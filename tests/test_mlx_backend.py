"""The MLX backend: protocol, checks and launcher wiring without MLX; LoRA and head math with MLX when installed."""

import hashlib
import json
import math
import tempfile
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest
from fastapi.testclient import TestClient

from bielik_one import launch, mlx_backend
from bielik_one.api.remote_vllm import RemoteVllmEngine
from tests.test_launch import dry, package


class FakeRunner:
    """Head logprobs -1.0 per head id, vocabulary logprobs -2.0; generation spells "hello world"."""

    generation_defaults = {"temperature": 0.7}
    vocab_size = 1000

    def __init__(self):
        self.calls = []

    def decision(self, prompt, ids, adapter):
        self.calls.append(("decision" if adapter else "base_heads", list(prompt)))
        return mlx_backend.head_logprobs([-1.0, -1.0], {7: 0, 8: 1}, ids)

    def vocab(self, prompt, ids):
        self.calls.append(("vocab", list(prompt)))
        return [-2.0] * len(ids)

    def encode(self, text):
        return [ord(c) for c in text]

    def chat_ids(self, messages, kwargs):
        self.calls.append(("chat", kwargs))
        return [1, 2, 3]

    def generate(self, prompt_ids, params):
        self.calls.append(("generate", params))
        words = ["hello", " world"]
        steps = ((word, n, "stop" if n == len(words) else None) for n, word in enumerate(words, 1))
        yield from mlx_backend.stop_at(steps, params["stop"])


def client(base_row_heads=False):
    runner = FakeRunner()
    return TestClient(mlx_backend.build_app(runner, "qwen-base", base_row_heads)), runner


def read(model, ids, **extra):
    return {"model": model, "prompt": [5, 6, 7], "max_tokens": 1, "logprob_token_ids": ids, **extra}


class ProtocolTests(unittest.TestCase):
    def test_models_list_base_and_decision_adapter(self):
        c, _ = client()
        cards = c.get("/v1/models").json()["data"]
        self.assertEqual([m["id"] for m in cards], ["qwen-base", "decision"])
        self.assertEqual(cards[1]["parent"], "qwen-base")
        self.assertEqual(c.get("/health").json(), {"status": "ok"})

    def test_decision_read_heads_and_clamped_non_heads(self):
        c, runner = client()
        out = c.post("/v1/completions", json=read("decision", [7, 8, 99])).json()
        table = out["choices"][0]["logprobs"]["top_logprobs"][0]
        self.assertEqual(table, {"token_id:7": -1.0, "token_id:8": -1.0, "token_id:99": mlx_backend.CLAMP})
        self.assertEqual(out["usage"]["prompt_tokens"], 3)
        self.assertEqual(runner.calls, [("decision", [5, 6, 7])])

    def test_base_row_flag_only_for_routed_packages(self):
        flag = {"vllm_xargs": {"systemone_head": "base"}}
        c, runner = client(base_row_heads=True)
        c.post("/v1/completions", json=read("qwen-base", [7, 99], **flag))
        c.post("/v1/completions", json=read("qwen-base", [7, 99]))
        self.assertEqual([k for k, _ in runner.calls], ["base_heads", "vocab"])
        # Without base_row_types vLLM has no logits processor: the flag is ignored and the canary stays finite.
        c, runner = client(base_row_heads=False)
        c.post("/v1/completions", json=read("qwen-base", [7, 99], **flag))
        self.assertEqual([k for k, _ in runner.calls], ["vocab"])

    def test_decision_reads_are_one_token_reads_over_token_ids(self):
        c, _ = client()
        for body in (
            read("decision", [7], max_tokens=2),
            read("decision", [7], prompt="text"),
            read("decision", [7], prompt=[]),
            read("decision", None),
            read("decision", [7, True]),
            read("decision", [7], prompt=[0] * mlx_backend.MAX_MODEL_LEN),
        ):
            self.assertEqual(c.post("/v1/completions", json=body).status_code, 400, body)
        self.assertEqual(c.post("/v1/completions", json=read("other", [7])).status_code, 404)

    def test_token_ids_outside_the_vocabulary_are_refused_before_the_model(self):
        c, runner = client()
        for body in (
            read("decision", [7, 1000]),
            read("decision", [7, -1]),
            read("decision", [7], prompt=[5, 1000]),
            read("qwen-base", [7], prompt=[-3]),
            {"prompt": [5, 10**12], "max_tokens": 1},
        ):
            r = c.post("/v1/completions", json=body)
            self.assertEqual(r.status_code, 400, body)
            self.assertIn("out of vocabulary", r.json()["error"]["message"])
            self.assertEqual(r.json()["error"]["type"], "BadRequestError")
        self.assertEqual(runner.calls, [])
        self.assertEqual(c.post("/v1/completions", json=read("decision", [7, 999])).status_code, 200)

    def test_chat_template_kwargs_are_template_variables_only(self):
        c, runner = client()
        body = {"messages": [{"role": "user", "content": "hi"}], "max_tokens": 1}
        for key in ("chat_template", "tokenize", "return_dict", "add_generation_prompt", "continue_final_message",
                    "tools", "return_tensors"):
            r = c.post("/v1/chat/completions", json={**body, "chat_template_kwargs": {key: "x"}})
            self.assertEqual(r.status_code, 400, key)
            self.assertIn(key, r.json()["error"]["message"])
        r = c.post("/v1/chat/completions", json={**body, "chat_template_kwargs": ["enable_thinking"]})
        self.assertEqual(r.status_code, 400)
        self.assertEqual(runner.calls, [])
        r = c.post("/v1/chat/completions", json={**body, "chat_template_kwargs": {"enable_thinking": True}})
        self.assertEqual(r.status_code, 200)
        self.assertEqual(runner.calls[0], ("chat", {"enable_thinking": True}))

    def test_chat_is_the_base_model_only(self):
        c, runner = client()
        body = {"messages": [{"role": "user", "content": "hi"}], "max_tokens": 5,
                "chat_template_kwargs": {"enable_thinking": False}}
        out = c.post("/v1/chat/completions", json=body).json()
        self.assertEqual(out["choices"][0]["message"]["content"], "hello world")
        self.assertEqual(out["usage"], {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5})
        self.assertEqual(runner.calls[0], ("chat", {"enable_thinking": False}))
        self.assertEqual(runner.calls[1][1]["temperature"], 0.7)  # generation_config default
        self.assertEqual(c.post("/v1/chat/completions", json={**body, "model": "decision"}).status_code, 400)
        self.assertEqual(c.post("/v1/chat/completions", json={**body, "model": "x"}).status_code, 404)
        self.assertEqual(c.post("/v1/chat/completions", json={"messages": []}).status_code, 400)
        self.assertEqual(c.post("/v1/chat/completions", content=b"[1]").status_code, 400)

    def test_streaming_chat_and_text_completion(self):
        c, _ = client()
        with c.stream("POST", "/v1/chat/completions",
                      json={"messages": [{"role": "user", "content": "hi"}], "stream": True}) as r:
            events = [line[6:] for line in r.iter_lines() if line.startswith("data: ")]
        self.assertEqual(events[-1], "[DONE]")
        chunks = [json.loads(e) for e in events[:-1]]
        self.assertEqual(chunks[0]["choices"][0]["delta"]["role"], "assistant")
        self.assertEqual("".join(ch["choices"][0]["delta"]["content"] for ch in chunks), "hello world")
        self.assertEqual(chunks[-1]["choices"][0]["finish_reason"], "stop")
        self.assertEqual({ch["object"] for ch in chunks}, {"chat.completion.chunk"})
        out = c.post("/v1/completions", json={"prompt": "ab", "stop": " wor", "temperature": 0}).json()
        self.assertEqual(out["choices"][0]["text"], "hello")
        self.assertEqual(out["choices"][0]["finish_reason"], "stop")
        self.assertEqual(out["usage"]["prompt_tokens"], 2)

    def test_streaming_holds_back_a_possible_stop_string(self):
        """A stream never sends text the stop string later cuts off: it ends as the non-streamed answer does."""
        c, _ = client()
        cases = [("hello world", ""), ("lo w", "hel"), ("lo wx", "hello world"), ("d!", "hello world"),
                 (["x", "o w"], "hell")]
        for stop, text in cases:
            body = {"prompt": "ab", "stop": stop}
            full = c.post("/v1/completions", json=body).json()["choices"][0]
            with c.stream("POST", "/v1/completions", json={**body, "stream": True}) as r:
                events = [line[6:] for line in r.iter_lines() if line.startswith("data: ")]
            chunks = [json.loads(e)["choices"][0] for e in events[:-1]]
            self.assertEqual(full["text"], text, stop)
            self.assertEqual("".join(ch["text"] for ch in chunks), text, stop)
            self.assertEqual(chunks[-1]["finish_reason"], full["finish_reason"], stop)

    def test_out_of_range_sampling_is_refused_before_the_stream_starts(self):
        c, runner = client()
        for extra in ({"min_p": 2}, {"temperature": -1}, {"top_p": 0}, {"top_k": -2}, {"seed": "1"}):
            body = {"messages": [{"role": "user", "content": "hi"}], "stream": True, **extra}
            r = c.post("/v1/chat/completions", json=body)
            self.assertEqual(r.status_code, 400, extra)
            self.assertEqual(r.json()["error"]["type"], "BadRequestError")
        self.assertEqual([k for k, _ in runner.calls if k == "generate"], [])

    def test_unsupported_sampling_parameters_are_refused_unless_neutral(self):
        """Penalties and the like are not implemented: refused before the stream starts, not silently ignored."""
        c, runner = client()
        for extra in ({"frequency_penalty": 3}, {"presence_penalty": -3}, {"repetition_penalty": 0},
                      {"repetition_penalty": 1.1}, {"min_tokens": 5}, {"ignore_eos": True}, {"logit_bias": {"5": 1}},
                      {"bad_words": ["x"]}, {"stop_token_ids": [2]}, {"allowed_token_ids": [5]}):
            body = {"messages": [{"role": "user", "content": "hi"}], "stream": True, **extra}
            r = c.post("/v1/chat/completions", json=body)
            self.assertEqual(r.status_code, 400, extra)
            self.assertIn("not supported", r.json()["error"]["message"])
        self.assertEqual([k for k, _ in runner.calls if k == "generate"], [])
        neutral = {"frequency_penalty": 0, "presence_penalty": 0.0, "repetition_penalty": 1, "min_tokens": 0,
                   "ignore_eos": False, "logit_bias": {}, "bad_words": [], "stop_token_ids": [], "allowed_token_ids": None}
        self.assertEqual(mlx_backend.sampling(neutral, {}, 10, 1000)["max_tokens"], mlx_backend.MAX_MODEL_LEN - 10)
        with self.assertRaises(ValueError):  # a generation_config that sets a penalty is refused too
            mlx_backend.sampling({}, {"repetition_penalty": 1.05}, 10, 1000)

    def test_official_frontend_reads_through_the_backend(self):
        """RemoteVllmEngine's read (anchor, canary, clamping checks) accepts the backend's answers."""
        c, _ = client(base_row_heads=True)

        def handler(request):
            r = c.post(request.url.path, content=request.content, headers={"Content-Type": "application/json"})
            return httpx.Response(r.status_code, content=r.content)

        metadata = dict(choice_capacity=255, labels_capacity=1, label_ids=[7], anchor_id=8, bias=[0.0],
                        threshold=0.5, source_repo="test/decision", base_row_types=["noul"], base_alias="qwen-base")
        engine = RemoteVllmEngine(metadata, "http://backend", tokenizer=object(),
                                  client=httpx.Client(transport=httpx.MockTransport(handler)))
        try:
            table, tokens = engine._read([5, 6, 7], [7], "choice")
            self.assertEqual((table, tokens), ({7: 0.0}, 3))
            table, _ = engine._read([5, 6, 7], [7], "noul")  # flagged base row: canary comes back clamped
            self.assertEqual(table, {7: 0.0})
        finally:
            engine.close()


class SamplingTests(unittest.TestCase):
    def test_request_values_then_generation_config_then_vllm_defaults(self):
        s = mlx_backend.sampling({"top_k": -1, "top_p": 1}, {"temperature": 0.6, "top_p": 0.95}, 10, 1000)
        self.assertEqual((s["temperature"], s["top_p"], s["top_k"], s["min_p"]), (0.6, 0.0, 0, 0.0))
        self.assertEqual(s["max_tokens"], mlx_backend.MAX_MODEL_LEN - 10)
        s = mlx_backend.sampling({"temperature": 0, "max_completion_tokens": 4, "stop": "x"}, {}, 10, 1000)
        self.assertEqual((s["temperature"], s["max_tokens"], s["stop"]), (0.0, 4, ["x"]))

    def test_invalid(self):
        for body in ({"max_tokens": 0}, {"max_tokens": mlx_backend.MAX_MODEL_LEN}, {"stop": [1]}, {"n": 2}):
            with self.assertRaises(ValueError):
                mlx_backend.sampling(body, {}, 10, 1000)

    def test_ranges_as_vllm_checks_them(self):
        """mlx_lm raises on these only when the first token is sampled, after a stream has started."""
        for body in ({"temperature": -0.1}, {"temperature": math.nan}, {"temperature": math.inf}, {"top_p": 0},
                     {"top_p": 1.5}, {"top_p": math.nan}, {"top_k": -2}, {"min_p": 2}, {"min_p": -0.1},
                     {"min_p": math.nan}, {"seed": "1"}, {"seed": True}, {"seed": 2**63}, {"seed": 0.5}):
            with self.assertRaises(ValueError, msg=body):
                mlx_backend.sampling(body, {}, 10, 1000)
        with self.assertRaises(ValueError):  # a generation_config default is checked like a request value
            mlx_backend.sampling({}, {"min_p": 2.0}, 10, 1000)
        s = mlx_backend.sampling({"temperature": 1, "min_p": 1, "top_k": 999, "seed": -1}, {}, 10, 1000)
        self.assertEqual((s["min_p"], s["top_k"], s["seed"]), (1.0, 999, 2**64 - 1))
        for top_k in (0, -1, 1000, 10**9):  # off; mlx_lm refuses a top_k that keeps the whole vocabulary
            self.assertEqual(mlx_backend.sampling({"top_k": top_k}, {}, 10, 1000)["top_k"], 0)

    def test_stop_strings_are_bounded(self):
        """Stop strings are checked after every token of an untrusted request."""
        for stop in (["x"] * 17, ["x" * 1001]):
            with self.assertRaises(ValueError):
                mlx_backend.sampling({"stop": stop}, {}, 10, 1000)
        self.assertEqual(len(mlx_backend.sampling({"stop": ["x" * 1000] * 16}, {}, 10, 1000)["stop"]), 16)


class StopTests(unittest.TestCase):
    def run_steps(self, pieces, stops, finish="stop"):
        steps = [(p, n, finish if n == len(pieces) else None) for n, p in enumerate(pieces, 1)]
        return list(mlx_backend.stop_at(steps, stops))

    def test_text_that_may_start_a_stop_string_is_held_back(self):
        out = self.run_steps(["ab", "c<", "/e", "nd>x"], ["</end>"])
        self.assertEqual([t for t, *_ in out], ["ab", "abc", "abc", "abc"])
        self.assertEqual(out[-1][1:], (4, "stop", True))

    def test_held_text_is_released_on_mismatch_or_at_the_end(self):
        out = self.run_steps(["ab", "c<", "/x"], ["</end>"])
        self.assertEqual([t for t, *_ in out], ["ab", "abc", "abc</x"])
        self.assertEqual(out[-1][2:], ("stop", True))
        out = self.run_steps(["ab", "c</"], ["</end>"], finish="length")
        self.assertEqual(out[-1], ("abc</", 2, "length", True))  # an unfinished stop string is not a match

    def test_a_stop_string_split_over_many_pieces_is_found(self):
        out = self.run_steps(list("abc hello world xyz"), ["hello world", "o w"])
        self.assertEqual(out[-1][:3], ("abc hell", 11, "stop"))  # "o w" is complete at the 11th piece

    def test_earliest_stop_string_wins(self):
        out = self.run_steps(["hello world"], ["world", "lo"])
        self.assertEqual(out, [("hel", 1, "stop", True)])


QWEN_KEY = "base_model.model.model.language_model.layers.{}.{}.lora_{}.weight"
LLAMA_KEY = "base_model.model.model.layers.{}.{}.lora_{}.weight"


class AdapterMappingTests(unittest.TestCase):
    def test_peft_names_meet_mlx_module_paths(self):
        keys = [QWEN_KEY.format(0, "linear_attn.out_proj", ab) for ab in "AB"] + [
            QWEN_KEY.format(3, "self_attn.q_proj", ab) for ab in "AB"
        ]
        pairs = mlx_backend.lora_pairs(keys)
        self.assertEqual(set(pairs), {"layers.0.linear_attn.out_proj", "layers.3.self_attn.q_proj"})
        index = mlx_backend.module_index(
            ["language_model.model.layers.0.linear_attn.out_proj", "language_model.model.layers.3.self_attn.q_proj",
             "language_model.lm_head"]
        )
        self.assertEqual(set(index), set(pairs))
        llama = mlx_backend.lora_pairs([LLAMA_KEY.format(12, "mlp.down_proj", ab) for ab in "AB"])
        self.assertEqual(set(llama), set(mlx_backend.module_index(["model.layers.12.mlp.down_proj"])))

    def test_incomplete_or_foreign_tensors_are_refused(self):
        with self.assertRaises(ValueError):
            mlx_backend.lora_pairs([LLAMA_KEY.format(0, "mlp.up_proj", "A")])
        with self.assertRaises(ValueError):
            mlx_backend.lora_pairs(["base_model.model.lm_head.weight"])

    def test_ambiguous_suffix_is_not_mapped(self):
        index = mlx_backend.module_index(["a.layers.0.q", "b.layers.0.q", "a.layers.1.q"])
        self.assertEqual(index, {"layers.1.q": "a.layers.1.q"})

    def test_lora_scale_plain_lora_only(self):
        config = {"peft_type": "LORA", "r": 128, "lora_alpha": 256, "bias": "none", "use_dora": False,
                  "rank_pattern": {}, "alpha_pattern": {}, "modules_to_save": None}
        self.assertEqual(mlx_backend.lora_scale(config, 128), 2.0)
        for change in ({"use_dora": True}, {"use_rslora": True}, {"rank_pattern": {"q_proj": 8}},
                       {"modules_to_save": ["lm_head"]}, {"bias": "all"}, {"peft_type": "IA3"}):
            with self.assertRaises(ValueError):
                mlx_backend.lora_scale({**config, **change}, 128)
        with self.assertRaises(ValueError):
            mlx_backend.lora_scale(config, 64)


class PackageTests(unittest.TestCase):
    def test_adapter_packages_pin_their_base(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            package("llama-adapter", root)
            info = mlx_backend.adapter_package(root, "llama-adapter")
            self.assertEqual((info["repo"], info["revision"], info["alias"]), ("org/base", "r1", "bielik-base"))
            self.assertEqual(info["adapter"], root / "adapter")
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            package("qwen-adapter", root)
            info = mlx_backend.adapter_package(root, "qwen-adapter")
            self.assertEqual((info["repo"], info["alias"], info["adapter"]), (launch.QWEN_BASE, "qwen-base", root))
        with self.assertRaises(ValueError):
            mlx_backend.adapter_package(Path("."), "qwen-merged")

    def test_adapter_and_heads_must_match_heads_json(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            (root / "heads.json").write_text(json.dumps(
                {"weights_file": "heads.safetensors", "weights_sha256": "h", "adapter_sha256": "a"}))
            info = {"adapter": root}
            mlx_backend.check_adapter_manifest(root, info, {"adapter_model.safetensors": "a", "heads.safetensors": "h"})
            for sums in ({"adapter_model.safetensors": "x", "heads.safetensors": "h"}, {"heads.safetensors": "h"}):
                with self.assertRaises(ValueError):
                    mlx_backend.check_adapter_manifest(root, info, sums)


class BaseCheckTests(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.base = Path(self.dir.name)
        (self.base / "model-00001-of-00001.safetensors").write_bytes(b"weights")
        (self.base / "config.json").write_text("{}")
        siblings = [
            SimpleNamespace(rfilename="model-00001-of-00001.safetensors", lfs=SimpleNamespace(
                sha256=hashlib.sha256(b"weights").hexdigest())),
            SimpleNamespace(rfilename="config.json", lfs=None, blob_id=mlx_backend.git_blob_sha1(self.base / "config.json")),
            SimpleNamespace(rfilename="README.md", lfs=None, blob_id="0"),
            SimpleNamespace(rfilename="sub/model.safetensors", lfs=None, blob_id="0"),
        ]
        self.expected = mlx_backend.checksums_from_siblings(siblings)

    def tearDown(self):
        self.dir.cleanup()

    def test_pinned_files_only(self):
        self.assertEqual(set(self.expected), {"model-00001-of-00001.safetensors", "config.json"})
        mlx_backend.verify_base(self.base, self.expected)

    def test_extra_tokenizer_and_config_files_are_refused(self):
        """Files the loaders would read that the pinned revision does not have, e.g. a chat_template.jinja that
        transformers prefers to tokenizer_config.json's template."""
        for name in ("chat_template.jinja", "tokenizer_config.json", "hf_quant_config.json",
                     "additional_chat_templates/default.jinja"):
            path = self.base / name
            path.parent.mkdir(exist_ok=True)
            path.write_text("{}")
            with self.assertRaisesRegex(ValueError, f"{name} not part of the pinned revision"):
                mlx_backend.verify_base(self.base, self.expected)
            path.unlink()
        for name in ("README.md", ".gitattributes", "notes/chat_template.jinja"):  # not read by the loaders
            path = self.base / name
            path.parent.mkdir(exist_ok=True)
            path.write_text("x")
        mlx_backend.verify_base(self.base, self.expected)

    def test_pinned_template_directory_is_checked(self):
        siblings = [SimpleNamespace(rfilename=name, lfs=None, blob_id="0") for name in (
            "model.safetensors", "chat_template.jinja", "additional_chat_templates/tool_use.jinja", "sub/x.jinja")]
        self.assertEqual(set(mlx_backend.checksums_from_siblings(siblings)),
                         {"model.safetensors", "chat_template.jinja", "additional_chat_templates/tool_use.jinja"})

    def test_changed_missing_or_extra_weights_are_refused(self):
        (self.base / "config.json").write_text('{"x": 1}')
        with self.assertRaisesRegex(ValueError, "config.json differs"):
            mlx_backend.verify_base(self.base, self.expected)
        (self.base / "config.json").unlink()
        with self.assertRaisesRegex(ValueError, "missing"):
            mlx_backend.verify_base(self.base, self.expected)
        (self.base / "config.json").write_text("{}")
        (self.base / "model-extra.safetensors").write_bytes(b"x")
        with self.assertRaisesRegex(ValueError, "not part of the pinned revision"):
            mlx_backend.verify_base(self.base, self.expected)

    def test_git_blob_sha1_matches_git(self):
        # `printf hello | git hash-object --stdin`
        path = self.base / "x"
        path.write_bytes(b"hello")
        self.assertEqual(mlx_backend.git_blob_sha1(path), "b6fc4c620b67d95f953a5c1c1230aaab5db5a1b0")


class LaunchTests(unittest.TestCase):
    def test_engine_mlx_runs_the_mlx_backend_for_adapter_packages(self):
        for kind in ("qwen-adapter", "llama-adapter"):
            with tempfile.TemporaryDirectory() as d:
                package(kind, Path(d))
                out = dry("backend", d, "--engine", "mlx", "--base", d, "--cache-limit-gb", "2")
                cmd = out["command"]
                self.assertEqual(cmd[1:3], ["-m", "bielik_one.mlx_backend"])
                self.assertEqual(cmd[cmd.index("--port") + 1], str(launch.PORTS[kind][0]))
                self.assertEqual(cmd[cmd.index("--base") + 1], str(Path(d).resolve()))
                self.assertEqual(cmd[cmd.index("--cache-limit-gb") + 1], "2.0")
                self.assertNotIn("VLLM_USE_V2_MODEL_RUNNER", out["env"])

    def test_merged_packages_and_vllm_flags_are_refused(self):
        with tempfile.TemporaryDirectory() as d:
            package("qwen-merged", Path(d))
            with self.assertRaisesRegex(ValueError, "adapter packages"):
                dry("backend", d, "--engine", "mlx")
        with tempfile.TemporaryDirectory() as d:
            package("qwen-adapter", Path(d))
            for flags in (["--fp8"], ["--eager"], ["--max-num-seqs", "8"], ["--kv-cache-gb", "2"]):
                with self.assertRaisesRegex(ValueError, "not used by --engine mlx"):
                    dry("backend", d, "--engine", "mlx", *flags)
            for args in (["backend", d, "--base", d], ["frontend", d, "--engine", "mlx", "--skip-base-check"]):
                with self.assertRaisesRegex(ValueError, "--engine mlx"):
                    dry(*args)
            with self.assertRaisesRegex(ValueError, "positive"):
                dry("backend", d, "--engine", "mlx", "--cache-limit-gb", "0")


# -- with MLX (Apple silicon): the LoRA switch and the head readout on synthetic arrays -------------------------


@pytest.fixture
def mlx_model():
    pytest.importorskip("mlx.core")
    pytest.importorskip("mlx_lm")
    from bielik_one import mlx_model

    return mlx_model


def test_switched_lora_matches_peft_formula(mlx_model, tmp_path):
    import mlx.core as mx
    import mlx.nn as nn
    import numpy as np

    rng = np.random.default_rng(0)

    class Layer(nn.Module):
        def __init__(self):
            super().__init__()
            self.q = nn.Linear(6, 4, bias=False)

    class Block(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = [Layer()]

    model = Block()
    model.layers[0].q.weight = mx.array(rng.normal(size=(4, 6)).astype(np.float32))
    a, b = rng.normal(size=(2, 6)).astype(np.float32), rng.normal(size=(4, 2)).astype(np.float32)
    mx.save_safetensors(str(tmp_path / "adapter_model.safetensors"), {
        "base_model.model.layers.0.q.lora_A.weight": mx.array(a),
        "base_model.model.layers.0.q.lora_B.weight": mx.array(b),
    })
    (tmp_path / "adapter_config.json").write_text(json.dumps({"r": 2, "lora_alpha": 4}))
    modules, _ = mlx_model.attach_lora(model, tmp_path, 2, dtype=mx.float32)
    assert modules == 1
    x = rng.normal(size=(3, 6)).astype(np.float32)
    lora = model.layers[0].q
    assert isinstance(lora, mlx_model.SwitchedLoRALinear)
    w = np.array(lora.base.weight)
    with mx.stream(mx.cpu):  # float32 reference precision (GPU float32 matmuls may round differently)
        off = np.array(lora(mx.array(x)))
        mlx_model.Switch.on = True
        try:
            on = np.array(lora(mx.array(x)))
        finally:
            mlx_model.Switch.on = False
    np.testing.assert_allclose(off, x @ w.T, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(on, x @ w.T + 2.0 * (x @ a.T) @ b.T, rtol=1e-5, atol=1e-5)


def test_seeded_sampling_does_not_depend_on_the_thread(mlx_model):
    """A seeded request draws from its own key: stepping it on other threads (whose MLX random state differs, as
    Starlette's worker threads do) gives the same tokens, and the global random state is left alone."""
    import mlx.core as mx

    params = {"temperature": 1.0, "top_p": 0.9, "min_p": 0.0, "top_k": 0, "seed": 3, "max_tokens": 19}
    with mx.stream(mx.cpu):
        logprobs = mx.log(mx.softmax(mx.random.normal((1, 64), key=mx.random.key(0)), axis=-1))
        mx.eval(logprobs)
        sampler = mlx_model.request_sampler(params)
        same_thread = [sampler(logprobs).item() for _ in range(20)]
    sampler, threads = mlx_model.request_sampler(params), []

    def step(seed):
        mx.random.seed(seed)
        with mx.stream(mx.cpu):
            threads.append(sampler(logprobs).item())

    for n in range(20):
        worker = threading.Thread(target=step, args=(n,))
        worker.start()
        worker.join()
    assert threads == same_thread
    assert len(set(same_thread)) > 1  # sampled, not greedy
    mx.random.seed(11)
    before = mx.random.uniform().item()
    mx.random.seed(11)
    mlx_model.request_sampler(params)(logprobs)
    assert mx.random.uniform().item() == before


def test_heads_readout_is_log_softmax_over_heads(mlx_model, tmp_path):
    import mlx.core as mx
    import numpy as np

    rng = np.random.default_rng(1)
    weight = rng.normal(size=(3, 5)).astype(np.float32)
    weight[1] = 0  # anchor
    mx.save_safetensors(str(tmp_path / "heads.safetensors"),
                        {"weight": mx.array(weight), "token_ids": mx.array(np.array([40, 41, 42], dtype=np.int64))})
    (tmp_path / "heads.json").write_text(json.dumps({"weights_file": "heads.safetensors", "anchor_id": 41}))
    heads = mlx_model.Heads(tmp_path, 5)
    runner = object.__new__(mlx_model.MlxRunner)
    hidden = rng.normal(size=5).astype(np.float32)
    runner.heads, runner.lock = heads, threading.Lock()
    runner.model = SimpleNamespace(model=lambda x: mx.array(hidden)[None, None])
    with mx.stream(mx.cpu):
        values = runner.decision([1, 2], [42, 40, 7], adapter=True)
    logits = weight.astype(np.float64) @ hidden
    expected = logits - (logits.max() + math.log(np.exp(logits - logits.max()).sum()))
    np.testing.assert_allclose(values[:2], [expected[2], expected[0]], rtol=1e-5)
    assert values[2] == mlx_backend.CLAMP
    weight[1, 0] = 1.0  # a non-zero anchor row is refused, as by the vLLM plugin
    mx.save_safetensors(str(tmp_path / "heads.safetensors"),
                        {"weight": mx.array(weight), "token_ids": mx.array(np.array([40, 41, 42], dtype=np.int64))})
    with pytest.raises(ValueError, match="anchor"):
        mlx_model.Heads(tmp_path, 5)
