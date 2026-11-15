"""Per-request budget for public generation: n and the output tokens one chat/completions request may ask for."""

import contextlib
import io
import json
import tempfile
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from bielik_one import launch
from bielik_one.api import remote_vllm
from bielik_one.api import vllm_middleware as middleware
from bielik_one.api.remote_vllm import GenerationLimits, check_generation_budget, public_request_body
from bielik_one.api.server import RequestError

from tests.test_launch import dry, package
from tests.test_vllm_middleware import make_app

LIMITS = GenerationLimits(max_n=16, max_tokens=131_072, context_tokens=32_768)


def refused(body, limits=LIMITS):
    with pytest.raises(RequestError) as caught:
        check_generation_budget(body, limits)
    return caught.value


# -- what is counted ---------------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "body",
    [
        {"messages": [{"role": "user", "content": "hi"}]},  # one answer, up to the full context
        {"messages": [], "n": 4},  # four full-context answers: exactly the budget
        {"messages": [], "n": 16, "max_tokens": 8192},
        {"messages": [], "n": 16, "max_completion_tokens": 8192, "max_tokens": 10**9},  # chat prefers this field
        {"prompt": ["a"] * 1024, "max_tokens": 128},  # 1024 prompts x 128 tokens
        {"prompt": list(range(5000)), "n": 4},  # one prompt given as token ids, not 5000 prompts
        {"messages": [[{"role": "user", "content": "a"}]] * 8, "max_tokens": 16384},  # batch chat
        {"input": "hi", "max_output_tokens": 4096},  # responses API
        {"messages": [], "max_tokens": 10**9},  # longer than the context counts as the context
    ],
)
def test_requests_within_the_budget_pass(body):
    check_generation_budget(body, LIMITS)


def test_n_over_the_limit_is_refused():
    error = refused({"messages": [], "n": 1000, "max_tokens": 1})
    assert error.loc == ["body", "n"]
    assert "n is 1000; this server allows at most 16" in error.msg


def test_best_of_and_beam_width_are_counted():
    error = refused({"prompt": "x", "n": 1, "best_of": 64, "max_tokens": 1})
    assert error.loc == ["body", "best_of"]
    error = refused({"prompt": "x", "n": 17, "use_beam_search": True, "max_tokens": 1})  # n is the beam width
    assert error.loc == ["body", "n"]


@pytest.mark.parametrize(
    "body, field, total",
    [
        ({"messages": [], "n": 5}, "max_tokens", 5 * 32_768),  # max_tokens not set: full context per answer
        ({"messages": [], "n": 16, "max_tokens": 16_384}, "max_tokens", 16 * 16_384),
        ({"messages": [], "max_completion_tokens": 32_768, "n": 8}, "max_completion_tokens", 8 * 32_768),
        ({"prompt": ["a"] * 1024, "max_tokens": 256}, "max_tokens", 1024 * 256),  # many prompts
        ({"prompt": [[1, 2]] * 64, "max_tokens": 4096}, "max_tokens", 64 * 4096),  # many token-id prompts
        ({"messages": [[{"role": "user", "content": "a"}]] * 64, "max_tokens": 4096}, "max_tokens", 64 * 4096),
        ({"input": "hi", "max_output_tokens": 32_768, "n": 5}, "max_output_tokens", 5 * 32_768),
    ],
)
def test_requests_over_the_token_budget_are_refused(body, field, total):
    error = refused(body)
    assert error.loc == ["body", field]
    assert f"up to {total} output tokens" in error.msg and "at most 131072" in error.msg


def test_unset_max_tokens_is_named_in_the_message():
    assert "(max_tokens not set)" in refused({"messages": [], "n": 8}).msg


def test_fields_of_the_wrong_type_are_left_to_vllm():
    for body in ({"messages": [], "n": "1000"}, {"messages": [], "n": True, "max_tokens": "x"}, {"n": None}):
        check_generation_budget(body, LIMITS)


def test_limits_can_be_lifted():
    check_generation_budget({"prompt": ["a"] * 1024, "n": 16384}, GenerationLimits(max_n=None, max_tokens=None))


# -- public_request_body: the shared filter of the frontend and the middleware ---------------------------------------


def test_public_request_body_applies_the_budget_and_keeps_valid_bodies_as_sent():
    raw = json.dumps({"model": "chat", "messages": [], "n": 2, "max_tokens": 100}).encode()
    assert public_request_body(raw, {"decision"}, "chat") is raw
    with pytest.raises(RequestError):
        public_request_body(json.dumps({"model": "chat", "messages": [], "n": 1000}).encode(), {"decision"}, "chat")
    lifted = GenerationLimits(max_n=None, max_tokens=None)
    body = json.dumps({"model": "chat", "messages": [], "n": 1000}).encode()
    assert public_request_body(body, {"decision"}, "chat", lifted) is body


# -- separate frontend ----------------------------------------------------------------------------------------------


class Engine:
    backend_url = "http://127.0.0.1:9"  # nothing listens: an accepted request ends as 502
    aliases = {"decision"}
    temperatures = {}
    calibration_source = None
    max_model_len = 32_768

    def close(self):
        pass


@pytest.mark.parametrize("path", ["/v1/chat/completions", "/v1/completions"])
def test_frontend_refuses_over_budget_requests_before_vllm(path):
    app = remote_vllm.build_remote_app(Engine(), chat_model="chat")
    with TestClient(app) as client:
        response = client.post(path, json={"model": "chat", "prompt": "x", "messages": [], "n": 1000})
        assert response.status_code == 400
        assert "n is 1000" in response.json()["error"]["message"]  # chat routes answer in the OpenAI format
        response = client.post(path, json={"model": "chat", "prompt": "x", "messages": [], "n": 2, "max_tokens": 64})
        assert response.status_code == 502  # passed the filter, then found no vLLM


def test_frontend_uses_configured_limits():
    limits = GenerationLimits(max_n=2, max_tokens=1000)
    app = remote_vllm.build_remote_app(Engine(), chat_model="chat", generation_limits=limits)
    with TestClient(app) as client:
        assert client.post("/v1/chat/completions", json={"messages": [], "n": 3, "max_tokens": 1}).status_code == 400
        too_long = client.post("/v1/chat/completions", json={"messages": [], "n": 2, "max_tokens": 501})
        assert too_long.status_code == 400 and "at most 1000" in too_long.json()["error"]["message"]


# -- vLLM middleware ------------------------------------------------------------------------------------------------


def test_middleware_refuses_over_budget_requests_on_every_public_route():
    app, _, _, _ = make_app()
    with TestClient(app) as client:
        for path in ("/v1/chat/completions", "/v1/completions", "/v1/responses", "/v1/messages", "/invocations"):
            response = client.post(path, json={"model": "base-a", "messages": [], "n": 1000})
            assert response.status_code == 400, path
            assert "n is 1000" in response.json()["error"]["message"]
            ok = client.post(path, json={"model": "base-a", "messages": [], "n": 2, "max_tokens": 64})
            assert ok.status_code == 200, path


def test_middleware_settings_and_internal_reads():
    app, owner, _, _ = make_app(max_generation_n=2, max_generation_tokens=None)
    with TestClient(app) as client:
        assert client.post("/v1/completions", json={"model": "base-a", "prompt": "x", "n": 3}).status_code == 400
        assert client.post("/v1/completions", json={"model": "base-a", "prompt": ["x"] * 1024}).status_code == 200
        # The middleware's own decision reads are not public requests and are not filtered.
        internal = {"x-systemone-internal": owner.settings.internal_token}
        assert client.post("/v1/completions", json={"model": "decision", "prompt": "x", "n": 3}, headers=internal).status_code == 200


def test_middleware_settings_from_env():
    settings = middleware.Settings.from_env({"SYSTEMONE_PACKAGE": "/tmp/package"})
    assert (settings.max_generation_n, settings.max_generation_tokens) == (
        remote_vllm.DEFAULT_MAX_GENERATION_N, remote_vllm.DEFAULT_MAX_GENERATION_TOKENS)
    settings = middleware.Settings.from_env(
        {"SYSTEMONE_PACKAGE": "/tmp/package", "SYSTEMONE_MAX_GENERATION_N": "4", "SYSTEMONE_MAX_GENERATION_TOKENS": "none"})
    assert (settings.max_generation_n, settings.max_generation_tokens) == (4, None)


@pytest.mark.parametrize("value", ["0", "-1", "many", "1.5"])
@pytest.mark.parametrize("name", ["SYSTEMONE_MAX_GENERATION_N", "SYSTEMONE_MAX_GENERATION_TOKENS"])
def test_middleware_rejects_invalid_limits(name, value):
    with pytest.raises(RuntimeError, match=name):
        middleware.Settings.from_env({"SYSTEMONE_PACKAGE": "/tmp/package", name: value})


# -- launcher and frontend command line -----------------------------------------------------------------------------


def test_launcher_passes_limits_to_the_frontend_only_when_given():
    for kind in launch.PORTS:
        with tempfile.TemporaryDirectory() as d:
            package(kind, Path(d))
            command = dry("frontend", d)["command"]
            assert "--max-generation-n" not in command and "--max-generation-tokens" not in command
            command = dry("frontend", d, "--max-generation-n", "4", "--max-generation-tokens", "none")["command"]
            assert command[command.index("--max-generation-n") + 1] == "4"
            assert command[command.index("--max-generation-tokens") + 1] == "none"


@pytest.mark.parametrize("args", [["--max-generation-n", "0"], ["--max-generation-tokens", "-5"], ["--max-generation-n", "x"]])
def test_launcher_rejects_invalid_limits(args):
    with tempfile.TemporaryDirectory() as d:
        package("qwen-merged", Path(d))
        with contextlib.redirect_stderr(io.StringIO()), pytest.raises(SystemExit):
            launch.main(["frontend", d, *args, "--dry-run"])


def test_frontend_parser_defaults_and_none():
    import argparse

    parser = argparse.ArgumentParser()
    remote_vllm.add_generation_arguments(parser)
    defaults = parser.parse_args([])
    assert remote_vllm.generation_limits(defaults) == GenerationLimits()
    parsed = parser.parse_args(["--max-generation-n", "none", "--max-generation-tokens", "5000"])
    assert remote_vllm.generation_limits(parsed, 4096) == GenerationLimits(None, 5000, 4096)
    for args in (["--max-generation-n", "0"], ["--max-generation-tokens", "abc"]):
        with contextlib.redirect_stderr(io.StringIO()), pytest.raises(SystemExit):
            parser.parse_args(args)
