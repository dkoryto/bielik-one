import shlex
import unittest.mock
import hashlib, json, tempfile, unittest
from pathlib import Path
import io, contextlib

from bielik_one import launch


def package(kind, root, *, base_model="speakleash/Bielik-11B-v3.1-Instruct"):
    files = {
        "conversion.json": {
            "base_revision": "r1",
            "source_repo": "x/y",
            "threshold": 0.5,
        }
    }
    if kind == "llama-adapter":
        files["serving.json"] = {
            "base_model": base_model,
            "base_revision": "r1",
            "base_alias": "bielik-base",
            "rank": 64,
            "kv_cache_memory_bytes": 1,
        }
        files["heads.json"] = {"base_revision": "r1", "transformer_rank": 64}
    elif kind == "llama-merged":
        files["serving.json"] = {
            "general_alias": "bielik-general",
            "kv_cache_memory_bytes": 1,
        }
        files["config.json"] = {"architectures": ["LlamaForCausalLM"]}
    elif kind == "qwen-adapter":
        files["heads.json"] = {"base_revision": "r1"}
        (root / "adapter_model.safetensors").write_bytes(b"x")
    else:
        files["conversion.json"]["merged_model"] = True
    for name, value in files.items():
        (root / name).write_text(json.dumps(value))
    (root / "calibration.json").write_text("{}")
    sums = {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in root.iterdir()
        if p.is_file()
    }
    (root / "SHA256SUMS.json").write_text(json.dumps(sums))


def dry(*args):
    """Run the launcher; a refused package (argparse error, exit 2) surfaces as ValueError with its message."""
    out, err = io.StringIO(), io.StringIO()
    try:
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            launch.main([*args, "--dry-run"])
    except SystemExit as exc:
        raise ValueError(err.getvalue()) from exc
    return json.loads(out.getvalue())


def configure_required_tools(case):
    directory=tempfile.TemporaryDirectory(prefix="required chat tools ")
    case.addCleanup(directory.cleanup)
    tools=Path(directory.name)/"tools";tools.mkdir()
    for name in ("bielik_vllm_tool_parser.py","bielik_vllm_reasoning_parser.py","bielik_advanced_chat_template.jinja"):
        (tools/name).write_text("# external test fixture")
    patch=unittest.mock.patch.dict("os.environ", {"BIELIK_TOOLS_DIR":directory.name})
    patch.start();case.addCleanup(patch.stop)


class LaunchTests(unittest.TestCase):
    def setUp(self):
        configure_required_tools(self)

    def test_all_kinds(self):
        for kind in launch.PORTS:
            with tempfile.TemporaryDirectory() as d:
                root = Path(d)
                package(kind, root)
                back = dry("backend", d)
                front = dry("frontend", d)
                self.assertEqual(back["kind"], kind)
                cmd = front["command"]
                self.assertEqual(
                    cmd[cmd.index("--calibration") + 1], str(root / "calibration.json")
                )
                self.assertIn("--backend-url", cmd)
                self.assertEqual(
                    cmd[cmd.index("--backend-url") + 1],
                    f"http://127.0.0.1:{launch.PORTS[kind][0]}",
                )
                raw = dry("frontend", d, "--no-calibration")["command"]
                self.assertIn("--no-calibration", raw)
                self.assertNotIn("--calibration", raw)
                self.assertNotIn("--quantization", back["command"])  # bfloat16 by default
                self.assertIn("--quantization", dry("backend", d, "--fp8")["command"])
                self.assertEqual(
                    back["env"].get("VLLM_PLUGINS"),
                    "systemone_heads" if "adapter" in kind else "",
                )

    def test_adapter_package_template_is_passed_to_frontend(self):
        for kind in ("llama-adapter", "qwen-adapter"):
            with tempfile.TemporaryDirectory() as d:
                root = Path(d)
                (root / "chat_template.jinja").write_text("PINNED")
                package(kind, root)
                cmd = dry("frontend", d)["command"]
                meta = json.loads(Path(cmd[cmd.index("--metadata") + 1]).read_text())
                self.assertEqual(
                    meta["chat_template_path"], str(root / "chat_template.jinja")
                )
            with tempfile.TemporaryDirectory() as d:
                package(kind, Path(d))
                cmd = dry("frontend", d)["command"]
                self.assertNotIn(
                    "chat_template_path",
                    json.loads(Path(cmd[cmd.index("--metadata") + 1]).read_text()),
                )

    def test_unlisted_or_missing_template_refused(self):
        with tempfile.TemporaryDirectory() as d:
            package("llama-adapter", Path(d))
            (Path(d) / "chat_template.jinja").write_text("STRAY")
            with self.assertRaises(ValueError):
                dry("frontend", d)
        with tempfile.TemporaryDirectory() as d:
            (Path(d) / "chat_template.jinja").write_text("PINNED")
            package("llama-adapter", Path(d))
            (Path(d) / "chat_template.jinja").unlink()
            with self.assertRaisesRegex(ValueError, "Missing file"):
                dry("frontend", d)

    def test_heads_manifest_cross_check(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            (root / "heads.safetensors").write_bytes(b"heads")
            package("llama-adapter", root)
            heads = json.loads((root / "heads.json").read_text())
            heads["weights_sha256"] = "0" * 64
            (root / "heads.json").write_text(json.dumps(heads))
            sums = json.loads((root / "SHA256SUMS.json").read_text())
            sums["heads.json"] = hashlib.sha256(
                (root / "heads.json").read_bytes()
            ).hexdigest()
            (root / "SHA256SUMS.json").write_text(json.dumps(sums))
            with self.assertRaisesRegex(ValueError, "Manifest mismatch"):
                dry("backend", d)

    def test_routed_adapter_package(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            package("llama-adapter", root)
            serving = json.loads((root / "serving.json").read_text())
            serving["base_row_types"] = ["choice", "noul", "score"]
            (root / "serving.json").write_text(json.dumps(serving))
            heads = json.loads((root / "heads.json").read_text())

            def resum():
                sums = {
                    p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                    for p in root.iterdir()
                    if p.is_file() and p.name != "SHA256SUMS.json"
                }
                (root / "SHA256SUMS.json").write_text(json.dumps(sums))

            resum()
            with self.assertRaisesRegex(ValueError, "base_row_heads"):
                dry("backend", d)
            heads["base_row_heads"] = True
            (root / "heads.json").write_text(json.dumps(heads))
            resum()
            back = dry("backend", d)["command"]
            self.assertEqual(
                back[back.index("--logits-processors") + 1],
                "systemone_vllm_heads.processor:SystemOneBaseHeadsProcessor",
            )
            cmd = dry("frontend", d)["command"]
            meta = json.loads(Path(cmd[cmd.index("--metadata") + 1]).read_text())
            self.assertEqual(meta["base_row_types"], ["choice", "noul", "score"])
            self.assertEqual(meta["base_alias"], "bielik-base")

    def test_checksum_mismatch_refuses(self):
        with tempfile.TemporaryDirectory() as d:
            package("llama-merged", Path(d))
            (Path(d) / "calibration.json").write_text('{"changed": 1}')
            with self.assertRaises(ValueError):
                dry("frontend", d)

    def test_auxiliary_files_can_be_edited_or_omitted(self):
        names = ("README.md", "LICENSE", ".gitattributes", "requirements.txt")
        for kind in launch.PORTS:
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as d:
                root = Path(d)
                for name in names:
                    (root / name).write_text("published content")
                package(kind, root)
                published = json.loads((root / "SHA256SUMS.json").read_text())
                for name in names:
                    (root / name).write_text("updated content")
                for component in ("backend", "frontend"):
                    self.assertEqual(dry(component, d)["kind"], kind)
                for name in names:
                    (root / name).unlink()
                for component in ("backend", "frontend"):
                    self.assertEqual(dry(component, d)["kind"], kind)
                self.assertEqual(launch.verify(root), published)

    def test_all_weight_files_remain_required_and_checksummed(self):
        names = (
            "model-00001.safetensors",
            "adapter/adapter_model.safetensors",
            "heads.safetensors",
        )
        for kind in launch.PORTS:
            for name in names:
                with self.subTest(kind=kind, file=name), tempfile.TemporaryDirectory() as d:
                    root = Path(d)
                    package(kind, root)
                    target = root / name
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_bytes(b"published weights")
                    sums = json.loads((root / "SHA256SUMS.json").read_text())
                    sums[name] = hashlib.sha256(target.read_bytes()).hexdigest()
                    (root / "SHA256SUMS.json").write_text(json.dumps(sums))
                    for component in ("backend", "frontend"):
                        self.assertEqual(dry(component, d)["kind"], kind)
                    target.write_bytes(b"corrupted weights")
                    for component in ("backend", "frontend"):
                        with self.assertRaisesRegex(ValueError, "Checksum mismatch"):
                            dry(component, d)
                    target.unlink()
                    for component in ("backend", "frontend"):
                        with self.assertRaisesRegex(ValueError, "Missing file"):
                            dry(component, d)

    def test_runtime_metadata_remains_checksummed(self):
        names = (
            "config.json", "conversion.json", "serving.json", "calibration.json",
            "tokenizer.json", "tokenizer_config.json", "chat_template.jinja",
        )
        for name in names:
            with self.subTest(file=name), tempfile.TemporaryDirectory() as d:
                root = Path(d)
                package("llama-merged", root)
                target = root / name
                if not target.exists():
                    target.write_text("published content")
                sums = json.loads((root / "SHA256SUMS.json").read_text())
                sums[name] = hashlib.sha256(target.read_bytes()).hexdigest()
                (root / "SHA256SUMS.json").write_text(json.dumps(sums))
                target.write_text("corrupted content")
                for component in ("backend", "frontend"):
                    with self.assertRaisesRegex(ValueError, "Checksum mismatch"):
                        dry(component, d)


if __name__ == "__main__":
    unittest.main()


class PackageSpecTests(unittest.TestCase):
    def test_local_dir_repo_id_and_errors(self):
        from unittest import mock

        with tempfile.TemporaryDirectory() as d:
            self.assertEqual(launch.resolve_package(d), Path(d).resolve())
            with self.assertRaises(ValueError):
                launch.resolve_package(d, "abc")
            with mock.patch("huggingface_hub.snapshot_download", return_value=d) as dl:
                self.assertEqual(launch.resolve_package("org/Model", "rev1"), Path(d))
                dl.assert_called_once_with("org/Model", revision="rev1")
        with self.assertRaises(ValueError):
            launch.resolve_package("./missing-dir")

    def test_hf_cache_layout_with_symlinks_verifies(self):
        with tempfile.TemporaryDirectory() as d:
            src = Path(d) / "src"
            src.mkdir()
            package("llama-merged", src)
            blobs = Path(d) / "blobs"
            blobs.mkdir()
            snap = Path(d) / "snapshots" / "rev"
            snap.mkdir(parents=True)
            for f in src.iterdir():
                (blobs / f.name).write_bytes(f.read_bytes())
                (snap / f.name).symlink_to(Path("../..") / "blobs" / f.name)
            launch.verify(snap)
            self.assertEqual(dry("backend", str(snap))["kind"], "llama-merged")

    def test_checksum_path_escape_refused(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            package("llama-merged", root)
            (root / "SHA256SUMS.json").write_text(json.dumps({"../x": "0"}))
            with self.assertRaises(ValueError):
                launch.verify(root)

    def test_auxiliary_names_do_not_bypass_path_validation(self):
        for name in ("../README.md", "/LICENSE", "..\\requirements.txt"):
            with self.subTest(file=name), tempfile.TemporaryDirectory() as d:
                root = Path(d)
                (root / "SHA256SUMS.json").write_text(json.dumps({name: "0" * 64}))
                with self.assertRaisesRegex(ValueError, "Invalid checksum path"):
                    launch.verify(root)

    def test_auxiliary_exclusions_apply_only_at_package_root(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            target = root / "adapter" / "README.md"
            target.parent.mkdir()
            target.write_text("changed")
            (root / "SHA256SUMS.json").write_text(
                json.dumps({"adapter/README.md": "0" * 64})
            )
            with self.assertRaisesRegex(ValueError, "Checksum mismatch"):
                launch.verify(root)


class EagerFlagTests(unittest.TestCase):
    def setUp(self):
        configure_required_tools(self)

    def test_graphs_by_default_eager_on_request(self):
        for kind in launch.PORTS:
            with tempfile.TemporaryDirectory() as d:
                package(kind, Path(d))
                self.assertNotIn("--enforce-eager", dry("backend", d)["command"], kind)
                self.assertIn("--enforce-eager", dry("backend", d, "--eager")["command"], kind)


class ProfileOverrideTests(unittest.TestCase):
    def setUp(self):
        configure_required_tools(self)

    def test_max_num_seqs_and_kv_cache_override(self):
        for kind in launch.PORTS:
            with tempfile.TemporaryDirectory() as d:
                package(kind, Path(d))
                base = dry("backend", d)["command"]
                cmd = dry("backend", d, "--max-num-seqs", "32", "--kv-cache-gb", "4")["command"]
                self.assertEqual(cmd[cmd.index("--max-num-seqs") + 1], "32", kind)
                self.assertEqual(cmd[cmd.index("--kv-cache-memory-bytes") + 1], str(4 * 2**30), kind)
                # nothing else changes
                strip = lambda c: [x for i, x in enumerate(c) if i == 0 or c[i - 1] not in ("--max-num-seqs", "--kv-cache-memory-bytes")]
                self.assertEqual(len(strip(cmd)), len(strip(base)))
                front = dry("frontend", d, "--max-num-seqs", "64")["command"]
                self.assertEqual(front[front.index("--workers") + 1], "64", kind)

    def test_max_body_mb_goes_to_the_frontend(self):
        for kind in launch.PORTS:
            with tempfile.TemporaryDirectory() as d:
                package(kind, Path(d))
                self.assertNotIn("--max-body-mb", dry("frontend", d)["command"])
                front = dry("frontend", d, "--max-body-mb", "4")["command"]
                self.assertEqual(front[front.index("--max-body-mb") + 1], "4.0", kind)

    def test_invalid_values_refused(self):
        with tempfile.TemporaryDirectory() as d:
            package("llama-merged", Path(d))
            for args in (["--max-num-seqs", "0"], ["--kv-cache-gb", "-1"], ["--max-body-mb", "0"]):
                with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                    launch.main(["backend", d, *args, "--dry-run"])


class PrintedCommandTests(unittest.TestCase):
    def setUp(self):
        configure_required_tools(self)

    """What --dry-run and --shell print is exactly what the launcher would execute, environment included."""

    def shell(self, *args):
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            self.assertEqual(launch.main([*args, "--shell"]), 0)
        return shlex.split(out.getvalue())

    def test_dry_run_env_includes_the_launcher_environment(self):
        for kind in launch.PORTS:
            with tempfile.TemporaryDirectory() as d:
                package(kind, Path(d))
                env = dry("backend", d)["env"]
                for key, value in launch.LAUNCH_ENV.items():
                    self.assertEqual(env[key], value, kind)
                if kind.endswith("adapter"):
                    self.assertEqual(env["VLLM_PLUGINS"], "systemone_heads")

    def test_shell_line_matches_the_dry_run(self):
        for kind in launch.PORTS:
            for component in ("backend",):
                with tempfile.TemporaryDirectory() as d:
                    package(kind, Path(d))
                    args = (component, d, "--max-num-seqs", "16", "--port", "9100")
                    expected = dry(*args)
                    words = self.shell(*args)
                    self.assertEqual(words[0], "env")
                    n = len(expected["env"])
                    self.assertEqual(dict(w.split("=", 1) for w in words[1 : 1 + n]), expected["env"])
                    self.assertEqual(words[1 + n :], expected["command"], (kind, component))

    def test_shell_starts_nothing(self):
        with tempfile.TemporaryDirectory() as d:
            package("llama-adapter", Path(d))
            with unittest.mock.patch.object(launch.os, "execvpe") as run:
                self.shell("backend", d)
            run.assert_not_called()

    def test_shell_is_backend_only(self):
        with tempfile.TemporaryDirectory() as d:
            package("llama-adapter", Path(d))
            err = io.StringIO()
            with contextlib.redirect_stderr(err), self.assertRaises(SystemExit):
                launch.main(["frontend", d, "--shell"])
            self.assertIn("--backend-url", err.getvalue())


class BielikChatParserTests(unittest.TestCase):
    def setUp(self):
        configure_required_tools(self)

    def make_package(self, root, model="speakleash/Bielik-11B-v3.1-Instruct"):
        (root / "chat_template.jinja").write_text("PINNED DECISION")
        package("llama-adapter", root)
        path = root / "serving.json"
        serving = json.loads(path.read_text())
        serving["base_model"] = model
        path.write_text(json.dumps(serving))
        sums_path = root / "SHA256SUMS.json"
        sums = json.loads(sums_path.read_text())
        sums.pop("SHA256SUMS.json", None)
        sums["serving.json"] = hashlib.sha256(path.read_bytes()).hexdigest()
        sums_path.write_text(json.dumps(sums))

    def make_tools(self, root):
        tools = root / "tools"
        tools.mkdir()
        for name in (
            "bielik_vllm_tool_parser.py", "bielik_vllm_reasoning_parser.py",
            "bielik_advanced_chat_template.jinja",
        ):
            (tools / name).write_text("EXTERNAL")
        return tools

    def test_supported_models_flags_and_separate_decision_template(self):
        for model, advanced in launch.BIELIK_CHAT_MODELS.items():
            with self.subTest(model=model), tempfile.TemporaryDirectory() as d:
                root = Path(d)
                self.make_package(root, model)
                tools = self.make_tools(root)
                before = dry("backend", d)["command"]
                after = dry("backend", d, "--bielik-tools-dir", d)["command"]
                self.assertEqual(before[before.index("--tool-call-parser")+1], "bielik")
                for flag in ("--enable-auto-tool-choice", "--exclude-tools-when-tool-choice-none"):
                    self.assertEqual(after.count(flag), 1)
                for flag, name in (("--tool-parser-plugin", "bielik_vllm_tool_parser.py"),
                                   ("--reasoning-parser-plugin", "bielik_vllm_reasoning_parser.py")):
                    self.assertEqual(after[after.index(flag) + 1], str(tools / name))
                for flag in ("--tool-call-parser", "--reasoning-parser"):
                    self.assertEqual(after[after.index(flag) + 1], "bielik")
                self.assertEqual("--chat-template" in after, advanced)
                if advanced:
                    self.assertEqual(after[after.index("--chat-template") + 1],
                                     str(tools / "bielik_advanced_chat_template.jinja"))
                front = dry("frontend", d)["command"]
                metadata = json.loads(Path(front[front.index("--metadata") + 1]).read_text())
                self.assertEqual(metadata["chat_template_path"], str(root / "chat_template.jinja"))
                self.assertEqual((root / "chat_template.jinja").read_text(), "PINNED DECISION")

    def test_environment_cli_override_opt_out_and_shell_from_other_cwd(self):
        with tempfile.TemporaryDirectory(prefix="chat parsers ") as d:
            root = Path(d)
            self.make_package(root)
            tools = self.make_tools(root)
            with unittest.mock.patch.dict("os.environ", {"BIELIK_TOOLS_DIR": "/missing/env/checkout"}):
                # Explicit CLI wins over the environment; paths survive spaces/foreign cwd.
                with contextlib.chdir(root.parent):
                    command = dry("backend", str(root), "--bielik-tools-dir", root.name)["command"]
                self.assertEqual(command[command.index("--tool-parser-plugin") + 1],
                                 str(tools / "bielik_vllm_tool_parser.py"))
                with self.assertRaisesRegex(ValueError, "parsers are required"):
                    dry("backend", d, "--no-bielik-chat-parsers")
                # A global backend setting does not alter frontend decision metadata.
                self.assertNotIn("--tool-call-parser", dry("frontend", d)["command"])
            with unittest.mock.patch.dict("os.environ", {"BIELIK_TOOLS_DIR": d}):
                enabled = dry("backend", d)["command"]
                self.assertEqual(enabled, command)
                out = io.StringIO()
                with contextlib.redirect_stdout(out):
                    launch.main(["backend", d, "--shell"])
                self.assertEqual(shlex.split(out.getvalue())[1 + len(launch.LAUNCH_ENV) + 2:], enabled)

    def test_invalid_models_modes_and_missing_resources_fail_before_exec(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            self.make_package(root)
            tools = self.make_tools(root)
            for missing in ("bielik_vllm_tool_parser.py", "bielik_vllm_reasoning_parser.py"):
                path = tools / missing
                path.unlink()
                with self.assertRaisesRegex(ValueError, missing):
                    dry("backend", d, "--bielik-tools-dir", d)
                path.write_text("EXTERNAL")
            for args in (("frontend", d), ("backend", d, "--engine", "mlx")):
                with self.assertRaisesRegex(ValueError, "vLLM backend options"):
                    dry(*args, "--bielik-tools-dir", d)
            self.make_package(root, "org/unrecognized-llama")
            with self.assertRaisesRegex(ValueError, "unsupported base model"):
                dry("backend", d, "--bielik-tools-dir", d)
        for kind in ("qwen-adapter", "qwen-merged", "llama-merged"):
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as d:
                package(kind, Path(d))
                with self.assertRaisesRegex(ValueError, "Bielik adapter or native packages only"):
                    dry("backend", d, "--bielik-tools-dir", d)

    def test_advanced_template_required_only_for_30(self):
        for model, advanced in launch.BIELIK_CHAT_MODELS.items():
            with self.subTest(model=model), tempfile.TemporaryDirectory() as d:
                root = Path(d)
                self.make_package(root, model)
                tools = self.make_tools(root)
                (tools / "bielik_advanced_chat_template.jinja").unlink()
                if advanced:
                    with self.assertRaisesRegex(ValueError, "bielik_advanced_chat_template.jinja"):
                        dry("backend", d, "--bielik-tools-dir", d)
                else:
                    dry("backend", d, "--bielik-tools-dir", d)


def test_bielik_adapter_requires_external_checkout(tmp_path,monkeypatch):
    import pytest
    monkeypatch.delenv('BIELIK_TOOLS_DIR',raising=False)
    package('llama-adapter',tmp_path)
    with pytest.raises(ValueError,match='parsers are required'):
        dry('backend',str(tmp_path))


def test_qwen_adapter_requires_its_own_builtin_parsers(tmp_path,monkeypatch):
    monkeypatch.setenv('BIELIK_TOOLS_DIR','/not-used-for-qwen')
    package('qwen-adapter',tmp_path)
    cmd=dry('backend',str(tmp_path))['command']
    assert cmd[cmd.index('--tool-call-parser')+1]=='qwen3_coder'
    assert cmd[cmd.index('--reasoning-parser')+1]=='qwen3'
    assert '--enable-auto-tool-choice' in cmd and '--tool-parser-plugin' not in cmd
