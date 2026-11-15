"""/health, /ping and /v1/models fail fast on a backend that accepts connections but does not answer."""

import json
import socket
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from unittest.mock import patch

import httpx
import pytest
from fastapi.testclient import TestClient

from bielik_one.api import remote_vllm


class Engine:
    aliases = {"decision"}
    temperatures = {}
    calibration_source = None
    max_model_len = 32768

    def __init__(self, url):
        self.backend_url = url

    def close(self):
        pass


@pytest.fixture
def hung_backend():
    """Accepts TCP connections and never sends a byte."""
    server = socket.socket()
    server.bind(("127.0.0.1", 0))
    server.listen(16)
    held = []
    stop = threading.Event()

    def accept():
        server.settimeout(0.2)
        while not stop.is_set():
            try:
                held.append(server.accept()[0])
            except OSError:
                continue

    thread = threading.Thread(target=accept, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.getsockname()[1]}"
    stop.set()
    thread.join()
    for conn in held:
        conn.close()
    server.close()


@pytest.fixture
def slow_backend():
    """Answers every request after `delay` seconds."""
    delay = {"seconds": 0.0}

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def _answer(self):
            time.sleep(delay["seconds"])
            if self.command == "POST":
                self.rfile.read(int(self.headers.get("content-length") or 0))
            body = json.dumps({"data": [{"id": "chat"}, {"id": "decision"}], "ok": True}).encode()
            self.send_response(200)
            self.send_header("content-type", "application/json")
            self.send_header("content-length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        do_GET = do_POST = _answer

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{server.server_port}", delay
    server.shutdown()
    server.server_close()


SHORT = httpx.Timeout(0.3, connect=0.3)


def test_probe_timeout_is_seconds_not_the_generation_hour():
    assert remote_vllm.PROBE_TIMEOUT.read <= 10 and remote_vllm.PROBE_TIMEOUT.connect <= 5


@pytest.mark.parametrize("path", ["/health", "/ping"])
def test_health_fails_fast_when_the_backend_hangs(hung_backend, path):
    with patch.object(remote_vllm, "PROBE_TIMEOUT", SHORT), TestClient(
        remote_vllm.build_remote_app(Engine(hung_backend), chat_model="chat")
    ) as client:
        started = time.monotonic()
        response = client.get(path)
        assert time.monotonic() - started < 3
    assert response.status_code == 503
    assert response.json()["code"] == "BACKEND_UNAVAILABLE"


def test_models_listing_times_out_when_the_backend_hangs(hung_backend):
    with patch.object(remote_vllm, "PROBE_TIMEOUT", SHORT), TestClient(
        remote_vllm.build_remote_app(Engine(hung_backend), chat_model="chat")
    ) as client:
        started = time.monotonic()
        response = client.get("/v1/models")
        assert time.monotonic() - started < 3
    assert response.status_code == 504
    assert response.json()["code"] == "TIMEOUT"


def test_probes_pass_and_chat_keeps_its_long_timeout(slow_backend):
    url, delay = slow_backend
    with patch.object(remote_vllm, "PROBE_TIMEOUT", SHORT), TestClient(
        remote_vllm.build_remote_app(Engine(url), chat_model="chat")
    ) as client:
        assert client.get("/health").json()["status"] == "ok"
        assert [m["id"] for m in client.get("/v1/models").json()["data"]] == ["chat"]
        # A generation slower than the probe timeout is still proxied: only the probes are short.
        delay["seconds"] = 1.0
        response = client.post("/v1/chat/completions", json={"model": "chat", "messages": []})
        assert response.status_code == 200 and response.json()["ok"] is True
        assert client.get("/health").status_code == 503  # the same delay fails the probe
