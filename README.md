# Bielik One

**Turn text into decisions with probabilities.** Serve Bielik One models through
one typed API, alongside ordinary OpenAI-compatible chat.

Ask a question about a document, message or other textual state:

- **`choice`** — choose one option.
- **`noul`** — estimate the probability of yes.
- **`score`** — rate the state on an ordered scale.
- **`labels`** — select any number of options, each with its own probability.

The decision API reads model outputs directly instead of generating and parsing
JSON. This repository contains the launcher and serving runtime; model weights
are hosted separately on Hugging Face.

[Quick start](#quick-start) · [Models](#models-and-serving-options) · [API reference](docs/api.md) · [Deployment](docs/serving.md#deployment-and-untrusted-text)

## Quick start

Requires **Python 3.12+**, an **NVIDIA GPU** and **vLLM 0.32.0**.
The 11B BF16 example below needs about **30 GB of GPU memory**. For smaller
GPUs, see [FP8 and memory settings](docs/serving.md#gpu-memory-and-performance).
For Apple silicon, see the [MLX guide](docs/serving.md#apple-silicon-mlx).

### 1. Install

```bash
pip install "bielik-one[vllm] @ git+https://github.com/speakleash/bielik-one"
```

### 2. Start the model and API

Run these in separate terminals. The model downloads to the Hugging Face cache
on first use; a local model directory works too.

```bash
# Terminal 1: GPU backend
bielik-one backend speakleash/Bielik-One-11B-v3.0-r1
```

```bash
# Terminal 2: HTTP API
bielik-one frontend speakleash/Bielik-One-11B-v3.0-r1
```

The API listens on **http://127.0.0.1:8026**. Interactive API docs are at
[/docs](http://127.0.0.1:8026/docs); readiness is at
[/health](http://127.0.0.1:8026/health).

### 3. Ask a question

```bash
curl http://127.0.0.1:8026/v1/systemone \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "decision",
    "state": "The invoice is unpaid. Its payment deadline was yesterday.",
    "questions": {
      "status": {
        "type": "choice",
        "instructions": "What is the status of this invoice?",
        "criteria": {
          "paid": "Paid",
          "overdue": "Unpaid and past the deadline",
          "not_due": "Unpaid, deadline not reached"
        }
      }
    }
  }'
```

The answer contains the selected option, probabilities and confidence. Put
several questions about the same state in one request to reuse cached prefixes.
See the [API reference](docs/api.md) for all four types and response fields.

## Models and serving options

Published decision models:

- [Bielik One 11B v3.0](https://huggingface.co/speakleash/Bielik-One-11B-v3.0-r1)
- [Bielik One 11B v3.1](https://huggingface.co/speakleash/Bielik-One-11B-v3.1-r1)
- [Bielik One 7B](https://huggingface.co/speakleash/Bielik-One-7B-v3.0-r1)
- [Bielik One 4B](https://huggingface.co/speakleash/Bielik-One-4B-r1)

Each has a **merged** export and a **`-Adapter`** package. The launcher detects
the package type. The 11B and 7B frontends default to port **8026**; 4B uses **8012**.

**Merged model:** use the quick-start commands with the model's repo ID. No heads
plugin is needed.

**LoRA adapter:** use the corresponding `-Adapter` repo and install the
[decision-heads plugin](docs/serving.md#adapter-and-merged-packages). One copy of
the original base weights serves both chat and decisions; the adapter is selected
only for decisions.

**Original Bielik 3.2:** use its existing answer tokens, with no adapter or extra
heads. Add `--native-tokens` to both commands:

```bash
bielik-one backend /path/to/bielik-3.2 --native-tokens
bielik-one frontend /path/to/bielik-3.2 --native-tokens
```

This serves decisions as `decision` and ordinary chat as `base`, on port **8026**.
In this mode, `labels` is always a series of independent `noul` questions. See
[Native Bielik 3.2](docs/native.md) for FP8, calibration and chat parser setup.

For **one server and one port**, the [vLLM middleware](docs/serving.md#single-server-middleware-adapter-packages)
supports adapter packages; [native middleware](docs/native.md#single-server-middleware)
supports original Bielik 3.2. Both expose `/v1/systemone` directly in vLLM,
without a separate frontend.

## Defaults that affect results

- **BF16** is the default precision. `--fp8` is optional and can change probabilities
  and decisions. See [GPU memory and performance](docs/serving.md#gpu-memory-and-performance).
- **Read twice is on**, with a **4096-token threshold** before repetition. Set
  `"read_twice": false` for the standard prompt; set
  `"read_twice_max_input_tokens": null` for unrestricted repetition within context.
- **Calibration loads automatically** when the model provides `calibration.json`.
  Send `"temperature": 1` for raw probabilities, or start with `--no-calibration`.
- **Labels use a dedicated head** in the decision packages by default. Their
  `"mode": "per_option"` alternative asks one yes/no question per option.
  Native Bielik 3.2 always uses the latter protocol.

To reproduce model-card numbers, match the [benchmark settings](docs/benchmarking.md#benchmarking)
rather than relying on serving defaults. `confidence` describes the output
distribution; it is not a calibrated probability of correctness.

## Documentation

- [API reference](docs/api.md) — requests, answers, confidence, labels modes, facts and errors.
- [Serving](docs/serving.md) — adapters, merged weights, GPU settings, middleware, MLX and deployment.
- [Native Bielik 3.2](docs/native.md) — original answer tokens, calibration, FP8 and chat parsers.
- [Calibration and benchmarking](docs/benchmarking.md) — probability metrics, inference profiles and comparison caveats.

## Development

```bash
pip install -e ".[test]"
pytest -q
```

Tests use synthetic backends and need no GPU. See [CONTRIBUTING.md](CONTRIBUTING.md)
for contributions and private security reporting.

## Contributors

- [Krzysztof Wróbel](https://github.com/djstrong) — Bielik One decision models, inference engine, calibration, serving tools and tests.
- [Filip Reka](https://github.com/FIlipHand) — original vLLM middleware integration, serving chat and typed decisions through one API server.
- [Grzegorz Brzezinka](https://github.com/agentGreg) — Apple-silicon (MLX) backend for adapter packages ([#5](https://github.com/speakleash/bielik-one/pull/5)) and opt-in deterministic calendar and number facts ([#6](https://github.com/speakleash/bielik-one/pull/6)).

Thank you to everyone who contributes code, reports issues and helps improve Bielik One.

## License

[Apache License 2.0](LICENSE). Model weights are distributed separately under
their own licenses.
