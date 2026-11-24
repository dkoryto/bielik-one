# Calibration and benchmarking

[Back to README](../README.md) · [API reference](api.md)

## Calibration

The published adapter and merged decision packages ship `calibration.json`: one temperature per question type, fitted on validation data only. The API applies it by default in every request mode: any number of option orders, `read_twice` and `questions_before_state`.

The temperatures were fitted for one option order on the standard prompt, and they carry over: for read-twice prompts and FP8 serving, temperatures refitted for those settings lowered test NLL by at most 0.012 against the shipped ones (Bielik-One-11B-v3.0-r1, Bielik-One-7B-v3.0-r1 and Bielik-One-4B-r1 for read-twice; these and Bielik-One-11B-v3.1-r1 for FP8). With 2 or 4 option orders and with `questions_before_state` the shipped temperatures gave a lower NLL than raw probabilities for every categorical type in a check on native validation (Bielik-One-4B-r1); a temperature fitted for `questions_before_state` would be somewhat higher.

Calibration never changes argmax or 0.5-threshold decisions; the `score` field is the expected level and does change. Raw probabilities: start the API with `--no-calibration`, or send `"temperature": 1`. `/health` reports whether calibration is active. The temperatures were fitted on bfloat16 outputs with the standard prompt.

For original Bielik 3.2 checkpoints, see [native calibration](native.md#calibration): the table is loaded beside the original weights and labels use their own temperature.

## Benchmarking

The reference results below concern the published adapter and merged decision models. Native Bielik 3.2 has its own labels protocol and calibration provenance (see [Native Bielik 3.2](native.md)).

The request options below change the numbers you get, so a result is only meaningful together with the settings it was measured with. Report them, and when you compare with another decision model, match its settings instead of comparing defaults: if that model reads the input twice or averages two option orders by default, run these models the same way too.

### Reference profile

bfloat16, one option order (`permutations: 1`), `read_twice: false`, `questions_before_state: false`, labels in `"mode": "head"`. This is the setting of the results on the model cards.
### Probabilities

Temperature scaling preserves categorical argmax and the 0.5 labels threshold, but changes probabilities and the expected `score` value. For probability metrics (NLL, Brier, ECE, KL) say which you used: the calibrated default, or raw with `"temperature": 1`. Calibration applies in every mode (see [Calibration](#calibration)), so raw probabilities need `"temperature": 1` explicitly.
### Labels

`"mode": "head"` reads all options in one pass. `"mode": "per_option"` asks one yes/no question per option, which is how decision models without a multi-label head answer such questions, so use it when comparing with them. For Bielik-One-4B-r1 it raises exact-set accuracy by 6.0 percentage points on the native test split, at one read per option.
### Read twice (default)

`"read_twice": true` repeats the input when the complete standard prompt has at most `read_twice_max_input_tokens` tokens (default **4096**, before repetition). Longer prompts use the standard input. Set `"read_twice": false` to disable it, or `"read_twice_max_input_tokens": null` to remove the length threshold. The exact repeated prompt plus one output token must always fit the server context; otherwise adapter and merged packages use the standard prompt; native Bielik 3.2 rejects a repeated prompt that exceeds context with 422. If the standard prompt also exceeds context, the request fails without truncation.

Each answer includes a compact `prompt` object: `variant` (`"standard"` or `"read_twice"`), `requested_read_twice`, `reason`, the effective `read_twice_max_input_tokens`, `max_context_tokens`, and read counts. Send `"prompt_details": true` for the per-read token lengths. The threshold is checked once per question: if any permutation or per-option label read exceeds it, all use the standard prompt. If repetition is eligible but exceeds context, adapter/merged packages fall back to standard while native 3.2 returns 422.

For `choice`, `labels`, and any non-semantic readout, both copies use the same current permutation, so each letter always denotes the same option within a prompt. Only semantic `noul` and `score` keep the original order in the first copy for a shared KV-cache prefix; the second copy is permuted while `yes`/`no` and score digits retain their meanings. This semantic fixed-prefix multi-permutation variant has not yet been rebenchmarked for quality.

This changes the behavior of requests that omit `read_twice`; reference benchmark requests must explicitly send `false`. The 4096 threshold is a provisional cost/accuracy tradeoff from 4B native validation, not a validated optimum for every model.

Earlier unrestricted read-twice measurements raised single-choice accuracy on the native test split by 3.5 percentage points (Bielik-One-11B-v3.0-r1), 3.3 (Bielik-One-7B-v3.0-r1) and 1.5 (Bielik-One-4B-r1), at about twice the prompt length; it was not measured for Bielik-One-11B-v3.1-r1.
### Facts

`"facts": true` (off in the reference profile) appends computed calendar and number facts to the state (see [Computed facts](api.md#computed-facts)). On the public dev split of the Tacit decision benchmark (2,852 items in Polish, English, German and Norwegian; default read twice, one option order, bf16 MLX conversions of the published models) it raised accuracy on the rule-based items by 9.5 percentage points (Bielik-One-4B-r1, 69.9% to 79.4%) and 10.9 (Bielik-One-11B-v3.1-r1, 66.2% to 77.1%), with the other items unchanged (within 0.1 point). Report it when you use it.
### Option orders

`"permutations": N` (up to 16) averages option orders to reduce position bias.
For a standard question, the logical cost is up to N reads, capped by the number
of distinct option orders. For `labels` in `per_option` mode with K options, the
cost is **K × min(N, 2)** reads per question: each option becomes a binary question
with at most two orders. This also applies to native Bielik 3.2 labels.

These are logical decision reads. Large native choice questions can require
additional HTTP chunk reads; see [native token reads](native.md).
### Repeatability

Batching under concurrent load changes floating-point order, so a few decisions near the boundary can differ between identical requests (98-100% agreement in our A100 measurements). For repeatable runs keep the `--max-num-seqs` profile fixed and report it. `--fp8` changes 2-18% of decisions depending on model and type; benchmark in bfloat16 unless FP8 is what you deploy.
### Long inputs

Nothing is truncated: a prompt over the 32,768-token context is refused with 422. Count it as an error instead of dropping the question, so every system is scored on the same denominator.
### Request layout

Questions about the same state can go in one request to reuse cached prefixes. Record the batching and concurrency profile, since floating-point differences can affect boundary decisions.
### External harnesses

`choice`, `noul` and `score` follow the TypeSafe System One schema, so a harness that calls `/v1/systemone` (for example a public Typed Decisions scorer) works against this API unchanged; `labels` and the options above are extensions.
