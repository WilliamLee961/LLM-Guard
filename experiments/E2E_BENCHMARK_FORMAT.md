# E2E Benchmark Format

This document fixes the benchmark input/output format for real-model end-to-end pressure tests.
It is based on the current dataset format in `experiments/datasets/sample_guard_eval.jsonl`
and the current benchmark output shape in `experiments/run_guard_benchmark.py`.

## 1. Input Dataset

Each input row is a JSON object in JSONL format.

Required fields:

| field | type | meaning |
| --- | --- | --- |
| `id` | `str` | stable sample id |
| `query` | `str` | user query |
| `is_harmful` | `bool` | whether the sample is harmful |
| `category` | `str` | business or safety category |

Recommended example:

```json
{"id":"h-001","category":"physical_harm","is_harmful":true,"query":"..."}
```

## 2. Run-Level Naming

One benchmark run should have one `run_id`.

Recommended naming:

```text
{timestamp}_{dataset_name}_{runtime_mode}_{difficulty}_{tag}
```

Example:

```text
20260407_sample_guard_eval_fast_normal_smoke1
```

## 3. Per-Sample Log Format

Per-sample logs should be stored as JSONL.
One output file per baseline is recommended.

Required output file naming:

```text
{run_id}_{baseline}.jsonl
```

Required per-sample fields:

| field | type | meaning |
| --- | --- | --- |
| `run_id` | `str` | unique run id |
| `timestamp` | `str` | run timestamp, `YYYYMMDD_HHMMSS` |
| `dataset_name` | `str` | dataset stem, for example `sample_guard_eval` |
| `dataset_path` | `str` | absolute or project-relative dataset path |
| `sample_id` | `str` | same as input `id` |
| `query` | `str` | original user query |
| `category` | `str` | sample category |
| `is_harmful` | `bool` | ground-truth harmful label |
| `baseline` | `str` | `no_guard`, `input_only`, `output_only`, `full_sidecar` |
| `runtime_mode` | `str` | `accurate` or `fast` |
| `difficulty` | `str` | guard detection difficulty |
| `judge_difficulty` | `str` | difficulty used by judge pass |
| `blocked_stage` | `str|null` | `input`, `output`, or `null` |
| `action` | `str` | final guard action name |
| `response_strategy_action` | `str` | normalized action from `response_strategy.action` |
| `response_strategy` | `object` | full response strategy object |
| `raw_answer` | `str` | direct model answer before final guard rewrite |
| `final_answer` | `str` | final answer shown to user |
| `judge_pred_violation` | `int` | `0` or `1` from judge pass |
| `judge_overall` | `str` | judge overall risk level, for example `L1` |
| `judge_risk_profile` | `object` | full judge profile |
| `latency_ms` | `float` | end-to-end latency for this sample |
| `raw_answer_chars` | `int` | character count of `raw_answer` |
| `final_answer_chars` | `int` | character count of `final_answer` |
| `error` | `str|null` | error message if the sample failed |

Optional fields:

| field | type | meaning |
| --- | --- | --- |
| `input_violation` | `int` | only for `full_sidecar` or full graph logs |
| `output_violation` | `int` | only for `full_sidecar` or full graph logs |
| `router_ctx_in` | `object` | input-side router output |
| `router_ctx_out` | `object` | output-side router output |
| `input_risk_profile` | `object` | input-side risk profile |
| `output_risk_profile` | `object` | output-side risk profile |
| `rag_docs_count` | `int` | number of retrieved docs |
| `trace` | `object` | timing or debug trace for internal stages |

Recommended JSONL example:

```json
{
  "run_id":"20260407_sample_guard_eval_fast_normal_smoke1",
  "timestamp":"20260407_213000",
  "dataset_name":"sample_guard_eval",
  "dataset_path":"experiments/datasets/sample_guard_eval.jsonl",
  "sample_id":"h-001",
  "query":"...",
  "category":"physical_harm",
  "is_harmful":true,
  "baseline":"full_sidecar",
  "runtime_mode":"fast",
  "difficulty":"normal",
  "judge_difficulty":"strict",
  "blocked_stage":"input",
  "action":"blocked_input_refuse",
  "response_strategy_action":"REFUSE",
  "response_strategy":{"action":"REFUSE","overall_risk_level":"L4"},
  "raw_answer":"",
  "final_answer":"...",
  "judge_pred_violation":0,
  "judge_overall":"L1",
  "judge_risk_profile":{"overall_risk_level":"L1"},
  "latency_ms":4321.55,
  "raw_answer_chars":0,
  "final_answer_chars":318,
  "error":null
}
```

## 4. Run Summary JSON Format

Each benchmark run should also produce one summary JSON file:

```text
{run_id}_summary.json
```

Required top-level fields:

| field | type | meaning |
| --- | --- | --- |
| `run_id` | `str` | unique run id |
| `timestamp` | `str` | run timestamp |
| `dataset` | `str` | dataset path |
| `dataset_name` | `str` | dataset stem |
| `difficulty` | `str` | guard difficulty |
| `runtime_mode` | `str` | `accurate` or `fast` |
| `judge_difficulty` | `str` | judge difficulty |
| `num_samples` | `int` | total number of samples |
| `baselines` | `object` | summary per baseline |

Each `baselines.{baseline}` object should contain:

| field | type | meaning |
| --- | --- | --- |
| `num_samples` | `int` | number of rows |
| `num_harmful` | `int` | harmful rows |
| `num_benign` | `int` | benign rows |
| `asr_harmful_output_rate` | `float` | harmful outputs that still passed the judge |
| `harmful_blocked_rate` | `float` | harmful samples blocked at input or output |
| `benign_refusal_rate` | `float` | benign samples refused or safe-answered |
| `benign_helpful_rate` | `float` | benign samples both safe and useful |
| `avg_latency_ms` | `float` | average latency |
| `p50_latency_ms` | `float` | median latency |
| `p95_latency_ms` | `float` | p95 latency |
| `p99_latency_ms` | `float` | p99 latency |
| `error_rate` | `float` | sample failure rate |

## 5. Flat Summary CSV Header

For comparison across multiple runs, also keep one flat CSV file.
One row per `(run_id, baseline)`.

Required CSV header:

```text
run_id,timestamp,dataset_name,dataset_path,baseline,runtime_mode,difficulty,judge_difficulty,num_samples,num_harmful,num_benign,asr_harmful_output_rate,harmful_blocked_rate,benign_refusal_rate,benign_helpful_rate,avg_latency_ms,p50_latency_ms,p95_latency_ms,p99_latency_ms,error_rate
```

## 6. Timing Scope

`latency_ms` must use end-to-end timing.

Start point:
- right before entering the tested entrypoint, such as `run_safe_qa(...)`

End point:
- right after the final result object is returned and `final_answer` is available

Do not mix cold-start and steady-state runs in one report unless explicitly labeled.

## 7. Error Logging Rules

If one sample fails:

1. keep the row
2. set `error` to the error message
3. set `latency_ms` to the observed elapsed time before failure
4. keep `final_answer` as empty string if not available

Recommended failure example:

```json
{
  "sample_id":"b-023",
  "baseline":"full_sidecar",
  "runtime_mode":"fast",
  "latency_ms":12001.77,
  "final_answer":"",
  "error":"TimeoutError: request exceeded 12s"
}
```

## 8. Minimal First-Version Decision

For the first real-model benchmark version, lock the standard as:

- input dataset: existing JSONL with `id/category/is_harmful/query`
- per-sample log: JSONL
- per-run summary: JSON
- cross-run comparison: CSV
- runtime modes to compare: `accurate`, `fast`
- baseline set to compare: `no_guard`, `input_only`, `output_only`, `full_sidecar`

This keeps the format aligned with the current repo and makes later extensions easy.
