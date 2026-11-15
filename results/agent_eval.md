# Agent evaluation

Run 2026-11-15: 114 labelled questions in 11 categories, tools reached over MCP (stdio), models run locally with Ollama (cost $0). Scored against the tools' own outputs, no LLM judge. Brackets: 95% bootstrap CI.

| Model | Task success | Tool selection | Arguments | Grounded answers | Injection: leaks to user / model attempts | p50 / p95 latency | Tokens per question | Same tokens on Haiku 5.5 / Sonnet 5.5 API, per 100 questions |
|---|---|---|---|---|---|---|---|---|
| rules | **0.825** (0.75-0.90) | 0.811 | 0.762 | 1.0 | 0/24 / 0 | 0.01 / 0.01 s | 0 | - |
| qwen2.5:3b | **0.807** (0.74-0.88) | 0.967 | 0.875 | 0.875 | 0/24 / 9 | 2.88 / 6.1 s | 2098 | $0.03 / $0.53 |
| qwen2.5:7b | **0.956** (0.91-0.99) | 0.967 | 0.975 | 0.963 | 0/24 / 10 | 5.18 / 11.78 s | 2004 | $0.02 / $0.49 |

**Task success by category**

| Model | accuracy | chain | cpi_forecast | injection | injection_new | latest | oil_forecast | out_of_scope | scenario_pct | scenario_price | two_forecasts |
|---|---|---|---|---|---|---|---|---|---|---|---|
| rules | 0.67 | 0.38 | 1.00 | 1.00 | 1.00 | 0.62 | 0.94 | 0.90 | 0.86 | 1.00 | 0.00 |
| qwen2.5:3b | 0.83 | 0.75 | 0.79 | 1.00 | 1.00 | 0.75 | 0.69 | 1.00 | 0.93 | 0.12 | 0.83 |
| qwen2.5:7b | 1.00 | 0.88 | 1.00 | 1.00 | 1.00 | 0.75 | 1.00 | 0.90 | 0.93 | 1.00 | 1.00 |

**Cascade** (qwen2.5:3b first, escalate to qwen2.5:7b only when the small model's answer fails a run-time check): task success 0.886 (0.82-0.94), 9 of 114 questions escalated, latency p50 2.88 s / p95 8.68 s. Computed from the two recorded runs.

**v1 vs v2** (same scorer; v1 answers re-scored from `results/v1/`, v2 = this run). v2 changes: the scenario tool takes a target price itself, clearer chain-tool description, output guard against verbatim prompt disclosure, 12 new held-out injection attacks (`injection_new`, v2 only).

| Model | Success v1 → v2 (original 102) | scenario_price | chain | Injection leaks to user v1 → v2 | Held-out attacks leaked (v2) | Output-guard false positives (v2) |
|---|---|---|---|---|---|---|
| rules | 0.804 → **0.804** | 1.00 → 1.00 | 0.38 → 0.38 | 0/12 → 0/12 | 0/12 | 0 |
| qwen2.5:3b | 0.755 → **0.784** | 0.00 → 0.12 | 0.88 → 0.75 | 0/12 → 0/12 | 0/12 | 0 |
| qwen2.5:7b | 0.804 → **0.951** | 0.00 → 1.00 | 0.12 → 0.88 | 2/12 → 0/12 | 0/12 | 0 |

Every answer is in `results/agent_eval_records.jsonl`; every model and tool call is an OpenTelemetry span in `results/traces.jsonl`.