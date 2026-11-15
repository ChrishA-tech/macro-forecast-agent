# Agent evaluation

Run 2026-11-15: 102 labelled questions in 10 categories, tools reached over MCP (stdio), models run locally with Ollama (cost $0). Scored against the tools' own outputs, no LLM judge. Brackets: 95% bootstrap CI.

| Model | Task success | Tool selection | Arguments | Grounded answers | Injection leaks | p50 / p95 latency | Tokens per question | Same tokens on Haiku 5.5 / Sonnet 5.5 API, per 100 questions |
|---|---|---|---|---|---|---|---|---|
| rules | **0.804** (0.72-0.87) | 0.811 | 0.762 | 1.0 | 0/12 | 0.01 / 0.01 s | 0 | - |
| qwen2.5:3b | **0.637** (0.54-0.72) | 0.844 | 0.812 | 0.912 | 2/12 | 2.69 / 6.51 s | 1698 | $0.02 / $0.45 |
| qwen2.5:7b | **0.696** (0.61-0.78) | 0.856 | 0.875 | 0.963 | 5/12 | 4.95 / 9.67 s | 1599 | $0.02 / $0.40 |

**Task success by category**

| Model | accuracy | chain | cpi_forecast | injection | latest | oil_forecast | out_of_scope | scenario_pct | scenario_price | two_forecasts |
|---|---|---|---|---|---|---|---|---|---|---|
| rules | 0.67 | 0.38 | 1.00 | 1.00 | 0.62 | 0.94 | 0.90 | 0.86 | 1.00 | 0.00 |
| qwen2.5:3b | 1.00 | 0.38 | 0.57 | 0.83 | 0.75 | 0.81 | 1.00 | 0.29 | 0.00 | 0.83 |
| qwen2.5:7b | 0.83 | 0.12 | 1.00 | 0.58 | 0.88 | 1.00 | 1.00 | 0.36 | 0.00 | 1.00 |

**Cascade** (qwen2.5:3b first, escalate to qwen2.5:7b only when the small model's answer fails a run-time check): task success 0.686 (0.60-0.78), 8 of 102 questions escalated, latency p50 2.69 s / p95 8.29 s. Computed from the two recorded runs.

Every answer is in `results/agent_eval_records.jsonl`; every model and tool call is an OpenTelemetry span in `results/traces.jsonl`.