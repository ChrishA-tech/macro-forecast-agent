# Macro Forecast Agent: guarded LangGraph + Claude over oil and UK inflation models

A **LangGraph** agent that answers questions about **Brent crude** and **UK CPI inflation** by calling
forecasting tools built in three companion projects, and uses **Claude** (optional) to route questions and
write answers that are **checked against the tool outputs**: every number in an answer must come from a tool, or
the answer is replaced by a deterministic one.

```
input_guard ──unsafe──▶ refuse
    │ safe
  route (Claude JSON route, validated; rule-based fallback) ──out of scope──▶ help
    │
  run_tools ─▶ compose (Claude or template) ─▶ ground_check ─▶ output_guard ─▶ answer
```

| Tool | What it does | From |
|---|---|---|
| `oil_forecast` | Brent forecast from the best model in the rolling backtest, with its backtest error | [oil-price-forecasting](https://github.com/ChrishA-tech/oil-price-forecasting) |
| `cpi_forecast` | UK CPI inflation forecasts (naive / AR / ARX) with backtest RMSE | [uk-cpi-forecasting](https://github.com/ChrishA-tech/uk-cpi-forecasting) |
| `oil_cpi_scenario` | "If Brent rises 20% (or goes to $100), what happens to UK inflation?" with 95% CIs and an extrapolation warning | uk-cpi-forecasting |
| `oil_forecast_to_cpi` | Chains the oil forecast into the CPI scenario | both |
| `model_accuracy`, `latest_data` | Backtest accuracy vs naive; latest data points and sources | both |
| Guardrails | Prompt-injection screening on input; PII/secret redaction on output | [langgraph-prompt-guard](https://github.com/ChrishA-tech/langgraph-prompt-guard) |

## Examples (real data)

<!-- EXAMPLES:START -->
_Not generated yet: `python -m macroagent.examples` after syncing real data._
<!-- EXAMPLES:END -->

## Run

```bash
pip install -r requirements.txt                 # + anthropic for Claude
python sync_vendor.py                           # copy packages, data and results from the sibling repos
python -m macroagent --demo                     # CLI demo questions
python -m macroagent "What happens to UK inflation if oil rises 20%?"
ANTHROPIC_API_KEY=... python -m macroagent "..."   # Claude routing + answers (grounding-checked)
streamlit run app/chat.py                       # chat UI (MACROAGENT_DEMO=1 for synthetic data)
uvicorn macroagent.api:app                      # POST /ask
python -m pytest -q                             # 25 tests on synthetic data, no API key needed
```

## Design notes

- **Grounding check:** numbers in the answer are compared with the tool facts (after rounding); horizons and
  years are allowed. A Claude answer that adds a number not in the facts is discarded and the template answer is
  returned, flagged in the response (`answer_source`).
- **Validated routing:** Claude's route must parse into a pydantic `Route` (allowed intents, horizon 1-24, oil
  change -90..500%); otherwise the rule-based router is used.
- **Scenarios are associational:** the oil → CPI effect comes from regression coefficients, not a causal model,
  and moves larger than anything in the data are flagged as extrapolation.
- **Guardrails first:** a flagged question never reaches the router or the LLM.
- `vendor/` contains copies of the three component packages (synced by `sync_vendor.py`); `vendor_data/` holds
  their cached real data and backtest results.
