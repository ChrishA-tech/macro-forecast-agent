"""
LangGraph agent:

  input_guard ──unsafe──▶ refuse ─────────────────────────────────────────────▶ END
      │safe
      ▼
    route ──out_of_scope──▶ help ─────────────────────────────────────────────▶ END
      │
      ▼
  run_tools ──▶ compose (template answer; Claude answer if enabled) ──▶ ground_check
                                                                            │
           (LLM answer with numbers not in the tool facts → template answer)│
                                                                            ▼
                                                                    output_guard ──▶ END

Components come from the three project packages (vendored in vendor/):
promptguard (input/output guardrails), oilcast (oil models), ukcpi (CPI models).
"""
from __future__ import annotations

import time
import uuid
from typing import Any, Callable, Dict, List, Optional, TypedDict

from . import VENDOR  # noqa: F401
from promptguard.guardrails import evaluate_prompt_safety
from functools import lru_cache
import os
from pathlib import Path
from promptguard.output_guard import redact_text

from .grounding import check_grounding, flatten_facts
from .router import Route, llm_route, rule_route
from .tools import (DataProvider, cpi_forecast, latest_data, model_accuracy, oil_cpi_scenario, oil_forecast,
                    oil_forecast_to_cpi, price_to_pct)

try:
    from langgraph.graph import END, StateGraph
    HAS_LANGGRAPH = True
except ImportError:                        # Python 3.9 / minimal installs
    HAS_LANGGRAPH = False
    END = "__END__"

LLM = Callable[[str, str], str]           # llm(system_prompt, user_message) -> text

ANSWER_PROMPT = """You are a UK macro forecasting assistant. Answer the question using ONLY the facts provided
(tool outputs). Every number you write must appear in the facts. Mention the model used, the forecast month,
and one sentence on uncertainty or accuracy if the facts include it. Scenario effects are associations from a
regression, not causal effects. 3-5 sentences, no headings."""


class AgentState(TypedDict, total=False):
    request_id: str
    question: str
    guard: Any
    route: Route
    tool_results: List[Dict[str, Any]]
    facts: Dict[str, Any]
    template_answer: str
    answer: str
    answer_source: str
    grounding: Dict[str, Any]
    redactions: List[str]
    status: str


def _fmt(v):
    return f"{v:+.2f}" if isinstance(v, float) else str(v)


def template_answer(route: Route, facts: Dict[str, Any]) -> str:
    f = facts
    if route.intent == "oil_forecast":
        target = f["forecast month"]
        t = (f"Using the {f['model']} model, Brent is forecast at ${f[f'Brent forecast for {target} (USD/bbl)']:.2f}/bbl for "
             f"{target}, versus ${f['latest Brent price (USD/bbl)']:.2f} in {f['latest month']} "
             f"({f['forecast change vs latest (%)']:+.1f}%).")
        rel = next((k for k in f if k.startswith("RMSE relative to naive")), None)
        rm = next((k for k in f if k.startswith("backtest RMSE")), None)
        if rm:
            t += f" In the rolling backtest its {rm.split('at ')[1].split(' horizon')[0]}-ahead RMSE was ${f[rm]}/bbl"
            t += f" ({f[rel]} times the naive forecast's error)." if rel else "."
        return t
    if route.intent == "cpi_forecast":
        t = (f"The ARX model (CPI lags, Bank Rate, Brent) forecasts UK CPI inflation at {f['ARX forecast of CPI inflation (%)']}% "
             f"for {f['forecast month']}, from {f['latest CPI inflation (%)']}% now; the AR model gives "
             f"{f['AR forecast of CPI inflation (%)']}% and the no-change benchmark {f['naive forecast of CPI inflation (%)']}%.")
        a = next((k for k in f if k.startswith("ARX backtest RMSE")), None)
        n = next((k for k in f if k.startswith("naive backtest RMSE")), None)
        if a and n:
            t += f" Backtest RMSE: ARX {f[a]}pp vs naive {f[n]}pp."
        return t
    if route.intent in ("oil_cpi_scenario", "oil_forecast_to_cpi"):
        pre = ""
        if route.intent == "oil_forecast_to_cpi":
            pre = (f"The {f['model']} oil forecast implies a {f['forecast change vs latest (%)']:+.1f}% move in Brent by "
                   f"{f['forecast month']}. ")
        hs = [k for k in f if k.startswith("CPI inflation change after")]
        def _m(k):
            n = k.split('after ')[1].split(' months')[0]
            return f"{n} month" + ("" if n == "1" else "s")
        parts = [f"{_fmt(f[k])}pp after {_m(k)} "
                 f"(95% CI {f[k.replace('CPI inflation change', '95% CI')][0]:+.2f} to {f[k.replace('CPI inflation change', '95% CI')][1]:+.2f})"
                 for k in hs]
        warn = (" Warning: this change is far outside the oil moves seen in the data, so the linear estimate is an "
                "extrapolation and should be treated with caution.") if f.get("extrapolation warning") else ""
        return (pre + f"A sustained {f['assumed Brent change (%)']:+.1f}% change in Brent is associated with UK CPI inflation "
                + "; ".join(parts) + ". These are regression associations, not causal effects." + warn)
    if route.intent == "model_accuracy":
        items = [f"{k.replace(' 1-month RMSE relative to naive', '')}: {v}" for k, v in f.items() if k != "note"]
        return ("1-month RMSE relative to the naive benchmark (below 1 = better): " + "; ".join(items) + ".") if items \
            else "No backtest results are available yet."
    if route.intent == "latest_data":
        return "; ".join(f"{k}: {v}" for k, v in f.items()) + "."
    return ""


HELP = ("I can forecast Brent crude oil and UK CPI inflation, estimate how oil-price changes relate to UK inflation, "
        "and report model accuracy. Try: \"Forecast Brent for the next 6 months\", \"What will UK inflation be in 12 "
        "months?\", \"What happens to UK inflation if oil rises 20%?\", \"How accurate are the models?\"")


CLASSIFIER_PATH = (Path(__file__).resolve().parent.parent / "vendor_data" / "langgraph-prompt-guard" / "models"
                   / "prompt_injection_clf.joblib")


@lru_cache(maxsize=1)
def default_classifier():
    """
    The prompt-guard repo's trained classifier, only if MACROAGENT_CLASSIFIER=on (else rules only).
    Opt-in because it wrongly flags many benign prompts unlike its training data (see langgraph-prompt-guard).
    """
    if os.environ.get("MACROAGENT_CLASSIFIER") != "on" or not CLASSIFIER_PATH.exists():
        return None
    try:
        from promptguard.classifier import InjectionClassifier
        return InjectionClassifier.load(CLASSIFIER_PATH)
    except Exception:          # scikit-learn missing or incompatible pickle: fall back to rules
        return None


def build_agent(provider: DataProvider, llm: Optional[LLM] = None, strict: bool = True, classifier=None):

    def input_guard(s: AgentState) -> dict:
        if classifier is not None:
            from promptguard.classifier import hybrid_assess
            return {"guard": hybrid_assess(s["question"], classifier, strict)}
        return {"guard": evaluate_prompt_safety(s["question"], strict_mode=strict)}

    def after_guard(s: AgentState) -> str:
        return "route" if s["guard"].is_safe else "refuse"

    def route(s: AgentState) -> dict:
        q = s["guard"].sanitized_prompt
        return {"route": llm_route(q, llm) if llm else rule_route(q)}

    def after_route(s: AgentState) -> str:
        return "help" if s["route"].intent == "out_of_scope" else "run_tools"

    def run_tools(s: AgentState) -> dict:
        r = s["route"]
        try:
            if r.intent == "oil_forecast":
                res = [oil_forecast(provider, r.horizon)]
            elif r.intent == "cpi_forecast":
                res = [cpi_forecast(provider, r.horizon)]
            elif r.intent == "oil_cpi_scenario":
                pct = r.oil_change_pct if r.oil_change_pct is not None else \
                    (price_to_pct(provider, r.target_price) if r.target_price else 10.0)
                res = [latest_data(provider), oil_cpi_scenario(provider, pct)] if r.target_price else [oil_cpi_scenario(provider, pct)]
            elif r.intent == "oil_forecast_to_cpi":
                res = [oil_forecast_to_cpi(provider, r.horizon)]
            elif r.intent == "model_accuracy":
                res = [model_accuracy(provider)]
            else:
                res = [latest_data(provider)]
            return {"tool_results": res, "facts": flatten_facts(res)}
        except Exception as e:     # tool failure -> graceful error answer
            return {"tool_results": [], "facts": {}, "status": "error",
                    "answer": f"The forecasting tools failed: {type(e).__name__}: {e}"}

    def after_tools(s: AgentState) -> str:
        return END if s.get("status") == "error" else "compose"

    def compose(s: AgentState) -> dict:
        tmpl = template_answer(s["route"], s["facts"])
        if llm is None:
            return {"template_answer": tmpl, "answer": tmpl, "answer_source": "template"}
        try:
            facts_txt = "\n".join(f"- {k}: {v}" for k, v in s["facts"].items())
            text = llm(ANSWER_PROMPT, f"Question: {s['question']}\n\nFacts:\n{facts_txt}")
            return {"template_answer": tmpl, "answer": text.strip(), "answer_source": "claude"}
        except Exception as e:
            return {"template_answer": tmpl, "answer": tmpl, "answer_source": f"template (LLM error: {type(e).__name__})"}

    def ground_check(s: AgentState) -> dict:
        ok, bad = check_grounding(s["answer"], s["facts"], s["question"])
        if ok or s["answer_source"] != "claude":
            return {"grounding": {"passed": ok, "ungrounded": bad}}
        return {"answer": s["template_answer"], "answer_source": "template (LLM answer failed grounding check)",
                "grounding": {"passed": False, "ungrounded": bad, "replaced": True}}

    def output_guard(s: AgentState) -> dict:
        clean, found = redact_text(s["answer"])
        return {"answer": clean, "redactions": found, "status": "success"}

    def refuse(s: AgentState) -> dict:
        return {"status": "blocked", "answer": "I can't help with that request: it was flagged by the input guardrail ("
                + ", ".join(s["guard"].matched_rules) + ")."}

    def help_node(s: AgentState) -> dict:
        return {"status": "out_of_scope", "answer": HELP}

    nodes = {"input_guard": input_guard, "route": route, "run_tools": run_tools, "compose": compose,
             "ground_check": ground_check, "output_guard": output_guard, "refuse": refuse, "help": help_node}
    cond = {"input_guard": (after_guard, {"route": "route", "refuse": "refuse"}),
            "route": (after_route, {"help": "help", "run_tools": "run_tools"}),
            "run_tools": (after_tools, {"compose": "compose", END: END})}
    edges = {"compose": "ground_check", "ground_check": "output_guard", "output_guard": END, "refuse": END, "help": END}

    if HAS_LANGGRAPH:
        g = StateGraph(AgentState)
        for n, fn in nodes.items():
            g.add_node(n, fn)
        g.set_entry_point("input_guard")
        for src, (router, mapping) in cond.items():
            g.add_conditional_edges(src, router, mapping)
        for src, dst in edges.items():
            g.add_edge(src, dst)
        return g.compile()
    from promptguard.workflow import _CompiledFallbackStateGraph
    return _CompiledFallbackStateGraph(nodes, "input_guard", cond, edges)


def ask(question: str, provider: DataProvider, llm: Optional[LLM] = None, classifier="auto") -> Dict[str, Any]:
    t0 = time.perf_counter()
    if classifier == "auto":
        classifier = default_classifier()
    state = build_agent(provider, llm, classifier=classifier).invoke({"request_id": uuid.uuid4().hex, "question": question})
    r = state.get("route")
    return {"request_id": state["request_id"], "status": state.get("status"), "answer": state.get("answer"),
            "answer_source": state.get("answer_source"), "intent": r.intent if r else None,
            "router": r.router if r else None, "facts": state.get("facts", {}),
            "sources": [t.get("source") for t in state.get("tool_results", [])],
            "grounding": state.get("grounding"), "redactions": state.get("redactions", []),
            "guard": {"risk_score": state["guard"].risk_score, "rules": state["guard"].matched_rules},
            "synthetic_data": provider.synthetic, "latency_ms": round((time.perf_counter() - t0) * 1000, 1)}
