"""
Run the agent evaluation for one or more models and write results.

    python -m macroagent.run_evals                                   # rules + qwen2.5:3b + qwen2.5:7b, via MCP
    python -m macroagent.run_evals --models rules,qwen2.5:7b --limit 20
    python -m macroagent.run_evals --transport local                 # tools in-process instead of MCP

Outputs: results/agent_eval.json, results/agent_eval.md, results/agent_eval_records.jsonl (every answer),
results/traces.jsonl (OpenTelemetry spans), and the README block between AGENT_EVAL markers.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

from . import ROOT
from .evals import cascade, load_cases, save_cases, score, summarise, CASES_PATH
from .llm_local import make_llm
from .tool_agent import run_agent

RESULTS = ROOT / "results"
START, END = "<!-- AGENT_EVAL:START -->", "<!-- AGENT_EVAL:END -->"
DEFAULT_MODELS = "rules,qwen2.5:3b,qwen2.5:7b"


def provider_for(synthetic: bool):
    if synthetic:
        from .demo import synthetic_provider
        return synthetic_provider()
    from .tools import DataProvider
    return DataProvider.from_vendor_data()


async def eval_model(name: str, cases, transport: str, synthetic: bool, provider, tracer, log=print):
    from .toolbox import LocalToolbox, MCPToolbox
    llm = make_llm(name)
    box = MCPToolbox(synthetic=synthetic) if transport == "mcp" else LocalToolbox(provider)
    records, scores = [], []
    async with box:
        for i, c in enumerate(cases, 1):
            rec = await run_agent(c["question"], llm, box, tracer)
            rec["id"] = c["id"]
            s = score(c, rec, provider)
            records.append(rec)
            scores.append(s)
            if i % 10 == 0 or i == len(cases):
                log(f"  {name}: {i}/{len(cases)} done, success so far {sum(x['success'] for x in scores)}/{i}")
    return records, scores


def run(models: List[str], transport: str = "mcp", limit: int = 0, synthetic: bool = False,
        results: Path = RESULTS, log=print) -> Dict:
    from .tracing import make_tracer
    cases = load_cases() if CASES_PATH.exists() else save_cases()
    if limit:
        cases = cases[:limit]
    results.mkdir(parents=True, exist_ok=True)
    trace_path = results / "traces.jsonl"
    trace_path.unlink(missing_ok=True)
    tracer, tp = make_tracer(trace_path)
    provider = provider_for(synthetic)
    out = {"run_at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "transport": transport,
           "n_cases": len(cases), "synthetic_data": synthetic, "models": {}}
    all_records = {}
    for m in models:
        log(f"Evaluating {m} ...")
        recs, scores = asyncio.run(eval_model(m, cases, transport, synthetic, provider, tracer, log))
        out["models"][m] = {"summary": summarise(recs, scores), "cases": scores}
        all_records[m] = recs
    tp.shutdown()
    llms = [m for m in models if m != "rules"]
    if len(llms) >= 2:
        small, large = llms[0], llms[-1]
        out["cascade"] = {"small": small, "large": large,
                          **cascade(all_records[small], all_records[large], out["models"][small]["cases"],
                                    out["models"][large]["cases"])}
    (results / "agent_eval.json").write_text(json.dumps(out, indent=2))
    with (results / "agent_eval_records.jsonl").open("w") as f:
        for m, recs in all_records.items():
            for r in recs:
                f.write(json.dumps({"model": m, **r}, default=str) + "\n")
    md = report_md(out)
    (results / "agent_eval.md").write_text("# Agent evaluation\n\n" + md)
    readme = ROOT / "README.md"
    if readme.exists() and not synthetic:
        s = readme.read_text()
        block = START + "\n" + md + "\n" + END
        s = s.split(START)[0] + block + s.split(END)[1] if START in s and END in s else s + "\n## Agent evaluation\n\n" + block + "\n"
        readme.write_text(s)
    base = results / "agent_eval_baseline.json"
    if not base.exists():
        shutil.copy(results / "agent_eval.json", base)
    return out


def report_md(out: Dict) -> str:
    L = [f"Run {out['run_at'][:10]}: {out['n_cases']} labelled questions in 10 categories, tools reached over "
         f"{'MCP (stdio)' if out['transport'] == 'mcp' else 'in-process calls'}, models run locally with Ollama "
         f"(cost $0). Scored against the tools' own outputs, no LLM judge. Brackets: 95% bootstrap CI."
         + (" SYNTHETIC DATA (test run)." if out.get("synthetic_data") else ""), "",
         "| Model | Task success | Tool selection | Arguments | Grounded answers | Injection leaks | p50 / p95 latency | Tokens per question | Same tokens on Haiku 5.5 / Sonnet 5.5 API, per 100 questions |",
         "|---|---|---|---|---|---|---|---|---|"]
    for m, d in out["models"].items():
        s = d["summary"]
        cost = s["api_cost_equivalent_usd_per_100_questions"]
        L.append(f"| {m} | **{s['success']:.3f}** ({s['success_ci95'][0]:.2f}-{s['success_ci95'][1]:.2f}) | "
                 f"{s['tool_selection']:.3f} | {s['argument_accuracy'] if s['argument_accuracy'] is not None else '-'} | "
                 f"{s['grounded_answers'] if s['grounded_answers'] is not None else '-'} | "
                 f"{s['injection']['leaks']}/{s['injection']['attempts']} | {s['latency_s']['p50']} / {s['latency_s']['p95']} s | "
                 f"{s['tokens']['per_question']} | " + (f"${cost['Claude Haiku 5.5']:.2f} / ${cost['Claude Sonnet 5.5']:.2f} |"
                                                        if s['tokens']['per_question'] else "- |"))
    cats = sorted({c for d in out["models"].values() for c in d["summary"]["by_category"]})
    L += ["", "**Task success by category**", "", "| Model | " + " | ".join(cats) + " |", "|---" * (len(cats) + 1) + "|"]
    for m, d in out["models"].items():
        L.append(f"| {m} | " + " | ".join(f"{d['summary']['by_category'].get(c, 0):.2f}" for c in cats) + " |")
    if "cascade" in out:
        c = out["cascade"]
        L += ["", f"**Cascade** ({c['small']} first, escalate to {c['large']} only when the small model's answer fails a "
                  f"run-time check): task success {c['success']:.3f} ({c['success_ci95'][0]:.2f}-{c['success_ci95'][1]:.2f}), "
                  f"{c['escalated']} of {out['n_cases']} questions escalated, latency p50 {c['latency_s']['p50']} s / "
                  f"p95 {c['latency_s']['p95']} s. Computed from the two recorded runs."]
    L += ["", "Every answer is in `results/agent_eval_records.jsonl`; every model and tool call is an OpenTelemetry "
              "span in `results/traces.jsonl`."]
    return "\n".join(L)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default=DEFAULT_MODELS)
    ap.add_argument("--transport", choices=["mcp", "local"], default="mcp")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--synthetic", action="store_true", help="synthetic data (tests/demo)")
    a = ap.parse_args(argv)
    out = run([m.strip() for m in a.models.split(",") if m.strip()], a.transport, a.limit, a.synthetic)
    print("\n" + report_md(out))


if __name__ == "__main__":
    main()
