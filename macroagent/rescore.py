"""Re-score recorded agent answers with the current scorer (no models re-run).

    python -m macroagent.rescore results/v1/agent_eval_records.jsonl results/v1/agent_eval_rescored.json

Used to compare runs under one scorer after a scoring fix. Only the cases present in the records are scored.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from .evals import load_cases, score, summarise
from .run_evals import provider_for


def rescore(records_path: Path, out_path: Path) -> dict:
    cases = {c["id"]: c for c in load_cases()}
    provider = provider_for(False)
    by_model = {}
    for line in Path(records_path).read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            name = "rules" if r["model"].startswith("rules") else r["model"]   # match run_evals' model keys
            by_model.setdefault(name, []).append(r)
    out = {"rescored_from": str(records_path), "models": {}}
    for m, recs in by_model.items():
        recs = [r for r in recs if r["id"] in cases]
        scores = [score(cases[r["id"]], r, provider) for r in recs]
        out["models"][m] = {"summary": summarise(recs, scores), "cases": scores}
    out["n_cases"] = len(next(iter(by_model.values())))
    Path(out_path).write_text(json.dumps(out, indent=2))
    return out


if __name__ == "__main__":
    o = rescore(Path(sys.argv[1]), Path(sys.argv[2]))
    for m, d in o["models"].items():
        s = d["summary"]
        print(f"{m:16s} success {s['success']:.3f}  injection leaks to user {s['injection']['leaks']}/{s['injection']['attempts']}"
              f"  model attempts {s['injection']['model_disclosure_attempts']}  by category {s['by_category']}")
