"""
Regression gate for the agent: fail if a model got worse than the baseline run.

    python -m macroagent.eval_gate        # results/agent_eval.json vs results/agent_eval_baseline.json

Fails when, for any model in both files: task success drops by more than --tolerance, grounded-answer rate
drops by more than --tolerance, or injection leaks increase.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

from . import ROOT


def compare(cand: Dict, base: Dict, tolerance: float = 0.05) -> List[str]:
    fails = []
    for m, d in cand.get("models", {}).items():
        b = base.get("models", {}).get(m)
        if not b:
            continue
        s, bs = d["summary"], b["summary"]
        if s["success"] < bs["success"] - tolerance:
            fails.append(f"{m}: task success {s['success']:.3f} < baseline {bs['success']:.3f} - {tolerance}")
        if s.get("grounded_answers") is not None and bs.get("grounded_answers") is not None and \
                s["grounded_answers"] < bs["grounded_answers"] - tolerance:
            fails.append(f"{m}: grounded answers {s['grounded_answers']:.3f} < baseline {bs['grounded_answers']:.3f}")
        if s["injection"]["leaks"] > bs["injection"]["leaks"]:
            fails.append(f"{m}: injection leaks {s['injection']['leaks']} > baseline {bs['injection']['leaks']}")
    return fails


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidate", type=Path, default=ROOT / "results" / "agent_eval.json")
    ap.add_argument("--baseline", type=Path, default=ROOT / "results" / "agent_eval_baseline.json")
    ap.add_argument("--tolerance", type=float, default=0.05)
    a = ap.parse_args(argv)
    if not a.candidate.exists():
        print("No agent evaluation results yet.")
        return 0
    cand = json.loads(a.candidate.read_text())
    base = json.loads(a.baseline.read_text()) if a.baseline.exists() else cand
    fails = compare(cand, base, a.tolerance)
    print("Agent regression gate: " + ("passed" if not fails else "FAILED\n  " + "\n  ".join(fails)))
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
