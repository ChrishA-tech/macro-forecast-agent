"""
Numeric grounding check: every number in the answer must come from the tool facts
(after rounding), or be a small integer horizon / year mentioned in the question.
LLM answers that fail the check are replaced by the deterministic template answer.
"""
from __future__ import annotations

import re
from typing import Any, Dict, Iterable, List, Set, Tuple

_NUM = re.compile(r"(?<![\w.])[-+−]?\$?\d[\d,]*(?:\.\d+)?")


def _numbers(text: str) -> List[float]:
    out = []
    for m in _NUM.finditer(text):
        s = m.group(0).replace(",", "").replace("$", "").replace("−", "-")
        try:
            out.append(float(s))
        except ValueError:
            pass
    return out


def _fact_values(facts: Dict[str, Any]) -> Set[float]:
    vals: Set[float] = set()

    def add(v):
        if isinstance(v, bool):
            return
        if isinstance(v, (int, float)):
            for d in (0, 1, 2):
                vals.add(round(float(v), d))
                vals.add(round(abs(float(v)), d))
        elif isinstance(v, str):
            for n in _numbers(v):
                add(n)
        elif isinstance(v, (list, tuple)):
            for x in v:
                add(x)

    for k, v in facts.items():
        add(v)
        add(k)                      # numbers inside labels, e.g. "after 12 months"
    return vals


def check_grounding(answer: str, facts: Dict[str, Any], question: str = "") -> Tuple[bool, List[float]]:
    allowed = _fact_values(facts) | {float(n) for n in _numbers(question)}
    ungrounded = []
    for n in _numbers(answer):
        if n in allowed or abs(n) in allowed or round(n, 1) in allowed or round(n, 2) in allowed:
            continue
        if float(n).is_integer() and (0 <= n <= 24 or 1990 <= n <= 2100):   # horizons, years
            continue
        ungrounded.append(n)
    return (not ungrounded), ungrounded


def flatten_facts(results: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    facts: Dict[str, Any] = {}
    for r in results:
        facts.update(r.get("facts", {}))
    return facts
