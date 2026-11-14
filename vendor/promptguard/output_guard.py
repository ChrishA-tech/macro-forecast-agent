"""
Output guardrail: scans LLM output before it is returned.

- Redacts personal data and secrets: e-mail addresses, phone numbers, payment card
  numbers (Luhn-checked), UK National Insurance numbers, IBANs, API keys / tokens,
  private-key blocks.
- Detects system-prompt leakage via a canary token embedded in the system prompt:
  if the canary appears in the output, the response is withheld.
"""
from __future__ import annotations

import re
import secrets
from typing import Any, Dict, List, Tuple

CANARY_PREFIX = "CANARY-"

_PATTERNS: List[Tuple[str, re.Pattern]] = [
    ("PRIVATE_KEY", re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----[\s\S]*?-----END [A-Z ]*PRIVATE KEY-----")),
    ("API_KEY", re.compile(r"\b(?:sk-(?:ant-)?[A-Za-z0-9_\-]{16,}|AKIA[0-9A-Z]{16}|ghp_[A-Za-z0-9]{36}|xox[baprs]-[A-Za-z0-9\-]{10,}|AIza[0-9A-Za-z_\-]{35})\b")),
    ("EMAIL", re.compile(r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b")),
    ("IBAN", re.compile(r"\b[A-Z]{2}\d{2}(?:\s?[A-Z0-9]{4}){3,7}(?:\s?[A-Z0-9]{1,3})?\b")),
    ("UK_NINO", re.compile(r"\b(?!BG|GB|NK|KN|TN|NT|ZZ)[A-CEGHJ-PR-TW-Z]{2}\s?\d{2}\s?\d{2}\s?\d{2}\s?[A-D]\b")),
    ("CARD_NUMBER", re.compile(r"(?<![\d])\d(?:[ \-]?\d){12,18}(?![\d])")),
    ("PHONE", re.compile(r"(?<!\w)(?:\+\d{1,3}[\s\-]?)?(?:\(?\d{2,5}\)?[\s\-]?)\d{3,4}[\s\-]?\d{3,4}(?!\w)")),
]


def new_canary() -> str:
    return CANARY_PREFIX + secrets.token_hex(8)


def _luhn_ok(digits: str) -> bool:
    d = [int(c) for c in digits][::-1]
    total = sum(d[0::2]) + sum(sum(divmod(2 * x, 10)) for x in d[1::2])
    return len(d) >= 13 and total % 10 == 0


def redact_text(text: str) -> Tuple[str, List[str]]:
    found: List[str] = []

    def sub(kind):
        def _r(m):
            raw = m.group(0)
            if kind == "CARD_NUMBER" and not _luhn_ok(re.sub(r"\D", "", raw)):
                return raw
            if kind == "PHONE":
                before, after = m.string[max(0, m.start() - 2):m.start()], m.string[m.end():m.end() + 2]
                part_of_longer_number = re.search(r"\d[ \-]?$", before) or re.match(r"^[ \-]?\d", after)
                if len(re.sub(r"\D", "", raw)) < 10 or part_of_longer_number:
                    return raw
            found.append(kind)
            return f"[REDACTED_{kind}]"
        return _r

    for kind, rx in _PATTERNS:
        text = rx.sub(sub(kind), text)
    return text, found


def scan_output(obj: Any, canary: str = "") -> Tuple[Any, List[str], bool]:
    """Recursively redact strings in a JSON-like object. Returns (clean_obj, redactions, canary_leaked)."""
    found: List[str] = []
    leaked = False

    def walk(x):
        nonlocal leaked
        if isinstance(x, str):
            if canary and canary in x:
                leaked = True
            clean, f = redact_text(x)
            found.extend(f)
            return clean
        if isinstance(x, list):
            return [walk(v) for v in x]
        if isinstance(x, dict):
            return {k: walk(v) for k, v in x.items()}
        return x

    return walk(obj), found, leaked
