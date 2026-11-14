"""
Input guardrail v2 (Nov 2026): prompt-injection, jailbreak and misuse screening.

Pipeline for each prompt:
  1. Normalise: Unicode NFKC, strip zero-width/control characters, map common
     Cyrillic/Greek homoglyphs to Latin, collapse whitespace.
  2. Build matching views: normalised text, a leetspeak-decoded view (1->i, 0->o, @->a ...),
     and a de-spaced view ("i g n o r e" -> "ignore").
  3. Decode embedded payloads (Base64, hex, \\x escapes) and scan the decoded text too.
  4. Score: max rule weight + 0.15 per additional rule (capped at 1.0).
     Blocked if score >= threshold (strict 0.50, lenient 0.75), so strict_mode
     really changes behaviour (v1 blocked on any rule regardless of mode).

guardrails_v1.py keeps the original rules for comparison in evaluation/.
"""
from __future__ import annotations

import base64
import binascii
import re
import unicodedata
from typing import List, Optional, Tuple

from .schemas import GuardrailAssessment

# (name, pattern, weight); patterns run case-insensitively on every matching view
INJECTION_RULES: List[Tuple[str, str, float]] = [
    ("DIRECT_OVERRIDE",
     r"\b(ignore|disregard|forget|bypass|override|skip)\b[\w\s,'-]{0,30}?\b(previous|prior|above|earlier|preceding|system|all|your|these|those|the)\b"
     r"[\w\s'-]{0,20}?\b(instructions?|prompts?|rules?|directions?|guidelines?|guardrails?|constraints?|context)\b", 0.95),
    ("DIRECT_OVERRIDE_MULTILINGUAL",
     r"\b(ignoriere|vergiss|ignorez|oubliez|ignora|olvida)\b.{0,40}?\b(anweisungen|vorherigen|instructions|instrucciones|reglas|regeln)\b", 0.95),
    ("NEW_INSTRUCTIONS",
     r"\b(new|updated|real|actual|revised)\s+(system\s+)?(instructions?|prompt|rules?)\s*[:\-]|\bthe\s+following\s+(overrides?|supersedes?)\b", 0.80),
    ("SYSTEM_PROMPT_EXFIL",
     r"\b(reveal|show|print|output|repeat|display|leak|tell\s+me|what\s+(is|are|was|were))\b.{0,30}?\b(your|the)\s+"
     r"(system\s+prompt|initial\s+(instructions|prompt)|hidden\s+(prompt|instructions)|developer\s+(message|instructions)|instructions\s+above|original\s+instructions)", 0.90),
    ("JAILBREAK_PERSONA",
     r"\b(do\s+anything\s+now|DAN\s+mode|developer\s+mode\s+(enabled|on)|jailbreak|jailbroken|"
     r"act\s+as\s+an?\s+(unrestricted|unfiltered|uncensored)|without\s+any\s+(restrictions|filters|rules))\b", 0.95),
    ("ROLEPLAY_OVERRIDE",
     r"\b(pretend|imagine)\s+(that\s+)?(you\s+are|you're|to\s+be)\b.{0,60}?\b(no|without|free\s+from|not\s+bound)\b.{0,20}?\b(rules|restrictions|limits|guidelines|filters)\b|"
     r"\byou\s+are\s+no\s+longer\b|\bfrom\s+now\s+on,?\s+you\s+(are|will|must)\b", 0.85),
    ("DELIMITER_ESCAPING", r"<\|im_start\|>|<\|im_end\|>|<\|endoftext\|>|\[/?INST\]|<<SYS>>|###\s*(system|assistant)\s*:|</?system>", 0.85),
    ("CODE_OR_SQL_INJECTION",
     r";\s*drop\s+table|\bunion\s+select\b|os\.system\(|subprocess\.(popen|run|call)\(|\beval\(|\bexec\(|__import__\(|rm\s+-rf\s+/", 0.80),
    ("DATA_EXFIL_MARKUP", r"!\[[^\]]*\]\(\s*https?://[^)]*[?&][^)]*=\s*[{\[]?", 0.70),
    ("HARMFUL_MISUSE",
     r"\b(build|write|create|make)\s+(a\s+)?(malware|ransomware|keylogger|botnet)\b|\bexploit\s+(a\s+|the\s+)?vulnerabilit(y|ies)\b|"
     r"\bransomware\s+payload\b|\bcredential\s+harvesting\b|\bphishing\s+(template|email|kit)\b", 1.00),
]
_COMPILED = [(n, re.compile(p, re.IGNORECASE | re.DOTALL), w) for n, p, w in INJECTION_RULES]

_ZERO_WIDTH = re.compile(r"[​-‏‪-‮⁠-⁤﻿]")
_CONTROL = re.compile(r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]")
_HOMOGLYPHS = str.maketrans({
    "а": "a", "е": "e", "о": "o", "р": "p", "с": "c", "у": "y", "х": "x", "і": "i", "ј": "j", "ѕ": "s",
    "А": "A", "Е": "E", "О": "O", "Р": "P", "С": "C", "Х": "X", "І": "I", "Ј": "J", "Ѕ": "S", "Т": "T",
    "ο": "o", "α": "a", "ε": "e", "ι": "i", "ν": "v", "Ο": "O", "Α": "A", "Ε": "E", "Ι": "I",
})
_LEET = str.maketrans({"0": "o", "1": "i", "3": "e", "4": "a", "5": "s", "7": "t", "@": "a", "$": "s", "!": "i"})
_SPACED = re.compile(r"\b(?:[A-Za-z][ .\-_]){3,}[A-Za-z]\b")
_B64 = re.compile(r"(?<![A-Za-z0-9+/])[A-Za-z0-9+/]{16,}={0,2}(?![A-Za-z0-9+/])")
_HEX = re.compile(r"(?:\\x[0-9a-fA-F]{2}){6,}|\b(?:[0-9a-fA-F]{2}){10,}\b")


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = _ZERO_WIDTH.sub("", text)
    text = _CONTROL.sub("", text)
    text = text.translate(_HOMOGLYPHS)
    return re.sub(r"[ \t]+", " ", text).strip()


def _views(text: str) -> List[str]:
    leet = text.translate(_LEET)
    despaced = _SPACED.sub(lambda m: re.sub(r"[ .\-_]", "", m.group(0)), text)
    return list(dict.fromkeys([text, leet, despaced]))


def _printable_ratio(s: str) -> float:
    return sum(c.isprintable() or c in "\n\t" for c in s) / max(len(s), 1)


def decode_payloads(text: str) -> List[Tuple[str, str]]:
    """Return (encoding, decoded_text) for Base64 / hex payloads that decode to readable text."""
    found = []
    for m in _B64.finditer(text):
        token = m.group(0)
        if token.isalpha() or token.isdigit():
            continue                                   # plain words / numbers, not encoded data
        try:
            raw = base64.b64decode(token + "=" * (-len(token) % 4), validate=True)
            s = raw.decode("utf-8")
        except (binascii.Error, UnicodeDecodeError, ValueError):
            continue
        if len(s) >= 8 and _printable_ratio(s) > 0.95 and re.search(r"[A-Za-z]{3,}\s+[A-Za-z]{2,}", s):
            found.append(("base64", s))
    for m in _HEX.finditer(text):
        token = m.group(0).replace("\\x", "")
        try:
            s = bytes.fromhex(token).decode("utf-8")
        except (ValueError, UnicodeDecodeError):
            continue
        if len(s) >= 6 and _printable_ratio(s) > 0.95 and re.search(r"[A-Za-z]{3,}", s):
            found.append(("hex", s))
    return found


def _scan(text: str) -> List[Tuple[str, float]]:
    hits = {}
    for view in _views(text):
        for name, rx, w in _COMPILED:
            if name not in hits and rx.search(view):
                hits[name] = w
    return list(hits.items())


def evaluate_prompt_safety(prompt: str, strict_mode: bool = True, threshold: Optional[float] = None) -> GuardrailAssessment:
    """Screens input text for prompt injection, jailbreaks, obfuscation and misuse."""
    if not isinstance(prompt, str) or not prompt.strip():
        return GuardrailAssessment(is_safe=False, risk_score=1.0, matched_rules=["EMPTY_OR_INVALID_INPUT"],
                                   sanitized_prompt="", reason="Input prompt is empty or non-string.")

    sanitized = normalize(prompt)
    obfuscated = sanitized != re.sub(r"[ \t]+", " ", _CONTROL.sub("", prompt)).strip()

    hits = _scan(sanitized)
    for enc, decoded in decode_payloads(sanitized):
        inner = _scan(normalize(decoded))
        if inner:
            hits += [(f"ENCODED_{enc.upper()}_{name}", w) for name, w in inner]
        else:
            hits.append((f"OBFUSCATED_{enc.upper()}_TEXT", 0.40))
    if obfuscated and hits:
        hits.append(("UNICODE_OBFUSCATION", 0.30))

    names = list(dict.fromkeys(n for n, _ in hits))
    weights = sorted((w for _, w in hits), reverse=True)
    risk = min(1.0, round(weights[0] + 0.15 * (len(weights) - 1), 4)) if weights else 0.0
    limit = threshold if threshold is not None else (0.50 if strict_mode else 0.75)
    is_safe = risk < limit
    reason = ("Input passed all prompt-protection guardrails." if not names else
              (f"Allowed with warnings (risk {risk} < {limit}): {', '.join(names)}" if is_safe else
               f"Blocked due to triggered safety rules: {', '.join(names)}"))
    return GuardrailAssessment(is_safe=is_safe, risk_score=risk, matched_rules=names,
                               sanitized_prompt=sanitized, reason=reason)
