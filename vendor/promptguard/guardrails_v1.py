"""Original v1 guardrail, frozen unchanged for comparison in evaluation/ (do not edit)."""
import re
from typing import Tuple, List
from .schemas import GuardrailAssessment

# Curated adversarial & prompt-injection rule patterns with severity weights
INJECTION_RULES: List[Tuple[str, str, float]] = [
    ("DIRECT_OVERRIDE", r"(?i)(ignore|disregard|forget|bypass)\s+(all\s+)?(previous|prior|above|system)\s+(instructions|prompts|rules|directions)", 0.95),
    ("SYSTEM_PROMPT_EXFIL", r"(?i)(reveal|show|print|output|repeat)\s+(your|the)\s+(system\s+prompt|initial\s+instructions|hidden\s+prompt|developer\s+message)", 0.90),
    ("JAILBREAK_PERSONA", r"(?i)\b(do\s+anything\s+now|DAN\s+mode|developer\s+mode\s+enabled|act\s+as\s+an?\s+unrestricted|jailbreak)\b", 0.95),
    ("DELIMITER_ESCAPING", r"(?:<\|im_start\|>|<\|endoftext\|>|\[INST\]|\[/INST\]|###\s*System:)", 0.85),
    ("CODE_OR_SQL_INJECTION", r"(?i)(?:;\s*DROP\s+TABLE|UNION\s+SELECT|os\.system\(|subprocess\.Popen\(|eval\(|exec\()", 0.80),
    ("HARMFUL_MISUSE", r"(?i)\b(build\s+a\s+malware|exploit\s+vulnerability|ransomware\s+payload|credential\s+harvesting|phishing\s+template)\b", 1.00),
]


def evaluate_prompt_safety(prompt: str, strict_mode: bool = True) -> GuardrailAssessment:
    """Screens input text for prompt injection, delimiter attacks, and misuse."""
    if not isinstance(prompt, str) or not prompt.strip():
        return GuardrailAssessment(
            is_safe=False,
            risk_score=1.0,
            matched_rules=["EMPTY_OR_INVALID_INPUT"],
            sanitized_prompt="",
            reason="Input prompt is empty or non-string."
        )

    # Normalize control characters while preserving printable text
    sanitized = re.sub(r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]", "", prompt).strip()

    matched_rules: List[str] = []
    max_weight = 0.0
    cumulative_bonus = 0.0

    for rule_name, pattern, weight in INJECTION_RULES:
        if re.search(pattern, sanitized):
            matched_rules.append(rule_name)
            if weight > max_weight:
                max_weight = weight
            else:
                cumulative_bonus += 0.15

    # Heuristic check for obfuscated Base64 payloads
    if re.search(r"\b[A-Za-z0-9+/]{48,}={0,2}\b", sanitized):
        matched_rules.append("OBFUSCATED_BASE64_PAYLOAD")
        max_weight = max(max_weight, 0.70)

    risk_score = min(1.0, round(max_weight + cumulative_bonus, 4))
    threshold = 0.50 if strict_mode else 0.75
    is_safe = (risk_score < threshold) and (len(matched_rules) == 0)

    reason = (
        "Input passed all prompt-protection guardrails."
        if is_safe
        else f"Blocked due to triggered safety rules: {', '.join(matched_rules)}"
    )

    return GuardrailAssessment(
        is_safe=is_safe,
        risk_score=risk_score,
        matched_rules=matched_rules,
        sanitized_prompt=sanitized,
        reason=reason,
    )
