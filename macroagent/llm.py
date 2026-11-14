"""Claude adapter: returns llm(system, user) -> text, as used by the router and answer composer."""
from __future__ import annotations

import os
from typing import Callable, Optional


def make_claude_llm(model: Optional[str] = None, max_tokens: int = 500, client=None) -> Callable[[str, str], str]:
    if client is None:
        import anthropic
        client = anthropic.Anthropic()
    model = model or os.environ.get("MACROAGENT_MODEL", "claude-sonnet-4-5")

    def call(system: str, user: str) -> str:
        msg = client.messages.create(model=model, max_tokens=max_tokens, system=system,
                                     messages=[{"role": "user", "content": user}])
        return "".join(getattr(b, "text", "") for b in msg.content).strip()

    call.model = model
    return call


def default_llm() -> Optional[Callable[[str, str], str]]:
    """Claude if ANTHROPIC_API_KEY is set and the SDK is installed, else None (deterministic mode)."""
    if not os.environ.get("ANTHROPIC_API_KEY"):
        return None
    try:
        return make_claude_llm()
    except ImportError:
        return None
