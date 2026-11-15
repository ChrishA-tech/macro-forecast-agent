"""
Chat backends with tool calling, behind one interface:

    llm.chat(messages, tools) -> ChatResult(content, tool_calls, prompt_tokens, completion_tokens, seconds)

  OllamaChat    free local open models via Ollama (http://localhost:11434), native tool calling
  RulesPolicy   no LLM: the original rule router + template answers, as a baseline the LLMs must beat
"""
from __future__ import annotations

import json
import re
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

OLLAMA = "http://localhost:11434"


@dataclass
class ChatResult:
    content: str = ""
    tool_calls: List[Dict[str, Any]] = field(default_factory=list)   # [{"name", "arguments"}]
    prompt_tokens: int = 0
    completion_tokens: int = 0
    seconds: float = 0.0


def _post(url: str, body: Dict, timeout: float) -> Dict:
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def ollama_models(host: str = OLLAMA) -> List[str]:
    try:
        with urllib.request.urlopen(host + "/api/tags", timeout=5) as r:
            return [m["name"] for m in json.loads(r.read()).get("models", [])]
    except (urllib.error.URLError, OSError, ValueError):
        return []


class OllamaChat:
    def __init__(self, model: str, host: str = OLLAMA, temperature: float = 0.0, num_ctx: int = 8192,
                 timeout: float = 300, seed: int = 7):
        self.model, self.host, self.timeout = model, host, timeout
        self.options = {"temperature": temperature, "num_ctx": num_ctx, "seed": seed}
        self.name = model

    def chat(self, messages: List[Dict[str, Any]], tools: Optional[List[Dict[str, Any]]] = None) -> ChatResult:
        body = {"model": self.model, "messages": messages, "stream": False, "options": self.options}
        if tools:
            body["tools"] = [{"type": "function", "function": {"name": t["name"], "description": t["description"],
                                                               "parameters": t["parameters"]}} for t in tools]
        t0 = time.perf_counter()
        r = _post(self.host + "/api/chat", body, self.timeout)
        msg = r.get("message", {})
        calls = []
        for c in msg.get("tool_calls") or []:
            f = c.get("function", {})
            args = f.get("arguments", {})
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except json.JSONDecodeError:
                    args = {}
            calls.append({"name": f.get("name", ""), "arguments": args})
        return ChatResult(content=msg.get("content", "") or "", tool_calls=calls,
                          prompt_tokens=int(r.get("prompt_eval_count", 0) or 0),
                          completion_tokens=int(r.get("eval_count", 0) or 0),
                          seconds=time.perf_counter() - t0)


class RulesPolicy:
    """Deterministic baseline: rule router picks the tool; template turns tool facts into the answer."""
    name = "rules (no LLM)"

    def chat(self, messages, tools=None) -> ChatResult:
        from .graph import HELP, template_answer
        from .router import rule_route
        question = next(m["content"] for m in messages if m["role"] == "user")
        r = rule_route(question)
        results = [json.loads(m["content"]) for m in messages if m["role"] == "tool"]
        called = [m.get("tool_name") for m in messages if m["role"] == "tool"]
        if r.intent == "out_of_scope":
            return ChatResult(content=HELP)
        if r.intent == "oil_cpi_scenario":
            if r.oil_change_pct is None and r.target_price is not None and "latest_data" not in called:
                return ChatResult(tool_calls=[{"name": "latest_data", "arguments": {}}])
            if "oil_cpi_scenario" not in called:
                pct = r.oil_change_pct
                if pct is None and r.target_price is not None:
                    last = results[0]["facts"]["latest Brent price (USD/bbl)"]
                    pct = round(100 * (r.target_price / last - 1), 1)
                return ChatResult(tool_calls=[{"name": "oil_cpi_scenario", "arguments": {"oil_change_pct": pct or 0.0}}])
            return ChatResult(content=template_answer(r, results[-1]["facts"]))
        if r.intent not in called:
            args = {} if r.intent in ("latest_data", "model_accuracy") else {"horizon": r.horizon}
            return ChatResult(tool_calls=[{"name": r.intent, "arguments": args}])
        return ChatResult(content=template_answer(r, results[-1]["facts"]))


def make_llm(name: str):
    return RulesPolicy() if name in ("rules", "rules (no LLM)") else OllamaChat(name)


_THINK = re.compile(r"<think>.*?</think>", re.DOTALL)


def strip_reasoning(text: str) -> str:
    """Some open models (e.g. Qwen3) emit <think>...</think> before the answer."""
    return _THINK.sub("", text or "").strip()
