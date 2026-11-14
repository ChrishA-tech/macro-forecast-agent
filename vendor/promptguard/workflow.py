"""
LangGraph workflow v2 (Nov 2026):

  input_guardrail ──safe──▶ llm_extraction ──valid──▶ output_guard ──▶ END
        │                       │  ▲
        │                       └──┘ invalid JSON/schema: self-repair retry
        │                       │      (error message fed back, max_attempts)
        │                       └──retries exhausted──▶ END (status=error)
        └──unsafe──▶ rejection ──▶ END

- input_guardrail: promptguard.guardrails v2 (normalisation, decoding, scoring)
- llm_extraction: any callable str -> JSON str (deterministic simulator by default,
  Claude via make_claude_extractor)
- output_guard: redacts PII/secrets; withholds output that leaks the canary token
- every run gets a request_id and one structured audit-log line (no raw prompt
  is logged: only its SHA-256)
"""
from __future__ import annotations

import hashlib
import json
import logging
import time
import uuid
from typing import Any, Callable, Dict, List, Optional, TypedDict

from .exceptions import SchemaExtractionError
from .extractor import extract_schema_constrained_json
from .guardrails import evaluate_prompt_safety
from .output_guard import scan_output
from .schemas import ExtractedEntitySchema, GuardrailAssessment, WorkflowResponse

try:
    from langgraph.graph import END, StateGraph
    HAS_LANGGRAPH = True
except ImportError:
    HAS_LANGGRAPH = False
    END = "__END__"

audit_log = logging.getLogger("promptguard.audit")


class AgentWorkflowState(TypedDict, total=False):
    request_id: str
    user_id: str
    raw_prompt: str
    strict_mode: bool
    canary: str
    guardrail_result: Optional[GuardrailAssessment]
    extracted_result: Optional[ExtractedEntitySchema]
    status: str
    error_detail: Optional[str]
    attempts: int
    max_attempts: int
    redactions: List[str]


class _CompiledFallbackStateGraph:
    """Minimal StateGraph runner with LangGraph's .invoke() interface (used when langgraph is not installed)."""

    def __init__(self, nodes, entry_point, conditional_edges, edges, recursion_limit: int = 25):
        self.nodes, self.entry_point = nodes, entry_point
        self.conditional_edges, self.edges = conditional_edges, edges
        self.recursion_limit = recursion_limit

    def invoke(self, state: Dict[str, Any]) -> Dict[str, Any]:
        current_state, node, steps = dict(state), self.entry_point, 0
        while node != END:
            steps += 1
            if steps > self.recursion_limit:
                raise RuntimeError("Recursion limit reached in fallback graph.")
            current_state.update(self.nodes[node](current_state) or {})
            if node in self.conditional_edges:
                router, mapping = self.conditional_edges[node]
                node = mapping[router(current_state)]
            else:
                node = self.edges.get(node, END)
        return current_state


def build_langgraph_workflow(llm_callable: Optional[Callable[[str], str]] = None):

    def input_guardrail_node(state: AgentWorkflowState) -> dict:
        return {"guardrail_result": evaluate_prompt_safety(state["raw_prompt"], state.get("strict_mode", True))}

    def route_after_guardrail(state: AgentWorkflowState) -> str:
        g = state.get("guardrail_result")
        return "llm_extraction_node" if g and g.is_safe else "rejection_node"

    def llm_extraction_node(state: AgentWorkflowState) -> dict:
        prompt = state["guardrail_result"].sanitized_prompt
        attempts = state.get("attempts", 0) + 1
        if state.get("error_detail"):                      # self-repair: feed the validation error back
            prompt += ("\n\n[Your previous reply failed validation: " + state["error_detail"][:300]
                       + " Reply again with ONLY the JSON object.]")
        try:
            extracted = extract_schema_constrained_json(prompt, llm_callable=llm_callable)
            return {"extracted_result": extracted, "status": "extracted", "error_detail": None, "attempts": attempts}
        except SchemaExtractionError as exc:
            return {"extracted_result": None, "status": "error", "error_detail": exc.message, "attempts": attempts}
        except Exception as exc:  # LLM/network failure: do not retry, report cleanly
            return {"extracted_result": None, "status": "error", "attempts": state.get("max_attempts", 2),
                    "error_detail": f"LLM call failed: {type(exc).__name__}: {exc}"}

    def route_after_extraction(state: AgentWorkflowState) -> str:
        if state["status"] == "extracted":
            return "output_guard_node"
        if state["attempts"] < state.get("max_attempts", 2):
            return "llm_extraction_node"
        return END

    def output_guard_node(state: AgentWorkflowState) -> dict:
        data = state["extracted_result"].model_dump()
        clean, redactions, leaked = scan_output(data, state.get("canary", ""))
        if leaked:
            return {"extracted_result": None, "status": "error", "redactions": redactions,
                    "error_detail": "Output withheld: it contained the system-prompt canary (prompt leakage)."}
        return {"extracted_result": ExtractedEntitySchema.model_validate(clean), "status": "success",
                "redactions": redactions}

    def rejection_node(state: AgentWorkflowState) -> dict:
        g = state.get("guardrail_result")
        return {"extracted_result": None, "status": "blocked",
                "error_detail": g.reason if g else "Blocked by safety guardrail."}

    nodes = {"input_guardrail_node": input_guardrail_node, "llm_extraction_node": llm_extraction_node,
             "output_guard_node": output_guard_node, "rejection_node": rejection_node}
    cond = {
        "input_guardrail_node": (route_after_guardrail, {"llm_extraction_node": "llm_extraction_node",
                                                         "rejection_node": "rejection_node"}),
        "llm_extraction_node": (route_after_extraction, {"output_guard_node": "output_guard_node",
                                                         "llm_extraction_node": "llm_extraction_node", END: END}),
    }
    edges = {"output_guard_node": END, "rejection_node": END}

    if HAS_LANGGRAPH:
        graph = StateGraph(AgentWorkflowState)
        for name, fn in nodes.items():
            graph.add_node(name, fn)
        graph.set_entry_point("input_guardrail_node")
        for src, (router, mapping) in cond.items():
            graph.add_conditional_edges(src, router, mapping)
        for src, dst in edges.items():
            graph.add_edge(src, dst)
        return graph.compile()
    return _CompiledFallbackStateGraph(nodes, "input_guardrail_node", cond, edges)


def run_protected_extraction_workflow(user_id: str, prompt: str, strict_mode: bool = True,
                                      llm_callable: Optional[Callable[[str], str]] = None,
                                      max_attempts: int = 2, canary: str = "",
                                      request_id: Optional[str] = None) -> WorkflowResponse:
    """Runs the compiled workflow and returns a validated WorkflowResponse (and writes one audit line)."""
    request_id = request_id or uuid.uuid4().hex
    t0 = time.perf_counter()
    app = build_langgraph_workflow(llm_callable=llm_callable)
    final = app.invoke({"request_id": request_id, "user_id": user_id, "raw_prompt": prompt,
                        "strict_mode": strict_mode, "canary": canary, "guardrail_result": None,
                        "extracted_result": None, "status": "pending", "error_detail": None,
                        "attempts": 0, "max_attempts": max(1, max_attempts), "redactions": []})
    g = final["guardrail_result"]
    resp = WorkflowResponse(status=final["status"], guardrail=g, extracted_data=final.get("extracted_result"),
                            error_detail=final.get("error_detail"), request_id=request_id,
                            extraction_attempts=final.get("attempts", 0),
                            output_redactions=sorted(set(final.get("redactions", []))))
    audit_log.info(json.dumps({
        "event": "prompt_guard", "request_id": request_id, "user_id": user_id, "status": resp.status,
        "risk_score": g.risk_score if g else None, "rules": g.matched_rules if g else [],
        "prompt_sha256": hashlib.sha256(prompt.encode("utf-8", "replace")).hexdigest(),
        "attempts": resp.extraction_attempts, "redactions": resp.output_redactions,
        "latency_ms": round((time.perf_counter() - t0) * 1000, 1)}))
    return resp
