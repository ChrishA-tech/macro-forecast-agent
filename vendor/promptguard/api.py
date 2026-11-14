"""
FastAPI service: POST /api/v1/extract runs the guarded LangGraph workflow.
Uses Claude when ANTHROPIC_API_KEY is set (PROMPTGUARD_LLM=claude), else the deterministic simulator.
Every response carries a request ID (body + X-Request-ID header) that matches the audit log.
"""
from __future__ import annotations

import logging
import os
import uuid

from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from .output_guard import new_canary
from .schemas import PromptInputRequest, WorkflowResponse
from .workflow import run_protected_extraction_workflow

logging.basicConfig(level=logging.INFO, format="%(message)s")
app = FastAPI(title="AI Safety Prompt Guard & LangGraph Extraction API", version="2.0.0",
              description="Input guardrail -> LLM extraction with self-repair -> output guardrail (PII/secrets, canary).")

_LLM = None
_CANARY = new_canary()


def _llm():
    global _LLM
    if _LLM is None and os.environ.get("PROMPTGUARD_LLM") == "claude" and os.environ.get("ANTHROPIC_API_KEY"):
        from .extractor import make_claude_extractor
        _LLM = make_claude_extractor(canary=_CANARY)
    return _LLM


@app.exception_handler(RequestValidationError)
async def validation_handler(request: Request, exc: RequestValidationError):
    rid = uuid.uuid4().hex
    return JSONResponse(status_code=422, headers={"X-Request-ID": rid},
                        content={"status": "error", "request_id": rid,
                                 "error_detail": "Invalid request", "errors": exc.errors()})


@app.get("/health")
def health_check():
    return {"status": "healthy", "service": "langgraph-prompt-guard",
            "llm": "claude" if _llm() is not None else "simulator"}


@app.post("/api/v1/extract", response_model=WorkflowResponse)
def extract_endpoint(payload: PromptInputRequest):
    result = run_protected_extraction_workflow(user_id=payload.user_id, prompt=payload.prompt,
                                               strict_mode=payload.strict_mode, llm_callable=_llm(),
                                               canary=_CANARY)
    code = {"success": 200, "blocked": status.HTTP_403_FORBIDDEN, "error": status.HTTP_422_UNPROCESSABLE_ENTITY}[result.status]
    return JSONResponse(status_code=code, content=result.model_dump(), headers={"X-Request-ID": result.request_id})
