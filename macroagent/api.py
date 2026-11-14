"""FastAPI: POST /ask {"question": "..."} -> grounded answer, facts, sources, guard info."""
from __future__ import annotations

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .graph import ask
from .llm import default_llm
from .tools import DataProvider

app = FastAPI(title="Macro Forecast Agent", version="1.0.0",
              description="Guarded LangGraph agent over Brent and UK CPI forecasting tools.")
_state = {}


def _provider() -> DataProvider:
    if "provider" not in _state:
        try:
            _state["provider"] = DataProvider.from_vendor_data()
        except FileNotFoundError as e:
            raise HTTPException(503, str(e)) from e
    return _state["provider"]


class AskRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=2000)
    use_llm: bool = True


@app.get("/health")
def health():
    return {"status": "healthy", "llm": "claude" if default_llm() else "none (deterministic)"}


@app.post("/ask")
def ask_endpoint(req: AskRequest):
    r = ask(req.question, _provider(), default_llm() if req.use_llm else None)
    if r["status"] == "blocked":
        raise HTTPException(403, detail=r)
    return r
