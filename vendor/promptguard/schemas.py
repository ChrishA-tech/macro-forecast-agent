from typing import List, Optional, Literal
from pydantic import BaseModel, Field, field_validator


class PromptInputRequest(BaseModel):
    user_id: str = Field(..., min_length=2, max_length=64, description="Unique caller identifier")
    prompt: str = Field(..., min_length=1, max_length=4000, description="Raw user prompt to screen and extract")
    strict_mode: bool = Field(default=True, description="Enforce stricter threshold on prompt risk score")

    @field_validator("prompt")
    @classmethod
    def must_not_be_whitespace(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("Prompt cannot be empty or whitespace only.")
        return v.strip()


class GuardrailAssessment(BaseModel):
    is_safe: bool = Field(..., description="Whether the prompt passed all safety checks")
    risk_score: float = Field(..., ge=0.0, le=1.0, description="Calculated adversarial risk score")
    matched_rules: List[str] = Field(default_factory=list, description="Triggered security rules")
    sanitized_prompt: str = Field(..., description="Normalized prompt string")
    reason: str = Field(..., description="Audit explanation of the guardrail decision")


class ExtractedEntitySchema(BaseModel):
    intent: Literal["financial_query", "document_summary", "data_extraction", "general_inquiry", "blocked"]
    entities: List[str] = Field(default_factory=list, description="Named entities or key subjects extracted")
    sentiment: Literal["positive", "neutral", "negative"] = Field(default="neutral")
    confidence: float = Field(..., ge=0.0, le=1.0)
    structured_payload: dict = Field(default_factory=dict, description="Domain-specific key-value attributes")


class WorkflowResponse(BaseModel):
    status: Literal["success", "blocked", "error"]
    guardrail: GuardrailAssessment
    extracted_data: Optional[ExtractedEntitySchema] = None
    error_detail: Optional[str] = None
    request_id: Optional[str] = Field(default=None, description="Correlation ID for the audit log")
    extraction_attempts: int = Field(default=0, description="LLM calls made (1 + self-repair retries)")
    output_redactions: List[str] = Field(default_factory=list, description="Types of data redacted by the output guard")
