class PromptSafetyError(Exception):
    """Raised when an input fails the AI safety guardrail check."""
    def __init__(self, message: str, risk_score: float, matched_rules: list[str]):
        super().__init__(message)
        self.message = message
        self.risk_score = risk_score
        self.matched_rules = matched_rules


class SchemaExtractionError(Exception):
    """Raised when the LLM output fails strict JSON schema validation."""
    def __init__(self, message: str, raw_output: str = ""):
        super().__init__(message)
        self.message = message
        self.raw_output = raw_output
