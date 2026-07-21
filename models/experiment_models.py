"""Pydantic models for experiment execution results (Module 3)."""

from pydantic import BaseModel, Field


class ExperimentResult(BaseModel):
    """Result from executing a computational experiment via DeepSolver."""
    hypothesis_id: str = Field(
        description="ID of the hypothesis that was tested"
    )
    plan_summary: str = Field(
        description="Short description of what was executed"
    )
    success: bool = Field(
        description="Whether the experiment code executed successfully"
    )
    final_code: str = Field(
        default="",
        description="The final working code (empty if execution failed completely)"
    )
    execution_output: str = Field(
        default="",
        description="Raw execution output including numerical results"
    )
    error_log: str = Field(
        default="",
        description="Error messages encountered during execution (empty if success)"
    )
