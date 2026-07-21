"""Pydantic models for result analysis and hypothesis evolution (Module 4).

包含基础分析结果模型和高级分析增强模型（统计检验、可视化、因果推断）。
"""

from pydantic import BaseModel, Field
from typing import Dict, List, Optional

from .hypothesis_models import Hypothesis


class AnalysisResult(BaseModel):
    """Analysis of a single experiment result against its hypothesis."""
    hypothesis_id: str = Field(
        description="ID of the hypothesis that was tested"
    )
    supports_hypothesis: bool = Field(
        description="Whether the experimental evidence supports the hypothesis"
    )
    confidence: float = Field(
        description="Statistical confidence in the conclusion (0.0 to 1.0)"
    )
    key_findings: List[str] = Field(
        description="Bullet-point key findings from the analysis"
    )
    statistical_summary: str = Field(
        description="Summary of any statistical analysis performed"
    )
    explanation: str = Field(
        description="Comprehensive interpretation of the results"
    )


class HypothesisEvolution(BaseModel):
    """How hypotheses evolved after one iteration of the discovery loop."""
    iteration: int = Field(
        default=0,
        description="Which discovery loop iteration this represents"
    )
    hypothesis_updates: List[Hypothesis] = Field(
        default_factory=list,
        description="Updated hypotheses with changed statuses"
    )
    new_hypotheses: List[Hypothesis] = Field(
        default_factory=list,
        description="Newly generated hypotheses for the next iteration"
    )
    eliminated_ids: List[str] = Field(
        default_factory=list,
        description="IDs of hypotheses eliminated in this iteration"
    )
    evolution_reasoning: str = Field(
        default="",
        description="Explanation of why the changes were made"
    )
    converged: bool = Field(
        default=False,
        description=(
            "Whether the discovery process has reached "
            "a satisfactory conclusion"
        )
    )


# ---------------------------------------------------------------------------
# 高级分析增强模型 - Advanced Analysis Enrichment Models
# ---------------------------------------------------------------------------


class StatisticalTestResult(BaseModel):
    """单个统计检验的结果。"""
    test_name: str = Field(
        description="Name of the statistical test (e.g., 't-test', 'chi-square')"
    )
    test_statistic: float = Field(
        description="Computed test statistic value"
    )
    p_value: float = Field(
        description="p-value of the test"
    )
    is_significant: bool = Field(
        description="Whether p < significance level (default 0.05)"
    )
    confidence_interval: Optional[List[float]] = Field(
        default=None,
        description="95% confidence interval [lower, upper]"
    )
    effect_size: Optional[float] = Field(
        default=None,
        description="Effect size measure (e.g., Cohen's d)"
    )
    interpretation: str = Field(
        description="Human-readable interpretation of the test result"
    )


class VisualizationArtifact(BaseModel):
    """生成的可视化图表的元数据。"""
    chart_type: str = Field(
        description="Chart type: 'bar', 'scatter', 'distribution', 'heatmap', 'line'"
    )
    file_path: str = Field(
        description="Absolute path to the saved chart image file"
    )
    title: str = Field(
        description="Chart title"
    )
    description: str = Field(
        description="Description of what the chart shows"
    )


class CausalRelationship(BaseModel):
    """识别出的单个因果关系。"""
    cause: str = Field(
        description="The causal variable name"
    )
    effect: str = Field(
        description="The effect variable name"
    )
    strength: float = Field(
        description="Edge weight / relationship strength (0.0-1.0)"
    )
    confidence: float = Field(
        description="Confidence in the causal direction (0.0-1.0)"
    )
    mechanism: str = Field(
        description="Brief description of hypothesized mechanism"
    )


class CausalInferenceResult(BaseModel):
    """贝叶斯因果推断的完整输出。"""
    relationships: List[CausalRelationship] = Field(
        default_factory=list,
        description="Identified causal edges"
    )
    network_structure: Dict[str, List[str]] = Field(
        default_factory=dict,
        description="Adjacency list: {node: [children]}"
    )
    confounders_identified: List[str] = Field(
        default_factory=list,
        description="Potential confounding variables identified"
    )
    model_fit_score: Optional[float] = Field(
        default=None,
        description="BIC or similar model fit metric"
    )
    network_diagram_path: Optional[str] = Field(
        default=None,
        description="Path to the visualized Bayesian network diagram"
    )


class EnrichedAnalysisResult(AnalysisResult):
    """扩展 AnalysisResult，包含程序化分析结果（统计检验、可视化、因果推断）。"""
    statistical_tests: List[StatisticalTestResult] = Field(
        default_factory=list,
        description="Results of statistical hypothesis tests"
    )
    visualizations: List[VisualizationArtifact] = Field(
        default_factory=list,
        description="Generated visualization chart metadata"
    )
    causal_inference: Optional[CausalInferenceResult] = Field(
        default=None,
        description="Bayesian causal inference results"
    )
    extracted_numerical_data: Dict[str, float] = Field(
        default_factory=dict,
        description="Parsed key-value numerical data from execution output"
    )
    enrichment_errors: List[str] = Field(
        default_factory=list,
        description="Non-fatal errors encountered during enrichment"
    )
