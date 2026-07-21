"""用于实验风险评估的 Pydantic 模型（Phase B.5）。

该模块定义了实验风险评估阶段使用的数据结构，
用于在实验执行前评估每个实验计划的计算可行性、潜在失败模式和替代策略。
"""

from pydantic import BaseModel, Field
from typing import List


class RiskAssessment(BaseModel):
    """单个实验计划的风险评估结果。"""
    hypothesis_id: str = Field(
        description="关联的假设 ID，对应 ExperimentPlan.hypothesis_id"
    )
    plan_method: str = Field(
        description="实验计划的计算方法（如 DFT、MD、ML_modeling 等）"
    )
    feasibility_score: float = Field(
        description="可行性评分（0.0 表示完全不可行，1.0 表示完全可行）"
    )
    risk_level: str = Field(
        description="风险等级：'low'（低）、'medium'（中）、'high'（高）或 'critical'（严重）"
    )
    is_feasible: bool = Field(
        description="最终可行性判定：True 表示可执行，False 表示应过滤"
    )
    missing_packages: List[str] = Field(
        default_factory=list,
        description="实验计划所需但当前环境中不可用的包列表"
    )
    failure_modes: List[str] = Field(
        default_factory=list,
        description="已识别的潜在失败模式（如 ImportError、超时、数据不足等）"
    )
    timeout_risk: str = Field(
        default="medium",
        description="预估的超时风险：'low'（低）、'medium'（中）或 'high'（高）"
    )
    alternative_suggestion: str = Field(
        default="",
        description="当计划被判定为高风险时，建议的更简单替代策略（可行计划为空）"
    )
    filtering_reason: str = Field(
        default="",
        description="计划被过滤的原因（可行计划为空）"
    )
    computational_cost_estimate: str = Field(
        default="moderate",
        description="计算成本预估：'trivial'（极低）、'moderate'（中等）、'heavy'（较高）或 'excessive'（过高）"
    )
