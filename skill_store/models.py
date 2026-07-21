"""
技能数据模型 — Pydantic 模型定义

定义技能的完整生命周期数据结构，包括：
- 技能元数据（领域、方法、工具链）
- 性能指标（成功率、执行时间）
- 溯源信息（来源会话、迭代）
- 迁移记录（跨域适配历史）
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import List, Optional

from pydantic import BaseModel, Field


class SkillPerformanceMetrics(BaseModel):
    """技能性能指标"""

    success_rate: float = 0.0
    total_uses: int = 0
    successful_uses: int = 0
    avg_execution_time_s: Optional[float] = None
    confidence: float = 0.5
    last_used: Optional[str] = None

    def record_use(self, success: bool, execution_time: Optional[float] = None):
        """记录一次技能使用结果"""
        self.total_uses += 1
        if success:
            self.successful_uses += 1
        self.success_rate = self.successful_uses / self.total_uses

        if execution_time is not None:
            if self.avg_execution_time_s is None:
                self.avg_execution_time_s = execution_time
            else:
                # 指数移动平均
                self.avg_execution_time_s = (
                    0.7 * self.avg_execution_time_s + 0.3 * execution_time
                )

        self.last_used = datetime.now().isoformat()


class SkillProvenance(BaseModel):
    """技能溯源信息"""

    session_id: str
    iteration: int
    hypothesis_id: str = ""
    user_id: str = "experiment_system"
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())


class TransferRecord(BaseModel):
    """跨域迁移记录"""

    source_domain: str
    target_domain: str
    adaptation_method: str = "llm_guided"  # "llm_guided" | "template_substitution"
    success: bool = False
    performance_delta: float = 0.0
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())


class Skill(BaseModel):
    """完整技能数据模型"""

    id: str = Field(default_factory=lambda: str(uuid.uuid4())[:12])
    version: int = 1
    name: str = ""
    domain: str = ""
    domain_tags: List[str] = Field(default_factory=list)
    method: str = ""
    tool_chain: List[str] = Field(default_factory=list)
    steps: List[str] = Field(default_factory=list)
    code_specification: str = ""
    final_code: str = ""
    success_criteria: str = ""
    preconditions: List[str] = Field(default_factory=list)
    postconditions: List[str] = Field(default_factory=list)
    performance_metrics: SkillPerformanceMetrics = Field(
        default_factory=SkillPerformanceMetrics
    )
    embedding: Optional[List[float]] = None
    provenance: SkillProvenance = Field(
        default_factory=lambda: SkillProvenance(
            session_id="unknown", iteration=0
        )
    )
    abstract_pattern: Optional[str] = None
    transfer_history: List[TransferRecord] = Field(default_factory=list)
    status: str = "active"  # active / deprecated / superseded
    created_at: str = Field(default_factory=lambda: datetime.now().isoformat())
    updated_at: str = Field(default_factory=lambda: datetime.now().isoformat())

    def to_embedding_text(self) -> str:
        """构造用于生成嵌入的文本表示"""
        parts = [
            self.domain,
            self.method,
            " ".join(self.steps),
            self.code_specification[:500] if self.code_specification else "",
            self.abstract_pattern or "",
        ]
        return " ".join(p for p in parts if p)

    def to_metadata(self) -> dict:
        """转换为 ChromaDB 元数据格式（仅支持基本类型）"""
        return {
            "domain": self.domain,
            "method": self.method,
            "status": self.status,
            "confidence": self.performance_metrics.confidence,
            "success_rate": self.performance_metrics.success_rate,
            "tool_chain": ",".join(self.tool_chain),
            "version": self.version,
            "name": self.name,
        }


class SkillQuery(BaseModel):
    """技能搜索查询"""

    query_text: str
    domain_filter: Optional[str] = None
    method_filter: Optional[str] = None
    tool_chain_filter: Optional[List[str]] = None
    min_confidence: float = 0.0
    top_k: int = 5
    include_cross_domain: bool = True


class SkillSearchResult(BaseModel):
    """技能搜索结果"""

    skill: Skill
    similarity_score: float
    is_cross_domain: bool = False
