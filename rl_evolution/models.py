"""
RL Evolution Data Models - RL假设演化数据模型

定义强化学习假设演化机制所需的所有数据结构：
- 假设特征向量
- 演化动作枚举
- 动作计划
- 奖励信号
- Bandit状态
"""

from __future__ import annotations

from enum import Enum
from typing import Dict, List, Optional
from pydantic import BaseModel, Field


class EvolutionAction(str, Enum):
    """假设演化的离散动作空间。"""
    MODIFY = "modify"              # 修改：调整假设陈述或置信度
    MERGE = "merge"                # 合并：与另一个假设融合
    ELIMINATE = "eliminate"         # 淘汰：从假设池中移除
    GENERATE_NEW = "generate_new"  # 生成新假设：从当前假设派生子代
    KEEP = "keep"                  # 保持：不做任何修改


# 所有可用的动作列表
ALL_ACTIONS: List[str] = [a.value for a in EvolutionAction]


class HypothesisFeatures(BaseModel):
    """单个假设的特征向量，用于RL状态表示。"""
    hypothesis_id: str = Field(description="假设唯一标识符")
    confidence: float = Field(description="当前置信度 [0.0, 1.0]")
    generation: int = Field(description="假设所处的代数")
    support_count: int = Field(default=0, description="本轮支持实验数")
    refute_count: int = Field(default=0, description="本轮驳斥实验数")
    status: str = Field(default="active", description="当前状态")
    age_iterations: int = Field(default=0, description="存活迭代数")


class HypothesisAction(BaseModel):
    """单个假设的动作分配。"""
    hypothesis_id: str = Field(description="目标假设ID")
    action: str = Field(description="选择的动作类型")
    score: float = Field(default=0.0, description="选择该动作的得分(UCB/采样值)")
    context: Optional[str] = Field(
        default=None,
        description="附加指导信息（如'merge with H3'）"
    )


class ActionPlan(BaseModel):
    """RL优化器的输出：每个假设的动作分配计划。"""
    actions: List[HypothesisAction] = Field(
        default_factory=list,
        description="每个假设的动作分配列表"
    )
    reasoning: str = Field(
        default="",
        description="动作选择的理由说明"
    )
    iteration: int = Field(default=0, description="所属迭代轮次")


class RewardSignal(BaseModel):
    """单次动作的奖励信号。"""
    iteration: int = Field(description="所属迭代轮次")
    hypothesis_id: str = Field(description="目标假设ID")
    action: str = Field(description="执行的动作")
    support_strength: float = Field(
        default=0.0, description="证据支持强度 [0.0, 1.0]"
    )
    novelty_gain: float = Field(
        default=0.0, description="新颖性增益 [0.0, 1.0]"
    )
    convergence_progress: float = Field(
        default=0.0, description="收敛进度 [0.0, 1.0]"
    )
    total_reward: float = Field(
        default=0.0, description="加权总奖励"
    )


class ArmStatistics(BaseModel):
    """单个Bandit臂的统计数据。"""
    name: str = Field(description="臂名称（动作类型）")
    pull_count: int = Field(default=0, description="被拉取次数")
    total_reward: float = Field(default=0.0, description="累计奖励")
    mean_reward: float = Field(default=0.0, description="平均奖励")
    # Thompson Sampling Beta分布参数
    alpha: float = Field(default=1.0, description="Beta分布alpha参数")
    beta_param: float = Field(default=1.0, description="Beta分布beta参数")


class BanditState(BaseModel):
    """可序列化的Bandit完整状态。"""
    algorithm: str = Field(default="thompson_sampling", description="算法类型")
    arm_stats: Dict[str, ArmStatistics] = Field(
        default_factory=dict, description="各臂统计数据"
    )
    history: List[RewardSignal] = Field(
        default_factory=list, description="历史奖励记录"
    )
    total_pulls: int = Field(default=0, description="总拉取次数")
    exploration_constant: float = Field(
        default=1.414, description="UCB1探索常数"
    )
