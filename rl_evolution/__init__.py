"""
RL Evolution Package - 基于强化学习的假设演化机制

本包实现 Multi-Armed Bandit 驱动的假设演化优化器，
为 LLM 演化智能体提供数据驱动的动作指导。

核心组件:
- RLEvolutionOptimizer: 顶层优化器（orchestrator 直接调用）
- BanditAlgorithm: MAB算法（UCB1 / Thompson Sampling）
- compute_reward: 奖励信号计算
- save_state / load_state: 状态持久化
"""

from .optimizer import RLEvolutionOptimizer
from .models import (
    ActionPlan,
    HypothesisAction,
    HypothesisFeatures,
    EvolutionAction,
    RewardSignal,
    ArmStatistics,
    BanditState,
    ALL_ACTIONS,
)
from .bandit import (
    BanditAlgorithm,
    UCB1Bandit,
    ThompsonSamplingBandit,
    create_bandit,
)
from .reward import compute_reward
from .persistence import save_state, load_state

__all__ = [
    "RLEvolutionOptimizer",
    "ActionPlan",
    "HypothesisAction",
    "HypothesisFeatures",
    "EvolutionAction",
    "RewardSignal",
    "ArmStatistics",
    "BanditState",
    "ALL_ACTIONS",
    "BanditAlgorithm",
    "UCB1Bandit",
    "ThompsonSamplingBandit",
    "create_bandit",
    "compute_reward",
    "save_state",
    "load_state",
]
