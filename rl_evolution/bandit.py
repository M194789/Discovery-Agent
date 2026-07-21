"""
Multi-Armed Bandit Implementations - 多臂老虎机算法实现

提供 UCB1 和 Thompson Sampling 两种 MAB 算法，
用于在假设演化操作之间做探索-利用权衡。
"""

from __future__ import annotations

import math
import random
from abc import ABC, abstractmethod
from typing import Dict, List, Optional

from .models import ArmStatistics, BanditState, ALL_ACTIONS


class BanditAlgorithm(ABC):
    """MAB算法抽象基类。"""

    def __init__(self, arms: List[str], exploration_constant: float = 1.414):
        self.arms = arms
        self.exploration_constant = exploration_constant
        self._stats: Dict[str, ArmStatistics] = {}
        self._total_pulls: int = 0
        self._initialize_arms()

    @abstractmethod
    def _initialize_arms(self):
        """初始化各臂的统计数据。"""
        ...

    @abstractmethod
    def select_arm(self, excluded: Optional[List[str]] = None) -> str:
        """选择一个臂（动作），排除被遮蔽的臂。"""
        ...

    def update(self, arm: str, reward: float) -> None:
        """根据奖励信号更新臂的统计数据。"""
        if arm not in self._stats:
            return
        stats = self._stats[arm]
        stats.pull_count += 1
        stats.total_reward += reward
        stats.mean_reward = stats.total_reward / stats.pull_count
        # Thompson Sampling 更新 Beta 分布参数
        # 奖励 clamp 到 [0, 1] 范围
        clamped = max(0.0, min(1.0, reward))
        stats.alpha += clamped
        stats.beta_param += (1.0 - clamped)
        self._total_pulls += 1

    def get_arm_stats(self) -> Dict[str, ArmStatistics]:
        """返回所有臂的统计数据副本。"""
        return {k: v.model_copy() for k, v in self._stats.items()}

    def get_total_pulls(self) -> int:
        return self._total_pulls

    def to_state(self) -> BanditState:
        """序列化为 BanditState 对象。"""
        return BanditState(
            algorithm=self._algorithm_name(),
            arm_stats={k: v.model_copy() for k, v in self._stats.items()},
            total_pulls=self._total_pulls,
            exploration_constant=self.exploration_constant,
        )

    @classmethod
    def from_state(cls, state: BanditState) -> "BanditAlgorithm":
        """从 BanditState 恢复算法实例。"""
        if state.algorithm == "ucb1":
            bandit = UCB1Bandit(
                arms=list(state.arm_stats.keys()) or ALL_ACTIONS,
                exploration_constant=state.exploration_constant,
            )
        else:
            bandit = ThompsonSamplingBandit(
                arms=list(state.arm_stats.keys()) or ALL_ACTIONS,
                exploration_constant=state.exploration_constant,
            )
        # 恢复统计
        for name, arm_stat in state.arm_stats.items():
            if name in bandit._stats:
                bandit._stats[name] = arm_stat.model_copy()
        bandit._total_pulls = state.total_pulls
        return bandit

    @abstractmethod
    def _algorithm_name(self) -> str:
        ...

    def _get_available_arms(self, excluded: Optional[List[str]] = None) -> List[str]:
        """获取可用臂列表（排除被遮蔽的）。"""
        if not excluded:
            return list(self.arms)
        available = [a for a in self.arms if a not in excluded]
        # 如果全部被排除，至少保留 "keep"
        if not available:
            return ["keep"] if "keep" in self.arms else [self.arms[0]]
        return available


class UCB1Bandit(BanditAlgorithm):
    """
    UCB1 (Upper Confidence Bound) 算法。

    选择公式: arm = argmax( mean_i + c * sqrt(ln(N) / n_i) )
    - mean_i: 臂i的平均奖励
    - c: 探索常数（默认 sqrt(2)）
    - N: 总拉取次数
    - n_i: 臂i的拉取次数
    """

    def _initialize_arms(self):
        for arm in self.arms:
            self._stats[arm] = ArmStatistics(
                name=arm, pull_count=0, total_reward=0.0,
                mean_reward=0.0, alpha=1.0, beta_param=1.0,
            )

    def _algorithm_name(self) -> str:
        return "ucb1"

    def select_arm(self, excluded: Optional[List[str]] = None) -> str:
        available = self._get_available_arms(excluded)

        # 冷启动：对未被拉取过的臂，随机选择一个
        unvisited = [a for a in available if self._stats[a].pull_count == 0]
        if unvisited:
            return random.choice(unvisited)

        # 如果总拉取次数为0（不应出现），随机选
        if self._total_pulls == 0:
            return random.choice(available)

        # UCB1 计算
        best_arm = available[0]
        best_score = -float("inf")
        ln_total = math.log(self._total_pulls)

        for arm in available:
            stats = self._stats[arm]
            if stats.pull_count == 0:
                return arm  # 安全检查
            ucb_score = stats.mean_reward + self.exploration_constant * math.sqrt(
                ln_total / stats.pull_count
            )
            if ucb_score > best_score:
                best_score = ucb_score
                best_arm = arm

        return best_arm

    def get_ucb_scores(self, excluded: Optional[List[str]] = None) -> Dict[str, float]:
        """返回各臂的UCB分数，用于日志记录。"""
        available = self._get_available_arms(excluded)
        scores = {}
        if self._total_pulls == 0:
            for arm in available:
                scores[arm] = 0.0
            return scores

        ln_total = math.log(max(1, self._total_pulls))
        for arm in available:
            stats = self._stats[arm]
            if stats.pull_count == 0:
                scores[arm] = float("inf")
            else:
                scores[arm] = stats.mean_reward + self.exploration_constant * math.sqrt(
                    ln_total / stats.pull_count
                )
        return scores


class ThompsonSamplingBandit(BanditAlgorithm):
    """
    Thompson Sampling 算法（Beta-Bernoulli 模型）。

    每个臂用 Beta(alpha, beta) 分布建模。
    选择时从各臂的Beta分布采样，选择最高采样值的臂。
    更新时：alpha += reward, beta += (1 - reward)。

    使用信息性先验（cold start priors）加速收敛。
    """

    # 各动作的先验参数（基于领域知识的初始信念）
    DEFAULT_PRIORS: Dict[str, tuple] = {
        "modify": (3.0, 2.0),        # 偏正：修改通常有效
        "merge": (2.0, 3.0),         # 偏负：合并常失败
        "eliminate": (2.0, 2.0),     # 中性
        "generate_new": (3.0, 2.0),  # 偏正：探索有价值
        "keep": (2.0, 2.0),          # 中性
    }

    def _initialize_arms(self):
        for arm in self.arms:
            alpha, beta = self.DEFAULT_PRIORS.get(arm, (1.0, 1.0))
            self._stats[arm] = ArmStatistics(
                name=arm, pull_count=0, total_reward=0.0,
                mean_reward=0.0, alpha=alpha, beta_param=beta,
            )

    def _algorithm_name(self) -> str:
        return "thompson_sampling"

    def select_arm(self, excluded: Optional[List[str]] = None) -> str:
        available = self._get_available_arms(excluded)

        best_arm = available[0]
        best_sample = -1.0

        for arm in available:
            stats = self._stats[arm]
            # 从 Beta(alpha, beta) 分布采样
            sample = random.betavariate(
                max(0.01, stats.alpha), max(0.01, stats.beta_param)
            )
            if sample > best_sample:
                best_sample = sample
                best_arm = arm

        return best_arm

    def get_expected_values(self, excluded: Optional[List[str]] = None) -> Dict[str, float]:
        """返回各臂的期望值 E[Beta] = alpha/(alpha+beta)，用于日志记录。"""
        available = self._get_available_arms(excluded)
        values = {}
        for arm in available:
            stats = self._stats[arm]
            total = stats.alpha + stats.beta_param
            values[arm] = stats.alpha / total if total > 0 else 0.5
        return values


def create_bandit(algorithm: str = "thompson_sampling",
                  exploration_constant: float = 1.414) -> BanditAlgorithm:
    """工厂函数：创建指定类型的Bandit实例。"""
    if algorithm == "ucb1":
        return UCB1Bandit(arms=ALL_ACTIONS, exploration_constant=exploration_constant)
    else:
        return ThompsonSamplingBandit(
            arms=ALL_ACTIONS, exploration_constant=exploration_constant
        )
