"""
Reward Computation - 奖励信号计算

实现论文中的奖励公式:
    r = alpha * support_strength + beta * novelty_gain + gamma * convergence_progress

根据实验结果和假设演化结果计算每个动作的奖励值。
"""

from __future__ import annotations

from typing import List, Optional

from .models import RewardSignal, EvolutionAction


def compute_support_strength(
    action: str,
    hypothesis_id: str,
    analyses: List,
    evolution: Optional[object] = None,
) -> float:
    """
    计算证据支持强度分量。

    根据不同动作类型有不同的计算逻辑:
    - modify: 分析后置信度（支持则高分）
    - eliminate: 如果被淘汰的假设确实被驳斥 → 高分（正确决策）
    - merge: 融合后假设置信度
    - generate_new: 新假设如果被后续实验支持 → 高分
    - keep: 如果假设仍然活跃/被确认 → 中等分数
    """
    # 查找该假设的分析结果
    matching_analysis = None
    for a in analyses:
        if hasattr(a, 'hypothesis_id') and a.hypothesis_id == hypothesis_id:
            matching_analysis = a
            break

    if action == EvolutionAction.ELIMINATE.value:
        # 淘汰动作：如果分析结果驳斥了假设（不支持），说明淘汰是正确的
        if matching_analysis:
            if not matching_analysis.supports_hypothesis:
                return 1.0  # 正确淘汰被驳斥的假设
            else:
                return 0.2  # 淘汰了仍有希望的假设（可能过早）
        return 0.5  # 无分析结果时给中性分数

    elif action == EvolutionAction.MODIFY.value:
        # 修改动作：看修改后分析是否支持
        if matching_analysis:
            if matching_analysis.supports_hypothesis:
                return matching_analysis.confidence
            else:
                return 0.3  # 修改后仍被驳斥
        return 0.5

    elif action == EvolutionAction.MERGE.value:
        # 合并动作：检查合并后假设的置信度
        if matching_analysis and matching_analysis.supports_hypothesis:
            return min(1.0, matching_analysis.confidence + 0.1)
        return 0.4

    elif action == EvolutionAction.GENERATE_NEW.value:
        # 生成新假设：根据是否有后续支持判断
        # 在当前轮次无法完全评估（需要下一轮实验），给予中性偏正的分数
        if matching_analysis:
            if matching_analysis.supports_hypothesis:
                return matching_analysis.confidence
            else:
                return 0.2
        return 0.5  # 新假设尚未被测试

    elif action == EvolutionAction.KEEP.value:
        # 保持动作：如果假设仍然活跃且有支持，说明保持是合理的
        if matching_analysis:
            if matching_analysis.supports_hypothesis:
                return matching_analysis.confidence * 0.8  # 稍低于修改
            else:
                return 0.2  # 应该做点什么但选择了保持
        return 0.4

    return 0.5  # 默认中性


def compute_novelty_gain(
    novelty_before: float,
    novelty_after: float,
) -> float:
    """
    计算新颖性增益分量。

    novelty_gain = normalize(novelty_after - novelty_before)
    归一化到 [0, 1]，使用 max(0, delta) / 0.3 并截断到 1.0。
    如果新颖性下降了，返回 0。
    """
    delta = novelty_after - novelty_before
    if delta <= 0:
        return 0.0
    # 以 0.3 的增益为满分标准化
    normalized = min(1.0, delta / 0.3)
    return normalized


def compute_convergence_progress(
    evolution: Optional[object],
    hypotheses: List,
) -> float:
    """
    计算收敛进度分量。

    - 如果已收敛 → 1.0
    - 否则 → (已确认 + 已驳斥) / 总假设数
    """
    if evolution and hasattr(evolution, 'converged') and evolution.converged:
        return 1.0

    if not hypotheses:
        return 0.0

    resolved = 0
    total = len(hypotheses)
    for h in hypotheses:
        if hasattr(h, 'status'):
            if h.status in ("confirmed", "refuted", "eliminated"):
                resolved += 1

    return resolved / total if total > 0 else 0.0


def compute_reward(
    action: str,
    hypothesis_id: str,
    analyses: List,
    novelty_before: float,
    novelty_after: float,
    evolution: Optional[object],
    hypotheses: List,
    alpha: float = 0.5,
    beta: float = 0.2,
    gamma: float = 0.3,
    iteration: int = 0,
) -> RewardSignal:
    """
    计算完整的奖励信号。

    r = alpha * support_strength + beta * novelty_gain + gamma * convergence_progress

    参数:
        action: 执行的动作类型
        hypothesis_id: 目标假设ID
        analyses: 本轮分析结果列表
        novelty_before: 演化前新颖性评分
        novelty_after: 演化后新颖性评分
        evolution: HypothesisEvolution 对象
        hypotheses: 演化后的假设列表
        alpha: support_strength 权重
        beta: novelty_gain 权重
        gamma: convergence_progress 权重
        iteration: 当前迭代轮次

    返回:
        RewardSignal 对象
    """
    support = compute_support_strength(action, hypothesis_id, analyses, evolution)
    novelty = compute_novelty_gain(novelty_before, novelty_after)
    convergence = compute_convergence_progress(evolution, hypotheses)

    total = alpha * support + beta * novelty + gamma * convergence

    return RewardSignal(
        iteration=iteration,
        hypothesis_id=hypothesis_id,
        action=action,
        support_strength=support,
        novelty_gain=novelty,
        convergence_progress=convergence,
        total_reward=total,
    )


# ---------------------------------------------------------------------------
# 跨域迁移奖励 — 用于元学习反馈
# ---------------------------------------------------------------------------

def compute_transfer_reward(
    transfer_success: bool,
    performance_delta: float = 0.0,
    domain_similarity: float = 0.3,
) -> float:
    """
    计算跨域技能迁移的奖励信号。

    当一个技能从源域成功迁移到目标域时，产生正向奖励；
    失败时产生负向奖励。奖励值可以注入到 RL 优化器中，
    影响未来的假设演化策略。

    参数:
        transfer_success: 迁移后技能是否执行成功
        performance_delta: 迁移后性能变化（正=提升, 负=下降）
        domain_similarity: 源域和目标域的相似度 [0,1]

    返回:
        奖励值 [-1.0, 1.0]
    """
    if transfer_success:
        # 成功迁移：基础奖励 + 性能提升加成 + 远域迁移加分
        base_reward = 0.6
        performance_bonus = min(0.2, max(0.0, performance_delta))
        # 相似度越低的域迁移成功，奖励越高（说明模式更通用）
        distance_bonus = 0.2 * (1.0 - domain_similarity)
        return min(1.0, base_reward + performance_bonus + distance_bonus)
    else:
        # 失败迁移：温和惩罚
        base_penalty = -0.3
        # 高相似度域的迁移失败惩罚更重（本不该失败）
        similarity_penalty = -0.2 * domain_similarity
        return max(-1.0, base_penalty + similarity_penalty)

