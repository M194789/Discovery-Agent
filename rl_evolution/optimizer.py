"""
RL Evolution Optimizer - RL假设演化优化器

顶层优化器类，协调 Bandit 选择、动作遮蔽、奖励计算和状态更新。
这是 orchestrator 直接调用的入口。
"""

from __future__ import annotations

from typing import Dict, List, Optional

from .models import (
    ActionPlan, HypothesisAction, HypothesisFeatures,
    EvolutionAction, ALL_ACTIONS, RewardSignal,
)
from .bandit import BanditAlgorithm, create_bandit
from .reward import compute_reward


class RLEvolutionOptimizer:
    """
    基于强化学习的假设演化优化器。

    使用 Multi-Armed Bandit 算法（UCB1 或 Thompson Sampling）
    为每个假设选择最优演化动作，并根据实验结果更新策略。

    与 LLM 的协作模式：
    - RL 选择动作类型（modify/merge/eliminate/generate_new/keep）
    - LLM 根据 RL 指导生成具体演化内容
    """

    def __init__(
        self,
        algorithm: str = "thompson_sampling",
        alpha: float = 0.5,
        beta: float = 0.2,
        gamma: float = 0.3,
        exploration_constant: float = 1.414,
        session_id: str = "",
    ):
        """
        初始化 RL 优化器。

        参数:
            algorithm: MAB算法类型 ("ucb1" 或 "thompson_sampling")
            alpha: 奖励公式中 support_strength 权重
            beta: 奖励公式中 novelty_gain 权重
            gamma: 奖励公式中 convergence_progress 权重
            exploration_constant: UCB1 探索常数 c
            session_id: 会话ID（用于持久化）
        """
        self.algorithm = algorithm
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.exploration_constant = exploration_constant
        self.session_id = session_id

        # 创建 Bandit 实例
        self.bandit: BanditAlgorithm = create_bandit(
            algorithm=algorithm,
            exploration_constant=exploration_constant,
        )

        # 历史奖励记录
        self.reward_history: List[RewardSignal] = []

    def select_actions(
        self,
        hypotheses: List,
        analyses: List,
        iteration: int,
    ) -> ActionPlan:
        """
        为每个假设选择最优演化动作。

        流程:
        1. 提取每个假设的特征
        2. 根据假设状态计算动作遮蔽
        3. 使用 Bandit 为每个假设选择动作
        4. 组装 ActionPlan

        参数:
            hypotheses: 当前所有假设列表
            analyses: 本轮分析结果列表
            iteration: 当前迭代轮次

        返回:
            ActionPlan 包含每个假设的动作分配
        """
        actions: List[HypothesisAction] = []
        active_count = sum(
            1 for h in hypotheses
            if hasattr(h, 'status') and h.status == "active"
        )

        # 构建分析映射 hypothesis_id -> analysis
        analysis_map = {}
        for a in analyses:
            if hasattr(a, 'hypothesis_id'):
                analysis_map[a.hypothesis_id] = a

        reasoning_parts = []

        for hyp in hypotheses:
            hyp_id = hyp.id if hasattr(hyp, 'id') else str(hyp)
            status = hyp.status if hasattr(hyp, 'status') else "active"
            confidence = hyp.confidence if hasattr(hyp, 'confidence') else 0.5

            # 获取该假设的分析结果
            analysis = analysis_map.get(hyp_id)
            is_supported = (
                analysis.supports_hypothesis if analysis else False
            )

            # 计算动作遮蔽
            excluded = self._compute_action_mask(
                status=status,
                confidence=confidence,
                is_supported=is_supported,
                active_count=active_count,
                iteration=iteration,
            )

            # Bandit 选择动作
            selected_action = self.bandit.select_arm(excluded=excluded)

            # 获取分数（用于日志和LLM指导）
            score = self._get_action_score(selected_action, excluded)

            # 生成上下文信息
            context = self._generate_context(
                selected_action, hyp, hypotheses, analysis
            )

            actions.append(HypothesisAction(
                hypothesis_id=hyp_id,
                action=selected_action,
                score=score,
                context=context,
            ))

            reasoning_parts.append(
                f"{hyp_id}({status},conf={confidence:.2f})->{selected_action}"
                f"(score={score:.2f})"
            )

        reasoning = (
            f"Iteration {iteration}: MAB({self.algorithm}) selected actions "
            f"based on {self.bandit.get_total_pulls()} historical observations. "
            + "; ".join(reasoning_parts)
        )

        return ActionPlan(
            actions=actions,
            reasoning=reasoning,
            iteration=iteration,
        )

    def update_from_outcome(
        self,
        action_plan: ActionPlan,
        evolution: Optional[object],
        analyses: List,
        hypotheses_after: List,
        novelty_before: float,
        novelty_after: float,
        iteration: int = 0,
    ) -> List[RewardSignal]:
        """
        根据演化结果计算奖励并更新 Bandit。

        参数:
            action_plan: 先前的动作计划
            evolution: HypothesisEvolution 结果对象
            analyses: 本轮分析结果
            hypotheses_after: 演化后的假设列表
            novelty_before: 演化前的新颖性评分
            novelty_after: 演化后的新颖性评分
            iteration: 当前迭代轮次

        返回:
            本轮产生的所有 RewardSignal 列表
        """
        rewards: List[RewardSignal] = []

        for ha in action_plan.actions:
            reward_signal = compute_reward(
                action=ha.action,
                hypothesis_id=ha.hypothesis_id,
                analyses=analyses,
                novelty_before=novelty_before,
                novelty_after=novelty_after,
                evolution=evolution,
                hypotheses=hypotheses_after,
                alpha=self.alpha,
                beta=self.beta,
                gamma=self.gamma,
                iteration=iteration,
            )
            rewards.append(reward_signal)

            # 更新 Bandit 臂统计
            self.bandit.update(ha.action, reward_signal.total_reward)

        # 追加到历史记录
        self.reward_history.extend(rewards)

        return rewards

    def get_statistics(self) -> Dict:
        """获取当前 Bandit 的统计信息（用于日志和调试）。"""
        stats = self.bandit.get_arm_stats()
        return {
            "algorithm": self.algorithm,
            "total_pulls": self.bandit.get_total_pulls(),
            "arms": {
                name: {
                    "pulls": arm.pull_count,
                    "mean_reward": round(arm.mean_reward, 3),
                    "alpha": round(arm.alpha, 2),
                    "beta": round(arm.beta_param, 2),
                }
                for name, arm in stats.items()
            },
            "total_rewards_recorded": len(self.reward_history),
        }

    def _compute_action_mask(
        self,
        status: str,
        confidence: float,
        is_supported: bool,
        active_count: int,
        iteration: int,
    ) -> List[str]:
        """
        计算动作遮蔽规则：哪些动作对当前假设不可用。

        规则:
        - confirmed 状态 → 只能 keep 或 eliminate
        - refuted 状态 → 只能 eliminate 或 generate_new
        - 活跃假设太少 → 不能 merge 或 eliminate
        - 活跃假设太多 → 不能 generate_new
        - 第一轮 → 不能 merge
        - 高置信度且被支持 → 不需要 modify 或 eliminate
        """
        excluded: List[str] = []

        if status == "confirmed":
            excluded.extend(["modify", "merge", "generate_new"])
        elif status == "refuted":
            excluded.extend(["modify", "merge", "keep"])
        else:
            # active 状态的通用规则
            if active_count <= 1:
                excluded.extend(["merge", "eliminate"])
            if active_count >= 10:
                excluded.append("generate_new")
            if iteration == 1:
                excluded.append("merge")
            if confidence >= 0.8 and is_supported:
                excluded.extend(["modify", "eliminate"])

        # 去重
        return list(set(excluded))

    def _get_action_score(
        self, action: str, excluded: Optional[List[str]]
    ) -> float:
        """获取选中动作的得分，用于日志记录。"""
        if hasattr(self.bandit, 'get_ucb_scores'):
            scores = self.bandit.get_ucb_scores(excluded)
            return scores.get(action, 0.0)
        elif hasattr(self.bandit, 'get_expected_values'):
            values = self.bandit.get_expected_values(excluded)
            return values.get(action, 0.0)
        return 0.0

    def _generate_context(
        self,
        action: str,
        hypothesis,
        all_hypotheses: List,
        analysis,
    ) -> Optional[str]:
        """为动作生成额外的上下文指导信息。"""
        if action == EvolutionAction.MERGE.value:
            # 为合并动作找到最相关的候选假设
            hyp_id = hypothesis.id if hasattr(hypothesis, 'id') else ""
            candidates = [
                h.id for h in all_hypotheses
                if hasattr(h, 'id') and h.id != hyp_id
                and hasattr(h, 'status') and h.status == "active"
            ]
            if candidates:
                return f"Consider merging with: {', '.join(candidates[:2])}"

        elif action == EvolutionAction.MODIFY.value:
            if analysis and hasattr(analysis, 'key_findings'):
                findings = analysis.key_findings[:2] if analysis.key_findings else []
                if findings:
                    return f"Modify based on findings: {'; '.join(findings)}"

        elif action == EvolutionAction.GENERATE_NEW.value:
            if analysis and hasattr(analysis, 'key_findings'):
                findings = analysis.key_findings[:1] if analysis.key_findings else []
                if findings:
                    return f"Derive new hypothesis exploring: {findings[0]}"

        return None
