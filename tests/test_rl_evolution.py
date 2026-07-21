"""
Unit Tests for RL Evolution Module
RL假设演化模块单元测试
"""

import sys
import os
from pathlib import Path

# 确保项目根目录在 sys.path 中
project_root = str(Path(__file__).resolve().parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import pytest
from unittest.mock import MagicMock


# ============================================================
# Test Models
# ============================================================

class TestModels:
    """测试 RL 数据模型。"""

    def test_evolution_action_enum(self):
        from rl_evolution.models import EvolutionAction
        assert EvolutionAction.MODIFY.value == "modify"
        assert EvolutionAction.MERGE.value == "merge"
        assert EvolutionAction.ELIMINATE.value == "eliminate"
        assert EvolutionAction.GENERATE_NEW.value == "generate_new"
        assert EvolutionAction.KEEP.value == "keep"

    def test_all_actions_list(self):
        from rl_evolution.models import ALL_ACTIONS
        assert len(ALL_ACTIONS) == 5
        assert "modify" in ALL_ACTIONS
        assert "keep" in ALL_ACTIONS

    def test_hypothesis_features_creation(self):
        from rl_evolution.models import HypothesisFeatures
        feat = HypothesisFeatures(
            hypothesis_id="H1",
            confidence=0.75,
            generation=2,
            support_count=1,
            refute_count=0,
            status="active",
            age_iterations=1,
        )
        assert feat.hypothesis_id == "H1"
        assert feat.confidence == 0.75
        assert feat.generation == 2

    def test_action_plan_creation(self):
        from rl_evolution.models import ActionPlan, HypothesisAction
        plan = ActionPlan(
            actions=[
                HypothesisAction(hypothesis_id="H1", action="modify", score=0.8),
                HypothesisAction(hypothesis_id="H2", action="eliminate", score=0.7),
            ],
            reasoning="Test reasoning",
            iteration=1,
        )
        assert len(plan.actions) == 2
        assert plan.actions[0].action == "modify"
        assert plan.iteration == 1

    def test_reward_signal_creation(self):
        from rl_evolution.models import RewardSignal
        reward = RewardSignal(
            iteration=1,
            hypothesis_id="H1",
            action="modify",
            support_strength=0.8,
            novelty_gain=0.3,
            convergence_progress=0.5,
            total_reward=0.6,
        )
        assert reward.total_reward == 0.6
        assert reward.action == "modify"

    def test_arm_statistics_defaults(self):
        from rl_evolution.models import ArmStatistics
        stats = ArmStatistics(name="modify")
        assert stats.pull_count == 0
        assert stats.mean_reward == 0.0
        assert stats.alpha == 1.0
        assert stats.beta_param == 1.0


# ============================================================
# Test Bandit Algorithms
# ============================================================

class TestUCB1Bandit:
    """测试 UCB1 算法。"""

    def test_creation(self):
        from rl_evolution.bandit import UCB1Bandit, ALL_ACTIONS
        from rl_evolution.models import ALL_ACTIONS
        bandit = UCB1Bandit(arms=ALL_ACTIONS)
        assert bandit.get_total_pulls() == 0
        stats = bandit.get_arm_stats()
        assert len(stats) == 5

    def test_cold_start_selects_unvisited(self):
        from rl_evolution.bandit import UCB1Bandit
        from rl_evolution.models import ALL_ACTIONS
        bandit = UCB1Bandit(arms=ALL_ACTIONS)
        # 冷启动时应该选择未访问过的臂
        arm = bandit.select_arm()
        assert arm in ALL_ACTIONS

    def test_update_increases_pull_count(self):
        from rl_evolution.bandit import UCB1Bandit
        from rl_evolution.models import ALL_ACTIONS
        bandit = UCB1Bandit(arms=ALL_ACTIONS)
        bandit.update("modify", 0.8)
        stats = bandit.get_arm_stats()
        assert stats["modify"].pull_count == 1
        assert stats["modify"].mean_reward == 0.8
        assert bandit.get_total_pulls() == 1

    def test_ucb_prefers_high_reward_arm(self):
        from rl_evolution.bandit import UCB1Bandit
        from rl_evolution.models import ALL_ACTIONS
        bandit = UCB1Bandit(arms=ALL_ACTIONS, exploration_constant=0.1)
        # 给每个臂都拉一次（初始化）
        for arm in ALL_ACTIONS:
            bandit.update(arm, 0.3)
        # 给 modify 很高的奖励
        for _ in range(10):
            bandit.update("modify", 0.95)
        # 低探索常数下，UCB应该偏向高奖励臂
        selections = [bandit.select_arm() for _ in range(20)]
        assert selections.count("modify") > 10  # 大部分应该选择 modify

    def test_exclusion_works(self):
        from rl_evolution.bandit import UCB1Bandit
        from rl_evolution.models import ALL_ACTIONS
        bandit = UCB1Bandit(arms=ALL_ACTIONS)
        # 排除所有臂除了 "keep"
        excluded = [a for a in ALL_ACTIONS if a != "keep"]
        arm = bandit.select_arm(excluded=excluded)
        assert arm == "keep"

    def test_ucb_scores(self):
        from rl_evolution.bandit import UCB1Bandit
        from rl_evolution.models import ALL_ACTIONS
        bandit = UCB1Bandit(arms=ALL_ACTIONS)
        for arm in ALL_ACTIONS:
            bandit.update(arm, 0.5)
        scores = bandit.get_ucb_scores()
        assert len(scores) == 5
        # 所有臂都被拉过一次，分数应该都有值
        for score in scores.values():
            assert score > 0


class TestThompsonSamplingBandit:
    """测试 Thompson Sampling 算法。"""

    def test_creation_with_priors(self):
        from rl_evolution.bandit import ThompsonSamplingBandit
        from rl_evolution.models import ALL_ACTIONS
        bandit = ThompsonSamplingBandit(arms=ALL_ACTIONS)
        stats = bandit.get_arm_stats()
        # 验证先验设置正确
        assert stats["modify"].alpha == 3.0
        assert stats["modify"].beta_param == 2.0
        assert stats["merge"].alpha == 2.0
        assert stats["merge"].beta_param == 3.0

    def test_select_arm_returns_valid(self):
        from rl_evolution.bandit import ThompsonSamplingBandit
        from rl_evolution.models import ALL_ACTIONS
        bandit = ThompsonSamplingBandit(arms=ALL_ACTIONS)
        for _ in range(50):
            arm = bandit.select_arm()
            assert arm in ALL_ACTIONS

    def test_update_modifies_posterior(self):
        from rl_evolution.bandit import ThompsonSamplingBandit
        from rl_evolution.models import ALL_ACTIONS
        bandit = ThompsonSamplingBandit(arms=ALL_ACTIONS)
        initial_alpha = bandit.get_arm_stats()["modify"].alpha
        bandit.update("modify", 1.0)
        updated_alpha = bandit.get_arm_stats()["modify"].alpha
        assert updated_alpha > initial_alpha

    def test_high_reward_shifts_preference(self):
        from rl_evolution.bandit import ThompsonSamplingBandit
        from rl_evolution.models import ALL_ACTIONS
        bandit = ThompsonSamplingBandit(arms=ALL_ACTIONS)
        # 大量高奖励给 eliminate
        for _ in range(50):
            bandit.update("eliminate", 0.95)
        # 其他臂给低奖励
        for arm in ALL_ACTIONS:
            if arm != "eliminate":
                for _ in range(5):
                    bandit.update(arm, 0.1)
        # 多次选择，eliminate 应该被频繁选中
        selections = [bandit.select_arm() for _ in range(100)]
        assert selections.count("eliminate") > 50

    def test_expected_values(self):
        from rl_evolution.bandit import ThompsonSamplingBandit
        from rl_evolution.models import ALL_ACTIONS
        bandit = ThompsonSamplingBandit(arms=ALL_ACTIONS)
        values = bandit.get_expected_values()
        assert len(values) == 5
        # 初始先验下 modify 的期望值应该是 3/(3+2)=0.6
        assert abs(values["modify"] - 0.6) < 0.01

    def test_exclusion_works(self):
        from rl_evolution.bandit import ThompsonSamplingBandit
        from rl_evolution.models import ALL_ACTIONS
        bandit = ThompsonSamplingBandit(arms=ALL_ACTIONS)
        excluded = ["modify", "merge", "eliminate", "generate_new"]
        arm = bandit.select_arm(excluded=excluded)
        assert arm == "keep"


class TestBanditSerialization:
    """测试 Bandit 状态序列化和反序列化。"""

    def test_to_state_and_from_state_ucb1(self):
        from rl_evolution.bandit import UCB1Bandit, BanditAlgorithm
        from rl_evolution.models import ALL_ACTIONS
        bandit = UCB1Bandit(arms=ALL_ACTIONS)
        bandit.update("modify", 0.8)
        bandit.update("eliminate", 0.6)

        state = bandit.to_state()
        assert state.algorithm == "ucb1"
        assert state.total_pulls == 2

        restored = BanditAlgorithm.from_state(state)
        assert restored.get_total_pulls() == 2
        stats = restored.get_arm_stats()
        assert stats["modify"].pull_count == 1

    def test_to_state_and_from_state_thompson(self):
        from rl_evolution.bandit import ThompsonSamplingBandit, BanditAlgorithm
        from rl_evolution.models import ALL_ACTIONS
        bandit = ThompsonSamplingBandit(arms=ALL_ACTIONS)
        bandit.update("generate_new", 0.9)

        state = bandit.to_state()
        assert state.algorithm == "thompson_sampling"

        restored = BanditAlgorithm.from_state(state)
        stats = restored.get_arm_stats()
        assert stats["generate_new"].pull_count == 1


# ============================================================
# Test Reward Computation
# ============================================================

class TestRewardComputation:
    """测试奖励信号计算。"""

    def _make_analysis(self, hyp_id, supports, confidence):
        """创建模拟的分析结果。"""
        analysis = MagicMock()
        analysis.hypothesis_id = hyp_id
        analysis.supports_hypothesis = supports
        analysis.confidence = confidence
        analysis.key_findings = ["finding1"]
        return analysis

    def test_modify_supported_gives_high_support(self):
        from rl_evolution.reward import compute_support_strength
        analysis = self._make_analysis("H1", True, 0.85)
        result = compute_support_strength("modify", "H1", [analysis])
        assert result == 0.85

    def test_modify_refuted_gives_low_support(self):
        from rl_evolution.reward import compute_support_strength
        analysis = self._make_analysis("H1", False, 0.7)
        result = compute_support_strength("modify", "H1", [analysis])
        assert result == 0.3

    def test_eliminate_refuted_gives_high_reward(self):
        from rl_evolution.reward import compute_support_strength
        analysis = self._make_analysis("H1", False, 0.8)
        result = compute_support_strength("eliminate", "H1", [analysis])
        assert result == 1.0

    def test_eliminate_supported_gives_low_reward(self):
        from rl_evolution.reward import compute_support_strength
        analysis = self._make_analysis("H1", True, 0.8)
        result = compute_support_strength("eliminate", "H1", [analysis])
        assert result == 0.2

    def test_novelty_gain_positive(self):
        from rl_evolution.reward import compute_novelty_gain
        # 0.15 / 0.3 = 0.5 归一化
        result = compute_novelty_gain(0.5, 0.65)
        assert abs(result - 0.5) < 0.01

    def test_novelty_gain_negative(self):
        from rl_evolution.reward import compute_novelty_gain
        result = compute_novelty_gain(0.7, 0.5)
        assert result == 0.0

    def test_novelty_gain_capped_at_one(self):
        from rl_evolution.reward import compute_novelty_gain
        result = compute_novelty_gain(0.0, 0.5)
        assert result == 1.0  # 0.5/0.3 > 1.0, capped

    def test_convergence_progress_converged(self):
        from rl_evolution.reward import compute_convergence_progress
        evolution = MagicMock()
        evolution.converged = True
        result = compute_convergence_progress(evolution, [])
        assert result == 1.0

    def test_convergence_progress_partial(self):
        from rl_evolution.reward import compute_convergence_progress
        evolution = MagicMock()
        evolution.converged = False
        # 3个假设，2个已解决
        hyps = [MagicMock(status="confirmed"), MagicMock(status="refuted"), MagicMock(status="active")]
        result = compute_convergence_progress(evolution, hyps)
        assert abs(result - 2/3) < 0.01

    def test_compute_reward_full(self):
        from rl_evolution.reward import compute_reward
        analysis = self._make_analysis("H1", True, 0.8)
        evolution = MagicMock()
        evolution.converged = False
        hyps = [MagicMock(status="confirmed"), MagicMock(status="active")]

        reward = compute_reward(
            action="modify",
            hypothesis_id="H1",
            analyses=[analysis],
            novelty_before=0.3,
            novelty_after=0.45,
            evolution=evolution,
            hypotheses=hyps,
            alpha=0.5, beta=0.2, gamma=0.3,
            iteration=2,
        )
        # support = 0.8, novelty = (0.15/0.3)=0.5, convergence = 1/2=0.5
        expected = 0.5 * 0.8 + 0.2 * 0.5 + 0.3 * 0.5
        assert abs(reward.total_reward - expected) < 0.01
        assert reward.action == "modify"
        assert reward.hypothesis_id == "H1"


# ============================================================
# Test Optimizer
# ============================================================

class TestRLEvolutionOptimizer:
    """测试 RL 优化器。"""

    def _make_hypothesis(self, hyp_id, status="active", confidence=0.5):
        hyp = MagicMock()
        hyp.id = hyp_id
        hyp.status = status
        hyp.confidence = confidence
        hyp.generation = 1
        return hyp

    def _make_analysis(self, hyp_id, supports=True, confidence=0.7):
        analysis = MagicMock()
        analysis.hypothesis_id = hyp_id
        analysis.supports_hypothesis = supports
        analysis.confidence = confidence
        analysis.key_findings = ["finding1"]
        return analysis

    def test_optimizer_creation(self):
        from rl_evolution import RLEvolutionOptimizer
        opt = RLEvolutionOptimizer(
            algorithm="thompson_sampling",
            alpha=0.5, beta=0.2, gamma=0.3,
        )
        assert opt.algorithm == "thompson_sampling"
        assert opt.bandit is not None
        assert len(opt.reward_history) == 0

    def test_select_actions_basic(self):
        from rl_evolution import RLEvolutionOptimizer
        opt = RLEvolutionOptimizer(algorithm="thompson_sampling")

        hyps = [
            self._make_hypothesis("H1", "active", 0.6),
            self._make_hypothesis("H2", "active", 0.4),
        ]
        analyses = [
            self._make_analysis("H1", True, 0.7),
            self._make_analysis("H2", False, 0.6),
        ]

        plan = opt.select_actions(hyps, analyses, iteration=2)
        assert len(plan.actions) == 2
        assert plan.actions[0].hypothesis_id == "H1"
        assert plan.actions[1].hypothesis_id == "H2"
        # 动作应该在有效范围内
        from rl_evolution.models import ALL_ACTIONS
        for ha in plan.actions:
            assert ha.action in ALL_ACTIONS

    def test_action_masking_confirmed(self):
        from rl_evolution import RLEvolutionOptimizer
        opt = RLEvolutionOptimizer(algorithm="ucb1")

        hyps = [self._make_hypothesis("H1", "confirmed", 0.9)]
        analyses = [self._make_analysis("H1", True, 0.9)]

        plan = opt.select_actions(hyps, analyses, iteration=3)
        # confirmed 只允许 keep 或 eliminate
        assert plan.actions[0].action in ["keep", "eliminate"]

    def test_action_masking_refuted(self):
        from rl_evolution import RLEvolutionOptimizer
        opt = RLEvolutionOptimizer(algorithm="ucb1")

        hyps = [self._make_hypothesis("H1", "refuted", 0.2)]
        analyses = [self._make_analysis("H1", False, 0.8)]

        plan = opt.select_actions(hyps, analyses, iteration=3)
        # refuted 只允许 eliminate 或 generate_new
        assert plan.actions[0].action in ["eliminate", "generate_new"]

    def test_update_from_outcome(self):
        from rl_evolution import RLEvolutionOptimizer
        from rl_evolution.models import ActionPlan, HypothesisAction
        opt = RLEvolutionOptimizer(algorithm="thompson_sampling")

        plan = ActionPlan(
            actions=[
                HypothesisAction(hypothesis_id="H1", action="modify", score=0.7),
            ],
            reasoning="test",
            iteration=1,
        )
        evolution = MagicMock()
        evolution.converged = False

        analyses = [self._make_analysis("H1", True, 0.8)]
        hyps_after = [self._make_hypothesis("H1", "active", 0.8)]

        rewards = opt.update_from_outcome(
            action_plan=plan,
            evolution=evolution,
            analyses=analyses,
            hypotheses_after=hyps_after,
            novelty_before=0.3,
            novelty_after=0.4,
            iteration=1,
        )
        assert len(rewards) == 1
        assert rewards[0].action == "modify"
        assert rewards[0].total_reward > 0
        assert opt.bandit.get_total_pulls() == 1
        assert len(opt.reward_history) == 1

    def test_get_statistics(self):
        from rl_evolution import RLEvolutionOptimizer
        opt = RLEvolutionOptimizer(algorithm="ucb1")
        stats = opt.get_statistics()
        assert stats["algorithm"] == "ucb1"
        assert stats["total_pulls"] == 0
        assert "arms" in stats

    def test_optimizer_learning_over_iterations(self):
        """验证多轮迭代后 Bandit 偏好会发生变化。"""
        from rl_evolution import RLEvolutionOptimizer
        from rl_evolution.models import ActionPlan, HypothesisAction
        opt = RLEvolutionOptimizer(
            algorithm="thompson_sampling",
            alpha=0.5, beta=0.2, gamma=0.3,
        )

        # 模拟3轮高奖励的 modify 和低奖励的 eliminate
        for i in range(10):
            opt.bandit.update("modify", 0.9)
            opt.bandit.update("eliminate", 0.2)

        stats = opt.get_statistics()
        assert stats["arms"]["modify"]["mean_reward"] > stats["arms"]["eliminate"]["mean_reward"]


# ============================================================
# Test Persistence
# ============================================================

class TestPersistence:
    """测试状态持久化。"""

    def test_save_and_load(self, tmp_path):
        from rl_evolution.bandit import ThompsonSamplingBandit
        from rl_evolution.models import ALL_ACTIONS, RewardSignal
        from rl_evolution.persistence import save_state, load_state

        bandit = ThompsonSamplingBandit(arms=ALL_ACTIONS)
        bandit.update("modify", 0.8)
        bandit.update("eliminate", 0.6)

        history = [
            RewardSignal(
                iteration=1, hypothesis_id="H1", action="modify",
                support_strength=0.8, novelty_gain=0.3,
                convergence_progress=0.5, total_reward=0.6,
            ),
        ]

        path = save_state(bandit, history, "test_session", str(tmp_path))
        assert os.path.exists(path)

        loaded = load_state("test_session", str(tmp_path))
        assert loaded is not None
        assert loaded.algorithm == "thompson_sampling"
        assert loaded.total_pulls == 2
        assert loaded.arm_stats["modify"].pull_count == 1

    def test_load_nonexistent_returns_none(self, tmp_path):
        from rl_evolution.persistence import load_state
        result = load_state("nonexistent_session", str(tmp_path))
        assert result is None


# ============================================================
# Test create_bandit factory
# ============================================================

class TestFactory:
    """测试工厂函数。"""

    def test_create_ucb1(self):
        from rl_evolution.bandit import create_bandit, UCB1Bandit
        bandit = create_bandit("ucb1")
        assert isinstance(bandit, UCB1Bandit)

    def test_create_thompson(self):
        from rl_evolution.bandit import create_bandit, ThompsonSamplingBandit
        bandit = create_bandit("thompson_sampling")
        assert isinstance(bandit, ThompsonSamplingBandit)

    def test_create_default(self):
        from rl_evolution.bandit import create_bandit, ThompsonSamplingBandit
        bandit = create_bandit()
        assert isinstance(bandit, ThompsonSamplingBandit)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
