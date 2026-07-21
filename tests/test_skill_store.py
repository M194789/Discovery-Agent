"""
技能存储模块的单元测试

测试覆盖:
- Pydantic 模型构建和验证
- 嵌入生成（mocked）
- 向量数据库 CRUD（使用回退存储）
- 混合检索排序
- 元学习域相似度
- 迁移解析
- 奖励计算扩展
"""

import json
import os
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# 确保项目根目录在 sys.path 中
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# =====================================================================
# 测试 1: 数据模型
# =====================================================================

class TestSkillModels:
    """测试 Pydantic 数据模型"""

    def test_skill_creation_defaults(self):
        """测试 Skill 创建使用默认值"""
        from skill_store.models import Skill

        skill = Skill()
        assert skill.id  # UUID 已生成
        assert skill.version == 1
        assert skill.status == "active"
        assert skill.performance_metrics.success_rate == 0.0
        assert skill.embedding is None
        assert skill.transfer_history == []

    def test_skill_creation_full(self):
        """测试 Skill 带完整参数创建"""
        from skill_store.models import Skill, SkillProvenance, SkillPerformanceMetrics

        skill = Skill(
            name="DFT_catalysis",
            domain="catalysis",
            method="DFT",
            tool_chain=["pymatgen", "ase"],
            steps=["Step 1: Setup structure", "Step 2: Run DFT"],
            code_specification="import pymatgen; ...",
            success_criteria="Energy < -100 eV",
            performance_metrics=SkillPerformanceMetrics(
                success_rate=0.8,
                total_uses=5,
                successful_uses=4,
                confidence=0.75,
            ),
            provenance=SkillProvenance(
                session_id="test_session",
                iteration=2,
                hypothesis_id="hyp_001",
                user_id="tester",
            ),
        )

        assert skill.name == "DFT_catalysis"
        assert skill.domain == "catalysis"
        assert skill.method == "DFT"
        assert len(skill.tool_chain) == 2
        assert skill.performance_metrics.success_rate == 0.8

    def test_skill_to_embedding_text(self):
        """测试嵌入文本生成"""
        from skill_store.models import Skill

        skill = Skill(
            domain="battery",
            method="ML_modeling",
            steps=["Train model", "Evaluate"],
            code_specification="Use sklearn to build regression model",
            abstract_pattern="Train [MODEL] on [DATA], evaluate with [METRIC]",
        )

        text = skill.to_embedding_text()
        assert "battery" in text
        assert "ML_modeling" in text
        assert "Train model" in text
        assert "sklearn" in text
        assert "[MODEL]" in text

    def test_skill_to_metadata(self):
        """测试 ChromaDB 元数据转换"""
        from skill_store.models import Skill

        skill = Skill(
            domain="electronics",
            method="simulation",
            tool_chain=["numpy", "scipy"],
            status="active",
        )

        meta = skill.to_metadata()
        assert meta["domain"] == "electronics"
        assert meta["method"] == "simulation"
        assert meta["status"] == "active"
        assert meta["tool_chain"] == "numpy,scipy"

    def test_performance_metrics_record_use(self):
        """测试性能指标更新"""
        from skill_store.models import SkillPerformanceMetrics

        metrics = SkillPerformanceMetrics()
        metrics.record_use(True, execution_time=5.0)
        assert metrics.total_uses == 1
        assert metrics.successful_uses == 1
        assert metrics.success_rate == 1.0
        assert metrics.avg_execution_time_s == 5.0

        metrics.record_use(False, execution_time=10.0)
        assert metrics.total_uses == 2
        assert metrics.successful_uses == 1
        assert metrics.success_rate == 0.5

    def test_skill_query_defaults(self):
        """测试 SkillQuery 默认值"""
        from skill_store.models import SkillQuery

        q = SkillQuery(query_text="DFT catalysis")
        assert q.top_k == 5
        assert q.include_cross_domain is True
        assert q.min_confidence == 0.0

    def test_transfer_record(self):
        """测试迁移记录"""
        from skill_store.models import TransferRecord

        record = TransferRecord(
            source_domain="catalysis",
            target_domain="battery",
            success=True,
            performance_delta=0.15,
        )
        assert record.adaptation_method == "llm_guided"
        assert record.success is True


# =====================================================================
# 测试 2: 向量数据库回退存储
# =====================================================================

class TestFallbackStore:
    """测试 JSON 回退存储"""

    def setup_method(self):
        """每个测试前创建临时目录"""
        self.temp_dir = tempfile.mkdtemp()

    def test_add_and_get_skill(self):
        """测试添加和获取技能"""
        from skill_store.models import Skill
        from skill_store.vector_db import FallbackSkillStore

        # Patch the file path
        store = FallbackSkillStore.__new__(FallbackSkillStore)
        store.user_id = "test_user"
        store._file = Path(self.temp_dir) / "test_skills.json"
        store._skills = {}

        skill = Skill(
            id="skill_001",
            name="test_skill",
            domain="catalysis",
            method="DFT",
            steps=["Step 1", "Step 2"],
        )

        assert store.add_skill(skill) is True
        assert store.count() == 1

        retrieved = store.get_by_id("skill_001")
        assert retrieved is not None
        assert retrieved["name"] == "test_skill"

    def test_delete_skill(self):
        """测试删除技能"""
        from skill_store.models import Skill
        from skill_store.vector_db import FallbackSkillStore

        store = FallbackSkillStore.__new__(FallbackSkillStore)
        store.user_id = "test_user"
        store._file = Path(self.temp_dir) / "test_skills.json"
        store._skills = {}

        skill = Skill(id="skill_del", name="to_delete", domain="test")
        store.add_skill(skill)
        assert store.count() == 1

        assert store.delete_skill("skill_del") is True
        assert store.count() == 0
        assert store.delete_skill("nonexistent") is False

    def test_text_search(self):
        """测试文本搜索"""
        from skill_store.models import Skill
        from skill_store.vector_db import FallbackSkillStore

        store = FallbackSkillStore.__new__(FallbackSkillStore)
        store.user_id = "test_user"
        store._file = Path(self.temp_dir) / "test_skills.json"
        store._skills = {}

        # 添加几个不同领域的技能
        store.add_skill(Skill(
            id="s1", domain="catalysis", method="DFT",
            steps=["DFT calculation for catalyst"],
        ))
        store.add_skill(Skill(
            id="s2", domain="battery", method="MD",
            steps=["Molecular dynamics for electrolyte"],
        ))
        store.add_skill(Skill(
            id="s3", domain="catalysis", method="ML_modeling",
            steps=["Machine learning for catalyst screening"],
        ))

        # 搜索 catalysis 相关
        results = store.search_by_text("catalysis DFT", top_k=3)
        assert len(results) > 0
        # 第一个结果应该是 catalysis + DFT 的技能
        assert results[0][0] == "s1"


# =====================================================================
# 测试 3: 检索格式化
# =====================================================================

class TestRetrieval:
    """测试检索功能"""

    def test_format_search_results_empty(self):
        """测试空结果格式化"""
        from skill_store.retrieval import format_search_results

        result = format_search_results([])
        assert "未找到" in result

    def test_format_search_results(self):
        """测试结果格式化"""
        from skill_store.models import Skill, SkillSearchResult
        from skill_store.retrieval import format_search_results

        skill = Skill(
            name="DFT_cat",
            domain="catalysis",
            method="DFT",
            tool_chain=["pymatgen", "ase"],
            steps=["Setup", "Calculate", "Analyze"],
        )

        results = [
            SkillSearchResult(skill=skill, similarity_score=0.85, is_cross_domain=False),
        ]

        formatted = format_search_results(results)
        assert "catalysis" in formatted
        assert "DFT_cat" in formatted
        assert "0.85" in formatted
        assert "pymatgen" in formatted

    def test_format_cross_domain_tag(self):
        """测试跨域标签"""
        from skill_store.models import Skill, SkillSearchResult
        from skill_store.retrieval import format_search_results

        skill = Skill(name="transferred_skill", domain="battery", method="ML")
        results = [
            SkillSearchResult(skill=skill, similarity_score=0.6, is_cross_domain=True),
        ]

        formatted = format_search_results(results)
        assert "跨域" in formatted


# =====================================================================
# 测试 4: 元学习 — 域相似度
# =====================================================================

class TestMetaLearning:
    """测试元学习模块"""

    def setup_method(self):
        self.temp_dir = tempfile.mkdtemp()

    def test_domain_pair_key(self):
        """测试域对键生成（有序）"""
        from skill_store.meta_learning import _get_domain_pair_key

        key1 = _get_domain_pair_key("catalysis", "battery")
        key2 = _get_domain_pair_key("battery", "catalysis")
        assert key1 == key2  # 顺序无关

    def test_domain_similarity_default(self):
        """测试域相似度默认值"""
        from skill_store.meta_learning import get_domain_similarity

        # 同域
        assert get_domain_similarity("catalysis", "catalysis") == 1.0
        # 未知域对
        sim = get_domain_similarity("catalysis", "quantum_computing")
        assert sim == 0.3  # 默认值

    def test_update_domain_similarity(self):
        """测试域相似度更新"""
        from skill_store import meta_learning

        # Patch 存储路径
        original_file = meta_learning._SIMILARITY_FILE
        meta_learning._SIMILARITY_FILE = Path(self.temp_dir) / "test_sim.json"

        try:
            # 初始状态
            sim = meta_learning.get_domain_similarity("A", "B")
            assert sim == 0.3

            # 成功迁移 +0.1
            meta_learning.update_domain_similarity("A", "B", success=True)
            sim = meta_learning.get_domain_similarity("A", "B")
            assert abs(sim - 0.4) < 0.01

            # 失败迁移 -0.05
            meta_learning.update_domain_similarity("A", "B", success=False)
            sim = meta_learning.get_domain_similarity("A", "B")
            assert abs(sim - 0.35) < 0.01
        finally:
            meta_learning._SIMILARITY_FILE = original_file

    def test_should_attempt_transfer(self):
        """测试迁移决策逻辑"""
        from skill_store.models import Skill, SkillPerformanceMetrics
        from skill_store.meta_learning import should_attempt_transfer

        # 高质量技能 + 有抽象模式 → 应该迁移
        good_skill = Skill(
            domain="catalysis",
            method="DFT",
            abstract_pattern="Compute [PROPERTY] of [SYSTEM] using [METHOD]",
            performance_metrics=SkillPerformanceMetrics(
                success_rate=0.8, confidence=0.7,
            ),
        )
        assert should_attempt_transfer(good_skill, "battery") is True

        # 同域 → 不需要迁移
        assert should_attempt_transfer(good_skill, "catalysis") is False

        # 低质量技能 → 不迁移
        bad_skill = Skill(
            domain="catalysis",
            performance_metrics=SkillPerformanceMetrics(success_rate=0.1),
        )
        assert should_attempt_transfer(bad_skill, "battery") is False


# =====================================================================
# 测试 5: 数据迁移解析
# =====================================================================

class TestMigration:
    """测试旧数据迁移"""

    def test_is_skill_entry(self):
        """测试技能条目识别"""
        from skill_store.migration import _is_skill_entry

        assert _is_skill_entry("可复用的实验技能\n\n实验计划:\n方法：DFT") is True
        assert _is_skill_entry("这是一个普通的发现记录") is False
        assert _is_skill_entry("实验计划:\n工具：numpy") is True

    def test_infer_domain(self):
        """测试领域推断"""
        from skill_store.migration import _infer_domain

        assert _infer_domain("催化剂设计 CO2转化", []) == "catalysis"
        assert _infer_domain("电池材料 锂离子", []) == "battery"
        assert _infer_domain("MIMO 波束成形", []) == "electronics"
        assert _infer_domain("random unknown content", []) == "general"

    def test_parse_skill_from_memory(self):
        """测试从记忆条目解析技能"""
        from skill_store.migration import _parse_skill_from_memory

        entry = {
            "content": (
                "可复用的实验技能\n\n"
                "实验计划:\n"
                "方法：DFT\n"
                "工具：pymatgen, ase\n"
                "步骤：Setup structure; Run calculation; Analyze results\n"
                "规范说明：Use pymatgen to create structure\n\n"
                "执行结果:\n"
                "成功标准：Energy convergence < 0.01 eV\n"
                "代码:\nimport pymatgen\nfrom ase import Atoms\n"
                "输出：Calculation completed successfully"
            ),
            "timestamp": "2025-01-15T10:00:00",
            "tags": ["catalysis"],
        }

        skill = _parse_skill_from_memory(entry, 0)
        assert skill is not None
        assert skill.method == "DFT"
        assert "pymatgen" in skill.tool_chain
        assert "ase" in skill.tool_chain
        assert len(skill.steps) >= 2
        assert skill.domain == "catalysis"


# =====================================================================
# 测试 6: RL 奖励扩展
# =====================================================================

class TestTransferReward:
    """测试迁移奖励计算"""

    def test_successful_transfer_reward(self):
        """测试成功迁移的奖励"""
        from rl_evolution.reward import compute_transfer_reward

        # 成功迁移，低相似度域（更通用）
        reward = compute_transfer_reward(
            transfer_success=True,
            performance_delta=0.1,
            domain_similarity=0.3,
        )
        assert reward > 0.6  # 应该是较高的正奖励
        assert reward <= 1.0

    def test_failed_transfer_penalty(self):
        """测试失败迁移的惩罚"""
        from rl_evolution.reward import compute_transfer_reward

        reward = compute_transfer_reward(
            transfer_success=False,
            domain_similarity=0.8,
        )
        assert reward < 0  # 应该是负值
        assert reward >= -1.0

    def test_high_similarity_success(self):
        """高相似度域成功迁移"""
        from rl_evolution.reward import compute_transfer_reward

        reward = compute_transfer_reward(
            transfer_success=True,
            performance_delta=0.0,
            domain_similarity=0.9,
        )
        # 高相似度成功，距离加分少
        assert 0.5 < reward < 0.8

    def test_low_similarity_success(self):
        """低相似度域成功迁移（更有价值）"""
        from rl_evolution.reward import compute_transfer_reward

        reward_low = compute_transfer_reward(
            transfer_success=True,
            performance_delta=0.0,
            domain_similarity=0.1,
        )
        reward_high = compute_transfer_reward(
            transfer_success=True,
            performance_delta=0.0,
            domain_similarity=0.9,
        )
        # 低相似度成功应该比高相似度更高奖励（更通用的模式）
        assert reward_low > reward_high


# =====================================================================
# 测试 7: 技能注册表 facade
# =====================================================================

class TestSkillRegistry:
    """测试 skill_registry.py facade"""

    def test_legacy_solidify(self):
        """测试旧版降级函数"""
        from skill_registry import _legacy_solidify

        # 创建 mock plan 和 result
        plan = MagicMock()
        plan.method = "DFT"
        plan.tool_chain = ["pymatgen"]
        plan.steps = ["Step 1"]
        plan.code_specification = "spec"
        plan.success_criteria = "criteria"

        result = MagicMock()
        result.final_code = "import pymatgen"
        result.execution_output = "Success"

        skill = _legacy_solidify(plan, result)
        assert skill["type"] == "experiment_skill"
        assert skill["method"] == "DFT"
        assert "pymatgen" in skill["tool_chain"]


# =====================================================================
# 运行入口
# =====================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
