"""
技能注册表 — 将成功的实验工作流固化为可复用的技能。

作为 skill_store 包的 facade，提供统一的技能管理 API。
当 skill_store 不可用时，优雅降级到旧的 dict + memory_ops 行为。
"""

from __future__ import annotations

import asyncio
from datetime import datetime
from typing import List, Optional

from models.hypothesis_models import ExperimentPlan
from models.experiment_models import ExperimentResult

# 尝试导入新的 skill_store 系统
_skill_store_available: Optional[bool] = None


def _check_skill_store() -> bool:
    """检查 skill_store 包是否可用"""
    global _skill_store_available
    if _skill_store_available is not None:
        return _skill_store_available
    try:
        from skill_store.models import Skill  # noqa: F401
        from skill_store.vector_db import create_skill_db  # noqa: F401
        _skill_store_available = True
    except ImportError as e:
        print(f"  Skill store 不可用 ({e})，使用旧版 dict 模式")
        _skill_store_available = False
    return _skill_store_available


# ---------------------------------------------------------------------------
# 新版 API（skill_store 可用时）
# ---------------------------------------------------------------------------

def solidify_skill(
    plan: ExperimentPlan,
    result: ExperimentResult,
    session_id: str = "unknown",
    iteration: int = 0,
    domain: str = "",
    user_id: str = "experiment_system",
    hypothesis_id: str = "",
) -> dict:
    """
    将成功的实验工作流打包为可复用的技能。

    参数:
        plan: 成功的实验计划
        result: 成功的实验结果
        session_id: 会话 ID
        iteration: 迭代轮次
        domain: 研究领域
        user_id: 用户 ID
        hypothesis_id: 关联的假设 ID

    返回:
        技能字典（兼容旧接口），或 Skill.model_dump()
    """
    if not _check_skill_store():
        # 降级：旧行为
        return _legacy_solidify(plan, result)

    from skill_store.models import Skill, SkillProvenance, SkillPerformanceMetrics
    from skill_store.meta_learning import generate_abstract_pattern_sync
    from skill_store import embeddings
    from skill_store.vector_db import create_skill_db
    from skill_store.retrieval import save_skill_data
    from skill_store.migration import migrate_existing_skills, is_migration_needed

    # 首次调用时触发数据迁移
    if is_migration_needed():
        try:
            migrate_existing_skills(user_id)
        except Exception as e:
            print(f"  Warning: 数据迁移失败: {e}")

    # 构建技能名称
    name = f"{plan.method}_{domain[:15]}" if domain else f"{plan.method}_skill"

    # 创建 Skill 对象
    skill = Skill(
        name=name,
        domain=domain,
        method=plan.method,
        tool_chain=plan.tool_chain,
        steps=plan.steps,
        code_specification=plan.code_specification,
        final_code=result.final_code or "",
        success_criteria=plan.success_criteria,
        performance_metrics=SkillPerformanceMetrics(
            confidence=0.7,
            success_rate=1.0,
            total_uses=1,
            successful_uses=1,
        ),
        provenance=SkillProvenance(
            session_id=session_id,
            iteration=iteration,
            hypothesis_id=hypothesis_id,
            user_id=user_id,
        ),
    )

    # 生成抽象模式（同步 LLM 调用）
    try:
        abstract = generate_abstract_pattern_sync(skill)
        if abstract:
            skill.abstract_pattern = abstract
    except Exception as e:
        print(f"  Warning: 抽象模式生成跳过: {e}")

    # 生成嵌入
    try:
        embedding = embeddings.embed_text(skill.to_embedding_text())
        if embedding:
            skill.embedding = embedding
    except Exception as e:
        print(f"  Warning: 嵌入生成跳过: {e}")

    # 存入向量数据库
    try:
        db = create_skill_db(user_id)
        db.add_skill(skill)
    except Exception as e:
        print(f"  Warning: 向量数据库写入失败: {e}")

    # 保存完整数据到 JSON
    try:
        save_skill_data(skill, user_id)
    except Exception as e:
        print(f"  Warning: 技能数据保存失败: {e}")

    print(f"  技能已固化: {skill.name} (ID: {skill.id})")
    return skill.model_dump()


async def save_skill(skill_dict: dict, user_id: str) -> str:
    """
    保存技能（兼容旧接口）。

    如果传入的是新版 Skill 的 dict，直接保存。
    否则按旧格式处理。

    参数:
        skill_dict: 技能字典
        user_id: 用户 ID

    返回:
        状态消息
    """
    if not _check_skill_store():
        return await _legacy_save_skill(skill_dict, user_id)

    # 新版已在 solidify_skill 中完成存储
    skill_id = skill_dict.get("id", "unknown")
    return f"技能已保存到向量数据库 (ID: {skill_id})"


async def find_applicable_skills(
    query: str,
    user_id: str,
    domain: Optional[str] = None,
    method: Optional[str] = None,
    top_k: int = 5,
    include_cross_domain: bool = True,
) -> str:
    """
    搜索与问题相关的技能。

    参数:
        query: 搜索查询
        user_id: 用户 ID
        domain: 域过滤（可选）
        method: 方法过滤（可选）
        top_k: 返回数量
        include_cross_domain: 是否包含跨域技能

    返回:
        格式化的匹配技能字符串
    """
    if not _check_skill_store():
        return await _legacy_find_skills(query, user_id)

    from skill_store.models import SkillQuery
    from skill_store.retrieval import hybrid_search, format_search_results

    skill_query = SkillQuery(
        query_text=query,
        domain_filter=domain,
        method_filter=method,
        top_k=top_k,
        include_cross_domain=include_cross_domain,
    )

    results = hybrid_search(skill_query, user_id)
    return format_search_results(results)


async def find_applicable_skills_rich(
    query: str,
    user_id: str,
    domain: Optional[str] = None,
    method: Optional[str] = None,
    top_k: int = 3,
    include_cross_domain: bool = True,
) -> str:
    """
    搜索与问题相关的技能，返回富文本上下文（含代码规格和工具链）。

    专供实验规划器使用，输出完整的 code_specification、final_code、
    steps 等信息以支持代码结构复用。

    参数:
        query: 搜索查询
        user_id: 用户 ID
        domain: 域过滤（可选）
        method: 方法过滤（可选）
        top_k: 返回数量
        include_cross_domain: 是否包含跨域技能

    返回:
        富文本格式的匹配技能字符串
    """
    if not _check_skill_store():
        return await _legacy_find_skills(query, user_id)

    from skill_store.models import SkillQuery
    from skill_store.retrieval import hybrid_search, format_rich_skill_context

    skill_query = SkillQuery(
        query_text=query,
        domain_filter=domain,
        method_filter=method,
        top_k=top_k,
        include_cross_domain=include_cross_domain,
    )

    results = hybrid_search(skill_query, user_id)
    return format_rich_skill_context(results)


async def adapt_skill_for_domain(
    skill_id: str,
    target_domain: str,
    target_problem: str,
    user_id: str = "experiment_system",
) -> Optional[dict]:
    """
    将技能适配到新领域。

    参数:
        skill_id: 源技能 ID
        target_domain: 目标领域
        target_problem: 目标问题
        user_id: 用户 ID

    返回:
        适配后的技能字典，失败返回 None
    """
    if not _check_skill_store():
        return None

    from skill_store.retrieval import load_skill_data, save_skill_data
    from skill_store.meta_learning import adapt_skill_to_domain, should_attempt_transfer
    from skill_store.vector_db import create_skill_db
    from skill_store import embeddings

    source_skill = load_skill_data(skill_id, user_id)
    if not source_skill:
        return None

    if not should_attempt_transfer(source_skill, target_domain):
        return None

    adapted = await adapt_skill_to_domain(source_skill, target_domain, target_problem)
    if not adapted:
        return None

    # 生成嵌入
    embedding = await embeddings.async_embed_text(adapted.to_embedding_text())
    if embedding:
        adapted.embedding = embedding

    # 存储
    db = create_skill_db(user_id)
    db.add_skill(adapted)
    save_skill_data(adapted, user_id)

    return adapted.model_dump()


def update_skill_metrics(
    skill_id: str,
    success: bool,
    execution_time: Optional[float] = None,
    user_id: str = "experiment_system",
) -> bool:
    """
    更新技能的性能指标。

    参数:
        skill_id: 技能 ID
        success: 本次使用是否成功
        execution_time: 执行时间
        user_id: 用户 ID

    返回:
        是否更新成功
    """
    if not _check_skill_store():
        return False

    from skill_store.retrieval import load_skill_data, update_skill_data
    from skill_store.vector_db import create_skill_db

    skill = load_skill_data(skill_id, user_id)
    if not skill:
        return False

    # 更新指标
    skill.performance_metrics.record_use(success, execution_time)
    skill.updated_at = datetime.now().isoformat()

    # 保存更新
    update_skill_data(skill_id, user_id, {
        "performance_metrics": skill.performance_metrics.model_dump(),
        "updated_at": skill.updated_at,
    })

    # 更新向量数据库中的元数据
    db = create_skill_db(user_id)
    if hasattr(db, 'update_metadata'):
        db.update_metadata(skill_id, {
            "confidence": skill.performance_metrics.confidence,
            "success_rate": skill.performance_metrics.success_rate,
        })

    return True


# ---------------------------------------------------------------------------
# 旧版降级函数
# ---------------------------------------------------------------------------

def _legacy_solidify(plan: ExperimentPlan, result: ExperimentResult) -> dict:
    """旧版技能打包（dict 格式）"""
    return {
        "type": "experiment_skill",
        "method": plan.method,
        "tool_chain": plan.tool_chain,
        "steps": plan.steps,
        "code_specification": plan.code_specification,
        "success_criteria": plan.success_criteria,
        "final_code": result.final_code,
        "execution_summary": result.execution_output[:500],
    }


async def _legacy_save_skill(skill: dict, user_id: str) -> str:
    """旧版技能保存"""
    import memory_ops

    plan_summary = (
        f"方法：{skill['method']}\n"
        f"工具：{', '.join(skill['tool_chain'])}\n"
        f"步骤：{'; '.join(skill['steps'])}\n"
        f"规范说明：{skill['code_specification'][:300]}"
    )
    result_summary = (
        f"成功标准：{skill['success_criteria']}\n"
        f"代码:\n{skill['final_code'][:500]}\n"
        f"输出：{skill['execution_summary']}"
    )
    return await memory_ops.save_experiment_skill(plan_summary, result_summary, user_id)


async def _legacy_find_skills(query: str, user_id: str) -> str:
    """旧版技能搜索"""
    import memory_ops
    return await memory_ops.load_relevant_skills(query, user_id)
