"""
语义检索 — 混合搜索引擎

结合向量相似度搜索和元数据过滤，支持跨域技能发现。
实现 hybrid search：vector similarity + metadata filtering + cross-domain expansion。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

from .models import Skill, SkillQuery, SkillSearchResult
from . import embeddings
from .vector_db import SkillVectorDB, FallbackSkillStore, create_skill_db


# 技能完整数据的本地缓存文件
_SKILL_DATA_DIR = Path(__file__).resolve().parent.parent / "saved_code" / "skill_data"


def _get_skill_data_path(user_id: str) -> Path:
    """获取技能完整数据存储路径"""
    return _SKILL_DATA_DIR / f"{user_id}_skills.json"


def save_skill_data(skill: Skill, user_id: str) -> bool:
    """
    将完整技能数据保存到 JSON（ChromaDB 只存嵌入和元数据，完整数据存 JSON）。

    参数:
        skill: 技能对象
        user_id: 用户 ID

    返回:
        是否成功
    """
    _SKILL_DATA_DIR.mkdir(parents=True, exist_ok=True)
    data_path = _get_skill_data_path(user_id)

    try:
        skills_data = {}
        if data_path.exists():
            with open(data_path, "r", encoding="utf-8") as f:
                skills_data = json.load(f)

        # 存储时排除 embedding（太大），embedding 已在 ChromaDB 中
        skill_dict = skill.model_dump()
        skill_dict.pop("embedding", None)
        skills_data[skill.id] = skill_dict

        with open(data_path, "w", encoding="utf-8") as f:
            json.dump(skills_data, f, indent=2, ensure_ascii=False)
        return True
    except Exception as e:
        print(f"  Warning: 技能数据保存失败: {e}")
        return False


def load_skill_data(skill_id: str, user_id: str) -> Optional[Skill]:
    """
    从 JSON 加载完整技能数据。

    参数:
        skill_id: 技能 ID
        user_id: 用户 ID

    返回:
        Skill 对象或 None
    """
    data_path = _get_skill_data_path(user_id)
    if not data_path.exists():
        return None

    try:
        with open(data_path, "r", encoding="utf-8") as f:
            skills_data = json.load(f)

        if skill_id in skills_data:
            return Skill(**skills_data[skill_id])
        return None
    except Exception:
        return None


def load_all_skills(user_id: str) -> List[Skill]:
    """加载用户的所有技能"""
    data_path = _get_skill_data_path(user_id)
    if not data_path.exists():
        return []

    try:
        with open(data_path, "r", encoding="utf-8") as f:
            skills_data = json.load(f)
        return [Skill(**v) for v in skills_data.values()]
    except Exception:
        return []


def update_skill_data(skill_id: str, user_id: str, updates: dict) -> bool:
    """更新技能数据的特定字段"""
    data_path = _get_skill_data_path(user_id)
    if not data_path.exists():
        return False

    try:
        with open(data_path, "r", encoding="utf-8") as f:
            skills_data = json.load(f)

        if skill_id not in skills_data:
            return False

        skills_data[skill_id].update(updates)
        with open(data_path, "w", encoding="utf-8") as f:
            json.dump(skills_data, f, indent=2, ensure_ascii=False)
        return True
    except Exception:
        return False


def _load_domain_affinity() -> Dict[str, float]:
    """加载域对亲和度数据"""
    affinity_path = _SKILL_DATA_DIR.parent / "domain_similarity.json"
    if affinity_path.exists():
        try:
            with open(affinity_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {}


def _get_domain_pair_key(d1: str, d2: str) -> str:
    """生成有序的域对键"""
    return "|".join(sorted([d1.lower(), d2.lower()]))


def hybrid_search(
    query: SkillQuery,
    user_id: str = "experiment_system",
) -> List[SkillSearchResult]:
    """
    混合语义检索。

    算法:
    1. 生成查询嵌入
    2. 向量搜索（带元数据过滤）
    3. 跨域扩展搜索
    4. 域对亲和度加权
    5. 合并去重排序

    参数:
        query: 搜索查询
        user_id: 用户 ID

    返回:
        排序后的搜索结果列表
    """
    db = create_skill_db(user_id)

    # 如果是回退存储，使用简化搜索
    if isinstance(db, FallbackSkillStore):
        return _fallback_search(db, query, user_id)

    # 生成查询嵌入
    query_embedding = embeddings.embed_text(query.query_text)
    if query_embedding is None:
        # embedding 不可用，回退到文本搜索
        return _fallback_search(
            FallbackSkillStore(user_id) if not isinstance(db, FallbackSkillStore) else db,
            query,
            user_id,
        )

    results: Dict[str, SkillSearchResult] = {}

    # 1. 主搜索：带域过滤
    where_filters = _build_where_filters(query)
    primary_results = db.search(
        query_embedding=query_embedding,
        top_k=query.top_k * 2,
        where_filters=where_filters if where_filters else None,
    )

    for skill_id, similarity, metadata in primary_results:
        if similarity < 0.1:
            continue
        skill = load_skill_data(skill_id, user_id)
        if skill and skill.performance_metrics.confidence >= query.min_confidence:
            results[skill_id] = SkillSearchResult(
                skill=skill,
                similarity_score=similarity,
                is_cross_domain=False,
            )

    # 2. 跨域扩展搜索
    if query.include_cross_domain and query.domain_filter:
        cross_results = db.search(
            query_embedding=query_embedding,
            top_k=query.top_k,
            where_filters=None,  # 不过滤域
        )

        domain_affinity = _load_domain_affinity()

        for skill_id, similarity, metadata in cross_results:
            if skill_id in results:
                continue  # 已经在主结果中

            skill_domain = metadata.get("domain", "")
            if skill_domain == query.domain_filter:
                continue  # 同域的跳过（已在主搜索中）

            # 计算跨域惩罚
            pair_key = _get_domain_pair_key(query.domain_filter, skill_domain)
            affinity = domain_affinity.get(pair_key, 0.3)  # 默认亲和度 0.3

            # 跨域分数 = 原始相似度 × 0.8 × 域对亲和度
            cross_score = similarity * 0.8 * (0.5 + 0.5 * affinity)

            if cross_score < 0.1:
                continue

            skill = load_skill_data(skill_id, user_id)
            if skill and skill.performance_metrics.confidence >= query.min_confidence:
                results[skill_id] = SkillSearchResult(
                    skill=skill,
                    similarity_score=cross_score,
                    is_cross_domain=True,
                )

    # 3. 排序并返回 top_k
    sorted_results = sorted(
        results.values(),
        key=lambda r: r.similarity_score,
        reverse=True,
    )
    return sorted_results[: query.top_k]


def _build_where_filters(query: SkillQuery) -> Optional[Dict]:
    """构建 ChromaDB where 过滤条件"""
    conditions = []

    if query.domain_filter:
        conditions.append({"domain": {"$eq": query.domain_filter}})
    if query.method_filter:
        conditions.append({"method": {"$eq": query.method_filter}})
    conditions.append({"status": {"$eq": "active"}})

    if not conditions:
        return None
    if len(conditions) == 1:
        return conditions[0]
    return {"$and": conditions}


def _fallback_search(
    store: FallbackSkillStore,
    query: SkillQuery,
    user_id: str,
) -> List[SkillSearchResult]:
    """回退搜索：使用 TF-IDF 文本匹配"""
    results = store.search_by_text(query.query_text, top_k=query.top_k * 2)

    output = []
    for skill_id, score in results:
        skill = load_skill_data(skill_id, user_id)
        if skill is None:
            # 尝试从回退存储加载
            sdata = store.get_by_id(skill_id)
            if sdata:
                try:
                    skill = Skill(**sdata)
                except Exception:
                    continue
            else:
                continue

        # 应用过滤
        if query.domain_filter and skill.domain != query.domain_filter:
            if not query.include_cross_domain:
                continue
            score *= 0.8  # 跨域惩罚

        if query.method_filter and skill.method != query.method_filter:
            continue

        if skill.performance_metrics.confidence < query.min_confidence:
            continue

        is_cross = (
            query.domain_filter is not None and skill.domain != query.domain_filter
        )
        output.append(SkillSearchResult(
            skill=skill,
            similarity_score=score,
            is_cross_domain=is_cross,
        ))

    output.sort(key=lambda r: r.similarity_score, reverse=True)
    return output[: query.top_k]


def format_search_results(results: List[SkillSearchResult]) -> str:
    """
    将搜索结果格式化为可读字符串（供 LLM 上下文使用）。

    参数:
        results: 搜索结果列表

    返回:
        格式化的字符串
    """
    if not results:
        return "未找到相关技能。"

    parts = []
    for i, r in enumerate(results, 1):
        s = r.skill
        cross_tag = " [跨域]" if r.is_cross_domain else ""
        parts.append(
            f"{i}. [{s.domain}] {s.name}{cross_tag} "
            f"(相似度={r.similarity_score:.2f}, 成功率={s.performance_metrics.success_rate:.0%})\n"
            f"   方法: {s.method} | 工具: {', '.join(s.tool_chain[:5])}\n"
            f"   步骤: {'; '.join(s.steps[:3])}"
        )
        if s.abstract_pattern:
            parts.append(f"   抽象模式: {s.abstract_pattern[:200]}")

    return "\n".join(parts)


def format_rich_skill_context(
    results: List[SkillSearchResult],
    max_rich_skills: int = 2,
    max_code_chars: int = 600,
    max_final_code_chars: int = 400,
) -> str:
    """
    将搜索结果格式化为富文本上下文（供实验规划器 LLM 使用）。

    与 format_search_results() 不同，此函数输出完整的 tool_chain、steps、
    code_specification 和 final_code，以便 LLM 直接复用代码结构。

    参数:
        results: 搜索结果列表
        max_rich_skills: 输出完整信息的技能数量上限（其余用简要格式）
        max_code_chars: code_specification 截断长度
        max_final_code_chars: final_code 截断长度

    返回:
        格式化的富文本字符串
    """
    if not results:
        return "未找到相关技能。"

    parts = []
    for i, r in enumerate(results, 1):
        s = r.skill
        cross_tag = " [跨域]" if r.is_cross_domain else ""

        if i <= max_rich_skills:
            # 富格式：包含完整代码规格和工具链
            section = [
                f"=== Reusable Skill {i}: {s.name}{cross_tag} ===",
                f"Domain: {s.domain} | Method: {s.method} | "
                f"Success Rate: {s.performance_metrics.success_rate:.0%}",
                f"Similarity: {r.similarity_score:.2f}",
                f"Tool Chain: {', '.join(s.tool_chain)}",
            ]
            if s.steps:
                section.append("Steps:")
                for j, step in enumerate(s.steps, 1):
                    section.append(f"  {j}. {step}")
            if s.success_criteria:
                section.append(f"Success Criteria: {s.success_criteria}")
            if s.code_specification:
                code_spec = s.code_specification[:max_code_chars]
                if len(s.code_specification) > max_code_chars:
                    code_spec += "..."
                section.append(f"Code Specification:\n  {code_spec}")
            if s.final_code:
                final = s.final_code[:max_final_code_chars]
                if len(s.final_code) > max_final_code_chars:
                    final += "..."
                section.append(f"Final Code Pattern:\n  {final}")
            if s.abstract_pattern:
                section.append(
                    f"Abstract Pattern: {s.abstract_pattern[:300]}"
                )
            parts.append("\n".join(section))
        else:
            # 简要格式（与原 format_search_results 一致）
            brief = (
                f"{i}. [{s.domain}] {s.name}{cross_tag} "
                f"(相似度={r.similarity_score:.2f}, "
                f"成功率={s.performance_metrics.success_rate:.0%})\n"
                f"   方法: {s.method} | 工具: {', '.join(s.tool_chain[:5])}\n"
                f"   步骤: {'; '.join(s.steps[:3])}"
            )
            parts.append(brief)

    return "\n\n".join(parts)
