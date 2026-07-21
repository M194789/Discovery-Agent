"""
元学习模块 — 跨领域技能迁移

实现:
A. 技能抽象：通过 LLM 提取域无关的计算模式
B. 域相似度矩阵：追踪领域间的迁移亲和度
C. 技能适配引擎：将技能从一个领域适配到另一个领域
D. 迁移效果追踪：记录和学习迁移成功率
"""

from __future__ import annotations

import asyncio
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .models import Skill, SkillProvenance, TransferRecord

# 域相似度矩阵存储路径
_SIMILARITY_FILE = Path(__file__).resolve().parent.parent / "saved_code" / "domain_similarity.json"


# ---------------------------------------------------------------------------
# A. 技能抽象 — 生成域无关的 abstract_pattern
# ---------------------------------------------------------------------------

_ABSTRACT_PATTERN_PROMPT = """你是一个科学工作流专家。给定以下实验技能描述，请提取出域无关的计算模式模板。

要求:
1. 将领域特定术语替换为通用占位符（如 [TARGET_SYSTEM], [ANALYSIS_METHOD], [DATA_SOURCE]）
2. 保留算法结构和逻辑流程
3. 用 2-4 句话描述抽象模式
4. 格式：直接输出抽象模式文本，不要额外解释

技能信息:
- 领域: {domain}
- 方法: {method}
- 步骤: {steps}
- 代码规范: {code_spec}
- 成功标准: {success_criteria}

请输出该技能的域无关抽象模式:"""


async def generate_abstract_pattern(skill: Skill) -> Optional[str]:
    """
    通过 LLM 生成技能的域无关抽象模式。

    参数:
        skill: 技能对象

    返回:
        抽象模式文本，失败返回 None
    """
    try:
        from config import client, MODEL_NAME

        prompt = _ABSTRACT_PATTERN_PROMPT.format(
            domain=skill.domain,
            method=skill.method,
            steps="; ".join(skill.steps[:5]),
            code_spec=skill.code_specification[:500],
            success_criteria=skill.success_criteria[:200],
        )

        response = await client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=300,
            temperature=0.3,
        )

        pattern = response.choices[0].message.content.strip()
        return pattern if pattern else None

    except Exception as e:
        print(f"  Warning: 抽象模式生成失败: {e}")
        return None


def generate_abstract_pattern_sync(skill: Skill) -> Optional[str]:
    """同步版本的抽象模式生成"""
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as pool:
                future = pool.submit(asyncio.run, generate_abstract_pattern(skill))
                return future.result(timeout=30)
        else:
            return loop.run_until_complete(generate_abstract_pattern(skill))
    except RuntimeError:
        return asyncio.run(generate_abstract_pattern(skill))
    except Exception as e:
        print(f"  Warning: 同步抽象模式生成失败: {e}")
        return None


# ---------------------------------------------------------------------------
# B. 域相似度矩阵
# ---------------------------------------------------------------------------

def _get_domain_pair_key(d1: str, d2: str) -> str:
    """生成有序的域对键"""
    return "|".join(sorted([d1.lower().strip(), d2.lower().strip()]))


def load_domain_similarity() -> Dict[str, float]:
    """加载域相似度矩阵"""
    if _SIMILARITY_FILE.exists():
        try:
            with open(_SIMILARITY_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {}


def save_domain_similarity(matrix: Dict[str, float]):
    """保存域相似度矩阵"""
    os.makedirs(str(_SIMILARITY_FILE.parent), exist_ok=True)
    try:
        with open(_SIMILARITY_FILE, "w", encoding="utf-8") as f:
            json.dump(matrix, f, indent=2, ensure_ascii=False)
    except Exception as e:
        print(f"  Warning: 域相似度保存失败: {e}")


def get_domain_similarity(domain_a: str, domain_b: str) -> float:
    """
    获取两个领域之间的相似度。

    参数:
        domain_a: 领域 A
        domain_b: 领域 B

    返回:
        相似度 [0, 1]，未知域对默认 0.3
    """
    if domain_a.lower() == domain_b.lower():
        return 1.0

    matrix = load_domain_similarity()
    key = _get_domain_pair_key(domain_a, domain_b)
    return matrix.get(key, 0.3)


def update_domain_similarity(domain_a: str, domain_b: str, success: bool):
    """
    根据迁移结果更新域对亲和度。

    参数:
        domain_a: 源域
        domain_b: 目标域
        success: 迁移是否成功
    """
    matrix = load_domain_similarity()
    key = _get_domain_pair_key(domain_a, domain_b)

    current = matrix.get(key, 0.3)
    if success:
        current = min(1.0, current + 0.1)
    else:
        current = max(0.0, current - 0.05)

    matrix[key] = round(current, 3)
    save_domain_similarity(matrix)


async def initialize_domain_similarity(domains: List[str]):
    """
    使用嵌入初始化域对相似度（仅对新域对执行）。

    参数:
        domains: 要初始化的领域列表
    """
    from . import embeddings

    matrix = load_domain_similarity()
    new_pairs = []

    for i, d1 in enumerate(domains):
        for d2 in domains[i + 1:]:
            key = _get_domain_pair_key(d1, d2)
            if key not in matrix:
                new_pairs.append((d1, d2, key))

    if not new_pairs:
        return

    # 批量生成域名嵌入
    all_domains = list(set(d for pair in new_pairs for d in (pair[0], pair[1])))
    domain_embeddings = await embeddings.async_embed_batch(all_domains)

    domain_vec_map = {}
    for domain_name, vec in zip(all_domains, domain_embeddings):
        if vec is not None:
            domain_vec_map[domain_name] = vec

    # 计算余弦相似度
    for d1, d2, key in new_pairs:
        if d1 in domain_vec_map and d2 in domain_vec_map:
            sim = _cosine_similarity(domain_vec_map[d1], domain_vec_map[d2])
            matrix[key] = round(sim, 3)
        else:
            matrix[key] = 0.3  # 默认值

    save_domain_similarity(matrix)


def _cosine_similarity(vec_a: List[float], vec_b: List[float]) -> float:
    """计算两个向量的余弦相似度"""
    dot_product = sum(a * b for a, b in zip(vec_a, vec_b))
    norm_a = sum(a * a for a in vec_a) ** 0.5
    norm_b = sum(b * b for b in vec_b) ** 0.5
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot_product / (norm_a * norm_b)


# ---------------------------------------------------------------------------
# C. 技能适配引擎
# ---------------------------------------------------------------------------

_ADAPTATION_PROMPT = """你是一个科学实验工作流适配专家。将以下实验技能从源领域适配到目标领域。

源技能信息:
- 源领域: {source_domain}
- 方法: {source_method}
- 步骤: {source_steps}
- 抽象模式: {abstract_pattern}
- 工具链: {source_tools}
- 代码规范: {source_code_spec}

适配目标:
- 目标领域: {target_domain}
- 目标问题: {target_problem}

要求:
1. 保留原始工作流的核心逻辑结构
2. 替换领域特定的方法、工具和数据源为目标领域的对应物
3. 调整参数和评估标准以适应目标领域

请以 JSON 格式输出适配后的技能:
{{
    "method": "适配后的方法名称",
    "tool_chain": ["适配后的工具列表"],
    "steps": ["适配后的步骤列表"],
    "code_specification": "适配后的代码规范（200字以内）",
    "success_criteria": "适配后的成功标准",
    "preconditions": ["输入要求"],
    "postconditions": ["输出描述"]
}}"""


async def adapt_skill_to_domain(
    source_skill: Skill,
    target_domain: str,
    target_problem: str,
) -> Optional[Skill]:
    """
    将技能从源领域适配到目标领域。

    参数:
        source_skill: 源技能
        target_domain: 目标领域
        target_problem: 目标问题描述

    返回:
        适配后的新技能，失败返回 None
    """
    try:
        from config import client, MODEL_NAME

        prompt = _ADAPTATION_PROMPT.format(
            source_domain=source_skill.domain,
            source_method=source_skill.method,
            source_steps="; ".join(source_skill.steps[:5]),
            abstract_pattern=source_skill.abstract_pattern or "无",
            source_tools=", ".join(source_skill.tool_chain),
            source_code_spec=source_skill.code_specification[:500],
            target_domain=target_domain,
            target_problem=target_problem,
        )

        response = await client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=800,
            temperature=0.4,
        )

        content = response.choices[0].message.content.strip()

        # 解析 JSON 响应
        # 处理可能的 markdown 代码块包裹
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0].strip()
        elif "```" in content:
            content = content.split("```")[1].split("```")[0].strip()

        adapted_data = json.loads(content)

        # 创建适配后的新技能
        adapted_skill = Skill(
            name=f"{adapted_data.get('method', source_skill.method)}_{target_domain[:10]}",
            domain=target_domain,
            domain_tags=source_skill.domain_tags + [source_skill.domain],
            method=adapted_data.get("method", source_skill.method),
            tool_chain=adapted_data.get("tool_chain", source_skill.tool_chain),
            steps=adapted_data.get("steps", source_skill.steps),
            code_specification=adapted_data.get("code_specification", ""),
            success_criteria=adapted_data.get("success_criteria", ""),
            preconditions=adapted_data.get("preconditions", []),
            postconditions=adapted_data.get("postconditions", []),
            abstract_pattern=source_skill.abstract_pattern,
            provenance=SkillProvenance(
                session_id=f"transfer_from_{source_skill.id}",
                iteration=0,
                hypothesis_id="",
                user_id=source_skill.provenance.user_id,
            ),
        )

        return adapted_skill

    except Exception as e:
        print(f"  Warning: 技能适配失败: {e}")
        return None


def should_attempt_transfer(
    source_skill: Skill,
    target_domain: str,
) -> bool:
    """
    判断是否应该尝试跨域迁移。

    条件:
    1. 域相似度 > 0.2
    2. 源技能成功率 > 0.3
    3. 源技能有 abstract_pattern（或成功率足够高）

    参数:
        source_skill: 源技能
        target_domain: 目标领域

    返回:
        是否应该尝试迁移
    """
    # 同域不需要迁移
    if source_skill.domain.lower() == target_domain.lower():
        return False

    # 检查域相似度
    similarity = get_domain_similarity(source_skill.domain, target_domain)
    if similarity < 0.2:
        return False

    # 检查源技能质量
    if source_skill.performance_metrics.success_rate < 0.3:
        return False

    # 有抽象模式或成功率高于 0.7
    if source_skill.abstract_pattern or source_skill.performance_metrics.success_rate > 0.7:
        return True

    return False


# ---------------------------------------------------------------------------
# D. 迁移效果追踪
# ---------------------------------------------------------------------------

def record_transfer_outcome(
    source_skill: Skill,
    target_domain: str,
    success: bool,
    performance_delta: float = 0.0,
) -> TransferRecord:
    """
    记录迁移结果。

    参数:
        source_skill: 源技能
        target_domain: 目标领域
        success: 是否成功
        performance_delta: 性能变化

    返回:
        迁移记录
    """
    record = TransferRecord(
        source_domain=source_skill.domain,
        target_domain=target_domain,
        adaptation_method="llm_guided",
        success=success,
        performance_delta=performance_delta,
    )

    # 更新域相似度矩阵
    update_domain_similarity(source_skill.domain, target_domain, success)

    return record


def get_transfer_statistics(user_id: str) -> Dict:
    """
    获取迁移统计信息。

    参数:
        user_id: 用户 ID

    返回:
        统计字典
    """
    from .retrieval import load_all_skills

    skills = load_all_skills(user_id)

    total_transfers = 0
    successful_transfers = 0
    domain_pairs: Dict[str, Dict] = {}

    for skill in skills:
        for record in skill.transfer_history:
            total_transfers += 1
            if record.success:
                successful_transfers += 1

            pair_key = _get_domain_pair_key(record.source_domain, record.target_domain)
            if pair_key not in domain_pairs:
                domain_pairs[pair_key] = {"total": 0, "success": 0}
            domain_pairs[pair_key]["total"] += 1
            if record.success:
                domain_pairs[pair_key]["success"] += 1

    return {
        "total_transfers": total_transfers,
        "successful_transfers": successful_transfers,
        "overall_success_rate": (
            successful_transfers / total_transfers if total_transfers > 0 else 0.0
        ),
        "domain_pairs": {
            k: {**v, "rate": v["success"] / v["total"] if v["total"] > 0 else 0}
            for k, v in domain_pairs.items()
        },
    }
