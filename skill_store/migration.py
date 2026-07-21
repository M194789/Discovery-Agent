"""
数据迁移工具 — 将旧 JSON 记忆中的技能迁移到向量数据库

扫描 experiment_system_memory.json 中的技能条目，
解析为 Skill 模型，生成嵌入并写入 ChromaDB。
"""

from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from .models import Skill, SkillProvenance, SkillPerformanceMetrics
from . import embeddings
from .vector_db import create_skill_db
from .retrieval import save_skill_data

# 旧记忆文件路径
_MEMORY_DIR = Path(__file__).resolve().parent.parent / "saved_code" / "memory"
_MIGRATION_FLAG_FILE = Path(__file__).resolve().parent.parent / "saved_code" / ".skill_migration_done"


def is_migration_needed() -> bool:
    """检查是否需要执行迁移"""
    return not _MIGRATION_FLAG_FILE.exists()


def _mark_migration_done():
    """标记迁移已完成"""
    _MIGRATION_FLAG_FILE.parent.mkdir(parents=True, exist_ok=True)
    _MIGRATION_FLAG_FILE.write_text(datetime.now().isoformat(), encoding="utf-8")


def _load_legacy_memories(user_id: str) -> list:
    """加载旧的记忆条目"""
    memory_file = _MEMORY_DIR / f"{user_id}_memory.json"
    if not memory_file.exists():
        return []
    try:
        with open(memory_file, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return []


def _is_skill_entry(content: str) -> bool:
    """判断记忆条目是否为技能"""
    skill_indicators = [
        "可复用的实验技能",
        "experiment_skill",
        "实验计划:",
        "执行结果:",
        "工具：",
        "步骤：",
    ]
    return any(indicator in content for indicator in skill_indicators)


def _parse_skill_from_memory(memory_entry: dict, index: int) -> Optional[Skill]:
    """
    尝试从旧格式记忆条目解析为 Skill 模型。

    旧格式示例:
    "可复用的实验技能\n\n实验计划:\n方法：data_analysis\n工具：numpy, scipy...\n步骤：...\n\n执行结果:\n成功标准：...\n代码:\n..."
    """
    content = memory_entry.get("content", "")
    timestamp = memory_entry.get("timestamp", datetime.now().isoformat())

    if not _is_skill_entry(content):
        return None

    # 解析方法
    method = ""
    method_match = re.search(r"方法[：:]\s*(\S+)", content)
    if method_match:
        method = method_match.group(1).strip()

    # 解析工具链
    tool_chain = []
    tools_match = re.search(r"工具[：:]\s*([^\n]+)", content)
    if tools_match:
        tool_chain = [t.strip() for t in tools_match.group(1).split(",") if t.strip()]

    # 解析步骤
    steps = []
    steps_match = re.search(r"步骤[：:]\s*([^\n]+)", content)
    if steps_match:
        steps = [s.strip() for s in steps_match.group(1).split(";") if s.strip()]

    # 解析代码规范
    code_spec = ""
    spec_match = re.search(r"规范说明[：:]\s*([^\n]+)", content)
    if spec_match:
        code_spec = spec_match.group(1).strip()

    # 解析成功标准
    success_criteria = ""
    criteria_match = re.search(r"成功标准[：:]\s*([^\n]+)", content)
    if criteria_match:
        success_criteria = criteria_match.group(1).strip()

    # 解析代码
    final_code = ""
    code_match = re.search(r"代码[：:]?\s*\n(.+?)(?:\n输出|$)", content, re.DOTALL)
    if code_match:
        final_code = code_match.group(1).strip()

    # 推断领域（从标签或内容关键词）
    domain = _infer_domain(content, memory_entry.get("tags", []))

    # 生成名称
    name = f"{method}_{domain}" if method and domain else f"skill_legacy_{index}"

    skill = Skill(
        version=1,
        name=name,
        domain=domain,
        method=method,
        tool_chain=tool_chain,
        steps=steps,
        code_specification=code_spec,
        final_code=final_code,
        success_criteria=success_criteria,
        performance_metrics=SkillPerformanceMetrics(
            confidence=0.5,  # 旧技能给予中等置信度
            success_rate=0.5,
            total_uses=1,
            successful_uses=1,
        ),
        provenance=SkillProvenance(
            session_id="legacy_migration",
            iteration=0,
            user_id="experiment_system",
            timestamp=timestamp,
        ),
        created_at=timestamp,
        updated_at=datetime.now().isoformat(),
    )

    return skill


def _infer_domain(content: str, tags: list) -> str:
    """从内容和标签推断领域"""
    domain_keywords = {
        "catalysis": ["催化", "catalyst", "catalysis", "CO2", "反应"],
        "battery": ["电池", "battery", "锂", "充放电", "电极"],
        "electronics": ["通信", "wireless", "信号", "MIMO", "波束"],
        "materials": ["材料", "合金", "晶体", "DFT", "分子动力学"],
    }

    content_lower = content.lower()
    tags_text = " ".join(tags).lower()
    combined = content_lower + " " + tags_text

    best_domain = ""
    best_score = 0

    for domain, keywords in domain_keywords.items():
        score = sum(1 for kw in keywords if kw.lower() in combined)
        if score > best_score:
            best_score = score
            best_domain = domain

    return best_domain if best_domain else "general"


def migrate_existing_skills(user_id: str = "experiment_system") -> int:
    """
    将旧记忆中的技能迁移到新向量数据库。

    参数:
        user_id: 用户 ID

    返回:
        成功迁移的技能数量
    """
    if not is_migration_needed():
        return 0

    print("  Skill store: 开始迁移旧技能数据...")

    memories = _load_legacy_memories(user_id)
    if not memories:
        print("  Skill store: 无旧数据需要迁移")
        _mark_migration_done()
        return 0

    # 筛选技能条目
    skill_entries = [
        (i, m) for i, m in enumerate(memories)
        if _is_skill_entry(m.get("content", ""))
    ]

    if not skill_entries:
        print("  Skill store: 未发现技能条目")
        _mark_migration_done()
        return 0

    print(f"  Skill store: 发现 {len(skill_entries)} 条技能记录，开始解析...")

    # 解析技能
    skills: List[Skill] = []
    for idx, (i, entry) in enumerate(skill_entries):
        skill = _parse_skill_from_memory(entry, idx)
        if skill:
            skills.append(skill)

    if not skills:
        print("  Skill store: 无可解析的技能")
        _mark_migration_done()
        return 0

    # 批量生成嵌入
    texts = [s.to_embedding_text() for s in skills]
    embedding_results = embeddings.embed_batch(texts)

    for skill, emb in zip(skills, embedding_results):
        skill.embedding = emb

    # 写入向量数据库和数据文件
    db = create_skill_db(user_id)
    success_count = 0

    for skill in skills:
        # 保存完整数据到 JSON
        save_skill_data(skill, user_id)

        # 写入向量数据库
        if hasattr(db, 'add_skill'):
            if db.add_skill(skill):
                success_count += 1
            else:
                success_count += 1  # JSON 保存成功也算

    _mark_migration_done()
    print(f"  Skill store: 迁移完成，成功 {success_count}/{len(skills)} 条")
    return success_count
