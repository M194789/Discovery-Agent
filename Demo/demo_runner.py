"""
统一的 Demo 运行器 — 异步生成器，逐阶段 yield 快照。

根据 domain 路由到对应的 mock 数据模块，
通过 log_callback 输出日志（替代 print），
通过 yield 返回结构化快照供 Gradio 消费。
"""

from __future__ import annotations

import asyncio
import json
import os
from datetime import datetime
from pathlib import Path
from typing import AsyncGenerator, Callable, Dict, List, Optional


# ---------------------------------------------------------------------------
# 领域路由
# ---------------------------------------------------------------------------

def _resolve_domain_module(domain: str):
    """根据 domain 字符串选择 mock 数据模块。"""
    d = domain.lower()
    if any(k in d for k in ("催化", "catalysis", "co2")):
        from Demo.demo_data import catalysis as mod
        return mod, "catalysis"
    if any(k in d for k in ("电池", "battery", "锂", "electrolyte")):
        from Demo.demo_data import battery as mod
        return mod, "battery"
    if any(k in d for k in ("电子", "通信", "electronic", "communication", "wireless")):
        from Demo.demo_data import electronics as mod
        return mod, "electronics"
    # 默认回退到催化
    from Demo.demo_data import catalysis as mod
    return mod, "custom"


# ---------------------------------------------------------------------------
# 专家名称映射
# ---------------------------------------------------------------------------

_EXPERT_NAMES = {
    "catalysis": ["计算化学专家", "材料物理专家", "机器学习方法专家"],
    "battery": ["计算化学专家", "材料物理专家", "机器学习方法专家"],
    "electronics": ["通信理论专家", "深度学习专家", "系统工程专家"],
    "custom": ["领域专家 A", "领域专家 B", "数据驱动专家"],
}

_DOMAIN_LABELS = {
    "catalysis": "催化科学",
    "battery": "电池材料",
    "electronics": "电子信息与通信工程",
    "custom": None,  # 使用用户输入
}


# ---------------------------------------------------------------------------
# 核心异步生成器
# ---------------------------------------------------------------------------

async def run_demo(
    domain: str,
    problem: str,
    max_iterations: int,
    log_callback: Callable[[str], None],
) -> AsyncGenerator[Dict, None]:
    """运行 Demo 发现流程，逐阶段 yield 快照字典。

    快照字典结构:
        phase          — "init" / "phase_a" / ... / "phase_e" / "complete"
        iteration      — 当前迭代号
        max_iterations — 总迭代数
        hypotheses     — 当前迭代的假设列表
        plans          — 当前迭代的实验方案列表
        results        — 当前迭代的实验结果列表
        analyses       — 当前迭代的分析结果列表
        evolution      — 当前迭代的进化结果字典（或 None）
        novelty_score  — 当前新颖性评分
        session_data   — 最终完整会话数据（仅 phase=="complete" 时有值）
    """
    mod, domain_key = _resolve_domain_module(domain)
    experts = _EXPERT_NAMES.get(domain_key, _EXPERT_NAMES["custom"])
    display_domain = _DOMAIN_LABELS.get(domain_key) or domain

    session_id = f"demo_{domain_key}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    def log(msg: str):
        log_callback(msg)

    # ------------------------------------------------------------------
    # 初始化
    # ------------------------------------------------------------------
    log("=" * 70)
    log(f"自主科学发现系统 - 演示模式")
    log("=" * 70)
    log(f"研究领域：{display_domain}")
    log(f"研究问题：{problem}")
    log(f"最大迭代次数：{max_iterations}")
    log(f"会话 ID：{session_id}")
    log(f"时间：{datetime.now().isoformat()}")
    log("=" * 70)

    log("")
    log("[INIT] 初始化阶段")
    log("  搜索记忆中的相关发现...")
    log("  加载相关技能...")
    log(f"  生成研究问题：{problem[:60]}...")
    await asyncio.sleep(0.3)

    all_iterations_data: List[Dict] = []
    novelty_score = 0.70
    current_hypotheses: List[Dict] = []
    current_plans: List[Dict] = []
    current_results: List[Dict] = []
    current_analyses: List[Dict] = []
    current_evolution: Optional[Dict] = None

    yield {
        "phase": "init",
        "iteration": 0,
        "max_iterations": max_iterations,
        "hypotheses": [],
        "plans": [],
        "results": [],
        "analyses": [],
        "evolution": None,
        "novelty_score": novelty_score,
        "session_data": None,
    }

    # ------------------------------------------------------------------
    # 迭代循环
    # ------------------------------------------------------------------
    converged = False
    for iteration in range(1, max_iterations + 1):
        log("")
        log("=" * 70)
        log(f"迭代 {iteration}/{max_iterations}")
        log("=" * 70)

        # ---- Phase A: 假设生成 ----
        log("")
        log(f"[PHASE A] 生成假设...")
        log(f"  {len(experts)} 个领域专家智能体正在生成假设...")
        for name in experts:
            log(f"     - {name}")

        current_hypotheses = mod.generate_hypotheses(iteration)
        log(f"  生成的假设:")
        for hyp in current_hypotheses:
            log(f"     [{hyp['id']}] {hyp['statement']}")
            log(f"         置信度：{hyp['confidence']:.2f}")

        log(f"  开始辩论优化...")
        log(f"     原始假设：{len(current_hypotheses)} 个")
        log(f"     保留：{len(current_hypotheses)} 个")
        await asyncio.sleep(0.3)

        yield {
            "phase": "phase_a",
            "iteration": iteration,
            "max_iterations": max_iterations,
            "hypotheses": current_hypotheses,
            "plans": [],
            "results": [],
            "analyses": [],
            "evolution": None,
            "novelty_score": novelty_score,
            "session_data": None,
        }

        # ---- Phase B: 实验规划 ----
        log("")
        log(f"[PHASE B] 规划实验...")
        current_plans = mod.plan_experiments(current_hypotheses, iteration)
        for plan in current_plans:
            log(f"  为假设 {plan['hypothesis_id']} 设计实验:")
            log(f"     方法：{plan['method']}")
            log(f"     工具：{', '.join(plan['tool_chain'])}")
            log(f"     复杂度：{plan['estimated_complexity']}")
            log(f"     步骤数：{len(plan['steps'])}")
        await asyncio.sleep(0.3)

        yield {
            "phase": "phase_b",
            "iteration": iteration,
            "max_iterations": max_iterations,
            "hypotheses": current_hypotheses,
            "plans": current_plans,
            "results": [],
            "analyses": [],
            "evolution": None,
            "novelty_score": novelty_score,
            "session_data": None,
        }

        # ---- Phase C: 实验执行 ----
        log("")
        log(f"[PHASE C] 执行实验...")
        current_results = mod.execute_experiments(current_plans)
        for result in current_results:
            status = "成功" if result["success"] else "失败"
            log(f"  实验 {result['hypothesis_id']}: {status}")
            if result["success"] and result.get("key_findings"):
                for finding in result["key_findings"][:2]:
                    log(f"     - {finding}")
            output_text = result.get("execution_output", "")
            if output_text:
                log(f"     输出：{output_text[:100]}...")
        await asyncio.sleep(0.3)

        yield {
            "phase": "phase_c",
            "iteration": iteration,
            "max_iterations": max_iterations,
            "hypotheses": current_hypotheses,
            "plans": current_plans,
            "results": current_results,
            "analyses": [],
            "evolution": None,
            "novelty_score": novelty_score,
            "session_data": None,
        }

        # ---- Phase D: 分析与进化 ----
        log("")
        log(f"[PHASE D] 分析结果并进化假设...")
        evo = mod.analyze_and_evolve(current_hypotheses, current_results, iteration)
        current_analyses = evo.get("analyses", [])
        current_evolution = evo

        log("  假设评估:")
        for ev in evo["evaluations"]:
            log(f"     {ev['hypothesis_id']}: {ev['verdict']} "
                f"(置信度: {ev['confidence_before']:.2f} -> {ev['confidence_after']:.2f})")

        log(f"  假设进化:")
        log(f"     新增假设：{len(evo['new_hypotheses'])} 个")
        log(f"     淘汰假设：{len(evo['eliminated_ids'])} 个")
        log(f"     更新假设：{len(evo['updated_ids'])} 个")

        converged = evo.get("converged", False)
        await asyncio.sleep(0.3)

        yield {
            "phase": "phase_d",
            "iteration": iteration,
            "max_iterations": max_iterations,
            "hypotheses": current_hypotheses,
            "plans": current_plans,
            "results": current_results,
            "analyses": current_analyses,
            "evolution": current_evolution,
            "novelty_score": novelty_score,
            "session_data": None,
        }

        # ---- Phase E: 记忆与技能 ----
        log("")
        log(f"[PHASE E] 保存到记忆并固化技能...")
        log(f"  保存发现到记忆...")
        successful_count = sum(1 for r in current_results if r["success"])
        log(f"  固化 {successful_count} 个成功实验为可复用技能...")

        novelty_score = min(0.95, novelty_score + 0.05)
        log(f"  新颖性评分更新：{novelty_score - 0.05:.2f} -> {novelty_score:.2f}")
        await asyncio.sleep(0.2)

        # 记录本轮迭代数据
        all_iterations_data.append({
            "iteration": iteration,
            "hypotheses": current_hypotheses,
            "plans": current_plans,
            "results": [
                {
                    "hypothesis_id": r["hypothesis_id"],
                    "plan_summary": r.get("plan_summary", ""),
                    "success": r["success"],
                    "final_code": r.get("final_code", ""),
                    "execution_output": r.get("execution_output", ""),
                    "error_log": r.get("error_log", ""),
                }
                for r in current_results
            ],
            "analysis": current_analyses,
            "evolution": {
                "new_hypotheses": evo.get("new_hypotheses", []),
                "eliminated_ids": evo.get("eliminated_ids", []),
                "evolution_reasoning": evo.get("evolution_reasoning", ""),
                "converged": converged,
            },
            "timestamp": datetime.now().isoformat(),
        })

        yield {
            "phase": "phase_e",
            "iteration": iteration,
            "max_iterations": max_iterations,
            "hypotheses": current_hypotheses,
            "plans": current_plans,
            "results": current_results,
            "analyses": current_analyses,
            "evolution": current_evolution,
            "novelty_score": novelty_score,
            "session_data": None,
        }

        if converged:
            log(f"\n  假设已收敛，提前结束迭代。")
            break

    # ------------------------------------------------------------------
    # 最终总结
    # ------------------------------------------------------------------
    log("")
    log("=" * 70)
    log("发现完成!")
    log("=" * 70)
    log(f"会话 ID: {session_id}")
    log(f"总迭代次数：{len(all_iterations_data)}")
    log(f"最终假设数量：{len(current_hypotheses)}")
    supported = sum(
        1 for a in current_analyses if a.get("supports_hypothesis", False)
    )
    log(f"支持的假设：{supported}")
    log(f"收敛：{'Yes' if converged else 'No'}")
    log(f"新颖性评分：{novelty_score:.2f}")
    log("=" * 70)

    session_data = {
        "session_id": session_id,
        "domain": display_domain,
        "problem": problem,
        "model_name": "demo-mock",
        "converged": converged,
        "overall_novelty_score": novelty_score,
        "current_iteration": len(all_iterations_data),
        "max_iterations": max_iterations,
        "iterations": all_iterations_data,
        "timestamp": datetime.now().isoformat(),
    }

    yield {
        "phase": "complete",
        "iteration": len(all_iterations_data),
        "max_iterations": max_iterations,
        "hypotheses": current_hypotheses,
        "plans": current_plans,
        "results": current_results,
        "analyses": current_analyses,
        "evolution": current_evolution,
        "novelty_score": novelty_score,
        "session_data": session_data,
    }


# ---------------------------------------------------------------------------
# JSON 保存
# ---------------------------------------------------------------------------

def save_session(session_data: Dict, save_dir: Path) -> str:
    """将完整会话数据保存为 JSON 文件，返回文件路径。"""
    save_dir.mkdir(parents=True, exist_ok=True)
    filename = f"{session_data['session_id']}.json"
    filepath = save_dir / filename
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(session_data, f, indent=2, ensure_ascii=False)
    return str(filepath)
