"""
Discovery-Agent 系统能力基准测试执行器

评估系统在多个科学领域（电池材料、催化科学、电子与通信）下的自主科学发现能力。
支持常规题定量评估（每题5次）和 GT 回顾题定性评估（每题1次）。

用法:
    python benchmarks/benchmark_runner.py                       # 运行全部基准测试
    python benchmarks/benchmark_runner.py --domain battery      # 仅运行电池领域
    python benchmarks/benchmark_runner.py --problem A1          # 仅运行指定问题
    python benchmarks/benchmark_runner.py --gt-only             # 仅运行 GT 回顾题
    python benchmarks/benchmark_runner.py --summary-only        # 仅从已有结果生成汇总
"""

from __future__ import annotations

import sys
from pathlib import Path

# 确保项目根目录在 sys.path 中（必须在 _utf8_env 导入之前设置）
_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import _utf8_env  # noqa: F401  # 全局 UTF-8 编码修复（必须在所有其他 import 之前）

import argparse
import asyncio
import json
import math
import os
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import List, Optional

# 禁用 OpenAI Agents SDK tracing（避免 DeepSeek key 被发送到 OpenAI）
try:
    from agents import set_tracing_disabled
    set_tracing_disabled(True)
except Exception:
    pass


# ---------------------------------------------------------------------------
# 基准测试问题定义
# ---------------------------------------------------------------------------

@dataclass
class BenchmarkProblem:
    """单个基准测试问题的定义"""
    problem_id: str          # 问题 ID: A1, A2, A3, B1, ..., C3
    domain: str              # 领域: battery, catalysis, electronics
    difficulty: str          # 难度: basic, intermediate, exploration
    statement: str           # 问题陈述
    mode: str = "regular"    # 模式: regular（定量）或 gt（定性）
    ground_truth: str = ""   # GT 答案（仅 GT 题有效）


# 领域A：电池材料
BATTERY_PROBLEMS = [
    BenchmarkProblem(
        problem_id="A1",
        domain="battery",
        difficulty="basic",
        statement="如何提高锂离子电池的能量密度？",
    ),
    BenchmarkProblem(
        problem_id="A2",
        domain="battery",
        difficulty="intermediate",
        statement="设计一种高稳定性固态电解质材料体系",
    ),
    BenchmarkProblem(
        problem_id="A3",
        domain="battery",
        difficulty="exploration",
        statement="针对钠离子电池负极的容量衰减机制提出改进策略",
    ),
    BenchmarkProblem(
        problem_id="A-GT",
        domain="battery",
        difficulty="gt_review",
        statement="在 LiFePO4 正极材料体系下，提出可显著提升倍率性能的改性策略",
        mode="gt",
        ground_truth="纳米化 + 碳包覆",
    ),
]

# 领域B：催化科学
CATALYSIS_PROBLEMS = [
    BenchmarkProblem(
        problem_id="B1",
        domain="catalysis",
        difficulty="basic",
        statement="如何提高 CO2 电化学还原的选择性？",
    ),
    BenchmarkProblem(
        problem_id="B2",
        domain="catalysis",
        difficulty="intermediate",
        statement="设计一种抗中毒的低成本燃料电池催化剂",
    ),
    BenchmarkProblem(
        problem_id="B3",
        domain="catalysis",
        difficulty="exploration",
        statement="光催化水分解中载流子复合的抑制策略",
    ),
    BenchmarkProblem(
        problem_id="B-GT",
        domain="catalysis",
        difficulty="gt_review",
        statement="针对氨合成反应，提出常温常压下替代 Haber-Bosch 的催化策略",
        mode="gt",
        ground_truth="单原子/电催化 N2 还原",
    ),
]

# 领域C：电子与通信
ELECTRONICS_PROBLEMS = [
    BenchmarkProblem(
        problem_id="C1",
        domain="electronics",
        difficulty="basic",
        statement="如何降低高频信号传输中的损耗？",
    ),
    BenchmarkProblem(
        problem_id="C2",
        domain="electronics",
        difficulty="intermediate",
        statement="设计一种低功耗毫米波天线阵列",
    ),
    BenchmarkProblem(
        problem_id="C3",
        domain="electronics",
        difficulty="exploration",
        statement="针对 6G 通信中的智能超表面优化策略",
    ),
    BenchmarkProblem(
        problem_id="C-GT",
        domain="electronics",
        difficulty="gt_review",
        statement="针对 5G 毫米波波束管理，提出降低波束切换时延的策略",
        mode="gt",
        ground_truth="AI 辅助波束预测 / 码本优化",
    ),
]

# 全部问题集
ALL_PROBLEMS = BATTERY_PROBLEMS + CATALYSIS_PROBLEMS + ELECTRONICS_PROBLEMS
REGULAR_PROBLEMS = [p for p in ALL_PROBLEMS if p.mode == "regular"]
GT_PROBLEMS = [p for p in ALL_PROBLEMS if p.mode == "gt"]


# ---------------------------------------------------------------------------
# 运行配置
# ---------------------------------------------------------------------------

# 全功能配置（12 个模块全部启用）
FULL_CONFIG = {
    "use_debate": True,
    "use_gan": True,
    "use_risk_assessment": True,
    "use_risk_static_only": False,
    "use_statistical_testing": True,
    "use_visualization": True,
    "use_causal_inference": True,
    "use_rl_evolution": True,
    "use_skill_solidify": True,
    "use_cross_domain": True,
    "use_literature": True,
    "use_memory": True,
}

# 运行参数
N_REGULAR_RUNS = 5         # 常规题每题运行次数
N_GT_RUNS = 1              # GT 题每题运行次数
MAX_ITERATIONS = 5         # 最大发现迭代轮数
MODEL_NAME = os.getenv("AGENT_MODEL_NAME", "deepseek-v4-pro")

# 输出根目录
OUTPUT_ROOT = _project_root / "benchmark_results"

# 领域到子目录的映射
DOMAIN_DIR_MAP = {
    "battery": "battery",
    "catalysis": "catalysis",
    "electronics": "electronics",
}


# ---------------------------------------------------------------------------
# 单次实验执行
# ---------------------------------------------------------------------------

async def run_single_experiment(
    problem: BenchmarkProblem,
    run_id: int,
) -> dict:
    """
    执行单次基准测试实验，返回 DiscoveryState.model_dump() 和运行时间。

    参数:
        problem: 基准测试问题定义
        run_id: 运行编号 (0-based)

    返回:
        {"state_dump": dict, "runtime_seconds": float}
    """
    from orchestrator import DiscoveryOrchestrator

    # 使用唯一 user_id 隔离记忆/技能
    user_id = f"benchmark_{problem.problem_id}_run{run_id}"

    orchestrator = DiscoveryOrchestrator(
        domain=problem.domain,
        initial_problem=problem.statement,
        max_iterations=MAX_ITERATIONS,
        model_name=MODEL_NAME,
        user_id=user_id,
        interactive=False,
        verbose=False,
        **FULL_CONFIG,
    )

    start_time = time.time()
    state = await orchestrator.run()
    elapsed = time.time() - start_time

    return {
        "state_dump": state.model_dump(),
        "runtime_seconds": elapsed,
    }


# ---------------------------------------------------------------------------
# 指标提取
# ---------------------------------------------------------------------------

def extract_benchmark_metrics(state_dump: dict, runtime_seconds: float) -> dict:
    """
    从单次实验结果中提取基准测试指标。
    复用 ablation_metrics.py 的提取逻辑。
    """
    from ablation_metrics import extract_metrics
    return extract_metrics(state_dump, runtime_seconds)


def compute_gt_hit_assessment(state_dump: dict, ground_truth: str) -> dict:
    """
    评估 GT 题的命中情况（定性分析）。
    从最终假设状态中分析是否逼近了 Ground Truth。

    返回:
        {"final_hypotheses": list, "ground_truth": str, "assessment_pending": True}
        (实际命中判定需后续 LLM-as-Judge 评估)
    """
    iterations = state_dump.get("iterations", [])

    # 提取最终迭代的确认假设
    final_hypotheses = []
    if iterations:
        last_iter = iterations[-1]
        for h in last_iter.get("hypotheses", []):
            final_hypotheses.append({
                "id": h.get("id", ""),
                "statement": h.get("statement", ""),
                "status": h.get("status", ""),
                "confidence": h.get("confidence", 0.0),
            })

    return {
        "ground_truth": ground_truth,
        "final_hypotheses": final_hypotheses,
        "confirmed_count": sum(1 for h in final_hypotheses if h["status"] == "confirmed"),
        "assessment_pending": True,  # 需要后续 LLM-as-Judge 进行命中判定
    }


# ---------------------------------------------------------------------------
# 结果保存与加载
# ---------------------------------------------------------------------------

def get_result_path(problem: BenchmarkProblem, run_id: int) -> Path:
    """获取结果文件的完整路径"""
    domain_dir = DOMAIN_DIR_MAP[problem.domain]
    sub_dir = "gt" if problem.mode == "gt" else "regular"
    filename = f"{problem.problem_id}_run{run_id}.json"
    return OUTPUT_ROOT / domain_dir / sub_dir / filename


def save_result(
    problem: BenchmarkProblem,
    run_id: int,
    state_dump: dict,
    metrics: dict,
    runtime_seconds: float,
    gt_assessment: Optional[dict] = None,
):
    """保存单次实验结果到 JSON 文件"""
    result_path = get_result_path(problem, run_id)
    result_path.parent.mkdir(parents=True, exist_ok=True)

    data = {
        "problem_id": problem.problem_id,
        "domain": problem.domain,
        "difficulty": problem.difficulty,
        "statement": problem.statement,
        "mode": problem.mode,
        "run_id": run_id,
        "config": FULL_CONFIG,
        "model_name": MODEL_NAME,
        "max_iterations": MAX_ITERATIONS,
        "metrics": metrics,
        "runtime_seconds": runtime_seconds,
        "timestamp": datetime.now().isoformat(),
        "state_dump": state_dump,
    }

    if gt_assessment:
        data["gt_assessment"] = gt_assessment

    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=str, ensure_ascii=False)


def load_result(result_path: Path) -> dict:
    """加载单次实验结果"""
    with open(result_path, "r", encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# 汇总报告生成
# ---------------------------------------------------------------------------

def generate_summary():
    """
    从所有已有结果文件生成 benchmark_summary.json。
    包含每个问题的聚合指标（均值 +/- 标准差）和领域对比分析。
    """
    from ablation_metrics import aggregate_metrics, compute_convergence_rate

    summary = {
        "generated_at": datetime.now().isoformat(),
        "config": FULL_CONFIG,
        "model_name": MODEL_NAME,
        "max_iterations": MAX_ITERATIONS,
        "n_regular_runs": N_REGULAR_RUNS,
        "n_gt_runs": N_GT_RUNS,
        "regular_results": {},
        "gt_results": {},
        "domain_comparison": {},
        "difficulty_comparison": {},
    }

    # 收集所有常规题结果
    for problem in REGULAR_PROBLEMS:
        metrics_list = []
        for run_id in range(N_REGULAR_RUNS):
            result_path = get_result_path(problem, run_id)
            if result_path.exists():
                try:
                    data = load_result(result_path)
                    metrics_list.append(data["metrics"])
                except Exception as e:
                    print(f"  Warning: Failed to load {result_path.name}: {e}")

        if metrics_list:
            summary["regular_results"][problem.problem_id] = {
                "domain": problem.domain,
                "difficulty": problem.difficulty,
                "statement": problem.statement,
                "n_completed": len(metrics_list),
                "aggregated": aggregate_metrics(metrics_list),
                "convergence_rate": compute_convergence_rate(metrics_list),
            }

    # 收集 GT 题结果（支持多次运行聚合）
    for problem in GT_PROBLEMS:
        gt_metrics_list = []
        gt_assessments = []
        gt_runtimes = []
        for run_id in range(N_GT_RUNS):
            result_path = get_result_path(problem, run_id)
            if result_path.exists():
                try:
                    data = load_result(result_path)
                    gt_metrics_list.append(data["metrics"])
                    gt_assessments.append({
                        "run_id": run_id,
                        **data.get("gt_assessment", {}),
                    })
                    gt_runtimes.append(data.get("runtime_seconds", 0.0))
                except Exception as e:
                    print(f"  Warning: Failed to load {result_path.name}: {e}")

        if gt_metrics_list:
            gt_entry = {
                "domain": problem.domain,
                "statement": problem.statement,
                "ground_truth": problem.ground_truth,
                "n_completed": len(gt_metrics_list),
                "gt_assessments": gt_assessments,
                "runtime_seconds_list": gt_runtimes,
            }
            if len(gt_metrics_list) > 1:
                gt_entry["aggregated"] = aggregate_metrics(gt_metrics_list)
                gt_entry["convergence_rate"] = compute_convergence_rate(gt_metrics_list)
            else:
                # 单次运行保持原行为以向后兼容
                gt_entry["metrics"] = gt_metrics_list[0]
                gt_entry["gt_assessment"] = gt_assessments[0] if gt_assessments else {}
                gt_entry["runtime_seconds"] = gt_runtimes[0] if gt_runtimes else 0.0
            summary["gt_results"][problem.problem_id] = gt_entry

    # 领域对比分析
    for domain in ["battery", "catalysis", "electronics"]:
        domain_metrics = []
        for pid, result in summary["regular_results"].items():
            if result["domain"] == domain:
                # 收集该领域所有 runs 的原始指标
                for run_id in range(N_REGULAR_RUNS):
                    prob = next((p for p in REGULAR_PROBLEMS if p.problem_id == pid), None)
                    if prob:
                        rp = get_result_path(prob, run_id)
                        if rp.exists():
                            try:
                                domain_metrics.append(load_result(rp)["metrics"])
                            except Exception:
                                pass
        if domain_metrics:
            summary["domain_comparison"][domain] = {
                "n_runs": len(domain_metrics),
                "aggregated": aggregate_metrics(domain_metrics),
                "convergence_rate": compute_convergence_rate(domain_metrics),
            }

    # 难度对比分析
    for difficulty in ["basic", "intermediate", "exploration"]:
        diff_metrics = []
        for pid, result in summary["regular_results"].items():
            if result["difficulty"] == difficulty:
                for run_id in range(N_REGULAR_RUNS):
                    prob = next((p for p in REGULAR_PROBLEMS if p.problem_id == pid), None)
                    if prob:
                        rp = get_result_path(prob, run_id)
                        if rp.exists():
                            try:
                                diff_metrics.append(load_result(rp)["metrics"])
                            except Exception:
                                pass
        if diff_metrics:
            summary["difficulty_comparison"][difficulty] = {
                "n_runs": len(diff_metrics),
                "aggregated": aggregate_metrics(diff_metrics),
                "convergence_rate": compute_convergence_rate(diff_metrics),
            }

    # 保存汇总
    summary_path = OUTPUT_ROOT / "summary" / "benchmark_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str, ensure_ascii=False)

    print(f"\n  Summary saved to: {summary_path}")
    _print_summary_table(summary)

    return summary


def _print_summary_table(summary: dict):
    """在控制台打印简要汇总表格"""
    print(f"\n{'='*70}")
    print("  BENCHMARK RESULTS SUMMARY")
    print(f"{'='*70}")

    # 常规题概览
    if summary["regular_results"]:
        print(f"\n  {'Problem':<8} {'Domain':<12} {'Diff':<14} {'ESR':>6} {'NS':>6} "
              f"{'Conv':>6} {'Iter':>5} {'Time(s)':>8}")
        print(f"  {'-'*65}")
        for pid, r in sorted(summary["regular_results"].items()):
            agg = r["aggregated"]
            esr = agg.get("experiment_success_rate", {}).get("mean", 0)
            ns = agg.get("novelty_score", {}).get("mean", 0)
            conv = r["convergence_rate"]
            itr = agg.get("iterations_used", {}).get("mean", 0)
            rt = agg.get("total_runtime_seconds", {}).get("mean", 0)
            print(f"  {pid:<8} {r['domain']:<12} {r['difficulty']:<14} "
                  f"{esr:>6.3f} {ns:>6.3f} {conv:>6.2f} {itr:>5.1f} {rt:>8.0f}")

    # GT 题概览
    if summary["gt_results"]:
        print(f"\n  {'Problem':<8} {'Domain':<12} {'GT':<30} {'Status':<12}")
        print(f"  {'-'*65}")
        for pid, r in sorted(summary["gt_results"].items()):
            gt = r["ground_truth"][:28]
            status = "pending" if r.get("gt_assessment", {}).get("assessment_pending") else "assessed"
            print(f"  {pid:<8} {r['domain']:<12} {gt:<30} {status:<12}")

    print(f"\n{'='*70}")


# ---------------------------------------------------------------------------
# 主执行流程
# ---------------------------------------------------------------------------

async def run_benchmark(
    target_domain: Optional[str] = None,
    target_problem: Optional[str] = None,
    gt_only: bool = False,
    regular_only: bool = False,
):
    """
    执行基准测试。

    参数:
        target_domain: 仅运行指定领域 (battery/catalysis/electronics)
        target_problem: 仅运行指定问题 ID (如 "A1", "B-GT")
        gt_only: 仅运行 GT 回顾题
        regular_only: 仅运行常规题
    """
    # 确定要运行的问题集
    problems_to_run: List[BenchmarkProblem] = []

    if target_problem:
        # 运行指定问题
        matched = [p for p in ALL_PROBLEMS if p.problem_id == target_problem]
        if not matched:
            print(f"  ERROR: Unknown problem ID '{target_problem}'")
            print(f"  Available: {[p.problem_id for p in ALL_PROBLEMS]}")
            return
        problems_to_run = matched
    elif gt_only:
        problems_to_run = GT_PROBLEMS
    elif regular_only:
        problems_to_run = REGULAR_PROBLEMS
    else:
        problems_to_run = ALL_PROBLEMS

    # 按领域过滤
    if target_domain:
        problems_to_run = [p for p in problems_to_run if p.domain == target_domain]
        if not problems_to_run:
            print(f"  ERROR: No problems found for domain '{target_domain}'")
            return

    # 计算总运行数
    total_planned = sum(
        N_GT_RUNS if p.mode == "gt" else N_REGULAR_RUNS
        for p in problems_to_run
    )

    # 打印运行计划
    total_start = time.time()
    total_runs = 0
    total_skipped = 0
    total_errors = 0

    print(f"\n{'='*70}")
    print(f"  DISCOVERY-AGENT Benchmark Test")
    print(f"  Model: {MODEL_NAME} | Max iterations: {MAX_ITERATIONS}")
    print(f"  Regular runs: {N_REGULAR_RUNS}/problem | GT runs: {N_GT_RUNS}/problem")
    print(f"  Problems: {len(problems_to_run)} | Total planned runs: {total_planned}")
    print(f"  Output: {OUTPUT_ROOT}")
    print(f"{'='*70}\n")

    # 按领域分组执行
    domains_order = ["battery", "catalysis", "electronics"]
    for domain in domains_order:
        domain_problems = [p for p in problems_to_run if p.domain == domain]
        if not domain_problems:
            continue

        print(f"\n{'#'*60}")
        print(f"  Domain: {domain} ({len(domain_problems)} problems)")
        print(f"{'#'*60}")

        for problem in domain_problems:
            n_runs = N_GT_RUNS if problem.mode == "gt" else N_REGULAR_RUNS
            mode_label = "GT" if problem.mode == "gt" else "Regular"

            print(f"\n  --- [{mode_label}] {problem.problem_id}: "
                  f"{problem.statement[:50]}... ---")

            for run_id in range(n_runs):
                result_path = get_result_path(problem, run_id)

                # 断点续跑: 跳过已完成的
                if result_path.exists():
                    print(f"    SKIP (exists): {result_path.name}")
                    total_skipped += 1
                    continue

                # 执行实验
                print(f"    RUN: {problem.problem_id} | run {run_id}/{n_runs-1} | "
                      f"[{total_runs + total_skipped + 1}/{total_planned}]")

                try:
                    exp_result = await run_single_experiment(problem, run_id)
                    state_dump = exp_result["state_dump"]
                    runtime = exp_result["runtime_seconds"]

                    # 提取指标
                    metrics = extract_benchmark_metrics(state_dump, runtime)

                    # GT 题额外评估
                    gt_assessment = None
                    if problem.mode == "gt":
                        gt_assessment = compute_gt_hit_assessment(
                            state_dump, problem.ground_truth
                        )

                    # 保存结果
                    save_result(
                        problem, run_id, state_dump, metrics, runtime, gt_assessment
                    )

                    total_runs += 1
                    c_flag = "C" if metrics["converged"] else "X"
                    print(
                        f"    DONE [{c_flag}]: {runtime:.0f}s | "
                        f"ESR={metrics['experiment_success_rate']:.2f} | "
                        f"NS={metrics['novelty_score']:.2f} | "
                        f"Iter={metrics['iterations_used']}"
                    )

                except Exception as e:
                    print(f"    ERROR: {problem.problem_id} run{run_id}: {e}")
                    total_errors += 1
                    total_runs += 1

                    # 保存错误记录到日志
                    _save_error_log(problem, run_id, e)

    # 运行完成，生成汇总
    total_time = time.time() - total_start
    print(f"\n{'='*70}")
    print(f"  Benchmark Complete")
    print(f"  Executed: {total_runs} | Skipped: {total_skipped} | "
          f"Errors: {total_errors}")
    print(f"  Total time: {total_time:.0f}s ({total_time/3600:.1f}h)")
    print(f"{'='*70}")

    # 生成汇总报告
    generate_summary()


# ---------------------------------------------------------------------------
# 错误日志
# ---------------------------------------------------------------------------

def _save_error_log(problem: BenchmarkProblem, run_id: int, error: Exception):
    """保存运行错误日志"""
    import traceback

    log_dir = OUTPUT_ROOT / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{problem.problem_id}_run{run_id}.log"

    with open(log_path, "w", encoding="utf-8") as f:
        f.write(f"Problem: {problem.problem_id}\n")
        f.write(f"Domain: {problem.domain}\n")
        f.write(f"Statement: {problem.statement}\n")
        f.write(f"Run ID: {run_id}\n")
        f.write(f"Timestamp: {datetime.now().isoformat()}\n")
        f.write(f"Error: {error}\n")
        f.write(f"\nTraceback:\n")
        f.write(traceback.format_exc())


# ---------------------------------------------------------------------------
# CLI 入口
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Discovery-Agent 系统能力基准测试执行器"
    )
    parser.add_argument(
        "--domain", default=None,
        choices=["battery", "catalysis", "electronics"],
        help="仅运行指定领域"
    )
    parser.add_argument(
        "--problem", default=None,
        help="仅运行指定问题 ID (如 A1, B2, C-GT)"
    )
    parser.add_argument(
        "--gt-only", action="store_true",
        help="仅运行 GT 回顾题（3题各1次）"
    )
    parser.add_argument(
        "--regular-only", action="store_true",
        help="仅运行常规题（9题各5次）"
    )
    parser.add_argument(
        "--summary-only", action="store_true",
        help="跳过实验执行，仅从已有结果生成汇总报告"
    )
    parser.add_argument(
        "--n-runs", type=int, default=None,
        help=f"覆盖常规题的运行次数 (默认: {N_REGULAR_RUNS})"
    )
    parser.add_argument(
        "--n-gt-runs", type=int, default=None,
        help=f"覆盖 GT 题的运行次数 (默认: {N_GT_RUNS})"
    )
    parser.add_argument(
        "--max-iterations", type=int, default=None,
        help=f"覆盖最大迭代轮数 (默认: {MAX_ITERATIONS})"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # 应用参数覆盖
    global N_REGULAR_RUNS, N_GT_RUNS, MAX_ITERATIONS
    if args.n_runs is not None:
        N_REGULAR_RUNS = args.n_runs
    if args.n_gt_runs is not None:
        N_GT_RUNS = args.n_gt_runs
    if args.max_iterations is not None:
        MAX_ITERATIONS = args.max_iterations

    # 确保输出目录存在
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    if args.summary_only:
        print("Summary-only mode: generating summary from existing results...")
        generate_summary()
        return

    asyncio.run(run_benchmark(
        target_domain=args.domain,
        target_problem=args.problem,
        gt_only=args.gt_only,
        regular_only=args.regular_only,
    ))


if __name__ == "__main__":
    main()
