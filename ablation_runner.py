"""
消融实验自动执行器

按组顺序执行所有消融变体实验，支持断点续跑和报告生成。

用法:
    python ablation_runner.py                          # 运行所有组
    python ablation_runner.py --group A                # 仅运行 Group A
    python ablation_runner.py --group A --config A1-No-Debate  # 仅运行单个配置
    python ablation_runner.py --report-only            # 仅从已有结果生成报告
"""

from __future__ import annotations

import _utf8_env  # noqa: F401  # 全局 UTF-8 编码修复（必须在所有其他 import 之前）

import argparse
import asyncio
import json
import os
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path

# 确保项目根目录在 sys.path 中
_project_root = Path(__file__).resolve().parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

# 禁用 OpenAI Agents SDK tracing（避免 DeepSeek key 被发送到 OpenAI）
try:
    from agents import set_tracing_disabled
    set_tracing_disabled(True)
except Exception:
    pass


# ---------------------------------------------------------------------------
# 消融配置定义
# ---------------------------------------------------------------------------

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

ALL_CONFIGS = {
    # 基线
    "FULL": {**FULL_CONFIG},
    # Group A: 假设生成组件消融
    "A1-No-Debate":   {**FULL_CONFIG, "use_debate": False},
    "A2-No-GAN":      {**FULL_CONFIG, "use_gan": False},
    "A3-Debate-Only": {**FULL_CONFIG, "use_gan": False},
    "A4-GAN-Only":    {**FULL_CONFIG, "use_debate": False},
    "A5-Vanilla-LLM": {**FULL_CONFIG, "use_debate": False, "use_gan": False},
    # Group B: 实验执行组件消融
    "B1-No-Risk":          {**FULL_CONFIG, "use_risk_assessment": False},
    "B2-Static-Risk":      {**FULL_CONFIG, "use_risk_static_only": True},
    "B3-No-Auto-Analysis": {**FULL_CONFIG, "use_statistical_testing": False,
                            "use_visualization": False, "use_causal_inference": False},
    "B4-Stats-Only":       {**FULL_CONFIG, "use_visualization": False,
                            "use_causal_inference": False},
    "B5-No-Causal":        {**FULL_CONFIG, "use_causal_inference": False},
    # Group C: 演化与记忆组件消融
    "C1-No-RL":           {**FULL_CONFIG, "use_rl_evolution": False},
    "C2-No-Skill":        {**FULL_CONFIG, "use_skill_solidify": False},
    "C3-No-Cross-Domain": {**FULL_CONFIG, "use_cross_domain": False},
    "C4-No-Literature":   {**FULL_CONFIG, "use_literature": False},
    "C5-No-Memory":       {**FULL_CONFIG, "use_memory": False},
    # Group D: 组合消融
    "D1-No-Debate-GAN":      {**FULL_CONFIG, "use_debate": False, "use_gan": False},
    "D2-No-Risk-Analysis":   {**FULL_CONFIG, "use_risk_assessment": False,
                              "use_statistical_testing": False,
                              "use_visualization": False, "use_causal_inference": False},
    "D3-No-Evolution-Skill": {**FULL_CONFIG, "use_rl_evolution": False,
                              "use_skill_solidify": False},
}

# 分组定义
GROUPS = {
    "A": ["FULL", "A1-No-Debate", "A2-No-GAN", "A3-Debate-Only",
          "A4-GAN-Only", "A5-Vanilla-LLM"],
    "B": ["FULL", "B1-No-Risk", "B2-Static-Risk", "B3-No-Auto-Analysis",
          "B4-Stats-Only", "B5-No-Causal"],
    "C": ["FULL", "C1-No-RL", "C2-No-Skill", "C3-No-Cross-Domain",
          "C4-No-Literature", "C5-No-Memory"],
    "D": ["FULL", "D1-No-Debate-GAN", "D2-No-Risk-Analysis",
          "D3-No-Evolution-Skill"],
}

# 域与种子问题映射
DOMAIN_SEEDS = {
    "battery": "How to improve energy density of Li-ion batteries?",
    "catalysis": "Design a catalyst for selective CO2 reduction",
    "electronics": "Find materials with high electron mobility for transistor channels",
}

# 预实验重复次数
N_RUNS = 5

# 最大迭代轮数
MAX_ITERATIONS = 5

# 输出根目录
OUTPUT_ROOT = _project_root / "ablation_formal"


# ---------------------------------------------------------------------------
# 单次实验执行
# ---------------------------------------------------------------------------

async def run_single_experiment(
    config: dict,
    config_name: str,
    domain: str,
    seed_problem: str,
    run_id: int,
) -> dict:
    """
    执行单次消融实验，返回 DiscoveryState.model_dump() 和运行时间。
    """
    from orchestrator import DiscoveryOrchestrator

    # 每次运行使用唯一 user_id 以隔离记忆/技能
    user_id = f"ablation_{config_name}_{domain}_{run_id}"

    orchestrator = DiscoveryOrchestrator(
        domain=domain,
        initial_problem=seed_problem,
        max_iterations=MAX_ITERATIONS,
        model_name=os.getenv("AGENT_MODEL_NAME", "deepseek-v4-pro"),
        user_id=user_id,
        interactive=False,
        verbose=False,
        **config,
    )

    start_time = time.time()
    state = await orchestrator.run()
    elapsed = time.time() - start_time

    return {
        "state_dump": state.model_dump(),
        "runtime_seconds": elapsed,
    }


# ---------------------------------------------------------------------------
# 结果保存与加载
# ---------------------------------------------------------------------------

def save_result(
    result_path: Path,
    config_name: str,
    domain: str,
    run_id: int,
    config: dict,
    state_dump: dict,
    metrics: dict,
    runtime_seconds: float,
):
    """保存单次实验结果到 JSON 文件"""
    result_path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "config_name": config_name,
        "domain": domain,
        "run_id": run_id,
        "switches": config,
        "metrics": metrics,
        "runtime_seconds": runtime_seconds,
        "timestamp": datetime.now().isoformat(),
        "state_dump": state_dump,
    }
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=str, ensure_ascii=False)


def load_result(result_path: Path) -> dict:
    """加载单次实验结果"""
    with open(result_path, "r", encoding="utf-8") as f:
        return json.load(f)


def result_filename(config_name: str, domain: str, run_id: int) -> str:
    """生成结果文件名"""
    return f"{config_name}_{domain}_run{run_id}.json"


# ---------------------------------------------------------------------------
# 分组摘要生成
# ---------------------------------------------------------------------------

def save_group_summary(group_dir: Path, group_name: str):
    """为一组实验生成摘要文件"""
    from ablation_metrics import aggregate_metrics, compute_convergence_rate

    summary = {"group": group_name, "configs": {}}

    # 遍历组目录中的所有结果文件
    for result_file in sorted(group_dir.glob("*.json")):
        if result_file.name == "group_summary.json":
            continue
        try:
            data = load_result(result_file)
            config_name = data["config_name"]
            if config_name not in summary["configs"]:
                summary["configs"][config_name] = []
            summary["configs"][config_name].append(data["metrics"])
        except Exception as e:
            print(f"  Warning: Failed to load {result_file}: {e}")

    # 聚合每个配置的指标
    aggregated = {}
    for config_name, metric_list in summary["configs"].items():
        aggregated[config_name] = {
            "aggregated": aggregate_metrics(metric_list),
            "convergence_rate": compute_convergence_rate(metric_list),
            "n_runs": len(metric_list),
        }

    summary["aggregated"] = aggregated
    summary_path = group_dir / "group_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"  Group {group_name} summary saved to {summary_path}")


# ---------------------------------------------------------------------------
# 主执行流程
# ---------------------------------------------------------------------------

async def run_ablation_study(
    target_group: str | None = None,
    target_config: str | None = None,
):
    """
    执行消融实验。

    参数:
        target_group: 仅运行指定组（如 "A"），None 表示全部
        target_config: 仅运行指定配置（如 "A1-No-Debate"），None 表示全部
    """
    from ablation_metrics import extract_metrics

    # FULL 基线结果缓存：(domain, run_id) -> (state_dump, metrics, runtime)
    full_cache: dict = {}

    groups_to_run = (
        [(target_group, GROUPS[target_group])]
        if target_group and target_group in GROUPS
        else list(GROUPS.items())
    )

    total_start = time.time()
    total_runs = 0
    total_skipped = 0

    print(f"\n{'='*70}")
    print(f"  DISCOVERY-AGENT Ablation Study")
    print(f"  Model: {os.getenv('AGENT_MODEL_NAME', 'deepseek-v4-pro')} | Runs per config: {N_RUNS}")
    print(f"  Max iterations: {MAX_ITERATIONS} | Domains: {len(DOMAIN_SEEDS)}")
    print(f"  Output: {OUTPUT_ROOT}")
    print(f"{'='*70}\n")

    for group_name, config_names in groups_to_run:
        group_dir = OUTPUT_ROOT / f"Group{group_name}"
        group_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n{'#'*60}")
        print(f"  Group {group_name}: {len(config_names)} configs")
        print(f"{'#'*60}")

        configs_to_run = (
            [target_config] if target_config and target_config in config_names
            else config_names
        )

        for config_name in configs_to_run:
            config = ALL_CONFIGS[config_name]
            print(f"\n  --- Config: {config_name} ---")

            for domain, seed in DOMAIN_SEEDS.items():
                for run_id in range(1, N_RUNS + 1):
                    fname = result_filename(config_name, domain, run_id)
                    result_path = group_dir / fname

                    # 断点续跑: 跳过已完成的
                    if result_path.exists():
                        print(f"    SKIP (exists): {fname}")
                        total_skipped += 1

                        # 缓存 FULL 基线结果
                        if config_name == "FULL":
                            try:
                                cached = load_result(result_path)
                                full_cache[(domain, run_id)] = (
                                    cached["state_dump"],
                                    cached["metrics"],
                                    cached.get("runtime_seconds", 0),
                                )
                            except Exception:
                                pass
                        continue

                    # FULL 去重: 后续组复用 Group A 的 FULL 结果
                    if config_name == "FULL" and (domain, run_id) in full_cache:
                        sd, mt, rt = full_cache[(domain, run_id)]
                        save_result(result_path, config_name, domain, run_id,
                                    config, sd, mt, rt)
                        print(f"    COPY (cached): {fname}")
                        total_skipped += 1
                        continue

                    # 执行实验
                    print(f"    RUN: {config_name} | {domain} | run {run_id}/{N_RUNS}")
                    try:
                        exp_result = await run_single_experiment(
                            config, config_name, domain, seed, run_id
                        )
                        state_dump = exp_result["state_dump"]
                        runtime = exp_result["runtime_seconds"]
                        metrics = extract_metrics(state_dump, runtime)

                        save_result(result_path, config_name, domain, run_id,
                                    config, state_dump, metrics, runtime)

                        # 缓存 FULL 基线
                        if config_name == "FULL":
                            full_cache[(domain, run_id)] = (state_dump, metrics, runtime)

                        total_runs += 1
                        c_flag = "C" if metrics["converged"] else "X"
                        print(
                            f"    DONE [{c_flag}]: {runtime:.0f}s | "
                            f"ESR={metrics['experiment_success_rate']:.2f} | "
                            f"NS={metrics['novelty_score']:.2f}"
                        )
                    except Exception as e:
                        print(f"    ERROR: {config_name} {domain} run{run_id}: {e}")
                        total_runs += 1

        # 保存组摘要
        save_group_summary(group_dir, group_name)

    total_time = time.time() - total_start
    print(f"\n{'='*70}")
    print(f"  Ablation Study Complete")
    print(f"  Total runs: {total_runs} | Skipped: {total_skipped}")
    print(f"  Total time: {total_time:.0f}s ({total_time/3600:.1f}h)")
    print(f"{'='*70}")

    # 生成最终报告
    generate_report()


# ---------------------------------------------------------------------------
# 报告生成入口
# ---------------------------------------------------------------------------

def generate_report():
    """调用报告生成模块"""
    try:
        from ablation_report import generate_ablation_report
        report_path = OUTPUT_ROOT / "ablation_report.md"
        figures_dir = OUTPUT_ROOT / "figures"
        generate_ablation_report(OUTPUT_ROOT, report_path, figures_dir)
        print(f"\n  Report generated: {report_path}")
    except Exception as e:
        print(f"\n  Warning: Report generation failed: {e}")
        import traceback
        traceback.print_exc()


# ---------------------------------------------------------------------------
# CLI 入口
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Discovery-Agent 消融实验自动执行器"
    )
    parser.add_argument(
        "--group", default=None, choices=["A", "B", "C", "D"],
        help="仅运行指定组 (A/B/C/D)"
    )
    parser.add_argument(
        "--config", default=None,
        help="仅运行指定配置 (如 A1-No-Debate)"
    )
    parser.add_argument(
        "--report-only", action="store_true",
        help="跳过实验执行，仅从已有结果生成报告"
    )
    parser.add_argument(
        "--n-runs", type=int, default=N_RUNS,
        help=f"每个配置的重复次数 (默认: {N_RUNS})"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    global N_RUNS
    N_RUNS = args.n_runs

    if args.report_only:
        print("Report-only mode: generating report from existing results...")
        generate_report()
        return

    asyncio.run(run_ablation_study(
        target_group=args.group,
        target_config=args.config,
    ))


if __name__ == "__main__":
    main()
