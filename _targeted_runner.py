"""
定向消融实验执行器（临时脚本）

任务范围（仅本机执行，其他设备已完成 electronics 和 C5）:
1. C4-No-Literature × battery × run1~5
2. C4-No-Literature × catalysis × run1~5
3. C3-No-Cross-Domain × catalysis × run4 (补跑空文件)
4. C3-No-Cross-Domain × electronics × run3 (补跑空文件)

复用 ablation_runner 的 ALL_CONFIGS / DOMAIN_SEEDS / run_single_experiment / save_result。
断点续跑：已存在文件自动跳过。
"""

from __future__ import annotations

import _utf8_env  # noqa: F401  # UTF-8 编码修复（必须最先 import）

import asyncio
import os
import sys
import time
from pathlib import Path

_project_root = Path(__file__).resolve().parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

try:
    from agents import set_tracing_disabled
    set_tracing_disabled(True)
except Exception:
    pass

from ablation_runner import (
    ALL_CONFIGS,
    DOMAIN_SEEDS,
    OUTPUT_ROOT,
    run_single_experiment,
    save_result,
    result_filename,
    save_group_summary,
)
from ablation_metrics import extract_metrics


# 定向任务清单：(config_name, domain, run_id)
TARGET_TASKS = [
    # C4-No-Literature: battery 5 + catalysis 5
    ("C4-No-Literature", "battery", 1),
    ("C4-No-Literature", "battery", 2),
    ("C4-No-Literature", "battery", 3),
    ("C4-No-Literature", "battery", 4),
    ("C4-No-Literature", "battery", 5),
    ("C4-No-Literature", "catalysis", 1),
    ("C4-No-Literature", "catalysis", 2),
    ("C4-No-Literature", "catalysis", 3),
    ("C4-No-Literature", "catalysis", 4),
    ("C4-No-Literature", "catalysis", 5),
    # C3-No-Cross-Domain: 补跑两个空文件
    ("C3-No-Cross-Domain", "catalysis", 4),
    ("C3-No-Cross-Domain", "electronics", 3),
]

GROUP_DIR = OUTPUT_ROOT / "GroupC"


async def main():
    GROUP_DIR.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*70}")
    print(f"  Targeted Ablation Runner (本机仅跑指定任务)")
    print(f"  Model: {os.getenv('AGENT_MODEL_NAME', 'deepseek-v4-pro')}")
    print(f"  Tasks: {len(TARGET_TASKS)} | Output: {GROUP_DIR}")
    print(f"{'='*70}\n")

    total_start = time.time()
    done = 0
    skipped = 0
    failed = 0

    for idx, (config_name, domain, run_id) in enumerate(TARGET_TASKS, 1):
        fname = result_filename(config_name, domain, run_id)
        result_path = GROUP_DIR / fname

        # 断点续跑：跳过非空已存在文件（<2KB 视为残留空文件，应重跑）
        if result_path.exists() and result_path.stat().st_size >= 2048:
            print(f"  [{idx}/{len(TARGET_TASKS)}] SKIP (exists): {fname}")
            skipped += 1
            continue

        # 移除残留空文件
        if result_path.exists():
            print(f"  [{idx}/{len(TARGET_TASKS)}] REMOVE empty residual: {fname} ({result_path.stat().st_size}B)")
            result_path.unlink()

        config = ALL_CONFIGS[config_name]
        seed = DOMAIN_SEEDS[domain]
        print(f"\n  [{idx}/{len(TARGET_TASKS)}] RUN: {config_name} | {domain} | run {run_id}")
        try:
            exp_result = await run_single_experiment(
                config, config_name, domain, seed, run_id
            )
            state_dump = exp_result["state_dump"]
            runtime = exp_result["runtime_seconds"]
            metrics = extract_metrics(state_dump, runtime)

            save_result(result_path, config_name, domain, run_id,
                        config, state_dump, metrics, runtime)

            done += 1
            c_flag = "C" if metrics["converged"] else "X"
            print(
                f"    DONE [{c_flag}]: {runtime:.0f}s | "
                f"ESR={metrics['experiment_success_rate']:.2f} | "
                f"NS={metrics['novelty_score']:.2f}"
            )
        except Exception as e:
            failed += 1
            print(f"    ERROR: {config_name} {domain} run{run_id}: {e}")
            import traceback
            traceback.print_exc()

    total_time = time.time() - total_start
    print(f"\n{'='*70}")
    print(f"  Targeted Run Complete")
    print(f"  Done: {done} | Skipped: {skipped} | Failed: {failed}")
    print(f"  Total time: {total_time:.0f}s ({total_time/3600:.1f}h)")
    print(f"{'='*70}")

    # 任务全部完成后刷新 group_summary
    try:
        save_group_summary(GROUP_DIR, "C")
    except Exception as e:
        print(f"  Warning: group summary failed: {e}")


if __name__ == "__main__":
    asyncio.run(main())
