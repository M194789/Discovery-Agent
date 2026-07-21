"""
消融实验报告生成模块

从消融实验结果中进行统计分析、生成可视化图表和 Markdown 报告。
"""

from __future__ import annotations

import json
import math
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

from ablation_metrics import (
    aggregate_metrics, compute_convergence_rate,
    compute_avg_convergence_iterations, compare_to_baseline,
)


# ---------------------------------------------------------------------------
# 统计分析函数
# ---------------------------------------------------------------------------

def run_paired_ttest(
    full_values: List[float], variant_values: List[float]
) -> Tuple[float, float]:
    """
    配对 t 检验（或独立样本 t 检验当样本量太小时）。
    返回 (t_stat, p_value)。样本不足时返回 (0.0, 1.0)。
    """
    if len(full_values) < 2 or len(variant_values) < 2:
        return 0.0, 1.0
    try:
        from scipy import stats
        if len(full_values) == len(variant_values):
            t_stat, p_val = stats.ttest_rel(full_values, variant_values)
        else:
            t_stat, p_val = stats.ttest_ind(full_values, variant_values)
        return (float(t_stat) if not math.isnan(t_stat) else 0.0,
                float(p_val) if not math.isnan(p_val) else 1.0)
    except Exception:
        return 0.0, 1.0


def run_one_way_anova(groups: List[List[float]]) -> Tuple[float, float]:
    """单因素方差分析。返回 (f_stat, p_value)。"""
    valid = [g for g in groups if len(g) >= 2]
    if len(valid) < 2:
        return 0.0, 1.0
    try:
        from scipy import stats
        f_stat, p_val = stats.f_oneway(*valid)
        return (float(f_stat) if not math.isnan(f_stat) else 0.0,
                float(p_val) if not math.isnan(p_val) else 1.0)
    except Exception:
        return 0.0, 1.0


def compute_cohens_d(
    full_values: List[float], variant_values: List[float]
) -> float:
    """计算 Cohen's d 效应量"""
    if len(full_values) < 2 or len(variant_values) < 2:
        return 0.0
    m1, m2 = _mean(full_values), _mean(variant_values)
    s1, s2 = _std(full_values), _std(variant_values)
    pooled = math.sqrt((s1**2 + s2**2) / 2) if (s1 + s2) > 0 else 1.0
    return (m1 - m2) / pooled if pooled > 0 else 0.0


def apply_bonferroni(p_values: List[float], n_comparisons: int) -> List[float]:
    """Bonferroni 多重比较校正"""
    return [min(p * n_comparisons, 1.0) for p in p_values]


def _mean(values: List[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _std(values: List[float]) -> float:
    if len(values) < 2:
        return 0.0
    m = _mean(values)
    return math.sqrt(sum((v - m)**2 for v in values) / (len(values) - 1))


# ---------------------------------------------------------------------------
# 可视化函数
# ---------------------------------------------------------------------------

def _check_matplotlib():
    """检查 matplotlib 是否可用"""
    try:
        import matplotlib
        matplotlib.use("Agg")
        return True
    except ImportError:
        print("  Warning: matplotlib not installed, skipping visualizations")
        return False


def generate_radar_chart(
    metrics_by_config: Dict[str, dict],
    output_dir: Path,
    primary_metrics: List[str] = None,
):
    """生成雷达图: 各变体在主要指标上的综合表现"""
    if not _check_matplotlib():
        return
    import matplotlib.pyplot as plt
    import numpy as np

    if primary_metrics is None:
        primary_metrics = [
            "converged", "novelty_score", "experiment_success_rate",
            "mean_confidence",
        ]

    labels = primary_metrics
    num_vars = len(labels)
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))

    for config_name, agg in metrics_by_config.items():
        values = [agg.get(m, {}).get("mean", 0) for m in primary_metrics]
        values += values[:1]
        ax.plot(angles, values, linewidth=1.5, label=config_name)
        ax.fill(angles, values, alpha=0.05)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, size=9)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=7)
    ax.set_title("Ablation Study - Primary Metrics Radar Chart", size=13, pad=20)

    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / "radar_chart_all.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def generate_heatmap(
    metrics_by_config: Dict[str, dict],
    output_dir: Path,
    metrics_list: List[str] = None,
):
    """生成热力图: 变体 x 指标矩阵"""
    if not _check_matplotlib():
        return
    import matplotlib.pyplot as plt
    import numpy as np

    if metrics_list is None:
        metrics_list = [
            "converged", "novelty_score", "experiment_success_rate",
            "mean_confidence", "total_hypotheses", "plan_filter_rate",
        ]

    configs = list(metrics_by_config.keys())
    data = []
    for cfg in configs:
        row = [metrics_by_config[cfg].get(m, {}).get("mean", 0) for m in metrics_list]
        data.append(row)
    data = np.array(data)

    fig, ax = plt.subplots(figsize=(max(10, len(metrics_list)*1.5), max(8, len(configs)*0.5)))
    im = ax.imshow(data, aspect="auto", cmap="RdYlGn")

    ax.set_xticks(range(len(metrics_list)))
    ax.set_xticklabels(metrics_list, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(configs)))
    ax.set_yticklabels(configs, fontsize=8)

    for i in range(len(configs)):
        for j in range(len(metrics_list)):
            ax.text(j, i, f"{data[i, j]:.2f}", ha="center", va="center",
                    fontsize=7, color="black")

    ax.set_title("Ablation Study - Metrics Heatmap", pad=12)
    fig.colorbar(im, ax=ax, shrink=0.6)

    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / "heatmap_primary_metrics.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def generate_box_plots(
    metric_name: str,
    distributions_by_config: Dict[str, List[float]],
    output_dir: Path,
):
    """生成箱线图: 关键指标的分布差异"""
    if not _check_matplotlib():
        return
    import matplotlib.pyplot as plt

    configs = list(distributions_by_config.keys())
    data = [distributions_by_config[c] for c in configs]

    fig, ax = plt.subplots(figsize=(max(10, len(configs)*0.8), 6))
    ax.boxplot(data, labels=configs, vert=True)
    ax.set_xticklabels(configs, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel(metric_name)
    ax.set_title(f"Ablation Study - {metric_name} Distribution")
    ax.grid(True, alpha=0.3)

    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / f"boxplot_{metric_name}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def generate_waterfall_chart(
    component_contributions: Dict[str, float],
    baseline_score: float,
    output_dir: Path,
):
    """生成瀑布图: 各组件的边际贡献"""
    if not _check_matplotlib():
        return
    import matplotlib.pyplot as plt
    import numpy as np

    components = list(component_contributions.keys())
    values = list(component_contributions.values())

    fig, ax = plt.subplots(figsize=(12, 6))
    cumulative = baseline_score
    positions = range(len(components) + 1)

    # 基线柱
    ax.bar(0, baseline_score, color="gray", alpha=0.7)
    ax.text(0, baseline_score + 0.01, f"{baseline_score:.3f}", ha="center", fontsize=8)

    for i, (comp, val) in enumerate(zip(components, values)):
        color = "green" if val >= 0 else "red"
        ax.bar(i + 1, val, bottom=cumulative, color=color, alpha=0.7)
        ax.text(i + 1, cumulative + val + 0.01, f"{val:+.3f}", ha="center", fontsize=7)
        cumulative += val

    labels = ["Vanilla\nBaseline"] + [c.replace("-", "\n") for c in components]
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, fontsize=7, rotation=45, ha="right")
    ax.set_ylabel("Performance Score")
    ax.set_title("Component Marginal Contributions (Waterfall)")
    ax.axhline(y=cumulative, color="blue", linestyle="--", alpha=0.5, label=f"FULL={cumulative:.3f}")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / "waterfall_contributions.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# 加载所有结果
# ---------------------------------------------------------------------------

def load_all_results(output_root: Path) -> Dict[str, List[dict]]:
    """
    加载所有组的所有结果文件，按 config_name 分组。
    返回: {config_name: [result_dict, ...]}
    """
    all_results: Dict[str, List[dict]] = {}

    for group_dir in sorted(output_root.glob("Group*")):
        if not group_dir.is_dir():
            continue
        for result_file in sorted(group_dir.glob("*.json")):
            if result_file.name == "group_summary.json":
                continue
            try:
                data = json.loads(result_file.read_text(encoding="utf-8"))
                config_name = data.get("config_name", "unknown")
                if config_name not in all_results:
                    all_results[config_name] = []
                all_results[config_name].append(data)
            except Exception:
                continue

    return all_results


# ---------------------------------------------------------------------------
# Markdown 报告生成
# ---------------------------------------------------------------------------

def generate_ablation_report(
    output_root: Path,
    report_path: Path,
    figures_dir: Path,
):
    """
    生成消融实验汇总 Markdown 报告。
    """
    all_results = load_all_results(output_root)
    if not all_results:
        print("  No results found, cannot generate report.")
        return

    # 聚合指标
    aggregated: Dict[str, dict] = {}
    raw_metrics: Dict[str, List[dict]] = {}
    for config_name, results in all_results.items():
        metrics_list = [r["metrics"] for r in results if "metrics" in r]
        if metrics_list:
            aggregated[config_name] = aggregate_metrics(metrics_list)
            raw_metrics[config_name] = metrics_list

    baseline_agg = aggregated.get("FULL", {})
    baseline_metrics = raw_metrics.get("FULL", [])

    # --- 生成可视化 ---
    figures_dir.mkdir(parents=True, exist_ok=True)
    try:
        generate_radar_chart(aggregated, figures_dir)
        generate_heatmap(aggregated, figures_dir)

        # 为关键指标生成箱线图
        for metric in ["novelty_score", "experiment_success_rate", "mean_confidence"]:
            distributions = {}
            for cfg, ml in raw_metrics.items():
                distributions[cfg] = [m.get(metric, 0) for m in ml]
            generate_box_plots(metric, distributions, figures_dir)

        # 计算组件边际贡献（MC = FULL - FULL_without_component）
        _generate_waterfall(aggregated, baseline_agg, figures_dir)
    except Exception as e:
        print(f"  Warning: Visualization generation partially failed: {e}")

    # --- 统计分析 ---
    stat_results = _run_statistical_analysis(raw_metrics, baseline_metrics)

    # --- 生成 Markdown ---
    lines = _build_markdown(aggregated, raw_metrics, baseline_agg, stat_results)

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"  Report written to {report_path}")


def _generate_waterfall(aggregated, baseline_agg, figures_dir):
    """计算并生成瀑布图"""
    # 组件映射: 变体名 -> 组件名
    component_map = {
        "A5-Vanilla-LLM": "Debate+GAN",
        "A1-No-Debate": "Multi-Expert Debate",
        "A2-No-GAN": "GAN Refinement",
        "B1-No-Risk": "Risk Assessment",
        "B3-No-Auto-Analysis": "Auto Analysis",
        "B5-No-Causal": "Causal Inference",
        "C1-No-RL": "RL Evolution",
        "C2-No-Skill": "Skill Solidify",
        "C3-No-Cross-Domain": "Cross-Domain",
        "C4-No-Literature": "Literature Mining",
        "C5-No-Memory": "Memory System",
    }

    # 综合性能分数 = mean(converged, novelty, ESR, confidence)
    def _perf_score(agg):
        metrics = ["converged", "novelty_score", "experiment_success_rate", "mean_confidence"]
        vals = [agg.get(m, {}).get("mean", 0) for m in metrics]
        return sum(vals) / len(vals) if vals else 0

    full_score = _perf_score(baseline_agg)

    # 找到 Vanilla 作为基线
    vanilla_agg = aggregated.get("A5-Vanilla-LLM", {})
    vanilla_score = _perf_score(vanilla_agg) if vanilla_agg else 0

    contributions = {}
    for variant, component in component_map.items():
        if variant in aggregated:
            variant_score = _perf_score(aggregated[variant])
            mc = full_score - variant_score
            contributions[component] = round(mc, 4)

    if contributions:
        generate_waterfall_chart(contributions, vanilla_score, figures_dir)


def _run_statistical_analysis(
    raw_metrics: Dict[str, List[dict]],
    baseline_metrics: List[dict],
) -> Dict[str, dict]:
    """对每个变体执行统计检验"""
    results = {}
    primary = ["novelty_score", "experiment_success_rate", "mean_confidence"]
    n_comparisons = len(raw_metrics) - 1  # 除 FULL 外

    for config_name, metrics_list in raw_metrics.items():
        if config_name == "FULL":
            continue

        config_stats = {}
        for metric in primary:
            full_vals = [m.get(metric, 0) for m in baseline_metrics]
            var_vals = [m.get(metric, 0) for m in metrics_list]
            t_stat, p_val = run_paired_ttest(full_vals, var_vals)
            d = compute_cohens_d(full_vals, var_vals)
            config_stats[metric] = {
                "t_stat": round(t_stat, 4),
                "p_value": round(p_val, 4),
                "cohens_d": round(d, 4),
                "significant": p_val < 0.05,
            }
        results[config_name] = config_stats

    # Bonferroni 校正
    if n_comparisons > 0:
        for metric in primary:
            p_vals = [results[c][metric]["p_value"]
                      for c in results if metric in results[c]]
            corrected = apply_bonferroni(p_vals, n_comparisons)
            for i, c in enumerate(results):
                if metric in results[c] and i < len(corrected):
                    results[c][metric]["p_corrected"] = round(corrected[i], 4)
                    results[c][metric]["significant_corrected"] = corrected[i] < 0.05

    return results


def _build_markdown(
    aggregated: Dict[str, dict],
    raw_metrics: Dict[str, List[dict]],
    baseline_agg: dict,
    stat_results: Dict[str, dict],
) -> List[str]:
    """构建 Markdown 报告"""
    lines = [
        "# Discovery-Agent 消融实验报告",
        "",
        "## 1. 实验元信息",
        "",
        f"- **生成日期**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"- **模型**: {os.getenv('AGENT_MODEL_NAME', 'deepseek-v4-pro')}",
        f"- **每配置重复次数**: {len(next(iter(raw_metrics.values()), []))}",
        f"- **最大迭代轮数**: 5",
        f"- **测试领域**: battery, catalysis, electronics",
        f"- **变体总数**: {len(aggregated)}",
        "",
        "> **注意**: 当前为预实验 (n=2)，样本量较小，统计结论仅供参考。",
        "",
        "---",
        "",
        "## 2. 主结果表",
        "",
        "| 变体 | 收敛率(CR) | 迭代数(ACI) | 新颖性(NS) | 成功率(ESR) | 置信度(MC) | 运行时间(s) |",
        "|------|-----------|------------|-----------|-----------|-----------|------------|",
    ]

    for config_name in aggregated:
        agg = aggregated[config_name]
        cr = compute_convergence_rate(raw_metrics.get(config_name, []))
        aci = compute_avg_convergence_iterations(raw_metrics.get(config_name, []))

        def _fmt(key):
            d = agg.get(key, {})
            m, s = d.get("mean", 0), d.get("std", 0)
            sig = ""
            if config_name != "FULL" and config_name in stat_results:
                if stat_results[config_name].get(key, {}).get("significant"):
                    sig = "*"
            return f"{m:.3f}+{s:.3f}{sig}"

        lines.append(
            f"| {config_name} | {cr:.2f} | {aci:.1f} | "
            f"{_fmt('novelty_score')} | {_fmt('experiment_success_rate')} | "
            f"{_fmt('mean_confidence')} | "
            f"{agg.get('total_runtime_seconds', {}).get('mean', 0):.0f} |"
        )

    lines.extend([
        "",
        "\\* 表示与 FULL 差异显著 (p < 0.05, 配对 t 检验)",
        "",
        "---",
        "",
    ])

    # 分组分析
    from ablation_runner import GROUPS
    for group_name, config_names in GROUPS.items():
        lines.extend([
            f"## 3.{group_name}. Group {group_name} 分析",
            "",
        ])
        for config_name in config_names:
            if config_name == "FULL" or config_name not in aggregated:
                continue
            comp = compare_to_baseline(aggregated[config_name], baseline_agg)
            key_changes = []
            for m in ["novelty_score", "experiment_success_rate", "mean_confidence"]:
                if m in comp:
                    pct = comp[m]["pct_change"]
                    direction = "+" if pct >= 0 else ""
                    key_changes.append(f"{m}: {direction}{pct:.1f}%")
            if key_changes:
                lines.append(f"- **{config_name}**: {', '.join(key_changes)}")
        lines.extend(["", "---", ""])

    # 组件贡献
    lines.extend([
        "## 4. 组件边际贡献分析",
        "",
        "边际贡献 MC(Component) = Performance(FULL) - Performance(FULL \\ Component)",
        "",
        "| 组件 | 对应变体 | MC (综合分数) |",
        "|------|---------|-------------|",
    ])

    component_map = {
        "Multi-Expert Debate": "A1-No-Debate",
        "GAN Refinement": "A2-No-GAN",
        "Risk Assessment": "B1-No-Risk",
        "Auto Analysis": "B3-No-Auto-Analysis",
        "Causal Inference": "B5-No-Causal",
        "RL Evolution": "C1-No-RL",
        "Skill Solidify": "C2-No-Skill",
        "Cross-Domain": "C3-No-Cross-Domain",
        "Literature Mining": "C4-No-Literature",
        "Memory System": "C5-No-Memory",
    }

    def _perf(agg):
        metrics = ["converged", "novelty_score", "experiment_success_rate", "mean_confidence"]
        vals = [agg.get(m, {}).get("mean", 0) for m in metrics]
        return sum(vals) / len(vals)

    full_perf = _perf(baseline_agg) if baseline_agg else 0.0
    for comp, variant in component_map.items():
        if variant in aggregated:
            var_perf = _perf(aggregated[variant])
            mc = full_perf - var_perf
            lines.append(f"| {comp} | {variant} | {mc:+.4f} |")

    lines.extend([
        "",
        "![Waterfall](figures/waterfall_contributions.png)",
        "",
        "---",
        "",
    ])

    # 统计细节
    lines.extend([
        "## 5. 统计检验详情",
        "",
        "| 变体 | 指标 | t-stat | p-value | p-corrected | Cohen's d | 显著 |",
        "|------|------|--------|---------|-------------|-----------|------|",
    ])
    for config_name, config_stats in stat_results.items():
        for metric, stats in config_stats.items():
            sig_mark = "Y" if stats.get("significant_corrected") else "N"
            lines.append(
                f"| {config_name} | {metric} | "
                f"{stats['t_stat']:.3f} | {stats['p_value']:.4f} | "
                f"{stats.get('p_corrected', 'N/A')} | "
                f"{stats['cohens_d']:.3f} | {sig_mark} |"
            )

    lines.extend([
        "",
        "---",
        "",
        "## 6. 局限性声明",
        "",
        "1. **样本量**: 当前为预实验 (n=2)，统计检验功效很低，结论仅为探索性",
        "2. **领域覆盖**: 仅测试 3 个领域，推广性有限",
        "3. **模型依赖**: 结果可能随 LLM 模型变化而不同",
        "4. **种子固定**: 使用固定种子问题可能引入偏差",
        "",
        "---",
        "",
        "## 7. 可视化图表",
        "",
        "![Radar](figures/radar_chart_all.png)",
        "",
        "![Heatmap](figures/heatmap_primary_metrics.png)",
        "",
        "![Novelty](figures/boxplot_novelty_score.png)",
        "",
        "![ESR](figures/boxplot_experiment_success_rate.png)",
        "",
        "![Confidence](figures/boxplot_mean_confidence.png)",
        "",
    ]
    )

    return lines
