"""
消融实验指标提取模块

从 DiscoveryState.model_dump() 字典中提取主要、次要和资源效率指标，
用于消融实验的定量分析和报告生成。
"""

from __future__ import annotations

import math
from typing import Dict, List, Any, Optional


# ---------------------------------------------------------------------------
# 单次实验指标提取
# ---------------------------------------------------------------------------

def extract_metrics(state_dict: dict, runtime_seconds: float = 0.0) -> dict:
    """
    从单次实验的 DiscoveryState dump 中提取所有指标。

    参数:
        state_dict: DiscoveryState.model_dump() 的结果
        runtime_seconds: 外部计时的总运行时间（秒）

    返回:
        包含所有指标的平面字典
    """
    iterations = state_dict.get("iterations", [])

    return {
        # 主要指标
        "converged": state_dict.get("converged", False),
        "iterations_used": state_dict.get("current_iteration", 0),
        "novelty_score": state_dict.get("overall_novelty_score", 0.0),
        "experiment_success_rate": _compute_experiment_success_rate(iterations),
        "mean_confidence": _compute_mean_confidence(iterations),
        # 次要指标
        "total_hypotheses": _compute_hypothesis_count(iterations),
        "plan_filter_rate": _compute_plan_filter_rate(iterations),
        "statistical_significance_rate": _compute_stat_sig_rate(iterations),
        "visualization_count": _compute_visualization_count(iterations),
        "causal_edge_count": _compute_causal_edge_count(iterations),
        # 资源效率指标
        "total_runtime_seconds": runtime_seconds,
        "code_execution_failure_rate": _compute_code_failure_rate(iterations),
    }


# ---------------------------------------------------------------------------
# 主要指标计算
# ---------------------------------------------------------------------------

def _compute_experiment_success_rate(iterations: List[dict]) -> float:
    """实验成功率: 成功实验数 / 总实验数"""
    total, success = 0, 0
    for it in iterations:
        for r in it.get("results", []):
            total += 1
            if r.get("success"):
                success += 1
    return success / total if total > 0 else 0.0


def _compute_mean_confidence(iterations: List[dict]) -> float:
    """平均置信度: 所有分析结果 confidence 的均值"""
    confidences = []
    for it in iterations:
        for a in it.get("analysis", []):
            c = a.get("confidence")
            if c is not None:
                confidences.append(float(c))
    return sum(confidences) / len(confidences) if confidences else 0.0


# ---------------------------------------------------------------------------
# 次要指标计算
# ---------------------------------------------------------------------------

def _compute_hypothesis_count(iterations: List[dict]) -> int:
    """总假设数: 所有迭代中出现的唯一假设 ID 数"""
    ids = set()
    for it in iterations:
        for h in it.get("hypotheses", []):
            hid = h.get("id", "")
            if hid:
                ids.add(hid)
    return len(ids)


def _compute_plan_filter_rate(iterations: List[dict]) -> float:
    """计划过滤率: 被风险评估过滤的计划比率"""
    total_plans = 0
    filtered = 0
    for it in iterations:
        plans = it.get("plans", [])
        risk_assessments = it.get("risk_assessments", [])
        total_plans += len(plans)
        for ra in risk_assessments:
            if isinstance(ra, dict) and not ra.get("is_feasible", True):
                filtered += 1
    return filtered / total_plans if total_plans > 0 else 0.0


def _compute_stat_sig_rate(iterations: List[dict]) -> float:
    """统计检验显著率"""
    total, sig = 0, 0
    for it in iterations:
        for a in it.get("analysis", []):
            for t in a.get("statistical_tests", []):
                total += 1
                if t.get("is_significant"):
                    sig += 1
    return sig / total if total > 0 else 0.0


def _compute_visualization_count(iterations: List[dict]) -> int:
    """可视化生成总数"""
    count = 0
    for it in iterations:
        for a in it.get("analysis", []):
            count += len(a.get("visualizations", []))
    return count


def _compute_causal_edge_count(iterations: List[dict]) -> int:
    """因果推断边总数"""
    count = 0
    for it in iterations:
        for a in it.get("analysis", []):
            ci = a.get("causal_inference")
            if ci and isinstance(ci, dict):
                count += len(ci.get("relationships", []))
    return count


# ---------------------------------------------------------------------------
# 资源效率指标
# ---------------------------------------------------------------------------

def _compute_code_failure_rate(iterations: List[dict]) -> float:
    """代码执行失败率: 1 - 成功率"""
    sr = _compute_experiment_success_rate(iterations)
    return 1.0 - sr


# ---------------------------------------------------------------------------
# 跨 run 聚合
# ---------------------------------------------------------------------------

def aggregate_metrics(metric_list: List[dict]) -> dict:
    """
    对多次 run 的指标计算 mean ± std。

    参数:
        metric_list: 多次 extract_metrics() 的返回值列表

    返回:
        {metric_name: {"mean": float, "std": float}, ...}
    """
    if not metric_list:
        return {}

    keys = [k for k in metric_list[0] if isinstance(metric_list[0][k], (int, float, bool))]
    result = {}
    for key in keys:
        values = [float(m.get(key, 0)) for m in metric_list]
        n = len(values)
        mean = sum(values) / n
        variance = sum((v - mean) ** 2 for v in values) / n if n > 1 else 0.0
        std = math.sqrt(variance)
        result[key] = {"mean": round(mean, 4), "std": round(std, 4)}
    return result


def compute_convergence_rate(metric_list: List[dict]) -> float:
    """收敛率: converged=True 的比率"""
    if not metric_list:
        return 0.0
    return sum(1 for m in metric_list if m.get("converged")) / len(metric_list)


def compute_avg_convergence_iterations(metric_list: List[dict]) -> float:
    """平均收敛迭代数: 仅计算 converged=True 的 runs"""
    converged = [m["iterations_used"] for m in metric_list if m.get("converged")]
    return sum(converged) / len(converged) if converged else 0.0


def compare_to_baseline(variant_agg: dict, baseline_agg: dict) -> dict:
    """
    计算变体相对于基线的差异（delta 和百分比变化）。

    参数:
        variant_agg: aggregate_metrics() 的结果（变体）
        baseline_agg: aggregate_metrics() 的结果（FULL 基线）

    返回:
        {metric: {"delta": float, "pct_change": float}, ...}
    """
    result = {}
    for key in baseline_agg:
        if key not in variant_agg:
            continue
        base_mean = baseline_agg[key]["mean"]
        var_mean = variant_agg[key]["mean"]
        delta = var_mean - base_mean
        pct = (delta / base_mean * 100) if base_mean != 0 else 0.0
        result[key] = {"delta": round(delta, 4), "pct_change": round(pct, 2)}
    return result
