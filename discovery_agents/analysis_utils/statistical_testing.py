"""
统计检验模块 - Statistical Testing Module

对实验结果中的数值数据执行假设检验、置信区间计算和显著性分析。
自动根据数据特征选择合适的统计检验方法。

依赖: scipy.stats, numpy (项目已有)
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

from .data_parser import ParsedNumericalData

# 延迟导入，优雅降级
_scipy_available = True
_numpy_available = True

try:
    import numpy as np
except ImportError:
    np = None  # type: ignore[assignment]
    _numpy_available = False

try:
    from scipy import stats as sp_stats
except ImportError:
    sp_stats = None  # type: ignore[assignment]
    _scipy_available = False

from models.analysis_models import StatisticalTestResult


# ---------------------------------------------------------------------------
# 工具函数
# ---------------------------------------------------------------------------

_ALPHA = 0.05  # 默认显著性水平


def _cohens_d(group1: List[float], group2: List[float]) -> float:
    """计算 Cohen's d 效应量。"""
    n1, n2 = len(group1), len(group2)
    if n1 < 2 or n2 < 2:
        return 0.0
    m1, m2 = np.mean(group1), np.mean(group2)
    s1, s2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
    pooled_std = math.sqrt(((n1 - 1) * s1 + (n2 - 1) * s2) / (n1 + n2 - 2))
    if pooled_std == 0:
        return 0.0
    return float(abs(m1 - m2) / pooled_std)


def _cohens_d_one_sample(values: List[float], mu0: float) -> float:
    """计算单样本 Cohen's d。"""
    if len(values) < 2:
        return 0.0
    m = np.mean(values)
    s = np.std(values, ddof=1)
    if s == 0:
        return 0.0
    return float(abs(m - mu0) / s)


def _bootstrap_ci(
    values: List[float], n_bootstrap: int = 1000, ci: float = 0.95
) -> Tuple[float, float]:
    """Bootstrap 法计算置信区间（适用于小样本）。"""
    arr = np.array(values)
    n = len(arr)
    means = np.array([
        np.mean(np.random.choice(arr, size=n, replace=True))
        for _ in range(n_bootstrap)
    ])
    alpha_half = (1 - ci) / 2
    lo = float(np.percentile(means, alpha_half * 100))
    hi = float(np.percentile(means, (1 - alpha_half) * 100))
    return (lo, hi)


def _is_normal(values: List[float]) -> bool:
    """Shapiro-Wilk 正态性检验（n >= 3）。"""
    if len(values) < 3 or len(values) > 5000:
        return True  # 样本太小/大时默认假设正态
    _, p = sp_stats.shapiro(values)
    return p > _ALPHA


# ---------------------------------------------------------------------------
# 单项检验函数
# ---------------------------------------------------------------------------


def _one_sample_test(
    values: List[float], label: str, baseline: float = 0.0
) -> Optional[StatisticalTestResult]:
    """单样本 t 检验：检验均值是否显著不等于基线值。"""
    if len(values) < 2:
        return None

    arr = np.array(values)
    mean_val = float(np.mean(arr))

    # 选择参数/非参数检验
    if _is_normal(values):
        stat, p = sp_stats.ttest_1samp(arr, baseline)
        test_name = "one_sample_t_test"
    else:
        # Wilcoxon signed-rank (非参数替代)
        shifted = arr - baseline
        # 需要非零差值
        nonzero = shifted[shifted != 0]
        if len(nonzero) < 2:
            return None
        stat, p = sp_stats.wilcoxon(nonzero)
        test_name = "wilcoxon_signed_rank"

    stat = float(stat)
    p = float(p)

    # 置信区间
    if len(values) < 30:
        ci = _bootstrap_ci(values)
    else:
        se = float(np.std(arr, ddof=1) / math.sqrt(len(arr)))
        ci = (mean_val - 1.96 * se, mean_val + 1.96 * se)

    effect = _cohens_d_one_sample(values, baseline)

    return StatisticalTestResult(
        test_name=test_name,
        test_statistic=stat,
        p_value=p,
        is_significant=p < _ALPHA,
        confidence_interval=list(ci),
        effect_size=effect,
        interpretation=(
            f"{label}: mean={mean_val:.4f} vs baseline={baseline}. "
            f"{'Statistically significant' if p < _ALPHA else 'Not significant'} "
            f"(p={p:.4g}, effect size d={effect:.3f})"
        ),
    )


def _two_sample_test(
    group1: List[float],
    group2: List[float],
    label1: str,
    label2: str,
) -> Optional[StatisticalTestResult]:
    """双样本检验：比较两组数据的均值差异。"""
    if len(group1) < 2 or len(group2) < 2:
        return None

    normal1 = _is_normal(group1)
    normal2 = _is_normal(group2)

    if normal1 and normal2:
        stat, p = sp_stats.ttest_ind(group1, group2, equal_var=False)
        test_name = "welch_t_test"
    else:
        stat, p = sp_stats.mannwhitneyu(
            group1, group2, alternative='two-sided'
        )
        test_name = "mann_whitney_u"

    stat = float(stat)
    p = float(p)
    effect = _cohens_d(group1, group2)
    m1, m2 = float(np.mean(group1)), float(np.mean(group2))

    return StatisticalTestResult(
        test_name=test_name,
        test_statistic=stat,
        p_value=p,
        is_significant=p < _ALPHA,
        effect_size=effect,
        interpretation=(
            f"{label1} (mean={m1:.4f}) vs {label2} (mean={m2:.4f}). "
            f"{'Statistically significant difference' if p < _ALPHA else 'No significant difference'} "
            f"(p={p:.4g}, d={effect:.3f})"
        ),
    )


def _correlation_test(
    x: List[float], y: List[float], label_x: str, label_y: str
) -> Optional[StatisticalTestResult]:
    """相关性检验：检验两变量之间是否存在显著相关。"""
    if len(x) < 3 or len(y) < 3 or len(x) != len(y):
        return None

    # 选择 Pearson 或 Spearman
    normal_x = _is_normal(x)
    normal_y = _is_normal(y)

    if normal_x and normal_y:
        stat, p = sp_stats.pearsonr(x, y)
        test_name = "pearson_correlation"
    else:
        stat, p = sp_stats.spearmanr(x, y)
        test_name = "spearman_correlation"

    stat = float(stat)
    p = float(p)

    strength = "strong" if abs(stat) > 0.7 else "moderate" if abs(stat) > 0.4 else "weak"
    direction = "positive" if stat > 0 else "negative"

    return StatisticalTestResult(
        test_name=test_name,
        test_statistic=stat,
        p_value=p,
        is_significant=p < _ALPHA,
        effect_size=float(stat ** 2),  # r²
        interpretation=(
            f"{label_x} vs {label_y}: {strength} {direction} correlation "
            f"(r={stat:.4f}, p={p:.4g}, r²={stat**2:.4f})"
        ),
    )


def _metric_significance_test(
    value: float, label: str
) -> Optional[StatisticalTestResult]:
    """
    单值显著性检验：当只有单个指标值时，
    检验其是否显著偏离常见基线（如 0.5 随机基线）。
    """
    # 对于 0-1 范围的指标（如 accuracy, R², AUC），用 0.5 作基线
    if 0.0 <= value <= 1.0:
        baseline = 0.5
        # 使用二项检验近似
        # 假设这是一个比例，用正态近似
        se = math.sqrt(baseline * (1 - baseline) / 100)  # 假设 n=100
        z = (value - baseline) / se if se > 0 else 0.0
        p = 2 * (1 - sp_stats.norm.cdf(abs(z)))  # 双侧
        return StatisticalTestResult(
            test_name="z_test_vs_baseline",
            test_statistic=float(z),
            p_value=float(p),
            is_significant=p < _ALPHA,
            confidence_interval=[
                value - 1.96 * se, value + 1.96 * se
            ],
            interpretation=(
                f"{label}={value:.4f} vs random baseline {baseline}. "
                f"{'Statistically significant' if p < _ALPHA else 'Not significant'} "
                f"(z={z:.4f}, p={p:.4g})"
            ),
        )
    return None


# ---------------------------------------------------------------------------
# 主入口
# ---------------------------------------------------------------------------


def run_statistical_tests(
    parsed_data: ParsedNumericalData,
    hypothesis_statement: str = "",
) -> List[StatisticalTestResult]:
    """
    对解析出的数值数据自动执行统计检验。

    参数:
        parsed_data: 从实验输出中解析的结构化数值
        hypothesis_statement: 假设陈述（用于上下文参考）

    返回:
        StatisticalTestResult 列表
    """
    if not _scipy_available or not _numpy_available:
        return []

    if parsed_data.parse_confidence == 0.0:
        return []

    results: List[StatisticalTestResult] = []

    # --- 1. 对 metrics 中的单个指标执行单值检验 ---
    for key, value in parsed_data.metrics.items():
        # 跳过置信区间本身
        if key.startswith("ci_"):
            continue
        test = _metric_significance_test(value, key)
        if test:
            results.append(test)

    # --- 2. 对 table 数据执行比较检验 ---
    for table in parsed_data.tables:
        columns = list(table.keys())
        column_values = list(table.values())

        # 如果只有一列数据，执行单样本检验
        if len(columns) == 1 and len(column_values[0]) >= 2:
            test = _one_sample_test(column_values[0], columns[0])
            if test:
                results.append(test)

        # 两列数据：执行双样本检验 + 相关性检验
        elif len(columns) == 2:
            v1, v2 = column_values[0], column_values[1]
            if len(v1) >= 2 and len(v2) >= 2:
                # 双样本比较
                if len(v1) == len(v2):
                    test = _two_sample_test(v1, v2, columns[0], columns[1])
                    if test:
                        results.append(test)
                    # 相关性
                    corr = _correlation_test(v1, v2, columns[0], columns[1])
                    if corr:
                        results.append(corr)

        # 多列数据：执行所有两两相关性检验
        elif len(columns) >= 3:
            for i in range(len(columns)):
                for j in range(i + 1, len(columns)):
                    vi, vj = column_values[i], column_values[j]
                    if len(vi) >= 3 and len(vj) >= 3 and len(vi) == len(vj):
                        corr = _correlation_test(
                            vi, vj, columns[i], columns[j]
                        )
                        if corr:
                            results.append(corr)

    # --- 3. 对 time_series 数据执行趋势检验 ---
    for name, values in parsed_data.time_series:
        if len(values) >= 4:
            # 分前后两半，检验是否有改善
            mid = len(values) // 2
            first_half = values[:mid]
            second_half = values[mid:]
            test = _two_sample_test(
                first_half, second_half,
                f"{name}_early", f"{name}_late"
            )
            if test:
                results.append(test)

    # --- 4. 对 metrics 中的多个指标做两两相关性 ---
    metric_keys = [
        k for k, v in parsed_data.metrics.items()
        if not k.startswith("ci_")
    ]
    if len(metric_keys) >= 3 and parsed_data.tables:
        # 尝试从表格中找到对应列进行相关性分析
        for table in parsed_data.tables:
            cols = list(table.keys())
            vals = list(table.values())
            for i in range(len(cols)):
                for j in range(i + 1, len(cols)):
                    if len(vals[i]) >= 3 and len(vals[j]) >= 3:
                        if len(vals[i]) == len(vals[j]):
                            corr = _correlation_test(
                                vals[i], vals[j], cols[i], cols[j]
                            )
                            if corr and not any(
                                r.test_name == corr.test_name
                                and r.interpretation == corr.interpretation
                                for r in results
                            ):
                                results.append(corr)

    return results
