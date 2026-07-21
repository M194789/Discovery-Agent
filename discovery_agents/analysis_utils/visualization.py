"""
可视化生成模块 - Visualization Generator

自动根据解析出的数值数据类型生成适当的图表。
遵循 generate_figures.py 的 matplotlib 配置模式。

依赖: matplotlib, numpy (项目已有), seaborn (可选增强)
"""

from __future__ import annotations

import os
from datetime import datetime
from typing import Dict, List, Optional

from .data_parser import ParsedNumericalData
from models.analysis_models import VisualizationArtifact

# 延迟导入，优雅降级
_matplotlib_available = True
_numpy_available = True
_seaborn_available = True

try:
    import matplotlib
    matplotlib.use('Agg')  # 非交互后端
    import matplotlib.pyplot as plt
except ImportError:
    plt = None  # type: ignore[assignment]
    _matplotlib_available = False

try:
    import numpy as np
except ImportError:
    np = None  # type: ignore[assignment]
    _numpy_available = False

try:
    import seaborn as sns
except ImportError:
    sns = None  # type: ignore[assignment]
    _seaborn_available = False


def _setup_matplotlib():
    """配置 matplotlib 全局参数（中文字体、DPI 等）。"""
    if not _matplotlib_available:
        return
    plt.rcParams['font.sans-serif'] = [
        'SimHei', 'Microsoft YaHei', 'DejaVu Sans'
    ]
    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams['figure.dpi'] = 150
    plt.rcParams['savefig.dpi'] = 300
    plt.rcParams['savefig.bbox'] = 'tight'


def _ensure_output_dir(output_dir: str) -> str:
    """确保输出目录存在，失败时降级到临时目录。"""
    try:
        os.makedirs(output_dir, exist_ok=True)
        return output_dir
    except OSError:
        import tempfile
        fallback = os.path.join(tempfile.gettempdir(), "discovery_figures")
        os.makedirs(fallback, exist_ok=True)
        return fallback


def _gen_filename(
    output_dir: str, hypothesis_id: str, chart_type: str
) -> str:
    """生成唯一的图表文件名。"""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{hypothesis_id}_{chart_type}_{ts}.png"
    return os.path.join(output_dir, filename)


# ---------------------------------------------------------------------------
# 图表生成函数
# ---------------------------------------------------------------------------


def _generate_metrics_bar_chart(
    metrics: Dict[str, float],
    hypothesis_id: str,
    output_dir: str,
) -> Optional[VisualizationArtifact]:
    """
    为单组指标生成条形图。
    当 metrics 中有 2+ 个指标时触发。
    """
    if len(metrics) < 2:
        return None

    # 过滤掉 CI 相关键
    filtered = {
        k: v for k, v in metrics.items()
        if not k.startswith("ci_")
    }
    if len(filtered) < 2:
        return None

    fig, ax = plt.subplots(1, 1, figsize=(max(8, len(filtered) * 1.2), 5))

    keys = list(filtered.keys())
    values = list(filtered.values())

    # 根据值的范围选择颜色
    colors = []
    for v in values:
        if 0 <= v <= 1:
            colors.append('#2ecc71' if v >= 0.7 else '#f39c12' if v >= 0.4 else '#e74c3c')
        else:
            colors.append('#3498db')

    bars = ax.bar(range(len(keys)), values, color=colors, alpha=0.85,
                  edgecolor='white', linewidth=1.5)

    # 值标签
    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height() + max(abs(v) for v in values) * 0.02,
            f'{val:.4g}',
            ha='center', va='bottom', fontsize=9, fontweight='bold',
        )

    ax.set_xticks(range(len(keys)))
    ax.set_xticklabels(keys, rotation=45, ha='right', fontsize=9)
    ax.set_ylabel('Value', fontsize=11)
    ax.set_title(
        f'Key Metrics - {hypothesis_id}',
        fontsize=13, fontweight='bold',
    )
    ax.grid(True, axis='y', alpha=0.3)

    filepath = _gen_filename(output_dir, hypothesis_id, "bar")
    plt.tight_layout()
    plt.savefig(filepath)
    plt.close(fig)

    return VisualizationArtifact(
        chart_type="bar",
        file_path=filepath,
        title=f"Key Metrics - {hypothesis_id}",
        description=f"Bar chart of {len(filtered)} extracted metrics: {', '.join(keys[:5])}",
    )


def _generate_distribution_plot(
    data: List[float],
    series_name: str,
    hypothesis_id: str,
    output_dir: str,
) -> Optional[VisualizationArtifact]:
    """
    为数值序列生成分布图（直方图 + KDE）。
    """
    if len(data) < 5:
        return None

    fig, ax = plt.subplots(1, 1, figsize=(8, 5))

    arr = np.array(data)

    if _seaborn_available:
        sns.histplot(arr, kde=True, ax=ax, color='#3498db', alpha=0.7)
    else:
        ax.hist(arr, bins='auto', color='#3498db', alpha=0.7,
                edgecolor='white', linewidth=1.0)

    # 标注均值和标准差
    mean_val = float(np.mean(arr))
    std_val = float(np.std(arr, ddof=1)) if len(arr) > 1 else 0
    ax.axvline(mean_val, color='#e74c3c', linestyle='--', linewidth=2,
               label=f'Mean={mean_val:.4g}')
    ax.axvline(mean_val + std_val, color='#f39c12', linestyle=':',
               linewidth=1.5, label=f'+1 SD={mean_val + std_val:.4g}')
    ax.axvline(mean_val - std_val, color='#f39c12', linestyle=':',
               linewidth=1.5, label=f'-1 SD={mean_val - std_val:.4g}')

    ax.set_xlabel(series_name, fontsize=11)
    ax.set_ylabel('Frequency', fontsize=11)
    ax.set_title(
        f'Distribution: {series_name} - {hypothesis_id}',
        fontsize=13, fontweight='bold',
    )
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    filepath = _gen_filename(output_dir, hypothesis_id, "distribution")
    plt.tight_layout()
    plt.savefig(filepath)
    plt.close(fig)

    return VisualizationArtifact(
        chart_type="distribution",
        file_path=filepath,
        title=f"Distribution: {series_name} - {hypothesis_id}",
        description=(
            f"Histogram with KDE for {series_name} "
            f"(n={len(data)}, mean={mean_val:.4g}, std={std_val:.4g})"
        ),
    )


def _generate_line_chart(
    values: List[float],
    series_name: str,
    hypothesis_id: str,
    output_dir: str,
) -> Optional[VisualizationArtifact]:
    """
    为时序数据生成折线图。
    """
    if len(values) < 3:
        return None

    fig, ax = plt.subplots(1, 1, figsize=(10, 5))

    x = list(range(1, len(values) + 1))
    ax.plot(x, values, 'o-', color='#3498db', linewidth=2, markersize=6)

    # 添加趋势线
    z = np.polyfit(x, values, 1)
    p_line = np.poly1d(z)
    ax.plot(x, p_line(x), '--', color='#e74c3c', linewidth=1.5,
            alpha=0.7, label=f'Trend (slope={z[0]:.4g})')

    ax.set_xlabel('Step', fontsize=11)
    ax.set_ylabel(series_name, fontsize=11)
    ax.set_title(
        f'Trend: {series_name} - {hypothesis_id}',
        fontsize=13, fontweight='bold',
    )
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    filepath = _gen_filename(output_dir, hypothesis_id, "line")
    plt.tight_layout()
    plt.savefig(filepath)
    plt.close(fig)

    return VisualizationArtifact(
        chart_type="line",
        file_path=filepath,
        title=f"Trend: {series_name} - {hypothesis_id}",
        description=(
            f"Line chart for {series_name} over {len(values)} steps "
            f"(trend slope={z[0]:.4g})"
        ),
    )


def _generate_correlation_heatmap(
    table: Dict[str, List[float]],
    hypothesis_id: str,
    output_dir: str,
) -> Optional[VisualizationArtifact]:
    """
    为多变量表格数据生成相关性热力图。
    至少需要 3 个变量。
    """
    cols = list(table.keys())
    if len(cols) < 3:
        return None

    # 确保所有列长度相同
    min_len = min(len(v) for v in table.values())
    if min_len < 3:
        return None

    data = np.array([table[c][:min_len] for c in cols])
    corr_matrix = np.corrcoef(data)

    fig, ax = plt.subplots(1, 1, figsize=(max(6, len(cols) * 1.2),
                                          max(5, len(cols) * 1.0)))

    if _seaborn_available:
        sns.heatmap(
            corr_matrix, annot=True, fmt='.3f',
            xticklabels=cols, yticklabels=cols,
            cmap='RdBu_r', center=0, vmin=-1, vmax=1,
            ax=ax, square=True,
        )
    else:
        im = ax.imshow(corr_matrix, cmap='RdBu_r', vmin=-1, vmax=1)
        ax.set_xticks(range(len(cols)))
        ax.set_yticks(range(len(cols)))
        ax.set_xticklabels(cols, rotation=45, ha='right')
        ax.set_yticklabels(cols)
        # 添加数值标注
        for i in range(len(cols)):
            for j in range(len(cols)):
                ax.text(j, i, f'{corr_matrix[i, j]:.3f}',
                        ha='center', va='center', fontsize=8)
        fig.colorbar(im, ax=ax)

    ax.set_title(
        f'Correlation Matrix - {hypothesis_id}',
        fontsize=13, fontweight='bold',
    )

    filepath = _gen_filename(output_dir, hypothesis_id, "heatmap")
    plt.tight_layout()
    plt.savefig(filepath)
    plt.close(fig)

    return VisualizationArtifact(
        chart_type="heatmap",
        file_path=filepath,
        title=f"Correlation Matrix - {hypothesis_id}",
        description=(
            f"Correlation heatmap of {len(cols)} variables: "
            f"{', '.join(cols[:5])}"
        ),
    )


def _generate_scatter_plot(
    x: List[float],
    y: List[float],
    label_x: str,
    label_y: str,
    hypothesis_id: str,
    output_dir: str,
) -> Optional[VisualizationArtifact]:
    """为两个变量生成散点图。"""
    if len(x) < 3 or len(y) < 3 or len(x) != len(y):
        return None

    fig, ax = plt.subplots(1, 1, figsize=(8, 6))

    ax.scatter(x, y, c='#3498db', alpha=0.7, edgecolors='white',
               linewidth=0.5, s=60)

    # 添加回归线
    z = np.polyfit(x, y, 1)
    p_line = np.poly1d(z)
    x_sorted = sorted(x)
    ax.plot(x_sorted, p_line(x_sorted), '--', color='#e74c3c',
            linewidth=1.5, alpha=0.7)

    # 计算 R²
    r_squared = float(np.corrcoef(x, y)[0, 1] ** 2)

    ax.set_xlabel(label_x, fontsize=11)
    ax.set_ylabel(label_y, fontsize=11)
    ax.set_title(
        f'{label_x} vs {label_y} - {hypothesis_id} (R²={r_squared:.4f})',
        fontsize=12, fontweight='bold',
    )
    ax.grid(True, alpha=0.3)

    filepath = _gen_filename(output_dir, hypothesis_id, "scatter")
    plt.tight_layout()
    plt.savefig(filepath)
    plt.close(fig)

    return VisualizationArtifact(
        chart_type="scatter",
        file_path=filepath,
        title=f"{label_x} vs {label_y} - {hypothesis_id}",
        description=(
            f"Scatter plot with regression line. R²={r_squared:.4f}, "
            f"n={len(x)} data points"
        ),
    )


# ---------------------------------------------------------------------------
# 主入口
# ---------------------------------------------------------------------------


def generate_visualizations(
    parsed_data: ParsedNumericalData,
    hypothesis_id: str,
    output_dir: str,
) -> List[VisualizationArtifact]:
    """
    根据解析出的数值数据自动生成图表。

    参数:
        parsed_data: 从实验输出中解析的结构化数值
        hypothesis_id: 假设ID（用于标题和文件名）
        output_dir: 图表输出目录

    返回:
        VisualizationArtifact 列表
    """
    if not _matplotlib_available or not _numpy_available:
        return []

    if parsed_data.parse_confidence == 0.0:
        return []

    _setup_matplotlib()
    output_dir = _ensure_output_dir(output_dir)

    artifacts: List[VisualizationArtifact] = []

    # 1. 指标条形图
    if len(parsed_data.metrics) >= 2:
        chart = _generate_metrics_bar_chart(
            parsed_data.metrics, hypothesis_id, output_dir
        )
        if chart:
            artifacts.append(chart)

    # 2. 时序折线图
    for name, values in parsed_data.time_series:
        chart = _generate_line_chart(
            values, name, hypothesis_id, output_dir
        )
        if chart:
            artifacts.append(chart)
        # 如果序列足够长，也生成分布图
        if len(values) >= 10:
            dist_chart = _generate_distribution_plot(
                values, name, hypothesis_id, output_dir
            )
            if dist_chart:
                artifacts.append(dist_chart)

    # 3. 表格数据的可视化
    for table in parsed_data.tables:
        cols = list(table.keys())
        vals = list(table.values())

        # 多变量热力图
        if len(cols) >= 3:
            chart = _generate_correlation_heatmap(
                table, hypothesis_id, output_dir
            )
            if chart:
                artifacts.append(chart)

        # 两变量散点图
        if len(cols) == 2 and len(vals[0]) >= 3:
            if len(vals[0]) == len(vals[1]):
                chart = _generate_scatter_plot(
                    vals[0], vals[1], cols[0], cols[1],
                    hypothesis_id, output_dir,
                )
                if chart:
                    artifacts.append(chart)

        # 表格中各列的分布图
        for col_name, col_vals in zip(cols, vals):
            if len(col_vals) >= 10:
                chart = _generate_distribution_plot(
                    col_vals, col_name, hypothesis_id, output_dir
                )
                if chart:
                    artifacts.append(chart)

    return artifacts
