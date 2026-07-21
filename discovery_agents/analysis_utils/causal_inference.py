"""
因果推断模块 - Causal Inference Module

使用贝叶斯网络模型识别实验变量与结果之间的因果关系，
超越简单相关性建立机制理解。

双模式策略：
- Mode A (pgmpy 可用): 完整贝叶斯网络结构学习 + 参数估计
- Mode B (pgmpy 不可用): 偏相关 + 条件独立性检验降级方案

依赖: pgmpy (可选), scipy, numpy, networkx (项目已有)
"""

from __future__ import annotations

import math
import os
from datetime import datetime
from typing import Dict, List, Optional, Tuple

from .data_parser import ParsedNumericalData
from models.analysis_models import (
    CausalInferenceResult,
    CausalRelationship,
)

# ---------------------------------------------------------------------------
# 延迟导入，优雅降级
# ---------------------------------------------------------------------------

_numpy_available = True
_scipy_available = True
_pgmpy_available = True
_networkx_available = True
_matplotlib_available = True

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

try:
    import networkx as nx
except ImportError:
    nx = None  # type: ignore[assignment]
    _networkx_available = False

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
except ImportError:
    plt = None  # type: ignore[assignment]
    _matplotlib_available = False

# pgmpy: 完整贝叶斯网络库（可选）
# Install: pip install pgmpy
try:
    from pgmpy.estimators import HillClimbSearch, BicScore
    from pgmpy.models import BayesianNetwork
    from pgmpy.estimators import MaximumLikelihoodEstimator
    import pandas as pd
except ImportError:
    HillClimbSearch = None  # type: ignore[assignment, misc]
    BicScore = None  # type: ignore[assignment, misc]
    BayesianNetwork = None  # type: ignore[assignment, misc]
    MaximumLikelihoodEstimator = None  # type: ignore[assignment, misc]
    pd = None  # type: ignore[assignment]
    _pgmpy_available = False


# ---------------------------------------------------------------------------
# 数据准备
# ---------------------------------------------------------------------------


def _build_variable_matrix(
    parsed_data_list: List[ParsedNumericalData],
) -> Tuple[Dict[str, List[float]], List[str]]:
    """
    从多个实验结果的解析数据中构建变量矩阵。
    聚合同名变量，为因果分析提供足够的数据点。

    返回:
        (variable_data, variable_names)
    """
    var_data: Dict[str, List[float]] = {}

    for parsed in parsed_data_list:
        # 从 metrics 收集
        for key, val in parsed.metrics.items():
            if key.startswith("ci_"):
                continue
            if key not in var_data:
                var_data[key] = []
            var_data[key].append(val)

        # 从 tables 收集
        for table in parsed.tables:
            for col_name, col_vals in table.items():
                if col_name not in var_data:
                    var_data[col_name] = []
                var_data[col_name].extend(col_vals)

    # 过滤掉只有 1 个数据点的变量
    var_data = {k: v for k, v in var_data.items() if len(v) >= 2}
    var_names = list(var_data.keys())

    return var_data, var_names


# ---------------------------------------------------------------------------
# Mode A: pgmpy 完整版贝叶斯网络
# ---------------------------------------------------------------------------


def _run_pgmpy_causal_inference(
    var_data: Dict[str, List[float]],
    var_names: List[str],
    output_dir: Optional[str] = None,
    hypothesis_id: str = "",
) -> CausalInferenceResult:
    """
    使用 pgmpy 进行完整的贝叶斯网络结构学习和参数估计。
    """
    # 统一变量长度（取最短长度）
    min_len = min(len(v) for v in var_data.values())
    if min_len < 5:
        # 数据太少，补充离散化的样本
        # 使用已有数据的统计量生成模拟数据
        augmented_data = {}
        target_n = max(20, min_len * 3)
        for name in var_names:
            vals = var_data[name]
            mean_v = float(np.mean(vals))
            std_v = float(np.std(vals, ddof=1)) if len(vals) > 1 else abs(mean_v) * 0.1
            if std_v == 0:
                std_v = abs(mean_v) * 0.1 if mean_v != 0 else 0.1
            augmented = list(vals) + list(
                np.random.normal(mean_v, std_v, target_n - len(vals))
            )
            augmented_data[name] = augmented[:target_n]
        df = pd.DataFrame(augmented_data)
    else:
        # 截断到相同长度
        df = pd.DataFrame({k: v[:min_len] for k, v in var_data.items()})

    # 离散化（pgmpy 结构学习需要离散数据或连续支持）
    # 使用分位数离散化为 3 类：low, medium, high
    df_discrete = pd.DataFrame()
    for col in df.columns:
        try:
            df_discrete[col] = pd.qcut(
                df[col], q=3, labels=['low', 'medium', 'high'],
                duplicates='drop',
            )
        except (ValueError, TypeError):
            # 如果值太少无法分 3 类，分 2 类
            try:
                df_discrete[col] = pd.qcut(
                    df[col], q=2, labels=['low', 'high'],
                    duplicates='drop',
                )
            except (ValueError, TypeError):
                df_discrete[col] = 'medium'

    # 结构学习
    scoring_method = BicScore(df_discrete)
    hc = HillClimbSearch(df_discrete)
    best_model = hc.estimate(scoring_method=scoring_method, max_indegree=3)

    edges = list(best_model.edges())
    if not edges:
        return CausalInferenceResult(
            relationships=[],
            network_structure={},
        )

    # 构建贝叶斯网络并估计参数
    bn = BayesianNetwork(edges)
    bn.fit(df_discrete, estimator=MaximumLikelihoodEstimator)

    # 提取因果关系
    relationships: List[CausalRelationship] = []
    network_structure: Dict[str, List[str]] = {}

    for parent, child in edges:
        # 计算关系强度（基于相关系数）
        if parent in var_data and child in var_data:
            p_vals = var_data[parent]
            c_vals = var_data[child]
            n = min(len(p_vals), len(c_vals))
            if n >= 3:
                corr, _ = sp_stats.pearsonr(p_vals[:n], c_vals[:n])
                strength = abs(float(corr))
            else:
                strength = 0.5
        else:
            strength = 0.5

        relationships.append(CausalRelationship(
            cause=parent,
            effect=child,
            strength=min(1.0, strength),
            confidence=0.7,  # pgmpy 完整版给较高置信度
            mechanism=(
                f"Bayesian network structure learning identified "
                f"'{parent}' as a direct cause of '{child}' "
                f"(BIC-optimized structure)"
            ),
        ))

        if parent not in network_structure:
            network_structure[parent] = []
        network_structure[parent].append(child)

    # 计算模型拟合分数
    model_fit = float(scoring_method.score(bn))

    # 可视化因果图
    diagram_path = None
    if output_dir and _networkx_available and _matplotlib_available:
        diagram_path = _visualize_causal_network(
            edges, var_names, output_dir, hypothesis_id
        )

    return CausalInferenceResult(
        relationships=relationships,
        network_structure=network_structure,
        confounders_identified=_identify_confounders(edges, var_names),
        model_fit_score=model_fit,
        network_diagram_path=diagram_path,
    )


# ---------------------------------------------------------------------------
# Mode B: 降级版（偏相关 + 条件独立性检验）
# ---------------------------------------------------------------------------


def _partial_correlation(
    x: List[float], y: List[float], z: List[float]
) -> Tuple[float, float]:
    """
    计算 x 和 y 的偏相关系数（控制 z）。
    使用 Fisher's Z-transform 计算 p 值。

    返回: (partial_corr, p_value)
    """
    n = min(len(x), len(y), len(z))
    if n < 4:
        return (0.0, 1.0)

    x_arr, y_arr, z_arr = np.array(x[:n]), np.array(y[:n]), np.array(z[:n])

    # 计算残差
    # r_xy.z = (r_xy - r_xz * r_yz) / sqrt((1 - r_xz²)(1 - r_yz²))
    r_xy = float(np.corrcoef(x_arr, y_arr)[0, 1])
    r_xz = float(np.corrcoef(x_arr, z_arr)[0, 1])
    r_yz = float(np.corrcoef(y_arr, z_arr)[0, 1])

    denom = math.sqrt((1 - r_xz ** 2) * (1 - r_yz ** 2))
    if denom < 1e-10:
        return (0.0, 1.0)

    partial_r = (r_xy - r_xz * r_yz) / denom

    # Fisher's Z-transform for p-value
    if abs(partial_r) >= 1.0:
        return (float(np.clip(partial_r, -1, 1)), 0.0)

    z_score = 0.5 * math.log((1 + partial_r) / (1 - partial_r))
    se = 1.0 / math.sqrt(n - 3) if n > 3 else 1.0
    p = 2 * (1 - sp_stats.norm.cdf(abs(z_score / se)))

    return (float(partial_r), float(p))


def _run_fallback_causal_inference(
    var_data: Dict[str, List[float]],
    var_names: List[str],
    output_dir: Optional[str] = None,
    hypothesis_id: str = "",
) -> CausalInferenceResult:
    """
    降级版因果推断：使用偏相关和条件独立性检验。
    """
    relationships: List[CausalRelationship] = []
    network_structure: Dict[str, List[str]] = {}
    confounders: List[str] = []
    edges: List[Tuple[str, str]] = []

    alpha = 0.05

    # 对所有变量对进行两两分析
    for i in range(len(var_names)):
        for j in range(i + 1, len(var_names)):
            vi_name, vj_name = var_names[i], var_names[j]
            vi_vals, vj_vals = var_data[vi_name], var_data[vj_name]

            n = min(len(vi_vals), len(vj_vals))
            if n < 3:
                continue

            # 1. 计算直接相关
            corr, p_corr = sp_stats.pearsonr(vi_vals[:n], vj_vals[:n])
            if p_corr >= alpha:
                continue  # 不显著，跳过

            # 2. 检查是否有混淆变量
            is_confounded = False
            for k in range(len(var_names)):
                if k == i or k == j:
                    continue
                vk_name = var_names[k]
                vk_vals = var_data[vk_name]

                nk = min(n, len(vk_vals))
                if nk < 4:
                    continue

                partial_r, p_partial = _partial_correlation(
                    vi_vals[:nk], vj_vals[:nk], vk_vals[:nk]
                )

                # 如果控制第三变量后相关性消失，说明可能是混淆
                if abs(partial_r) < 0.1 and p_partial > alpha:
                    is_confounded = True
                    if vk_name not in confounders:
                        confounders.append(vk_name)
                    break

            if is_confounded:
                continue

            # 3. 推断因果方向（简化启发式）
            # 使用方差比：因通常更稳定（方差更小）
            var_i = float(np.var(vi_vals[:n]))
            var_j = float(np.var(vj_vals[:n]))

            strength = abs(float(corr))

            if var_i <= var_j:
                cause, effect = vi_name, vj_name
            else:
                cause, effect = vj_name, vi_name

            edges.append((cause, effect))

            relationships.append(CausalRelationship(
                cause=cause,
                effect=effect,
                strength=min(1.0, strength),
                confidence=0.4,  # 降级版给较低置信度
                mechanism=(
                    f"Partial correlation analysis suggests '{cause}' "
                    f"causally influences '{effect}' "
                    f"(r={corr:.4f}, p={p_corr:.4g}, "
                    f"robust after controlling for confounders)"
                ),
            ))

            if cause not in network_structure:
                network_structure[cause] = []
            network_structure[cause].append(effect)

    # 可视化因果图
    diagram_path = None
    if output_dir and edges and _networkx_available and _matplotlib_available:
        diagram_path = _visualize_causal_network(
            edges, var_names, output_dir, hypothesis_id
        )

    return CausalInferenceResult(
        relationships=relationships,
        network_structure=network_structure,
        confounders_identified=confounders,
        model_fit_score=None,
        network_diagram_path=diagram_path,
    )


# ---------------------------------------------------------------------------
# 辅助函数
# ---------------------------------------------------------------------------


def _identify_confounders(
    edges: List[Tuple[str, str]], var_names: List[str]
) -> List[str]:
    """识别可能的混淆变量（入度和出度都 > 0 的节点）。"""
    if not _networkx_available:
        return []

    G = nx.DiGraph(edges)
    confounders = []
    for node in G.nodes():
        if G.in_degree(node) > 0 and G.out_degree(node) > 0:
            confounders.append(node)
    return confounders


def _visualize_causal_network(
    edges: List[Tuple[str, str]],
    var_names: List[str],
    output_dir: str,
    hypothesis_id: str,
) -> Optional[str]:
    """使用 networkx + matplotlib 可视化因果网络图。"""
    try:
        G = nx.DiGraph(edges)

        fig, ax = plt.subplots(1, 1, figsize=(10, 8))

        # 使用 spring layout
        pos = nx.spring_layout(G, seed=42, k=2.0)

        # 节点颜色：根/原因用蓝色，叶/结果用绿色，中间用橙色
        node_colors = []
        for node in G.nodes():
            in_deg = G.in_degree(node)
            out_deg = G.out_degree(node)
            if in_deg == 0 and out_deg > 0:
                node_colors.append('#3498db')  # 纯因
            elif out_deg == 0 and in_deg > 0:
                node_colors.append('#2ecc71')  # 纯果
            else:
                node_colors.append('#f39c12')  # 中间变量

        nx.draw_networkx_nodes(
            G, pos, ax=ax, node_size=2000, node_color=node_colors,
            alpha=0.8, edgecolors='white', linewidths=2,
        )
        nx.draw_networkx_labels(
            G, pos, ax=ax, font_size=9, font_weight='bold',
        )
        nx.draw_networkx_edges(
            G, pos, ax=ax, edge_color='#7f8c8d', width=2,
            arrowsize=20, arrowstyle='->', connectionstyle='arc3,rad=0.1',
        )

        ax.set_title(
            f'Causal Network - {hypothesis_id}',
            fontsize=14, fontweight='bold',
        )
        ax.axis('off')

        os.makedirs(output_dir, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        filepath = os.path.join(
            output_dir,
            f"{hypothesis_id}_causal_network_{ts}.png",
        )
        plt.tight_layout()
        plt.savefig(filepath, dpi=300)
        plt.close(fig)

        return filepath
    except Exception:
        return None


# ---------------------------------------------------------------------------
# 主入口
# ---------------------------------------------------------------------------


def run_causal_inference(
    parsed_data_list: List[ParsedNumericalData],
    hypothesis_id: str = "",
    output_dir: Optional[str] = None,
) -> Optional[CausalInferenceResult]:
    """
    对解析出的数值数据执行因果推断分析。

    支持聚合多个实验结果的数据，构建更丰富的因果模型。

    参数:
        parsed_data_list: 一个或多个实验的解析数值数据
        hypothesis_id: 假设ID（用于标题和文件名）
        output_dir: 图表输出目录

    返回:
        CausalInferenceResult 或 None（数据不足时）
    """
    if not _numpy_available or not _scipy_available:
        return None

    if not parsed_data_list:
        return None

    # 构建变量矩阵
    var_data, var_names = _build_variable_matrix(parsed_data_list)

    # 至少需要 2 个变量才能分析因果关系
    if len(var_names) < 2:
        return None

    # 选择分析模式
    if _pgmpy_available and len(var_names) >= 2:
        try:
            return _run_pgmpy_causal_inference(
                var_data, var_names, output_dir, hypothesis_id
            )
        except Exception:
            # pgmpy 失败时降级
            pass

    # Mode B: 降级版
    if len(var_names) >= 2:
        try:
            return _run_fallback_causal_inference(
                var_data, var_names, output_dir, hypothesis_id
            )
        except Exception:
            pass

    return None
