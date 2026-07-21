"""
compute_ablation_stats.py
计算消融实验的推断性统计指标：
  - Mann-Whitney U 检验（单侧：FULL > ablated）
  - Holm-Bonferroni 多重检验校正
  - Hedges' g 效应量（小样本偏差修正）
  - FULL novelty 均值的 Bootstrap 95% CI（B=2000）

输出：
  - docs/tables/ablation_stats_inference.json  （原始数值供 xlsx/md 使用）
  - 控制台打印 Table1 追加列预览
"""

import json
import math
import numpy as np
from pathlib import Path
from scipy import stats

# ---------------------------------------------------------------------------
# 路径配置
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
ABL_DIR  = BASE_DIR / "ablation_formal"
OUT_JSON = BASE_DIR / "docs" / "tables" / "ablation_stats_inference.json"

METRIC   = "novelty_score"   # 主检验指标
B_BOOT   = 2000              # Bootstrap 重采样次数
SEED     = 42
RNG      = np.random.default_rng(SEED)

# ---------------------------------------------------------------------------
# 数据加载（FULL 仅取 GroupA）
# ---------------------------------------------------------------------------
def load_all() -> dict[str, list[float]]:
    data: dict[str, list[float]] = {}
    for group_dir in sorted(ABL_DIR.glob("Group*")):
        if not group_dir.is_dir():
            continue
        for fp in sorted(group_dir.glob("*.json")):
            if fp.name == "group_summary.json":
                continue
            try:
                d = json.loads(fp.read_text(encoding="utf-8"))
                cfg = d.get("config_name", "unknown")
                m   = d.get("metrics", {})
                v   = m.get(METRIC)
                if v is None:
                    continue
                # FULL 只取 GroupA
                if cfg == "FULL" and group_dir.name != "GroupA":
                    continue
                data.setdefault(cfg, []).append(float(v))
            except Exception:
                continue
    return data


# ---------------------------------------------------------------------------
# Hedges' g（偏差修正效应量）
# ---------------------------------------------------------------------------
def hedges_g(x1: list[float], x2: list[float]) -> float:
    """
    x1 = FULL（参考组），x2 = ablated（对比组）
    g > 0 表示 FULL 优于 ablated
    """
    n1, n2 = len(x1), len(x2)
    if n1 < 2 or n2 < 2:
        return float("nan")
    m1, m2 = np.mean(x1), np.mean(x2)
    s1, s2 = np.std(x1, ddof=1), np.std(x2, ddof=1)
    df = n1 + n2 - 2
    sp = math.sqrt(((n1 - 1) * s1**2 + (n2 - 1) * s2**2) / df)
    if sp == 0:
        return float("nan")
    d = (m1 - m2) / sp
    # 小样本修正因子 J
    J = 1 - 3 / (4 * df - 1)
    return d * J


# ---------------------------------------------------------------------------
# Holm-Bonferroni 校正
# ---------------------------------------------------------------------------
def holm_correct(pvals: list[float]) -> list[float]:
    """返回 Holm 校正后的 p 值列表（与输入顺序对应）"""
    n = len(pvals)
    order   = sorted(range(n), key=lambda i: pvals[i])
    adj     = [0.0] * n
    running = 0.0
    for rank, idx in enumerate(order):
        corrected = pvals[idx] * (n - rank)
        running   = max(running, corrected)
        adj[idx]  = min(running, 1.0)
    return adj


# ---------------------------------------------------------------------------
# Bootstrap CI for mean
# ---------------------------------------------------------------------------
def bootstrap_ci(values: list[float], ci: float = 0.95, b: int = B_BOOT):
    arr  = np.array(values)
    boot = RNG.choice(arr, size=(b, len(arr)), replace=True).mean(axis=1)
    lo   = float(np.percentile(boot, (1 - ci) / 2 * 100))
    hi   = float(np.percentile(boot, (1 + ci) / 2 * 100))
    return lo, hi


# ---------------------------------------------------------------------------
# 显著性星号
# ---------------------------------------------------------------------------
def sig_stars(p: float) -> str:
    if p < 0.001:
        return "***"
    elif p < 0.01:
        return "**"
    elif p < 0.05:
        return "*"
    else:
        return "ns"


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
def main():
    data = load_all()

    full_vals = data.get("FULL", [])
    print(f"FULL n = {len(full_vals)}")

    # Bootstrap CI for FULL
    ci_lo, ci_hi = bootstrap_ci(full_vals)
    full_mean = float(np.mean(full_vals))
    print(f"FULL novelty: {full_mean:.4f}  95% CI [{ci_lo:.4f}, {ci_hi:.4f}]")

    # 消融配置列表（排除 FULL）
    ablated_cfgs = sorted(k for k in data if k != "FULL")

    # Mann-Whitney U 单侧检验（H1: FULL > ablated）
    raw_pvals = []
    for cfg in ablated_cfgs:
        vals = data[cfg]
        stat, p = stats.mannwhitneyu(full_vals, vals, alternative="greater")
        raw_pvals.append(p)

    # Holm 校正
    adj_pvals = holm_correct(raw_pvals)

    # Hedges' g 和结果整理
    results = {}
    print(f"\n{'Config':<25} {'n':>3}  {'raw p':>7}  {'Holm p':>7}  {'sig':>4}  {'g':>6}")
    print("-" * 62)
    for i, cfg in enumerate(ablated_cfgs):
        vals = data[cfg]
        g    = hedges_g(full_vals, vals)
        raw_p  = raw_pvals[i]
        holm_p = adj_pvals[i]
        stars  = sig_stars(holm_p)
        print(f"{cfg:<25} {len(vals):>3}  {raw_p:>7.4f}  {holm_p:>7.4f}  {stars:>4}  {g:>6.3f}")
        results[cfg] = {
            "n":          len(vals),
            "raw_p":      round(raw_p,  4),
            "holm_p":     round(holm_p, 4),
            "sig":        stars,
            "hedges_g":   round(g, 3),
            "novelty_mean": round(float(np.mean(vals)), 4),
        }

    # 输出 JSON
    output = {
        "FULL": {
            "n":           len(full_vals),
            "novelty_mean": round(full_mean, 4),
            "bootstrap_ci_95": [round(ci_lo, 4), round(ci_hi, 4)],
            "holm_p":      "—",
            "sig":         "—",
            "hedges_g":    "—",
        },
        **results,
        "_meta": {
            "test":       "Mann-Whitney U (one-sided: FULL > ablated)",
            "correction": "Holm-Bonferroni",
            "effect_size":"Hedges' g",
            "bootstrap_b": B_BOOT,
            "bootstrap_seed": SEED,
            "metric":     METRIC,
        }
    }

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[OK] 结果已保存至 {OUT_JSON}")
    print(f"\nFULL Bootstrap 95% CI: [{ci_lo:.4f}, {ci_hi:.4f}]")


if __name__ == "__main__":
    main()
