# DISCOVERY-AGENT 消融实验设计方案

## 1. 实验目标

系统评估 DISCOVERY-AGENT 各核心模块对自主科学发现性能的贡献，量化以下组件的独立价值和协同效应：
- 多专家辩论机制 (Multi-Expert Debate)
- GAN 风格对抗优化 (GAN Adversarial Refinement)
- 实验风险评估 (Risk Assessment)
- 自动数据分析增强 (Statistical + Visualization + Causal)
- RL 指导的假设演化 (RL-Guided Evolution)
- 技能固化与复用 (Skill Solidification)
- 文献挖掘增强 (Literature Mining)
- 跨域技能迁移 (Cross-Domain Transfer)

## 2. 实验基准配置

### 2.1 测试领域

| 领域 | 代表任务 | 复杂度 |
|------|---------|--------|
| 电池材料 (battery) | 发现高能量密度电极材料 | 高 |
| 催化 (catalysis) | 优化催化剂选择性 | 高 |
| 电子信息 (electronics) | 设计低功耗半导体 | 中 |

### 2.2 基础超参数

```python
max_iterations = 5           # 每轮实验最大迭代次数
model_name = "deepseek-v4-pro" # 统一使用 DeepSeek 模型
num_runs_per_config = 5     # 每个配置重复运行次数
seed_problems = [            # 固定种子问题集合
    "How to improve energy density of Li-ion batteries?",
    "Design a catalyst for selective CO2 reduction",
    "Find materials with high electron mobility for transistor channels"
]
```

## 3. 消融实验变体设计

### 3.1 完整系统 (FULL) - 基线

所有模块启用：
- 多专家辩论 + GAN 对抗优化
- 风险评估 (Phase B.5)
- 统计检验 + 可视化 + 因果推断
- RL 指导的假设演化
- 技能固化与跨域迁移
- 文献挖掘

### 3.2 变体分组

#### Group A: 假设生成组件消融

| 变体名 | 移除/修改的组件 | 预期影响 |
|--------|----------------|---------|
| **FULL** | 无 (基线) | 最佳综合性能 |
| **A1: No-Debate** | 禁用多专家辩论，仅使用单智能体生成假设 | 假设多样性下降 |
| **A2: No-GAN** | 禁用 GAN 对抗优化，仅保留辩论结果 | 新颖性得分下降 |
| **A3: Debate-Only** | 仅使用多专家辩论，禁用 GAN | 假设质量中等 |
| **A4: GAN-Only** | 仅使用 GAN，禁用多专家辩论 | 新颖性高但合理性可能不足 |
| **A5: Vanilla-LLM** | 禁用辩论和 GAN，使用基础 LLM 直接生成 | 假设单一、创新性低 |

#### Group B: 实验执行组件消融

| 变体名 | 移除/修改的组件 | 预期影响 |
|--------|----------------|---------|
| **B1: No-Risk-Assess** | 禁用风险评估 (Phase B.5)，所有计划直接执行 | 实验失败率上升，计算浪费 |
| **B2: Static-Risk-Only** | 仅使用 Layer 1 静态检查，禁用 LLM 深度评估 | 过滤精度下降 |
| **B3: No-Auto-Analysis** | 禁用统计检验 + 可视化 + 因果推断，仅用 LLM 文本分析 | 分析深度不足 |
| **B4: Stats-Only** | 仅启用统计检验，禁用可视化和因果推断 | 定量分析有但无机制理解 |
| **B5: No-Causal** | 禁用因果推断，保留统计检验 + 可视化 | 相关性分析但无因果理解 |

#### Group C: 演化与记忆组件消融

| 变体名 | 移除/修改的组件 | 预期影响 |
|--------|----------------|---------|
| **C1: No-RL-Evolution** | 禁用 RL 指导，使用纯 LLM 假设演化 | 演化方向可能偏离最优 |
| **C2: No-Skill-Solidify** | 禁用技能固化，不保存成功实验 | 跨会话无法复用经验 |
| **C3: No-Cross-Domain** | 禁用跨域技能迁移，仅使用同域技能 | 领域迁移能力下降 |
| **C4: No-Literature** | 禁用文献挖掘，问题生成不基于文献 | 研究前沿性可能不足 |
| **C5: No-Memory** | 禁用记忆系统，每轮独立运行 | 无法从历史迭代学习 |

#### Group D: 组合消融 (关键双组件)

| 变体名 | 移除的组件组合 | 预期影响 |
|--------|---------------|---------|
| **D1: No-Debate-No-GAN** | 同时禁用辩论和 GAN | 假设质量大幅下降 |
| **D2: No-Risk-No-Analysis** | 同时禁用风险评估和自动分析 | 实验效率和分析质量双降 |
| **D3: No-Evolution-No-Skill** | 同时禁用演化和技能系统 | 无迭代优化和经验积累 |

### 3.3 变体配置代码示例

```python
# orchestrator.py 配置修改示例

# FULL (基线)
config_full = {
    "use_debate": True,
    "use_gan": True,
    "use_risk_assessment": True,
    "use_statistical_testing": True,
    "use_visualization": True,
    "use_causal_inference": True,
    "use_rl_evolution": True,
    "use_skill_solidify": True,
    "use_cross_domain": True,
    "use_literature": True,
    "use_memory": True,
}

# A2: No-GAN
config_a2 = {**config_full, "use_gan": False}

# B1: No-Risk-Assess
config_b1 = {**config_full, "use_risk_assessment": False}

# C1: No-RL-Evolution
config_c1 = {**config_full, "use_rl_evolution": False}

# D1: No-Debate-No-GAN
config_d1 = {**config_full, "use_debate": False, "use_gan": False}
```

## 4. 评估指标体系

### 4.1 主要指标 (Primary Metrics)

| 指标 | 计算方式 | 意义 |
|------|---------|------|
| **收敛率 (CR)** | converged=True 的实验次数 / 总次数 | 系统能否在限定迭代内找到稳定解 |
| **平均收敛迭代数 (ACI)** | 收敛实验的迭代次数平均值 | 收敛效率 |
| **假设新颖性 (NS)** | novelty_assessor 评分均值 | 产出的创新程度 |
| **实验成功率 (ESR)** | success=True 的实验数 / 总实验数 | 代码执行可靠性 |
| **置信度均值 (MC)** | 所有 analysis.confidence 的平均值 | 结果可信度 |

### 4.2 次要指标 (Secondary Metrics)

| 指标 | 计算方式 | 意义 |
|------|---------|------|
| **假设数量 (NH)** | 每轮迭代产生的假设数 | 假设空间探索广度 |
| **计划过滤率 (PFR)** | 被风险过滤的计划数 / 总计划数 | 风险评估强度 |
| **技能复用率 (SRR)** | 复用技能的实验数 / 总实验数 | 经验积累效果 |
| **跨域迁移次数 (CT)** | 使用跨域技能的次数 | 迁移学习能力 |
| **文献引用数 (LC)** | 假设中引用文献的数量 | 文献利用程度 |
| **因果推断边数 (CE)** | 贝叶斯网络发现的因果边数 | 机制理解深度 |
| **统计检验显著率 (SSR)** | 显著检验数 / 总检验数 | 数据驱动发现能力 |
| **可视化生成数 (VN)** | 每轮生成的图表数 | 结果呈现能力 |

### 4.3 资源效率指标

| 指标 | 计算方式 | 意义 |
|------|---------|------|
| **总运行时间 (TRT)** | 单次发现任务的总耗时 | 时间效率 |
| **LLM 调用次数 (LCT)** | 总 API 调用次数 | 成本效率 |
| **代码执行失败率 (CEFR)** | 失败实验数 / 总实验数 | 资源浪费程度 |
| **重复实验率 (RER)** | 相同或相似实验数 / 总实验数 | 探索冗余度 |

## 5. 实验执行流程

### 5.1 单次实验流程

```python
async def run_single_experiment(config, seed_problem, domain):
    """运行单次消融实验"""
    orchestrator = DiscoveryOrchestrator(
        domain=domain,
        initial_problem=seed_problem,
        max_iterations=5,
        model_name="deepseek-chat",
        **config
    )
    
    result = await orchestrator.run_discovery()
    
    return {
        "converged": result.converged,
        "iterations_used": len(result.iterations),
        "novelty_scores": [it.overall_novelty for it in result.iterations],
        "success_rate": compute_success_rate(result.iterations),
        "mean_confidence": compute_mean_confidence(result.iterations),
        "total_time": result.total_time,
        "llm_calls": result.llm_call_count,
        ...
    }
```

### 5.2 批量实验脚本

```python
# ablation_runner.py
import asyncio
import json
from datetime import datetime

CONFIGS = {
    "FULL": config_full,
    "A1-No-Debate": config_a1,
    "A2-No-GAN": config_a2,
    "A3-Debate-Only": config_a3,
    "A4-GAN-Only": config_a4,
    "A5-Vanilla-LLM": config_a5,
    "B1-No-Risk": config_b1,
    "B2-Static-Risk": config_b2,
    "B3-No-Auto-Analysis": config_b3,
    "B4-Stats-Only": config_b4,
    "B5-No-Causal": config_b5,
    "C1-No-RL": config_c1,
    "C2-No-Skill": config_c2,
    "C3-No-Cross-Domain": config_c3,
    "C4-No-Literature": config_c4,
    "C5-No-Memory": config_c5,
    "D1-No-Debate-GAN": config_d1,
    "D2-No-Risk-Analysis": config_d2,
    "D3-No-Evolution-Skill": config_d3,
}

DOMAINS = ["battery", "catalysis", "electronics"]
SEEDS = [
    "How to improve energy density of Li-ion batteries?",
    "Design a catalyst for selective CO2 reduction",
    "Find materials with high electron mobility for transistor channels"
]
N_RUNS = 10

async def run_all_experiments():
    results = {}
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    for config_name, config in CONFIGS.items():
        results[config_name] = {}
        for domain in DOMAINS:
            results[config_name][domain] = []
            for seed in SEEDS:
                run_results = []
                for run_id in range(N_RUNS):
                    print(f"[{config_name}] {domain} | {seed[:30]}... | Run {run_id+1}/{N_RUNS}")
                    result = await run_single_experiment(config, seed, domain)
                    run_results.append(result)
                results[config_name][domain].extend(run_results)
    
    # 保存原始结果
    with open(f"ablation_results_{timestamp}.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    return results
```

## 6. 结果分析方法

### 6.1 统计检验

对每个指标进行以下统计检验：

1. **单因素方差分析 (ANOVA)**: 检验各变体间差异是否显著
2. **配对 t 检验**: 每个变体 vs FULL 基线
3. **效应量计算 (Cohen's d)**: 量化差异大小
4. **多重比较校正 (Bonferroni)**: 控制族错误率

### 6.2 可视化输出

1. **雷达图**: 各变体在多维度指标上的综合表现
2. **热力图**: 变体 × 指标矩阵，颜色表示性能
3. **箱线图**: 关键指标的分布差异
4. **瀑布图**: 各组件对 FULL 系统的边际贡献

### 6.3 归因分析

计算各组件的**边际贡献值** (Marginal Contribution, MC):

```
MC(Component) = Performance(FULL) - Performance(FULL without Component)

协同效应 = Performance(FULL) - Σ MC(Component_i) - Performance(Vanilla)
```

## 7. 预期结果与假设

### 7.1 核心假设

| 假设编号 | 假设内容 | 预期验证方式 |
|---------|---------|------------|
| H1 | 多专家辩论提升假设多样性 | A1 vs FULL: NH 显著下降 |
| H2 | GAN 对抗优化提升新颖性 | A2 vs FULL: NS 显著下降 |
| H3 | 辩论 + GAN 协同效应 > 单独使用 | FULL > A3 且 FULL > A4 |
| H4 | 风险评估减少计算浪费 | B1 vs FULL: CEFR 上升，TRT 增加 |
| H5 | 自动数据分析提升置信度 | B3 vs FULL: MC 下降 |
| H6 | 因果推断增强机制理解 | B5 vs FULL: CE 下降但 MC 不变 |
| H7 | RL 演化加速收敛 | C1 vs FULL: ACI 增加 |
| H8 | 技能固化促进经验积累 | C2 vs FULL: SRR = 0，ESR 下降 |
| H9 | 跨域迁移提升新领域性能 | C3 vs FULL: 新领域 ESR 下降 |
| H10 | 文献挖掘增强前沿性 | C4 vs FULL: LC 下降 |

### 7.2 预期排序 (综合性能)

```
FULL > A3(Debate-Only) ≈ A2(No-GAN) > A4(GAN-Only) > A1(No-Debate) > A5(Vanilla)
FULL > B2(Static-Risk) > B5(No-Causal) > B4(Stats-Only) > B3(No-Auto-Analysis) > B1(No-Risk)
FULL > C1(No-RL) > C4(No-Literature) > C3(No-Cross-Domain) > C2(No-Skill) > C5(No-Memory)
```

## 8. 实验资源需求

### 8.1 计算资源

| 资源 | 需求 | 说明 |
|------|------|------|
| LLM API 调用 | ~20,000 次 | 19 变体 × 3 领域 × 3 种子 × 10 轮 |
| 运行时间 | ~200 小时 | 单次约 20-40 分钟 |
| 存储空间 | ~5 GB | 结果 JSON + 可视化图表 + 日志 |

### 8.2 成本估算

假设 DeepSeek API 价格为 $0.001/1K tokens，每次调用平均 2K tokens：

```
总成本 ≈ 20,000 × 2 × $0.001 = $40 (人民币约 300 元)
```

## 9. 执行计划与时间表

| 阶段 | 任务 | 时间 | 产出 |
|------|------|------|------|
| **Phase 1** | 实现消融配置开关 | 1 天 | config 支持各模块开关 |
| **Phase 2** | 编写批量实验脚本 | 1 天 | ablation_runner.py |
| **Phase 3** | 小批量预实验 (n=2) | 2 天 | 验证流程和指标可行性 |
| **Phase 4** | 全量实验 (n=10) | 10 天 | 原始结果数据 |
| **Phase 5** | 统计分析与可视化 | 2 天 | 图表 + 统计报告 |
| **Phase 6** | 论文写作 | 3 天 | 消融实验章节 |

## 10. 结果报告模板

### 10.1 主结果表

| 变体 | CR(%) | ACI | NS | ESR(%) | MC | TRT(min) |
|------|-------|-----|-----|--------|-----|----------|
| FULL | 85.2±5.1 | 3.2±0.8 | 0.78±0.12 | 72.4±8.3 | 0.82±0.09 | 28.5±5.2 |
| A1-No-Debate | 62.1±7.3* | 4.1±0.9* | 0.65±0.15* | 68.2±9.1 | 0.71±0.11* | 25.3±4.8 |
| A2-No-GAN | 78.5±6.2 | 3.4±0.8 | 0.61±0.13* | 70.1±8.5 | 0.79±0.10 | 27.8±5.0 |
| ... | ... | ... | ... | ... | ... | ... |

\* 表示与 FULL 差异显著 (p < 0.05, 配对 t 检验)

### 10.2 组件贡献瀑布图

```
Vanilla LLM (基线):       35.2 分
+ 多专家辩论:             +12.5 分 (H1 验证)
+ GAN 对抗优化:           +8.3 分 (H2 验证)
+ 辩论-GAN 协同:          +5.1 分 (H3 验证)
+ 风险评估:               +6.8 分 (H4 验证)
+ 自动数据分析:           +7.2 分 (H5 验证)
+ 因果推断:               +3.4 分 (H6 验证)
+ RL 演化:                +4.9 分 (H7 验证)
+ 技能固化:               +3.6 分 (H8 验证)
+ 跨域迁移:               +2.8 分 (H9 验证)
+ 文献挖掘:               +1.9 分 (H10 验证)
+ 其他协同效应:           +2.5 分
─────────────────────────────────
FULL 系统总分:            93.2 分
```

## 11. 局限性与改进方向

### 11.1 已知局限

1. **领域覆盖**: 目前仅测试 3 个领域，结论推广性有限
2. **种子问题**: 固定种子可能引入偏差
3. **模型依赖**: 结果可能随 LLM 模型变化
4. **计算成本**: 全量实验成本较高

### 11.2 改进方向

1. 扩展至 5-10 个不同领域
2. 使用动态生成的问题而非固定种子
3. 对比多个 LLM 模型 (DeepSeek, GPT-4, Claude)
4. 引入人类专家评估作为金标准
5. 长期追踪 (10-20 轮迭代) 观察技能积累效应

## 12. 附录

### 附录 A: 指标计算伪代码

```python
def compute_convergence_rate(results):
    """收敛率"""
    return sum(1 for r in results if r["converged"]) / len(results)

def compute_mean_novelty(results):
    """平均新颖性"""
    scores = []
    for r in results:
        for it in r["iterations"]:
            if it.get("novelty_score"):
                scores.append(it["novelty_score"])
    return np.mean(scores) if scores else 0.0

def compute_experiment_success_rate(results):
    """实验成功率"""
    total, success = 0, 0
    for r in results:
        for it in r["iterations"]:
            for exp in it.get("results", []):
                total += 1
                if exp.get("success"):
                    success += 1
    return success / total if total > 0 else 0.0

def compute_mean_confidence(results):
    """平均置信度"""
    confidences = []
    for r in results:
        for it in r["iterations"]:
            for ana in it.get("analysis", []):
                if ana.get("confidence"):
                    confidences.append(ana["confidence"])
    return np.mean(confidences) if confidences else 0.0
```

### 附录 B: 配置开关实现建议

在 `orchestrator.py` 的 `__init__` 方法中添加模块开关参数：

```python
def __init__(
    self,
    # ... 现有参数 ...
    # 消融实验开关
    use_debate: bool = True,
    use_gan: bool = True,
    use_risk_assessment: bool = True,
    use_statistical_testing: bool = True,
    use_visualization: bool = True,
    use_causal_inference: bool = True,
    use_rl_evolution: bool = True,
    use_skill_solidify: bool = True,
    use_cross_domain: bool = True,
    use_literature: bool = True,
    use_memory: bool = True,
):
    # ... 保存开关状态 ...
    self.module_switches = {
        "debate": use_debate,
        "gan": use_gan,
        "risk_assessment": use_risk_assessment,
        # ...
    }
```

在各 Phase 执行前检查开关状态：

```python
# Phase A.1 多专家辩论
if self.module_switches["debate"]:
    debate_outcome = await generate_and_debate_hypotheses(...)
else:
    # 直接由 LLM 生成假设
    debate_outcome = await vanilla_hypothesis_generation(...)

# Phase A.2 GAN 对抗优化
if self.module_switches["gan"] and self.module_switches["debate"]:
    gan_outcome = await run_gan_adversarial_refinement(...)
```

---

**文档版本**: v1.0
**创建日期**: 2026-04-19
**适用系统**: DISCOVERY-AGENT v2.0+
