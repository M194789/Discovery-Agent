# 系统能力基准测试实验方案

## 1. 实验目标

评估 Discovery-Agent 在多个科学领域（电池材料、催化、电子与通信）下的自主科学发现能力，量化系统在假设生成质量、实验执行成功率、收敛效率、结果新颖性等维度的表现。

## 2. 实验设计

### 2.1 测试领域与问题集

选取工程已有 Demo 覆盖的 3 个领域，每个领域设计 3 个不同难度/类型的**常规题**，并额外构造 1 个**Ground Truth 回顾性问题**用于定性验证：

**领域 A：电池材料（常规题）**
- A1（基础）：如何提高锂离子电池的能量密度？
- A2（进阶）：设计一种高稳定性固态电解质材料体系
- A3（探索）：针对钠离子电池负极的容量衰减机制提出改进策略
- A-GT（回顾）：在 LiFePO4 正极材料体系下，提出可显著提升倍率性能的改性策略（GT：纳米化 + 碳包覆）

**领域 B：催化科学（常规题）**
- B1（基础）：如何提高 CO2 电化学还原的选择性？
- B2（进阶）：设计一种抗中毒的低成本燃料电池催化剂
- B3（探索）：光催化水分解中载流子复合的抑制策略
- B-GT（回顾）：针对氨合成反应，提出常温常压下替代 Haber-Bosch 的催化策略（GT：单原子/电催化 N2 还原）

**领域 C：电子与通信（常规题）**
- C1（基础）：如何降低高频信号传输中的损耗？
- C2（进阶）：设计一种低功耗毫米波天线阵列
- C3（探索）：针对 6G 通信中的智能超表面优化策略
- C-GT（回顾）：针对 5G 毫米波波束管理，提出降低波束切换时延的策略（GT：AI 辅助波束预测 / 码本优化）

### 2.2 运行配置

使用 `DiscoveryOrchestrator` 的默认完整配置（12 个模块开关全部为 `True`：`use_debate / use_gan / use_risk_assessment / use_statistical_testing / use_visualization / use_causal_inference / use_rl_evolution / use_skill_solidify / use_cross_domain / use_literature / use_memory`，`use_risk_static_only=False`）：
- 底层模型：`deepseek-v4-pro`
- 最大迭代次数：`MAX_DISCOVERY_ITERATIONS = 5`
- GAN 对抗轮数：`GAN_ADVERSARIAL_ROUNDS = 3`
- 风险评估阈值：`RISK_FEASIBILITY_THRESHOLD = 0.4`
- 实验超时：`DEEP_SOLVER_TIMEOUT = 1800` 秒

**分级运行策略**：
- **常规题（A1-A3 / B1-B3 / C1-C3，共 9 题）**：每题独立运行 `N = 5` 次（不同 `user_id` 隔离技能/记忆），用于**定量**指标统计（均值 ± 标准差）。
- **GT 回顾题（A-GT / B-GT / C-GT，共 3 题）**：每题独立运行 `N = 1` 次，用于**定性**评估系统是否能命中或逼近已知 Ground Truth 答案。

### 2.3 评估指标体系

#### 维度 1：假设生成质量（主观 + 客观）

| 指标 | 计算方法 | 数据来源 |
|------|----------|----------|
| 假设数量 | 每轮生成的假设总数 | `orchestrator.py` 状态日志 |
| 假设多样性 | 假设间语义相似度（嵌入距离） | `skill_store/embeddings.py`（需二次封装用于假设向量化） |
| 科学合理性评分 | **LLM-as-Judge**：`deepseek-v4-pro` 多数投票打分 1-5 | 自动化评审脚本 |
| GT 命中度（仅 GT 题） | 最终确认假设与 GT 答案的语义匹配（命中/部分命中/未命中） | LLM-as-Judge 定性判定 |
| 可验证性 | 能否转化为可执行实验计划 | 实验规划器输出 |

#### 维度 2：实验执行能力

| 指标 | 计算方法 | 数据来源 |
|------|----------|----------|
| 实验成功率 | 成功执行的实验数 / 总计划数 | `ExperimentResult.success` |
| 平均重试次数 | 每次实验的平均重试次数 | `experiment_executor.py` 日志 |
| 实验类型分布 | DFT / MD / ML / simulation / data_analysis 占比 | `ExperimentPlan.method` |
| 代码可运行率 | 首次生成即可运行的代码比例 | 执行日志分析 |

#### 维度 3：收敛效率

| 指标 | 计算方法 | 数据来源 |
|------|----------|----------|
| 收敛率 | 收敛的运行次数 / 总运行次数 | `DiscoveryState.converged` |
| 平均收敛迭代数 | 收敛运行的平均迭代轮次 | 状态日志 |
| 活跃假设清零率 | 因无活跃假设而停止的比例 | 状态日志 |
| 最大迭代耗尽率 | 因达到最大迭代而强制收敛的比例 | 状态日志 |

#### 维度 4：结果质量与分析深度

| 指标 | 计算方法 | 数据来源 |
|------|----------|----------|
| 统计检验覆盖率 | 应用统计检验的分析比例 | `result_analyzer.py` 输出 |
| 可视化生成率 | 产生图表的分析比例 | `discovery_agents/analysis_utils/visualization.py` |
| 因果推断触发率 | 触发因果推断模块的比例 | `discovery_agents/analysis_utils/causal_inference.py` |
| 假设状态分布 | confirmed / refuted / active / eliminated 占比 | 最终假设状态 |

#### 维度 5：资源效率

| 指标 | 计算方法 | 数据来源 |
|------|----------|----------|
| 总运行时间 | 单次发现流程的 wall-clock 时间 | 系统计时 |
| LLM API 调用次数 | 每轮迭代的 API 调用次数 | 日志统计 |
| 风险评估过滤率 | 被过滤的实验计划比例 | `discovery_agents/risk_assessor.py` 双层评估输出 |
| 风险评估 LLM 跳过率 | Layer 1 短路跳过 LLM 的比例 | `discovery_agents/risk_assessor.py` 日志（`RISK_SKIP_LLM_FOR_LOW=True`） |

### 2.4 对比分析维度

除单系统评估外，还进行以下对比：

1. **领域间对比**：分析系统在不同科学领域的适用性差异
2. **问题难度对比**：基础/进阶/探索三类问题的表现差异
3. **迭代进程分析**：追踪每轮迭代的假设数量、实验成功率、置信度变化曲线

## 3. 实验执行流程

### 3.1 自动化执行脚本

基于 [ablation_runner.py](ablation_runner.py) 的 `run_single_experiment` 模式（已具备 `user_id` 隔离、`model_name` 环境变量读取、JSON 持久化能力），编写 `benchmark_runner.py`：

```python
# 伪代码
async def run_benchmark(regular_problems, gt_problems, n_regular=5, n_gt=1):
    results = []
    # 常规题：5 次定量
    for problem in regular_problems:
        for run_id in range(n_regular):
            user_id = f"benchmark_{problem.id}_{run_id}"  # 隔离记忆/技能
            orch = DiscoveryOrchestrator(
                domain=problem.domain,
                initial_problem=problem.statement,
                max_iterations=5,
                model_name=os.getenv("AGENT_MODEL_NAME", "deepseek-v4-pro"),
                user_id=user_id,
                interactive=False,
                verbose=False,
            )
            result = await orch.run()
            results.append(extract_metrics(result, orch.state, mode="quantitative"))
    # GT 回顾题：1 次定性
    for problem in gt_problems:
        user_id = f"benchmark_gt_{problem.id}"
        orch = DiscoveryOrchestrator(
            domain=problem.domain,
            initial_problem=problem.statement,
            max_iterations=5,
            model_name=os.getenv("AGENT_MODEL_NAME", "deepseek-v4-pro"),
            user_id=user_id,
            interactive=False,
            verbose=False,
        )
        result = await orch.run()
        results.append(extract_metrics(result, orch.state, mode="qualitative_gt", gt=problem.ground_truth))
    return results
```

### 3.2 指标提取与日志解析

编写 `benchmark_metrics.py` 模块，自动从 `DiscoveryState` 和日志中提取上述指标：
- 解析 `state.iterations` 中的假设、实验、分析结果
- 统计 `state.current_iteration`（实际迭代数）
- 读取 `state.converged` 和收敛原因
- 计算假设嵌入相似度（复用 `skill_store/embeddings.py`）

### 3.3 结果持久化

每次运行保存为独立 JSON：
```
benchmark_results/
  battery_A1_run0.json
  battery_A1_run1.json
  ...
  catalysis_B3_run4.json
  electronics_C2_run2.json
  battery_A-GT_run0.json
  catalysis_B-GT_run0.json
  electronics_C-GT_run0.json
```

汇总后生成 `benchmark_summary.json` 和可视化图表。

## 4. 预期结果与分析

### 4.1 预期发现

- **领域差异**：电池和催化领域（材料科学）可能更依赖 DFT/MD 方法，实验成功率受包可用性影响较大；电子领域更依赖 simulation/data_analysis，成功率可能更高。
- **迭代进程**：前 1-2 轮假设数量较多，第 3-4 轮逐步收敛，第 5 轮可能因达到上限而强制停止。
- **风险评估效果**：Layer 1 静态检查预计可短路 30-50% 的明确可行/不可行计划，显著减少 LLM API 调用。

### 4.2 结果呈现

- 表格：各领域/问题的平均指标对比
- 图表：迭代进程曲线（假设数、置信度、成功率随迭代变化）
- 热力图：领域 x 问题难度 x 指标的多维可视化
- 案例分析：选取典型运行，展示完整的发现循环轨迹

## 5. 局限性与说明

1. **实验执行限制**：所有实验为本地 Python 计算，不涉及真实物理实验或云平台资源。
2. **评估自动化**：科学合理性评分和 GT 命中度均采用 **LLM-as-Judge（`deepseek-v4-pro` 多数投票）**，避免人工评审的主观性与人力成本；建议进行 3 次独立打分取多数结果。
3. **LLM 随机性**：即使常规题运行 5 次，LLM 的随机性仍可能导致结果波动，建议报告均值 +/- 标准差；GT 题仅运行 1 次且结果为定性判定，不做方差分析。
4. **人机协作**：当前系统人机协作功能缺失，本实验仅评估全自动模式下的系统能力。

## 6. 实验辅助脚本设计

### 6.1 benchmark_runner.py

位置：`./benchmarks/benchmark_runner.py`

职责：
- 加载问题集配置
- 循环调用 `DiscoveryOrchestrator` 执行发现流程
- 捕获标准输出/错误输出
- 记录 wall-clock 时间
- 将 `DiscoveryState` 序列化为 JSON

### 6.2 benchmark_metrics.py

位置：`./benchmarks/benchmark_metrics.py`

职责：
- 从 JSON 状态文件中提取结构化的评估指标
- 计算统计量（均值、标准差、中位数）
- 生成对比表格和 CSV 输出

### 6.3 benchmark_visualize.py

位置：`./benchmarks/benchmark_visualize.py`

职责：
- 读取 `benchmark_summary.json`
- 生成 Matplotlib 图表（迭代曲线、箱线图、热力图）
- 输出 PNG 格式的可视化结果
