# Discovery-Agent 🔬

基于**多智能体协作**与**闭环假设演化**的自主科学发现系统。系统以大语言模型为核心，模拟真实科研工作流：从**科学问题生成**出发，经过**假设生成与辩论、实验规划与执行、结果分析**，再通过**强化学习驱动的假设演化**形成闭环迭代，并借助**技能库**实现跨任务经验复用。

> 采用 OpenAI 兼容接口，默认使用 DeepSeek 后端。

---

## ✨ 核心特性

- 🧠 **端到端闭环**：问题 → 假设 → 实验 → 分析 → 演化 的自动迭代（`orchestrator.py`）
- 🤝 **多智能体协作**：领域专家生成竞争性假设 + 辩论精炼 + GAN 式对抗验证
- 🎯 **新颖性评估**：基于语义相似度量化假设与已有研究的差异
- ⚗️ **实验自动化**：自动生成并在受控沙箱中执行 Python 实验代码
- 🛡️ **风险评估**：过滤不可行 / 高风险的实验方案，节省算力与 API 调用
- 📈 **RL 假设演化**：Thompson Sampling / UCB1 多臂老虎机指导假设优化
- 📚 **文献挖掘**：通过 Tavily 检索为假设与新颖性判断提供外部证据
- 🧰 **技能库**：向量数据库存储可复用实验技能，支持跨领域迁移与元学习

---

## 🏗️ 项目结构

```
Discovery-Agent/
├── main.py                     # 命令行入口
├── app.py                      # Gradio Web 交互界面
├── demo.py                     # 离线演示（无需 API，使用 mock 数据）
├── orchestrator.py             # 中央闭环协调器（总指挥）
├── config.py                   # 共享配置（模型客户端、路径、超参数）
├── memory_ops.py               # 发现历史记忆读写
├── skill_registry.py           # 可复用技能注册
├── ui_bridge.py                # Web UI 与后端桥接
├── discovery_agents/           # 各智能体模块
│   ├── problem_generator.py    #   科学问题生成
│   ├── novelty_assessor.py     #   新颖性评估
│   ├── hypothesis_generator.py #   假设生成 / 辩论 / GAN 对抗精炼
│   ├── experiment_planner.py   #   实验方案规划
│   ├── experiment_executor.py  #   实验代码执行
│   ├── result_analyzer.py      #   结果分析与假设演化
│   ├── risk_assessor.py        #   实验风险评估
│   ├── literature_pipeline.py  #   文献检索管线
│   └── analysis_utils/         #   统计检验、因果推断、可视化、数据解析
├── models/                     # Pydantic 数据模型（问题/假设/实验/分析/状态/风险）
├── rl_evolution/               # 强化学习假设演化（bandit / optimizer / reward / persistence）
├── skill_store/                # 技能向量库（embeddings / retrieval / vector_db / meta_learning）
├── mcp_servers_and_tools/
│   └── direct_tools/           # search_tool（Tavily 检索）、memory_tools（记忆）
├── Demo/                       # 演示应用与示例数据
├── tests/                      # 单元测试
├── requirements.txt            # 依赖清单
└── .env.example                # 环境变量模板
```

---

## 🚀 快速开始

### 1. 环境要求

- Python **3.11+**（推荐）

### 2. 安装依赖

```bash
pip install -r requirements.txt
```

### 3. 配置 API 密钥

复制环境变量模板并填入你自己的密钥：

```bash
# Windows (PowerShell)
Copy-Item .env.example .env

# macOS / Linux
cp .env.example .env
```

编辑 `.env`：

```env
OPENAI_API_KEY=sk-your-deepseek-api-key-here
OPENAI_BASE_URL=https://api.deepseek.com/v1
AGENT_MODEL_NAME=deepseek-v4-pro
REQUIRE_OPENAI_API_KEY=true
TAVILY_API_KEY=tvly-your-tavily-api-key-here
```

> 系统使用 OpenAI 兼容接口，因此也可切换到 OpenAI、通义千问等任意兼容后端，只需修改 `OPENAI_BASE_URL` 与 `AGENT_MODEL_NAME`。

### 4. 运行

**命令行模式：**

```bash
python main.py --domain "catalysis" \
    --problem "Design efficient CO2 conversion catalysts at low temperature" \
    --max-iterations 3 --verbose
```

若省略 `--problem`，智能体将自动生成研究问题：

```bash
python main.py --domain "battery materials" --interactive
```

**Web 界面模式：**

```bash
python app.py
```

启动后浏览器访问 `http://localhost:7866`。

**离线演示（无需任何 API 密钥）：**

```bash
python demo.py
```

---

## ⚙️ 命令行参数

| 参数 | 说明 | 默认值 |
| --- | --- | --- |
| `--domain` | 研究领域（如 `catalysis`、`battery materials`）**（必填）** | — |
| `--problem` | 种子问题陈述；省略则由智能体自动生成 | `None` |
| `--max-iterations` | 发现循环的最大迭代次数 | `3` |
| `--model` | 大语言模型名称 | 环境变量 `AGENT_MODEL_NAME` |
| `--user-id` | 记忆操作的用户标识符 | `experiment_system` |
| `--interactive` | 启用人机交互检查点（human-in-the-loop） | 关闭 |
| `--verbose` | 打印详细的智能体输出 | 关闭 |

---

## 🔬 发现流程

系统由 `DiscoveryOrchestrator` 驱动，每一轮迭代执行如下闭环：

1. **问题生成** — 结合领域知识与文献识别研究空白
2. **新颖性评估** — 量化问题 / 假设与既有研究的语义差异
3. **假设生成** — 多智能体生成竞争性假设，经辩论与 GAN 式对抗精炼
4. **风险评估** — 过滤不可行或高风险方案
5. **实验规划与执行** — 自动生成实验方案并运行 Python 代码
6. **结果分析** — 统计检验、因果推断与可视化
7. **假设演化** — 基于 RL（多臂老虎机）修正 / 融合 / 淘汰假设，进入下一轮

达到收敛条件或最大迭代次数后终止。

---

## 🧩 主要依赖

`openai` / `openai-agents`（多智能体编排）、`pydantic`（数据校验）、`numpy` / `scipy` / `pandas`（科学计算）、`chromadb`（技能向量库）、`tavily-python`（文献检索）、`matplotlib` / `seaborn`（可视化）、`networkx` / `scikit-learn`（因果与相似度）、`gradio`（Web UI）。完整版本见 [`requirements.txt`](requirements.txt)。

---

## 📊 复现数据（Reproducibility Data）

基准测试与消融实验的原始结果 + 配套复现附录已发布至 Zenodo：

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.21906104.svg)](https://doi.org/10.5281/zenodo.21906104)

- 内容：330 个消融 run（19 配置 × 3 领域 × 5 run）+ 9 基准题 / 3 GT 题（各 5 run）的结果 JSON，及算法 / 超参 / 复现说明附录。
- 数据集 DOI：https://doi.org/10.5281/zenodo.21906104

---

## 📄 许可证

本项目基于 [MIT License](LICENSE) 开源。
