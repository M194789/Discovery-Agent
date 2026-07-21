"""
Result Analyzer and Hypothesis Evolution Agents (Module 4) - 结果分析与假设演化模块（模块4）

Analyzes experiment results and evolves hypotheses for the next iteration
of the discovery loop.
分析实验结果，为下一轮发现循环演化假设。

这个模块是系统的"实验报告分析员"：
- 读取实验执行的结果（成功/失败、输出数据）
- 判断实验结果是否支持假设
- 区分"技术失败"（代码报错）和"科学驳斥"（假设错误）
- 更新假设状态（confirmed/refuted/active等）
- 决定研究是否收敛（可以结束了）
"""

from __future__ import annotations  # 启用未来版本的类型注解

import _utf8_env  # noqa: F401  # 全局 UTF-8 编码修复

import asyncio       # 异步编程库
from typing import List, Optional  # 类型提示

# === 导入智能体框架 ===
from agents import Agent, Runner               # 智能体和运行器
from agents import OpenAIChatCompletionsModel  # OpenAI兼容模型

# === 导入配置 ===
from config import client, MODEL_NAME, AGENT_MAX_TURNS
# client = API客户端, MODEL_NAME = 模型名称, AGENT_MAX_TURNS = 最大对话轮数

# === 导入数据模型 ===
from models.problem_models import ResearchProblem       # 研究问题
from models.hypothesis_models import Hypothesis         # 假设
from models.experiment_models import ExperimentResult   # 实验结果
from models.analysis_models import (
    AnalysisResult, HypothesisEvolution,  # 分析结果、假设演化
    EnrichedAnalysisResult,               # 增强版分析结果
)

# ---------------------------------------------------------------------------
# Tool loading - 工具加载（延迟加载，避免循环导入）
# ---------------------------------------------------------------------------

_analysis_tools = None    # 分析工具缓存
_evolution_tools = None   # 演化工具缓存


def _get_analysis_tools():
    """
    获取分析工具列表（延迟加载）。
    分析智能体可以调用这些工具来辅助分析。
    """
    global _analysis_tools  # 使用全局变量缓存
    if _analysis_tools is None:
        _analysis_tools = []
        # DeepSeek模型不支持工具调用，跳过
        if "deepseek" in MODEL_NAME.lower():
            return _analysis_tools
        # 导入工具加载函数
        from config import load_tavily_search, load_workspace_tools
        ws = load_workspace_tools()  # 加载工作空间工具
        _analysis_tools = [
            ws.execute_code,           # 执行代码工具（可以运行分析代码）
            load_tavily_search()       # Tavily搜索工具（可以查资料）
        ]
    return _analysis_tools


def _get_evolution_tools():
    """
    获取假设演化工具列表（延迟加载）。
    演化智能体可以调用这些工具。
    """
    global _evolution_tools  # 使用全局变量缓存
    if _evolution_tools is None:
        _evolution_tools = []
        # DeepSeek模型不支持工具调用，跳过
        if "deepseek" in MODEL_NAME.lower():
            return _evolution_tools
        # 导入搜索工具
        from config import load_tavily_search
        _evolution_tools = [load_tavily_search()]
        # 尝试加载记忆搜索工具（可能不可用）
        try:
            from config import load_memory_tools
            mem = load_memory_tools()
            _evolution_tools.append(mem.search_memory)
        except Exception:
            pass  # 记忆工具不可用也没关系
    return _evolution_tools


def _is_deepseek() -> bool:
    """
    检查是否使用DeepSeek模型。
    DeepSeek需要特殊处理（JSON输出、不支持工具等）。
    """
    return "deepseek" in MODEL_NAME.lower()


# ---------------------------------------------------------------------------
# Analysis agent - 分析智能体（分析单个实验结果）
# ---------------------------------------------------------------------------

ANALYSIS_PROMPT = """
ROLE: Experiment Result Analyst - 实验结果分析师

You analyze computational experiment results to determine whether they support
or refute a scientific hypothesis.
你分析计算实验结果，判断它们是支持还是驳斥科学假设。

TASK:
给定假设、实验计划摘要和原始执行输出：

1. **Parse Results** - 从执行输出中提取关键数值和数据
2. **Statistical Analysis** - 如有需要，编写并执行分析代码进行额外处理
3. **Evaluate** - 判断结果是否支持假设，并分配置信度
4. **Summarize** - 生成关键发现、统计摘要和综合解释
5. **Output** - 返回结构化的AnalysisResult

CRITICAL - DISTINGUISH TECHNICAL vs SCIENTIFIC FAILURES:
【关键 - 区分技术失败和科学失败】

- **Technical failure** 技术失败（代码错误、缺少依赖、编码问题、导入错误、超时）：
  设置 supports_hypothesis=True
  置信度根据失败类型判断：
  - 如果是简单的导入错误/语法错误（容易修复）：confidence=0.5
  - 如果是复杂的执行错误/超时：confidence=0.4
  - 在explanation中注明"TECHNICAL_FAILURE"和具体错误类型
  假设**未被驳斥**，只是代码执行有问题

- **Scientific refutation** 科学驳斥（实验成功运行但结果与假设矛盾）：
  设置 supports_hypothesis=False，给出适当置信度（0.6-0.9）
  
- **Scientific support** 科学支持（实验运行且结果支持假设）：
  设置 supports_hypothesis=True，给出适当置信度（0.7-0.95）

CONSTRAINTS:
- 只基于实际数据下结论，不做假设
- 技术失败**不应**驳斥假设 - 它们表示需要重试
- 在stderr中查找：ImportError, ModuleNotFoundError, SyntaxError,
  UnicodeDecodeError, timeout - 这些都是技术失败
- 最多10次工具调用
"""


def create_analysis_agent() -> Agent:
    """
    Create a result analysis agent.
    创建结果分析智能体。
    
    这个智能体负责分析单个实验结果，判断它是否支持假设。
    """
    instructions = ANALYSIS_PROMPT  # 使用分析提示词作为工作说明
    
    # 如果是DeepSeek模型，添加JSON格式要求
    # 因为DeepSeek不支持结构化输出，需要在提示词里明确要求JSON
    if _is_deepseek():
        instructions += (
            "\n\nIMPORTANT: Return a valid JSON object with fields: "
            "hypothesis_id (string), supports_hypothesis (boolean), "
            "confidence (0.0-1.0), key_findings (list of strings), "
            "statistical_summary (string), explanation (string)."
        )
    
    # 组装智能体参数
    kwargs = {
        "name": "ResultAnalysisAgent",  # 智能体名称
        "instructions": instructions,    # 工作说明（提示词）
        "model": OpenAIChatCompletionsModel(
            model=MODEL_NAME, openai_client=client
        ),
    }
    # 只有非DeepSeek模型才能使用结构化输出和工具
    if not _is_deepseek():
        kwargs["output_type"] = AnalysisResult  # 输出类型：分析结果
        kwargs["tools"] = _get_analysis_tools()  # 可用工具列表
    
    return Agent(**kwargs)  # 创建并返回智能体


# ---------------------------------------------------------------------------
# Hypothesis evolution agent - 假设演化智能体（更新假设状态）
# ---------------------------------------------------------------------------

EVOLUTION_PROMPT = """
ROLE: Hypothesis Evolution Strategist - 假设演化策略师

You are a scientific strategist who decides how hypotheses should evolve based
on experimental evidence.
你是科学策略师，基于实验证据决定假设如何演化。

TASK:
给定所有当前假设、它们的分析结果和研究问题：

1. **Update Statuses** - 更新每个假设的状态：
   - "confirmed": 强科学证据支持（实验成功且结果支持）
   - "refuted": 科学证据明确驳斥（实验成功但结果矛盾）
   - "active": 尚无定论需要更多测试，或发生了技术失败
   - "merged": 与另一个假设合并
   - "eliminated": 被删除（冗余、不可行等）

2. **Generate New Hypotheses** - 基于结果中的模式，提出1-3个下一代假设：
   - 建立在已确认的发现之上
   - 解决失败实验揭示的空白
   - 探索数据中发现的关系
   - 使用递增的代数编号，引用parent_ids

3. **Assess Convergence** - 评估是否收敛（可以结束研究）：
   【重要】收敛判断应该积极但合理，避免无限循环
   
   如果满足以下任一条件，**必须**设置 converged=True：
   - 至少一个假设被"confirmed"且置信度 >= 0.6
   - 当前迭代次数 >= 2 且至少一个实验成功运行并返回有意义结果
   - 大多数假设（>50%）是"confirmed"或"refuted"（很少"active"剩余）
   - 目标属性已满意地实现（有明确数值结果支持）
   - 连续两轮迭代没有产生新的有价值的假设
   - 已经探索了所有主要的研究方向
   
   只有以下情况设置 converged=False：
   - 所有实验都因TECHNICAL_FAILURE失败，且没有成功实验，或
   - 完全没有获得任何有意义的结果来评估假设
   
   【决策指南】
   - 如果有成功实验且结果支持某些假设，倾向于收敛
   - 如果实验已经验证了关键假设，即使还有active假设也可以收敛
   - 不要为了继续迭代而故意设置 converged=False

4. **Output** - 返回结构化的HypothesisEvolution。

CRITICAL - HANDLING TECHNICAL FAILURES:
【关键 - 处理技术失败】
- 如果分析提到"TECHNICAL_FAILURE", "ImportError", "ModuleNotFoundError",
  "encoding error", "timeout", 或 "execution failed"：
  - 保持假设状态为"active"（**不要**设为"refuted"）
  - 这些是代码执行问题，不是科学驳斥
  - 在新假设中提出使用更简单代码的 refined experiments

CONSTRAINTS:
- 始终提供清晰的evolution_reasoning（演化推理）
- 新假设必须具体且可计算测试
- 如果至少一个实验成功，倾向于收敛
- **永远不要**因技术/代码失败将假设标记为"refuted"
- 最多10次工具调用
"""


def create_evolution_agent() -> Agent:
    """
    Create a hypothesis evolution agent.
    创建假设演化智能体。
    
    这个智能体负责根据实验结果更新假设状态，决定研究是否收敛。
    """
    instructions = EVOLUTION_PROMPT  # 使用演化提示词作为工作说明
    
    # 如果是DeepSeek模型，添加详细的JSON格式示例
    # 因为DeepSeek不支持结构化输出，需要给出明确的格式示例
    if _is_deepseek():
        instructions += '''

OUTPUT FORMAT: Return a valid JSON object EXACTLY like this example:
【输出格式：返回一个有效的JSON对象，严格按照这个示例】

```json
{
  "hypothesis_updates": [
    {
      "id": "H1",
      "statement": "Updated hypothesis statement",
      "scientific_basis": "Why this hypothesis is valid",
      "predicted_outcome": "Expected result",
      "confidence": 0.75,
      "generation": 2,
      "status": "confirmed"
    }
  ],
  "new_hypotheses": [],
  "eliminated_ids": ["H2"],
  "converged": true,
  "evolution_reasoning": "Hypothesis H1 was confirmed by experiment with confidence 0.75. Since at least one hypothesis is confirmed and we are at iteration 2+, the discovery has converged."
}
```

CRITICAL - SET converged TO true IF:
【关键 - 在以下情况设置 converged 为 true】
- At least one hypothesis has status "confirmed" with confidence >= 0.6
- Current iteration is 2 or higher AND at least one experiment succeeded with meaningful results
- Most hypotheses (>50%) are "confirmed" or "refuted" (few remain "active")
- Target properties have been satisfactorily achieved
- No new valuable hypotheses were generated in the last iteration

IMPORTANT: Be proactive in setting converged=true. If experiments succeeded and validated key hypotheses, set converged=true to avoid infinite loops.
'''
    
    # 组装智能体参数
    kwargs = {
        "name": "HypothesisEvolutionAgent",  # 智能体名称
        "instructions": instructions,         # 工作说明（提示词）
        "model": OpenAIChatCompletionsModel(
            model=MODEL_NAME, openai_client=client
        ),
    }
    # 只有非DeepSeek模型才能使用结构化输出和工具
    if not _is_deepseek():
        kwargs["output_type"] = HypothesisEvolution  # 输出类型：假设演化
        kwargs["tools"] = _get_evolution_tools()      # 可用工具列表
    
    return Agent(**kwargs)  # 创建并返回智能体


# ---------------------------------------------------------------------------
# Top-level functions - 顶层函数（供其他模块调用）
# ---------------------------------------------------------------------------


async def analyze_results(
    results: List[ExperimentResult],      # 输入：实验结果列表
    hypotheses: List[Hypothesis],         # 输入：假设列表（用于查找对应假设）
    problem: ResearchProblem,             # 输入：研究问题（提供上下文）
    user_id: str = "experiment_system",   # 输入：用户ID
    output_dir: Optional[str] = None,     # 输入：图表输出目录（可选）
    use_statistical_testing: bool = True,  # 消融开关: 统计检验
    use_visualization: bool = True,        # 消融开关: 可视化
    use_causal_inference: bool = True,     # 消融开关: 因果推断
) -> List[EnrichedAnalysisResult]:        # 输出：增强版分析结果列表
    """
    Analyze each experiment result against its hypothesis.
    分析每个实验结果与其假设的关系。

    为了提高效率，并行运行多个分析智能体。
    分析完成后，通过程序化增强层添加统计检验、可视化和因果推断。

    参数:
        results: 要分析的实验结果
        hypotheses: 当前假设（用于上下文查找）
        problem: 研究问题上下文
        user_id: 用户标识符
        output_dir: 可视化图表输出目录（可选）

    返回:
        增强版分析结果列表（包含统计检验、可视化、因果推断）
    """
    agent = create_analysis_agent()  # 创建分析智能体
    
    # 创建假设ID到假设对象的映射，方便查找
    hyp_map = {h.id: h for h in hypotheses}

    async def _analyze_one(exp_result: ExperimentResult) -> AnalysisResult | None:
        """
        分析单个实验结果的内部函数。
        """
        # 根据实验结果中的hypothesis_id找到对应的假设
        hyp = hyp_map.get(exp_result.hypothesis_id)
        if not hyp:
            return None  # 找不到对应假设，跳过

        # 组装输入文本（给AI看的"分析报告模板"）
        input_text = (
            f"Research Problem: {problem.problem_statement}\n\n"  # 研究问题
            f"Hypothesis ({hyp.id}):\n"                          # 假设ID
            f"  Statement: {hyp.statement}\n"                    # 假设陈述
            f"  Predicted outcome: {hyp.predicted_outcome}\n\n"  # 预期结果
            f"Experiment Summary: {exp_result.plan_summary}\n"   # 实验摘要
            f"Execution Success: {exp_result.success}\n\n"       # 执行是否成功
            f"Execution Output:\n{exp_result.execution_output}\n\n"  # 执行输出
        )
        # 如果有错误日志，也加上
        if exp_result.error_log:
            input_text += f"Error Log:\n{exp_result.error_log}\n"

        try:
            # 调用分析智能体进行分析
            res = await Runner().run(
                agent, input_text, max_turns=AGENT_MAX_TURNS
            )
            output = res.final_output  # 获取分析结果
            
            # 如果是DeepSeek模型，需要手动解析JSON输出
            if _is_deepseek() and isinstance(output, str):
                import json
                import re
                try:
                    # 清理可能的markdown标记
                    cleaned = re.sub(
                        r'^```(?:json)?\s*|\s*```$', '', output.strip()
                    )
                    data = json.loads(cleaned)  # 解析JSON
                    output = AnalysisResult(**data)  # 转换为对象
                except (json.JSONDecodeError, TypeError) as e:
                    print(
                        f"    Warning: Failed to parse analysis for "
                        f"{exp_result.hypothesis_id}: {e}"
                    )
                    return None  # 解析失败，返回None
            
            return output  # 返回分析结果
        except Exception as e:
            hid = exp_result.hypothesis_id
            print(f"    Warning: Analysis failed for {hid}: {e}")
            return None  # 分析失败，返回None

    # 为每个实验结果创建一个分析任务
    tasks = [_analyze_one(r) for r in results]
    # 并行执行所有分析任务
    raw = await asyncio.gather(*tasks, return_exceptions=True)

    # 收集成功的分析结果
    analyses: List[AnalysisResult] = []
    for item in raw:
        if isinstance(item, Exception):
            print(f"    Warning: Analysis task raised: {item}")
        elif item is not None:
            analyses.append(item)

    # === 后处理增强层：统计检验、可视化、因果推断 ===
    enriched = await _enrich_analyses(
        analyses, results, hypotheses, output_dir,
        use_statistical_testing=use_statistical_testing,
        use_visualization=use_visualization,
        use_causal_inference=use_causal_inference,
    )
    return enriched


async def evolve_hypotheses(
    hypotheses: List[Hypothesis],         # 输入：所有当前假设
    analyses: List[AnalysisResult],       # 输入：本轮的分析结果
    iteration: int,                       # 输入：当前迭代编号
    problem: ResearchProblem,             # 输入：研究问题
    user_id: str = "experiment_system",   # 输入：用户ID
    rl_action_plan=None,                  # 输入：RL优化器的动作计划（可选）
) -> HypothesisEvolution:                 # 输出：假设演化结果
    """
    Evolve hypotheses based on analysis results.
    基于分析结果演化假设。

    这是发现循环的"大脑"，决定：
    - 哪些假设被证实/证伪
    - 是否生成新假设
    - 研究是否收敛（可以结束）

    参数:
        hypotheses: 所有当前假设
        analyses: 本轮迭代的分析结果
        iteration: 当前迭代编号
        problem: 研究问题上下文
        user_id: 用户标识符

    返回:
        描述假设更新和新假设的HypothesisEvolution对象
    """
    agent = create_evolution_agent()  # 创建演化智能体

    # 格式化假设列表（给AI看的简洁格式）
    hyp_text = "\n".join(
        f"- [{h.id}] (gen={h.generation}, status={h.status}) {h.statement}"
        for h in hypotheses
    )
    
    # 格式化分析结果（给AI看的简洁格式）
    analysis_text = "\n".join(
        f"- Hypothesis {a.hypothesis_id}: "
        f"{'SUPPORTS' if a.supports_hypothesis else 'REFUTES'} "
        f"(confidence={a.confidence:.2f})\n"
        f"  Key findings: {'; '.join(a.key_findings)}"
        for a in analyses
    )

    # 组装输入文本（给AI看的"演化决策报告"）
    input_text = (
        f"User ID: {user_id}\n\n"
        f"Research Problem: {problem.problem_statement}\n"  # 研究问题
        f"Domain: {problem.domain}\n"                       # 领域
        f"Target properties: {', '.join(problem.target_properties)}\n\n"  # 目标属性
        f"Current iteration: {iteration}\n\n"               # 当前迭代
        f"Current Hypotheses:\n{hyp_text}\n\n"              # 当前假设
        f"Analysis Results:\n{analysis_text}\n\n"           # 分析结果
        "Based on these results, evolve the hypotheses for the next iteration."
        # 基于这些结果，为下一轮迭代演化假设
    )

    # === RL 优化器指导注入 ===
    # 如果提供了 RL 动作计划，将其作为结构化建议追加到输入中
    if rl_action_plan is not None:
        rl_guidance_lines = [
            "\n\n--- RL OPTIMIZER GUIDANCE ---",
            "Based on accumulated experimental evidence across iterations, "
            "the RL optimizer recommends the following actions:",
        ]
        actions = (
            rl_action_plan.actions
            if hasattr(rl_action_plan, 'actions')
            else []
        )
        for ha in actions:
            action_str = ha.action.upper() if hasattr(ha, 'action') else "UNKNOWN"
            hyp_id = ha.hypothesis_id if hasattr(ha, 'hypothesis_id') else "?"
            score = ha.score if hasattr(ha, 'score') else 0.0
            line = f"- {hyp_id}: {action_str} (confidence_score={score:.2f})"
            if hasattr(ha, 'context') and ha.context:
                line += f" [{ha.context}]"
            rl_guidance_lines.append(line)
        rl_guidance_lines.extend([
            "",
            "You SHOULD follow these recommendations unless you have "
            "strong scientific reasons to deviate.",
            "If you deviate from a recommendation, explain why in "
            "evolution_reasoning.",
            "--- END RL GUIDANCE ---",
        ])
        input_text += "\n".join(rl_guidance_lines)

    # 调用演化智能体进行决策
    result = await Runner().run(agent, input_text, max_turns=AGENT_MAX_TURNS)
    
    # 如果是DeepSeek模型，需要手动解析JSON输出
    output = result.final_output
    if _is_deepseek() and isinstance(output, str):
        import json
        import re
        try:
            # 清理可能的markdown标记
            cleaned = re.sub(
                r'^```(?:json)?\s*|\s*```$', '', output.strip()
            )
            data = json.loads(cleaned)  # 解析JSON
            
            # 解析假设对象（将字典转换为Hypothesis对象）
            if "hypothesis_updates" in data:
                data["hypothesis_updates"] = [
                    Hypothesis(**h) for h in data["hypothesis_updates"]
                ]
            if "new_hypotheses" in data:
                data["new_hypotheses"] = [
                    Hypothesis(**h) for h in data["new_hypotheses"]
                ]
            
            # 确保converged字段被正确解析
            if "converged" not in data:
                # 从上下文推断收敛状态
                updates = data.get("hypothesis_updates", [])
                confirmed_count = sum(
                    1 for h in updates
                    if isinstance(h, dict) and h.get("status") == "confirmed"
                    or hasattr(h, 'status') and h.status == "confirmed"
                )
                # 如果在迭代>=2时有任何假设被确认，认为已收敛
                if confirmed_count > 0 and iteration >= 2:
                    data["converged"] = True
                    print(f"    Inferred converged=True (confirmed={confirmed_count}, iter={iteration})")
                else:
                    data["converged"] = False
            
            output = HypothesisEvolution(**data)  # 转换为对象
        except (json.JSONDecodeError, TypeError) as e:
            raise ValueError(f"Failed to parse evolution output: {e}")
    
    return output


# ---------------------------------------------------------------------------
# Advanced Analysis Enrichment - 高级分析增强层
# ---------------------------------------------------------------------------


async def _enrich_analyses(
    analyses: List[AnalysisResult],
    results: List[ExperimentResult],
    hypotheses: List[Hypothesis],
    output_dir: Optional[str] = None,
    use_statistical_testing: bool = True,   # 消融开关: 统计检验
    use_visualization: bool = True,          # 消融开关: 可视化
    use_causal_inference: bool = True,       # 消融开关: 因果推断
) -> List[EnrichedAnalysisResult]:
    """
    对 LLM 分析结果进行程序化增强：统计检验、可视化、因果推断。

    每个增强步骤独立 try/except，任何失败都不会中断管道。

    参数:
        analyses: LLM 生成的分析结果
        results: 原始实验结果（用于提取数值数据）
        hypotheses: 假设列表
        output_dir: 图表输出目录

    返回:
        EnrichedAnalysisResult 列表
    """
    # 延迟导入增强模块（避免在工具不可用时报错）
    try:
        from discovery_agents.analysis_utils.data_parser import (
            parse_numerical_data, ParsedNumericalData,
        )
        from discovery_agents.analysis_utils.statistical_testing import (
            run_statistical_tests,
        )
        from discovery_agents.analysis_utils.visualization import (
            generate_visualizations,
        )
        from discovery_agents.analysis_utils.causal_inference import (
            run_causal_inference,
        )
        _enrichment_available = True
    except ImportError as e:
        print(f"    Warning: Analysis enrichment modules not available: {e}")
        _enrichment_available = False

    # 构建实验结果映射 (hypothesis_id -> ExperimentResult)
    result_map = {r.hypothesis_id: r for r in results}
    hyp_map = {h.id: h for h in hypotheses}

    # 用于因果推断的跨实验聚合数据
    all_parsed_data: list = []

    enriched_list: List[EnrichedAnalysisResult] = []

    for analysis in analyses:
        errors: List[str] = []
        stat_tests = []
        viz_artifacts = []
        causal_result = None
        extracted_data = {}
        parsed_data = None

        if _enrichment_available:
            # --- 1. 解析数值数据 ---
            exp_result = result_map.get(analysis.hypothesis_id)
            if exp_result and exp_result.execution_output:
                try:
                    parsed_data = parse_numerical_data(
                        exp_result.execution_output
                    )
                    extracted_data = dict(parsed_data.metrics)
                    if parsed_data.parse_confidence > 0:
                        all_parsed_data.append(parsed_data)
                except Exception as e:
                    errors.append(f"Data parsing failed: {e}")

            # --- 2. 统计检验 ---
            # 【步骤2】对解析出的数值数据进行统计检验
            if use_statistical_testing and parsed_data and parsed_data.parse_confidence > 0:
                # 判断条件：开启了统计检验开关 且 成功解析出数值数据 且 解析置信度大于0
                try:
                    # 尝试执行统计检验
                    hyp = hyp_map.get(analysis.hypothesis_id)
                    # 根据当前分析的假设ID，从假设映射表中查找对应的假设对象
                    hyp_stmt = hyp.statement if hyp else ""
                    # 获取该假设的陈述文本，如果找不到则使用空字符串
                    stat_tests = run_statistical_tests(
                        # 调用统计检验模块，返回检验结果列表
                        parsed_data, hyp_stmt
                        # 传入解析后的数值数据和假设陈述，用于针对性检验
                    )
                except Exception as e:
                    # 捕获统计检验过程中的任何异常
                    errors.append(f"Statistical testing failed: {e}")
                    # 将错误信息记录到当前分析的错误列表中，不中断整体流程

            # --- 3. 可视化 ---
            # 【步骤3】根据解析出的数值数据生成可视化图表
            if use_visualization and parsed_data and parsed_data.parse_confidence > 0 and output_dir:
                # 判断条件：开启了可视化开关 且 有解析数据 且 置信度>0 且 指定了输出目录
                try:
                    # 尝试生成可视化图表
                    viz_artifacts = generate_visualizations(
                        # 调用可视化模块，返回生成的图表文件路径列表
                        parsed_data, analysis.hypothesis_id, output_dir
                        # 传入数值数据、假设ID（用于命名图表）、输出目录路径
                    )
                except Exception as e:
                    # 捕获可视化生成过程中的任何异常
                    errors.append(f"Visualization generation failed: {e}")
                    # 将错误信息记录到错误列表中，不中断流程

        # 构建 EnrichedAnalysisResult
        # 【整合】将原始LLM分析结果与程序化增强结果合并，构建增强版分析结果对象
        enriched = EnrichedAnalysisResult(
            # 创建增强版分析结果对象，继承并扩展原始分析结果
            # 继承原始 AnalysisResult 的所有字段
            # 以下字段来自LLM智能体生成的原始分析结果
            hypothesis_id=analysis.hypothesis_id,
            # 假设ID：标识该分析结果对应的假设
            supports_hypothesis=analysis.supports_hypothesis,
            # 是否支持假设：LLM判断实验结果是否支持对应假设
            confidence=analysis.confidence,
            # 置信度：LLM给出的置信度评分（0.0-1.0）
            key_findings=analysis.key_findings,
            # 关键发现：LLM提取的实验结果要点列表
            statistical_summary=analysis.statistical_summary,
            # 统计摘要：LLM生成的统计描述文本
            explanation=analysis.explanation,
            # 解释说明：LLM对分析结果的详细解释
            # 新增字段
            # 以下字段是本函数新增的程序化增强结果
            statistical_tests=stat_tests,
            # 统计检验结果：程序化执行的假设检验、置信区间等
            visualizations=viz_artifacts,
            # 可视化图表：生成的图表文件路径列表
            causal_inference=causal_result,
            # 因果推断结果：初始为None，将在后续跨实验因果推断步骤中填充
            extracted_numerical_data=extracted_data,
            # 提取的数值数据：从实验输出中解析出的结构化数值指标
            enrichment_errors=errors,
            # 增强过程错误：记录统计检验、可视化等步骤中的错误信息
        )
        enriched_list.append(enriched)
        # 将当前增强分析结果添加到总列表中

    # --- 4. 因果推断（跨实验聚合） ---
    # 【步骤4】基于所有实验的聚合数据进行跨实验因果推断
    if use_causal_inference and _enrichment_available and all_parsed_data and len(all_parsed_data) >= 1:
        # 判断条件：开启了因果推断开关 且 增强模块可用 且 有至少一组聚合数据
        try:
            # 尝试执行跨实验因果推断
            causal = run_causal_inference(
                # 调用因果推断模块，返回因果关系分析结果
                all_parsed_data,
                # 传入所有实验解析出的数值数据列表（跨实验聚合）
                hypothesis_id="cross_experiment",
                # 使用特殊标识符表示这是跨实验的因果分析，而非单个实验
                output_dir=output_dir,
                # 图表输出目录，因果图等可视化结果将保存至此
            )
            if causal and causal.relationships:
                # 如果成功推断出因果关系且关系列表非空
                # 将因果推断结果附加到所有增强结果上
                for enriched in enriched_list:
                    # 遍历所有已生成的增强分析结果
                    enriched.causal_inference = causal
                    # 将跨实验因果推断结果统一附加到每个分析结果中
        except Exception as e:
            # 捕获因果推断过程中的任何异常
            for enriched in enriched_list:
                # 遍历所有增强分析结果
                enriched.enrichment_errors.append(
                    f"Causal inference failed: {e}"
                    # 将因果推断失败信息记录到每个结果的错误列表中
                )

    return enriched_list
    # 返回所有增强版分析结果的列表
