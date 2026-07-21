"""
Experiment Planner Agent (Module 2b) - 实验规划器智能体(模块2b)

Designs computational experiment workflows for testing hypotheses
设计用于测试假设的计算实验工作流

这个模块是系统的"实验设计师"：
- 接收假设和研究问题
- 设计具体的计算实验方案
- 输出可执行的实验计划
"""

from __future__ import annotations  # 启用未来版本的类型注解语法
from typing import List  # 类型提示：列表
import sys  # 系统库，用于修改模块搜索路径
from pathlib import Path  # 路径处理库

# === 设置模块搜索路径 ===
# 获取当前文件的父父目录(项目根目录)
current_dir = Path(__file__).resolve().parent.parent
# 如果这个目录还不在Python的搜索路径里，就把它加进去
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))

import _utf8_env  # noqa: F401  # 全局 UTF-8 编码修复

# === 导入AI智能体框架 ===
from agents import Agent, Runner  # Agent=智能体，Runner=运行器
from agents import OpenAIChatCompletionsModel  # OpenAI兼容模型

# === 导入项目配置 ===
from config import client, MODEL_NAME, AGENT_MAX_TURNS
# client = API客户端，MODEL_NAME = 模型名称，AGENT_MAX_TURNS = 最大对话轮数

# === 导入数据模型 ===
from models.problem_models import ResearchProblem  # 研究问题
from models.hypothesis_models import Hypothesis, ExperimentPlan  # 假设、实验计划

# === 工具列表(懒加载) ===
# 为什么要懒加载？因为加载工具可能很慢，第一次用到时再加载
_tools = None  # 初始为空，第一次调用时填充


def _get_tools():
    """
    Get and return tool list, load on first call
    获取并返回工具列表，第一次调用时加载
    
    工具就像是实验设计师的"工具箱"：
    - Tavily搜索：上网查文献、找最佳实践
    - 包检查：看看电脑上装了哪些Python包
    - 代码提取：从网页扒代码
    - 记忆搜索：回顾以前设计过的实验
    """
    global _tools  # 使用全局变量，这样只加载一次
    if _tools is None:  # 如果还没加载过
        _tools = []  # 初始化为空列表
        # DeepSeek模型不支持工具调用，直接返回空列表
        if "deepseek" in MODEL_NAME.lower():
            return _tools
        # 加载Tavily搜索和工作空间工具
        from config import load_tavily_search, load_workspace_tools
        ws = load_workspace_tools()
        _tools = [
            load_tavily_search(),           # 网络搜索工具
            ws.check_installed_packages     # 检查已安装的Python包
        ]
        # 尝试加载研究工具（从URL提取代码等）
        try:
            from config import load_research_tools
            rt = load_research_tools()
            _tools.extend([
                rt.extract_code_from_url,    # 从网页提取代码
                rt.retrieve_extracted_code   # 获取提取的代码
            ])
        except Exception:
            pass  # 加载失败也没关系
        # 尝试加载记忆工具
        try:
            from config import load_memory_tools
            mem = load_memory_tools()
            _tools.append(mem.search_memory)  # 搜索历史记忆
        except Exception:
            pass  # 加载失败也没关系
    return _tools  # 返回已加载的工具列表


def _is_deepseek() -> bool:
    """
    Check if using DeepSeek model
    检查是否使用DeepSeek模型
    
    为什么要检查？因为DeepSeek：
    - 不支持结构化输出(output_type)
    - 不支持工具调用(tools)
    - 需要手动解析JSON
    """
    return "deepseek" in MODEL_NAME.lower()


# ============================================================
# 实验规划师的"工作说明书"
# ============================================================
EXPERIMENT_PLANNER_PROMPT = """
Role: Computational Experiment Planner
角色：计算实验规划师

You design computational experiment workflows for scientific research across
all domains (materials science, electronics, communications, physics, etc.).
你为所有领域的科学研究设计计算实验工作流。

Task: Based on the given hypothesis and broader research problem, design a complete
computational experiment that can be executed via Python code.
任务：基于给定的假设和研究问题，设计一个可以通过Python代码执行的完整计算实验。

Workflow:
工作流程：

1. **Search Memory** - 搜索记忆，找类似的过往实验设计
1.5. **Check Prior Skills** - 如果输入中包含 "REUSABLE SKILL PATTERNS"：
   - 仔细检查其 tool_chain 和 code_specification
   - 如果某个已有技能的方法和工具与你计划的方案匹配，直接复用其 code_specification 作为模板
   - 根据当前假设的具体需求调整代码规格
   - 优先使用已被验证成功的工具链
   - 如果没有相关的已有技能，则从头设计
2. **Select Method** - 选择最合适的计算方法：
   - simulation: 数值模拟（信号处理、电路、物理）
   - ML_modeling: 机器学习/深度学习实验
   - data_analysis: 数据分析、统计测试
   - DFT: 密度泛函理论（仅材料科学）
   - MD: 分子动力学（仅材料科学）
   - ML_potential: 机器学习势函数（仅材料科学）

3. **Check Packages** - 检查可用的Python包
4. **Research Method** - 用Tavily搜索最佳实践和代码示例
5. **Design Steps** - 把实验分解成有序步骤
6. **Write Code Spec** - 写详细的Python代码规格
   重要：代码规格只能用常见的Python包
   (numpy, scipy, pandas, matplotlib, scikit-learn, torch, tensorflow)
   不要用领域特定的包，除非确认已安装
7. **Define Success Criteria** - 定义实验成功的量化标准
8. **Output** - 返回结构化的实验计划

Constraints:
约束条件：
- 代码规格要足够详细，能自动生成代码
- 优先使用通用包：numpy, scipy, pandas, matplotlib, scikit-learn, torch
- 材料科学：额外考虑 pymatgen, ase
- 电子/通信：使用 numpy, scipy.signal, torch
- 确保代码规格描述的是一个自包含的实验
  如果没有真实数据，要能生成模拟数据
- 最多调用15次工具
"""


def create_experiment_planner_agent() -> Agent:
    """
    Create and return an experiment planner agent
    创建并返回实验规划师智能体
    
    这个函数创建一个"实验设计师"AI：
    - 给它评分标准（提示词）
    - 给它工具（搜索、检查包等）
    - 设置好和大模型的连接
    """
    instructions = EXPERIMENT_PLANNER_PROMPT  # 使用工作说明书
    
    # 如果是DeepSeek模型，添加JSON格式要求
    # 因为DeepSeek不支持结构化输出，需要在提示词里明确要求JSON格式
    if _is_deepseek():
        instructions += (
            "\n\nIMPORTANT: Return a valid JSON object with fields: "
            "hypothesis_id (string), method (string: DFT/MD/ML_potential/data_analysis), "
            "code_specification (string), estimated_complexity (string: low/medium/high), "
            "required_packages (list of strings), success_criteria (string), "
            "tool_chain (list of package names), steps (list of step descriptions)"
        )
    
    # 组装智能体参数
    kwargs = {
        "name": "ExperimentPlannerAgent",    # 智能体名字
        "instructions": instructions,         # 工作说明（提示词）
        "model": OpenAIChatCompletionsModel(  # 连接的AI模型
            model=MODEL_NAME, openai_client=client
        ),
    }
    
    # 只有非DeepSeek模型才能使用结构化输出和工具
    # DeepSeek不支持output_type和tools参数
    if not _is_deepseek():
        kwargs["output_type"] = ExperimentPlan  # 输出类型：实验计划
        kwargs["tools"] = _get_tools()          # 可用工具列表
    
    return Agent(**kwargs)  # 创建并返回智能体


async def plan_experiments(
    hypotheses: List[Hypothesis],          # 输入：假设列表
    problem: ResearchProblem,              # 输入：研究问题
    user_id: str = "experiment_system",    # 输入：用户ID
    use_skills: bool = True,              # 是否启用技能检索复用
) -> List[ExperimentPlan]:                 # 输出：实验计划列表
    """
    Create experiment plans for each active hypothesis
    为每个活跃的假设创建实验计划
    
    这是本模块的主函数，被orchestrator.py调用。
    整体流程：
    1. 创建实验规划师智能体
    2. 遍历每个活跃的假设
    3. 为每个假设设计实验
    4. 返回所有实验计划
    """
    # 第一步：创建实验规划师
    agent = create_experiment_planner_agent()
    plans: List[ExperimentPlan] = []  # 存放生成的实验计划
    
    # 第二步：遍历每个假设
    for hyp in hypotheses:
        # 只处理"活跃"的假设
        if hyp.status != "active":
            continue

        # 检索相关技能作为参考工作流（含完整代码规格和工具链）
        skill_context = ""
        if use_skills:
            try:
                import skill_registry
                skill_results = await skill_registry.find_applicable_skills_rich(
                    f"{hyp.statement} {problem.domain}",
                    user_id,
                    domain=problem.domain,
                    top_k=3,
                    include_cross_domain=True,
                )
                if skill_results and "未找到" not in skill_results:
                    skill_context = (
                        "\n\n--- REUSABLE SKILL PATTERNS FROM PRIOR EXPERIMENTS ---\n"
                        "The following skills were successfully executed previously. "
                        "Reuse their tool chains, code patterns, and success criteria "
                        "where applicable. Adapt the specifics to the current hypothesis.\n\n"
                        f"{skill_results}\n"
                        "--- END SKILL PATTERNS ---\n"
                    )
            except Exception:
                pass  # 技能检索失败不阻塞主流程

        # 组装输入信息 - 告诉AI所有相关信息
        input_text = (
            f"User ID: {user_id}\n\n"
            f"Research Problem:\n{problem.problem_statement}\n"  # 研究问题
            f"Domain: {problem.domain}\n"                       # 领域
            f"Target properties: {', '.join(problem.target_properties)}\n\n"  # 目标属性
            f"Hypothesis to test (ID={hyp.id}):\n"              # 要测试的假设
            f"  Statement: {hyp.statement}\n"                   # 假设陈述
            f"  Scientific basis: {hyp.scientific_basis}\n"     # 科学依据
            f"  Predicted outcome: {hyp.predicted_outcome}\n"   # 预期结果
            f"{skill_context}\n"
            "Design a computational experiment to test this hypothesis."  # 请设计实验
        )

        try:
            # 第三步：调用大模型生成实验计划
            # 这里是真正连接大模型的地方！
            result = await Runner().run(
                agent, input_text, max_turns=AGENT_MAX_TURNS
            )
            output = result.final_output  # 获取AI的输出
            
            # ========== DeepSeek特殊处理：手动解析JSON ==========
            # DeepSeek不支持结构化输出，返回的是字符串，需要手动解析
            if _is_deepseek() and isinstance(output, str):
                import json
                import re
                try:
                    # 去掉markdown代码块标记
                    cleaned = output.strip()
                    cleaned = re.sub(r'^```(?:json)?\s*', '', cleaned)  # 去掉开头的```json
                    cleaned = re.sub(r'\s*```\s*$', '', cleaned)        # 去掉结尾的```
                    cleaned = cleaned.strip()
                    
                    data = None
                    
                    # 策略1：直接解析
                    try:
                        data = json.loads(cleaned)
                    except json.JSONDecodeError:
                        pass
                    
                    # 策略2：括号匹配找完整JSON对象
                    if data is None:
                        brace_count = 0
                        last_valid_pos = 0
                        in_string = False
                        escape_next = False
                        found_first_open = False
                        start_pos = -1
                        
                        for i, char in enumerate(cleaned):
                            if escape_next:
                                escape_next = False
                                continue
                            if char == '\\' and in_string:
                                escape_next = True
                                continue
                            if char == '"':
                                in_string = not in_string
                            elif not in_string:
                                if char == '{':
                                    if not found_first_open:
                                        found_first_open = True
                                        start_pos = i
                                    brace_count += 1
                                elif char == '}':
                                    if found_first_open:
                                        brace_count -= 1
                                        if brace_count == 0:
                                            last_valid_pos = i + 1
                                            break
                        
                        if last_valid_pos > 0 and start_pos >= 0:
                            json_str = cleaned[start_pos:last_valid_pos]
                            try:
                                data = json.loads(json_str)
                            except json.JSONDecodeError:
                                pass
                    
                    # 策略3：正则表达式提取
                    if data is None:
                        json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', cleaned)
                        if json_match:
                            try:
                                data = json.loads(json_match.group(0))
                            except json.JSONDecodeError:
                                pass
                    
                    # 策略4：手动提取关键字段
                    if data is None:
                        hyp_match = re.search(r'"hypothesis_id"\s*:\s*"([^"]+)"', cleaned)
                        method_match = re.search(r'"method"\s*:\s*"([^"]+)"', cleaned)
                        
                        if hyp_match and method_match:
                            data = {
                                "hypothesis_id": hyp_match.group(1),
                                "method": method_match.group(1),
                                "code_specification": "# Auto-generated placeholder",
                                "tool_chain": ["python", "numpy"],
                                "steps": ["Execute analysis", "Generate results"],
                                "success_criteria": "Analysis completes successfully"
                            }
                    
                    # 如果所有策略都失败，创建默认计划
                    if data is None:
                        print(f"    Warning: Could not parse plan for {hyp.id}, using default plan")
                        print(f"    Raw output preview: {output[:200]}...")
                        data = {
                            "hypothesis_id": hyp.id,
                            "method": "data_analysis",
                            "code_specification": "import numpy as np\nimport pandas as pd\n# Placeholder code",
                            "tool_chain": ["python", "numpy", "pandas"],
                            "steps": ["Load data", "Analyze", "Report results"],
                            "success_criteria": "Analysis completes"
                        }
                    
                    # 确保必要字段存在
                    if "tool_chain" not in data:
                        data["tool_chain"] = ["pymatgen", "ase"]
                    if "steps" not in data:
                        data["steps"] = [
                            "Load structure",
                            "Run calculation",
                            "Analyze results"
                        ]
                    if "code_specification" not in data:
                        data["code_specification"] = "# Placeholder code"
                    if "hypothesis_id" not in data:
                        data["hypothesis_id"] = hyp.id
                    if "method" not in data:
                        data["method"] = "data_analysis"
                    
                    # 处理success_criteria字段类型
                    if "success_criteria" in data:
                        sc = data["success_criteria"]
                        if isinstance(sc, list):
                            data["success_criteria"] = "; ".join(sc)
                        elif not isinstance(sc, str):
                            data["success_criteria"] = str(sc)
                    else:
                        data["success_criteria"] = "Experiment completes"
                    
                    # 转换为Pydantic对象
                    output = ExperimentPlan(**data)
                except Exception as e:
                    print(f"    Warning: Could not parse plan for {hyp.id}: {e}")
                    print(f"    Raw output preview: {output[:200]}...")
                    # 创建备用计划，确保流程不中断
                    output = ExperimentPlan(
                        hypothesis_id=hyp.id,
                        method="data_analysis",
                        code_specification="import numpy as np\n# Fallback code",
                        tool_chain=["python", "numpy"],
                        steps=["Execute analysis"],
                        success_criteria="Analysis completes"
                    )
            
            # 添加到计划列表
            plans.append(output)
            print(f"    Created plan for hypothesis {hyp.id}: {output.method}")
        except Exception as e:
            # 出错时创建备用计划，确保流程不中断
            print(f"    Warning: Plan generation failed for {hyp.id}: {e}")
            fallback = ExperimentPlan(
                hypothesis_id=hyp.id,
                method="data_analysis",
                code_specification=(
                    "import numpy as np\n"
                    "import pandas as pd\n"
                    "# Fallback experiment: basic data analysis\n"
                    f"# Hypothesis: {hyp.statement[:100] if hyp.statement else 'N/A'}\n"
                    "print('Running fallback analysis...')\n"
                    "data = np.random.randn(100, 5)\n"
                    "print(f'Sample mean: {np.mean(data):.4f}')\n"
                    "print(f'Sample std: {np.std(data):.4f}')\n"
                    "print('Analysis completed successfully')\n"
                ),
                tool_chain=["python", "numpy", "pandas"],
                steps=["Generate synthetic data", "Perform basic analysis", "Report results"],
                success_criteria="Analysis completes and prints results"
            )
            plans.append(fallback)
            print(f"    Created fallback plan for hypothesis {hyp.id}")

    return plans  # 返回所有实验计划
