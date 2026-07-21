"""
Problem Generator Agent (Module 1) - 问题生成器智能体（模块1）

Mines literature to identify research gaps and generates a formal
research problem with novelty assessment.
挖掘文献以识别研究空白，并生成带有新颖性评估的正式研究问题。

增强功能:
- 程序化文献挖掘流水线（Tavily 多路搜索 + 知识图谱 + 因果推理 + 趋势分析）
- 向量记忆查询避免重复探索
- 所有预处理在智能体调用前完成，上下文注入提示词

这个模块是整个科学发现系统的"选题专家":
- 输入: 研究领域(如"催化剂") + 可选的初始问题
- 输出: 一个结构化的 ResearchProblem 对象
- 工作流程: 文献流水线预处理 -> 查记忆 -> 查文献 -> 找空白 -> 定问题 -> 评新颖
"""

from __future__ import annotations  # 启用未来版本的类型注解语法，让代码更简洁

import sys
from pathlib import Path

# 确保项目根目录在 sys.path 中（_utf8_env 位于项目根目录）
_project_root = str(Path(__file__).resolve().parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import _utf8_env  # noqa: F401  # 全局 UTF-8 编码修复

# === 导入 AI 智能体框架 ===
from agents import Agent, Runner  # Agent=智能体类，Runner=运行器（让智能体开始工作）
from agents import OpenAIChatCompletionsModel  # 用于连接 OpenAI 兼容的 API

# === 导入项目配置 ===
from config import client, MODEL_NAME, AGENT_MAX_TURNS
# client = API 客户端（已经配置好密钥的）
# MODEL_NAME = 用哪个模型，如 "o3" 或 "deepseek-chat"
# AGENT_MAX_TURNS = 智能体最多对话几轮（防止死循环）

# === 导入数据模型 ===
from models.problem_models import ResearchProblem  # 研究问题的数据结构定义

# === 工具列表（懒加载） ===
# 为什么要懒加载？因为加载工具可能很慢，启动时不加载，用到时再加载
_tools = None  # 初始为空，第一次调用时才填充


def _get_tools():
    """
    获取智能体可以使用的工具列表
    
    工具就像是智能体的"双手"：
    - Tavily 搜索：让智能体能上网查文献
    - Memory 搜索：让智能体能查以前的发现记录
    
    返回：工具列表（可能为空，如果工具都加载失败的话）
    """
    global _tools  # 使用全局变量，这样只加载一次
    
    if _tools is None:  # 如果还没加载过
        _tools = []  # 初始化为空列表
        
        # === 特殊处理：DeepSeek 模型不支持工具调用 ===
        # DeepSeek 有兼容性问题，所以直接跳过工具加载
        if "deepseek" in MODEL_NAME.lower():
            return _tools  # 返回空列表
        
        # === 尝试加载 Tavily 搜索工具 ===
        # Tavily 是一个网络搜索 API，用来查文献
        try:
            from config import load_tavily_search  # 从配置里导入加载函数
            _tools.append(load_tavily_search())    # 加载并添加到工具列表
        except Exception:
            pass  # 加载失败也没关系，不影响运行，只是没有搜索功能
        
        # === 尝试加载记忆工具 ===
        # 记忆工具用来搜索以前的发现记录
        try:
            from config import load_memory_tools  # 从配置里导入加载函数
            mem = load_memory_tools()             # 加载记忆工具集
            _tools.append(mem.search_memory)      # 只取其中的搜索功能
        except Exception:
            pass  # 加载失败也没关系，不影响运行
    
    return _tools  # 返回工具列表


# === 智能体的"工作说明书" ===
# 这段提示词告诉AI它的身份、任务、工作流程和约束条件
PROBLEM_GENERATOR_PROMPT = """
ROLE: Scientific Problem Generator
角色: 科学问题生成器

You are a research scientist who identifies impactful, computationally-testable
research problems by analyzing the current literature landscape.
你是一个研究科学家,通过分析当前文献来识别有影响力的、可计算验证的研究问题。

TASK:
任务:
Given a scientific domain (and optionally an initial problem statement from the
user), produce a formal ResearchProblem.
给定一个科学领域(以及用户可选的初始问题),生成一个正式的研究问题。

WORKFLOW:
工作流程:
1. **Search Memory** - call `search_memory` to check for past discoveries in
   this domain.
   搜索记忆 - 调用 search_memory 检查过去的发现

2. **Literature Survey** - use `tavily_search` (search_depth="advanced",
   max_results=5) with 3+ diverse queries covering:
   文献调研 - 使用 tavily_search 搜索3+个查询,覆盖:
   - Recent review articles / trends in the domain (最近的综述/趋势)
   - Known open challenges and research gaps (已知的难题和空白)
   - Computational methods commonly applied (常用的计算方法)

3. **Identify Gaps** - from the search results, identify 1-2 concrete research
   gaps where computational methods could make progress.
   识别空白 - 找出1-2个具体的研究空白

4. **Formulate Problem** - synthesize a clear, specific, computationally
   testable problem statement.  If the user provided an initial problem, enrich
   and validate it rather than replacing it.
   制定问题 - 综合出一个清晰、具体的问题陈述
   如果用户提供了初始问题,完善它而不是替换它

5. **Assess Novelty** - search for the closest existing work to this problem.
   Score novelty from 0.0 (well-explored) to 1.0 (highly novel).
   评估新颖性 - 搜索最接近的已有工作,打分从0.0(已被充分研究)到1.0(非常新颖)

6. **Output** - return a structured ResearchProblem with all fields filled.
   输出 - 返回一个填充完整的结构化研究问题

CONSTRAINTS:
约束条件:
- Focus on problems solvable with computational chemistry / materials science
  tools (DFT, MD, ML potentials, data mining).
  专注于可用计算工具解决的问题(密度泛函、分子动力学、机器学习等)
- The problem must have clear, measurable target properties.
  问题必须有清晰、可测量的目标属性
- Maximum 15 tool calls.
  最多调用15次工具

CONTEXT USAGE:
上下文使用:
- The system has already performed automated literature search, knowledge graph
  construction, causal reasoning, and trend analysis BEFORE your invocation.
  系统已在你被调用前自动完成了文献搜索、知识图谱构建、因果推理和趋势分析。
- If LITERATURE CONTEXT is provided above your input, USE it as your primary
  information source rather than relying solely on training knowledge.
  如果提供了文献上下文,将其作为主要信息来源。
- Cite provided URLs in supporting_literature and source_urls fields.
  在 supporting_literature 和 source_urls 字段中引用提供的 URL。
- Prioritize gaps identified by the automated analysis.
  优先考虑自动分析识别出的研究空白。
- AVOID research topics flagged with high prior work overlap (>0.6).
  避免与历史研究高度重叠(>0.6)的主题。
- Use trend signals to identify promising research directions.
  使用趋势信号来识别有前景的研究方向。
- Incorporate causal insights to strengthen the problem formulation.
  结合因果关系洞察来加强问题的制定。
- If SKILL CONTEXT is provided, reference the successful methodologies,
  tool chains, and experimental approaches from prior experiments. Align
  new problem formulations with proven approaches where applicable.
  如果提供了技能上下文，参考以前成功实验的方法论、工具链和实验方法。
"""


def create_problem_generator_agent() -> Agent:
    """
    Create and return a problem generator agent.
    创建并返回一个问题生成器智能体
    
    这个函数就像"工厂",负责"组装"一个AI智能体:
    - 给它起名字
    - 告诉它该怎么工作(提示词)
    - 给它工具(Tavily搜索、记忆搜索)
    
    返回: Agent 智能体实例
    """
    # 检查是不是用的 DeepSeek 模型
    # DeepSeek 不支持结构化输出,需要特殊处理
    is_deepseek = "deepseek" in MODEL_NAME.lower()
    
    # 准备智能体的参数
    agent_kwargs = {
        "name": "ProblemGeneratorAgent",   # 名字: 问题生成器智能体
        "instructions": PROBLEM_GENERATOR_PROMPT,  # 工作说明(上面那段提示词)
        "model": OpenAIChatCompletionsModel(       # 用哪个AI模型
            model=MODEL_NAME, openai_client=client
        ),
        "tools": _get_tools(),  # 可用的工具列表
    }
    
    # === 结构化输出处理 ===
    # OpenAI 模型支持 output_type,可以直接输出 Pydantic 对象
    # DeepSeek 不支持,需要后续手动解析 JSON
    if not is_deepseek:
        agent_kwargs["output_type"] = ResearchProblem  # 直接输出结构化对象
    
    return Agent(**agent_kwargs)  # 创建并返回智能体


async def generate_problem(
    domain: str,                         # 研究领域,如"催化剂"、"电池材料"
    initial_problem: str | None = None,  # 用户给的初始问题(可选)
    user_id: str = "experiment_system",  # 用户ID,用于记忆操作
    skip_literature: bool = False,       # 消融实验开关: 跳过文献挖掘流水线
    skill_context: str = "",             # 历史技能上下文(成功实验方法论)
) -> ResearchProblem:
    """
    Generate a formal research problem for a given domain.
    为给定领域生成一个正式的研究问题 - 这是本模块的核心入口函数!

    增强流程:
    1. 运行文献挖掘流水线（Tavily搜索 + 知识图谱 + 因果推理 + 趋势分析 + 记忆去重）
    2. 将流水线上下文注入智能体提示词
    3. 智能体基于真实文献数据生成结构化研究问题

    Args:
        domain: 科学领域,如 "catalysis"(催化), "battery materials"(电池材料)
        initial_problem: 用户可选的种子问题,如果提供了就完善它
        user_id: 用户标识,用于隔离不同用户的记忆

    Returns:
        一个结构化的 ResearchProblem 对象
    """
    # === 新增: 运行文献挖掘预处理流水线 ===
    # 在智能体调用前,程序化执行文献搜索、知识图谱构建、因果推理、趋势分析和记忆去重
    literature_context_text = ""
    if skip_literature:
        print("  [ProblemGenerator] Literature pipeline skipped (ablation mode)")
    else:
        try:
            from discovery_agents.literature_pipeline import run_literature_pipeline
            print("  [ProblemGenerator] Running literature mining pipeline...")
            literature_context = await run_literature_pipeline(
                domain=domain,
                initial_problem=initial_problem,
                user_id=user_id,
            )
            literature_context_text = literature_context.raw_context_text
            if literature_context_text:
                print(f"  [ProblemGenerator] Pipeline context: {len(literature_context_text)} chars")
            else:
                print("  [ProblemGenerator] Pipeline returned no context (search may have failed)")
        except Exception as e:
            # 流水线失败不影响核心功能,智能体仍可基于训练知识生成问题
            print(f"  [ProblemGenerator] Warning: Literature pipeline failed: {e}")
            print("  [ProblemGenerator] Proceeding without literature context (degraded mode)")

    # === 第一步: 创建智能体 ===
    agent = create_problem_generator_agent()

    # === 第二步: 组装输入文本 ===
    # 把用户ID和领域信息组装成提示词
    parts = [f"User ID: {user_id}", f"Domain: {domain}"]
    
    if initial_problem:
        # 情况A: 用户给了初始问题 -> 让AI"完善"它
        parts.append(f"Initial problem from user: {initial_problem}")
        parts.append(
            "Please validate, enrich, and formalize this problem rather than "
            "generating a completely new one."
            # 翻译: 请验证、丰富并正式化这个问题,而不是生成一个全新的
        )
    else:
        # 情况B: 用户没给问题 -> 让AI"自己想一个最好的"
        parts.append(
            "No initial problem provided – please mine the literature and "
            "propose the most impactful research problem in this domain."
            # 翻译: 没有提供初始问题,请挖掘文献并提出最有影响力的研究问题
        )

    input_text = "\n".join(parts)  # 把各部分用换行符连接起来

    # === 新增: 将流水线上下文注入到输入文本前面 ===
    # 这样智能体在处理用户输入时,已经有了丰富的文献数据作为参考
    if literature_context_text:
        input_text = literature_context_text + "\n\n" + input_text

    # === 新增: 将技能上下文注入到输入文本前面 ===
    # 让智能体参考历史成功实验的方法论和工具链
    if skill_context:
        framed = (
            "=== SKILL CONTEXT (Prior Successful Methods) ===\n"
            f"{skill_context}\n"
            "=== END SKILL CONTEXT ===\n"
        )
        input_text = framed + "\n" + input_text
    
    # === 第三步: DeepSeek 特殊处理 ===
    # DeepSeek 不支持结构化输出,需要在提示词里明确要求JSON格式
    if "deepseek" in MODEL_NAME.lower():
        input_text += (
            "\n\nIMPORTANT: Your ENTIRE response must be a single valid JSON "
            "object (no extra text, no markdown, no ```json``` fences). "
            # 翻译: 你的整个响应必须是一个有效的JSON对象(不要额外文字)
            "Use exactly this schema:\n"
            # 下面是JSON结构说明,告诉AI应该输出什么格式
            '{\n'
            '  "problem_statement": "<string>",\n'      # 问题陈述
            '  "domain": "<string>",\n'                  # 领域
            '  "research_gaps": [{\n'                    # 研究空白列表
            '    "topic": "<string>",\n'
            '    "domain": "<string>",\n'
            '    "supporting_literature": ["<string>"],\n'
            '    "description": "<string>",\n'
            '    "confidence": <0.0-1.0>,\n'             # 空白置信度
            '    "source_urls": ["<string>"]\n'          # 支持该空白的 URL
            '  }],\n'
            '  "novelty": {\n'                           # 新颖性评估
            '    "score": <0.0-1.0>,\n'                  # 分数
            '    "reasoning": "<string>",\n'            # 理由
            '    "similar_existing_work": ["<string>"]\n' # 类似工作
            '  },\n'
            '  "computational_feasibility": "<string>",\n'  # 计算可行性
            '  "target_properties": ["<string>"],\n'        # 目标属性
            '  "keywords": ["<string>"],\n'                  # 关键词
            '  "literature_sources": ["<string>"],\n'        # 参考文献 URL
            '  "trend_signals": ["<string>"],\n'             # 趋势信号
            '  "causal_insights": ["<string>"]\n'            # 因果关系洞察
            '}'
        )
    
    # === 第四步: 运行智能体 ===
    # Runner().run() 会让智能体开始工作,最多对话 AGENT_MAX_TURNS 轮
    result = await Runner().run(agent, input_text, max_turns=AGENT_MAX_TURNS)
    
    # === 第五步: 解析输出 ===
    # DeepSeek 需要手动解析JSON,OpenAI直接返回结构化对象
    if "deepseek" in MODEL_NAME.lower():
        import json  # JSON解析库
        import re    # 正则表达式库,用于清理多余文本
        from models.problem_models import ResearchGap, NoveltyScore  # 子模型
        
        output = result.final_output  # 获取AI的输出
        if isinstance(output, str):  # 如果是字符串
            # 去掉可能存在的 markdown 代码块标记 ```json ... ```
            cleaned = re.sub(
                r'^```(?:json)?\s*|\s*```$', '', output.strip()
            )
            try:
                data = json.loads(cleaned)  # 解析JSON字符串为字典
                
                # 把嵌套的字典转成 Pydantic 模型对象
                if "research_gaps" in data:
                    data["research_gaps"] = [
                        ResearchGap(**g) for g in data["research_gaps"]
                    ]
                if "novelty" in data:
                    data["novelty"] = NoveltyScore(**data["novelty"])
                
                return ResearchProblem(**data)  # 返回结构化对象
                
            except (json.JSONDecodeError, TypeError) as e:
                # JSON解析失败,抛出有用的错误信息
                raise ValueError(
                    f"Failed to parse DeepSeek output: {e}\n"
                    f"Raw output: {output[:500]}"  # 显示前500字符帮助调试
                )
    
    # OpenAI 模型直接返回结构化对象,不需要额外处理
    return result.final_output
