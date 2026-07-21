"""
Hypothesis Generator Agent (Module 2a) - 假设生成器智能体(模块2a)

Multiple domain expert agents generate competing hypotheses, then debate
多个领域专家智能体生成竞争性假设,然后进行辩论筛选

这个模块是整个系统的"头脑风暴团队":
- 3位专家(化学家、物理学家、ML专家)并行提出假设
- 1位辩论调解员评判筛选
- GAN风格的生成器-验证器对抗优化,最大化科学新颖性
- 最终输出精练后的假设列表
"""

from __future__ import annotations  # 启用未来版本的类型注解语法
import asyncio  # 异步编程库,让3个专家可以"同时"工作,不用排队
import json  # JSON解析库,用于手动解析大模型返回的JSON字符串
import re  # 正则表达式库,用于清理大模型输出中的markdown标记
from typing import List, Optional  # 类型提示工具
import os  # 操作系统库
import sys  # 系统库,用于修改模块搜索路径
from pathlib import Path  # 路径处理库

# === 设置模块搜索路径 ===
# 获取当前文件的父父目录(项目根目录)
current_dir = Path(__file__).resolve().parent.parent
# 如果这个目录还不在Python的搜索路径里,就把它加进去
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))

import _utf8_env  # noqa: F401  # 全局 UTF-8 编码修复

# === 导入AI智能体框架 ===
from agents import Agent, Runner  # Agent=智能体类, Runner=运行器(让智能体开始工作)
from agents import OpenAIChatCompletionsModel  # 用于连接OpenAI兼容的API

# === 导入项目配置 ===
from config import client, MODEL_NAME, AGENT_MAX_TURNS, GAN_ADVERSARIAL_ROUNDS
# client = API客户端(已经配置好密钥的)
# MODEL_NAME = 用哪个模型,如"o3"或"deepseek-chat"
# AGENT_MAX_TURNS = 智能体最多对话几轮(防止死循环)
# GAN_ADVERSARIAL_ROUNDS = GAN对抗优化轮数(默认3轮)

# === 导入数据模型 ===
from models.problem_models import ResearchProblem  # 研究问题的数据结构
from models.hypothesis_models import (
    Hypothesis, DebateOutcome,  # 假设、辩论结果
    VerifierAssessment, GANRoundResult, GANOutcome,  # GAN对抗相关模型
)

# === 工具列表(懒加载) ===
# 为什么要懒加载?因为加载工具可能很慢,启动时不加载,用到时再加载
_tools = None  # 初始为空,第一次调用时才填充


def _get_tools():
    """
    Get and return tool list, load on first call
    获取并返回工具列表,第一次调用时加载
    
    工具就像是智能体的"双手":
    - Tavily搜索: 让智能体能上网查文献
    - Memory搜索: 让智能体能查以前的发现记录
    
    返回: 工具列表(可能为空,如果工具都加载失败的话)
    """
    global _tools  # 使用全局变量,这样只加载一次
    if _tools is None:  # 如果还没加载过
        _tools = []  # 初始化为空列表
        # DeepSeek模型不支持工具调用,直接返回空列表
        if "deepseek" in MODEL_NAME.lower():
            return _tools
        # 加载Tavily搜索工具(网络搜索API,用来查文献)
        from config import load_tavily_search
        _tools = [load_tavily_search()]
        # 尝试加载记忆工具(用来搜索以前的发现记录)
        try:
            from config import load_memory_tools
            mem = load_memory_tools()
            _tools.append(mem.search_memory)  # 只取其中的搜索功能
        except Exception:
            pass  # 加载失败也没关系,不影响运行
    return _tools


def _is_deepseek() -> bool:
    """
    Check if using DeepSeek model
    检查是不是使用DeepSeek模型
    
    为什么要单独检查?因为DeepSeek:
    - 不支持结构化输出(output_type)
    - 不支持工具调用(tools)
    - 需要手动解析JSON
    """
    return "deepseek" in MODEL_NAME.lower()


def _create_agent_kwargs(
    name: str,           # 智能体名字
    instructions: str,   # 工作说明(提示词)
    output_type=None,    # 输出类型(Pydantic模型)
    tools=None           # 可用工具列表
) -> dict:
    """
    Create agent kwargs ensuring DeepSeek compatibility
    创建智能体参数,确保DeepSeek兼容性
    
    这是一个"工厂函数",统一创建智能体参数:
    - 名字、提示词、模型都是必须的
    - 工具和输出类型可选
    - DeepSeek模型不能有output_type
    """
    kwargs = {
        "name": name,                # 智能体名字
        "instructions": instructions,  # 工作说明
        "model": OpenAIChatCompletionsModel(  # 连接到AI模型
            model=MODEL_NAME, openai_client=client
        ),
    }
    # 如果有工具,添加工具
    if tools is not None:
        kwargs["tools"] = tools
    # 只有非DeepSeek模型才能添加output_type
    if output_type is not None and not _is_deepseek():
        kwargs["output_type"] = output_type
    return kwargs


# ============================================================
# 三位专家的"工作说明书" - 每位专家有不同的专业视角
# ============================================================

# === 专家1: 化学家 ===
# 从分子层面看问题: 电子结构、反应机理、结构-性质关系
EXPERT_CHEMIST_PROMPT = """
You are an expert computational chemist specializing in catalysis and materials.
你是一个专注于催化和材料的计算化学专家。

Generate a novel, testable hypothesis for the given research problem.
为给定的研究问题生成一个新颖、可测试的假设。

Your hypothesis should be based on first-principles understanding of:
你的假设应该基于第一性原理的理解:
- Electronic structure and bonding (电子结构和成键)
- Reaction mechanisms and kinetics (反应机理和动力学)
- Structure-property relationships (结构-性质关系)

IMPORTANT: Return a valid JSON object with these exact fields:
重要: 返回一个包含以下字段的有效JSON对象:
- id: a unique string identifier (e.g., "hyp_001") (唯一标识符)
- statement: the hypothesis statement (string) (假设陈述)
- scientific_basis: scientific reasoning (string) (科学依据)
- predicted_outcome: expected experimental result (string) (预期结果)
- confidence: confidence level 0.0-1.0 (number) (置信度)
- generation: generation number, use 1 (integer) (代数,用1)
- status: "active" (string) (状态)
"""

# === 专家2: 物理学家 ===
# 从物理原理看问题: 热力学、量子力学、输运现象
EXPERT_PHYSICIST_PROMPT = """
You are an expert computational physicist specializing in condensed matter.
你是一个专注于凝聚态物理的计算物理学家。

Generate a novel, testable hypothesis for the given research problem.
为给定的研究问题生成一个新颖、可测试的假设。

Your hypothesis should be based on physical principles:
你的假设应该基于物理原理:
- Thermodynamics and statistical mechanics (热力学和统计力学)
- Quantum mechanics and band structure (量子力学和能带结构)
- Transport phenomena (输运现象)

IMPORTANT: Return a valid JSON object with these exact fields:
重要: 返回一个包含以下字段的有效JSON对象:
- id: a unique string identifier (e.g., "hyp_002") (唯一标识符)
- statement: the hypothesis statement (string) (假设陈述)
- scientific_basis: scientific reasoning (string) (科学依据)
- predicted_outcome: expected experimental result (string) (预期结果)
- confidence: confidence level 0.0-1.0 (number) (置信度)
- generation: generation number, use 1 (integer) (代数,用1)
- status: "active" (string) (状态)
"""

# === 专家3: 机器学习专家 ===
# 从数据角度看问题: 模式识别、统计学习、高通量筛选
EXPERT_ML_PROMPT = """
You are an expert in machine learning for materials science.
你是一个专注于材料科学的机器学习专家。

Generate a novel, testable hypothesis for the given research problem.
为给定的研究问题生成一个新颖、可测试的假设。

Your hypothesis should leverage:
你的假设应该利用:
- Data-driven pattern recognition (数据驱动的模式识别)
- Statistical learning theory (统计学习理论)
- High-throughput screening approaches (高通量筛选方法)

IMPORTANT: Return a valid JSON object with these exact fields:
重要: 返回一个包含以下字段的有效JSON对象:
- id: a unique string identifier (e.g., "hyp_003") (唯一标识符)
- statement: the hypothesis statement (string) (假设陈述)
- scientific_basis: scientific reasoning (string) (科学依据)
- predicted_outcome: expected experimental result (string) (预期结果)
- confidence: confidence level 0.0-1.0 (number) (置信度)
- generation: generation number, use 1 (integer) (代数,用1)
- status: "active" (string) (状态)
"""

# ============================================================
# 辩论调解员的"工作说明书" - 负责评判和筛选假设
# ============================================================
DEBATE_MEDIATOR_PROMPT = """
You are a debate mediator evaluating competing scientific hypotheses.
你是一个评估竞争性科学假设的辩论调解员。

Given multiple hypotheses for the same research problem:
给定同一研究问题的多个假设:

1. Analyze each hypothesis for:
   分析每个假设的:
   - Scientific validity and grounding (科学有效性和基础)
   - Novelty and originality (新颖性和原创性)
   - Testability with computational methods (计算方法的可测试性)
   - Potential impact (潜在影响)

2. Compare hypotheses against each other
   相互比较假设

3. Decide which to keep, refine, or eliminate
   决定保留、改进或淘汰哪些

4. Output a structured DebateOutcome with refined hypotheses
   输出包含精炼假设的结构化辩论结果

IMPORTANT: Return a valid JSON object with these exact fields:
重要: 返回一个包含以下字段的有效JSON对象:
- original_hypotheses: array of hypothesis objects (原始假设数组)
- refined_hypotheses: array of refined hypothesis objects (精炼后的假设数组)
- eliminated_count: number of eliminated hypotheses (integer) (淘汰数量)
- debate_summary: summary of the debate (string) (辩论总结)
"""


# ============================================================
# 假设字段类型修复工具函数(模块级,供辩论解析和GAN阶段共用)
# ============================================================
def fix_hypothesis(hyp):
    """
    Fix hypothesis field types from LLM JSON output
    修复大模型JSON输出中假设字段的类型问题

    大模型有时会:
    - 把数字写成字符串: "1" 而不是 1
    - 把置信度写成 "high" 而不是 0.8
    - 漏掉某些必填字段
    这个函数统一修复这些问题
    """
    if not isinstance(hyp, dict):  # 如果不是字典,直接返回
        return hyp

    # --- 修复id: 确保是字符串 ---
    if 'id' in hyp and not isinstance(hyp['id'], str):
        hyp['id'] = str(hyp['id'])

    # --- 修复generation: 确保是整数 ---
    if 'generation' in hyp:
        if isinstance(hyp['generation'], str):
            try:
                hyp['generation'] = int(hyp['generation'])
            except ValueError:
                hyp['generation'] = 1
        elif not isinstance(hyp['generation'], int):
            hyp['generation'] = 1
    else:
        hyp['generation'] = 1

    # --- 修复confidence: 确保是浮点数 ---
    if 'confidence' in hyp:
        if isinstance(hyp['confidence'], str):
            confidence_map = {
                'high': 0.8, 'h': 0.8,
                'medium': 0.5, 'med': 0.5, 'm': 0.5,
                'low': 0.3, 'l': 0.3,
            }
            lower_val = hyp['confidence'].lower().strip()
            if lower_val in confidence_map:
                hyp['confidence'] = confidence_map[lower_val]
            else:
                try:
                    hyp['confidence'] = float(hyp['confidence'])
                except ValueError:
                    hyp['confidence'] = 0.5
        elif not isinstance(hyp['confidence'], (int, float)):
            hyp['confidence'] = 0.5
    else:
        hyp['confidence'] = 0.5

    # --- 确保必填字符串字段存在 ---
    if 'statement' not in hyp or not hyp['statement']:
        hyp['statement'] = "No statement provided"
    if 'scientific_basis' not in hyp or not hyp['scientific_basis']:
        hyp['scientific_basis'] = "Based on expert analysis"
    if 'predicted_outcome' not in hyp or not hyp['predicted_outcome']:
        hyp['predicted_outcome'] = "To be determined"
    if 'status' not in hyp or not hyp['status']:
        hyp['status'] = "active"

    return hyp


def _parse_json_from_llm(raw_output: str) -> dict:
    """
    Parse JSON from LLM string output with markdown cleanup
    从大模型的字符串输出中解析JSON,自动清理markdown标记

    返回: 解析后的字典,解析失败则抛出异常
    """
    cleaned = re.sub(r'^```(?:json)?\s*|\s*```$', '', raw_output.strip())
    return json.loads(cleaned)


# ============================================================
# 运行单个专家的函数
# ============================================================
async def _run_expert_agent(
    agent: Agent,                              # 要运行的专家智能体
    problem: ResearchProblem,                  # 研究问题
    prior_hypotheses: Optional[List[Hypothesis]],  # 之前的假设(避免重复)
    skill_context: str = "",                   # 历史技能上下文(成功实验方法论)
) -> Hypothesis:
    """
    Run a single expert agent to generate a hypothesis
    运行单个专家智能体生成假设
    
    这个函数负责"让一个专家工作":
    - 告诉专家:问题是什么、领域是什么、目标属性是什么
    - 如果有历史假设,告诉专家"别重复这些"
    - 运行后解析输出,如果解析失败就创建一个基本假设
    """
    # 如果有之前的假设,告诉专家避免重复
    prior_text = ""
    if prior_hypotheses:
        prior_text = "\nPreviously proposed hypotheses (avoid repeating):\n" + "\n".join(
            f"- {h.statement}" for h in prior_hypotheses
        )
    
    # 组装输入文本 - 告诉专家所有相关信息
    input_text = (
        f"Research Problem: {problem.problem_statement}\n"  # 研究问题
        f"Domain: {problem.domain}\n"                       # 领域
        f"Target properties: {', '.join(problem.target_properties)}\n"  # 目标属性
        f"{prior_text}\n\n"                                  # 历史假设(如果有)
        "Generate a novel hypothesis."                       # 请生成新假设
    )

    # === 注入技能上下文 ===
    if skill_context:
        input_text = (
            "=== SKILL CONTEXT (Prior Successful Methods) ===\n"
            f"{skill_context}\n"
            "=== END SKILL CONTEXT ===\n\n"
        ) + input_text
    
    # === 运行智能体 ===
    # 这里是真正连接大模型的地方!
    result = await Runner().run(agent, input_text, max_turns=AGENT_MAX_TURNS)
    output = result.final_output  # 获取AI的输出
    
    # === DeepSeek需要手动解析JSON ===
    if _is_deepseek() and isinstance(output, str):
        try:
            data = _parse_json_from_llm(output)
            output = Hypothesis(**data)  # 转换为Pydantic对象
        except (json.JSONDecodeError, TypeError) as e:
            # 解析失败,创建一个基本假设(容错处理)
            print(f"    Warning: Could not parse hypothesis: {e}")
            output = Hypothesis(
                id=f"hyp_{asyncio.get_event_loop().time()}",  # 生成唯一ID
                statement=output[:200] if len(output) > 200 else output,  # 截取前200字
                scientific_basis="Generated by expert agent",
                predicted_outcome="To be determined through experiment"
            )
    
    return output  # 返回假设对象


# ============================================================
# 核心函数: 生成假设并进行辩论
# 这是本模块的主入口,被orchestrator.py调用
# ============================================================
async def generate_and_debate_hypotheses(
    problem: ResearchProblem,                          # 输入: 研究问题
    prior_hypotheses: Optional[List[Hypothesis]] = None,  # 输入: 之前的假设(避免重复)
    user_id: str = "experiment_system",                # 输入: 用户ID(用于记忆系统)
    skill_context: str = "",                           # 输入: 历史技能上下文
) -> DebateOutcome:  # 输出: 辩论结果(包含精炼后的假设)
    """
    Generate hypotheses from multiple experts and debate to refine
    让多位专家生成假设,然后通过辩论精炼
    
    整体流程:
    1. 创建3个专家智能体(化学家、物理学家、ML专家)
    2. 并行运行3个专家,各自生成1个假设
    3. 创建辩论调解员智能体
    4. 调解员评判、筛选、精炼这些假设
    5. 返回辩论结果
    
    这就像一个学术研讨会:
    - 3位不同背景的科学家各自提出自己的理论
    - 然后一位主持人综合评判,选出最好的方案
    """
    
    # ========== 第一阶段: 创建3个专家智能体 ==========
    # 每个专家有不同的专业背景和思维方式
    
    # === 专家1: 化学家 ===
    # 从分子、电子结构、反应机理的角度思考
    chemist_agent = Agent(
        **_create_agent_kwargs(              # 使用工厂函数创建参数
            "ExpertChemist",                 # 名字: 化学专家
            EXPERT_CHEMIST_PROMPT,           # 工作说明: 化学家的提示词
            None if _is_deepseek() else Hypothesis,  # 输出类型(DeepSeek不支持)
            _get_tools()                     # 可用工具(搜索等)
        )
    )
    
    # === 专家2: 物理学家 ===
    # 从热力学、量子力学、能带结构的角度思考
    physicist_agent = Agent(
        **_create_agent_kwargs(
            "ExpertPhysicist",               # 名字: 物理专家
            EXPERT_PHYSICIST_PROMPT,         # 工作说明: 物理学家的提示词
            None if _is_deepseek() else Hypothesis,
            _get_tools()
        )
    )
    
    # === 专家3: 机器学习专家 ===
    # 从数据驱动、模式识别、高通量筛选的角度思考
    ml_agent = Agent(
        **_create_agent_kwargs(
            "ExpertML",                      # 名字: ML专家
            EXPERT_ML_PROMPT,                # 工作说明: ML专家的提示词
            None if _is_deepseek() else Hypothesis,
            _get_tools()
        )
    )
    
    # ========== 第二阶段: 并行运行3个专家 ==========
    # asyncio.gather的魔力: 3个专家"同时"工作,而不是排队
    # 这就像3个人同时在各自的房间里思考,大大节省时间!
    print("  Running expert agents in parallel...")  # 日志: 并行运行中
    print("  正在并行运行专家智能体...")
    
    expert_results = await asyncio.gather(
        _run_expert_agent(chemist_agent, problem, prior_hypotheses, skill_context),   # 化学家工作
        _run_expert_agent(physicist_agent, problem, prior_hypotheses, skill_context), # 物理学家工作
        _run_expert_agent(ml_agent, problem, prior_hypotheses, skill_context),        # ML专家工作
        return_exceptions=True  # 重要: 某个专家失败不影响其他专家
        # 如果不设这个,一个专家失败整个gather就失败了
    )
    
    # ========== 收集专家的假设 ==========
    # 遍历每个专家的结果,成功的加入列表,失败的打印错误
    original_hypotheses = []  # 存放所有成功生成的假设
    for i, result in enumerate(expert_results):
        if isinstance(result, Exception):  # 如果是异常(专家失败了)
            print(f"    Expert {i+1} failed: {result}")
            print(f"    专家{i+1}失败: {result}")
        else:  # 专家成功生成了假设
            original_hypotheses.append(result)  # 加入列表
            # 打印假设的前60个字符(预览)
            print(f"    Expert {i+1} generated hypothesis: {result.statement[:60]}...")
            print(f"    专家{i+1}生成假设: {result.statement[:60]}...")
    
    # ========== 边界情况: 所有专家都失败了 ==========
    if not original_hypotheses:  # 如果一个假设都没有
        print("  Warning: No hypotheses generated, returning empty outcome")
        print("  警告: 没有生成任何假设,返回空结果")
        return DebateOutcome(
            original_hypotheses=[],          # 原始假设: 空
            refined_hypotheses=[],           # 精炼假设: 空
            eliminated_count=0,              # 淘汰数量: 0
            debate_summary="No hypotheses could be generated"  # 总结
        )
    
    # ========== 第三阶段: 辩论阶段 ==========
    # 创建辩论调解员,让它评判和筛选假设
    print("  Running debate mediation...")
    print("  正在运行辩论调解...")
    
    # 创建辩论调解员智能体
    debate_agent = Agent(
        **_create_agent_kwargs(
            "DebateMediator",                # 名字: 辩论调解员
            DEBATE_MEDIATOR_PROMPT,          # 工作说明: 调解员的提示词
            None if _is_deepseek() else DebateOutcome,  # 输出类型
            # 注意: 调解员没有工具,只需要思考和判断
        )
    )
    
    # ========== 组装辩论输入 ==========
    # 把所有假设格式化成文本,让调解员能理解
    hypotheses_text = "\n\n".join(
        f"Hypothesis {i+1}:\n"                      # 假设编号
        f"  Statement: {h.statement}\n"             # 假设陈述
        f"  Basis: {h.scientific_basis}\n"          # 科学依据
        f"  Predicted: {h.predicted_outcome}"       # 预期结果
        for i, h in enumerate(original_hypotheses)  # 遍历所有假设
    )
    
    # 完整的辩论输入
    debate_input = (
        f"Research Problem: {problem.problem_statement}\n\n"  # 研究问题
        f"Proposed Hypotheses:\n{hypotheses_text}\n\n"        # 所有假设
        "Evaluate and refine these hypotheses."               # 请评估和精炼
    )
    
    # ========== 运行辩论调解员 ==========
    try:
        # 这里是真正调用大模型的地方!
        debate_result = await Runner().run(
            debate_agent,          # 辩论调解员智能体
            debate_input,          # 辩论输入(问题+所有假设)
            max_turns=AGENT_MAX_TURNS  # 最大对话轮数
        )
        output = debate_result.final_output  # 获取最终输出
        
        # ========== DeepSeek特殊处理: 手动解析JSON ==========
        # DeepSeek不支持结构化输出,返回的是字符串,需要手动解析
        if _is_deepseek() and isinstance(output, str):
            try:
                data = _parse_json_from_llm(output)
                
                # --- 使用模块级fix_hypothesis修复假设列表 ---
                if 'original_hypotheses' in data and isinstance(data['original_hypotheses'], list):
                    data['original_hypotheses'] = [fix_hypothesis(h) for h in data['original_hypotheses']]
                
                if 'refined_hypotheses' in data and isinstance(data['refined_hypotheses'], list):
                    data['refined_hypotheses'] = [fix_hypothesis(h) for h in data['refined_hypotheses']]
                
                # --- 修复淘汰数量: 确保是整数 ---
                if 'eliminated_count' in data and isinstance(data['eliminated_count'], str):
                    try:
                        data['eliminated_count'] = int(data['eliminated_count'])
                    except ValueError:
                        data['eliminated_count'] = 0  # 转换失败用0
                
                # 最终转换为Pydantic对象
                output = DebateOutcome(**data)
                
            except (json.JSONDecodeError, TypeError) as e:
                # JSON解析失败,打印警告并返回所有原始假设
                print(f"    Warning: Could not parse debate outcome: {e}")
                print(f"    警告: 无法解析辩论结果: {e}")
                # 容错: 保留所有原始假设,不做筛选
                output = DebateOutcome(
                    original_hypotheses=original_hypotheses,
                    refined_hypotheses=original_hypotheses,  # 精炼=原始(未筛选)
                    eliminated_count=0,
                    debate_summary="Debate parsing failed, keeping all hypotheses"
                )
        
        return output  # 返回辩论结果
        
    except Exception as e:
        # ========== 异常处理: 辩论整体失败 ==========
        # 可能是网络问题、API限流等导致辩论失败
        # 容错策略: 保留所有原始假设,让流程继续
        print(f"  Warning: Debate failed: {e}, keeping all hypotheses")
        print(f"  警告: 辩论失败: {e}, 保留所有假设")
        return DebateOutcome(
            original_hypotheses=original_hypotheses,
            refined_hypotheses=original_hypotheses,  # 精炼=原始
            eliminated_count=0,
            debate_summary=f"Debate failed: {e}"  # 记录失败原因
        )


# ============================================================
# GAN 风格双智能体对抗优化 - 最大化科学新颖性
# ============================================================
# 在多专家辩论之后运行, 由"范式颠覆者"(生成器)和"文献怀疑论者"(验证器)
# 通过多轮对抗迭代, 产出既大胆又有科学依据的高创新假设

# === GAN 生成器智能体的"工作说明书" ===
# 生成器的核心任务: 挑战现有范式, 提出颠覆性假设
GAN_GENERATOR_PROMPT = """
You are a revolutionary scientific thinker — a 'paradigm disruptor'.
你是一位革命性的科学思想家——"范式颠覆者"。

Your mission: Generate 1-2 DISRUPTIVE hypotheses that fundamentally challenge
the conventional hypotheses provided below. These are NOT incremental improvements
— they must represent paradigm shifts.
你的使命：生成1-2个从根本上挑战下方常规假设的颠覆性假设。这不是渐进式改进——
必须代表范式转变。

Think using these strategies:
使用以下策略思考：
- Cross-domain analogies (跨领域类比: 从其他科学领域借鉴机制)
- Contrarian interpretations of known data (逆向解读: 已知数据是否存在另一种解释?)
- Scaling law violations (标度律违反: 在极端条件下规律是否失效?)
- Emergent phenomena from simple rules (涌现现象: 简单规则能否产生复杂行为?)
- Unconventional mechanism pathways (非常规机制: 是否存在未被考虑的路径?)

CRITICAL: Do NOT produce variations of the conventional hypotheses.
Propose fundamentally DIFFERENT paradigms.
关键：不要产生常规假设的变体。提出根本不同的范式。

If you receive feedback from a previous round's Verifier, use it to refine
your proposals — make them bolder yet more scientifically defensible.
如果收到上一轮验证器的反馈，利用它改进你的提案——使其更大胆且更具科学依据。

IMPORTANT: Return a valid JSON object with this structure:
重要：返回以下结构的有效JSON对象：
{
  "disruptive_hypotheses": [
    {
      "id": "gan_hyp_001",
      "statement": "hypothesis statement",
      "scientific_basis": "scientific reasoning for this paradigm shift",
      "predicted_outcome": "expected experimental result",
      "confidence": 0.6,
      "generation": 1,
      "status": "proposed"
    }
  ]
}
"""

# === GAN 验证器智能体的"工作说明书" ===
# 验证器的核心任务: 评估颠覆性假设的科学合理性, 不过分保守也不过分宽松
GAN_VERIFIER_PROMPT = """
You are a rigorous scientific skeptic and literature expert.
你是一位严谨的科学怀疑论者和文献专家。

Your mission: Evaluate each disruptive hypothesis for scientific plausibility.
你的使命：评估每个颠覆性假设的科学合理性。

The bar is "scientifically defensible", NOT "already proven".
We WANT bold ideas — your job is to filter out the impossible, not the unproven.
标准是"科学上可辩护"，而不是"已被证明"。
我们需要大胆的想法——你的任务是过滤掉不可能的，而不是未被证明的。

Evaluation criteria:
评估标准：
1. Does it violate established physical/chemical laws? → REJECT
   是否违反已确立的物理/化学定律？→ 拒绝
2. Does it contradict overwhelming experimental evidence? → REJECT
   是否与大量实验证据矛盾？→ 拒绝
3. Is there any theoretical or empirical basis, even indirect? → ACCEPT or REVISE
   是否有任何理论或实验依据（即使是间接的）？→ 接受或修改
4. Is it actually novel, or does prior work already exist? → Score novelty
   是否真正新颖，还是已有先前工作？→ 评分

Use your search tools to find relevant literature evidence when available.
有搜索工具时，使用它们查找相关文献证据。

IMPORTANT: Return a valid JSON object with this structure:
重要：返回以下结构的有效JSON对象：
{
  "assessments": [
    {
      "hypothesis_id": "gan_hyp_001",
      "plausibility_score": 0.7,
      "novelty_score": 0.85,
      "literature_conflicts": ["Paper X shows..."],
      "literature_support": ["Theory Y suggests..."],
      "verdict": "accept",
      "revision_suggestions": ""
    }
  ],
  "round_summary": "Overall assessment of this round..."
}

Verdict must be one of: "accept", "revise", "reject"
判定必须是以下之一: "accept"(接受), "revise"(需修改), "reject"(拒绝)
"""


# ============================================================
# GAN 对抗优化核心函数
# ============================================================
async def run_gan_adversarial_refinement(
    problem: ResearchProblem,                          # 输入: 研究问题
    debate_hypotheses: List[Hypothesis],               # 输入: 辩论精炼后的假设
    num_rounds: int = GAN_ADVERSARIAL_ROUNDS,          # 输入: 对抗轮数(默认3轮)
    user_id: str = "experiment_system",                # 输入: 用户ID
    skill_context: str = "",                           # 输入: 历史技能上下文
) -> GANOutcome:  # 输出: GAN对抗优化的完整结果
    """
    Run GAN-style adversarial refinement to maximize scientific novelty
    运行GAN风格的对抗优化以最大化科学新颖性

    整体流程:
    1. 对每一轮(共num_rounds轮):
       a. 生成器提出1-2个颠覆性假设(挑战常规假设的范式)
       b. 验证器评估每个假设(文献检索+合理性判断)
       c. 接受的加入最终列表, 需修改的反馈给下一轮, 拒绝的丢弃
    2. 将通过验证的GAN假设追加合并到辩论假设列表

    这就像一场学术答辩:
    - "颠覆者"提出大胆理论
    - "审查员"严格评审
    - 经过多轮打磨, 只有站得住脚的创新性假设才能通过
    """
    print("  Running GAN adversarial refinement...")
    print("  正在运行GAN对抗性优化...")

    rounds: List[GANRoundResult] = []    # 各轮结果记录
    all_accepted: List[Hypothesis] = []  # 所有轮次中通过验证的假设
    verifier_feedback = ""               # 上一轮验证器的反馈(初始为空)

    try:
        for round_num in range(1, num_rounds + 1):
            print(f"    GAN Round {round_num}/{num_rounds}...")
            print(f"    GAN 第{round_num}/{num_rounds}轮...")

            # ========== 阶段1: 运行生成器 ==========
            proposed = await _run_gan_generator(
                problem, debate_hypotheses, verifier_feedback, round_num, skill_context
            )

            if not proposed:
                # 生成器本轮失败,记录空轮次并继续
                print(f"    GAN Generator failed in round {round_num}, skipping")
                rounds.append(GANRoundResult(
                    round_number=round_num,
                    proposed_hypotheses=[],
                    revision_feedback=verifier_feedback,
                ))
                continue

            for h in proposed:
                print(f"      [Generator] {h.statement[:70]}...")

            # ========== 阶段2: 运行验证器 ==========
            assessments, round_summary = await _run_gan_verifier(
                problem, debate_hypotheses, proposed
            )

            # ========== 阶段3: 处理验证结果 ==========
            round_accepted = []
            revision_parts = []

            # 为每个假设匹配验证结果
            assessment_map = {a.hypothesis_id: a for a in assessments}
            for hyp in proposed:
                assessment = assessment_map.get(hyp.id)
                if assessment is None:
                    # 验证器没有评估这个假设(可能因解析问题遗漏),乐观接受
                    round_accepted.append(hyp)
                    continue

                if assessment.verdict == "accept":
                    round_accepted.append(hyp)
                    print(f"      [Verifier] ACCEPT: {hyp.id} "
                          f"(plausibility={assessment.plausibility_score:.2f}, "
                          f"novelty={assessment.novelty_score:.2f})")
                elif assessment.verdict == "revise":
                    revision_parts.append(
                        f"Hypothesis '{hyp.statement[:80]}': {assessment.revision_suggestions}"
                    )
                    print(f"      [Verifier] REVISE: {hyp.id} - {assessment.revision_suggestions[:60]}...")
                else:  # reject
                    print(f"      [Verifier] REJECT: {hyp.id} "
                          f"(conflicts: {', '.join(assessment.literature_conflicts[:2])})")

            # 构建下一轮的反馈
            verifier_feedback = "\n".join(revision_parts) if revision_parts else ""

            # 记录本轮结果
            rounds.append(GANRoundResult(
                round_number=round_num,
                proposed_hypotheses=proposed,
                assessments=assessments,
                accepted_hypotheses=round_accepted,
                revision_feedback=verifier_feedback,
            ))
            all_accepted.extend(round_accepted)

        # ========== 合并: 辩论假设 + GAN通过假设 ==========
        # 将通过验证的GAN假设状态改为active
        for h in all_accepted:
            h.status = "active"

        final_hypotheses = list(debate_hypotheses) + all_accepted

        # 构建总结
        gan_summary = (
            f"GAN adversarial refinement completed: {len(rounds)} rounds, "
            f"{sum(len(r.proposed_hypotheses) for r in rounds)} total proposed, "
            f"{len(all_accepted)} accepted."
        )

        print(f"    GAN Summary: {gan_summary}")
        print(f"    GAN 总结: {len(all_accepted)}个颠覆性假设通过验证")

        return GANOutcome(
            input_hypotheses=debate_hypotheses,
            rounds=rounds,
            final_hypotheses=final_hypotheses,
            total_proposed=sum(len(r.proposed_hypotheses) for r in rounds),
            total_accepted=len(all_accepted),
            gan_summary=gan_summary,
        )

    except Exception as e:
        # ========== 整体容错: GAN阶段失败不影响主流程 ==========
        print(f"  Warning: GAN adversarial refinement failed: {e}")
        print(f"  警告: GAN对抗优化失败: {e}, 保留辩论结果")
        return GANOutcome(
            input_hypotheses=debate_hypotheses,
            rounds=rounds,
            final_hypotheses=list(debate_hypotheses),
            total_proposed=sum(len(r.proposed_hypotheses) for r in rounds),
            total_accepted=0,
            gan_summary=f"GAN refinement failed: {e}",
        )


# ============================================================
# GAN 内部辅助函数: 运行生成器
# ============================================================
async def _run_gan_generator(
    problem: ResearchProblem,
    conventional_hypotheses: List[Hypothesis],
    verifier_feedback: str,
    round_num: int,
    skill_context: str = "",                           # 历史技能上下文
) -> List[Hypothesis]:
    """
    Run the GAN Generator agent to produce disruptive hypotheses
    运行GAN生成器智能体产生颠覆性假设

    参数:
        problem: 研究问题
        conventional_hypotheses: 常规假设(辩论精炼后的, 作为"要挑战的对象")
        verifier_feedback: 上一轮验证器的反馈(第一轮为空)
        round_num: 当前轮次编号
    返回:
        颠覆性假设列表(可能为空)
    """
    # 格式化常规假设为文本
    conventional_text = "\n".join(
        f"  - [{h.id}] {h.statement}" for h in conventional_hypotheses
    )

    # 组装生成器输入
    feedback_section = ""
    if verifier_feedback:
        feedback_section = (
            f"\n\n=== VERIFIER FEEDBACK FROM PREVIOUS ROUND ===\n"
            f"(Use this to improve your proposals)\n"
            f"(利用以下反馈改进你的提案)\n"
            f"{verifier_feedback}\n"
            f"=== END FEEDBACK ===\n"
        )

    input_text = (
        f"Research Problem: {problem.problem_statement}\n"
        f"Domain: {problem.domain}\n"
        f"Target properties: {', '.join(problem.target_properties)}\n\n"
        f"=== CONVENTIONAL HYPOTHESES (challenge these!) ===\n"
        f"(以下是常规假设, 请提出根本不同的范式来挑战它们!)\n"
        f"{conventional_text}\n"
        f"{feedback_section}\n"
        f"Generate 1-2 DISRUPTIVE hypotheses for round {round_num}. "
        f"Use IDs like 'gan_hyp_{round_num:02d}_01', 'gan_hyp_{round_num:02d}_02'."
    )

    # === 注入技能上下文 ===
    if skill_context:
        input_text = (
            "=== SKILL CONTEXT (Prior Successful Methods) ===\n"
            f"{skill_context}\n"
            "=== END SKILL CONTEXT ===\n\n"
        ) + input_text

    try:
        # 创建生成器智能体 - 给予搜索工具用于查找灵感
        generator_agent = Agent(
            **_create_agent_kwargs(
                "GANGenerator",         # 名字: GAN生成器
                GAN_GENERATOR_PROMPT,   # 工作说明: 颠覆者的提示词
                tools=_get_tools(),     # 给予搜索工具
            )
        )

        result = await Runner().run(generator_agent, input_text, max_turns=AGENT_MAX_TURNS)
        output = result.final_output

        # 解析输出 - 统一处理(不区分DeepSeek/非DeepSeek,都手动解析)
        if isinstance(output, str):
            try:
                data = _parse_json_from_llm(output)
            except (json.JSONDecodeError, TypeError):
                # JSON解析失败, 尝试从文本中创建一个基本假设
                print(f"    Warning: Could not parse generator JSON, creating fallback")
                return [Hypothesis(
                    id=f"gan_hyp_{round_num:02d}_01",
                    statement=output[:300] if len(output) > 300 else output,
                    scientific_basis="Generated by GAN generator agent",
                    predicted_outcome="To be determined",
                    confidence=0.5,
                    generation=round_num,
                    status="proposed",
                )]
        else:
            # 结构化输出(非DeepSeek模型可能直接返回对象)
            # 尝试转为字典处理
            if hasattr(output, 'model_dump'):
                data = output.model_dump()
            else:
                data = {"disruptive_hypotheses": []}

        # 提取假设列表
        raw_hypotheses = data.get("disruptive_hypotheses", [])
        if not isinstance(raw_hypotheses, list):
            raw_hypotheses = []

        # 修复并转换为Hypothesis对象
        hypotheses = []
        for i, raw_h in enumerate(raw_hypotheses[:2]):  # 最多取2个
            fixed = fix_hypothesis(raw_h)
            if isinstance(fixed, dict):
                # 确保GAN假设有正确的ID前缀和状态
                if not fixed.get('id', '').startswith('gan_'):
                    fixed['id'] = f"gan_hyp_{round_num:02d}_{i+1:02d}"
                fixed['status'] = 'proposed'
                if 'generation' not in fixed or fixed['generation'] == 1:
                    fixed['generation'] = round_num
                try:
                    hypotheses.append(Hypothesis(**fixed))
                except Exception as e:
                    print(f"    Warning: Could not create hypothesis: {e}")

        return hypotheses

    except Exception as e:
        print(f"    Warning: GAN Generator failed: {e}")
        return []


# ============================================================
# GAN 内部辅助函数: 运行验证器
# ============================================================
async def _run_gan_verifier(
    problem: ResearchProblem,
    conventional_hypotheses: List[Hypothesis],
    proposed_hypotheses: List[Hypothesis],
) -> tuple:
    """
    Run the GAN Verifier agent to evaluate disruptive hypotheses
    运行GAN验证器智能体评估颠覆性假设

    参数:
        problem: 研究问题
        conventional_hypotheses: 常规假设(作为上下文参考)
        proposed_hypotheses: 本轮生成器提出的颠覆性假设
    返回:
        (assessments列表, round_summary字符串)
    """
    # 格式化常规假设(作为参考上下文)
    conventional_text = "\n".join(
        f"  - [{h.id}] {h.statement}" for h in conventional_hypotheses
    )

    # 格式化待评估的颠覆性假设
    proposed_text = "\n\n".join(
        f"Disruptive Hypothesis [{h.id}]:\n"
        f"  Statement: {h.statement}\n"
        f"  Scientific Basis: {h.scientific_basis}\n"
        f"  Predicted Outcome: {h.predicted_outcome}\n"
        f"  Confidence: {h.confidence}"
        for h in proposed_hypotheses
    )

    input_text = (
        f"Research Problem: {problem.problem_statement}\n"
        f"Domain: {problem.domain}\n\n"
        f"=== CONVENTIONAL HYPOTHESES (for context) ===\n"
        f"{conventional_text}\n\n"
        f"=== DISRUPTIVE HYPOTHESES TO EVALUATE ===\n"
        f"{proposed_text}\n\n"
        f"Evaluate each disruptive hypothesis. For each, provide plausibility_score, "
        f"novelty_score, literature evidence, and a verdict (accept/revise/reject)."
    )

    try:
        # 创建验证器智能体 - 给予搜索工具用于文献验证
        verifier_agent = Agent(
            **_create_agent_kwargs(
                "GANVerifier",          # 名字: GAN验证器
                GAN_VERIFIER_PROMPT,    # 工作说明: 怀疑论者的提示词
                tools=_get_tools(),     # 给予搜索工具(验证器需要查文献!)
            )
        )

        result = await Runner().run(verifier_agent, input_text, max_turns=AGENT_MAX_TURNS)
        output = result.final_output

        # 解析验证结果
        if isinstance(output, str):
            try:
                data = _parse_json_from_llm(output)
            except (json.JSONDecodeError, TypeError):
                print(f"    Warning: Could not parse verifier JSON, auto-accepting all")
                # 解析失败,乐观回退: 自动接受所有假设
                return _fallback_accept_all(proposed_hypotheses)
        else:
            if hasattr(output, 'model_dump'):
                data = output.model_dump()
            else:
                return _fallback_accept_all(proposed_hypotheses)

        # 提取评估列表
        raw_assessments = data.get("assessments", [])
        round_summary = data.get("round_summary", "Assessment completed")

        assessments = []
        for raw_a in raw_assessments:
            if not isinstance(raw_a, dict):
                continue
            # 修复评估字段类型
            try:
                # 确保分数是浮点数
                for score_field in ('plausibility_score', 'novelty_score'):
                    if score_field in raw_a:
                        if isinstance(raw_a[score_field], str):
                            try:
                                raw_a[score_field] = float(raw_a[score_field])
                            except ValueError:
                                raw_a[score_field] = 0.5
                    else:
                        raw_a[score_field] = 0.5

                # 确保verdict是有效值
                if raw_a.get('verdict') not in ('accept', 'revise', 'reject'):
                    raw_a['verdict'] = 'accept'  # 默认乐观接受

                # 确保列表字段存在
                for list_field in ('literature_conflicts', 'literature_support'):
                    if list_field not in raw_a or not isinstance(raw_a[list_field], list):
                        raw_a[list_field] = []

                if 'revision_suggestions' not in raw_a:
                    raw_a['revision_suggestions'] = ""
                if 'hypothesis_id' not in raw_a:
                    raw_a['hypothesis_id'] = ""

                assessments.append(VerifierAssessment(**raw_a))
            except Exception as e:
                print(f"    Warning: Could not parse assessment: {e}")

        return assessments, round_summary

    except Exception as e:
        print(f"    Warning: GAN Verifier failed: {e}, auto-accepting all")
        return _fallback_accept_all(proposed_hypotheses)


def _fallback_accept_all(
    proposed_hypotheses: List[Hypothesis],
) -> tuple:
    """
    Fallback: auto-accept all hypotheses when verifier fails
    回退策略: 验证器失败时自动接受所有假设

    返回: (assessments列表, round_summary字符串)
    """
    assessments = [
        VerifierAssessment(
            hypothesis_id=h.id,
            plausibility_score=0.5,
            novelty_score=0.7,
            verdict="accept",
            revision_suggestions="",
        )
        for h in proposed_hypotheses
    ]
    return assessments, "Verifier failed, optimistic fallback: all hypotheses accepted"


# ============================================================
# Vanilla 假设生成（消融实验用）
# 禁用多专家辩论时，使用单个 LLM 直接生成假设
# ============================================================

VANILLA_GENERATOR_PROMPT = """
You are a generalist scientist. Given a research problem, generate 2-3 novel,
testable hypotheses from different angles (chemical, physical, data-driven).

IMPORTANT: Return a valid JSON object with this exact structure:
{
  "hypotheses": [
    {
      "id": "hyp_v1",
      "statement": "...",
      "scientific_basis": "...",
      "predicted_outcome": "...",
      "confidence": 0.5,
      "generation": 1,
      "status": "active"
    }
  ]
}
"""


async def vanilla_hypothesis_generation(
    problem: ResearchProblem,
    prior_hypotheses: Optional[List[Hypothesis]] = None,
    user_id: str = "experiment_system",
    skill_context: str = "",                           # 历史技能上下文
) -> DebateOutcome:
    """
    Vanilla LLM hypothesis generation — 消融实验用
    用单个 LLM 调用直接生成 2-3 个假设，无多专家辩论。
    返回 DebateOutcome 以保持接口兼容。
    """
    # 构建输入
    prior_text = ""
    if prior_hypotheses:
        prior_text = "\nPreviously proposed hypotheses (avoid repeating):\n" + "\n".join(
            f"- {h.statement}" for h in prior_hypotheses
        )

    input_text = (
        f"Research Problem: {problem.problem_statement}\n"
        f"Domain: {problem.domain}\n"
        f"Target properties: {', '.join(problem.target_properties)}\n"
        f"{prior_text}\n\n"
        "Generate 2-3 novel hypotheses from different scientific perspectives."
    )

    # === 注入技能上下文 ===
    if skill_context:
        input_text = (
            "=== SKILL CONTEXT (Prior Successful Methods) ===\n"
            f"{skill_context}\n"
            "=== END SKILL CONTEXT ===\n\n"
        ) + input_text

    # 创建并运行单个智能体
    agent_kwargs = _create_agent_kwargs(
        name="vanilla_hypothesis_generator",
        instructions=VANILLA_GENERATOR_PROMPT,
    )
    agent = Agent(**agent_kwargs)

    try:
        result = await Runner().run(agent, input_text, max_turns=AGENT_MAX_TURNS)
        output = result.final_output

        # 解析 JSON 输出
        hypotheses = []
        if isinstance(output, str):
            try:
                data = _parse_json_from_llm(output)
                raw_hyps = data.get("hypotheses", []) if isinstance(data, dict) else []
                if not raw_hyps and isinstance(data, list):
                    raw_hyps = data
                for i, raw_h in enumerate(raw_hyps):
                    raw_h = fix_hypothesis(raw_h)
                    if 'id' not in raw_h or not raw_h['id']:
                        raw_h['id'] = f"hyp_v{i+1}"
                    hypotheses.append(Hypothesis(**raw_h))
            except (json.JSONDecodeError, TypeError) as e:
                print(f"    Warning: Vanilla generation parse failed: {e}")

        # 至少产出一个回退假设
        if not hypotheses:
            hypotheses = [Hypothesis(
                id="hyp_v1",
                statement=f"Investigate novel approaches for: {problem.problem_statement[:100]}",
                scientific_basis="Generated by vanilla LLM (parse fallback)",
                predicted_outcome="To be determined through computational experiments",
                confidence=0.5,
                generation=1,
                status="active",
            )]

        return DebateOutcome(
            original_hypotheses=hypotheses,
            refined_hypotheses=hypotheses,
            eliminated_count=0,
            debate_summary="Vanilla LLM generation (no multi-expert debate)",
        )

    except Exception as e:
        print(f"    Error: Vanilla hypothesis generation failed: {e}")
        fallback = [Hypothesis(
            id="hyp_v1",
            statement=f"Explore {problem.domain}: {problem.problem_statement[:80]}",
            scientific_basis="Fallback hypothesis due to generation error",
            predicted_outcome="To be determined",
            confidence=0.3,
            generation=1,
            status="active",
        )]
        return DebateOutcome(
            original_hypotheses=fallback,
            refined_hypotheses=fallback,
            eliminated_count=0,
            debate_summary=f"Vanilla LLM generation failed: {e}",
        )
