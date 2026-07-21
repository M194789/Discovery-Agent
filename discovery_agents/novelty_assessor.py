"""
新颖性评估智能体（跨领域模块）

通过对比现有文献，量化评估科学发现或研究方向的新颖性程度。
"""

from __future__ import annotations  # 启用未来版本注解，支持 Python 3.7+ 的新特性

import _utf8_env  # noqa: F401  # 全局 UTF-8 编码修复

from agents import Agent, Runner  # 从 openai-agents SDK 导入智能体和运行器
from agents import OpenAIChatCompletionsModel  # 导入 OpenAI 聊天完成模型包装器

from config import client, MODEL_NAME, AGENT_MAX_TURNS  # 导入配置：API 客户端、模型名称、最大回合数
from models.problem_models import NoveltyScore  # 导入新颖性评分模型

_tools = None  # 全局变量：缓存已加载的工具列表，避免重复初始化


def _get_tools():
    """获取并返回工具列表，首次调用时加载工具"""
    global _tools  # 声明使用全局变量
    if _tools is None:  # 如果工具列表尚未加载
        _tools = []  # 初始化为空列表
        # 如果使用 DeepSeek 模型，跳过工具加载以避免兼容性问题
        if "deepseek" in MODEL_NAME.lower():
            return _tools
        from config import load_tavily_search  # 导入 Tavily 搜索工具
        _tools = [load_tavily_search()]  # 添加 Tavily 搜索工具
        try:
            from config import load_memory_tools  # 尝试导入记忆工具
            mem = load_memory_tools()
            _tools.append(mem.search_memory)  # 添加记忆搜索工具
        except Exception:
            pass  # 记忆工具加载失败则忽略
    return _tools  # 返回已加载的工具列表


def _is_deepseek() -> bool:
    """检查是否正在使用 DeepSeek 模型"""
    return "deepseek" in MODEL_NAME.lower()  # 检查模型名称是否包含 "deepseek"


# 新颖性评估智能体的系统提示词
NOVELTY_ASSESSOR_PROMPT = """
ROLE: Scientific Novelty Assessor

You evaluate the novelty of a scientific discovery or research direction.

TASK:
Given a description of the current state of a discovery (problem statement,
hypotheses, key findings), produce a novelty score.

SCORING GUIDE:
- 0.0-0.3: Incremental / well-explored territory (many similar papers exist)
- 0.3-0.5: Low-moderate novelty (topic is known but approach is slightly different)
- 0.5-0.7: Moderate novelty with new aspects (some differentiation from literature)
- 0.7-0.85: Significant novelty, clear differentiation (few similar works)
- 0.85-1.0: Highly novel, little prior work (groundbreaking approach)

IMPORTANT SCORING PRINCIPLES:
- If you performed literature search and found FEW similar works, score HIGHER (0.6-0.8)
- If you could NOT search literature (no tools available), base score on problem uniqueness
- If the problem involves NOVEL combinations of methods/domains, score at least 0.6
- If experiments successfully validated hypotheses, add 0.1 to base score
- AVOID defaulting to 0.5 - make a clear assessment

CONSTRAINTS:
- Be honest and calibrated
- Maximum 10 tool calls
"""


def create_novelty_assessor_agent() -> Agent:
    """创建并返回新颖性评估智能体"""
    instructions = NOVELTY_ASSESSOR_PROMPT  # 使用基础提示词
    # 如果使用 DeepSeek 模型，添加 JSON 格式指令和示例
    if _is_deepseek():
        instructions += '''

OUTPUT FORMAT: Return a valid JSON object EXACTLY like this example:
```json
{
  "score": 0.72,
  "reasoning": "The approach combines deep learning with signal processing in a novel way. While similar work exists in [cite works], this specific combination for [application] is relatively unexplored.",
  "similar_existing_work": ["Paper A (2023) - similar but different domain", "Paper B (2022) - related method"]
}
```

The score MUST be a decimal number between 0.0 and 1.0.
Do NOT return score as 0.5 unless you have specific reasoning for moderate novelty.
'''
    
    kwargs = {
        "name": "NoveltyAssessorAgent",  # 智能体名称
        "instructions": instructions,  # 系统指令
        "model": OpenAIChatCompletionsModel(  # 使用 OpenAI 兼容模型
            model=MODEL_NAME, openai_client=client  # 指定模型和客户端
        ),
    }
    # 如果不是 DeepSeek 模型，添加结构化输出类型和工具
    if not _is_deepseek():
        kwargs["output_type"] = NoveltyScore  # 设置输出类型为 NoveltyScore
        kwargs["tools"] = _get_tools()  # 添加工具列表
    
    return Agent(**kwargs)  # 创建并返回智能体实例


async def assess_novelty(
    description: str,
    user_id: str = "experiment_system",
) -> NoveltyScore:
    """评估科学发现描述的新颖性

    参数:
        description: 待评估的发现状态描述文本
        user_id: 用于记忆操作的用户标识符

    返回:
        包含评分、推理和相似工作的 NoveltyScore 对象
    """
    agent = create_novelty_assessor_agent()  # 创建新颖性评估智能体
    input_text = f"User ID: {user_id}\n\nDiscovery to assess:\n{description}"  # 构建输入文本
    result = await Runner().run(agent, input_text, max_turns=AGENT_MAX_TURNS)  # 运行智能体
    
    # 解析 DeepSeek JSON 输出
    output = result.final_output  # 获取最终输出
    if _is_deepseek() and isinstance(output, str):  # 如果是 DeepSeek 且输出为字符串
        import json  # 导入 JSON 模块
        import re  # 导入正则表达式模块
        try:
            cleaned = re.sub(
                r'^```(?:json)?\s*|\s*```$', '', output.strip()  # 移除 Markdown 代码块标记
            )
            
            # 多策略 JSON 提取
            data = None
            
            # 策略 1: 直接解析
            try:
                data = json.loads(cleaned)
            except json.JSONDecodeError:
                pass
            
            # 策略 2: 查找完整的 JSON 对象
            if data is None:
                brace_count = 0
                start_pos = -1
                end_pos = -1
                in_string = False
                
                for i, char in enumerate(cleaned):
                    if char == '"' and (i == 0 or cleaned[i-1] != '\\'):
                        in_string = not in_string
                    elif not in_string:
                        if char == '{':
                            if start_pos == -1:
                                start_pos = i
                            brace_count += 1
                        elif char == '}':
                            brace_count -= 1
                            if brace_count == 0 and start_pos != -1:
                                end_pos = i + 1
                                break
                
                if start_pos != -1 and end_pos != -1:
                    try:
                        data = json.loads(cleaned[start_pos:end_pos])
                    except json.JSONDecodeError:
                        pass
            
            # 策略 3: 正则表达式提取
            if data is None:
                json_match = re.search(r'\{[\s\S]*?"score"[\s\S]*?\}', cleaned)
                if json_match:
                    try:
                        data = json.loads(json_match.group(0))
                    except json.JSONDecodeError:
                        pass
            
            # 策略 4: 直接提取 score 数值（即使 JSON 格式不完整）
            if data is None:
                score_match = re.search(
                    r'["\']?score["\']?\s*[:\s]\s*([\d.]+)',
                    cleaned,
                    re.IGNORECASE
                )
                if score_match:
                    try:
                        extracted_score = float(score_match.group(1))
                        if 0.0 <= extracted_score <= 1.0:
                            data = {
                                "score": extracted_score,
                                "reasoning": "Score extracted from partial output",
                                "similar_work": []
                            }
                            print(f"    Extracted novelty score: {extracted_score}")
                    except ValueError:
                        pass
            
            # 如果所有策略都失败，基于内容启发式评分（而不是默认 0.5）
            if data is None:
                print(f"    Warning: Could not parse novelty JSON. Raw: {output[:150]}...")
                # 检查输出中是否有暗示高/低新颖性的关键词
                lower_out = output.lower()
                if any(w in lower_out for w in ['highly novel', 'groundbreaking', 'unique', 'first']):
                    inferred_score = 0.75
                elif any(w in lower_out for w in ['significant', 'novel approach', 'new method']):
                    inferred_score = 0.65
                elif any(w in lower_out for w in ['incremental', 'well-known', 'common', 'established']):
                    inferred_score = 0.35
                else:
                    inferred_score = 0.55  # 略高于中等，避免总是 0.5
                data = {
                    "score": inferred_score,
                    "reasoning": "Score inferred from output keywords",
                    "similar_work": []
                }
            
            # 确保必要字段存在
            if "score" not in data:
                data["score"] = 0.5
            if "reasoning" not in data:
                data["reasoning"] = "Assessment completed"
            if "similar_work" not in data:
                data["similar_work"] = []
            
            # 确保 score 是 float
            if isinstance(data["score"], str):
                try:
                    data["score"] = float(data["score"])
                except ValueError:
                    data["score"] = 0.5
            
            output = NoveltyScore(**data)  # 创建 NoveltyScore 对象
        except Exception as e:  # 解析失败
            print(f"    Warning: Novelty parsing error: {e}, using default")
            output = NoveltyScore(
                score=0.5,
                reasoning="Unable to parse novelty assessment",
                similar_work=[]
            )
    
    return output  # 返回新颖性评分结果
