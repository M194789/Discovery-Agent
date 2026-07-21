"""
实验系统的记忆操作模块（模块 5）。

封装了 CASCADE 的 Mem0 记忆工具，适配科学发现场景。
如果记忆服务不可用，所有操作都会优雅降级。
"""

from __future__ import annotations  # 启用未来版本的注解功能，支持 Python 3.7+ 的新特性

import json  # 导入 JSON 处理模块，用于序列化和反序列化数据
from typing import Optional  # 从 typing 模块导入 Optional 类型提示，用于声明可选类型

# 延迟导入标志 —— 记忆工具会触发 Mem0 初始化，需要数据库连接
# 我们在使用时才导入，避免阻塞启动过程
_memory_available: Optional[bool] = None  # 全局标志，记录记忆服务是否可用，None 表示尚未检查
_search_memory = None  # 全局变量，缓存搜索记忆的函数引用
_save_to_memory = None  # 全局变量，缓存保存记忆的函数引用


def _ensure_memory_tools():
    """延迟导入记忆工具，首次调用时才初始化。"""
    global _memory_available, _search_memory, _save_to_memory  # 声明使用全局变量
    if _memory_available is not None:  # 如果已经检查过记忆服务可用性
        return _memory_available  # 直接返回已知的结果，避免重复初始化
    try:
        from config import load_memory_tools  # 从配置文件导入记忆工具加载函数
        mem = load_memory_tools()  # 加载记忆工具模块
        _search_memory = mem.search_memory  # 获取搜索记忆函数的引用
        _save_to_memory = mem.save_memory  # 获取保存记忆函数的引用（注意：函数名是 save_memory）
        _memory_available = True  # 标记记忆服务可用
        print("  记忆服务已连接")  # 打印成功消息
    except Exception as e:  # 捕获所有异常
        _memory_available = False  # 标记记忆服务不可用
        print(f"  记忆服务不可用 ({e}) —— 将在无持久化记忆模式下运行")  # 打印警告信息
    return _memory_available  # 返回记忆服务可用性标志


async def search_prior_discoveries(query: str, user_id: str) -> str:
    """搜索记忆中类似的过往实验和发现。

    参数:
        query: 搜索查询字符串
        user_id: 用户标识符

    返回:
        格式化的相关记忆字符串，如果记忆不可用则返回空结果消息。
    """
    if not _ensure_memory_tools():  # 检查记忆服务是否可用
        return "未找到相关记忆（记忆服务不可用）。"  # 如果不可用，返回提示信息
    try:
        # 直接调用函数，不使用 on_invoke_tool
        result = _search_memory(query=query, user_id=user_id)
        return result  # 返回搜索结果
    except Exception as e:  # 捕获所有异常
        print(f"  警告：记忆搜索失败：{e}")  # 打印警告信息
        return "未找到相关记忆。"  # 返回默认消息


async def save_discovery_to_memory(
    iteration_summary: str,  # 本轮迭代的总结文本
    user_id: str,  # 用户标识符
) -> str:
    """将本轮迭代的关键发现保存到记忆中。

    参数:
        iteration_summary: 迭代总结内容
        user_id: 用户标识符

    返回:
        保存结果消息。
    """
    if not _ensure_memory_tools():  # 检查记忆服务是否可用
        return "跳过记忆保存（记忆服务不可用）。"  # 如果不可用，返回跳过消息
    try:
        # 直接调用函数，不使用 on_invoke_tool
        result = _save_to_memory(content=iteration_summary, user_id=user_id)
        return "记忆保存成功" if result else "记忆保存失败"
    except Exception as e:  # 捕获所有异常
        print(f"  警告：记忆保存失败：{e}")  # 打印警告信息
        return f"记忆保存失败：{e}"  # 返回错误消息


async def save_experiment_skill(
    plan_summary: str,  # 实验计划的总结文本
    result_summary: str,  # 实验结果的总结文本
    user_id: str,  # 用户标识符
) -> str:
    """将成功的实验工作流作为可复用技能保存到记忆中。

    参数:
        plan_summary: 实验计划总结
        result_summary: 实验结果总结
        user_id: 用户标识符

    返回:
        保存结果消息。
    """
    # 同时保存到旧记忆系统（保持兼容）
    content = (
        f"可复用的实验技能\n\n"
        f"实验计划:\n{plan_summary}\n\n"
        f"执行结果:\n{result_summary}"
    )
    return await save_discovery_to_memory(content, user_id)


async def load_relevant_skills(query: str, user_id: str) -> str:
    """搜索与问题相关的已保存技能。

    优先使用 skill_registry 的向量语义搜索，
    不可用时回退到旧的记忆关键词搜索。

    参数:
        query: 搜索查询字符串
        user_id: 用户标识符

    返回:
        相关技能的搜索结果。
    """
    try:
        import skill_registry
        result = await skill_registry.find_applicable_skills(query, user_id)
        if result and "未找到" not in result:
            return result
    except Exception:
        pass

    # 回退到旧的记忆搜索
    skill_query = f"可复用的实验技能工作流：{query}"
    return await search_prior_discoveries(skill_query, user_id)
