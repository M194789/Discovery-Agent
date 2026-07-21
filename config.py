"""
实验系统的共享配置模块。

集中管理环境设置、模型客户端创建和路径解析。
必须在导入任何 CASCADE 工具模块之前导入此模块，以设置工作空间路径。
"""

from __future__ import annotations  # 启用未来版本的注解功能，支持 Python 3.7+ 的新特性

import os  # 导入操作系统接口模块，用于访问环境变量和文件系统操作
import sys  # 导入系统模块，用于操作 Python 运行时环境（如 sys.path）
from pathlib import Path  # 从 pathlib 导入 Path 类，用于跨平台的路径操作

from dotenv import load_dotenv  # 从 python-dotenv 导入 load_dotenv 函数，用于从 .env 文件加载环境变量

# ---------------------------------------------------------------------------
# 路径解析
# ---------------------------------------------------------------------------

EXPERIMENT_DIR = Path(__file__).resolve().parent  # 获取当前配置文件所在的目录（Experiment 目录）
PROJECT_ROOT = EXPERIMENT_DIR.parent  # 获取项目根目录（Experiment 的父目录）

# 从当前目录加载 .env 文件
load_dotenv(EXPERIMENT_DIR / ".env", override=True)  # 加载 .env 文件并覆盖已存在的环境变量

# 将项目根目录添加到 sys.path 以便 CASCADE 导入正常工作
if str(PROJECT_ROOT) not in sys.path:  # 如果项目根目录不在 sys.path 中
    sys.path.insert(0, str(PROJECT_ROOT))  # 将项目根目录插入到 sys.path 的开头

# ---------------------------------------------------------------------------
# 工作空间路径 —— 必须在导入 workspace_tools 之前设置
# ---------------------------------------------------------------------------

_temp_code = str(EXPERIMENT_DIR / "temp_code")  # 构建临时代码目录路径：Experiment/temp_code
_saved_code = str(EXPERIMENT_DIR / "saved_code")  # 构建已保存代码目录路径：Experiment/saved_code

os.makedirs(_temp_code, exist_ok=True)  # 创建临时代码目录（如果已存在则不报错）
os.makedirs(_saved_code, exist_ok=True)  # 创建已保存代码目录（如果已存在则不报错）

if not os.getenv("CODE_STORAGE_DIR"):  # 如果环境变量 CODE_STORAGE_DIR 未设置
    os.environ["CODE_STORAGE_DIR"] = _temp_code  # 设置为临时代码目录路径
if not os.getenv("SAVED_FILES_DIR"):  # 如果环境变量 SAVED_FILES_DIR 未设置
    os.environ["SAVED_FILES_DIR"] = _saved_code  # 设置为已保存代码目录路径

# ---------------------------------------------------------------------------
# 模型配置
# ---------------------------------------------------------------------------

MODEL_NAME = os.getenv("AGENT_MODEL_NAME", "deepseek-reasoner")  # 从环境变量获取模型名称，默认为"o3"
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")  # OpenAI API 基础 URL，默认为官方地址
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "")  # Tavily 搜索 API 密钥，用于文献检索
_require_api_key = os.getenv("REQUIRE_OPENAI_API_KEY", "true").lower() == "true"  # 是否需要 API 密钥，默认为 true

from openai import AsyncOpenAI  # noqa: E402  # 导入异步 OpenAI 客户端类（E402 忽略导入位置警告）

_api_key = os.getenv("OPENAI_API_KEY") if _require_api_key else "EMPTY"  # 如果需要 API 密钥则从环境变量获取，否则设为"EMPTY"
if not _api_key and _require_api_key:  # 如果没有 API 密钥但需要它
    _api_key = "MISSING"  # 设为"MISSING"，将错误延迟到实际 API 调用时

client = AsyncOpenAI(
    api_key=_api_key,  # API 密钥
    base_url=OPENAI_BASE_URL,  # API 基础 URL
    timeout=500.0,  # 超时时间（秒）
    max_retries=3,  # 最大重试次数
)  # 创建异步 OpenAI 客户端实例

# ---------------------------------------------------------------------------
# 常量
# ---------------------------------------------------------------------------

MAX_DISCOVERY_ITERATIONS = 5  # 最大发现迭代次数：5 轮
DEFAULT_USER_ID = "experiment_system"  # 默认用户标识符
DEEP_SOLVER_TIMEOUT = 1800  # DeepSolver 超时时间：1800 秒（30 分钟）
AGENT_MAX_TURNS = 10  # 智能体最大回合数：10（降低以避免长时间等待）
GAN_ADVERSARIAL_ROUNDS = 3  # GAN对抗优化轮数: 默认3轮(生成器→验证器→反馈 × 3)
RISK_FEASIBILITY_THRESHOLD = 0.4  # 风险评估可行性阈值：低于此分数的实验计划将被过滤
RISK_SKIP_LLM_FOR_LOW = True  # 对于低复杂度且所有包可用的计划，跳过LLM评估以节省API调用

# ---------------------------------------------------------------------------
# RL Evolution Configuration - 强化学习假设演化配置
# ---------------------------------------------------------------------------
RL_EVOLUTION_ENABLED = True  # 是否启用RL指导的假设演化（False则回退到纯LLM模式）
RL_ALGORITHM = "thompson_sampling"  # MAB算法: "ucb1" 或 "thompson_sampling"
RL_REWARD_ALPHA = 0.5   # 奖励公式中 support_strength（证据支持）的权重
RL_REWARD_BETA = 0.2    # 奖励公式中 novelty_gain（新颖性增益）的权重
RL_REWARD_GAMMA = 0.3   # 奖励公式中 convergence_progress（收敛进度）的权重
RL_EXPLORATION_CONSTANT = 1.414  # UCB1 探索常数 c（越大越倾向探索）
RL_PERSISTENCE_ENABLED = True  # 是否持久化Bandit状态（跨会话学习）

# ---------------------------------------------------------------------------
# Skill Store Configuration - 技能向量存储配置
# ---------------------------------------------------------------------------
SKILL_STORE_BACKEND = os.getenv("SKILL_STORE_BACKEND", "chromadb")  # 后端: "chromadb" 或 "json"
SKILL_EMBEDDING_MODEL = os.getenv("SKILL_EMBEDDING_MODEL", "text-embedding-3-small")  # 嵌入模型
SKILL_VECTOR_DB_DIR = str(EXPERIMENT_DIR / "saved_code" / "skill_vectordb")  # 向量数据库存储路径
SKILL_CROSS_DOMAIN_ENABLED = os.getenv("SKILL_CROSS_DOMAIN", "true").lower() == "true"  # 跨域迁移开关
SKILL_META_LEARNING_ENABLED = os.getenv("SKILL_META_LEARNING", "true").lower() == "true"  # 元学习开关

# 确保向量数据库目录存在
os.makedirs(SKILL_VECTOR_DB_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# 工具导入辅助函数
# ---------------------------------------------------------------------------
# mcp_servers_and_tools/direct_tools/__init__.py 导入了 research_tools
# 而 research_tools 依赖 supabase, selenium, crawl4ai, neo4j 等重量级库
# 我们使用 importlib 直接导入单个工具模块，避免触发重量级的 __init__.py
# ---------------------------------------------------------------------------

import importlib.util as _ilu  # 导入 importlib.util 模块用于动态导入，简称为_ilu


def _import_module_direct(module_name: str, file_path: str):
    """通过文件路径导入模块，绕过包的 __init__.py。"""
    spec = _ilu.spec_from_file_location(module_name, file_path)  # 从文件位置创建模块规范
    mod = _ilu.module_from_spec(spec)  # 根据规范创建模块对象
    spec.loader.exec_module(mod)  # 执行模块代码以加载模块
    return mod  # 返回加载的模块对象


def load_tavily_search():
    """导入 tavily_search 工具而不触发重量级的 __init__.py 导入。"""
    mod = _import_module_direct(
        "search_tool",  # 模块名称
        str(EXPERIMENT_DIR / "mcp_servers_and_tools" / "direct_tools" / "search_tool.py"),  # search_tool.py 的完整路径
    )
    return mod.tavily_search  # 返回 tavily_search 函数


def load_workspace_tools():
    """直接导入工作空间工具。"""
    mod = _import_module_direct(
        "workspace_tools",  # 模块名称
        str(EXPERIMENT_DIR / "mcp_servers_and_tools" / "direct_tools" / "workspace_tools.py"),  # workspace_tools.py 的完整路径
    )
    return mod  # 返回整个 workspace_tools 模块


def load_memory_tools():
    """导入记忆工具（如果 Mem0/Supabase 未配置可能会失败）。"""
    mod = _import_module_direct(
        "memory_tools",  # 模块名称
        str(EXPERIMENT_DIR / "mcp_servers_and_tools" / "direct_tools" / "memory_tools.py"),  # memory_tools.py 的完整路径
    )
    return mod  # 返回整个 memory_tools 模块


def load_research_tools():
    """导入研究工具（如果 supabase/selenium 未安装可能会失败）。"""
    mod = _import_module_direct(
        "research_tools",  # 模块名称
        str(EXPERIMENT_DIR / "mcp_servers_and_tools" / "direct_tools" / "research_tools.py"),  # research_tools.py 的完整路径
    )
    return mod  # 返回整个 research_tools 模块
