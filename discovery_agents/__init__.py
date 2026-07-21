"""实验系统的智能体工厂模块。

该模块是 discovery_agents 包的入口点，负责导出所有核心智能体功能函数，
供系统其他部分（如 orchestrator）统一调用。
"""

# 从 openai-agents SDK 导入核心类
from agents import Agent, Runner, OpenAIChatCompletionsModel

from .problem_generator import generate_problem
from .novelty_assessor import assess_novelty
from .hypothesis_generator import generate_and_debate_hypotheses, run_gan_adversarial_refinement
from .experiment_planner import plan_experiments
from .experiment_executor import run_experiments
from .result_analyzer import analyze_results, evolve_hypotheses
from .risk_assessor import assess_experiment_risks

__all__ = [
    # openai-agents SDK 类
    "Agent",
    "Runner",
    "OpenAIChatCompletionsModel",
    # 本模块功能函数
    "generate_problem",
    "assess_novelty",
    "generate_and_debate_hypotheses",
    "run_gan_adversarial_refinement",
    "plan_experiments",
    "run_experiments",
    "analyze_results",
    "evolve_hypotheses",
    "assess_experiment_risks",
]
