"""用于整体发现状态追踪的 Pydantic 模型（模块 5）。

该模块定义了自主科学发现过程中用于记录和追踪整个发现会话状态的数据结构，
包括每轮迭代的快照和完整会话的历史记录。
"""

from pydantic import BaseModel, Field  # 从 pydantic 导入 BaseModel（模型基类）和 Field（字段定义工具）
from typing import Dict, List, Optional  # 从 typing 模块导入类型提示：Dict, List（列表）和 Optional（可选类型）
from datetime import datetime  # 从 datetime 模块导入日期时间类，用于生成时间戳

from .problem_models import ResearchProblem  # 从当前包的 problem_models 模块导入研究问题类
from .hypothesis_models import Hypothesis, ExperimentPlan, GANOutcome  # 从 hypothesis_models 模块导入假设类、实验计划类和GAN结果类
from .experiment_models import ExperimentResult  # 从 experiment_models 模块导入实验结果类
from .analysis_models import AnalysisResult, HypothesisEvolution, EnrichedAnalysisResult  # 从 analysis_models 模块导入分析结果类和假设演化类
from .risk_models import RiskAssessment  # 从 risk_models 模块导入风险评估类


class DiscoveryIteration(BaseModel):
    """发现循环单轮迭代的状态快照。"""
    iteration: int = Field(description="迭代编号（从 1 开始）")  # 记录这是第几轮迭代
    problem: ResearchProblem = Field(description="正在研究的研究问题对象")  # 本轮迭代要解决的研究问题
    hypotheses: List[Hypothesis] = Field(
        description="本轮迭代中活跃的所有假设"  # 当前迭代中所有活跃的假设集合
    )
    plans: List[ExperimentPlan] = Field(
        default_factory=list,  # 默认值为空列表
        description="本轮迭代创建的实验计划列表"  # 记录本轮生成的所有实验计划
    )
    results: List[ExperimentResult] = Field(
        default_factory=list,  # 默认值为空列表
        description="本轮迭代的实验结果列表"  # 记录本轮执行实验后获得的所有结果
    )
    analysis: List[AnalysisResult] = Field(
        default_factory=list,  # 默认值为空列表
        description="本轮迭代的分析结果列表"  # 记录对实验结果进行分析后得到的所有分析结论
    )
    evolution: Optional[HypothesisEvolution] = Field(
        default=None,  # 默认值为 None，表示尚未进行演化
        description="本轮迭代结束时假设如何演化的信息（可能为 None）"  # 记录假设演化的详细情况，包括新增、淘汰、更新的假设
    )
    gan_outcome: Optional[GANOutcome] = Field(
        default=None,  # 默认值为 None，表示未运行或跳过了 GAN 阶段
        description="GAN 对抗优化阶段的结果（可能为 None）"  # 记录生成器-验证器对抗产生的颠覆性假设及评估
    )
    risk_assessments: List[RiskAssessment] = Field(
        default_factory=list,  # 默认值为空列表
        description="Phase B.5 风险评估结果，记录每个实验计划的可行性评估和过滤原因"
    )
    rl_action_plan: Optional[Dict] = Field(
        default=None,
        description="RL优化器为本轮迭代推荐的动作计划（假设ID→动作映射）"
    )
    timestamp: str = Field(
        default_factory=lambda: datetime.now().isoformat(),  # 使用 lambda 函数生成当前的 ISO 格式时间戳
        description="本轮迭代开始时的 ISO 格式时间戳"  # 记录迭代启动的具体时间，便于追踪和回溯
    )


class DiscoveryState(BaseModel):
    """自主发现会话的完整状态。"""
    session_id: str = Field(description="唯一运行标识符")  # 本次发现会话的唯一 ID，用于区分不同的运行实例
    domain: str = Field(description="研究领域")  # 科学研究领域，如"电池材料"、"催化"等
    initial_problem: Optional[str] = Field(
        default=None,  # 默认值为 None，表示用户未提供种子问题
        description="用户提供的种子问题（如果由智能体生成则为 None）"  # 记录用户是否指定了初始问题，None 表示完全由智能体生成
    )
    problem: Optional[ResearchProblem] = Field(
        default=None,  # 默认值为 None，在问题生成阶段后会被赋值
        description="形式化的研究问题对象"  # 经过智能体形式化定义的完整研究问题对象
    )
    iterations: List[DiscoveryIteration] = Field(
        default_factory=list,  # 默认值为空列表，随着迭代进行逐步添加
        description="所有已完成迭代的历史记录列表"  # 按顺序存储每一轮迭代的完整状态快照
    )
    current_iteration: int = Field(
        default=0,  # 默认值为 0，表示尚未开始任何迭代
        description="当前正在进行的迭代编号"  # 记录当前进行到第几轮迭代，完成后会添加到 iterations 列表
    )
    max_iterations: int = Field(description="配置的最大迭代次数")  # 允许执行的最大迭代轮次，达到此值后停止发现流程
    converged: bool = Field(
        default=False,  # 默认值为 False，表示尚未收敛
        description="发现过程是否已经收敛"  # True 表示已达到稳定状态或找到满意解，False 表示仍在探索中
    )
    overall_novelty_score: float = Field(
        default=0.0,  # 默认值为 0.0，随着发现过程推进会更新
        description="发现过程的累积新颖性度量"  # 综合评估整个发现会话产生成果的新颖程度，分数越高表示越创新
    )
    model_name: str = Field(description="本次会话使用的大语言模型名称")  # 记录使用的 LLM 模型，如"o3"、"deepseek"等，用于元数据追踪
