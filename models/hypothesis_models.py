"""用于假设生成和实验规划的 Pydantic 模型（模块 2）。

该模块定义了自主科学发现系统中用于表示假设、辩论结果和实验计划的数据结构。
"""

from pydantic import BaseModel, Field  # 从 pydantic 导入 BaseModel（模型基类）和 Field（字段定义工具）
from typing import List  # 从 typing 模块导入 List 类型提示，用于声明列表类型的字段


class Hypothesis(BaseModel):
    """由领域专家智能体生成的科学假设。"""
    id: str = Field(description="唯一标识符（例如：'H1', 'H2'）")  # 假设的唯一 ID，用于追踪和引用
    statement: str = Field(description="假设的文本陈述")  # 假设的具体内容描述
    scientific_basis: str = Field(
        description="支持该假设的文献或推理依据"  # 假设的科学基础，包括引用的文献或理论推导
    )
    predicted_outcome: str = Field(
        description="如果该假设为真，实验应该展示的结果"  # 预测的实验结果，用于验证假设
    )
    confidence: float = Field(
        description="专家智能体对该假设的置信度（0.0 到 1.0）"  # 置信度分数，0.0 表示完全不确定，1.0 表示完全确定
    )
    generation: int = Field(
        description="生成该假设的迭代轮次（从 1 开始）"  # 记录假设是在第几轮迭代中生成的，第一代假设为 1
    )
    parent_ids: List[str] = Field(
        default_factory=list,  # 默认值为空列表
        description="该假设演化自的父代假设 ID 列表（第一代为空）"  # 记录演化关系，第一代假设没有父代，后续代数记录来源
    )
    status: str = Field(
        default="active",  # 默认状态为"active"（活跃）
        description="假设状态：active（活跃）、confirmed（已确认）、refuted（已反驳）、merged（已合并）或 eliminated（已淘汰）"  # 假设的当前生命周期状态
    )


class DebateOutcome(BaseModel):
    """多智能体辩论后产生的结果，记录了竞争性假设的对抗性优化过程。"""
    original_hypotheses: List[Hypothesis] = Field(
        description="专家智能体提交的初始假设集合（辩论前）"  # 辩论开始前所有专家提出的原始假设
    )
    refined_hypotheses: List[Hypothesis] = Field(
        description="经过对抗性辩论和优化后的假设集合"  # 辩论结束后保留并优化的假设集
    )
    debate_summary: str = Field(
        description="辩论过程中的关键论点、反驳论点和推理总结"  # 辩论的详细摘要，包括各方观点和最终推理
    )
    eliminated_count: int = Field(
        description="在辩论过程中被淘汰的假设数量"  # 统计有多少假设因质量不佳而被剔除
    )


class VerifierAssessment(BaseModel):
    """GAN 验证器智能体对单个颠覆性假设的评估结果。"""
    hypothesis_id: str = Field(description="被评估的假设ID")
    plausibility_score: float = Field(
        description="科学合理性评分（0.0 到 1.0），基于文献证据的合理性判定"
    )
    novelty_score: float = Field(
        description="新颖性评分（0.0 到 1.0），衡量假设对既有范式的挑战程度"
    )
    literature_conflicts: List[str] = Field(
        default_factory=list,
        description="与现有文献的冲突列表"
    )
    literature_support: List[str] = Field(
        default_factory=list,
        description="支持性文献证据列表"
    )
    verdict: str = Field(
        description="判定结果：'accept'（接受）、'revise'（需修改）或 'reject'（拒绝）"
    )
    revision_suggestions: str = Field(
        default="",
        description="修改建议（当 verdict 为 'revise' 时提供具体改进方向）"
    )


class GANRoundResult(BaseModel):
    """GAN 对抗循环中单轮的结果记录。"""
    round_number: int = Field(description="轮次编号（从 1 开始）")
    proposed_hypotheses: List[Hypothesis] = Field(
        description="生成器智能体本轮提出的颠覆性假设列表"
    )
    assessments: List[VerifierAssessment] = Field(
        default_factory=list,
        description="验证器智能体对每个假设的评估列表"
    )
    accepted_hypotheses: List[Hypothesis] = Field(
        default_factory=list,
        description="通过验证的假设列表（verdict 为 'accept'）"
    )
    revision_feedback: str = Field(
        default="",
        description="发送给下一轮生成器的反馈摘要（包含 'revise' 和 'reject' 的原因）"
    )


class GANOutcome(BaseModel):
    """GAN 风格双智能体对抗优化的完整输出结果。"""
    input_hypotheses: List[Hypothesis] = Field(
        description="辩论精炼后作为输入的常规假设列表"
    )
    rounds: List[GANRoundResult] = Field(
        default_factory=list,
        description="各轮对抗的详细记录"
    )
    final_hypotheses: List[Hypothesis] = Field(
        description="最终假设列表（辩论假设 + 通过 GAN 验证的颠覆性假设）"
    )
    total_proposed: int = Field(
        default=0,
        description="生成器在所有轮次中总共提出的假设数量"
    )
    total_accepted: int = Field(
        default=0,
        description="通过验证器验证的假设总数"
    )
    gan_summary: str = Field(
        default="",
        description="GAN 对抗过程的叙述性总结"
    )


class ExperimentPlan(BaseModel):
    """用于测试假设的计算实验详细计划。"""
    hypothesis_id: str = Field(
        description="该实验所要测试的假设 ID"  # 关联到具体的假设，指明实验的目的
    )
    method: str = Field(
        description="计算方法（例如：'DFT'（密度泛函理论）、'MD'（分子动力学）、'ML_potential'（机器学习势函数）、'data_analysis'（数据分析））"  # 指定使用的计算方法类型
    )
    tool_chain: List[str] = Field(
        description="所需的工具和包列表（例如：['pymatgen', 'ase', 'chgnet']）"  # 实现实验所需的软件栈
    )
    steps: List[str] = Field(
        description="按顺序排列的实验步骤（自然语言描述）"  # 实验操作的有序列表，用自然语言描述每个步骤
    )
    code_specification: str = Field(
        description="DeepSolver 的详细规范说明，描述代码应完成的功能"  # 给代码生成器的详细指令，说明如何实现实验
    )
    success_criteria: str = Field(
        description="判断实验是否成功的定量标准"  # 明确的数值或条件标准，用于评估实验结果
    )
    estimated_complexity: str = Field(
        default="medium",  # 默认复杂度为"medium"（中等）
        description="预估的复杂度等级：'low'（低）、'medium'（中）或 'high'（高）"  # 实验实现的难度评估
    )
