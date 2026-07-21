"""Pydantic models for problem generation and assessment (Module 1)."""  # 模块文档字符串：说明本模块用于问题生成和评估的Pydantic模型（模块1）

from pydantic import BaseModel, Field  # 从pydantic库导入基础模型类和字段类，用于数据验证和序列化
from typing import List  # 从typing模块导入List类型，用于类型注解


class ResearchGap(BaseModel):  # 定义研究空白类，继承自Pydantic的BaseModel，用于表示现有研究中的空白点
    """A gap identified in the existing research landscape."""  # 类文档字符串：说明这是现有研究领域中识别出的空白
    topic: str = Field(description="The specific research gap topic")  # 研究空白的具体主题，字符串类型
    domain: str = Field(description="Scientific domain (e.g., catalysis, battery materials)")  # 科学领域（例如：催化、电池材料）
    supporting_literature: List[str] = Field(  # 支持该研究空白的文献列表
        description="References or URLs from literature search supporting this gap"  # 文献搜索中支持该空白的参考文献或URL
    )
    description: str = Field(  # 研究空白的详细描述
        description="Natural language description of why this is a meaningful gap"  # 用自然语言描述为什么这是一个有意义的研究空白
    )
    confidence: float = Field(  # 空白置信度评分
        default=0.5, description="Confidence that this is a genuine gap (0.0-1.0)"
    )
    source_urls: List[str] = Field(  # 支持空白识别的来源 URL
        default_factory=list, description="URLs from automated search supporting gap identification"
    )


class NoveltyScore(BaseModel):  # 定义新颖性评分类，继承自Pydantic的BaseModel，用于量化评估研究问题或发现的新颖性
    """Quantified novelty assessment for a research problem or discovery."""  # 类文档字符串：说明这是对研究问题或发现的量化新颖性评估
    score: float = Field(  # 新颖性得分，浮点数类型
        description="Novelty score from 0.0 (well-explored) to 1.0 (maximally novel)"  # 新颖性得分范围：0.0（已被充分研究）到1.0（最大新颖性）
    )
    reasoning: str = Field(  # 评分理由，字符串类型
        description="Explanation of why this novelty score was assigned"  # 解释为什么给出这个新颖性评分
    )
    similar_existing_work: List[str] = Field(  # 相似现有工作的列表
        default_factory=list,  # 默认值为空列表，如果没有提供则自动创建空列表
        description="Closest existing works found during novelty assessment"  # 在新颖性评估过程中发现的最接近的现有工作
    )


class ResearchProblem(BaseModel):  # 定义研究问题类，继承自Pydantic的BaseModel，表示一个正式定义的研究问题，可用于假设生成
    """A formally defined research problem ready for hypothesis generation."""  # 类文档字符串：说明这是一个正式定义的研究问题，准备好进行假设生成
    problem_statement: str = Field(  # 问题陈述，字符串类型
        description="Clear, concise statement of the research problem"  # 清晰、简洁的研究问题陈述
    )
    domain: str = Field(description="Scientific domain")  # 科学领域
    research_gaps: List[ResearchGap] = Field(  # 研究空白列表，类型为ResearchGap对象的列表
        description="Research gaps from which this problem was derived"  # 该问题所源自的研究空白
    )
    novelty: NoveltyScore = Field(  # 新颖性评分，类型为NoveltyScore对象
        description="Assessed novelty of this research problem"  # 该研究问题的评估新颖性
    )
    computational_feasibility: str = Field(  # 计算可行性说明，字符串类型
        description="Why this problem can be addressed with computational methods"  # 解释为什么这个问题可以用计算方法解决
    )
    target_properties: List[str] = Field(  # 目标属性列表
        description="Measurable outcomes or properties to investigate"  # 要研究的可测量结果或属性
    )
    keywords: List[str] = Field(  # 关键词列表
        description="Keywords for indexing and memory retrieval"  # 用于索引和记忆检索的关键词
    )
    literature_sources: List[str] = Field(  # 参考文献来源 URL 列表
        default_factory=list, description="URLs of literature consulted during problem generation"
    )
    trend_signals: List[str] = Field(  # 趋势信号描述列表
        default_factory=list, description="Identified research trend directions in the domain"
    )
    causal_insights: List[str] = Field(  # 因果关系洞察列表
        default_factory=list, description="Key causal relationships identified from literature"
    )
