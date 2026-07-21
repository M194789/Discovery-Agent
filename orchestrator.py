"""
Discovery Orchestrator (Module 5)
发现编排器（模块5）

Central closed-loop coordinator driving autonomous scientific discovery
中央闭环协调器，驱动自主科学发现流程

这个文件是整个系统的"总指挥"，负责协调各个智能体模块，
按照"提出问题→生成假设→做实验→分析结果→演化假设"的流程运行。
"""

from __future__ import annotations  # 启用未来版本的类型注解语法，让代码更简洁

import _utf8_env  # noqa: F401  # 全局 UTF-8 编码修复（必须在所有其他 import 之前）

import json      # JSON处理库，用于保存发现结果到文件
import uuid      # 唯一标识符生成库，用于给每次发现任务生成唯一ID
import time      # 时间库，用于计时统计每轮花了多久
from datetime import datetime  # 日期时间库（这里实际未使用，可能是预留的）
from typing import List, Optional, Callable, Awaitable  # 类型提示工具
import sys       # 系统库，用于修改Python的模块搜索路径
from pathlib import Path  # 路径处理库，用于跨平台的文件路径操作

# === 设置模块搜索路径 ===
# 获取当前文件所在目录的绝对路径
current_dir = Path(__file__).resolve().parent
# 如果这个目录还不在Python的搜索路径里，就把它加进去
# 这样才能正确导入同目录下的其他模块
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))

# === 导入项目内部模块 ===
# 从配置文件导入：最大迭代次数、默认用户ID
from config import MAX_DISCOVERY_ITERATIONS, DEFAULT_USER_ID

# 从数据模型导入各种"数据模板"
from models.problem_models import ResearchProblem      # 研究问题的数据结构
from models.hypothesis_models import Hypothesis, ExperimentPlan  # 假设、实验计划
from models.experiment_models import ExperimentResult  # 实验结果
from models.analysis_models import AnalysisResult, HypothesisEvolution  # 分析结果、假设演化
from models.discovery_state import DiscoveryIteration, DiscoveryState  # 迭代记录、总状态

# 导入记忆操作和技能注册模块
import memory_ops, skill_registry  # 用于保存/读取发现历史和可复用技能

# === RL 假设演化优化器导入 ===
from config import (
    RL_EVOLUTION_ENABLED, RL_ALGORITHM, RL_REWARD_ALPHA,
    RL_REWARD_BETA, RL_REWARD_GAMMA, RL_EXPLORATION_CONSTANT,
    RL_PERSISTENCE_ENABLED,
)


class DiscoveryOrchestrator:
    """
    Coordinate end-to-end autonomous scientific discovery loop
    协调端到端的自主科学发现循环
    
    这是整个系统的核心类，就像一个"项目经理"，
    负责安排各个"员工"（智能体模块）的工作顺序。
    """

    def __init__(
        self,
        domain: str,  # 研究领域，比如"催化剂"、"电池材料"、"电子信息"
        initial_problem: Optional[str] = None,  # 用户提供的初始问题（可选，不提供则AI自动生成）
        max_iterations: int = MAX_DISCOVERY_ITERATIONS,  # 最多跑几轮，默认5轮
        model_name: str = "o3",  # 使用哪个AI模型，如"o3"、"deepseek-chat"
        user_id: str = DEFAULT_USER_ID,  # 用户ID，用于隔离不同用户的记忆
        interactive: bool = False,  # 是否开启交互模式（每一步都等人确认）
        verbose: bool = False,  # 是否开启详细日志模式（打印更多信息）
        log_callback: Optional[Callable[[str], None]] = None,  # 日志回调函数，用于把日志发送到其他地方（如网页）
        checkpoint_callback: Optional[Callable[[str], Awaitable[bool]]] = None,  # 检查点回调，用于异步等待用户确认
        # === 消融实验模块开关 ===
        use_debate: bool = True,             # 多专家辩论
        use_gan: bool = True,                # GAN 对抗优化
        use_risk_assessment: bool = True,    # 风险评估 (Phase B.5)
        use_risk_static_only: bool = False,  # 仅静态风险检查（B2 变体）
        use_statistical_testing: bool = True, # 统计检验
        use_visualization: bool = True,       # 可视化
        use_causal_inference: bool = True,    # 因果推断
        use_rl_evolution: bool = True,        # RL 指导的假设演化
        use_skill_solidify: bool = True,      # 技能固化与复用
        use_cross_domain: bool = True,        # 跨域技能迁移
        use_literature: bool = True,          # 文献挖掘
        use_memory: bool = True,              # 记忆系统
    ):
        """
        初始化编排器，设置各种参数
        """
        # 把参数都存到对象属性里，后面要用
        self.domain = domain                      # 研究领域
        self.initial_problem = initial_problem    # 初始问题
        self.max_iterations = max_iterations      # 最大轮数
        self.model_name = model_name              # 模型名称
        self.user_id = user_id                    # 用户ID
        self.interactive = interactive            # 交互模式开关
        self.verbose = verbose                    # 详细日志开关
        self.log_callback = log_callback          # 日志回调
        self.checkpoint_callback = checkpoint_callback  # 检查点回调

        # === 消融实验模块开关 ===
        self.use_debate = use_debate
        self.use_gan = use_gan
        self.use_risk_assessment = use_risk_assessment
        self.use_risk_static_only = use_risk_static_only
        self.use_statistical_testing = use_statistical_testing
        self.use_visualization = use_visualization
        self.use_causal_inference = use_causal_inference
        self.use_rl_evolution = use_rl_evolution
        self.use_skill_solidify = use_skill_solidify
        self.use_cross_domain = use_cross_domain
        self.use_literature = use_literature
        self.use_memory = use_memory

        # 创建"状态记录本"，记录这次发现任务的所有信息
        self.state = DiscoveryState(
            session_id=str(uuid.uuid4())[:8],  # 生成8位随机ID作为"实验编号"，如"abc123de"
            domain=domain,                      # 研究领域
            initial_problem=initial_problem,    # 初始问题
            max_iterations=max_iterations,      # 最大轮数
            model_name=model_name,              # 模型名称
        )

        # === 初始化 RL 假设演化优化器 ===
        self.rl_optimizer = None
        if self.use_rl_evolution and RL_EVOLUTION_ENABLED:
            try:
                from rl_evolution import RLEvolutionOptimizer
                from rl_evolution.persistence import load_state
                from rl_evolution.bandit import BanditAlgorithm

                self.rl_optimizer = RLEvolutionOptimizer(
                    algorithm=RL_ALGORITHM,
                    alpha=RL_REWARD_ALPHA,
                    beta=RL_REWARD_BETA,
                    gamma=RL_REWARD_GAMMA,
                    exploration_constant=RL_EXPLORATION_CONSTANT,
                    session_id=self.state.session_id,
                )
                # 尝试加载先前会话的学习状态
                if RL_PERSISTENCE_ENABLED:
                    prior_state = load_state(self.state.session_id)
                    if prior_state:
                        self.rl_optimizer.bandit = BanditAlgorithm.from_state(
                            prior_state
                        )
            except Exception as e:
                print(f"  Warning: RL optimizer init failed: {e}")
                self.rl_optimizer = None

    async def _checkpoint(self, message: str) -> bool:
        """
        检查点函数：在关键节点等待用户确认是否继续
        
        参数:
            message: 要显示给用户的消息
        返回:
            True表示继续，False表示中止
        """
        # 如果没开交互模式，直接返回True（自动继续）
        if not self.interactive:
            return True
        
        # 如果有自定义的回调函数（比如网页上的确认按钮），就用它
        if self.checkpoint_callback:
            return await self.checkpoint_callback(message)
        
        # 否则在命令行里等用户输入
        print(f"\n{'='*60}")
        print(f"CHECKPOINT: {message}")  # 打印检查点消息
        print(f"{'='*60}")
        response = input("Proceed? (yes/no): ").strip().lower()  # 等待用户输入
        return response in ("y", "yes", "")  # 输入y/yes/回车都算同意

    def _log(self, msg: str):
        """
        普通日志函数：打印消息，并可选地发送到回调
        每次都会打印，不管verbose是否开启
        """
        try:
            print(msg)  # 打印到控制台
        except UnicodeEncodeError:
            print(msg.encode("utf-8", errors="replace").decode("utf-8", errors="replace"))
        if self.log_callback:
            self.log_callback(msg)  # 如果有回调，也发送过去（比如显示到网页）

    def _vlog(self, msg: str):
        """
        详细日志函数：只有在verbose模式下才打印
        用于输出一些"啰嗦"的详细信息
        """
        if self.verbose:  # 只有开启了详细模式才打印
            print(msg)
            if self.log_callback:
                self.log_callback(msg)

    async def run(self) -> DiscoveryState:
        """
        Execute complete discovery loop
        执行完整的发现循环 —— 这是整个编排器的核心方法！
        
        返回:
            DiscoveryState: 包含整个发现过程的完整状态
        """
        # ========== 开始计时 ==========
        total_start = time.time()  # 记录开始时间
        
        # 打印欢迎信息/任务摘要
        self._log(f"\n{'#'*60}")
        self._log(f"# Autonomous Scientific Discovery - Session {self.state.session_id}")
        self._log(f"# Domain: {self.domain}")  # 研究领域
        self._log(f"# Max Iterations: {self.max_iterations}")  # 最大轮数
        self._log(f"# Model: {self.model_name}")  # 使用的模型
        self._log(f"{'#'*60}\n")

        # ========== 初始化阶段 [INIT] ==========
        
        # --- 步骤1: 搜索历史记忆 ---
        # 去"记忆库"里翻翻，看以前有没有做过类似研究
        self._log("[INIT] Searching prior discoveries in memory...")
        if self.use_memory:
            prior_knowledge = await memory_ops.search_prior_discoveries(
                f"{self.domain} scientific discovery",  # 搜索关键词
                self.user_id  # 用户ID（不同用户的记忆是隔离的）
            )
            self._vlog(f"  Prior knowledge: {prior_knowledge[:200]}...")  # 详细模式下打印前200字符
        else:
            prior_knowledge = ""
            self._log("  Memory disabled (ablation mode)")

        # --- 步骤2: 加载相关技能（语义搜索 + 跨域迁移）---
        # 看看有没有以前成功的实验方法可以复用
        self._log("[INIT] Loading relevant skills...")
        if self.use_skill_solidify:
            skills = await skill_registry.find_applicable_skills(
                self.domain,  # 搜索查询
                self.user_id,  # 用户ID
                domain=self.domain,  # 域过滤
                include_cross_domain=self.use_cross_domain,  # 跨域搜索开关
            )
            self._vlog(f"  Skills found: {skills[:200]}...")
        else:
            skills = ""
            self._log("  Skill loading disabled (ablation mode)")

        # --- 步骤3: 生成研究问题 ---
        # 让AI根据领域（和用户提供的初始问题）生成一个正式的研究问题
        self._log("[INIT] Generating research problem...")
        from discovery_agents.problem_generator import generate_problem  # 延迟导入，避免循环依赖

        try:
            # 调用问题生成器
            problem = await generate_problem(
                self.domain,           # 研究领域
                self.initial_problem,  # 用户的初始问题（可能为None）
                self.user_id,          # 用户ID
                skip_literature=(not self.use_literature),  # 消融开关
                skill_context=skills,  # 历史技能上下文
            )
            self.state.problem = problem  # 把问题存到状态里
            self._log(f"  Problem: {problem.problem_statement}")  # 打印问题陈述
            self._log(f"  Novelty Score: {problem.novelty.score:.2f}")  # 打印新颖性评分
            self._vlog(f"  Target Properties: {problem.target_properties}")  # 详细模式下打印目标属性
        except Exception as e:
            # 如果问题生成失败，这是致命错误，直接返回
            self._log(f"  Fatal Error: Problem generation failed: {e}")
            return self.state  # 没问题就没法研究，直接结束

        # 检查点：让用户确认是否继续（如果开了交互模式）
        if not await self._checkpoint(
            f"Research problem generated:\n  {problem.problem_statement}\n"
            f"  Novelty: {problem.novelty.score:.2f}"
        ):
            self._log("User aborted")  # 用户选择中止
            return self.state

        # --- 步骤4: 评估初始新颖性 ---
        self._log("[INIT] Assessing initial novelty...")
        from discovery_agents.novelty_assessor import assess_novelty  # 延迟导入
        try:
            initial_novelty = await assess_novelty(
                problem.problem_statement,  # 问题陈述
                self.user_id
            )
            self.state.overall_novelty_score = initial_novelty.score  # 存储新颖性分数
            self._log(f"  Initial Novelty: {initial_novelty.score:.2f}")
        except Exception as e:
            # 新颖性评估失败不是致命的，只是警告
            self._log(f"  Warning: Novelty assessment failed: {e}")

        # ========== 发现循环 ==========
        # 这是核心的"假设→实验→分析→演化"闭环
        current_hypotheses: List[Hypothesis] = []  # 当前的假设列表，初始为空
        
        # 开始迭代，最多跑max_iterations轮
        for iteration in range(1, self.max_iterations + 1):
            iter_start = time.time()  # 记录本轮开始时间
            self.state.current_iteration = iteration  # 更新当前轮数
            rl_action_plan = None  # RL动作计划（本轮）
            
            # 打印轮次分隔线
            self._log(f"\n{'='*60}")
            self._log(f"  Iteration {iteration}/{self.max_iterations}")
            self._log(f"{'='*60}")

            # ========== [PHASE A] 假设生成与辩论 ==========
            self._log(f"\n[PHASE A] Generating hypotheses (Iteration {iteration})...")

            gan_outcome = None  # 初始化GAN结果(可能在异常时保持为None)

            try:
                # 如果不是第一轮，把之前的假设传进去（避免重复）
                prior = current_hypotheses if iteration > 1 else None
                
                if self.use_debate:
                    # 调用假设生成器：3个专家各提假设，然后辩论筛选
                    from discovery_agents.hypothesis_generator import (
                        generate_and_debate_hypotheses,
                    )
                    debate_outcome = await generate_and_debate_hypotheses(
                        problem,      # 研究问题
                        prior,        # 之前的假设（第一轮为None）
                        self.user_id,  # 用户ID
                        skill_context=skills,  # 历史技能上下文
                    )
                else:
                    # 消融模式：使用单个 LLM 直接生成假设，无多专家辩论
                    from discovery_agents.hypothesis_generator import (
                        vanilla_hypothesis_generation,
                    )
                    self._log("  Debate disabled (ablation) -> vanilla LLM generation")
                    debate_outcome = await vanilla_hypothesis_generation(
                        problem, prior, self.user_id,
                        skill_context=skills,  # 历史技能上下文
                    )
                
                # 获取辩论后保留的假设
                current_hypotheses = debate_outcome.refined_hypotheses
                
                # 打印辩论结果
                self._log(
                    f"  Debate: {len(debate_outcome.original_hypotheses)} proposals, "  # 原始提案数
                    f"{debate_outcome.eliminated_count} eliminated, "  # 被淘汰数
                    f"{len(current_hypotheses)} refined"  # 精炼后保留数
                )
                # 打印每个保留的假设
                for h in current_hypotheses:
                    self._log(f"    [{h.id}] {h.statement[:80]}...")  # 只显示前80字符
                self._vlog(f"  Debate Summary: {debate_outcome.debate_summary[:200]}")

                # ========== [PHASE A.2] GAN 对抗优化 ==========
                if self.use_gan:
                    # 在辩论之后运行GAN风格的生成器-验证器对抗, 最大化科学新颖性
                    self._log(f"\n[PHASE A.2] Running GAN adversarial refinement...")
                    self._log(f"  正在运行GAN对抗性优化...")
                    from discovery_agents.hypothesis_generator import run_gan_adversarial_refinement

                    try:
                        gan_outcome = await run_gan_adversarial_refinement(
                            problem, current_hypotheses, user_id=self.user_id,
                            skill_context=skills,  # 历史技能上下文
                        )
                        current_hypotheses = gan_outcome.final_hypotheses
                        self._log(
                            f"  GAN: {gan_outcome.total_proposed} proposed, "
                            f"{gan_outcome.total_accepted} accepted, "
                            f"{len(gan_outcome.rounds)} rounds completed"
                        )
                        # 单独打印GAN产生的假设
                        for h in current_hypotheses:
                            if h.id.startswith("gan_"):
                                self._log(f"    [GAN] [{h.id}] {h.statement[:80]}...")
                        self._vlog(f"  GAN Summary: {gan_outcome.gan_summary[:200]}")
                    except Exception as e:
                        self._log(f"  Warning: GAN refinement failed: {e}, using debate results")
                        self._log(f"  警告: GAN对抗优化失败: {e}, 保留辩论结果")
                        gan_outcome = None
                else:
                    self._log("\n[PHASE A.2] GAN refinement disabled (ablation mode)")

            except Exception as e:
                self._log(f"  Error: Hypothesis generation failed: {e}")
                break  # 假设生成失败是严重错误，跳出循环

            # 检查点
            if not await self._checkpoint(
                f"{len(current_hypotheses)} hypotheses remaining after debate"
            ):
                self._log("User aborted")
                break

            # ========== [PHASE B] 实验规划 ==========
            self._log(f"\n[PHASE B] Planning experiments...")
            from discovery_agents.experiment_planner import plan_experiments

            try:
                # 为每个假设设计实验方案
                plans = await plan_experiments(
                    current_hypotheses,  # 当前的假设列表
                    problem,             # 研究问题
                    self.user_id,
                    use_skills=self.use_skill_solidify,  # 消融时可禁用技能检索
                )
                self._log(f"  Created {len(plans)} experiment plans")
                # 打印每个实验计划的概要
                for p in plans:
                    self._log(f"    {p.hypothesis_id}: {p.method} ({p.estimated_complexity})")
            except Exception as e:
                self._log(f"  Error: Experiment planning failed: {e}")
                plans = []  # 失败了就设为空列表

            # 如果没有实验计划，跳过本轮的执行阶段
            if not plans:
                self._log("  No plans created, skipping execution")
                continue  # 跳到下一轮

            # ========== [PHASE B.5] 实验风险评估 ==========
            risk_assessments = []
            if self.use_risk_assessment:
                self._log(f"\n[PHASE B.5] Assessing experiment risks...")
                from discovery_agents.risk_assessor import assess_experiment_risks
                from models.risk_models import RiskAssessment

                # B2 变体: 仅静态检查时临时覆盖配置
                _orig_skip_llm = None
                if self.use_risk_static_only:
                    import config as _cfg
                    _orig_skip_llm = _cfg.RISK_SKIP_LLM_FOR_LOW
                    _cfg.RISK_SKIP_LLM_FOR_LOW = True
                    self._log("  Static-risk-only mode (ablation): LLM layer skipped for all plans")

                try:
                    risk_assessments = await assess_experiment_risks(plans, self.user_id)

                    # 分离可行和不可行的计划
                    feasible_plans = []
                    filtered_plans = []
                    for plan, assessment in zip(plans, risk_assessments):
                        if assessment.is_feasible:
                            feasible_plans.append(plan)
                        else:
                            filtered_plans.append((plan, assessment))

                    # 安全措施：永远保留至少一个计划（分数最高的那个）
                    if not feasible_plans and risk_assessments:
                        best = max(risk_assessments, key=lambda a: a.feasibility_score)
                        best_plan = plans[risk_assessments.index(best)]
                        feasible_plans = [best_plan]
                        self._log(f"  All plans risky; keeping best: {best_plan.hypothesis_id}")

                    # 日志输出
                    self._log(f"  {len(feasible_plans)}/{len(plans)} plans passed risk assessment")
                    for plan, assessment in filtered_plans:
                        self._log(f"    FILTERED {plan.hypothesis_id}: {assessment.filtering_reason}")
                        if assessment.alternative_suggestion:
                            self._vlog(f"      Alternative: {assessment.alternative_suggestion}")

                    plans = feasible_plans  # 用可行计划替换原列表
                except Exception as e:
                    self._log(f"  Warning: Risk assessment failed: {e}, proceeding with all plans")
                finally:
                    # 恢复配置
                    if _orig_skip_llm is not None:
                        _cfg.RISK_SKIP_LLM_FOR_LOW = _orig_skip_llm
            else:
                self._log(f"\n[PHASE B.5] Risk assessment disabled (ablation mode)")

            # 如果风险评估后没有可行计划，跳过执行
            if not plans:
                self._log("  No feasible plans after risk assessment, skipping execution")
                continue

            # ========== [PHASE C] 实验执行 ==========
            self._log(f"\n[PHASE C] Executing experiments...")
            from discovery_agents.experiment_executor import run_experiments

            try:
                # 执行所有实验计划（AI生成代码并运行）
                results = await run_experiments(plans, self.user_id)
                # 统计成功数
                successes = sum(1 for r in results if r.success)
                self._log(f"  {successes}/{len(results)} experiments successful")
            except Exception as e:
                self._log(f"  Error: Experiment execution failed: {e}")
                results = []  # 失败了就设为空列表

            # ========== [PHASE D] 结果分析与假设演化 ==========
            self._log(f"\n[PHASE D] Analyzing results and evolving hypotheses...")
            from discovery_agents.result_analyzer import (
                analyze_results,     # 分析结果
                evolve_hypotheses,   # 演化假设
            )

            analyses: List[AnalysisResult] = []  # 分析结果列表
            evolution: Optional[HypothesisEvolution] = None  # 演化结果

            # 构建会话级图表目录
            from config import EXPERIMENT_DIR
            figures_dir = str(
                EXPERIMENT_DIR / "saved_code"
                / f"figures_{self.state.session_id}"
            )

            # 只有有实验结果才进行分析
            if results:
                try:
                    # 分析每个实验结果（含增强层：统计检验、可视化、因果推断）
                    analyses = await analyze_results(
                        results,             # 实验结果
                        current_hypotheses,  # 当前假设
                        problem,             # 研究问题
                        self.user_id,
                        output_dir=figures_dir,  # 图表输出目录
                        use_statistical_testing=self.use_statistical_testing,
                        use_visualization=self.use_visualization,
                        use_causal_inference=self.use_causal_inference,
                    )
                    self._log(f"  Completed {len(analyses)} analyses")
                    
                    # 打印每个分析的结论
                    for a in analyses:
                        verdict = "SUPPORTS" if a.supports_hypothesis else "REFUTES"
                        self._log(
                            f"    {a.hypothesis_id}: {verdict} "
                            f"(confidence {a.confidence:.2f})"
                        )
                        # 打印增强分析结果摘要
                        if hasattr(a, 'statistical_tests') and a.statistical_tests:
                            sig = sum(1 for t in a.statistical_tests if t.is_significant)
                            self._log(
                                f"      Statistical tests: {len(a.statistical_tests)} "
                                f"({sig} significant)"
                            )
                        if hasattr(a, 'visualizations') and a.visualizations:
                            self._log(
                                f"      Visualizations: {len(a.visualizations)} charts generated"
                            )
                        if hasattr(a, 'causal_inference') and a.causal_inference:
                            n_rels = len(a.causal_inference.relationships)
                            self._log(
                                f"      Causal inference: {n_rels} relationships identified"
                            )
                except Exception as e:
                    self._log(f"  Warning: Analysis failed: {e}")

            # 只有有分析结果才进行假设演化
            if analyses:
                try:
                    # === RL 优化器：选择动作 ===
                    rl_action_plan = None
                    novelty_before = self.state.overall_novelty_score
                    if self.rl_optimizer:
                        try:
                            rl_action_plan = self.rl_optimizer.select_actions(
                                current_hypotheses, analyses, iteration
                            )
                            self._log(f"  [RL] Optimizer selected actions:")
                            for ha in rl_action_plan.actions:
                                self._log(
                                    f"    {ha.hypothesis_id} -> "
                                    f"{ha.action.upper()} (score={ha.score:.2f})"
                                )
                        except Exception as rl_err:
                            self._log(
                                f"  Warning: RL action selection failed: {rl_err}"
                            )
                            rl_action_plan = None

                    # 根据分析结果演化假设（传入RL指导）
                    evolution = await evolve_hypotheses(
                        current_hypotheses,  # 当前假设
                        analyses,            # 分析结果
                        iteration,           # 当前轮次
                        problem,             # 研究问题
                        self.user_id,
                        rl_action_plan=rl_action_plan,  # RL动作指导
                    )
                    
                    # 打印演化结果
                    self._log(
                        f"  Evolution: {len(evolution.new_hypotheses)} new hypotheses, "
                        f"{len(evolution.eliminated_ids)} eliminated, "
                        f"converged={evolution.converged}"  # 是否收敛
                    )
                    self._vlog(f"  Reasoning: {evolution.evolution_reasoning[:200]}")

                    # === 更新假设列表 ===
                    # 建立ID到更新后假设的映射
                    updated_map = {h.id: h for h in evolution.hypothesis_updates}
                    next_hypotheses = []
                    
                    for h in current_hypotheses:
                        # 跳过被淘汰的假设
                        if h.id in evolution.eliminated_ids:
                            continue
                        # 如果有更新版，用更新版
                        if h.id in updated_map:
                            next_hypotheses.append(updated_map[h.id])
                        else:
                            # 否则保持原样
                            next_hypotheses.append(h)
                    
                    # 加入新生成的假设
                    next_hypotheses.extend(evolution.new_hypotheses)
                    current_hypotheses = next_hypotheses  # 更新当前假设列表

                    # === RL 优化器：更新奖励 ===
                    if self.rl_optimizer and rl_action_plan:
                        try:
                            novelty_after = self.state.overall_novelty_score
                            rewards = self.rl_optimizer.update_from_outcome(
                                action_plan=rl_action_plan,
                                evolution=evolution,
                                analyses=analyses,
                                hypotheses_after=current_hypotheses,
                                novelty_before=novelty_before,
                                novelty_after=novelty_after,
                                iteration=iteration,
                            )
                            # 打印奖励统计
                            avg_reward = (
                                sum(r.total_reward for r in rewards) / len(rewards)
                                if rewards else 0.0
                            )
                            self._log(
                                f"  [RL] Rewards updated: avg={avg_reward:.3f} "
                                f"(n={len(rewards)})"
                            )
                            stats = self.rl_optimizer.get_statistics()
                            arm_summary = " | ".join(
                                f"{name}(n={info['pulls']},\u03bc={info['mean_reward']:.2f})"
                                for name, info in stats.get("arms", {}).items()
                                if info['pulls'] > 0
                            )
                            if arm_summary:
                                self._vlog(f"  [RL] Arms: {arm_summary}")

                            # 持久化 Bandit 状态
                            if RL_PERSISTENCE_ENABLED:
                                try:
                                    from rl_evolution.persistence import save_state
                                    save_state(
                                        self.rl_optimizer.bandit,
                                        self.rl_optimizer.reward_history,
                                        self.state.session_id,
                                    )
                                except Exception:
                                    pass  # 持久化失败不影响主流程
                        except Exception as rl_err:
                            self._log(
                                f"  Warning: RL reward update failed: {rl_err}"
                            )

                except Exception as e:
                    self._log(f"  Warning: Evolution failed: {e}")

            # ========== [PHASE E] 保存记忆与技能 ==========
            self._log(f"\n[PHASE E] Saving to memory and solidifying skills...")

            # 构建本轮的文字总结
            iter_summary = self._build_iteration_summary(
                iteration, problem, current_hypotheses, plans, results, analyses
            )
            # 追加GAN统计信息到摘要
            if gan_outcome is not None:
                iter_summary += (
                    f"\n\nGAN Adversarial Refinement: "
                    f"{gan_outcome.total_proposed} proposed, "
                    f"{gan_outcome.total_accepted} accepted"
                )
            # 保存到记忆库
            if self.use_memory:
                await memory_ops.save_discovery_to_memory(iter_summary, self.user_id)

            # 把成功的实验固化为"技能"，以后可以复用（向量数据库 + 语义索引）
            if self.use_skill_solidify:
                for plan, result in zip(plans, results):
                    if result.success:  # 只有成功的实验才值得存
                        try:
                            sk = skill_registry.solidify_skill(
                                plan, result,
                                session_id=self.state.session_id,
                                iteration=self.state.current_iteration,
                                domain=self.domain,
                                user_id=self.user_id,
                                hypothesis_id=plan.hypothesis_id if hasattr(plan, 'hypothesis_id') else "",
                            )  # 打包成技能并存入向量数据库
                            await skill_registry.save_skill(sk, self.user_id)
                            self._vlog(f"  Skill solidified: {plan.method}")
                        except Exception as e:
                            self._vlog(f"  Warning: Skill save failed: {e}")

            # 更新新颖性评分（基于新发现）
            try:
                updated_novelty = await assess_novelty(
                    f"{problem.problem_statement}\nKey findings: {iter_summary[:500]}",
                    self.user_id,
                )
                self.state.overall_novelty_score = updated_novelty.score
                self._log(f"  Updated novelty score: {updated_novelty.score:.2f}")
            except Exception:
                pass  # 评估失败就算了，不是关键功能

            # 创建本轮的迭代记录
            iteration_record = DiscoveryIteration(
                iteration=iteration,
                problem=problem,
                hypotheses=current_hypotheses,
                plans=plans,
                results=results,
                analysis=analyses,
                evolution=evolution,
                gan_outcome=gan_outcome,
                risk_assessments=risk_assessments,
                rl_action_plan=(
                    rl_action_plan.model_dump() if rl_action_plan else None
                ),
            )
            self.state.iterations.append(iteration_record)  # 添加到状态中
            
            # 打印本轮用时
            iter_time = time.time() - iter_start
            self._log(f"\n  Iteration {iteration} completed in {iter_time:.1f}s")

            # ========== 收敛判断 ==========
            # 如果AI判断已经收敛（找到答案了），就停止
            if evolution and evolution.converged:
                self._log(f"\n  Converged at iteration {iteration}")
                self.state.converged = True
                break

            # 如果没有活跃的假设了（都被确认或反驳了），也停止
            active = [h for h in current_hypotheses if h.status == "active"]
            if not active:
                self._log("  No active hypotheses remaining - stopping")
                self.state.converged = True
                break
            
            # 后备收敛判断：如果达到最大迭代次数且有成功实验，强制收敛
            if iteration >= self.max_iterations:
                successful_experiments = [r for r in results if r.success]
                if successful_experiments:
                    self._log(f"\n  Max iterations reached with {len(successful_experiments)} successful experiments - forcing convergence")
                    self.state.converged = True
                    break

        # ========== 最终总结 ==========
        total_time = time.time() - total_start  # 计算总用时
        
        # 打印最终报告
        self._log(f"\n{'#'*60}")
        self._log(f"# Discovery Complete - Session {self.state.session_id}")
        self._log(f"# Iterations: {self.state.current_iteration}")  # 实际跑了几轮
        self._log(f"# Converged: {self.state.converged}")  # 是否收敛
        self._log(f"# Final Novelty Score: {self.state.overall_novelty_score:.2f}")  # 最终新颖性
        self._log(f"# Total Time: {total_time:.1f}s")  # 总用时
        self._log(f"{'#'*60}")

        # 保存最终总结到记忆库
        final_summary = self._build_final_summary()
        if self.use_memory:
            await memory_ops.save_discovery_to_memory(final_summary, self.user_id)
        
        # 保存完整状态到磁盘（JSON文件）
        self._save_state_to_disk()
        
        return self.state  # 返回完整的发现状态

    def _build_iteration_summary(
        self,
        iteration: int,              # 轮次
        problem: ResearchProblem,    # 研究问题
        hypotheses: List[Hypothesis],  # 假设列表
        plans: List[ExperimentPlan],   # 实验计划列表
        results: List[ExperimentResult],  # 实验结果列表
        analyses: List[AnalysisResult],   # 分析结果列表
    ) -> str:
        """
        Build text summary of this iteration for memory storage
        构建单轮迭代的文字总结，用于存储到记忆库
        
        返回一个字符串，包含本轮的关键信息
        """
        lines = [
            f"Discovery Iteration {iteration}",      # 迭代编号
            f"Problem: {problem.problem_statement}", # 研究问题
            f"Domain: {problem.domain}",             # 领域
            "",
            "Hypotheses:",  # 假设部分
        ]
        # 列出每个假设
        for h in hypotheses:
            lines.append(f"  [{h.id}] ({h.status}) {h.statement}")

        # 实验统计
        lines.append(f"\nExperiments: {len(results)} executed, "
                      f"{sum(1 for r in results if r.success)} successful")

        # 关键发现
        if analyses:
            lines.append("\nKey Findings:")
            for a in analyses:
                verdict = "supports" if a.supports_hypothesis else "refutes"
                lines.append(f"  {a.hypothesis_id}: {verdict} - {', '.join(a.key_findings[:2])}")
        
        return "\n".join(lines)  # 拼接成字符串

    def _build_final_summary(self) -> str:
        """
        Build comprehensive summary of entire discovery session
        构建整个发现会话的综合总结
        
        返回一个字符串，包含整个发现过程的概要
        """
        lines = [
            f"Autonomous Discovery Session {self.state.session_id}",  # 会话ID
            f"Domain: {self.domain}",  # 领域
            f"Problem: {self.state.problem.problem_statement if self.state.problem else 'N/A'}",  # 问题
            f"Iterations: {self.state.current_iteration}",  # 迭代次数
            f"Converged: {self.state.converged}",  # 是否收敛
            f"Novelty: {self.state.overall_novelty_score:.2f}",  # 新颖性
            "",
        ]
        
        # 列出每一轮的信息
        for it in self.state.iterations:
            lines.append(f"--- Iteration {it.iteration} ---")
            # 列出假设
            for h in it.hypotheses:
                lines.append(f"  [{h.id}] ({h.status}) {h.statement[:60]}")
            # 列出分析结论
            for a in it.analysis:
                verdict = "supports" if a.supports_hypothesis else "refutes"
                lines.append(f"  Analysis {a.hypothesis_id}: {verdict}")
            # 列出新假设数
            if it.evolution:
                lines.append(f"  New Hypotheses: {len(it.evolution.new_hypotheses)}")
            # 列出GAN对抗优化结果
            if it.gan_outcome:
                lines.append(
                    f"  GAN Refinement: {it.gan_outcome.total_proposed} proposed, "
                    f"{it.gan_outcome.total_accepted} accepted"
                )
        
        return "\n".join(lines)

    def _save_state_to_disk(self):
        """
        Save discovery state to JSON file for persistence
        把发现状态保存成JSON文件，方便以后查看和分析
        
        文件保存在 saved_code/discovery_<session_id>.json
        """
        from config import EXPERIMENT_DIR  # 导入实验目录配置
        
        # 确定输出目录和文件路径
        output_dir = EXPERIMENT_DIR / "saved_code"
        output_dir.mkdir(parents=True, exist_ok=True)  # 如果目录不存在就创建
        output_path = output_dir / f"discovery_{self.state.session_id}.json"

        try:
            # 打开文件写入JSON
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(
                    self.state.model_dump(),  # 把Pydantic模型转成字典
                    f, 
                    indent=2,           # 缩进2空格，便于阅读
                    default=str,        # 不认识的类型转成字符串
                    ensure_ascii=False  # 允许中文等非ASCII字符
                )
            self._log(f"\n  State saved to: {output_path}")
        except Exception as e:
            self._log(f"\n  Warning: Failed to save state to disk: {e}")
