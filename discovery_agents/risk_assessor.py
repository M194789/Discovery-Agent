"""
Experiment Risk Assessor (Phase B.5) - 实验风险评估器

Evaluates computational feasibility and risk of experiment plans before execution.
在实验执行前评估实验计划的计算可行性和风险。

这个模块是系统的"风险评估师"：
- 接收实验计划列表
- 静态检查包可用性和方法兼容性（Layer 1）
- 对中高复杂度计划调用LLM进行深度评估（Layer 2）
- 过滤不可行的计划，减少浪费的计算资源
"""

from __future__ import annotations

import json
import re
from typing import List
import sys
from pathlib import Path

# === 设置模块搜索路径 ===
current_dir = Path(__file__).resolve().parent.parent
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))

import _utf8_env  # noqa: F401  # 全局 UTF-8 编码修复

# === 导入AI智能体框架 ===
from agents import Agent, Runner
from agents import OpenAIChatCompletionsModel

# === 导入项目配置 ===
from config import (
    client, MODEL_NAME, AGENT_MAX_TURNS,
    RISK_FEASIBILITY_THRESHOLD, RISK_SKIP_LLM_FOR_LOW,
)

# === 导入数据模型 ===
from models.hypothesis_models import ExperimentPlan
from models.risk_models import RiskAssessment


def _is_deepseek() -> bool:
    """检查是否使用DeepSeek模型"""
    return "deepseek" in MODEL_NAME.lower()


# ============================================================
# Layer 1: 静态可行性检查（无需LLM调用）
# ============================================================

# 方法与关键依赖包的映射
_METHOD_CRITICAL_PACKAGES = {
    "DFT": ["pymatgen"],
    "MD": ["ase"],
    "ML_potential": ["torch", "tensorflow"],
}

# 复杂度到基础分数的映射
_COMPLEXITY_BASE_SCORES = {
    "low": 0.9,
    "medium": 0.7,
    "high": 0.5,
}

# 复杂度到超时风险的映射
_COMPLEXITY_TIMEOUT_RISK = {
    "low": "low",
    "medium": "medium",
    "high": "high",
}

# 方法到计算成本的映射
_METHOD_COST_ESTIMATE = {
    "DFT": "heavy",
    "MD": "heavy",
    "ML_potential": "heavy",
    "ML_modeling": "moderate",
    "simulation": "moderate",
    "data_analysis": "trivial",
}


def _get_available_packages() -> list:
    """
    获取当前环境中已安装的科学计算包列表。
    复用 experiment_executor 中的检测逻辑。
    """
    available = []
    check_list = [
        ('numpy', 'numpy'),
        ('scipy', 'scipy'),
        ('pandas', 'pandas'),
        ('matplotlib', 'matplotlib'),
        ('sklearn', 'scikit-learn'),
        ('torch', 'pytorch'),
        ('tensorflow', 'tensorflow'),
        ('pymatgen', 'pymatgen'),
        ('ase', 'ase'),
        ('rdkit', 'rdkit'),
        ('networkx', 'networkx'),
        ('sympy', 'sympy'),
        ('seaborn', 'seaborn'),
        ('plotly', 'plotly'),
        ('numba', 'numba'),
        ('h5py', 'h5py'),
        ('yaml', 'pyyaml'),
        ('requests', 'requests'),
        ('bs4', 'beautifulsoup4'),
        ('lxml', 'lxml'),
        ('PIL', 'pillow'),
        ('cv2', 'opencv-python'),
    ]
    for import_name, display_name in check_list:
        try:
            __import__(import_name)
            available.append(display_name)
        except ImportError:
            pass
    available.extend([
        'json', 'os', 'sys', 'math', 'random',
        'collections', 'itertools', 'csv', 'statistics'
    ])
    return available


def _normalize_package_name(name: str) -> str:
    """标准化包名用于比较（小写、去除版本号等）"""
    return name.lower().strip().split("==")[0].split(">=")[0].split("<=")[0]


def _static_feasibility_check(plan: ExperimentPlan) -> dict:
    """
    Layer 1: 静态可行性检查
    
    不调用LLM，纯粹基于规则检查：
    1. 包可用性
    2. 方法-工具兼容性
    3. 复杂度评估
    
    返回一个dict，包含初步评估结果。
    """
    available_pkgs = _get_available_packages()
    available_normalized = {_normalize_package_name(p) for p in available_pkgs}

    # --- 检查缺失包 ---
    missing = []
    for pkg in plan.tool_chain:
        pkg_norm = _normalize_package_name(pkg)
        # 跳过通用名（python本身、标准库等）
        if pkg_norm in ("python", "python3"):
            continue
        if pkg_norm not in available_normalized:
            missing.append(pkg)

    # --- 检查方法关键依赖 ---
    method_upper = plan.method.upper() if plan.method else ""
    critical_missing = []
    for method_key, required_pkgs in _METHOD_CRITICAL_PACKAGES.items():
        if method_key.upper() in method_upper:
            for rp in required_pkgs:
                rp_norm = _normalize_package_name(rp)
                if rp_norm not in available_normalized:
                    critical_missing.append(rp)

    # --- 计算基础分数 ---
    complexity = plan.estimated_complexity.lower() if plan.estimated_complexity else "medium"
    base_score = _COMPLEXITY_BASE_SCORES.get(complexity, 0.7)

    # 根据缺失包调整分数
    if plan.tool_chain:
        missing_ratio = len(missing) / len(plan.tool_chain)
    else:
        missing_ratio = 0.0

    if missing_ratio > 0.5:
        # 超过50%包缺失 -> 严重扣分
        base_score = min(base_score, 0.1)
    elif critical_missing:
        # 关键包缺失 -> 大幅扣分
        base_score = min(base_score, 0.2)
    elif missing:
        # 有非关键包缺失 -> 适度扣分
        base_score = base_score * (1 - missing_ratio * 0.5)

    # --- 识别失败模式 ---
    failure_modes = []
    if missing:
        failure_modes.append(f"ImportError: missing packages {missing}")
    if critical_missing:
        failure_modes.append(
            f"Critical dependency missing for {plan.method}: {critical_missing}"
        )
    if complexity == "high":
        failure_modes.append("High complexity may cause timeout or resource exhaustion")
    if len(plan.steps) > 10:
        failure_modes.append("Many steps increase chance of intermediate failure")

    # --- 超时风险 ---
    timeout_risk = _COMPLEXITY_TIMEOUT_RISK.get(complexity, "medium")
    if critical_missing:
        timeout_risk = "high"  # 关键包缺失一定会导致快速失败，但标记为高风险

    # --- 计算成本预估 ---
    cost_estimate = _METHOD_COST_ESTIMATE.get(plan.method, "moderate")
    if complexity == "high" and cost_estimate != "excessive":
        # 高复杂度方法提升成本等级
        cost_map = {"trivial": "moderate", "moderate": "heavy", "heavy": "excessive"}
        cost_estimate = cost_map.get(cost_estimate, cost_estimate)

    # --- 风险等级 ---
    if base_score >= 0.8:
        risk_level = "low"
    elif base_score >= 0.6:
        risk_level = "medium"
    elif base_score >= 0.4:
        risk_level = "high"
    else:
        risk_level = "critical"

    return {
        "feasibility_score": round(base_score, 2),
        "risk_level": risk_level,
        "missing_packages": missing,
        "critical_missing": critical_missing,
        "failure_modes": failure_modes,
        "timeout_risk": timeout_risk,
        "computational_cost_estimate": cost_estimate,
        "needs_llm": base_score < 0.8 and complexity != "low",
    }


# ============================================================
# Layer 2: LLM 风险评估智能体
# ============================================================

RISK_ASSESSOR_PROMPT = """
Role: Experiment Risk Assessor
角色：实验风险评估师

You evaluate the computational feasibility and risk of experiment plans
before they are executed. Your goal is to prevent wasted computation
by identifying plans that are likely to fail.
你在实验执行前评估实验计划的计算可行性和风险，
目标是识别可能失败的计划以防止计算资源浪费。

Task: Given an experiment plan and its static analysis results,
provide a comprehensive risk assessment.
任务：给定实验计划及其静态分析结果，提供全面的风险评估。

Assessment Criteria:
评估标准：

1. **Package Availability** - 所有需要的包是否可用？
2. **Method Feasibility** - 选择的计算方法是否适合这个问题？
3. **Code Complexity** - 代码实现的复杂度是否在合理范围？
4. **Data Requirements** - 是否需要真实数据？合成数据是否足够？
5. **Timeout Risk** - 是否可能在超时限制内完成？
6. **Success Likelihood** - 实验成功的总体可能性？

Scoring Guide:
评分指南：
- 0.0-0.2: Infeasible（不可行） - 关键依赖缺失或方法根本不适用
- 0.2-0.4: Very risky（高风险） - 多个严重问题，很可能失败
- 0.4-0.6: Moderate risk（中等风险） - 有一些问题但可能成功
- 0.6-0.8: Low risk（低风险） - 小问题，大概率成功
- 0.8-1.0: Feasible（可行） - 所有条件满足，预期成功

If the plan is high-risk, suggest a simpler alternative that uses
only available packages and lower complexity methods.
如果计划是高风险的，建议使用可用包和更低复杂度方法的替代方案。

Constraints:
约束条件：
- Be pragmatic: synthetic data experiments with common packages are usually feasible
- data_analysis methods with numpy/pandas/scipy are almost always feasible
- Consider that the system can generate synthetic data if real data is unavailable
- Maximum 5 tool calls
"""


def _create_risk_assessor_agent() -> Agent:
    """创建并返回风险评估智能体"""
    instructions = RISK_ASSESSOR_PROMPT

    if _is_deepseek():
        instructions += '''

OUTPUT FORMAT: Return a valid JSON object EXACTLY like this example:
```json
{
  "feasibility_score": 0.75,
  "risk_level": "medium",
  "failure_modes": ["Possible timeout for large dataset", "May need fallback for missing package X"],
  "timeout_risk": "medium",
  "alternative_suggestion": "",
  "computational_cost_estimate": "moderate"
}
```

The feasibility_score MUST be a decimal number between 0.0 and 1.0.
risk_level MUST be one of: "low", "medium", "high", "critical".
timeout_risk MUST be one of: "low", "medium", "high".
computational_cost_estimate MUST be one of: "trivial", "moderate", "heavy", "excessive".
'''

    kwargs = {
        "name": "RiskAssessorAgent",
        "instructions": instructions,
        "model": OpenAIChatCompletionsModel(
            model=MODEL_NAME, openai_client=client
        ),
    }

    if not _is_deepseek():
        kwargs["output_type"] = RiskAssessment

    return Agent(**kwargs)


def _parse_deepseek_risk_output(output: str, plan: ExperimentPlan,
                                 static_result: dict) -> RiskAssessment:
    """
    解析DeepSeek模型的字符串输出为RiskAssessment对象。
    采用与novelty_assessor.py相同的多策略JSON提取。
    """
    try:
        cleaned = re.sub(r'^```(?:json)?\s*|\s*```$', '', output.strip())

        data = None

        # 策略1: 直接解析
        try:
            data = json.loads(cleaned)
        except json.JSONDecodeError:
            pass

        # 策略2: 括号匹配找完整JSON对象
        if data is None:
            brace_count = 0
            start_pos = -1
            end_pos = -1
            in_string = False
            for i, char in enumerate(cleaned):
                if char == '"' and (i == 0 or cleaned[i - 1] != '\\'):
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

        # 策略3: 正则表达式提取
        if data is None:
            json_match = re.search(
                r'\{[\s\S]*?"feasibility_score"[\s\S]*?\}', cleaned
            )
            if json_match:
                try:
                    data = json.loads(json_match.group(0))
                except json.JSONDecodeError:
                    pass

        # 策略4: 直接提取关键数值
        if data is None:
            score_match = re.search(
                r'["\']?feasibility_score["\']?\s*[:\s]\s*([\d.]+)',
                cleaned, re.IGNORECASE
            )
            if score_match:
                try:
                    extracted = float(score_match.group(1))
                    if 0.0 <= extracted <= 1.0:
                        data = {"feasibility_score": extracted}
                except ValueError:
                    pass

        # 所有策略失败：使用静态检查结果
        if data is None:
            print(f"    Warning: Could not parse risk JSON for {plan.hypothesis_id}")
            data = {}

        # 合并LLM输出与静态检查结果
        score = data.get("feasibility_score", static_result["feasibility_score"])
        if isinstance(score, str):
            try:
                score = float(score)
            except ValueError:
                score = static_result["feasibility_score"]

        risk_level = data.get("risk_level", static_result["risk_level"])
        if risk_level not in ("low", "medium", "high", "critical"):
            risk_level = static_result["risk_level"]

        failure_modes = data.get("failure_modes", static_result["failure_modes"])
        if isinstance(failure_modes, str):
            failure_modes = [failure_modes]

        timeout_risk = data.get("timeout_risk", static_result["timeout_risk"])
        if timeout_risk not in ("low", "medium", "high"):
            timeout_risk = static_result["timeout_risk"]

        alternative = data.get("alternative_suggestion", "")
        if not isinstance(alternative, str):
            alternative = str(alternative)

        cost_est = data.get(
            "computational_cost_estimate",
            static_result["computational_cost_estimate"]
        )
        if cost_est not in ("trivial", "moderate", "heavy", "excessive"):
            cost_est = static_result["computational_cost_estimate"]

        is_feasible = score >= RISK_FEASIBILITY_THRESHOLD
        filtering_reason = "" if is_feasible else (
            f"Feasibility score {score:.2f} below threshold {RISK_FEASIBILITY_THRESHOLD}"
        )

        return RiskAssessment(
            hypothesis_id=plan.hypothesis_id,
            plan_method=plan.method,
            feasibility_score=round(score, 2),
            risk_level=risk_level,
            is_feasible=is_feasible,
            missing_packages=static_result["missing_packages"],
            failure_modes=failure_modes,
            timeout_risk=timeout_risk,
            alternative_suggestion=alternative,
            filtering_reason=filtering_reason,
            computational_cost_estimate=cost_est,
        )

    except Exception as e:
        print(f"    Warning: Risk parsing error for {plan.hypothesis_id}: {e}")
        # 解析完全失败时，基于静态检查结果构建评估
        return _build_assessment_from_static(plan, static_result)


def _build_assessment_from_static(
    plan: ExperimentPlan,
    static_result: dict,
) -> RiskAssessment:
    """基于静态检查结果构建RiskAssessment对象"""
    score = static_result["feasibility_score"]
    is_feasible = score >= RISK_FEASIBILITY_THRESHOLD

    # 为不可行计划生成替代建议
    alternative = ""
    if not is_feasible and static_result["missing_packages"]:
        alternative = (
            f"Consider using data_analysis method with numpy/scipy/pandas "
            f"instead of {plan.method}, to avoid dependency on "
            f"{', '.join(static_result['missing_packages'])}"
        )

    filtering_reason = ""
    if not is_feasible:
        reasons = []
        if static_result.get("critical_missing"):
            reasons.append(
                f"Critical packages missing: {static_result['critical_missing']}"
            )
        if static_result["missing_packages"]:
            reasons.append(
                f"Missing packages: {static_result['missing_packages']}"
            )
        if score < RISK_FEASIBILITY_THRESHOLD:
            reasons.append(
                f"Feasibility score {score:.2f} below threshold "
                f"{RISK_FEASIBILITY_THRESHOLD}"
            )
        filtering_reason = "; ".join(reasons)

    return RiskAssessment(
        hypothesis_id=plan.hypothesis_id,
        plan_method=plan.method,
        feasibility_score=score,
        risk_level=static_result["risk_level"],
        is_feasible=is_feasible,
        missing_packages=static_result["missing_packages"],
        failure_modes=static_result["failure_modes"],
        timeout_risk=static_result["timeout_risk"],
        alternative_suggestion=alternative,
        filtering_reason=filtering_reason,
        computational_cost_estimate=static_result["computational_cost_estimate"],
    )


# ============================================================
# 公共接口
# ============================================================

async def assess_experiment_risks(
    plans: List[ExperimentPlan],
    user_id: str = "experiment_system",
) -> List[RiskAssessment]:
    """
    评估实验计划列表的风险和可行性。

    采用两层评估策略：
    - Layer 1 (静态检查): 检查包可用性、方法兼容性、复杂度
    - Layer 2 (LLM评估): 对非明确可行的计划调用LLM进行深度评估

    参数:
        plans: 待评估的实验计划列表
        user_id: 用户标识符

    返回:
        每个计划对应一个RiskAssessment的列表
    """
    assessments: List[RiskAssessment] = []

    for plan in plans:
        print(f"    Assessing risk for {plan.hypothesis_id} ({plan.method})...")

        # === Layer 1: 静态检查 ===
        static_result = _static_feasibility_check(plan)

        # 如果静态检查已经明确不可行（关键包缺失或分数极低），
        # 直接构建评估结果，无需调用LLM
        if static_result["feasibility_score"] < RISK_FEASIBILITY_THRESHOLD:
            print(f"      Static check: INFEASIBLE (score={static_result['feasibility_score']:.2f})")
            assessment = _build_assessment_from_static(plan, static_result)
            assessments.append(assessment)
            continue

        # 如果配置允许，对低复杂度且全部包可用的计划跳过LLM
        complexity = plan.estimated_complexity.lower() if plan.estimated_complexity else "medium"
        if (RISK_SKIP_LLM_FOR_LOW
                and complexity == "low"
                and not static_result["missing_packages"]):
            print(f"      Static check: FEASIBLE (low complexity, all packages available)")
            assessment = _build_assessment_from_static(plan, static_result)
            assessments.append(assessment)
            continue

        # 如果静态检查分数高且无缺失包，也跳过LLM
        if (static_result["feasibility_score"] >= 0.8
                and not static_result["missing_packages"]):
            print(f"      Static check: FEASIBLE (score={static_result['feasibility_score']:.2f})")
            assessment = _build_assessment_from_static(plan, static_result)
            assessments.append(assessment)
            continue

        # === Layer 2: LLM 深度评估 ===
        print(f"      Static check: UNCERTAIN (score={static_result['feasibility_score']:.2f}), invoking LLM...")
        try:
            agent = _create_risk_assessor_agent()

            # 组装LLM输入
            available_pkgs = _get_available_packages()
            input_text = (
                f"User ID: {user_id}\n\n"
                f"Experiment Plan to Assess:\n"
                f"  Hypothesis ID: {plan.hypothesis_id}\n"
                f"  Method: {plan.method}\n"
                f"  Tool Chain: {', '.join(plan.tool_chain)}\n"
                f"  Steps: {'; '.join(plan.steps)}\n"
                f"  Code Specification: {plan.code_specification[:500]}\n"
                f"  Success Criteria: {plan.success_criteria}\n"
                f"  Estimated Complexity: {plan.estimated_complexity}\n\n"
                f"Static Analysis Results:\n"
                f"  Missing Packages: {static_result['missing_packages'] or 'None'}\n"
                f"  Preliminary Score: {static_result['feasibility_score']:.2f}\n"
                f"  Identified Failure Modes: {static_result['failure_modes'] or 'None'}\n\n"
                f"Available Packages: {', '.join(available_pkgs)}\n\n"
                f"Please assess the feasibility and risk of this experiment plan."
            )

            result = await Runner().run(agent, input_text, max_turns=AGENT_MAX_TURNS)
            output = result.final_output

            # 解析输出
            if _is_deepseek() and isinstance(output, str):
                assessment = _parse_deepseek_risk_output(output, plan, static_result)
            elif isinstance(output, RiskAssessment):
                # 非DeepSeek模型：结构化输出，需要补充静态检查的字段
                assessment = RiskAssessment(
                    hypothesis_id=plan.hypothesis_id,
                    plan_method=plan.method,
                    feasibility_score=output.feasibility_score,
                    risk_level=output.risk_level,
                    is_feasible=output.feasibility_score >= RISK_FEASIBILITY_THRESHOLD,
                    missing_packages=static_result["missing_packages"],
                    failure_modes=output.failure_modes or static_result["failure_modes"],
                    timeout_risk=output.timeout_risk,
                    alternative_suggestion=output.alternative_suggestion,
                    filtering_reason="" if output.feasibility_score >= RISK_FEASIBILITY_THRESHOLD else (
                        f"Feasibility score {output.feasibility_score:.2f} "
                        f"below threshold {RISK_FEASIBILITY_THRESHOLD}"
                    ),
                    computational_cost_estimate=output.computational_cost_estimate,
                )
            elif isinstance(output, str):
                # 非DeepSeek但返回了字符串（异常情况）
                assessment = _parse_deepseek_risk_output(output, plan, static_result)
            else:
                assessment = _build_assessment_from_static(plan, static_result)

            status = "FEASIBLE" if assessment.is_feasible else "INFEASIBLE"
            print(f"      LLM assessment: {status} (score={assessment.feasibility_score:.2f})")
            assessments.append(assessment)

        except Exception as e:
            # LLM调用失败，回退到静态结果（默认可行，不阻塞流程）
            print(f"      Warning: LLM risk assessment failed: {e}, using static result")
            assessment = _build_assessment_from_static(plan, static_result)
            assessments.append(assessment)

    return assessments
