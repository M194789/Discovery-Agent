"""
Experiment Executor (Module 3) - 实验执行器（模块3）

Executes computational experiments by generating and running Python code via LLM.
通过LLM生成并运行Python代码来执行计算实验。

这个模块是系统的"实验技术员"：
- 接收实验计划（设计好的实验方案）
- 让AI编写Python代码
- 在沙箱环境中安全执行代码
- 收集执行结果，支持错误重试
"""

from __future__ import annotations  # 启用未来版本的类型注解语法

import _utf8_env  # noqa: F401  # 全局 UTF-8 编码修复

import asyncio       # 异步编程库，让代码执行不阻塞主程序
import json          # JSON处理（虽然这里没直接用，但可能用于调试）
import re            # 正则表达式，用于清理代码格式
import subprocess    # 子进程管理，用于运行生成的代码
import tempfile     # 临时文件，存放生成的Python代码
from typing import List, Optional  # 类型提示

# === 修复直接模块使用时的导入问题 ===
import sys
from pathlib import Path
# 获取当前文件的父父目录（项目根目录）
current_dir = Path(__file__).resolve().parent.parent
# 如果项目根目录不在Python搜索路径中，添加进去
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))

# === 导入配置 ===
from config import DEEP_SOLVER_TIMEOUT, client, MODEL_NAME
# DEEP_SOLVER_TIMEOUT = 代码执行超时时间（秒）
# client = API客户端，MODEL_NAME = 模型名称

# === 导入数据模型 ===
from models.hypothesis_models import ExperimentPlan   # 实验计划
from models.experiment_models import ExperimentResult  # 实验结果

# 代码生成/执行失败时的最大重试次数
_MAX_CODE_RETRIES = 3  # 最多重试3次（总共4次尝试）


def _get_available_packages() -> list:
    """
    Check which scientific packages are actually installed.
    检查哪些科学计算包实际安装了。
    
    为什么要检查？因为AI写代码时需要知道：
    - 哪些包可以用（已安装的）
    - 哪些包不能用（没安装的）
    这样就不会写出"import xxx"然后报ImportError的代码
    """
    available = []  # 存放已安装的包列表
    
    # 要检查的科学计算包列表
    # 格式：(导入名, 显示名)
    check_list = [
        ('numpy', 'numpy'),           # 数值计算基础
        ('scipy', 'scipy'),           # 科学计算
        ('pandas', 'pandas'),         # 数据处理
        ('matplotlib', 'matplotlib'), # 数据可视化
        ('sklearn', 'scikit-learn'),  # 机器学习
        ('torch', 'pytorch'),         # 深度学习(PyTorch)
        ('tensorflow', 'tensorflow'), # 深度学习(TensorFlow)
        ('pymatgen', 'pymatgen'),     # 材料科学
        ('ase', 'ase'),               # 原子模拟
        ('rdkit', 'rdkit'),           # 化学信息学
        ('networkx', 'networkx'),     # 图/网络分析
        ('sympy', 'sympy'),           # 符号计算
        ('seaborn', 'seaborn'),       # 统计数据可视化
        ('plotly', 'plotly'),         # 交互式图表
        ('numba', 'numba'),           # JIT编译加速
        ('h5py', 'h5py'),             # HDF5文件格式
        ('yaml', 'pyyaml'),           # YAML配置文件
        ('requests', 'requests'),     # HTTP请求
        ('bs4', 'beautifulsoup4'),    # HTML解析
        ('lxml', 'lxml'),             # XML解析
        ('PIL', 'pillow'),            # 图像处理
        ('cv2', 'opencv-python'),     # 计算机视觉
    ]
    
    # 尝试导入每个包，能导入说明已安装
    for import_name, display_name in check_list:
        try:
            __import__(import_name)  # 尝试导入
            available.append(display_name)  # 成功，加入列表
        except ImportError:
            pass  # 失败，说明没装，跳过
    
    # 标准库肯定都有，直接加进去
    available.extend([
        'json', 'os', 'sys', 'math', 'random',
        'collections', 'itertools', 'csv', 'statistics'
    ])
    
    return available  # 返回已安装的包列表


async def _generate_experiment_code(
    plan: ExperimentPlan,                   # 输入：实验计划
    previous_error: str | None = None,      # 输入：上次的错误（重试时用）
) -> str:                                   # 输出：生成的Python代码
    """
    Use LLM to generate Python code for the experiment.
    使用大模型生成实验的Python代码。
    
    这是"让AI写代码"的核心函数：
    1. 告诉AI实验要做什么
    2. 告诉AI电脑上有哪些包可用
    3. 如果是重试，告诉AI上次错在哪
    4. AI返回一段Python代码
    """
    # 在函数内部导入，避免循环导入问题
    from agents import Agent, Runner, OpenAIChatCompletionsModel

    # 获取已安装的包列表
    available_pkgs = _get_available_packages()
    available_str = ', '.join(available_pkgs)  # 转成字符串，方便放入提示词

    # 组装错误反馈（如果是重试）
    error_feedback = ""
    if previous_error:
        error_feedback = (
            f"\n\n上次执行失败，错误信息：\n{previous_error[:500]}\n"  # 只取前500字符
            f"请修复这个错误。不要用任何导致ImportError的包。"
            f"只使用下面列出的已安装的包。"
        )

    # === 组装提示词 ===
    # 这是给AI的"任务书"，非常详细
    prompt = f"""你是一个计算科学专家。请写Python代码执行以下实验。

实验方法: {plan.method}

要实现的步骤:
{chr(10).join(f"  {i+1}. {s}" for i, s in enumerate(plan.steps))}

详细规格:
{plan.code_specification}

成功标准:
{plan.success_criteria}
{error_feedback}

关键规则:
1. 写完整的、可运行的Python代码
2. **严格限制**: 只能用这些已确认安装的包: {available_str}
3. **禁止导入**: 列表之外的任何包都会导致ImportError，绝对不要尝试
4. **替代方案**: 如果需要的专业包没装，用numpy/scipy/pandas/标准库实现相同功能
5. **错误处理**: 每个可能出错的地方都要加try-except，确保程序不会崩溃
6. **结果输出**: 在结尾用print()清晰打印所有数值结果，格式: "Result: XXX"
7. **文件编码**: 所有文件操作都用 encoding='utf-8'
8. **字符限制**: print里避免非ASCII字符，用英文输出
9. **图表保存**: 画图保存文件（plt.savefig）而不是plt.show()
10. **模拟数据**: 如果没有真实数据，用numpy生成合理的模拟数据
11. **代码检查**: 生成代码前，检查每个import是否在允许列表中

只输出Python代码，不要markdown格式，不要解释。"""

    # 创建代码生成智能体
    agent = Agent(
        name="CodeGenerator",                # 名字：代码生成器
        instructions=prompt,                  # 工作说明（上面的提示词）
        model=OpenAIChatCompletionsModel(     # 连接的AI模型
            model=MODEL_NAME, openai_client=client
        ),
    )

    # === 调用大模型生成代码 ===
    # 这里是真正连接大模型的地方！
    result = await Runner().run(agent, "Generate the experiment code.", max_turns=5)
    code = result.final_output  # 获取AI生成的代码

    # 清理代码（去掉可能的markdown标记）
    if isinstance(code, str):
        # 去掉 ```python 和 ``` 这样的标记
        code = re.sub(r'^```(?:python)?\s*|\s*```$', '', code.strip())

    return code  # 返回清理后的代码


async def _execute_python_code(code: str, timeout: int = 300) -> dict:
    """
    Execute Python code in a subprocess and capture output.
    在子进程中执行Python代码，捕获输出。
    
    安全执行机制：
    1. 代码写入临时文件（隔离）
    2. 在子进程中运行（崩溃不影响主程序）
    3. 设置超时保护（防止死循环）
    4. 捕获所有输出（stdout和stderr）
    5. 自动清理临时文件
    """
    
    # 把代码写入临时文件
    # delete=False 表示不自动删除，我们需要手动控制
    # encoding='utf-8' 确保中文不会乱码
    with tempfile.NamedTemporaryFile(
        mode='w', suffix='.py', delete=False, encoding='utf-8'
    ) as f:
        f.write(code)  # 写入代码
        temp_file = f.name  # 获取临时文件路径
    
    try:
        # 在子进程中运行代码
        # 传递 PYTHONUTF8=1 确保子进程也使用 UTF-8 编码，避免 GBK 崩溃
        import os as _os
        _env = _os.environ.copy()
        _env["PYTHONUTF8"] = "1"
        _env["PYTHONIOENCODING"] = "utf-8"
        process = await asyncio.create_subprocess_exec(
            sys.executable, "-X", "utf8", temp_file,    # 命令：python -X utf8 xxx.py
            stdout=asyncio.subprocess.PIPE,  # 捕获标准输出
            stderr=asyncio.subprocess.PIPE,  # 捕获错误输出
            cwd=str(Path(temp_file).parent),  # 设置工作目录
            env=_env,  # 传递 UTF-8 环境变量
        )
        
        try:
            # 等待执行完成，设置超时
            stdout, stderr = await asyncio.wait_for(
                process.communicate(),
                timeout=timeout  # 超时时间（秒）
            )
            
            # 解码输出（用utf-8，出错时用替换模式）
            stdout_str = stdout.decode('utf-8', errors='replace')
            stderr_str = stderr.decode('utf-8', errors='replace')
            
            return {
                "success": process.returncode == 0,  # 返回码0表示成功
                "stdout": stdout_str,   # 标准输出（print的内容）
                "stderr": stderr_str,   # 错误输出（报错信息）
                "returncode": process.returncode  # 返回码
            }
            
        except asyncio.TimeoutError:
            # 超时了，杀掉进程
            process.kill()
            return {
                "success": False,
                "stdout": "",
                "stderr": f"执行超时（超过{timeout}秒）",
                "returncode": -1
            }
            
    except Exception as e:
        # 其他异常
        return {
            "success": False,
            "stdout": "",
            "stderr": f"执行错误: {str(e)}",
            "returncode": -1
        }
    finally:
        # 清理临时文件（无论成功与否都要清理）
        try:
            Path(temp_file).unlink()
        except:
            pass  # 清理失败也没关系


class RealExperimentSolver:
    """
    真实实验执行器 - 使用LLM生成并执行代码
    
    注意：这个类目前是一个占位符/兼容层，
    实际功能由下面的 execute_experiment() 函数提供
    """
    
    async def solve(self, query: str, user_id: str = "experiment_system"):
        """
        解析query中的实验计划，生成并执行代码
        
        注意：当前实现需要直接传递plan对象，
        而不是通过query字符串。
        这是一个过渡方案 - 实际应该直接调用 execute_experiment()
        """
        # 从query解析实验计划（当前未实现）
        # query由_compose_query生成
        
        # 目前需要直接传递plan对象
        # 这是临时解决方案 - 实际应该直接传递plan对象
        return {
            "success": False,
            "final_code": "# Error: RealExperimentSolver requires direct plan access",
            "execution_results": "Please use execute_experiment() instead"
        }


def _compose_query(plan: ExperimentPlan) -> str:
    """
    Translate an ExperimentPlan into a DeepSolver query string.
    将实验计划转换为DeepSolver查询字符串。
    
    这个函数把结构化的实验计划转换成一段描述文本，
    用于传递给AI理解。
    """
    # 将步骤列表格式化为编号列表
    steps_text = "\n".join(f"  {i+1}. {s}" for i, s in enumerate(plan.steps))
    # 将工具链格式化为逗号分隔的字符串
    tools_text = ", ".join(plan.tool_chain) if plan.tool_chain else "any appropriate tools"

    return (
        f"计算实验请求\n"
        f"================================\n\n"
        f"方法: {plan.method}\n"
        f"需要的包/工具: {tools_text}\n\n"
        f"实验步骤:\n{steps_text}\n\n"
        f"详细规格:\n{plan.code_specification}\n\n"
        f"成功标准: {plan.success_criteria}\n\n"
        f"重要：清晰打印所有数值结果。"
        f"输出必须包含评估成功标准所需的数据。"
    )


async def execute_experiment(
    plan: ExperimentPlan,                   # 输入：实验计划
    user_id: str = "experiment_system",     # 输入：用户ID
) -> ExperimentResult:                      # 输出：实验结果
    """
    Execute a single experiment plan with automatic retry on failure.
    执行单个实验计划，失败时自动重试。

    当发生技术失败（ImportError、SyntaxError等）时，
    代码生成器会被再次调用，并传入错误反馈以便修复问题。

    参数:
        plan: 要执行的实验计划
        user_id: 用于记忆上下文的用户标识符

    返回:
        包含执行结果的ExperimentResult对象
    """
    previous_error = None  # 上次的错误信息
    last_code = ""         # 上次生成的代码
    last_output = ""       # 上次的输出

    # 最多尝试 _MAX_CODE_RETRIES + 1 次（1次正常 + 2次重试）
    for attempt in range(_MAX_CODE_RETRIES + 1):
        # 如果不是第一次，显示重试标记
        attempt_label = f" (第{attempt+1}次尝试)" if attempt > 0 else ""
        print(f"    正在为 {plan.hypothesis_id} 生成代码{attempt_label}...")

        try:
            # 第一步：生成代码（如果是重试，会带上错误反馈）
            code = await _generate_experiment_code(plan, previous_error=previous_error)

            # 检查代码是否太短（可能生成失败了）
            if not code or len(code) < 50:
                previous_error = "生成的代码太短或为空。请写完整代码。"
                continue  # 跳过这次，进入下一次重试

            last_code = code  # 保存代码
            print(f"    正在执行 {plan.hypothesis_id} 的代码{attempt_label}...")

            # 第二步：执行生成的代码
            exec_result = await _execute_python_code(code, timeout=DEEP_SOLVER_TIMEOUT)

            # 合并输出（stdout和stderr）
            full_output = exec_result["stdout"]
            if exec_result["stderr"]:
                full_output += f"\n\n[错误输出]:\n{exec_result['stderr']}"
            last_output = full_output

            # 如果成功，立即返回结果
            if exec_result["success"]:
                print(f"    {plan.hypothesis_id} 代码执行成功！")
                return ExperimentResult(
                    hypothesis_id=plan.hypothesis_id,
                    plan_summary=f"{plan.method}: {plan.steps[0] if plan.steps else 'N/A'}",
                    success=True,
                    final_code=code,
                    execution_output=full_output,
                    error_log="",
                )

            # 执行失败，准备错误反馈用于重试
            stderr = exec_result["stderr"]
            
            # 检查是否是可重试的技术错误
            retryable_keywords = [
                "ImportError", "ModuleNotFoundError",  # 导入错误
                "SyntaxError", "IndentationError",     # 语法错误
                "NameError", "TypeError",              # 命名/类型错误
                "FileNotFoundError", "AttributeError", # 文件/属性错误
                "UnicodeEncodeError", "UnicodeDecodeError",  # 编码错误
            ]
            # 检查错误信息中是否包含任何可重试的关键词
            is_retryable = any(kw in stderr for kw in retryable_keywords)

            # 如果是可重试错误，且还没达到最大重试次数
            if is_retryable and attempt < _MAX_CODE_RETRIES:
                previous_error = stderr[:800]  # 保存错误信息（限制长度）
                print(f"    检测到可重试错误，将重新生成代码: {stderr[:100]}...")
                continue  # 继续下一次尝试
            else:
                # 不可重试或已达最大重试次数，返回失败结果
                return ExperimentResult(
                    hypothesis_id=plan.hypothesis_id,
                    plan_summary=f"{plan.method}: {plan.steps[0] if plan.steps else 'N/A'}",
                    success=False,
                    final_code=code,
                    execution_output=full_output,
                    error_log=stderr,
                )

        except asyncio.TimeoutError:
            # 整体超时
            return ExperimentResult(
                hypothesis_id=plan.hypothesis_id,
                plan_summary=f"{plan.method} (超时)",
                success=False,
                final_code=last_code,
                error_log=f"实验超时（超过{DEEP_SOLVER_TIMEOUT}秒）",
            )
        except Exception as e:
            # 其他异常，尝试重试
            if attempt < _MAX_CODE_RETRIES:
                previous_error = str(e)
                continue
            return ExperimentResult(
                hypothesis_id=plan.hypothesis_id,
                plan_summary=f"{plan.method} (错误)",
                success=False,
                final_code=last_code,
                error_log=str(e),
            )

    # 理论上不会执行到这里，但为了安全加兜底
    return ExperimentResult(
        hypothesis_id=plan.hypothesis_id,
        plan_summary=f"{plan.method}: 所有重试已用尽",
        success=False,
        final_code=last_code,
        execution_output=last_output,
        error_log=previous_error or "所有重试都失败了",
    )


async def run_experiments(
    plans: List[ExperimentPlan],             # 输入：实验计划列表
    user_id: str = "experiment_system",     # 输入：用户ID
) -> List[ExperimentResult]:                # 输出：实验结果列表
    """
    Execute all experiment plans sequentially.
    顺序执行所有实验计划。

    为什么要顺序执行？
    因为实验可能会操作共享的temp_code工作空间中的文件，
    并行执行可能会导致文件冲突。

    参数:
        plans: 要执行的实验计划列表
        user_id: 用于DeepSolver的用户标识符

    返回:
        实验结果列表
    """
    results: List[ExperimentResult] = []  # 存放所有结果

    # 遍历每个实验计划
    for i, plan in enumerate(plans):
        print(f"    执行实验 {i+1}/{len(plans)}，假设: {plan.hypothesis_id}...")
        
        # 执行单个实验
        result = await execute_experiment(plan, user_id)
        
        # 打印结果状态
        status = "成功" if result.success else "失败"
        print(f"    实验 {i+1} 结果: {status}")
        
        # 添加到结果列表
        results.append(result)

    return results  # 返回所有实验结果
