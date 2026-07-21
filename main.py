#!/usr/bin/env python3
"""
自主科学发现系统 — 命令行入口点。

用法:
    python -m Experiment.main --domain "catalysis" --problem "Design efficient CO2 conversion catalysts"
    python -m Experiment.main --domain "battery materials" --max-iterations 3 --interactive --verbose
"""

from __future__ import annotations  # 启用未来版本的注解功能，支持 Python 3.7+ 的新特性

import argparse  # 导入命令行参数解析模块，用于处理用户输入的参数
import asyncio  # 导入异步 IO 模块，用于运行异步任务
import os  # 导入操作系统接口模块，用于访问环境变量等
import sys  # 导入系统模块，用于操作 Python 运行时环境


def parse_args() -> argparse.Namespace:
    """解析命令行参数并返回命名空间对象。"""
    parser = argparse.ArgumentParser(
        description="基于 CASCADE 构建的自主科学发现系统",  # 程序的描述信息
        formatter_class=argparse.RawDescriptionHelpFormatter,  # 保持帮助信息的原始格式
        epilog="""
示例:
  python -m Experiment.main --domain "catalysis" \\
      --problem "Design efficient CO2 conversion catalysts at low temperature" \\
      --max-iterations 3 --verbose

  python -m Experiment.main --domain "battery materials" --interactive
        """,  # 帮助信息末尾的使用示例
    )
    parser.add_argument(
        "--domain",
        required=True,  # 该参数为必需参数
        help="研究领域（例如：'catalysis', 'battery materials', 'drug design'）",  # 参数帮助信息
    )
    parser.add_argument(
        "--problem",
        default=None,  # 默认值为 None，表示不指定种子问题
        help="可选的种子问题陈述。如果省略，智能体将自动生成一个问题。",  # 参数帮助信息
    )
    parser.add_argument(
        "--max-iterations",
        type=int,  # 参数类型转换为整数
        default=3,  # 默认迭代次数为 3
        help="发现循环的最大迭代次数（默认：3）",  # 参数帮助信息
    )
    parser.add_argument(
        "--model",
        default=None,  # 默认值为 None，使用环境变量或默认模型
        help="大语言模型名称（默认：环境变量 AGENT_MODEL_NAME 或 'o3'）",  # 参数帮助信息
    )
    parser.add_argument(
        "--user-id",
        default="experiment_system",  # 默认用户 ID 为"experiment_system"
        help="记忆操作用户标识符（默认：'experiment_system'）",  # 参数帮助信息
    )
    parser.add_argument(
        "--interactive",
        action="store_true",  # 布尔标志，存在即为 True，不存在为 False
        help="启用人机交互检查点（human-in-the-loop）",  # 参数帮助信息
    )
    parser.add_argument(
        "--verbose",
        action="store_true",  # 布尔标志，存在即为 True，不存在为 False
        help="打印详细的智能体输出信息",  # 参数帮助信息
    )
    return parser.parse_args()  # 解析命令行参数并返回命名空间对象


def main():
    """主函数，执行自主科学发现流程。"""
    args = parse_args()  # 解析命令行参数

    # 如果指定了模型名称，覆盖环境变量
    if args.model:  # 检查是否提供了 --model 参数
        os.environ["AGENT_MODEL_NAME"] = args.model  # 设置环境变量 AGENT_MODEL_NAME 为用户指定的模型名称

    # 在设置环境变量后导入配置模块（配置模块在导入时读取环境变量）
    # 添加当前目录到 sys.path 以支持直接导入
    import sys  # 导入系统模块
    from pathlib import Path  # 从 pathlib 导入 Path 类用于路径操作
    current_dir = Path(__file__).resolve().parent.parent  # 获取当前文件所在目录的父目录（项目根目录）
    if str(current_dir) not in sys.path:  # 如果项目根目录不在 sys.path 中
        sys.path.insert(0, str(current_dir))  # 将项目根目录添加到 sys.path 的开头，以便导入项目模块
    
    from config import MODEL_NAME  # noqa: F401  # 导入模型名称（触发配置加载）
    from orchestrator import DiscoveryOrchestrator  # 从编排器模块导入发现编排器类

    orchestrator = DiscoveryOrchestrator(
        domain=args.domain,  # 研究领域，来自命令行参数
        initial_problem=args.problem,  # 初始问题，可为 None（由智能体生成）
        max_iterations=args.max_iterations,  # 最大迭代次数
        model_name=args.model or os.getenv("AGENT_MODEL_NAME", "o3"),  # 模型名称：优先使用命令行参数，其次环境变量，最后默认值"o3"
        user_id=args.user_id,  # 用户 ID，用于记忆操作
        interactive=args.interactive,  # 是否启用交互模式
        verbose=args.verbose,  # 是否启用详细输出模式
    )

    state = asyncio.run(orchestrator.run())  # 异步运行编排器的主循环，执行完整的科学发现流程

    # 根据收敛状态设置退出码
    if state.converged:  # 检查发现流程是否已收敛（成功完成）
        sys.exit(0)  # 成功则退出码为 0
    else:
        sys.exit(1)  # 未收敛则退出码为 1，表示未完成


if __name__ == "__main__":
    main()  # 当脚本直接运行时，调用主函数执行科学发现流程
