"""
自主科学发现系统 — Gradio Web 交互界面。

启动方式:
    python app.py

浏览器访问 http://localhost:7866
"""
# 上面的三引号是 Python 的多行字符串，也叫"文档字符串"（docstring）
# 它的作用是说明这个文件是干什么的
# 当有人用 help(app) 或者在 IDE 里查看时，能看到这段说明
# 这里告诉我们：这是一个 Web 界面，用 Gradio 做的，启动后访问 localhost:7866

from __future__ import annotations

import _utf8_env  # noqa: F401  # 强制 UTF-8 输出（修复 Windows GBK 控制台 emoji 崩溃）

import asyncio
import os
import queue
import sys
from pathlib import Path
from typing import Optional

# 确保项目根目录在 sys.path 中
_project_dir = Path(__file__).resolve().parent
if str(_project_dir) not in sys.path:
    sys.path.insert(0, str(_project_dir))

# Disable OpenAI Agents tracing to avoid API key issues
try:
    from agents import set_tracing_disabled
    set_tracing_disabled(True)
except ImportError:
    # If agents package is not available or incompatible, skip tracing setup
    pass

import gradio as gr

from ui_bridge import LogEmitter, HistoryManager, StateRenderer

# 历史管理器实例
_saved_dir = _project_dir / "saved_code"
_history_mgr = HistoryManager(_saved_dir)

# 当前运行任务引用（用于停止功能）
_current_task: Optional[asyncio.Task] = None


# ---------------------------------------------------------------------------
# 辅助函数
# ---------------------------------------------------------------------------

def build_enhanced_problem(
    problem: str,
    methods: list,
    targets: str,
    constraints: str,
) -> Optional[str]:
    """将高级参数拼入 initial_problem 字符串。"""
    parts = []
    if problem and problem.strip():
        parts.append(problem.strip())
    if methods:
        parts.append(f"\n偏好的计算方法: {', '.join(methods)}")
    if targets and targets.strip():
        parts.append(f"\n目标性质: {targets.strip()}")
    if constraints and constraints.strip():
        parts.append(f"\n约束条件: {constraints.strip()}")
    return "\n".join(parts) if parts else None


async def _auto_approve(message: str) -> bool:
    """Gradio 模式下自动批准检查点。"""
    return True


# ---------------------------------------------------------------------------
# 核心运行函数 (async generator)
# outputs 顺序: [progress_md, log_output, hypotheses_df, results_df, analysis_md, download_file]
# ---------------------------------------------------------------------------

async def run_discovery(
    domain: str,
    problem: str,
    max_iter: int,
    model: str,
    methods: list,
    targets: str,
    constraints: str,
    interactive: bool,
    verbose: bool,
    user_id: str,
):
    """主发现任务的 async generator，用于流式更新 Gradio 组件。

    yields 6 个值的元组，对应 outputs: [progress_md, log_output, hypotheses_df, results_df, analysis_md, download_file]
    """
    global _current_task

    _no_update = gr.update()

    if not domain or not domain.strip():
        yield (
            "**错误**: 请输入研究领域",  # progress_md
            "错误：研究领域不能为空。",   # log_output
            _no_update,                   # hypotheses_df
            _no_update,                   # results_df
            _no_update,                   # analysis_md
            _no_update,                   # download_file
        )
        return

    # 设置模型环境变量
    if model and model.strip():
        os.environ["AGENT_MODEL_NAME"] = model.strip()

    emitter = LogEmitter()
    emitter.max_iterations = int(max_iter)

    enhanced_problem = build_enhanced_problem(problem, methods, targets, constraints)

    # yield 初始状态
    yield (
        StateRenderer.progress_markdown("正在初始化...", 0, int(max_iter)),
        "正在启动发现流程...\n",
        [],
        [],
        "*等待运行...*",
        gr.update(value=None, visible=False),
    )

    # 延迟导入（避免在 Gradio 启动时触发重量级依赖）
    from orchestrator import DiscoveryOrchestrator

    orch = DiscoveryOrchestrator(
        domain=domain.strip(),
        initial_problem=enhanced_problem,
        max_iterations=int(max_iter),
        model_name=model.strip() if model else os.getenv("AGENT_MODEL_NAME", "o3"),
        user_id=user_id.strip() if user_id else "experiment_system",
        interactive=interactive,
        verbose=verbose,
        log_callback=emitter.emit,
        checkpoint_callback=_auto_approve,
    )

    # 后台启动编排器
    loop = asyncio.get_event_loop()
    task = loop.create_task(orch.run())
    _current_task = task

    # 流式 yield 日志更新
    last_log_len = 0
    last_log_content = ""
    no_change_count = 0
    try:
        while not task.done():
            # 排空队列（消息已通过 emit 累积到 full_log）
            while not emitter.queue.empty():
                try:
                    emitter.queue.get_nowait()
                except queue.Empty:
                    break

            current_log = emitter.get_full_log()
            current_len = len(current_log)
            
            # 只在日志有实际变化时才更新UI
            if current_len != last_log_len:
                last_log_len = current_len
                last_log_content = current_log
                no_change_count = 0

                # 中间状态：尝试提取当前迭代的结构化数据
                hyp_rows = []
                res_rows = []
                ana_md = "*运行中...*"
                if orch.state and orch.state.iterations:
                    last_iter = orch.state.iterations[-1]
                    hyp_rows = StateRenderer.hypotheses_to_dataframe(last_iter.hypotheses)
                    res_rows = StateRenderer.results_to_dataframe(last_iter.results, last_iter.plans)
                    if last_iter.analysis:
                        ana_md = StateRenderer.analysis_to_markdown(last_iter.analysis)

                yield (
                    StateRenderer.progress_markdown(
                        emitter.phase, emitter.iteration, emitter.max_iterations,
                        elapsed=emitter.get_elapsed()
                    ),
                    current_log,
                    hyp_rows if hyp_rows else _no_update,
                    res_rows if res_rows else _no_update,
                    ana_md,
                    _no_update,
                )
            else:
                # 日志无变化，增加计数
                no_change_count += 1
                # 如果连续10次(5秒)无变化，减少刷新频率
                if no_change_count > 10:
                    await asyncio.sleep(1.0)
                    continue

            await asyncio.sleep(0.5)

        # 任务完成，获取最终结果
        state = task.result()
        final_log = emitter.get_full_log() + "\n--- 发现流程完成 ---\n"

        # 提取结构化数据
        final_hypotheses = []
        final_results = []
        final_plans = []
        final_analyses = []

        if state and hasattr(state, "iterations") and state.iterations:
            last_iter = state.iterations[-1]
            final_hypotheses = getattr(last_iter, "hypotheses", [])
            final_results = getattr(last_iter, "results", [])
            final_plans = getattr(last_iter, "plans", [])
            final_analyses = getattr(last_iter, "analysis", [])

        # 查找保存的 JSON 文件
        json_path = None
        if state and hasattr(state, "session_id"):
            candidate = _saved_dir / f"discovery_{state.session_id}.json"
            if candidate.exists():
                json_path = str(candidate)

        converged_str = "Yes" if (state and state.converged) else "No"
        novelty_str = f"{state.overall_novelty_score:.2f}" if state else "N/A"

        final_log += f"\n会话 ID: {state.session_id if state else 'N/A'}"
        final_log += f"\n收敛: {converged_str}"
        final_log += f"\n总新颖性: {novelty_str}"

        hyp_rows = StateRenderer.hypotheses_to_dataframe(final_hypotheses)
        res_rows = StateRenderer.results_to_dataframe(final_results, final_plans)
        ana_md = StateRenderer.analysis_to_markdown(final_analyses)

        yield (
            StateRenderer.progress_markdown(
                "已完成", emitter.max_iterations, emitter.max_iterations,
                elapsed=emitter.get_elapsed()
            ),
            final_log,
            hyp_rows if hyp_rows else [],
            res_rows if res_rows else [],
            ana_md,
            gr.update(value=json_path, visible=True) if json_path else gr.update(value=None, visible=False),
        )

    except asyncio.CancelledError:
        yield (
            "**已停止**",
            emitter.get_full_log() + "\n--- 用户已停止发现流程 ---\n",
            _no_update,
            _no_update,
            _no_update,
            _no_update,
        )
    except Exception as e:
        yield (
            f"**错误**: {str(e)[:200]}",
            emitter.get_full_log() + f"\n--- 错误: {e} ---\n",
            _no_update,
            _no_update,
            _no_update,
            _no_update,
        )
    finally:
        _current_task = None


def stop_discovery():
    """停止当前运行的发现任务。"""
    global _current_task
    if _current_task and not _current_task.done():
        _current_task.cancel()
        return "已发送停止信号..."
    return "当前没有运行中的任务"


# ---------------------------------------------------------------------------
# 历史记录 Tab 事件处理
# ---------------------------------------------------------------------------

def load_history_list(keyword: str = "", domain_filter: str = ""):
    """加载历史会话列表。"""
    if keyword or domain_filter:
        rows = _history_mgr.search_sessions(keyword, domain_filter)
    else:
        rows = _history_mgr.list_sessions()
    return rows


def on_history_select(evt: gr.SelectData, history_data):
    """点击历史记录行时，显示详情。"""
    if evt.index is None or history_data is None:
        return "*请选择一条记录*", gr.update(value=None, visible=False)

    row_idx = evt.index[0] if isinstance(evt.index, (list, tuple)) else evt.index
    if row_idx < 0 or row_idx >= len(history_data):
        return "*无效选择*", gr.update(value=None, visible=False)

    row = history_data[row_idx]
    filename = row[0]

    data = _history_mgr.load_session(filename)
    if not data:
        return f"*无法加载文件: {filename}*", gr.update(value=None, visible=False)

    detail_md = _history_mgr.render_detail_markdown(data)
    file_path = _history_mgr.get_file_path(filename)

    if file_path:
        return detail_md, gr.update(value=file_path, visible=True)
    return detail_md, gr.update(value=None, visible=False)


# ---------------------------------------------------------------------------
# Gradio 界面构建
# ---------------------------------------------------------------------------

def create_app() -> gr.Blocks:
    """构建 Gradio Blocks 应用。"""

    with gr.Blocks(
        title="自主科学发现系统",
    ) as app:

        gr.Markdown(
            "# 自主科学发现系统\n"
            "基于多智能体协作与闭环反馈的 AI 驱动科研平台"
        )

        with gr.Tabs():
            # ==============================================================
            # Tab 1: 新发现
            # ==============================================================
            with gr.Tab("新发现"):
                with gr.Row():
                    # ---------- 左栏：输入控制区 ----------
                    with gr.Column(scale=1):
                        gr.Markdown("### 研究参数")

                        domain_input = gr.Dropdown(
                            label="研究领域",
                            choices=[
                                "catalysis",
                                "battery materials",
                                "drug design",
                                "nanotechnology",
                                "electronic information",
                                "polymer science",
                                "quantum computing",
                            ],
                            allow_custom_value=True,
                            value="catalysis",
                            info="选择或输入自定义研究领域",
                        )

                        problem_input = gr.Textbox(
                            label="研究问题（可选）",
                            placeholder="留空则由 AI 自动生成研究问题...\n例如: Design efficient CO2 conversion catalysts at low temperature",
                            lines=3,
                        )

                        max_iter_input = gr.Slider(
                            label="最大迭代次数",
                            minimum=1,
                            maximum=5,
                            value=3,
                            step=1,
                            info="发现循环的最大迭代轮次",
                        )

                        model_input = gr.Dropdown(
                            label="模型选择",
                            choices=["o3", "gpt-4", "gpt-4o", "deepseek-chat", "deepseek-reasoner"],
                            allow_custom_value=True,
                            value="deepseek-reasoner",
                            info="大语言模型名称",
                        )

                        with gr.Accordion("高级设置", open=False):
                            methods_input = gr.CheckboxGroup(
                                label="计算方法偏好",
                                choices=["DFT", "MD", "ML_potential", "data_analysis"],
                                value=[],
                                info="选择偏好的计算方法（可多选，留空则由 AI 决定）",
                            )

                            targets_input = gr.Textbox(
                                label="目标性质",
                                placeholder="逐行输入目标性质，如:\n能量密度\n循环稳定性\n离子导电率",
                                lines=3,
                            )

                            constraints_input = gr.Textbox(
                                label="约束条件",
                                placeholder="输入约束条件，如:\n温度 < 800K\n使用非贵金属催化剂",
                                lines=2,
                            )

                            interactive_input = gr.Checkbox(
                                label="交互模式",
                                value=False,
                                info="启用后在关键检查点自动批准（UI 模式）",
                            )

                            verbose_input = gr.Checkbox(
                                label="详细输出",
                                value=True,
                                info="显示智能体的详细输出信息",
                            )

                            user_id_input = gr.Textbox(
                                label="用户 ID",
                                value="experiment_system",
                                info="用于记忆操作的用户标识",
                            )

                        with gr.Row():
                            start_btn = gr.Button(
                                "启动发现", variant="primary", scale=2
                            )
                            stop_btn = gr.Button(
                                "停止", variant="stop", scale=1
                            )

                    # ---------- 右栏：实时输出区 ----------
                    with gr.Column(scale=2):
                        progress_md = gr.Markdown(
                            value=StateRenderer.progress_markdown("等待启动", 0, 0),
                            elem_classes=["progress-box"],
                        )

                        log_output = gr.Textbox(
                            label="实时日志",
                            lines=18,
                            max_lines=500,
                            interactive=False,
                            elem_classes=["log-box"],
                            autoscroll=True,
                        )

                        gr.Markdown("### 结构化结果")

                        hypotheses_df = gr.Dataframe(
                            headers=["ID", "陈述", "置信度", "代数", "状态"],
                            label="假设列表",
                            interactive=False,
                            wrap=True,
                        )

                        results_df = gr.Dataframe(
                            headers=["假设ID", "方法", "成功", "输出摘要"],
                            label="实验结果",
                            interactive=False,
                            wrap=True,
                        )

                        analysis_md = gr.Markdown(
                            value="*等待运行...*",
                            label="分析摘要",
                        )

                        download_file = gr.File(
                            label="下载发现结果",
                            visible=False,
                        )

                # 事件绑定 — 启动发现
                start_btn.click(
                    fn=run_discovery,
                    inputs=[
                        domain_input, problem_input, max_iter_input, model_input,
                        methods_input, targets_input, constraints_input,
                        interactive_input, verbose_input, user_id_input,
                    ],
                    outputs=[
                        progress_md, log_output, hypotheses_df,
                        results_df, analysis_md, download_file,
                    ],
                )

                # 事件绑定 — 停止
                stop_btn.click(fn=stop_discovery, outputs=[log_output])

            # ==============================================================
            # Tab 2: 历史记录
            # ==============================================================
            with gr.Tab("历史记录"):
                with gr.Row():
                    history_domain_filter = gr.Dropdown(
                        label="领域筛选",
                        choices=[
                            "", "catalysis", "battery materials", "drug design",
                            "nanotechnology", "electronic information",
                        ],
                        value="",
                        allow_custom_value=True,
                    )
                    history_keyword = gr.Textbox(
                        label="关键词搜索",
                        placeholder="输入关键词...",
                    )
                    history_refresh_btn = gr.Button("刷新", scale=0)

                with gr.Row():
                    with gr.Column(scale=1):
                        history_table = gr.Dataframe(
                            headers=[
                                "文件名", "会话ID", "领域",
                                "迭代数", "收敛", "新颖性", "时间",
                            ],
                            label="历史会话列表",
                            interactive=False,
                            value=_history_mgr.list_sessions(),
                        )

                    with gr.Column(scale=1):
                        history_detail = gr.Markdown(
                            value="*请选择左侧的一条记录查看详情*",
                        )
                        history_download = gr.File(
                            label="下载 JSON",
                            visible=False,
                        )

                # 历史记录事件绑定
                history_refresh_btn.click(
                    fn=load_history_list,
                    inputs=[history_keyword, history_domain_filter],
                    outputs=[history_table],
                )

                history_keyword.submit(
                    fn=load_history_list,
                    inputs=[history_keyword, history_domain_filter],
                    outputs=[history_table],
                )

                history_table.select(
                    fn=on_history_select,
                    inputs=[history_table],
                    outputs=[history_detail, history_download],
                )

    return app


# ---------------------------------------------------------------------------
# 主入口
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    app = create_app()
    app.queue(default_concurrency_limit=1)
    app.launch(
        server_name="127.0.0.1",
        server_port=7866,
        share=True,
        show_error=True,
        quiet=False,
        inbrowser=False
    )
