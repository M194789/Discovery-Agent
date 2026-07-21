"""
自主科学发现系统 — Demo 演示 Gradio Web 交互界面。

启动方式:
    python Demo/demo_app.py

浏览器访问 http://localhost:7864

使用预设 mock 数据运行，无需 API 密钥。
"""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path
from typing import Optional

# 确保项目根目录在 sys.path 中
_demo_dir = Path(__file__).resolve().parent       # Demo/
_project_dir = _demo_dir.parent                    # Discovery-Agent/
if str(_project_dir) not in sys.path:
    sys.path.insert(0, str(_project_dir))

import gradio as gr

from ui_bridge import LogEmitter, HistoryManager, StateRenderer
from Demo.demo_runner import run_demo, save_session

# 历史管理器 — 指向 Demo/ 目录
_history_mgr = HistoryManager(_demo_dir)

# 当前运行任务引用（用于停止功能）
_current_task: Optional[asyncio.Task] = None


# ---------------------------------------------------------------------------
# 领域 → 默认问题映射
# ---------------------------------------------------------------------------

_DOMAIN_DEFAULTS = {
    "催化科学": "设计高效的二氧化碳转化催化剂",
    "电池材料": "开发高能量密度长寿命固态锂电池电解质",
    "电子信息与通信工程": "利用深度学习优化无线通信系统的频谱效率和误码率性能",
}

_PHASE_DISPLAY = {
    "init": "初始化",
    "phase_a": "假设生成与辩论",
    "phase_b": "实验规划",
    "phase_c": "实验执行",
    "phase_d": "结果分析与演化",
    "phase_e": "记忆与技能保存",
    "complete": "已完成",
}


# ---------------------------------------------------------------------------
# 辅助函数
# ---------------------------------------------------------------------------

def update_problem_default(domain: str):
    """当用户切换领域时，更新问题输入的默认值。"""
    default = _DOMAIN_DEFAULTS.get(domain, "")
    placeholder = default if default else "输入您的研究问题..."
    return gr.update(value=default, placeholder=placeholder)


# ---------------------------------------------------------------------------
# 核心运行函数 (async generator)
# outputs: [progress_md, log_output, hypotheses_df, results_df, analysis_md, download_file]
# ---------------------------------------------------------------------------

async def run_demo_discovery(
    domain: str,
    problem: str,
    max_iter: int,
):
    """主演示任务的 async generator，流式更新 Gradio 组件。"""
    global _current_task

    _no_update = gr.update()

    if not domain or not domain.strip():
        yield (
            "**错误**: 请选择或输入研究领域",
            "错误：研究领域不能为空。",
            _no_update,
            _no_update,
            _no_update,
            _no_update,
        )
        return

    domain = domain.strip()
    problem = problem.strip() if problem else _DOMAIN_DEFAULTS.get(domain, "自主发现研究问题")
    max_iter = int(max_iter)

    emitter = LogEmitter()
    emitter.max_iterations = max_iter

    # yield 初始状态
    yield (
        StateRenderer.progress_markdown("正在初始化...", 0, max_iter),
        "正在启动演示流程...\n",
        [],
        [],
        "*等待运行...*",
        gr.update(value=None, visible=False),
    )

    try:
        async for snapshot in run_demo(domain, problem, max_iter, emitter.emit):
            phase = snapshot["phase"]
            iteration = snapshot["iteration"]
            phase_display = _PHASE_DISPLAY.get(phase, phase)

            # 构建假设表
            hyp_rows = StateRenderer.hypotheses_to_dataframe(snapshot["hypotheses"])
            # 构建结果表
            res_rows = StateRenderer.results_to_dataframe(
                snapshot["results"], snapshot["plans"]
            )
            # 构建分析 markdown
            ana_md = StateRenderer.analysis_to_markdown(snapshot["analyses"])

            # 最终阶段：保存 JSON 并提供下载
            download_update = _no_update
            if phase == "complete" and snapshot.get("session_data"):
                json_path = save_session(snapshot["session_data"], _demo_dir)
                emitter.emit(f"\n状态已保存到：{json_path}")
                emitter.emit("\n演示完成!")
                download_update = gr.update(value=json_path, visible=True)

            progress_iter = iteration if phase != "complete" else max_iter
            yield (
                StateRenderer.progress_markdown(phase_display, progress_iter, max_iter),
                emitter.get_full_log(),
                hyp_rows if hyp_rows else _no_update,
                res_rows if res_rows else _no_update,
                ana_md if snapshot["analyses"] else _no_update,
                download_update,
            )

    except asyncio.CancelledError:
        yield (
            "**已停止**",
            emitter.get_full_log() + "\n--- 用户已停止演示 ---\n",
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


def stop_demo():
    """停止当前运行的演示任务。"""
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
        title="自主科学发现系统 - 演示模式",
    ) as app:

        gr.Markdown(
            "# 自主科学发现系统 - 演示模式\n"
            "基于多智能体协作与闭环反馈的 AI 驱动科研平台\n\n"
            "> 演示模式使用预设数据，**无需 API 密钥**即可体验完整发现流程。"
        )

        with gr.Tabs():
            # ==============================================================
            # Tab 1: 运行演示
            # ==============================================================
            with gr.Tab("运行演示"):
                with gr.Row():
                    # ---------- 左栏：输入控制区 ----------
                    with gr.Column(scale=1):
                        gr.Markdown("### 研究参数")

                        domain_input = gr.Dropdown(
                            label="研究领域",
                            choices=[
                                "催化科学",
                                "电池材料",
                                "电子信息与通信工程",
                            ],
                            allow_custom_value=True,
                            value="催化科学",
                            info="选择预设领域或输入自定义领域（自定义领域将使用催化科学的演示数据）",
                        )

                        problem_input = gr.Textbox(
                            label="研究问题",
                            value="设计高效的二氧化碳转化催化剂",
                            placeholder="输入您的研究问题...",
                            lines=3,
                        )

                        max_iter_input = gr.Slider(
                            label="最大迭代次数",
                            minimum=1,
                            maximum=5,
                            value=2,
                            step=1,
                            info="发现循环的最大迭代轮次",
                        )

                        with gr.Row():
                            start_btn = gr.Button(
                                "启动演示", variant="primary", scale=2
                            )
                            stop_btn = gr.Button(
                                "停止", variant="stop", scale=1
                            )

                        gr.Markdown(
                            "---\n"
                            "**提示**: 演示模式使用预设的 mock 数据模拟完整的科学发现流程，"
                            "包括假设生成、实验规划、执行、分析和进化等阶段。"
                        )

                    # ---------- 右栏：实时输出区 ----------
                    with gr.Column(scale=2):
                        progress_md = gr.Markdown(
                            value=(
                                "**当前阶段**: 等待启动\n\n"
                                "**迭代进度**: 0 / 0\n\n"
                                "`[--------------------]` 0%"
                            ),
                        )

                        log_output = gr.Textbox(
                            label="实时日志",
                            lines=18,
                            max_lines=50,
                            interactive=False,
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

                # 事件绑定 — 领域切换更新默认问题
                domain_input.change(
                    fn=update_problem_default,
                    inputs=[domain_input],
                    outputs=[problem_input],
                )

                # 事件绑定 — 启动演示
                start_btn.click(
                    fn=run_demo_discovery,
                    inputs=[domain_input, problem_input, max_iter_input],
                    outputs=[
                        progress_md, log_output, hypotheses_df,
                        results_df, analysis_md, download_file,
                    ],
                )

                # 事件绑定 — 停止
                stop_btn.click(fn=stop_demo, outputs=[log_output])

            # ==============================================================
            # Tab 2: 演示历史
            # ==============================================================
            with gr.Tab("演示历史"):
                with gr.Row():
                    history_domain_filter = gr.Dropdown(
                        label="领域筛选",
                        choices=[
                            "", "催化科学", "电池材料",
                            "电子信息与通信工程",
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
        server_port=7864,
        share=False,
        show_error=True,
        quiet=True,
        inbrowser=False,
    )
