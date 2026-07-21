"""
Gradio UI 桥接层。

提供日志捕获、状态渲染、历史记录管理等功能，
将 DiscoveryOrchestrator 的输出与 Gradio 界面解耦。
"""

from __future__ import annotations

import json
import os
import queue
import re
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import List, Optional


# ---------------------------------------------------------------------------
# 阶段映射表
# ---------------------------------------------------------------------------

PHASE_MAP = {
    "[INIT]": "初始化",
    "[PHASE A]": "假设生成与辩论",
    "[PHASE B]": "实验规划",
    "[PHASE C]": "实验执行",
    "[PHASE D]": "结果分析与演化",
    "[PHASE E]": "记忆与技能保存",
}

PHASE_ORDER = list(PHASE_MAP.keys())


# ---------------------------------------------------------------------------
# LogEmitter — 日志发射器
# ---------------------------------------------------------------------------

class LogEmitter:
    """接收编排器日志，通过线程安全队列转发给 Gradio 流式输出。"""

    # 子阶段顺序：INIT + 5个 PHASE，用于计算细粒度进度
    _SUB_PHASES = ["[INIT]", "[PHASE A]", "[PHASE B]", "[PHASE C]", "[PHASE D]", "[PHASE E]"]

    def __init__(self):
        self.queue: queue.Queue = queue.Queue()
        self.full_log: str = ""
        self.phase: str = "等待启动"
        self.phase_key: str = ""
        self.phase_detail: str = ""  # 当前阶段的详细描述
        self.iteration: int = 0
        self.max_iterations: int = 0
        self.sub_phase_index: int = -1  # 当前子阶段索引 (0-5)
        self.start_time: float = time.time()  # 启动时间戳
        self._lock = threading.Lock()

    def emit(self, msg: str):
        """log_callback 回调，由 orchestrator._log()/_vlog() 调用。"""
        with self._lock:
            self.full_log += msg + "\n"
            self._detect_phase(msg)
        self.queue.put(msg)

    def _detect_phase(self, msg: str):
        """从日志消息解析当前阶段、迭代号和子阶段索引。"""
        for key, name in PHASE_MAP.items():
            if key in msg:
                self.phase = name
                self.phase_key = key
                # 提取详细信息（去掉 phase tag 后的文本）
                detail = msg.split(key, 1)[-1].strip().rstrip("...")
                if detail:
                    self.phase_detail = detail
                break

        # 更新子阶段索引
        for i, sp in enumerate(self._SUB_PHASES):
            if sp in msg:
                self.sub_phase_index = i
                break

        # 匹配 "迭代 X/Y" 或 "Iteration X/Y" 格式
        m = re.search(r"(?:迭代|Iteration)\s+(\d+)/(\d+)", msg)
        if m:
            self.iteration = int(m.group(1))
            self.max_iterations = int(m.group(2))

    def get_elapsed(self) -> float:
        """返回已运行秒数。"""
        return time.time() - self.start_time

    def get_fine_progress(self) -> float:
        """计算细粒度进度百分比 (0.0 ~ 1.0)，基于迭代 + 子阶段。"""
        if self.max_iterations <= 0:
            return 0.0
        total_sub = len(self._SUB_PHASES)  # 每轮6个子阶段
        # INIT 阶段属于第0轮前的准备
        if self.iteration == 0:
            # INIT 阶段：按子阶段在第一轮前占一小部分
            sp = max(0, self.sub_phase_index)
            return (sp / total_sub) * (1 / (self.max_iterations + 0.5)) * 0.5
        # 正式迭代中的进度
        completed_iters = self.iteration - 1
        sp = max(0, self.sub_phase_index)
        iter_progress = completed_iters + (sp / total_sub)
        return min(iter_progress / self.max_iterations, 1.0)

    def get_full_log(self) -> str:
        """获取完整的累积日志文本（线程安全）。"""
        with self._lock:
            return self.full_log


# ---------------------------------------------------------------------------
# TeeStdout — stdout 重定向上下文管理器
# ---------------------------------------------------------------------------

class TeeStdout:
    """捕获 agents 中散落的 print() 输出，同时保留原始控制台输出。"""

    def __init__(self, emitter: LogEmitter):
        self.emitter = emitter
        self._original_stdout = None
        self._lock = threading.Lock()

    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = self
        return self

    def __exit__(self, *args):
        sys.stdout = self._original_stdout

    def write(self, text: str):
        if self._original_stdout:
            self._original_stdout.write(text)
        # 只转发非空非纯换行的内容
        stripped = text.strip()
        if stripped:
            self.emitter.emit(stripped)

    def flush(self):
        if self._original_stdout:
            self._original_stdout.flush()


# ---------------------------------------------------------------------------
# StateRenderer — 状态格式化
# ---------------------------------------------------------------------------

class StateRenderer:
    """将发现状态数据转为 Gradio 可展示的格式。"""

    @staticmethod
    def hypotheses_to_dataframe(hypotheses: list) -> list[list]:
        """假设列表 -> [[ID, 陈述, 置信度, 代数, 状态], ...]"""
        rows = []
        for h in hypotheses:
            if isinstance(h, dict):
                rows.append([
                    h.get("id", ""),
                    h.get("statement", "")[:100],
                    f"{h.get('confidence', 0):.2f}",
                    str(h.get("generation", 1)),
                    h.get("status", "active"),
                ])
            else:
                rows.append([
                    getattr(h, "id", ""),
                    getattr(h, "statement", "")[:100],
                    f"{getattr(h, 'confidence', 0):.2f}",
                    str(getattr(h, "generation", 1)),
                    getattr(h, "status", "active"),
                ])
        return rows

    @staticmethod
    def results_to_dataframe(results: list, plans: Optional[list] = None) -> list[list]:
        """实验结果 -> [[假设ID, 方法, 成功, 输出摘要], ...]"""
        plan_map = {}
        if plans:
            for p in plans:
                hid = p.get("hypothesis_id", "") if isinstance(p, dict) else getattr(p, "hypothesis_id", "")
                method = p.get("method", "") if isinstance(p, dict) else getattr(p, "method", "")
                plan_map[hid] = method

        rows = []
        for r in results:
            if isinstance(r, dict):
                hid = r.get("hypothesis_id", "")
                success = r.get("success", False)
                output = r.get("execution_output", "")[:120]
            else:
                hid = getattr(r, "hypothesis_id", "")
                success = getattr(r, "success", False)
                output = getattr(r, "execution_output", "")[:120]
            method = plan_map.get(hid, "")
            rows.append([hid, method, "Yes" if success else "No", output])
        return rows

    @staticmethod
    def analysis_to_markdown(analyses: list) -> str:
        """分析结果 -> Markdown 格式文本"""
        if not analyses:
            return "*暂无分析结果*"
        lines = ["### 分析结果\n"]
        for a in analyses:
            if isinstance(a, dict):
                hid = a.get("hypothesis_id", "")
                supports = a.get("supports_hypothesis", False)
                confidence = a.get("confidence", 0)
                findings = a.get("key_findings", [])
                explanation = a.get("explanation", "")
            else:
                hid = getattr(a, "hypothesis_id", "")
                supports = getattr(a, "supports_hypothesis", False)
                confidence = getattr(a, "confidence", 0)
                findings = getattr(a, "key_findings", [])
                explanation = getattr(a, "explanation", "")

            verdict = "**支持**" if supports else "**反驳**"
            lines.append(f"- **{hid}**: {verdict} (置信度: {confidence:.2f})")
            if findings:
                for f in findings[:3]:
                    lines.append(f"  - {f}")
            if explanation:
                lines.append(f"  - _{explanation[:200]}_")
            lines.append("")
        return "\n".join(lines)

    @staticmethod
    def progress_markdown(
        phase: str,
        iteration: int,
        max_iter: int,
        fine_progress: float = -1.0,
        elapsed: float = 0.0,
        phase_detail: str = "",
    ) -> str:
        """生成进度展示 Markdown（支持细粒度子阶段进度）。"""
        # 细粒度进度优先，否则按迭代粗略计算
        if fine_progress >= 0:
            pct = int(fine_progress * 100)
        elif max_iter <= 0:
            pct = 0
        else:
            pct = int((iteration / max_iter) * 100)
        pct = min(pct, 100)

        filled = int(20 * pct / 100)
        bar = "█" * filled + "░" * (20 - filled)

        # 格式化耗时
        if elapsed > 0:
            m, s = divmod(int(elapsed), 60)
            time_str = f"{m}分{s}秒" if m else f"{s}秒"
        else:
            time_str = "--"

        lines = [
            f"**当前阶段**: {phase}",
        ]
        if phase_detail:
            lines.append(f"  *{phase_detail}*")
        lines.append(f"")
        lines.append(f"**迭代进度**: {iteration} / {max_iter}  |  **已耗时**: {time_str}")
        lines.append(f"")
        lines.append(f"`{bar}` **{pct}%**")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# HistoryManager — 历史记录管理
# ---------------------------------------------------------------------------

class HistoryManager:
    """管理 saved_code/ 目录中的 JSON 发现历史。"""

    def __init__(self, saved_dir: Path):
        self.saved_dir = saved_dir

    def list_sessions(self) -> list[list]:
        """扫描所有 JSON 文件，返回摘要列表供 Dataframe 展示。

        返回: [[文件名, 会话ID, 领域, 迭代数, 收敛, 新颖性, 修改时间], ...]
        """
        rows = []
        if not self.saved_dir.exists():
            return rows

        for fp in sorted(self.saved_dir.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True):
            try:
                with open(fp, "r", encoding="utf-8") as f:
                    data = json.load(f)
            except Exception:
                continue

            session_id = data.get("session_id", fp.stem)
            domain = data.get("domain", "")
            iterations = data.get("iterations", [])
            iter_count = len(iterations) if isinstance(iterations, list) else data.get("current_iteration", 0)
            converged = data.get("converged", False)
            novelty = data.get("overall_novelty_score", 0.0)
            mtime = datetime.fromtimestamp(fp.stat().st_mtime).strftime("%Y-%m-%d %H:%M")

            rows.append([
                fp.name,
                str(session_id)[:8],
                domain,
                str(iter_count),
                "Yes" if converged else "No",
                f"{novelty:.2f}" if isinstance(novelty, (int, float)) else str(novelty),
                mtime,
            ])
        return rows

    def load_session(self, filename: str) -> Optional[dict]:
        """加载完整 JSON 数据。"""
        fp = self.saved_dir / filename
        if not fp.exists():
            return None
        try:
            with open(fp, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return None

    def get_file_path(self, filename: str) -> Optional[str]:
        """获取 JSON 文件的完整路径（供下载使用）。"""
        fp = self.saved_dir / filename
        return str(fp) if fp.exists() else None

    def render_detail_markdown(self, data: dict) -> str:
        """将 JSON 数据渲染为 Markdown 格式的详情展示。"""
        lines = []

        # 概览
        lines.append("## 会话概览\n")
        lines.append(f"- **会话 ID**: {data.get('session_id', 'N/A')}")
        lines.append(f"- **研究领域**: {data.get('domain', 'N/A')}")

        problem = data.get("problem")
        if isinstance(problem, dict):
            lines.append(f"- **研究问题**: {problem.get('problem_statement', 'N/A')}")
            novelty_obj = problem.get("novelty", {})
            if isinstance(novelty_obj, dict):
                lines.append(f"- **问题新颖性**: {novelty_obj.get('score', 'N/A')}")
            targets = problem.get("target_properties", [])
            if targets:
                lines.append(f"- **目标性质**: {', '.join(str(t) for t in targets)}")
        elif isinstance(problem, str):
            lines.append(f"- **研究问题**: {problem}")

        lines.append(f"- **模型**: {data.get('model_name', 'N/A')}")
        lines.append(f"- **收敛**: {'Yes' if data.get('converged') else 'No'}")
        lines.append(f"- **总新颖性**: {data.get('overall_novelty_score', 'N/A')}")
        lines.append("")

        # 迭代详情
        iterations = data.get("iterations", [])
        if not iterations:
            lines.append("*暂无迭代记录*")
            return "\n".join(lines)

        for it in iterations:
            iter_num = it.get("iteration", "?")
            lines.append(f"---\n### 迭代 {iter_num}\n")

            # 假设
            hypotheses = it.get("hypotheses", [])
            if hypotheses:
                lines.append("**假设:**\n")
                for h in hypotheses:
                    hid = h.get("id", "?")
                    stmt = h.get("statement", "")[:120]
                    status = h.get("status", "active")
                    conf = h.get("confidence", 0)
                    lines.append(f"- `{hid}` ({status}, {conf:.2f}): {stmt}")
                lines.append("")

            # 实验计划
            plans = it.get("plans", [])
            if plans:
                lines.append("**实验计划:**\n")
                for p in plans:
                    hid = p.get("hypothesis_id", "?")
                    method = p.get("method", "?")
                    complexity = p.get("estimated_complexity", "?")
                    lines.append(f"- `{hid}`: {method} (复杂度: {complexity})")
                lines.append("")

            # 实验结果
            results = it.get("results", [])
            if results:
                lines.append("**实验结果:**\n")
                for r in results:
                    hid = r.get("hypothesis_id", "?")
                    success = r.get("success", False)
                    icon = "OK" if success else "FAIL"
                    output = r.get("execution_output", "")[:150]
                    lines.append(f"- `{hid}`: [{icon}] {output}")
                lines.append("")

            # 分析
            analysis = it.get("analysis", [])
            if analysis:
                lines.append("**分析:**\n")
                for a in analysis:
                    hid = a.get("hypothesis_id", "?")
                    supports = a.get("supports_hypothesis", False)
                    verdict = "支持" if supports else "反驳"
                    conf = a.get("confidence", 0)
                    lines.append(f"- `{hid}`: {verdict} (置信度: {conf:.2f})")
                    for finding in a.get("key_findings", [])[:2]:
                        lines.append(f"  - {finding}")
                lines.append("")

            # 演化
            evo = it.get("evolution")
            if evo:
                new_count = len(evo.get("new_hypotheses", []))
                elim_count = len(evo.get("eliminated_ids", []))
                converged = evo.get("converged", False)
                lines.append(f"**演化**: 新增 {new_count} 个假设, 淘汰 {elim_count} 个, 收敛: {'Yes' if converged else 'No'}")
                reasoning = evo.get("evolution_reasoning", "")
                if reasoning:
                    lines.append(f"\n> {reasoning[:300]}")
                lines.append("")

        return "\n".join(lines)

    def search_sessions(self, keyword: str = "", domain_filter: str = "") -> list[list]:
        """按关键词和领域筛选会话列表。"""
        all_sessions = self.list_sessions()
        filtered = []
        for row in all_sessions:
            # row: [文件名, 会话ID, 领域, 迭代数, 收敛, 新颖性, 时间]
            domain = row[2].lower()
            session_text = " ".join(str(c) for c in row).lower()

            if domain_filter and domain_filter.lower() not in domain:
                continue
            if keyword and keyword.lower() not in session_text:
                continue
            filtered.append(row)
        return filtered
