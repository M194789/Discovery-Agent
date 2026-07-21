"""
RL State Persistence - RL状态持久化

支持将 Bandit 学习状态保存到 JSON 文件，
实现跨迭代和跨会话的知识积累。
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Optional

from .models import BanditState, ArmStatistics, RewardSignal
from .bandit import BanditAlgorithm


# 默认持久化目录
DEFAULT_PERSISTENCE_DIR = Path(__file__).resolve().parent.parent / "saved_code"


def get_state_path(session_id: str, base_dir: Optional[str] = None) -> Path:
    """获取会话特定的状态文件路径。"""
    directory = Path(base_dir) if base_dir else DEFAULT_PERSISTENCE_DIR
    return directory / f"rl_state_{session_id}.json"


def get_global_state_path(base_dir: Optional[str] = None) -> Path:
    """获取全局状态文件路径（跨会话共享）。"""
    directory = Path(base_dir) if base_dir else DEFAULT_PERSISTENCE_DIR
    return directory / "rl_state_global.json"


def save_state(
    bandit: BanditAlgorithm,
    reward_history: list,
    session_id: str,
    base_dir: Optional[str] = None,
) -> str:
    """
    将 Bandit 状态保存到 JSON 文件。

    同时更新会话特定文件和全局文件。

    参数:
        bandit: Bandit 算法实例
        reward_history: 历史奖励记录
        session_id: 会话 ID
        base_dir: 存储目录（可选）

    返回:
        保存的文件路径
    """
    state = bandit.to_state()
    # 将 RewardSignal 对象序列化
    state.history = [
        r if isinstance(r, RewardSignal) else RewardSignal(**r)
        for r in reward_history
    ]

    # 序列化为字典
    data = state.model_dump()
    data["last_updated"] = datetime.now().isoformat()
    data["session_id"] = session_id

    # 确保目录存在
    directory = Path(base_dir) if base_dir else DEFAULT_PERSISTENCE_DIR
    os.makedirs(str(directory), exist_ok=True)

    # 保存会话特定文件
    session_path = get_state_path(session_id, base_dir)
    with open(str(session_path), "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    # 同步更新全局文件（跨会话累积学习）
    global_path = get_global_state_path(base_dir)
    _merge_to_global(data, str(global_path))

    return str(session_path)


def load_state(
    session_id: str,
    base_dir: Optional[str] = None,
) -> Optional[BanditState]:
    """
    加载 Bandit 状态。

    优先加载会话特定文件，如果不存在则回退到全局文件。

    参数:
        session_id: 会话 ID
        base_dir: 存储目录（可选）

    返回:
        BanditState 对象，如果无可用状态则返回 None
    """
    # 尝试加载会话特定文件
    session_path = get_state_path(session_id, base_dir)
    if os.path.exists(str(session_path)):
        return _load_from_file(str(session_path))

    # 回退到全局文件（利用先前会话的学习）
    global_path = get_global_state_path(base_dir)
    if os.path.exists(str(global_path)):
        return _load_from_file(str(global_path))

    return None


def _load_from_file(file_path: str) -> Optional[BanditState]:
    """从 JSON 文件加载 BanditState。"""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # 重建 arm_stats
        arm_stats = {}
        for name, stats_data in data.get("arm_stats", {}).items():
            arm_stats[name] = ArmStatistics(**stats_data)

        # 重建 history
        history = []
        for r in data.get("history", []):
            history.append(RewardSignal(**r))

        return BanditState(
            algorithm=data.get("algorithm", "thompson_sampling"),
            arm_stats=arm_stats,
            history=history,
            total_pulls=data.get("total_pulls", 0),
            exploration_constant=data.get("exploration_constant", 1.414),
        )
    except (json.JSONDecodeError, TypeError, KeyError) as e:
        print(f"  Warning: Failed to load RL state from {file_path}: {e}")
        return None


def _merge_to_global(session_data: dict, global_path: str) -> None:
    """
    将当前会话的统计数据合并到全局文件。

    合并策略：取各臂的累积统计（加法合并）。
    """
    try:
        # 加载现有全局数据
        global_data = {}
        if os.path.exists(global_path):
            with open(global_path, "r", encoding="utf-8") as f:
                global_data = json.load(f)

        # 如果全局数据为空，直接使用当前会话数据
        if not global_data.get("arm_stats"):
            global_data = session_data.copy()
            global_data.pop("session_id", None)
        else:
            # 合并臂统计（取最新的会话数据，因为它包含了完整的累积）
            # 对于全局文件，我们保留拉取次数最多的版本
            session_arms = session_data.get("arm_stats", {})
            global_arms = global_data.get("arm_stats", {})

            for name, s_stats in session_arms.items():
                if name not in global_arms:
                    global_arms[name] = s_stats
                else:
                    g_stats = global_arms[name]
                    # 如果会话的数据更丰富（更多拉取次数），用它更新全局
                    s_pulls = s_stats.get("pull_count", 0) if isinstance(s_stats, dict) else 0
                    g_pulls = g_stats.get("pull_count", 0) if isinstance(g_stats, dict) else 0
                    if s_pulls > g_pulls:
                        global_arms[name] = s_stats

            global_data["arm_stats"] = global_arms
            global_data["total_pulls"] = max(
                global_data.get("total_pulls", 0),
                session_data.get("total_pulls", 0)
            )

        global_data["last_updated"] = datetime.now().isoformat()

        with open(global_path, "w", encoding="utf-8") as f:
            json.dump(global_data, f, indent=2, ensure_ascii=False)

    except Exception:
        pass  # 全局合并失败不影响主流程
