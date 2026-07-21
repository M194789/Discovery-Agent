"""
Windows GBK 编码全局修复模块

在 Windows 系统上，默认的控制台编码为 GBK (cp936)，无法处理 Unicode 科学符号
（如下标 ₂ U+2082、上标 ² U+00B2、特殊减号 − U+2212 等），导致 print() 崩溃。

本模块提供进程级别的彻底修复：
1. 设置 PYTHONUTF8=1 环境变量（影响后续子进程）
2. 重新配置 stdout/stderr 为 UTF-8 编码
3. 猴子补丁 builtins.print 作为终极保险

用法：在任何 Python 文件的最顶部（所有其他 import 之前）添加：
    import _utf8_env  # noqa: F401  # 强制 UTF-8 输出
"""

import builtins
import os
import sys

# ──────────────────────────────────────────────────
# 1. 设置环境变量（对当前进程的子进程生效）
# ──────────────────────────────────────────────────
os.environ["PYTHONUTF8"] = "1"
os.environ["PYTHONIOENCODING"] = "utf-8"

# ──────────────────────────────────────────────────
# 2. 重新配置当前进程的 stdout / stderr
# ──────────────────────────────────────────────────
def _reconfigure_streams():
    """强制将 stdout 和 stderr 切换为 UTF-8 编码。"""
    for stream_name in ("stdout", "stderr", "__stdout__", "__stderr__"):
        stream = getattr(sys, stream_name, None)
        if stream is None:
            continue
        encoding = getattr(stream, "encoding", None)
        if encoding and encoding.lower() not in ("utf-8", "utf8"):
            try:
                stream.reconfigure(encoding="utf-8", errors="replace")
            except Exception:
                pass


_reconfigure_streams()

# ──────────────────────────────────────────────────
# 3. 猴子补丁 builtins.print 作为终极保险
#    即使 reconfigure 失败，print 也不会因编码崩溃
# ──────────────────────────────────────────────────
_original_print = builtins.print


def _safe_print(*args, **kwargs):
    """编码安全的 print 替代品。"""
    try:
        _original_print(*args, **kwargs)
    except UnicodeEncodeError:
        # 将所有参数转为安全字符串
        safe_args = []
        for a in args:
            s = str(a)
            try:
                safe_args.append(
                    s.encode("utf-8", errors="replace").decode("ascii", errors="replace")
                )
            except Exception:
                safe_args.append("[encoding-error]")
        try:
            _original_print(*safe_args, **kwargs)
        except Exception:
            pass  # 彻底放弃输出，但绝不崩溃


builtins.print = _safe_print

# ──────────────────────────────────────────────────
# 确认修复已生效
# ──────────────────────────────────────────────────
_applied = True
