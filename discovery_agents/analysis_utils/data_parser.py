"""
数值数据解析器 - Numerical Data Parser

从实验执行输出的自由文本中提取结构化数值数据。
这是统计检验、可视化和因果推断三种技术的基础模块。

解析策略（多遍扫描）：
1. Key-Value 模式匹配: metric_name: 1.234, metric = 1.234
2. 表格检测: CSV-like 或空格对齐的表格
3. 标签数字提取: "accuracy is 0.95" 类型
4. 区段感知: 用标题行分组相关数字
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Tuple


@dataclass
class ParsedNumericalData:
    """从实验输出中解析出的结构化数值数据。"""
    metrics: Dict[str, float] = field(default_factory=dict)
    tables: List[Dict[str, List[float]]] = field(default_factory=list)
    time_series: List[Tuple[str, List[float]]] = field(default_factory=list)
    raw_numbers: List[float] = field(default_factory=list)
    parse_confidence: float = 0.0


# ---------------------------------------------------------------------------
# 正则表达式模式
# ---------------------------------------------------------------------------

# 匹配科学记数法和普通浮点数
_NUMBER_RE = re.compile(
    r'[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?'
)

# Key-Value 模式: "metric_name: 1.234" 或 "metric_name = 1.234"
# 支持带空格和下划线的名称
_KV_RE = re.compile(
    r'^\s*([A-Za-z][A-Za-z0-9_ /\-\(\)]{0,60}?)\s*[:=]\s*'
    r'([-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?)\s*'
    r'(%|[A-Za-z/\*\^]{0,15})?\s*$',
    re.MULTILINE,
)

# 自然语言数值: "accuracy is 0.95", "the result was 3.14"
_NL_VALUE_RE = re.compile(
    r'(?:is|was|were|equals?|of|about|approximately|roughly|around)\s+'
    r'([-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?)',
    re.IGNORECASE,
)

# 置信区间: "95% CI: (0.88, 0.96)" 或 "CI = [0.88, 0.96]"
_CI_RE = re.compile(
    r'(?:CI|confidence\s+interval)[^()\[\]]*'
    r'[\(\[]\s*([-+]?[\d.]+(?:[eE][-+]?\d+)?)\s*[,;]\s*'
    r'([-+]?[\d.]+(?:[eE][-+]?\d+)?)\s*[\)\]]',
    re.IGNORECASE,
)

# 区段标题行: "===RESULTS===", "## Summary", "--- Output ---"
_SECTION_RE = re.compile(
    r'^\s*(?:#{1,4}\s+.+|[=\-]{3,}.*[=\-]{3,}|[A-Z][A-Za-z\s]{2,30}:)\s*$',
    re.MULTILINE,
)

# 表格分隔符行
_TABLE_SEP_RE = re.compile(r'^\s*[\-\+\|=]{3,}\s*$')


def _safe_float(s: str) -> float | None:
    """安全地将字符串转为浮点数。"""
    try:
        return float(s)
    except (ValueError, OverflowError):
        return None


def _normalize_key(key: str) -> str:
    """规范化指标名称：去除多余空格，转小写下划线连接。"""
    key = key.strip()
    key = re.sub(r'[\s\-/]+', '_', key)
    key = re.sub(r'[()]+', '', key)
    key = key.lower().strip('_')
    return key


def _extract_key_value_pairs(text: str) -> Dict[str, float]:
    """
    第1遍：提取 key: value 和 key = value 模式的数值。
    """
    metrics: Dict[str, float] = {}
    for match in _KV_RE.finditer(text):
        key = _normalize_key(match.group(1))
        val = _safe_float(match.group(2))
        if key and val is not None and len(key) > 1:
            # 如果带百分号，转换为 0-1 范围
            unit = match.group(3) or ""
            if unit == "%" and val > 1:
                metrics[key] = val / 100.0
            else:
                metrics[key] = val
    return metrics


def _extract_nl_values(text: str, existing_keys: set) -> Dict[str, float]:
    """
    第2遍：从自然语言描述中提取数值。
    寻找 "the X is/was Y" 模式。
    """
    metrics: Dict[str, float] = {}
    # 扩展模式：尝试捕获变量名
    pattern = re.compile(
        r'(?:the\s+)?([A-Za-z][A-Za-z0-9_ ]{1,40}?)\s+'
        r'(?:is|was|were|equals?)\s+'
        r'([-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?)',
        re.IGNORECASE,
    )
    for match in pattern.finditer(text):
        key = _normalize_key(match.group(1))
        val = _safe_float(match.group(2))
        if key and val is not None and key not in existing_keys and len(key) > 1:
            metrics[key] = val
    return metrics


def _extract_tables(text: str) -> List[Dict[str, List[float]]]:
    """
    第3遍：检测并解析表格结构。
    支持 CSV-like 格式和空格/制表符对齐的表格。
    """
    tables: List[Dict[str, List[float]]] = []
    lines = text.split('\n')

    i = 0
    while i < len(lines):
        line = lines[i].strip()

        # 检测 CSV 头部行（至少两个逗号分隔的列名）
        if ',' in line and not _NUMBER_RE.match(line.split(',')[0].strip()):
            headers_raw = [h.strip() for h in line.split(',')]
            # 验证至少有2个非空头部
            headers = [_normalize_key(h) for h in headers_raw if h.strip()]
            if len(headers) >= 2:
                table_data: Dict[str, List[float]] = {h: [] for h in headers}
                j = i + 1
                while j < len(lines):
                    row_line = lines[j].strip()
                    if not row_line or _TABLE_SEP_RE.match(row_line):
                        j += 1
                        continue
                    cells = [c.strip() for c in row_line.split(',')]
                    if len(cells) != len(headers):
                        break
                    row_vals = [_safe_float(c) for c in cells]
                    if all(v is not None for v in row_vals):
                        for h, v in zip(headers, row_vals):
                            table_data[h].append(v)  # type: ignore[arg-type]
                        j += 1
                    else:
                        break
                # 只保留有数据的表格
                if any(len(v) > 0 for v in table_data.values()):
                    tables.append(table_data)
                    i = j
                    continue

        # 检测制表符/空格分隔的表格
        if '\t' in line or re.search(r'\s{2,}', line):
            parts = re.split(r'\t+|\s{2,}', line)
            headers = [_normalize_key(p) for p in parts if p.strip()]
            if len(headers) >= 2 and all(
                not _NUMBER_RE.fullmatch(h) for h in headers
            ):
                table_data = {h: [] for h in headers}
                j = i + 1
                while j < len(lines):
                    row_line = lines[j].strip()
                    if not row_line or _TABLE_SEP_RE.match(row_line):
                        j += 1
                        continue
                    cells = re.split(r'\t+|\s{2,}', row_line.strip())
                    cells = [c.strip() for c in cells if c.strip()]
                    if len(cells) != len(headers):
                        break
                    row_vals = [_safe_float(c) for c in cells]
                    if all(v is not None for v in row_vals):
                        for h, v in zip(headers, row_vals):
                            table_data[h].append(v)  # type: ignore[arg-type]
                        j += 1
                    else:
                        break
                if any(len(v) > 0 for v in table_data.values()):
                    tables.append(table_data)
                    i = j
                    continue

        i += 1

    return tables


def _extract_time_series(text: str) -> List[Tuple[str, List[float]]]:
    """
    第4遍：检测有序数值序列（如训练历史、迭代数据）。
    寻找 "epoch/step/iteration X: value" 类型的模式。
    """
    series: List[Tuple[str, List[float]]] = []

    # 寻找连续的带标签数值序列
    # 模式: "Epoch 1: 0.85", "Epoch 2: 0.90", ...
    seq_pattern = re.compile(
        r'(?:epoch|step|iteration|iter|round|trial)\s*'
        r'(?:#?\s*)?\d+\s*[:=,\-]\s*'
        r'([-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?)',
        re.IGNORECASE,
    )
    seq_values = [_safe_float(m.group(1)) for m in seq_pattern.finditer(text)]
    seq_values = [v for v in seq_values if v is not None]
    if len(seq_values) >= 3:
        series.append(("training_sequence", seq_values))

    # 寻找方括号中的数组: [1.0, 2.0, 3.0, ...]
    array_pattern = re.compile(
        r'\[(\s*[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?'
        r'(?:\s*,\s*[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?){2,}\s*)\]'
    )
    for match in array_pattern.finditer(text):
        nums = [_safe_float(n.strip()) for n in match.group(1).split(',')]
        nums = [n for n in nums if n is not None]
        if len(nums) >= 3:
            series.append(("array_data", nums))

    return series


def _extract_all_numbers(text: str) -> List[float]:
    """
    最后备用：提取文本中所有数值。
    过滤掉看起来像日期、版本号等的数字。
    """
    numbers: List[float] = []
    for match in _NUMBER_RE.finditer(text):
        val = _safe_float(match.group())
        if val is not None:
            # 过滤明显的非数据数字（行号、版本号等）
            s = match.group()
            # 跳过过长的纯整数（可能是ID或时间戳）
            if '.' not in s and 'e' not in s.lower() and len(s) > 8:
                continue
            numbers.append(val)
    return numbers


def _compute_parse_confidence(parsed: ParsedNumericalData) -> float:
    """
    计算解析置信度。
    基于提取到的数据量和类型给出 0.0-1.0 的评分。
    """
    score = 0.0

    # Key-Value 指标贡献最大
    n_metrics = len(parsed.metrics)
    if n_metrics > 0:
        score += min(0.4, n_metrics * 0.1)

    # 表格数据
    if parsed.tables:
        total_rows = sum(
            max(len(v) for v in t.values()) if t else 0
            for t in parsed.tables
        )
        score += min(0.3, total_rows * 0.03)

    # 时序数据
    if parsed.time_series:
        score += 0.15

    # 原始数字（低贡献）
    if parsed.raw_numbers and not parsed.metrics and not parsed.tables:
        score += min(0.15, len(parsed.raw_numbers) * 0.01)

    return min(1.0, score)


def parse_numerical_data(execution_output: str) -> ParsedNumericalData:
    """
    从实验执行输出中解析结构化数值数据。

    参数:
        execution_output: ExperimentResult.execution_output 的原始文本

    返回:
        ParsedNumericalData: 包含提取的指标、表格、时序和原始数字
    """
    if not execution_output or not execution_output.strip():
        return ParsedNumericalData()

    parsed = ParsedNumericalData()

    # 第1遍：Key-Value 对
    parsed.metrics = _extract_key_value_pairs(execution_output)

    # 第2遍：自然语言数值
    nl_metrics = _extract_nl_values(
        execution_output, set(parsed.metrics.keys())
    )
    parsed.metrics.update(nl_metrics)

    # 提取置信区间到指标中
    for ci_match in _CI_RE.finditer(execution_output):
        lo = _safe_float(ci_match.group(1))
        hi = _safe_float(ci_match.group(2))
        if lo is not None and hi is not None:
            parsed.metrics["ci_lower"] = lo
            parsed.metrics["ci_upper"] = hi

    # 第3遍：表格数据
    parsed.tables = _extract_tables(execution_output)

    # 第4遍：时序数据
    parsed.time_series = _extract_time_series(execution_output)

    # 备用：所有原始数字
    parsed.raw_numbers = _extract_all_numbers(execution_output)

    # 计算解析置信度
    parsed.parse_confidence = _compute_parse_confidence(parsed)

    return parsed
