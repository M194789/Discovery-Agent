"""
Literature Pipeline - 文献挖掘预处理流水线

在问题生成器智能体调用前，程序化执行：
1. 多路 Tavily 搜索（最新文献和预印本）
2. 轻量知识图谱构建（实体提取与关系推断）
3. 因果推理（因果链提取与置信度评估）
4. 趋势分析（时间模式与方向检测）
5. 研究空白识别（缺失边、低置信因果、显式空白语言）
6. 向量记忆去重（避免重复探索）

所有结果格式化为上下文文本注入智能体提示词。
"""

from __future__ import annotations

import _utf8_env  # noqa: F401  # 全局 UTF-8 编码修复

import asyncio
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# 数据结构
# ---------------------------------------------------------------------------

@dataclass
class SearchResult:
    """单条搜索结果"""
    title: str
    url: str
    content: str
    published_date: Optional[str] = None
    relevance_score: float = 0.0


@dataclass
class KnowledgeNode:
    """知识图谱节点"""
    entity: str
    entity_type: str  # "method" / "material" / "property" / "phenomenon"
    connections: List[Tuple[str, str]] = field(default_factory=list)
    # connections: [(relationship, target_entity), ...]


@dataclass
class CausalChain:
    """因果关系链"""
    cause: str
    effect: str
    mechanism: str
    confidence: float
    source_snippet: str


@dataclass
class TrendSignal:
    """趋势信号"""
    topic: str
    direction: str  # "emerging" / "declining" / "stable" / "breakthrough"
    evidence_count: int
    time_span: str
    key_papers: List[str] = field(default_factory=list)


@dataclass
class LiteratureContext:
    """流水线输出：聚合所有分析结果"""
    search_results: Dict[str, List[SearchResult]] = field(default_factory=dict)
    knowledge_graph: List[KnowledgeNode] = field(default_factory=list)
    causal_chains: List[CausalChain] = field(default_factory=list)
    trends: List[TrendSignal] = field(default_factory=list)
    identified_gaps: List[str] = field(default_factory=list)
    novelty_flags: Dict[str, float] = field(default_factory=dict)
    raw_context_text: str = ""


# ---------------------------------------------------------------------------
# 1. 多路文献搜索
# ---------------------------------------------------------------------------

def _generate_search_queries(domain: str, initial_problem: Optional[str] = None) -> List[str]:
    """根据领域和初始问题生成多角度搜索查询"""
    queries = [
        f"{domain} recent review 2024 2025 research trends",
        f"{domain} open challenges research gaps unsolved problems",
        f"{domain} computational methods DFT machine learning breakthroughs",
        f"{domain} preprint arxiv latest discoveries 2025",
    ]
    if initial_problem:
        queries.append(f"{initial_problem} state of the art recent progress")
    return queries


async def _execute_searches(
    domain: str,
    initial_problem: Optional[str] = None,
) -> Dict[str, List[SearchResult]]:
    """执行多路 Tavily 并行搜索"""
    try:
        from mcp_servers_and_tools.direct_tools.search_tool import tavily_multi_search
    except ImportError:
        # 如果直接导入失败，尝试通过 config 的方式
        import importlib.util as ilu
        from pathlib import Path
        search_path = Path(__file__).resolve().parent.parent / "mcp_servers_and_tools" / "direct_tools" / "search_tool.py"
        spec = ilu.spec_from_file_location("search_tool", str(search_path))
        mod = ilu.module_from_spec(spec)
        spec.loader.exec_module(mod)
        tavily_multi_search = mod.tavily_multi_search

    queries = _generate_search_queries(domain, initial_problem)
    print(f"  [Pipeline] Executing {len(queries)} Tavily searches...")

    raw_results = await tavily_multi_search(queries, search_depth="advanced", max_results=5)

    # 转换为 SearchResult 对象
    structured: Dict[str, List[SearchResult]] = {}
    total_count = 0
    for query, items in raw_results.items():
        structured[query] = [
            SearchResult(
                title=item.get("title", ""),
                url=item.get("url", ""),
                content=item.get("content", ""),
                published_date=item.get("published_date"),
                relevance_score=item.get("score", 0.0),
            )
            for item in items
        ]
        total_count += len(structured[query])

    print(f"  [Pipeline] Retrieved {total_count} search results across {len(queries)} queries")
    return structured


# ---------------------------------------------------------------------------
# 2. 知识图谱构建
# ---------------------------------------------------------------------------

# 实体类型的启发式模式
_METHOD_PATTERNS = re.compile(
    r'\b(DFT|density functional theory|molecular dynamics|MD simulation|'
    r'machine learning|deep learning|neural network|Monte Carlo|'
    r'first.principles|ab initio|VASP|Gaussian|LAMMPS|'
    r'random forest|gradient boosting|GNN|graph neural|'
    r'Bayesian optimization|active learning|transfer learning|'
    r'reinforcement learning|XGBoost|SVR|support vector)\b',
    re.IGNORECASE
)

_MATERIAL_PATTERNS = re.compile(
    r'\b([A-Z][a-z]?(?:\d*[A-Z][a-z]?\d*)+|'  # 化学式如 LiFePO4, TiO2
    r'graphene|perovskite|MOF|metal.organic framework|'
    r'zeolite|polymer|nanoparticle|quantum dot|'
    r'carbon nanotube|alloy|catalyst|electrolyte|'
    r'semiconductor|superconductor|ceramic)\b',
    re.IGNORECASE
)

_PROPERTY_PATTERNS = re.compile(
    r'\b(band gap|conductivity|stability|selectivity|activity|'
    r'efficiency|capacity|voltage|thermal|mechanical|'
    r'adsorption energy|binding energy|formation energy|'
    r'tensile strength|hardness|density|porosity|'
    r'catalytic activity|overpotential|Faradaic efficiency)\b',
    re.IGNORECASE
)


def _extract_entities(text: str) -> List[Tuple[str, str]]:
    """从文本中提取实体及其类型"""
    entities = []
    seen = set()

    for match in _METHOD_PATTERNS.finditer(text):
        ent = match.group(0).strip()
        key = ent.lower()
        if key not in seen:
            entities.append((ent, "method"))
            seen.add(key)

    for match in _MATERIAL_PATTERNS.finditer(text):
        ent = match.group(0).strip()
        key = ent.lower()
        if key not in seen and len(ent) > 2:
            entities.append((ent, "material"))
            seen.add(key)

    for match in _PROPERTY_PATTERNS.finditer(text):
        ent = match.group(0).strip()
        key = ent.lower()
        if key not in seen:
            entities.append((ent, "property"))
            seen.add(key)

    return entities


def _build_knowledge_graph(
    search_results: Dict[str, List[SearchResult]],
) -> List[KnowledgeNode]:
    """从搜索结果构建轻量知识图谱"""
    # 收集所有文本的实体
    all_entities: Dict[str, str] = {}  # entity_lower -> entity_type
    # 共现关系：(entity1, entity2) -> count
    cooccurrence: Counter = Counter()

    for query, results in search_results.items():
        for result in results:
            text = f"{result.title} {result.content}"
            entities = _extract_entities(text)

            for ent, etype in entities:
                all_entities[ent.lower()] = etype

            # 同一文档内的实体视为共现
            ent_names = [e[0].lower() for e in entities]
            for i in range(len(ent_names)):
                for j in range(i + 1, len(ent_names)):
                    pair = tuple(sorted([ent_names[i], ent_names[j]]))
                    cooccurrence[pair] += 1

    # 构建节点和连接
    node_connections: Dict[str, List[Tuple[str, str]]] = defaultdict(list)
    for (e1, e2), count in cooccurrence.items():
        if count >= 2:  # 至少共现 2 次才算有关系
            t1, t2 = all_entities.get(e1, "unknown"), all_entities.get(e2, "unknown")
            rel = _infer_relationship(t1, t2)
            node_connections[e1].append((rel, e2))
            node_connections[e2].append((rel, e1))

    nodes = []
    for entity_lower, etype in all_entities.items():
        nodes.append(KnowledgeNode(
            entity=entity_lower,
            entity_type=etype,
            connections=node_connections.get(entity_lower, []),
        ))

    # 按连接数排序（hub 节点优先）
    nodes.sort(key=lambda n: len(n.connections), reverse=True)
    print(f"  [Pipeline] Knowledge graph: {len(nodes)} entities, "
          f"{sum(len(n.connections) for n in nodes) // 2} relationships")
    return nodes


def _infer_relationship(type1: str, type2: str) -> str:
    """根据实体类型推断关系"""
    pair = frozenset({type1, type2})
    if pair == {"method", "material"}:
        return "applied_to"
    elif pair == {"method", "property"}:
        return "predicts"
    elif pair == {"material", "property"}:
        return "exhibits"
    elif type1 == type2 == "method":
        return "combined_with"
    elif type1 == type2 == "material":
        return "composed_with"
    return "related_to"


# ---------------------------------------------------------------------------
# 3. 因果推理
# ---------------------------------------------------------------------------

_CAUSAL_PATTERNS = [
    # (regex_pattern, cause_group, effect_group)
    (re.compile(r'(.{10,80}?)\s+(?:leads?\s+to|results?\s+in|causes?)\s+(.{10,80})', re.IGNORECASE), 1, 2),
    (re.compile(r'(.{10,80}?)\s+(?:enables?|facilitates?|promotes?)\s+(.{10,80})', re.IGNORECASE), 1, 2),
    (re.compile(r'(.{10,80}?)\s+(?:improves?|enhances?|increases?|boosts?)\s+(.{10,80})', re.IGNORECASE), 1, 2),
    (re.compile(r'(.{10,80}?)\s+(?:reduces?|decreases?|inhibits?|suppresses?)\s+(.{10,80})', re.IGNORECASE), 1, 2),
    (re.compile(r'(?:due\s+to|because\s+of|owing\s+to)\s+(.{10,80}?),?\s+(.{10,80})', re.IGNORECASE), 1, 2),
    (re.compile(r'(.{10,80}?)\s+(?:is\s+attributed\s+to|is\s+caused\s+by)\s+(.{10,80})', re.IGNORECASE), 2, 1),
]


def _extract_causal_chains(
    search_results: Dict[str, List[SearchResult]],
) -> List[CausalChain]:
    """从搜索结果中提取因果关系"""
    raw_chains: List[Tuple[str, str, str]] = []  # (cause, effect, source_snippet)

    for query, results in search_results.items():
        for result in results:
            # 按句子分割
            sentences = re.split(r'[.!?]\s+', result.content)
            for sentence in sentences:
                sentence = sentence.strip()
                if len(sentence) < 20:
                    continue

                for pattern, cause_grp, effect_grp in _CAUSAL_PATTERNS:
                    match = pattern.search(sentence)
                    if match:
                        cause = match.group(cause_grp).strip()
                        effect = match.group(effect_grp).strip()
                        # 过滤过短或过长的提取
                        if 5 < len(cause) < 200 and 5 < len(effect) < 200:
                            raw_chains.append((cause, effect, sentence[:200]))
                        break  # 每句只取第一个匹配

    # 合并相似因果链并计算置信度
    merged = _merge_causal_chains(raw_chains)
    print(f"  [Pipeline] Causal reasoning: {len(merged)} causal chains extracted")
    return merged


def _merge_causal_chains(
    raw_chains: List[Tuple[str, str, str]],
) -> List[CausalChain]:
    """合并相似的因果链，计算置信度"""
    if not raw_chains:
        return []

    # 简单去重：基于 cause+effect 的关键词重叠
    unique_chains: List[CausalChain] = []
    seen_keys: List[set] = []

    for cause, effect, snippet in raw_chains:
        cause_words = set(cause.lower().split())
        effect_words = set(effect.lower().split())
        chain_key = cause_words | effect_words

        # 检查是否与已有链相似
        merged = False
        for i, existing_key in enumerate(seen_keys):
            overlap = len(chain_key & existing_key) / max(len(chain_key | existing_key), 1)
            if overlap > 0.4:
                # 合并：增加置信度
                unique_chains[i].confidence = min(1.0, unique_chains[i].confidence + 0.2)
                merged = True
                break

        if not merged:
            unique_chains.append(CausalChain(
                cause=cause,
                effect=effect,
                mechanism=snippet,
                confidence=0.3,  # 单次出现的基础置信度
                source_snippet=snippet,
            ))
            seen_keys.append(chain_key)

    # 按置信度排序
    unique_chains.sort(key=lambda c: c.confidence, reverse=True)
    return unique_chains[:10]  # 最多返回 10 条


# ---------------------------------------------------------------------------
# 4. 趋势分析
# ---------------------------------------------------------------------------

_RECENCY_CUES = re.compile(
    r'\b(recently|novel|newly|emerging|state.of.the.art|'
    r'first time|cutting.edge|latest|breakthrough|pioneering|'
    r'2024|2025|2026)\b',
    re.IGNORECASE
)

_DECLINE_CUES = re.compile(
    r'\b(traditional|conventional|previously|former|'
    r'outdated|superseded|replaced by)\b',
    re.IGNORECASE
)

_YEAR_PATTERN = re.compile(r'\b(20[12]\d)\b')


def _analyze_trends(
    search_results: Dict[str, List[SearchResult]],
) -> List[TrendSignal]:
    """分析搜索结果中的趋势"""
    # 收集所有文本中的主题词频率
    topic_mentions: Counter = Counter()
    topic_recency: Dict[str, int] = defaultdict(int)  # 近期线索计数
    topic_decline: Dict[str, int] = defaultdict(int)   # 衰退线索计数
    topic_papers: Dict[str, List[str]] = defaultdict(list)
    topic_years: Dict[str, List[int]] = defaultdict(list)

    for query, results in search_results.items():
        for result in results:
            text = f"{result.title} {result.content}"
            entities = _extract_entities(text)

            for ent, etype in entities:
                ent_lower = ent.lower()
                topic_mentions[ent_lower] += 1
                topic_papers[ent_lower].append(result.title[:80])

                # 检查近期/衰退线索
                # 在实体周围 200 字符范围内查找
                for match in re.finditer(re.escape(ent), text, re.IGNORECASE):
                    start = max(0, match.start() - 100)
                    end = min(len(text), match.end() + 100)
                    context = text[start:end]

                    if _RECENCY_CUES.search(context):
                        topic_recency[ent_lower] += 1
                    if _DECLINE_CUES.search(context):
                        topic_decline[ent_lower] += 1

                    # 提取年份
                    for ym in _YEAR_PATTERN.finditer(context):
                        topic_years[ent_lower].append(int(ym.group(1)))

    # 生成趋势信号
    trends = []
    for topic, count in topic_mentions.most_common(20):
        if count < 2:
            continue

        recency = topic_recency.get(topic, 0)
        decline = topic_decline.get(topic, 0)
        years = topic_years.get(topic, [])

        # 判断方向
        if recency >= 3 and decline == 0:
            direction = "breakthrough"
        elif recency > decline:
            direction = "emerging"
        elif decline > recency:
            direction = "declining"
        else:
            direction = "stable"

        # 时间跨度
        if years:
            time_span = f"{min(years)}-{max(years)}"
        else:
            time_span = "unknown"

        trends.append(TrendSignal(
            topic=topic,
            direction=direction,
            evidence_count=count,
            time_span=time_span,
            key_papers=list(set(topic_papers.get(topic, [])))[:3],
        ))

    # 按 evidence_count 排序，breakthrough 和 emerging 优先
    direction_priority = {"breakthrough": 0, "emerging": 1, "stable": 2, "declining": 3}
    trends.sort(key=lambda t: (direction_priority.get(t.direction, 2), -t.evidence_count))

    print(f"  [Pipeline] Trend analysis: {len(trends)} signals detected")
    return trends[:15]


# ---------------------------------------------------------------------------
# 5. 研究空白识别
# ---------------------------------------------------------------------------

_GAP_LANGUAGE = re.compile(
    r'(remains?\s+unclear|not\s+yet\s+(?:investigated|explored|understood|studied)|'
    r'poorly\s+understood|limited\s+(?:understanding|knowledge|research)|'
    r'gap\s+in|challenge\s+(?:remains?|is)|'
    r'further\s+(?:research|investigation|study)\s+(?:is\s+)?(?:needed|required)|'
    r'open\s+(?:question|problem|challenge)|'
    r'unexplored|underexplored|overlooked)',
    re.IGNORECASE
)


def _identify_gaps(
    knowledge_graph: List[KnowledgeNode],
    causal_chains: List[CausalChain],
    search_results: Dict[str, List[SearchResult]],
) -> List[str]:
    """从多个维度识别研究空白"""
    gaps: List[Tuple[float, str]] = []  # (priority_score, description)

    # --- 维度1: 显式空白语言 ---
    for query, results in search_results.items():
        for result in results:
            sentences = re.split(r'[.!?]\s+', result.content)
            for sentence in sentences:
                if _GAP_LANGUAGE.search(sentence) and len(sentence) > 30:
                    gap_desc = sentence.strip()[:200]
                    gaps.append((0.9, f"[Literature] {gap_desc}"))

    # --- 维度2: 知识图谱缺失边 ---
    # 找出高连接度实体对之间的缺失关系
    connected_entities = set()
    entity_neighbors: Dict[str, set] = defaultdict(set)
    for node in knowledge_graph:
        for rel, target in node.connections:
            connected_entities.add(node.entity)
            connected_entities.add(target)
            entity_neighbors[node.entity].add(target)
            entity_neighbors[target].add(node.entity)

    # 检测潜在的缺失边（A->B, B->C 存在但 A-/>C）
    hub_nodes = [n for n in knowledge_graph if len(n.connections) >= 3]
    for hub in hub_nodes[:5]:  # 只检查 top 5 hub 节点
        neighbors = entity_neighbors[hub.entity]
        for n1 in list(neighbors)[:5]:
            for n2 in entity_neighbors.get(n1, set()):
                if n2 != hub.entity and n2 not in neighbors:
                    # n2 通过 n1 与 hub 间接相连，但没有直接关系
                    gaps.append((
                        0.7,
                        f"[Knowledge Graph] Unexplored connection: "
                        f"'{hub.entity}' and '{n2}' are indirectly linked "
                        f"through '{n1}' but not directly studied together"
                    ))

    # --- 维度3: 低置信度因果链 ---
    for chain in causal_chains:
        if chain.confidence < 0.5:
            gaps.append((
                0.6,
                f"[Causal] Poorly understood mechanism: "
                f"'{chain.cause}' -> '{chain.effect}' "
                f"(confidence={chain.confidence:.1f}, needs further investigation)"
            ))

    # 去重并排序
    seen = set()
    unique_gaps = []
    for priority, desc in sorted(gaps, key=lambda x: -x[0]):
        # 简单去重：前 50 字符相同视为重复
        key = desc[:50].lower()
        if key not in seen:
            unique_gaps.append(desc)
            seen.add(key)
        if len(unique_gaps) >= 5:
            break

    print(f"  [Pipeline] Gap identification: {len(unique_gaps)} gaps found")
    return unique_gaps


# ---------------------------------------------------------------------------
# 6. 向量记忆去重
# ---------------------------------------------------------------------------

async def _check_novelty_against_memory(
    candidates: List[str],
    user_id: str,
) -> Dict[str, float]:
    """检查候选主题与历史记忆的重叠度"""
    if not candidates:
        return {}

    try:
        import importlib.util as ilu
        from pathlib import Path
        mem_path = Path(__file__).resolve().parent.parent / "mcp_servers_and_tools" / "direct_tools" / "memory_tools.py"
        if not mem_path.exists():
            mem_path = Path(__file__).resolve().parent / "mcp_servers_and_tools" / "direct_tools" / "memory_tools.py"

        # 尝试多个可能的路径
        for candidate_path in [
            Path(__file__).resolve().parent.parent / "mcp_servers_and_tools" / "direct_tools" / "memory_tools.py",
            Path(__file__).resolve().parent / "mcp_servers_and_tools" / "direct_tools" / "memory_tools.py",
        ]:
            if candidate_path.exists():
                spec = ilu.spec_from_file_location("memory_tools", str(candidate_path))
                mod = ilu.module_from_spec(spec)
                spec.loader.exec_module(mod)
                result = mod.batch_similarity_check(candidates, user_id)
                print(f"  [Pipeline] Memory dedup check: {len(result)} candidates evaluated")
                return result

        print("  [Pipeline] Warning: memory_tools.py not found, skipping dedup")
        return {c: 0.0 for c in candidates}

    except Exception as e:
        print(f"  [Pipeline] Warning: Memory dedup check failed: {e}")
        return {c: 0.0 for c in candidates}


# ---------------------------------------------------------------------------
# 7. 上下文格式化
# ---------------------------------------------------------------------------

def _format_context_for_prompt(ctx: LiteratureContext) -> str:
    """将流水线结果格式化为智能体提示词上下文"""
    sections = []
    sections.append("=== LITERATURE CONTEXT (auto-gathered by research pipeline) ===")
    sections.append("")

    # --- 文献概览 ---
    total_results = sum(len(rs) for rs in ctx.search_results.values())
    if total_results > 0:
        sections.append("## Recent Literature Findings")
        seen_urls = set()
        count = 0
        for query, results in ctx.search_results.items():
            for r in results:
                if r.url not in seen_urls and count < 12:
                    snippet = r.content[:200].replace("\n", " ")
                    entry = f"- [{r.title}]({r.url})"
                    if r.published_date:
                        entry += f" ({r.published_date})"
                    entry += f": {snippet}..."
                    sections.append(entry)
                    seen_urls.add(r.url)
                    count += 1
        sections.append("")

    # --- 知识图谱洞察 ---
    if ctx.knowledge_graph:
        sections.append("## Knowledge Graph Insights")
        hub_nodes = [n for n in ctx.knowledge_graph if len(n.connections) >= 2][:5]
        if hub_nodes:
            sections.append("Hub entities (highly connected):")
            for node in hub_nodes:
                targets = [t for _, t in node.connections[:5]]
                sections.append(
                    f"- {node.entity} ({node.entity_type}): "
                    f"connected to {', '.join(targets)}"
                )
        # 桥接概念
        bridge_nodes = [
            n for n in ctx.knowledge_graph
            if len(set(t for _, t in n.connections)) >= 3
        ][:3]
        if bridge_nodes:
            sections.append("Bridge concepts (linking multiple domains):")
            for node in bridge_nodes:
                sections.append(f"- {node.entity}: bridges {len(node.connections)} connections")
        sections.append("")

    # --- 因果关系 ---
    if ctx.causal_chains:
        sections.append("## Causal Relationships Found")
        for chain in ctx.causal_chains[:6]:
            sections.append(
                f"- {chain.cause} -> {chain.effect} "
                f"(confidence={chain.confidence:.1f})"
            )
        sections.append("")

    # --- 趋势分析 ---
    if ctx.trends:
        sections.append("## Trend Analysis")
        for trend in ctx.trends[:8]:
            marker = {
                "breakthrough": "BREAKTHROUGH",
                "emerging": "EMERGING",
                "declining": "DECLINING",
                "stable": "STABLE",
            }.get(trend.direction, trend.direction.upper())
            sections.append(
                f"- [{marker}] {trend.topic} "
                f"(evidence: {trend.evidence_count}, span: {trend.time_span})"
            )
        sections.append("")

    # --- 研究空白 ---
    if ctx.identified_gaps:
        sections.append("## Identified Research Gaps (prioritized)")
        for i, gap in enumerate(ctx.identified_gaps, 1):
            sections.append(f"{i}. {gap}")
        sections.append("")

    # --- 记忆去重标记 ---
    if ctx.novelty_flags:
        high_overlap = {k: v for k, v in ctx.novelty_flags.items() if v > 0.4}
        if high_overlap:
            sections.append("## Prior Work Overlap Check")
            for topic, sim in sorted(high_overlap.items(), key=lambda x: -x[1]):
                flag = "HIGH - AVOID" if sim > 0.6 else "MODERATE"
                sections.append(f"- \"{topic}\" overlap with prior discoveries: {sim:.2f} ({flag})")
            sections.append("")

    sections.append("=== END LITERATURE CONTEXT ===")
    sections.append("")
    sections.append(
        "Use the above context to inform your analysis. "
        "Cite provided URLs in supporting_literature fields. "
        "Prioritize gaps identified by the automated analysis. "
        "AVOID research topics flagged with high prior work overlap (>0.6)."
    )

    return "\n".join(sections)


# ---------------------------------------------------------------------------
# 主入口：流水线编排
# ---------------------------------------------------------------------------

async def run_literature_pipeline(
    domain: str,
    initial_problem: Optional[str] = None,
    user_id: str = "experiment_system",
    timeout: float = 120.0,
) -> LiteratureContext:
    """
    运行完整的文献挖掘预处理流水线。

    参数:
        domain: 研究领域
        initial_problem: 可选的初始问题
        user_id: 用户标识（用于记忆去重）
        timeout: 总超时时间（秒）

    返回:
        LiteratureContext 聚合所有分析结果
    """
    print(f"  [Pipeline] Starting literature pipeline for domain: {domain}")
    ctx = LiteratureContext()

    try:
        # === 阶段1: 文献搜索（异步，可能耗时最长）===
        try:
            ctx.search_results = await asyncio.wait_for(
                _execute_searches(domain, initial_problem),
                timeout=60.0,
            )
        except asyncio.TimeoutError:
            print("  [Pipeline] Warning: Search timed out, proceeding with empty results")
            ctx.search_results = {}
        except Exception as e:
            print(f"  [Pipeline] Warning: Search failed: {e}")
            ctx.search_results = {}

        if not ctx.search_results or all(len(v) == 0 for v in ctx.search_results.values()):
            print("  [Pipeline] No search results available, returning minimal context")
            ctx.raw_context_text = ""
            return ctx

        # === 阶段2: 知识图谱构建（同步，纯 CPU）===
        try:
            ctx.knowledge_graph = _build_knowledge_graph(ctx.search_results)
        except Exception as e:
            print(f"  [Pipeline] Warning: Knowledge graph construction failed: {e}")
            ctx.knowledge_graph = []

        # === 阶段3: 因果链提取（同步，纯正则）===
        try:
            ctx.causal_chains = _extract_causal_chains(ctx.search_results)
        except Exception as e:
            print(f"  [Pipeline] Warning: Causal extraction failed: {e}")
            ctx.causal_chains = []

        # === 阶段4: 趋势分析（同步）===
        try:
            ctx.trends = _analyze_trends(ctx.search_results)
        except Exception as e:
            print(f"  [Pipeline] Warning: Trend analysis failed: {e}")
            ctx.trends = []

        # === 阶段5: 研究空白识别 ===
        try:
            ctx.identified_gaps = _identify_gaps(
                ctx.knowledge_graph, ctx.causal_chains, ctx.search_results
            )
        except Exception as e:
            print(f"  [Pipeline] Warning: Gap identification failed: {e}")
            ctx.identified_gaps = []

        # === 阶段6: 向量记忆去重 ===
        try:
            # 收集候选主题用于去重
            candidates = []
            # 从趋势中收集
            candidates.extend([t.topic for t in ctx.trends if t.direction in ("emerging", "breakthrough")])
            # 从空白中收集摘要
            candidates.extend([g[:100] for g in ctx.identified_gaps])

            if candidates:
                ctx.novelty_flags = await asyncio.wait_for(
                    _check_novelty_against_memory(candidates, user_id),
                    timeout=15.0,
                )
            else:
                ctx.novelty_flags = {}
        except asyncio.TimeoutError:
            print("  [Pipeline] Warning: Memory dedup check timed out")
            ctx.novelty_flags = {}
        except Exception as e:
            print(f"  [Pipeline] Warning: Memory dedup check failed: {e}")
            ctx.novelty_flags = {}

        # === 阶段7: 格式化上下文 ===
        ctx.raw_context_text = _format_context_for_prompt(ctx)
        print(f"  [Pipeline] Context assembled: {len(ctx.raw_context_text)} chars")

    except Exception as e:
        print(f"  [Pipeline] Fatal pipeline error: {e}")
        ctx.raw_context_text = ""

    return ctx


# ---------------------------------------------------------------------------
# 测试入口
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    async def _test():
        print("=" * 60)
        print("Testing Literature Pipeline")
        print("=" * 60)

        ctx = await run_literature_pipeline(
            domain="catalysis",
            initial_problem="design high-performance electrocatalysts for CO2 reduction",
        )

        print("\n" + "=" * 60)
        print("Pipeline Output Summary:")
        print(f"  Search results: {sum(len(v) for v in ctx.search_results.values())} items")
        print(f"  Knowledge graph nodes: {len(ctx.knowledge_graph)}")
        print(f"  Causal chains: {len(ctx.causal_chains)}")
        print(f"  Trends: {len(ctx.trends)}")
        print(f"  Identified gaps: {len(ctx.identified_gaps)}")
        print(f"  Novelty flags: {len(ctx.novelty_flags)}")
        print(f"  Context text length: {len(ctx.raw_context_text)} chars")

        if ctx.raw_context_text:
            print("\n" + "=" * 60)
            print("Formatted Context (first 2000 chars):")
            print(ctx.raw_context_text[:2000])

    asyncio.run(_test())
