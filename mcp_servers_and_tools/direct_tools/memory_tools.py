"""
Memory Tools - 基于本地 JSON 文件的记忆功能
支持 TF-IDF 向量相似度搜索（sklearn 可用时）和关键词匹配回退。
"""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

# 记忆存储目录
MEMORY_DIR = Path(__file__).resolve().parent.parent.parent / "saved_code" / "memory"
MEMORY_DIR.mkdir(parents=True, exist_ok=True)

# TF-IDF 可用性检测（懒加载）
_sklearn_available: Optional[bool] = None


def _check_sklearn() -> bool:
    """检查 scikit-learn 是否可用"""
    global _sklearn_available
    if _sklearn_available is None:
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer  # noqa: F401
            from sklearn.metrics.pairwise import cosine_similarity  # noqa: F401
            _sklearn_available = True
        except ImportError:
            _sklearn_available = False
    return _sklearn_available


def _get_memory_file(user_id: str) -> Path:
    """获取用户的记忆文件路径"""
    return MEMORY_DIR / f"{user_id}_memory.json"


def _load_memories(user_id: str) -> list:
    """加载用户的所有记忆条目"""
    memory_file = _get_memory_file(user_id)
    if not memory_file.exists():
        return []
    try:
        with open(memory_file, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return []


def _keyword_search(query: str, memories: list, limit: int) -> list:
    """
    关键词匹配搜索（原有算法，作为回退方案）

    参数:
        query: 搜索关键词
        memories: 记忆列表
        limit: 返回数量限制

    返回:
        (relevance, memory) 元组列表，按相关度降序
    """
    query_lower = query.lower()
    keywords = [w.strip() for w in query_lower.split() if len(w.strip()) >= 2]
    if not keywords:
        keywords = [query_lower]

    scored = []
    for mem in memories:
        content = mem.get("content", "").lower()
        match_count = sum(1 for kw in keywords if kw in content)
        if match_count > 0:
            relevance = match_count / len(keywords)
            scored.append((relevance, mem))

    scored.sort(key=lambda x: x[0], reverse=True)
    return scored[:limit]


def _tfidf_search(query: str, memories: list, limit: int) -> list:
    """
    TF-IDF 余弦相似度搜索（sklearn 可用时使用）

    参数:
        query: 搜索查询
        memories: 记忆列表
        limit: 返回数量限制

    返回:
        (similarity, memory) 元组列表，按相似度降序
    """
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity

    contents = [mem.get("content", "") for mem in memories]
    if not contents:
        return []

    # 构建 TF-IDF 矩阵：[query] + [所有记忆内容]
    all_texts = [query] + contents
    vectorizer = TfidfVectorizer(
        max_features=5000,
        stop_words="english",
        ngram_range=(1, 2),
    )

    try:
        tfidf_matrix = vectorizer.fit_transform(all_texts)
    except ValueError:
        # 所有文档为空或只有停用词时回退
        return _keyword_search(query, memories, limit)

    # 计算查询向量与每个记忆向量的余弦相似度
    query_vec = tfidf_matrix[0:1]
    doc_vecs = tfidf_matrix[1:]
    similarities = cosine_similarity(query_vec, doc_vecs).flatten()

    # 按相似度排序
    scored = []
    for idx, sim in enumerate(similarities):
        if sim > 0.01:  # 过滤极低相似度
            scored.append((float(sim), memories[idx]))

    scored.sort(key=lambda x: x[0], reverse=True)
    return scored[:limit]


def search_memory(query: str, user_id: str = "experiment_system", limit: int = 5) -> str:
    """
    搜索记忆

    参数:
        query: 搜索关键词
        user_id: 用户标识
        limit: 返回结果数量限制

    返回:
        搜索到的记忆内容，如果没有则返回提示信息
    """
    memories = _load_memories(user_id)
    if not memories:
        return "No prior discoveries found in memory."

    try:
        # 优先使用 TF-IDF 向量搜索，不可用时回退到关键词匹配
        if _check_sklearn():
            scored_matches = _tfidf_search(query, memories, limit)
        else:
            scored_matches = _keyword_search(query, memories, limit)

        if not scored_matches:
            return "No relevant memories found for the given query."

        # 格式化结果
        results = []
        for i, (score, mem) in enumerate(scored_matches, 1):
            timestamp = mem.get("timestamp", "unknown")
            content = mem.get("content", "")[:300]  # 限制长度
            results.append(f"{i}. [relevance={score:.2f}] [{timestamp}] {content}...")

        return "\n".join(results)

    except Exception as e:
        return f"Error searching memory: {e}"


def batch_similarity_check(
    candidates: List[str],
    user_id: str = "experiment_system",
) -> Dict[str, float]:
    """
    批量检查候选主题与历史记忆的相似度（用于去重）。

    参数:
        candidates: 候选主题/描述列表
        user_id: 用户标识

    返回:
        字典 {candidate: max_similarity}，相似度越高表示与历史越重叠
    """
    memories = _load_memories(user_id)
    if not memories or not candidates:
        return {c: 0.0 for c in candidates}

    contents = [mem.get("content", "") for mem in memories]

    if _check_sklearn():
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity

        all_texts = candidates + contents
        vectorizer = TfidfVectorizer(
            max_features=5000,
            stop_words="english",
            ngram_range=(1, 2),
        )
        try:
            tfidf_matrix = vectorizer.fit_transform(all_texts)
            candidate_vecs = tfidf_matrix[: len(candidates)]
            memory_vecs = tfidf_matrix[len(candidates) :]
            sim_matrix = cosine_similarity(candidate_vecs, memory_vecs)

            result = {}
            for i, candidate in enumerate(candidates):
                max_sim = float(sim_matrix[i].max()) if sim_matrix[i].size > 0 else 0.0
                result[candidate] = max_sim
            return result
        except ValueError:
            pass

    # 回退：关键词重叠度
    result = {}
    for candidate in candidates:
        cand_words = set(candidate.lower().split())
        max_overlap = 0.0
        for content in contents:
            mem_words = set(content.lower().split())
            if cand_words:
                overlap = len(cand_words & mem_words) / len(cand_words)
                max_overlap = max(max_overlap, overlap)
        result[candidate] = max_overlap
    return result


def save_memory(content: str, user_id: str = "experiment_system", tags: Optional[List[str]] = None) -> bool:
    """
    保存记忆

    参数:
        content: 记忆内容
        user_id: 用户标识
        tags: 标签列表

    返回:
        是否保存成功
    """
    memory_file = _get_memory_file(user_id)

    try:
        # 加载现有记忆
        if memory_file.exists():
            with open(memory_file, "r", encoding="utf-8") as f:
                memories = json.load(f)
        else:
            memories = []

        # 添加新记忆
        new_memory = {
            "content": content,
            "timestamp": datetime.now().isoformat(),
            "tags": tags or [],
            "id": f"mem_{len(memories) + 1}"
        }

        memories.append(new_memory)

        # 限制记忆数量（保留最近100条）
        if len(memories) > 100:
            memories = memories[-100:]

        # 保存
        with open(memory_file, "w", encoding="utf-8") as f:
            json.dump(memories, f, indent=2, ensure_ascii=False)

        return True

    except Exception as e:
        print(f"Error saving memory: {e}")
        return False


class MemoryTools:
    """记忆工具类，用于兼容原有的工具加载方式"""

    def __init__(self):
        self.search_memory = search_memory
        self.save_memory = save_memory
        self.batch_similarity_check = batch_similarity_check


if __name__ == "__main__":
    # 简单测试
    print("Testing memory tools...")
    print(f"  sklearn available: {_check_sklearn()}")

    # 测试保存
    ok = save_memory("Test discovery about catalysis with DFT methods", tags=["test"])
    print(f"  save_memory: {'OK' if ok else 'FAILED'}")

    # 测试搜索
    result = search_memory("catalysis DFT")
    print(f"  search_memory result:\n{result}")

    # 测试批量相似度检查
    sim = batch_similarity_check(["catalysis DFT methods", "quantum computing algorithms"])
    print(f"  batch_similarity_check: {sim}")
