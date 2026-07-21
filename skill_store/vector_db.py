"""
向量数据库层 — ChromaDB 持久化存储

提供技能嵌入的 CRUD 操作和语义搜索功能。
ChromaDB 不可用时优雅降级到基于 JSON + TF-IDF 的回退方案。
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .models import Skill

# ChromaDB 可用性检测
_chromadb_available: Optional[bool] = None
_client_instance = None

# 默认存储路径
_DEFAULT_DB_DIR = Path(__file__).resolve().parent.parent / "saved_code" / "skill_vectordb"


def _check_chromadb() -> bool:
    """检查 ChromaDB 是否可用"""
    global _chromadb_available
    if _chromadb_available is not None:
        return _chromadb_available
    try:
        import chromadb  # noqa: F401
        _chromadb_available = True
    except ImportError:
        _chromadb_available = False
        print("  Skill store: chromadb 不可用，回退到 JSON + TF-IDF")
    return _chromadb_available


def _get_client():
    """获取或创建 ChromaDB PersistentClient"""
    global _client_instance
    if _client_instance is not None:
        return _client_instance

    import chromadb

    db_dir = os.getenv("SKILL_VECTOR_DB_DIR", str(_DEFAULT_DB_DIR))
    os.makedirs(db_dir, exist_ok=True)

    _client_instance = chromadb.PersistentClient(path=db_dir)
    return _client_instance


def _get_collection(user_id: str):
    """获取用户的技能 collection"""
    client = _get_client()
    collection_name = f"skills_{user_id}"[:63]  # ChromaDB 名称长度限制
    return client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"},
    )


class SkillVectorDB:
    """技能向量数据库操作类"""

    def __init__(self, user_id: str = "experiment_system"):
        self.user_id = user_id
        self._available = _check_chromadb()

        if self._available:
            try:
                self._collection = _get_collection(user_id)
                print(f"  Skill store: ChromaDB 已连接 (collection: skills_{user_id})")
            except Exception as e:
                print(f"  Skill store: ChromaDB 初始化失败: {e}")
                self._available = False
                self._collection = None
        else:
            self._collection = None

    @property
    def is_available(self) -> bool:
        return self._available and self._collection is not None

    def add_skill(self, skill: Skill) -> bool:
        """
        添加或更新技能到向量数据库。

        参数:
            skill: 技能对象（必须包含 embedding）

        返回:
            是否成功
        """
        if not self.is_available:
            return False

        if skill.embedding is None:
            print(f"  Warning: Skill {skill.id} 没有 embedding，跳过向量存储")
            return False

        try:
            # 构建文档内容（用于全文搜索备份）
            document = skill.to_embedding_text()

            # 元数据（ChromaDB 只支持 str/int/float/bool）
            metadata = skill.to_metadata()

            self._collection.upsert(
                ids=[skill.id],
                embeddings=[skill.embedding],
                documents=[document],
                metadatas=[metadata],
            )
            return True
        except Exception as e:
            print(f"  Warning: 技能写入 ChromaDB 失败: {e}")
            return False

    def search(
        self,
        query_embedding: List[float],
        top_k: int = 5,
        where_filters: Optional[Dict] = None,
    ) -> List[Tuple[str, float, Dict]]:
        """
        语义搜索技能。

        参数:
            query_embedding: 查询向量
            top_k: 返回数量
            where_filters: ChromaDB where 过滤条件

        返回:
            [(skill_id, similarity_score, metadata)] 列表
        """
        if not self.is_available:
            return []

        try:
            kwargs = {
                "query_embeddings": [query_embedding],
                "n_results": top_k,
                "include": ["metadatas", "distances", "documents"],
            }
            if where_filters:
                kwargs["where"] = where_filters

            results = self._collection.query(**kwargs)

            if not results or not results["ids"] or not results["ids"][0]:
                return []

            output = []
            for i, skill_id in enumerate(results["ids"][0]):
                # ChromaDB 返回的是距离（越小越相似），转换为相似度
                distance = results["distances"][0][i] if results["distances"] else 0.0
                similarity = 1.0 - distance  # cosine distance → similarity
                metadata = results["metadatas"][0][i] if results["metadatas"] else {}
                output.append((skill_id, similarity, metadata))

            return output
        except Exception as e:
            print(f"  Warning: ChromaDB 搜索失败: {e}")
            return []

    def get_by_id(self, skill_id: str) -> Optional[Dict]:
        """
        按 ID 获取技能数据。

        参数:
            skill_id: 技能 ID

        返回:
            技能的元数据和文档，不存在返回 None
        """
        if not self.is_available:
            return None

        try:
            result = self._collection.get(
                ids=[skill_id],
                include=["metadatas", "documents", "embeddings"],
            )
            if result and result["ids"]:
                return {
                    "id": result["ids"][0],
                    "metadata": result["metadatas"][0] if result["metadatas"] else {},
                    "document": result["documents"][0] if result["documents"] else "",
                    "embedding": result["embeddings"][0] if result["embeddings"] else None,
                }
            return None
        except Exception as e:
            print(f"  Warning: ChromaDB get_by_id 失败: {e}")
            return None

    def update_metadata(self, skill_id: str, updates: Dict) -> bool:
        """
        更新技能的元数据。

        参数:
            skill_id: 技能 ID
            updates: 要更新的元数据字段

        返回:
            是否成功
        """
        if not self.is_available:
            return False

        try:
            self._collection.update(
                ids=[skill_id],
                metadatas=[updates],
            )
            return True
        except Exception as e:
            print(f"  Warning: ChromaDB 元数据更新失败: {e}")
            return False

    def delete_skill(self, skill_id: str) -> bool:
        """
        删除技能（从向量索引中移除）。

        参数:
            skill_id: 技能 ID

        返回:
            是否成功
        """
        if not self.is_available:
            return False

        try:
            self._collection.delete(ids=[skill_id])
            return True
        except Exception as e:
            print(f"  Warning: ChromaDB 删除失败: {e}")
            return False

    def count(self) -> int:
        """返回当前存储的技能数量"""
        if not self.is_available:
            return 0
        try:
            return self._collection.count()
        except Exception:
            return 0

    def list_all_ids(self) -> List[str]:
        """列出所有技能 ID"""
        if not self.is_available:
            return []
        try:
            result = self._collection.get(include=[])
            return result["ids"] if result else []
        except Exception:
            return []


# ---------------------------------------------------------------------------
# JSON 回退存储（chromadb 不可用时使用）
# ---------------------------------------------------------------------------

_FALLBACK_FILE = _DEFAULT_DB_DIR.parent / "skill_store_fallback.json"


class FallbackSkillStore:
    """基于 JSON 文件的回退技能存储"""

    def __init__(self, user_id: str = "experiment_system"):
        self.user_id = user_id
        self._file = _FALLBACK_FILE.parent / f"skills_{user_id}.json"
        self._skills: Dict[str, dict] = {}
        self._load()

    def _load(self):
        """从文件加载"""
        if self._file.exists():
            try:
                with open(self._file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                self._skills = {s["id"]: s for s in data if "id" in s}
            except Exception:
                self._skills = {}

    def _save(self):
        """保存到文件"""
        os.makedirs(str(self._file.parent), exist_ok=True)
        try:
            with open(self._file, "w", encoding="utf-8") as f:
                json.dump(list(self._skills.values()), f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"  Warning: 回退存储写入失败: {e}")

    def add_skill(self, skill: Skill) -> bool:
        """添加技能"""
        self._skills[skill.id] = skill.model_dump()
        self._save()
        return True

    def get_by_id(self, skill_id: str) -> Optional[dict]:
        """按 ID 获取"""
        return self._skills.get(skill_id)

    def search_by_text(self, query: str, top_k: int = 5) -> List[Tuple[str, float]]:
        """基于 TF-IDF 的文本搜索（回退方案）"""
        if not self._skills:
            return []

        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.metrics.pairwise import cosine_similarity

            skill_texts = []
            skill_ids = []
            for sid, sdata in self._skills.items():
                text = f"{sdata.get('domain', '')} {sdata.get('method', '')} " \
                       f"{' '.join(sdata.get('steps', []))} " \
                       f"{sdata.get('code_specification', '')[:300]}"
                skill_texts.append(text)
                skill_ids.append(sid)

            all_texts = [query] + skill_texts
            vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
            tfidf_matrix = vectorizer.fit_transform(all_texts)

            query_vec = tfidf_matrix[0:1]
            doc_vecs = tfidf_matrix[1:]
            similarities = cosine_similarity(query_vec, doc_vecs).flatten()

            scored = [
                (skill_ids[i], float(sim))
                for i, sim in enumerate(similarities)
                if sim > 0.01
            ]
            scored.sort(key=lambda x: x[1], reverse=True)
            return scored[:top_k]

        except ImportError:
            # sklearn 也不可用，用关键词匹配
            query_words = set(query.lower().split())
            scored = []
            for sid, sdata in self._skills.items():
                text = f"{sdata.get('domain', '')} {sdata.get('method', '')} " \
                       f"{' '.join(sdata.get('steps', []))}"
                text_words = set(text.lower().split())
                overlap = len(query_words & text_words)
                if overlap > 0:
                    scored.append((sid, overlap / max(len(query_words), 1)))
            scored.sort(key=lambda x: x[1], reverse=True)
            return scored[:top_k]

    def count(self) -> int:
        return len(self._skills)

    def list_all_ids(self) -> List[str]:
        return list(self._skills.keys())

    def delete_skill(self, skill_id: str) -> bool:
        if skill_id in self._skills:
            del self._skills[skill_id]
            self._save()
            return True
        return False


def create_skill_db(user_id: str = "experiment_system"):
    """
    工厂函数：创建技能数据库实例。

    优先使用 ChromaDB，不可用时回退到 JSON + TF-IDF。
    """
    if _check_chromadb():
        try:
            db = SkillVectorDB(user_id)
            if db.is_available:
                return db
        except Exception:
            pass

    print("  Skill store: 使用 JSON 回退存储")
    return FallbackSkillStore(user_id)
