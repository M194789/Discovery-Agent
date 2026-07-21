"""
嵌入层 — 使用 OpenAI text-embedding-3-small 生成密集向量

提供技能文本的嵌入生成功能，用于向量数据库存储和语义搜索。
支持单条和批量嵌入，降级时返回 None。
"""

from __future__ import annotations

import asyncio
from typing import List, Optional

# 可用性标志
_available: Optional[bool] = None
_embedding_model: str = "text-embedding-3-small"
_embedding_dimensions: int = 1536


def _check_available() -> bool:
    """检查 OpenAI embedding API 是否可用"""
    global _available
    if _available is not None:
        return _available
    try:
        from config import client  # noqa: F401
        _available = True
    except Exception:
        _available = False
    return _available


async def _async_embed_text(text: str) -> Optional[List[float]]:
    """异步调用 OpenAI embedding API"""
    from config import client

    try:
        response = await client.embeddings.create(
            model=_embedding_model,
            input=text[:8000],  # API 限制 token，截断过长文本
        )
        return response.data[0].embedding
    except Exception as e:
        print(f"  Warning: Embedding generation failed: {e}")
        return None


async def _async_embed_batch(texts: List[str]) -> List[Optional[List[float]]]:
    """异步批量生成嵌入"""
    from config import client

    try:
        # 截断每条文本
        truncated = [t[:8000] for t in texts]
        response = await client.embeddings.create(
            model=_embedding_model,
            input=truncated,
        )
        return [item.embedding for item in response.data]
    except Exception as e:
        print(f"  Warning: Batch embedding failed: {e}")
        return [None] * len(texts)


def embed_text(text: str) -> Optional[List[float]]:
    """
    生成文本的密集向量嵌入（同步接口）。

    参数:
        text: 要嵌入的文本

    返回:
        1536 维浮点向量，如果不可用返回 None
    """
    if not _check_available():
        return None

    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # 在已运行的事件循环中，创建新任务
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as pool:
                future = pool.submit(asyncio.run, _async_embed_text(text))
                return future.result(timeout=30)
        else:
            return loop.run_until_complete(_async_embed_text(text))
    except RuntimeError:
        # 没有事件循环，直接创建
        return asyncio.run(_async_embed_text(text))
    except Exception as e:
        print(f"  Warning: embed_text failed: {e}")
        return None


def embed_batch(texts: List[str]) -> List[Optional[List[float]]]:
    """
    批量生成密集向量嵌入（同步接口）。

    参数:
        texts: 文本列表

    返回:
        嵌入向量列表，失败的位置为 None
    """
    if not _check_available() or not texts:
        return [None] * len(texts)

    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as pool:
                future = pool.submit(asyncio.run, _async_embed_batch(texts))
                return future.result(timeout=60)
        else:
            return loop.run_until_complete(_async_embed_batch(texts))
    except RuntimeError:
        return asyncio.run(_async_embed_batch(texts))
    except Exception as e:
        print(f"  Warning: embed_batch failed: {e}")
        return [None] * len(texts)


async def async_embed_text(text: str) -> Optional[List[float]]:
    """
    异步生成文本嵌入（供 async 上下文直接调用）。

    参数:
        text: 要嵌入的文本

    返回:
        1536 维浮点向量，失败返回 None
    """
    if not _check_available():
        return None
    return await _async_embed_text(text)


async def async_embed_batch(texts: List[str]) -> List[Optional[List[float]]]:
    """
    异步批量生成嵌入（供 async 上下文直接调用）。

    参数:
        texts: 文本列表

    返回:
        嵌入向量列表
    """
    if not _check_available() or not texts:
        return [None] * len(texts)
    return await _async_embed_batch(texts)


def get_embedding_dimensions() -> int:
    """返回嵌入向量的维度数"""
    return _embedding_dimensions
