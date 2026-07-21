"""
Tavily Search Tool - Tavily API 集成模块

通过 httpx 异步 HTTP 客户端调用 Tavily Search API，
支持单查询和多查询并行搜索，用于文献挖掘和研究空白识别。

既可作为智能体工具（非 DeepSeek 模型），也可被 literature_pipeline 直接调用。
"""

from __future__ import annotations

import asyncio
import os
from typing import Optional


# Tavily API 端点
TAVILY_API_URL = "https://api.tavily.com/search"


def _get_tavily_api_key() -> str | None:
    """从环境变量读取 Tavily API Key"""
    return os.getenv("TAVILY_API_KEY") or None


async def tavily_search(
    query: str,
    search_depth: str = "advanced",
    max_results: int = 5,
    topic: str = "general",
    include_domains: list[str] | None = None,
) -> list[dict]:
    """
    执行单次 Tavily 搜索查询。

    参数:
        query: 搜索查询字符串
        search_depth: 搜索深度 ("basic" 或 "advanced")
        max_results: 最大返回结果数
        topic: 搜索主题类型 ("general" 或 "news")
        include_domains: 限定搜索域名列表（可选）

    返回:
        结果列表，每项包含 title, url, content, published_date, score
    """
    api_key = _get_tavily_api_key()
    if not api_key:
        print("  [Tavily] Warning: TAVILY_API_KEY not set, skipping search")
        return []

    try:
        import httpx
    except ImportError:
        print("  [Tavily] Warning: httpx not installed, skipping search")
        return []

    payload = {
        "api_key": api_key,
        "query": query,
        "search_depth": search_depth,
        "max_results": max_results,
        "topic": topic,
        "include_answer": False,
    }
    if include_domains:
        payload["include_domains"] = include_domains

    retries = 2
    for attempt in range(retries + 1):
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(TAVILY_API_URL, json=payload)

                if response.status_code == 429:
                    # 速率限制 - 指数退避重试
                    if attempt < retries:
                        wait = 2 ** (attempt + 1)
                        print(f"  [Tavily] Rate limited, retrying in {wait}s...")
                        await asyncio.sleep(wait)
                        continue
                    print("  [Tavily] Rate limit exceeded after retries")
                    return []

                response.raise_for_status()
                data = response.json()

                results = []
                for item in data.get("results", []):
                    results.append({
                        "title": item.get("title", ""),
                        "url": item.get("url", ""),
                        "content": item.get("content", ""),
                        "published_date": item.get("published_date"),
                        "score": item.get("score", 0.0),
                    })
                return results

        except httpx.TimeoutException:
            if attempt < retries:
                print(f"  [Tavily] Timeout on attempt {attempt + 1}, retrying...")
                continue
            print("  [Tavily] Search timed out after retries")
            return []
        except Exception as e:
            print(f"  [Tavily] Search error: {e}")
            return []

    return []


async def tavily_multi_search(
    queries: list[str],
    search_depth: str = "advanced",
    max_results: int = 5,
    topic: str = "general",
) -> dict[str, list[dict]]:
    """
    并行执行多个 Tavily 搜索查询。

    参数:
        queries: 查询字符串列表
        search_depth: 搜索深度
        max_results: 每个查询的最大结果数
        topic: 搜索主题类型

    返回:
        字典 {query_string: [results]}
    """
    if not queries:
        return {}

    tasks = [
        tavily_search(q, search_depth=search_depth, max_results=max_results, topic=topic)
        for q in queries
    ]

    results_list = await asyncio.gather(*tasks, return_exceptions=True)

    output = {}
    for query, result in zip(queries, results_list):
        if isinstance(result, Exception):
            print(f"  [Tavily] Query failed: {query[:50]}... -> {result}")
            output[query] = []
        else:
            output[query] = result

    return output


# === 智能体工具接口 ===
# 同步包装器，供 OpenAI Agents SDK 的 function tool 使用
def tavily_search_sync(
    query: str,
    search_depth: str = "advanced",
    max_results: int = 5,
) -> str:
    """
    Search the web for scientific literature and recent publications.
    搜索网络上的科学文献和最新出版物。

    Args:
        query: Search query string (搜索查询)
        search_depth: Search depth - "basic" or "advanced" (搜索深度)
        max_results: Maximum number of results (最大结果数)

    Returns:
        Formatted search results as text (格式化的搜索结果文本)
    """
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as pool:
                results = pool.submit(
                    asyncio.run,
                    tavily_search(query, search_depth, max_results)
                ).result(timeout=60)
        else:
            results = loop.run_until_complete(
                tavily_search(query, search_depth, max_results)
            )
    except Exception:
        results = asyncio.run(tavily_search(query, search_depth, max_results))

    if not results:
        return "No search results found."

    formatted = []
    for i, r in enumerate(results, 1):
        entry = f"{i}. {r['title']}\n   URL: {r['url']}\n   {r['content'][:300]}"
        if r.get("published_date"):
            entry += f"\n   Published: {r['published_date']}"
        formatted.append(entry)

    return "\n\n".join(formatted)


# 模块级导出 - 兼容 config.py 的 load_tavily_search() 接口
tavily_search_tool = tavily_search_sync


if __name__ == "__main__":
    async def _test():
        print("Testing Tavily search...")
        results = await tavily_search("catalysis research trends 2025", max_results=3)
        if results:
            for r in results:
                print(f"  - {r['title']}: {r['url']}")
            print(f"\nSuccess: {len(results)} results returned")
        else:
            print("No results (check API key)")

        print("\nTesting multi-search...")
        multi = await tavily_multi_search([
            "battery materials research gaps",
            "DFT machine learning catalysis",
        ], max_results=2)
        for q, rs in multi.items():
            print(f"  Query: {q[:40]}... -> {len(rs)} results")

    asyncio.run(_test())
