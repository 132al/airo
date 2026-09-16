# webtest/llm_recommender.py
import json
from .ollama_client import client


def pick_seed(user_query):
    """让 LLM 凭知识推荐 1 首种子"""
    prompt = f"""用户想要：{user_query}

请推荐 1 首最符合这个描述的歌。
要求：
- 选一首你确定真实存在的歌
- 歌名和歌手都要给
- 输出 JSON

输出格式：
{{"track_name": "...", "artist_name": "..."}}
"""
    resp = client.generate(prompt, max_tokens=100, temperature=0.7, format="json")
    try:
        data = json.loads(resp)
        return data.get("track_name"), data.get("artist_name")
    except Exception:
        return None, None


def filter_candidates(user_query, seed_info, candidates, top_n=5):
    """让 LLM 从候选里选 top_n 条"""
    if not candidates:
        return []

    lines = []
    for i, c in enumerate(candidates, 1):
        lines.append(
            f"{i}. 《{c.get('track_name', '')}》- {c.get('artist_name', '')}"
            f"（流派：{c.get('artist_genres', '未知')}）"
        )
    candidates_text = "\n".join(lines)

    prompt = f"""用户想要：{user_query}

系统根据种子《{seed_info.get('track_name', '')}》- {seed_info.get('artist_name', '')} 召回了以下 {len(candidates)} 首歌：

{candidates_text}

请从中选出最符合用户描述的 {top_n} 首，输出它们的序号。
要求：
- 输出 JSON
- 只输出序号数组，从 1 开始
- 按符合程度排序

输出格式：
{{"picks": [3, 7, 12, 1, 18]}}
"""
    resp = client.generate(prompt, max_tokens=200, temperature=0.3, format="json")
    try:
        data = json.loads(resp)
        picks = data.get("picks", [])
        # 转 0-based 索引，校验范围
        indices = []
        for p in picks:
            try:
                idx = int(p) - 1
                if 0 <= idx < len(candidates):
                    indices.append(idx)
            except Exception:
                continue
        return indices[:top_n]
    except Exception:
        # 失败时返回前 top_n 条
        return list(range(min(top_n, len(candidates))))