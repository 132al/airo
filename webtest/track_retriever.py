# webtest/track_retriever.py
import requests

QDRANT_URL = "http://127.0.0.1:6333"
COLLECTION = "spotify_tracks"


def _split_genres(genres_str):
    """把 'uk pop, pop, rock' 拆成小写列表"""
    return [g.strip().lower() for g in (genres_str or "").split(",") if g.strip()]


def find_tracks_by_tags(tags, avoid=None, limit=50):
    """
    用标签过滤歌曲。
    - tags: 要匹配的标签（OR 关系）
    - avoid: 要排除的标签关键词
    - 按命中数 + popularity 排序
    """
    if not tags:
        return []

    avoid = [a.lower() for a in (avoid or [])]

    should_conditions = [
        {"key": "artist_genres", "match": {"text": tag}}
        for tag in tags
    ]

    try:
        resp = requests.post(
            f"{QDRANT_URL}/collections/{COLLECTION}/points/scroll",
            json={
                "filter": {"should": should_conditions},
                "limit": limit * 4,  # 多召回，后面过滤
                "with_payload": True,
            },
            timeout=15,
        )
        points = resp.json().get("result", {}).get("points", [])
    except Exception as e:
        print(f"[歌曲过滤] 失败: {e}")
        return []

    tags_lower = [t.lower() for t in tags]

    candidates = []
    for p in points:
        pl = p.get("payload", {})
        genres_str = pl.get("artist_genres", "")
        genres = _split_genres(genres_str)

        # 排除：命中 avoid 里任一关键词
        if avoid:
            if any(a in genres_str.lower() for a in avoid):
                continue

        # 命中数：候选标签有几个在 genres 里
        hit_count = 0
        for t in tags_lower:
            if any(t in g for g in genres):
                hit_count += 1

        if hit_count == 0:
            continue

        candidates.append({
            "track_id": pl.get("track_id", ""),
            "track_name": pl.get("track_name", ""),
            "artist_name": pl.get("artist_name", ""),
            "artist_genres": genres_str,
            "popularity": pl.get("popularity", 0),
            "hit_count": hit_count,
        })

    # 按命中数优先，再按 popularity
    candidates.sort(
        key=lambda x: (x["hit_count"], x["popularity"]),
        reverse=True,
    )

    # 同一歌手最多 3 首
    result = []
    artist_count = {}
    for c in candidates:
        artist = c["artist_name"]
        if artist_count.get(artist, 0) >= 3:
            continue
        artist_count[artist] = artist_count.get(artist, 0) + 1
        result.append(c)
        if len(result) >= limit:
            break

    return result