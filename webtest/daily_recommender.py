# webtest/daily_recommender.py
"""
路线 C：画像 → 歌曲（"一键推荐"，类似网易云每日推荐）。

核心特征
--------
1. 无输入：画像本身就是查询条件。
2. 每次不同：结果经偏置随机取样，刷新就换一批，但不会跑偏。
3. 冷启动可用：完全没有反馈数据时，退化为"按用户级多样性偏置的热门推荐"。

三路召回
--------
① 流派索引召回  artist_genre_idx ∈ 画像流派（走 Qdrant integer 索引，毫秒级）
② 口味向量召回  画像质心（喜欢的歌的 64 维向量加权平均）做 ANN 检索
③ 歌手扩散召回  画像喜欢的歌手 → related_artist_idxs → 找"可能也会喜欢的新歌手"
"""

from qdrant_client import QdrantClient
from qdrant_client.http import models as qdrant_models

from webtest import profile_service
from webtest.ranking import merge_rrf, diversity_rerank, random_sample

QDRANT_URL = "http://127.0.0.1:6333"
COLLECTION = "spotify_tracks"

# 每路召回条数
CHANNEL_LIMIT = 300
ARTIST_CHANNEL_LIMIT = 200

# 流派召回的 popularity 下限（0~1 尺度）
GENRE_MIN_POPULARITY = 0.15

# RRF 各路权重：流派与向量为主，歌手扩散为辅（它更"发散"）
RRF_WEIGHTS = [1.0, 1.0, 0.6]

# 候选池大小（融合后进入重排的数量）
POOL_SIZE = 400

# 冷启动兜底：全局热门召回条数
COLD_POOL_SIZE = 500

_client = None


def _get_client():
    global _client
    if _client is None:
        _client = QdrantClient(url=QDRANT_URL, port=6333, timeout=30)
    return _client


def _payload_to_item(payload, score=0.0):
    """把 Qdrant payload 转成统一的结果条目"""
    return {
        "track_id": payload.get("track_id", ""),
        "track_name": payload.get("track_name", ""),
        "artist_name": payload.get("artist_name", ""),
        "artist_genres": payload.get("artist_genres", ""),
        "artist_genre_idx": payload.get("artist_genre_idx", 0),
        "artist_idx": payload.get("artist_idx", 0),
        "album_name": payload.get("album_name", ""),
        "popularity": payload.get("popularity", 0.0),
        "similarity": round(float(score or 0.0), 4),
    }


# ============================================================
# 通用过滤条件构造
# ============================================================
def _exclusion_clauses(profile, exclude_seen=True):
    """
    构造 must_not 条件：排除不喜欢的流派、不喜欢的歌、已听过的歌。
    这些条件对三路召回统一生效。

    【性能约束】track_id 没有 payload 索引，用 MatchAny 排除已听歌曲会在
    1238 万点上全表扫描。因此这里：
    - 只排除"流派"（artist_genre_idx 有索引，零成本）
    - 不喜欢的具体歌曲：在 Python 侧过滤（候选集只有几百条，可忽略）
    """
    must_not = []

    # 无流派（idx=0 是 <UNK>）噪声大，一律排除
    must_not.append(
        qdrant_models.FieldCondition(
            key="artist_genre_idx",
            match=qdrant_models.MatchValue(value=0),
        )
    )

    for idx in profile_service.to_disliked_genre_idxs(profile):
        must_not.append(
            qdrant_models.FieldCondition(
                key="artist_genre_idx",
                match=qdrant_models.MatchValue(value=idx),
            )
        )

    return must_not


def filter_excluded_tracks(items, profile, exclude_seen=True):
    """
    在 Python 侧剔除"用户已交互过"的歌曲。

    放在这里而不是 Qdrant 过滤，是因为 track_id 无索引（见上）。
    候选集只有几百条，集合判断的开销可忽略。
    """
    if not exclude_seen or not items:
        return items
    banned = set(profile_service.to_seen_track_ids(profile))
    if not banned:
        return items
    return [it for it in items if it.get("track_id") not in banned]


# ============================================================
# 通道①：流派索引召回
# ============================================================
def recall_by_genres(profile, limit=CHANNEL_LIMIT, must_not=None):
    """
    按画像里的流派 idx 做 OR 过滤，走 Qdrant integer 索引。

    这是"最懂用户品味"的一路：命中用户明确喜欢的流派。
    """
    genre_idxs = profile_service.to_genre_idxs(profile, top_n=5)
    if not genre_idxs:
        return []

    must = [
        qdrant_models.FieldCondition(
            key="artist_genre_idx",
            match=qdrant_models.MatchAny(any=genre_idxs),
        ),
        qdrant_models.FieldCondition(
            key="popularity",
            range=qdrant_models.Range(gte=GENRE_MIN_POPULARITY),
        ),
    ]

    try:
        records = _scroll(
            must=must,
            must_not=must_not,
            limit=limit,
            with_vectors=False,
        )
    except Exception as e:
        print(f"[每日推荐] 流派召回失败: {e}")
        return []

    items = []
    for r in records:
        pl = r.payload or {}
        # 权重 = 该流派在画像里的权重（精细排序：最喜欢的流派排更前）
        weight = 0.0
        for g in profile.get("preferred_genres", []):
            if g["idx"] == pl.get("artist_genre_idx"):
                weight = g["weight"]
                break
        item = _payload_to_item(pl, score=weight)
        item["_genre_weight"] = weight
        items.append(item)

    items.sort(key=lambda x: (x["_genre_weight"], x["popularity"]), reverse=True)
    return items


# ============================================================
# 通道②：口味向量召回
# ============================================================
def recall_by_centroid(profile, limit=CHANNEL_LIMIT, must_not=None):
    """
    用"流派锚定质心"做 ANN 检索。

    这一路能捕捉"流派标签之外的声学特征"—— 比如用户就是喜欢
    高能量、明亮、快节奏的听感，无论具体子流派标签是什么。
    质心构造见 profile_service.to_centroid（流派锚定，不依赖 track_id 反查）。
    """
    if not profile or profile.get("stage") == "cold":
        return []

    try:
        centroid = profile_service.to_centroid(profile, _get_client(), COLLECTION)
    except Exception as e:
        print(f"[每日推荐] 质心计算失败: {e}")
        return []

    if not centroid:
        return []

    query_filter = None
    if must_not:
        query_filter = qdrant_models.Filter(must_not=must_not)

    try:
        points = _get_client().query_points(
            collection_name=COLLECTION,
            query=centroid,
            query_filter=query_filter,
            limit=limit,
            with_payload=True,
            with_vectors=False,
        ).points
    except Exception as e:
        print(f"[每日推荐] 向量召回失败: {e}")
        return []

    items = []
    for p in points:
        score = getattr(p, "score", None)
        if score is None:
            score = 0.0
        items.append(_payload_to_item(p.payload or {}, score=score))
    return items


# ============================================================
# 通道③：关联歌手扩散召回
# ============================================================
def recall_by_related_artists(profile, limit=ARTIST_CHANNEL_LIMIT, must_not=None):
    """
    歌手扩散：喜欢的歌手 → related_artist_idxs → 这些"邻居歌手"的歌。

    为什么这样做：related_artist_idxs 是数据自带的 top10 关联歌手（artist_idx），
    而 artist_idx 有 Qdrant integer 索引。

    关键约束：track_id **没有索引**，所以要拿到 seed 歌手的 related_artist_idxs
    不能按 track_id 反查。改为用 artist_name（有全文索引）定位种子 ——
    名字来自画像，数量少（<=5），每次一个 MatchText 条件，走索引很快。
    """
    liked = profile.get("liked_tracks", [])
    if not liked:
        return []

    # 画像里已解析好的 artist_idx 优先（零查询）；否则用 artist_name 走全文索引
    artist_idxs = profile.get("liked_artist_idxs") or []
    neighbor_idxs = set()

    if artist_idxs:
        try:
            records = _scroll(
                must=[qdrant_models.FieldCondition(
                    key="artist_idx",
                    match=qdrant_models.MatchAny(any=list(artist_idxs)[:20]),
                )],
                limit=50,
                with_vectors=False,
            )
        except Exception as e:
            print(f"[每日推荐] artist_idx 查询失败: {e}")
            records = []
        for r in records:
            for idx in ((r.payload or {}).get("related_artist_idxs") or []):
                try:
                    idx = int(idx)
                except (TypeError, ValueError):
                    continue
                if idx:
                    neighbor_idxs.add(idx)

    if not neighbor_idxs:
        # 退化路径：用歌手名（text 索引）定位种子
        names = []
        for t in liked:
            n = (t.get("artist_name") or "").strip()
            if n and n not in names:
                names.append(n)
            if len(names) >= 3:
                break
        for name in names:
            try:
                records = _scroll(
                    must=[qdrant_models.FieldCondition(
                        key="artist_name",
                        match=qdrant_models.MatchText(text=name),
                    )],
                    limit=5,
                    with_vectors=False,
                )
            except Exception as e:
                print(f"[每日推荐] 歌名定位失败 {name}: {e}")
                continue
            for r in records:
                for idx in ((r.payload or {}).get("related_artist_idxs") or []):
                    try:
                        idx = int(idx)
                    except (TypeError, ValueError):
                        continue
                    if idx:
                        neighbor_idxs.add(idx)
            if len(neighbor_idxs) >= 30:
                break

    if not neighbor_idxs:
        return []

    neighbor_idxs = list(neighbor_idxs)[:40]
    must = [
        qdrant_models.FieldCondition(
            key="artist_idx",
            match=qdrant_models.MatchAny(any=neighbor_idxs),
        ),
        qdrant_models.FieldCondition(
            key="popularity",
            range=qdrant_models.Range(gte=GENRE_MIN_POPULARITY),
        ),
    ]

    try:
        records = _scroll(must=must, must_not=must_not, limit=limit, with_vectors=False)
    except Exception as e:
        print(f"[每日推荐] 关联歌手召回失败: {e}")
        return []

    items = [_payload_to_item(r.payload or {}, score=float((r.payload or {}).get("popularity") or 0))
             for r in records]
    items.sort(key=lambda x: x["popularity"], reverse=True)
    return items


# ============================================================
# 冷启动兜底：全局热门 + 用户级多样性
# ============================================================
def recall_cold_start(limit=COLD_POOL_SIZE, must_not=None, min_popularity=0.6):
    """
    没有任何反馈时的兜底：高质量热门池。

    用 popularity 门槛 + 随机取样，保证"每次不同"且不会太冷门。
    """
    must = [
        qdrant_models.FieldCondition(
            key="popularity",
            range=qdrant_models.Range(gte=min_popularity),
        ),
    ]
    try:
        records = _scroll(must=must, must_not=must_not, limit=limit, with_vectors=False)
    except Exception as e:
        print(f"[每日推荐] 冷启动召回失败: {e}")
        return []

    items = [_payload_to_item(r.payload or {}, score=float((r.payload or {}).get("popularity") or 0))
             for r in records]
    items.sort(key=lambda x: x["popularity"], reverse=True)
    return items


# ============================================================
# Qdrant scroll 封装
# ============================================================
def _scroll(must=None, must_not=None, limit=100, with_vectors=False):
    """统一的 scroll 调用（带过滤器组装）"""
    flt = None
    if must or must_not:
        flt = qdrant_models.Filter(must=must or None, must_not=must_not or None)
    records, _ = _get_client().scroll(
        collection_name=COLLECTION,
        scroll_filter=flt,
        limit=limit,
        with_payload=True,
        with_vectors=with_vectors,
    )
    return records


def fetch_by_track_ids(track_ids, with_vectors=False):
    """
    【已废弃 - 请勿使用】

    spotify_tracks 的 track_id 字段 **没有 payload 索引**（索引只有
    track_name / artist_name / artist_idx / artist_genre_idx / popularity / isrc），
    因此任何 track_id 过滤都会在 1238 万点上做全表扫描，实测超时 >30s。

    保留此函数仅为兼容，内部直接返回空列表。请改用：
    - 按 artist_idx / artist_genre_idx 等有索引的字段过滤
    - 或在业务侧把 artist_genre_idx 随反馈一起存下来
    """
    print("[每日推荐] fetch_by_track_ids 已废弃（track_id 无索引，会全表扫描）")
    return []


# ============================================================
# 主入口
# ============================================================
def recommend(profile, top_k=20, refresh=True, lam=None):
    """
    一键推荐主入口。

    Args:
        profile: profile_service.get(user) 返回的画像
        top_k: 返回条数
        refresh: True=偏置随机取样（每次刷新不同，像每日推荐）；
                 False=返回确定性的 MMR 重排结果
        lam: MMR 相关性权重。None 时按画像丰富度自动决定
             （冷启动 lam 低 → 多探索；画像丰富 lam 高 → 更精准）

    Returns:
        dict: {"results": [...], "channels": {...}, "stage": ..., "explore": bool}
    """
    stage = (profile or {}).get("stage", "cold")
    must_not = _exclusion_clauses(profile)

    channel_stats = {"genre": 0, "vector": 0, "artist": 0, "cold": 0}
    channels = []
    weights = []
    names = []

    if stage == "cold":
        # ---------------- 冷启动：热门池 ----------------
        pool = recall_cold_start(must_not=must_not)
        pool = filter_excluded_tracks(pool, profile)
        channel_stats["cold"] = len(pool)
        if not pool:
            return {"results": [], "channels": channel_stats, "stage": stage, "explore": True}

        # 热门池内部先做一次 MMR，保证歌手/流派多样
        pool = diversity_rerank(
            pool, top_k=min(len(pool), POOL_SIZE), lam=0.5,
            max_per_artist=2, max_per_genre_idx=8,
        )
        picked = random_sample(pool, top_k) if refresh else pool[:top_k]
        return {
            "results": _finalize(picked, profile),
            "channels": channel_stats,
            "stage": stage,
            "explore": True,
        }

    # ---------------- 三路召回 ----------------
    ch1 = recall_by_genres(profile, must_not=must_not)
    if ch1:
        channels.append(ch1); weights.append(RRF_WEIGHTS[0]); names.append("genre")
    channel_stats["genre"] = len(ch1)

    ch2 = recall_by_centroid(profile, must_not=must_not)
    if ch2:
        channels.append(ch2); weights.append(RRF_WEIGHTS[1]); names.append("vector")
    channel_stats["vector"] = len(ch2)

    ch3 = recall_by_related_artists(profile, must_not=must_not)
    if ch3:
        channels.append(ch3); weights.append(RRF_WEIGHTS[2]); names.append("artist")
    channel_stats["artist"] = len(ch3)

    if not channels:
        # 画像存在但召回全空（例如流派 idx 在库里没歌）→ 退化为冷启动池
        pool = recall_cold_start(must_not=must_not)
        pool = filter_excluded_tracks(pool, profile)
        channel_stats["cold"] = len(pool)
        if not pool:
            return {"results": [], "channels": channel_stats, "stage": stage, "explore": True}
        picked = random_sample(pool, top_k) if refresh else pool[:top_k]
        return {
            "results": _finalize(picked, profile),
            "channels": channel_stats,
            "stage": stage,
            "explore": True,
        }

    # ---------------- RRF 融合 ----------------
    merged = merge_rrf(channels, weights=weights)
    merged = filter_excluded_tracks(merged, profile)[:POOL_SIZE]

    # 基础分 = RRF 归一化分（0~1），让画像加成有可比尺度
    for m in merged:
        m["final_score"] = m.get("_rrf_score_norm", 0.0)
        m["similarity"] = m.get("_rrf_score_norm", 0.0)

    # ---------------- 画像加成 + 排序 ----------------
    merged = profile_service.apply_personalize_bonus(merged, profile)

    # ---------------- 多样性重排 ----------------
    if lam is None:
        # 信号越多越精准；信号少则多探索
        lam = {"warm": 0.55}.get(stage, 0.75)
    reranked = diversity_rerank(
        merged,
        top_k=min(len(merged), max(top_k * 4, 80)),
        lam=lam,
        max_per_artist=2 if stage == "hot" else 1,
        max_per_genre_idx=6 if stage == "hot" else 4,
    )

    # ---------------- 每次不同 ----------------
    picked = random_sample(reranked, top_k) if refresh else reranked[:top_k]

    return {
        "results": _finalize(picked, profile),
        "channels": channel_stats,
        "stage": stage,
        "explore": False,
    }


def _finalize(items, profile):
    """清理内部字段，保证输出契约与路线 A/B 一致"""
    out = []
    for it in items:
        # 内部专用字段不进 API 响应
        it.pop("_vector", None)
        it.pop("_channels", None)
        it.pop("_genre_weight", None)
        it.pop("_rrf_score", None)
        it.pop("_rrf_score_norm", None)
        it.setdefault("similarity", 0.0)
        if it["similarity"] is None:
            it["similarity"] = 0.0
        out.append(it)
    return out


def build_seed(profile):
    """构造前端展示用的 seed 信息"""
    names = profile_service.to_genre_names(profile, top_n=5)
    stage = (profile or {}).get("stage", "cold")
    label = {
        "cold": "为你探索",
        "warm": "为你推荐",
        "hot": "专属推荐",
    }.get(stage, "为你推荐")
    return {
        "track_name": label,
        "artist_name": "每日推荐",
        "artist_genres": ", ".join(names) if names else "热门精选",
        "lastfm_tags": [],
    }
