# webtest/ranking.py
"""
多路召回融合 + 多样性重排 —— 三条推荐路线共用。

为什么需要 RRF
--------------
三路召回（流派索引 / 口味向量 / 关联歌手）的分数**量纲不可比**：
- 流派召回按 popularity 排
- 向量召回是余弦相似度（0~1）
- 歌手扩散按 popularity 排

直接加权求和需要反复调参且随数据类型漂移。RRF（Reciprocal Rank
Fusion）只用"排名"不用"分数"，天然免疫量纲问题，是业界多路融合的
标准做法。

为什么需要 MMR
--------------
纯按相关性排序会导致结果高度同质（同一歌手/同一子流派刷屏）。
MMR（Maximal Marginal Relevance）在"相关性"与"与已选项的差异度"
之间做权衡，并叠加同歌手/同流派硬约束。
"""

import math

# RRF 平滑常数：经典取值 60
RRF_K = 60


def merge_rrf(channels, k=RRF_K, weights=None):
    """
    Reciprocal Rank Fusion 融合多路召回。

    Args:
        channels: list[list[dict]]，每路内部已按各自分数降序排列
        k: 平滑常数
        weights: list[float]，每路权重（默认等权）。可按"这路更可信"
                 给更高权重，例如流派召回 1.0、向量召回 1.0、歌手 0.6。

    Returns:
        list[dict]，已按 rrf_score 降序；每项含 `_rrf_score` 与
        `_channels`（命中了哪几路，便于调试与展示）
    """
    if weights is None:
        weights = [1.0] * len(channels)

    merged = {}
    for ch_idx, channel in enumerate(channels):
        w = weights[ch_idx] if ch_idx < len(weights) else 1.0
        for rank, item in enumerate(channel):
            tid = item.get("track_id")
            if not tid:
                continue
            entry = merged.get(tid)
            if entry is None:
                entry = dict(item)
                entry["_rrf_score"] = 0.0
                entry["_channels"] = []
                merged[tid] = entry
            entry["_rrf_score"] += w * (1.0 / (k + rank + 1))
            # 记录命中来源（用索引标识，调用方负责映射成名字）
            entry["_channels"].append(ch_idx)
            # 用"更靠前的一次"作为展示字段来源
            if rank == 0:
                entry.update(item)

    results = list(merged.values())
    results.sort(key=lambda x: x["_rrf_score"], reverse=True)

    # 归一化到 0~1，让前端相似度百分比有意义
    if results:
        top = results[0]["_rrf_score"]
        if top > 0:
            for r in results:
                r["_rrf_score_norm"] = round(r["_rrf_score"] / top, 6)
    return results


def _cosine(a, b):
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = 0.0
    na = 0.0
    nb = 0.0
    for x, y in zip(a, b):
        x = float(x)
        y = float(y)
        dot += x * y
        na += x * x
        nb += y * y
    if na <= 0 or nb <= 0:
        return 0.0
    return dot / (math.sqrt(na) * math.sqrt(nb))


def diversity_rerank(
    candidates,
    top_k=20,
    lam=0.7,
    max_per_artist=2,
    max_per_genre_idx=6,
    score_key="_final_score",
    vector_key="_vector",
    artist_key="artist_name",
    genre_key="artist_genre_idx",
):
    """
    MMR 多样性重排 + 硬约束。

    Args:
        candidates: 已按相关性降序的候选（需含 score_key 字段）
        top_k: 输出条数
        lam: 相关性权重。1.0 = 纯相关性，0.0 = 纯多样性。
             冷启动应调低（多探索），画像丰富时应调高。
        max_per_artist: 同歌手最多几首
        max_per_genre_idx: 同流派 idx 最多几首
        vector_key: 存放 64 维向量的字段名（缺省则不参与相似度计算，
                    退化为"只有硬约束的贪婪选择"）

    Returns: list[dict]
    """
    if not candidates:
        return []

    # 相关性归一化到 0~1
    scores = [float(c.get(score_key) or 0.0) for c in candidates]
    smin, smax = min(scores), max(scores)
    span = smax - smin

    pool = []
    for c, s in zip(candidates, scores):
        rel = 1.0 if span <= 0 else (s - smin) / span
        pool.append((c, rel))

    selected = []
    selected_vecs = []
    artist_count = {}
    genre_count = {}

    while pool and len(selected) < top_k:
        best_i = -1
        best_value = None

        for i, (cand, rel) in enumerate(pool):
            artist = (cand.get(artist_key) or "").strip()
            gidx = int(cand.get(genre_key) or 0)

            # ---- 硬约束：超限直接跳过 ----
            if artist and artist_count.get(artist, 0) >= max_per_artist:
                continue
            if gidx and genre_count.get(gidx, 0) >= max_per_genre_idx:
                continue

            # ---- 惩罚项：与已选集合的最大相似度 ----
            penalty = 0.0
            vec = cand.get(vector_key)
            if vec and selected_vecs:
                penalty = max(_cosine(vec, sv) for sv in selected_vecs)

            value = lam * rel - (1.0 - lam) * penalty
            if best_value is None or value > best_value:
                best_value = value
                best_i = i

        if best_i < 0:
            # 全部被硬约束卡住 → 逐级放宽，尽力填满 top_k
            if max_per_artist < top_k:
                max_per_artist += 1
                continue
            if max_per_genre_idx < top_k:
                max_per_genre_idx += 1
                continue
            break

        cand, _rel = pool.pop(best_i)
        selected.append(cand)
        vec = cand.get(vector_key)
        if vec:
            selected_vecs.append(vec)
        artist = (cand.get(artist_key) or "").strip()
        if artist:
            artist_count[artist] = artist_count.get(artist, 0) + 1
        gidx = int(cand.get(genre_key) or 0)
        if gidx:
            genre_count[gidx] = genre_count.get(gidx, 0) + 1

    return selected


def random_sample(candidates, k):
    """
    偏置随机取样 —— 用于"每次点开都不同的每日推荐"。

    不是纯随机：候选已按相关性降序，前一半是"高相关"，后一半是"长尾"。
    高相关段占 70% 名额（保证不跑偏），长尾段占 30%（提供探索性），
    最后整体打乱输出顺序。
    """
    import random

    if not candidates:
        return []
    if len(candidates) <= k:
        shuffled = list(candidates)
        random.shuffle(shuffled)
        return shuffled

    half = max(1, len(candidates) // 2)
    hot = candidates[:half]
    rest = candidates[half:]

    picked = []
    seen = set()

    def _take(pool, quota):
        tries = 0
        limit = max(20, quota * 20)
        while len(picked) < quota and tries < limit:
            tries += 1
            c = random.choice(pool)
            tid = c.get("track_id")
            if tid and tid in seen:
                continue
            if tid:
                seen.add(tid)
            picked.append(c)

    hot_quota = int(k * 0.7)
    _take(hot, hot_quota)
    _take(rest, k)

    # 仍不足（去重导致）→ 从全部候选补齐
    if len(picked) < k:
        for c in candidates:
            if len(picked) >= k:
                break
            tid = c.get("track_id")
            if tid and tid in seen:
                continue
            if tid:
                seen.add(tid)
            picked.append(c)

    random.shuffle(picked)
    return picked
