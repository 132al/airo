# webtest/views.py

import json
import math
import re
import time

import requests
from django.http import JsonResponse
from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt

# ===================== 配置 =====================
QDRANT_URL = "http://localhost:6333"
COLLECTION_NAME = "spotify_songs_test"

MIN_SIMILARITY = 0.60
MAX_ARTIST_COUNT = 3
DEFAULT_TOP_K = 20
DEFAULT_RECALL_MULTIPLIER = 3

# ===================== 辅助函数 =====================
def clean_track_name(name):
    """去除括号、方括号、' - ' 后缀"""
    if not name:
        return ""
    name = re.sub(r"\s*\([^)]*\)", "", name)
    name = re.sub(r"\s*\[[^\]]*\]", "", name)
    name = re.sub(r"\s*-\s*[^-]+$", "", name)
    return name.strip()


def safe_get(data, *keys, default=None):
    for key in keys:
        try:
            data = data[key]
        except (KeyError, TypeError, IndexError):
            return default
    return data


def parse_genres(genre_str):
    """
    解析流派字符串，返回流派列表
    支持格式: "j-pop,anime,j-rock" 或 "j-pop; anime; j-rock"
    """
    if not genre_str:
        return []
    # 用逗号或分号分割
    genres = re.split(r'[,;]\s*', genre_str)
    # 过滤空字符串并去重
    return [g.strip() for g in genres if g.strip()]


def build_genre_filter(genres, mode="match_any"):
    """
    构建 Qdrant 的流派过滤条件
    
    Args:
        genres: 流派列表，如 ['j-pop', 'anime', 'j-rock']
        mode: 
            - "match_any": 命中任意一个即可（宽松，推荐）
            - "match_all": 必须命中全部（严格）
            - "fuzzy": 模糊匹配（包含关键词即可）
    
    Returns:
        Qdrant filter 条件字典
    """
    if not genres:
        return None
    
    if mode == "match_all":
        # 必须同时包含所有流派（几乎不会命中，不推荐）
        return {"must": [{"key": "artist_genres", "match": {"text": g}} for g in genres]}
    
    elif mode == "fuzzy":
        # 模糊匹配：只要包含任意一个关键词即可
        # 注意：Qdrant 的 text match 本身就是模糊的，这里用 should 包装
        return {"should": [{"key": "artist_genres", "match": {"text": g}} for g in genres], "minimum_should_match": 1}
    
    else:  # "match_any" (默认)
        # 标准多流派匹配：命中任意一个
        return {"should": [{"key": "artist_genres", "match": {"text": g}} for g in genres], "minimum_should_match": 1}


# ===================== 视图 =====================
def index(request):
    return render(request, "index.html")


def data_browser(request):
    return render(request, "data_browser.html")


def recommend_api(request):
    # ----- 1. 参数解析 -----
    track_name = request.GET.get("q", "").strip()
    top_k = int(request.GET.get("top_k", DEFAULT_TOP_K))
    min_popularity = float(request.GET.get("min_popularity", 0.0))
    year_min = int(request.GET.get("year_min", 0))
    year_max = int(request.GET.get("year_max", 3000))
    
    weight_similarity = float(request.GET.get("weight_similarity", 0.5))
    weight_popularity = float(request.GET.get("weight_popularity", 0.5))
    weight_diversity = float(request.GET.get("weight_diversity", 0.3))
    
    genre_mode = request.GET.get("genre_mode", "match_any")
    max_genres = int(request.GET.get("max_genres", 3))

    total_weight = weight_similarity + weight_popularity
    if total_weight == 0:
        weight_similarity = 0.5
        weight_popularity = 0.5
    else:
        weight_similarity = weight_similarity / total_weight
        weight_popularity = weight_popularity / total_weight
    
    max_artist_count = max(1, min(3, int(3 - weight_diversity * 2)))

    if not track_name:
        return JsonResponse({"error": "请输入歌曲名"}, status=400)

    start_time = time.time()

    # ----- 2. 查找种子歌曲 -----
    scroll_payload = {
        "filter": {"must": [{"key": "track_name", "match": {"text": track_name}}]},
        "limit": 1,
        "with_payload": True,
        "with_vector": True,
    }

    try:
        resp = requests.post(
            f"{QDRANT_URL}/collections/{COLLECTION_NAME}/points/scroll",
            json=scroll_payload,
            timeout=10,
        )
        resp.raise_for_status()
        points = safe_get(resp.json(), "result", "points", default=[])
    except Exception as e:
        return JsonResponse({"error": f"查询失败: {str(e)}"}, status=500)

    if not points:
        return JsonResponse({"error": f"未找到歌曲: {track_name}"}, status=404)

    seed = points[0]
    seed_vector = seed.get("vector")
    seed_payload = seed.get("payload", {})
    seed_track = seed_payload.get("track_name", track_name)
    seed_artist = seed_payload.get("artist_name", "")
    seed_genre = seed_payload.get("artist_genres", "")
    seed_popularity = seed_payload.get("popularity", 50.0)

    if not seed_vector:
        return JsonResponse({"error": "歌曲向量为空"}, status=500)

    all_genres = parse_genres(seed_genre)
    main_genres = all_genres[:max_genres] if all_genres else []
    print(f"[流派] 种子: {seed_track} - {seed_artist}")
    print(f"[流派] 使用: {main_genres}")

    # ----- 3. 构建 Qdrant 过滤条件（只用流行度和年份）-----
    must_conditions = []

    # 流行度过滤
    if min_popularity > 0:
        must_conditions.append({"key": "popularity", "range": {"gte": min_popularity}})

    # 年份过滤
    if year_min > 0 or year_max < 3000:
        must_conditions.append({
            "key": "release_year",
            "range": {
                "gte": year_min if year_min > 0 else 0,
                "lte": year_max if year_max < 3000 else 3000,
            },
        })

    # ----- 4. 执行向量搜索（不限流派，召回 200 条）-----
    search_payload = {
        "vector": seed_vector,
        "limit": 200,  # 多召回一些，方便应用层过滤
        "with_payload": True,
        "with_vector": True,
    }

    if must_conditions:
        search_payload["filter"] = {"must": must_conditions}
        print(f"[过滤] 流行度: {min_popularity}, 年份: {year_min}-{year_max}")

    print(f"[搜索] 召回 200 条（不限流派）")

    try:
        resp = requests.post(
            f"{QDRANT_URL}/collections/{COLLECTION_NAME}/points/search",
            json=search_payload,
            timeout=30,
        )
        resp.raise_for_status()
        hits = safe_get(resp.json(), "result", default=[])
    except Exception as e:
        return JsonResponse({"error": f"搜索失败: {str(e)}"}, status=500)

    print(f"[召回] 获取 {len(hits)} 条候选")

    # ----- 5. 应用层流派过滤 + 去重 + 排序 -----
    import numpy as np
    from sklearn.metrics.pairwise import cosine_similarity

    seed_np = np.array(seed_vector).reshape(1, -1)
    seen_artists = {}
    seen_tracks = set()
    candidate_pool = []

    all_popularities = [seed_popularity]
    for hit in hits:
        payload = hit.get("payload", {})
        pop = payload.get("popularity", 0.0)
        if pop > 0:
            all_popularities.append(pop)
    
    max_pop = max(all_popularities) if all_popularities else 100
    min_pop = min(all_popularities) if all_popularities else 0
    pop_range = max_pop - min_pop if max_pop != min_pop else 1

    genre_matched_count = 0
    genre_unmatched_count = 0

    for hit in hits:
        payload = hit.get("payload", {})
        qdrant_score = hit.get("score", 0.0)
        hit_vector = hit.get("vector")

        track = payload.get("track_name", "")
        artist = payload.get("artist_name", "")
        if not track or not artist:
            continue

        if track == seed_track and artist == seed_artist:
            continue

        # 手动计算相似度
        manual_sim = qdrant_score
        if hit_vector is not None and len(hit_vector) == len(seed_vector):
            try:
                v2 = np.array(hit_vector).reshape(1, -1)
                manual_sim = float(cosine_similarity(seed_np, v2)[0][0])
            except Exception:
                pass

        # ===== 应用层流派过滤 =====
        track_genre = payload.get("artist_genres", "").lower()
        is_genre_match = False
        matched_genres = []
        
        if genre_mode != "none" and main_genres:
            for g in main_genres:
                if g.lower() in track_genre:
                    is_genre_match = True
                    matched_genres.append(g)
            if not is_genre_match:
                genre_unmatched_count += 1
                continue  # 跳过流派不匹配的
            else:
                genre_matched_count += 1
        else:
            is_genre_match = True

        # 去重
        clean_track = clean_track_name(track)
        key = (clean_track, artist)
        if key in seen_tracks:
            continue
        seen_tracks.add(key)

        if artist not in seen_artists:
            seen_artists[artist] = 0
        if seen_artists[artist] >= max_artist_count:
            continue
        seen_artists[artist] += 1

        # 计算综合分数
        popularity = payload.get("popularity", 0.0)
        normalized_pop = (popularity - min_pop) / pop_range if pop_range > 0 else 0.5
        combined_score = weight_similarity * manual_sim + weight_popularity * normalized_pop

        candidate_pool.append({
            "track_name": track,
            "artist_name": artist,
            "artist_genres": payload.get("artist_genres", ""),
            "popularity": popularity,
            "similarity": round(manual_sim, 4),
            "normalized_pop": round(normalized_pop, 4),
            "combined_score": round(combined_score, 4),
            "matched_genres": matched_genres,
            "genre_match": is_genre_match,
        })

    print(f"[过滤] 流派匹配: {genre_matched_count} 条, 未匹配: {genre_unmatched_count} 条")

    # 如果匹配的太少（<5），放宽限制，使用所有候选
    if genre_matched_count < 5 and genre_mode != "none":
        print(f"[放宽] 匹配结果太少 ({genre_matched_count} 条)，使用全部候选")
        # 重新构建候选池（不过滤流派）
        candidate_pool = []
        seen_tracks = set()
        seen_artists = {}
        
        for hit in hits:
            payload = hit.get("payload", {})
            qdrant_score = hit.get("score", 0.0)
            hit_vector = hit.get("vector")

            track = payload.get("track_name", "")
            artist = payload.get("artist_name", "")
            if not track or not artist:
                continue

            if track == seed_track and artist == seed_artist:
                continue

            manual_sim = qdrant_score
            if hit_vector is not None and len(hit_vector) == len(seed_vector):
                try:
                    v2 = np.array(hit_vector).reshape(1, -1)
                    manual_sim = float(cosine_similarity(seed_np, v2)[0][0])
                except Exception:
                    pass

            clean_track = clean_track_name(track)
            key = (clean_track, artist)
            if key in seen_tracks:
                continue
            seen_tracks.add(key)

            if artist not in seen_artists:
                seen_artists[artist] = 0
            if seen_artists[artist] >= max_artist_count:
                continue
            seen_artists[artist] += 1

            popularity = payload.get("popularity", 0.0)
            normalized_pop = (popularity - min_pop) / pop_range if pop_range > 0 else 0.5
            combined_score = weight_similarity * manual_sim + weight_popularity * normalized_pop

            candidate_pool.append({
                "track_name": track,
                "artist_name": artist,
                "artist_genres": payload.get("artist_genres", ""),
                "popularity": popularity,
                "similarity": round(manual_sim, 4),
                "normalized_pop": round(normalized_pop, 4),
                "combined_score": round(combined_score, 4),
                "matched_genres": [],
                "genre_match": False,
            })

    # 排序
    candidate_pool.sort(key=lambda x: x["combined_score"], reverse=True)
    final_results = candidate_pool[:top_k]

    elapsed = time.time() - start_time
    print(f"[性能] {elapsed:.2f}s, 召回 {len(hits)} 条, 候选 {len(candidate_pool)} 条, 返回 {len(final_results)} 条")

    return JsonResponse({
        "query": track_name,
        "seed": {
            "track": seed_track,
            "artist": seed_artist,
            "genre": seed_genre,
            "main_genres": main_genres,
            "all_genres": all_genres,
            "popularity": seed_popularity,
        },
        "filters": {
            "genre_mode": genre_mode,
            "genres_used": main_genres,
            "min_popularity": min_popularity,
            "year_range": f"{year_min}-{year_max}" if (year_min > 0 or year_max < 3000) else "不限",
        },
        "stats": {
            "recalled": len(hits),
            "genre_matched": genre_matched_count,
            "genre_unmatched": genre_unmatched_count,
            "candidates": len(candidate_pool),
            "returned": len(final_results),
        },
        "results": final_results,
    })


# ===================== 理由生成 =====================
@csrf_exempt
def reason_api(request):
    if request.method != "POST":
        return JsonResponse({"error": "POST required"}, status=405)

    try:
        data = json.loads(request.body)
        seed = data.get("seed", "这首歌")
        tracks = data.get("tracks", [])

        reasons = []
        for t in tracks[:10]:
            name = t.get("track_name", "未知歌曲")
            sim = t.get("similarity", 0)
            pop = t.get("popularity", 0)
            genres = t.get("artist_genres", "")
            matched = t.get("matched_genres", [])
            match_info = f"，命中流派: {', '.join(matched)}" if matched else ""
            
            reasons.append(
                f"《{name}》与《{seed}》声学相似度 {sim:.1%}，"
                f"流行度 {pop:.0f}，综合评分 {t.get('combined_score', 0):.3f}{match_info}。"
            )

        return JsonResponse({"reasons": reasons})

    except Exception as e:
        return JsonResponse({"error": f"生成理由失败: {str(e)}"}, status=400)


# ===================== 反馈收集 =====================
@csrf_exempt
def feedback_api(request):
    if request.method != "POST":
        return JsonResponse({"error": "POST required"}, status=405)

    try:
        data = json.loads(request.body)
        print("[反馈] 收到反馈:", data)
        return JsonResponse({"status": "ok"})

    except Exception as e:
        return JsonResponse({"error": f"处理反馈失败: {str(e)}"}, status=400)


# ===================== 数据浏览 =====================
def data_api(request):
    page = int(request.GET.get("page", 1))
    limit = int(request.GET.get("limit", 20))
    track_name = request.GET.get("q", "").strip()
    artist_name = request.GET.get("artist", "").strip()
    genre = request.GET.get("genre", "").strip()

    must_conditions = []
    if track_name:
        must_conditions.append({"key": "track_name", "match": {"text": track_name}})
    if artist_name:
        must_conditions.append({"key": "artist_name", "match": {"text": artist_name}})
    if genre:
        must_conditions.append({"key": "artist_genres", "match": {"text": genre}})

    count_payload = {}
    if must_conditions:
        count_payload["filter"] = {"must": must_conditions}

    try:
        resp = requests.post(
            f"{QDRANT_URL}/collections/{COLLECTION_NAME}/points/count",
            json=count_payload,
            timeout=10,
        )
        total = safe_get(resp.json(), "result", "count", default=0)
    except Exception:
        total = 0

    total_pages = (total + limit - 1) // limit if limit > 0 else 1

    scroll_payload = {
        "limit": limit,
        "offset": (page - 1) * limit,
        "with_payload": True,
    }
    if must_conditions:
        scroll_payload["filter"] = {"must": must_conditions}

    try:
        resp = requests.post(
            f"{QDRANT_URL}/collections/{COLLECTION_NAME}/points/scroll",
            json=scroll_payload,
            timeout=30,
        )
        resp.raise_for_status()
        points = safe_get(resp.json(), "result", "points", default=[])
    except Exception as e:
        return JsonResponse({"error": str(e)}, status=500)

    results = []
    for point in points:
        payload = point.get("payload", {})
        results.append({
            "track_id": payload.get("track_id", ""),
            "track_name": payload.get("track_name", "未知"),
            "artist_name": payload.get("artist_name", "未知"),
            "popularity": payload.get("popularity", 0),
            "artist_genres": payload.get("artist_genres", ""),
            "album_name": payload.get("album_name", ""),
        })

    return JsonResponse({
        "results": results,
        "total": total,
        "page": page,
        "limit": limit,
        "total_pages": total_pages,
    })