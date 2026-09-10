# webtest/views.py

import json
import re
import time
import requests
from django.http import JsonResponse
from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt
from webtest.embeat_similar import EmbeatSimilar
from webtest.meting_client import enrich_results_with_netease
from webtest.reason_generator import generate_recommend_reason

# ===================== 配置 =====================
QDRANT_URL = "http://localhost:6333"
COLLECTION_NAME = "spotify_tracks"

embeat = EmbeatSimilar(
    qdrant_url="http://127.0.0.1:6333",
    collection_name="spotify_tracks",
)


def safe_get(data, *keys, default=None):
    for key in keys:
        try:
            data = data[key]
        except (KeyError, TypeError, IndexError):
            return default
    return data


# ===================== 页面 =====================
def index(request):
    return render(request, "index.html")


def data_browser(request):
    return render(request, "data_browser.html")


# ===================== 推荐 API =====================
def recommend_api(request):
    track_name = request.GET.get("q", "").strip()
    artist_name = request.GET.get("artist", "").strip()
    top_k = int(request.GET.get("top_k", 20))

    if not track_name:
        return JsonResponse({"error": "请提供 q 参数"}, status=400)

    if not artist_name:
        try:
            resp = requests.post(
                f"{QDRANT_URL}/collections/{COLLECTION_NAME}/points/scroll",
                json={
                    "filter": {"must": [{"key": "track_name", "match": {"text": track_name}}]},
                    "limit": 20,
                    "with_payload": True,
                },
                timeout=10,
            )
            points = resp.json().get("result", {}).get("points", [])
            if not points:
                return JsonResponse({"error": f"未找到歌曲: {track_name}"}, status=404)
            points_sorted = sorted(
                points,
                key=lambda p: float(p["payload"].get("popularity") or 0),
                reverse=True,
            )
            artist_name = points_sorted[0]["payload"].get("artist_name", "")
        except Exception as e:
            return JsonResponse({"error": f"查询失败: {str(e)}"}, status=500)

    result = embeat.recommend(
        track_name=track_name,
        artist_name=artist_name,
        top_k=top_k,
    )

    # 为推荐结果补充网易云跳转链接
    if "results" in result and result["results"]:
        result["results"] = enrich_results_with_netease(result["results"])
        reason = generate_recommend_reason(
            seed=result.get("seed", {}),
            results=result["results"],
        )   
        result["reason"] = reason
    return JsonResponse(result)


# ===================== 数据浏览 API =====================
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