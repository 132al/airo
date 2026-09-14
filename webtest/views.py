# webtest/views.py

import json
import re
import time
import requests
from django.http import JsonResponse
from django.shortcuts import render, redirect
from django.views.decorators.csrf import csrf_exempt
from django.contrib.auth import login as auth_login, logout as auth_logout
from django.contrib.auth.models import User
from webtest.embeat_similar import EmbeatSimilar
from webtest.meting_client import enrich_results_with_netease
from webtest.reason_generator import generate_recommend_reason
from webtest.forms import RegisterForm, LoginForm
from webtest.models import UserProfile

# ===================== 配置 =====================
QDRANT_URL = "http://localhost:6333"
COLLECTION_NAME = "spotify_tracks"

embeat = EmbeatSimilar(
    qdrant_url="http://127.0.0.1:6333",
    collection_name="spotify_tracks",
)


# ===================== 登录系统 =====================
def login_view(request):
    if request.user.is_authenticated:
        return redirect("/")

    if request.method == "POST":
        form = LoginForm(request, data=request.POST)
        if form.is_valid():
            user = form.get_user()
            auth_login(request, user)
            next_url = request.GET.get("next") or "/"
            return redirect(next_url)
    else:
        form = LoginForm(request)

    return render(request, "login.html", {"form": form})


def register_view(request):
    if request.user.is_authenticated:
        return redirect("/")

    if request.method == "POST":
        form = RegisterForm(request.POST)
        if form.is_valid():
            user = User.objects.create_user(
                username=form.cleaned_data["username"],
                password=form.cleaned_data["password1"],
            )
            auth_login(request, user)
            return redirect("/")
    else:
        form = RegisterForm()

    return render(request, "register.html", {"form": form})


def logout_view(request):
    auth_logout(request)
    return redirect("/login/")


# ===================== 页面 =====================
def index(request):
    return render(request, "index.html")


def data_browser(request):
    return render(request, "data_browser.html")


def feedback_page(request):
    return render(request, "feedback.html")


# ===================== 工具函数 =====================
def safe_get(data, *keys, default=None):
    for key in keys:
        try:
            data = data[key]
        except (KeyError, TypeError, IndexError):
            return default
    return data


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

    user = request.user
    result = embeat.recommend(track_name=track_name, artist_name=artist_name, top_k=top_k)

    if "results" in result and result["results"]:
        result["results"] = enrich_results_with_netease(result["results"])

        # 根据用户档案重排
        profile = UserProfile.objects.filter(user=user).first()
        if profile:
            try:
                profile_data = json.loads(profile.profile_json or "{}")
                if profile_data:
                    result["results"] = personalize_results(result["results"], profile_data)
                    result["user_summary"] = profile_data.get("llm_summary", "")
            except Exception:
                pass

        # 附加当前用户对每首歌的已有反馈
        attach_user_feedback(result["results"], user)

        result["reason"] = ""

    return JsonResponse(result)


def attach_user_feedback(results, user):
    """给每条结果附加当前用户的反馈状态"""
    from webtest.models import UserFeedback
    track_ids = [r.get("track_id", "") for r in results if r.get("track_id")]
    if not track_ids:
        return
    rows = UserFeedback.objects.filter(user=user, track_id__in=track_ids)
    mapping = {row.track_id: row.feedback for row in rows}
    for r in results:
        r["_feedback"] = mapping.get(r.get("track_id", ""), "")


def personalize_results(results, profile_data):
    """根据用户档案重排推荐结果"""
    preferred_genres = profile_data.get("preferred_genres", {})
    disliked_genres = profile_data.get("disliked_genres", {})
    preferred_artists = profile_data.get("preferred_artists", {})

    max_pref = max(preferred_genres.values()) if preferred_genres else 1
    max_bad = max(disliked_genres.values()) if disliked_genres else 1
    max_artist = max(preferred_artists.values()) if preferred_artists else 1

    for r in results:
        bonus = 0.0
        genres = [g.strip().lower() for g in r.get("artist_genres", "").split(",") if g.strip()]
        artist = r.get("artist_name", "")

        for g in genres:
            if g in preferred_genres:
                bonus += (preferred_genres[g] / max_pref) * 0.05
            if g in disliked_genres:
                bonus -= (disliked_genres[g] / max_bad) * 0.05

        if artist in preferred_artists:
            bonus += (preferred_artists[artist] / max_artist) * 0.08

        r["_personal_bonus"] = round(bonus, 4)
        r["_final_score"] = r["similarity"] + bonus

    results.sort(key=lambda x: x["_final_score"], reverse=True)
    return results


# ===================== 理由 API =====================
@csrf_exempt
def reason_api(request):
    """为单首歌生成理由"""
    if request.method != "POST":
        return JsonResponse({"error": "POST required"}, status=405)

    try:
        data = json.loads(request.body)
        seed_track = data.get("seed_track", "")
        seed_artist = data.get("seed_artist", "")
        seed_genres = data.get("seed_genres", "")

        rec_track = data.get("track_name", "")
        rec_artist = data.get("artist_name", "")
        rec_genres = data.get("artist_genres", "")
        sim = float(data.get("similarity", 0))

        # 拿当前用户的档案
        user_summary = ""
        preferred_genres = {}
        liked_artists = {}
        profile = UserProfile.objects.filter(user=request.user).first()
        if profile:
            try:
                profile_data = json.loads(profile.profile_json or "{}")
                user_summary = profile_data.get("llm_summary", "")
                preferred_genres = profile_data.get("preferred_genres", {})
                liked_artists = profile_data.get("preferred_artists", {})
            except Exception:
                pass

        from webtest.reason_generator import generate_song_reason
        reason = generate_song_reason(
            seed_track=seed_track,
            seed_artist=seed_artist,
            seed_genres=seed_genres,
            rec_track=rec_track,
            rec_artist=rec_artist,
            rec_genres=rec_genres,
            similarity=sim,
            user_summary=user_summary,
            preferred_genres=preferred_genres,
            liked_artists=liked_artists,
        )

        return JsonResponse({"reason": reason})
    except Exception as e:
        return JsonResponse({"error": str(e)}, status=400)


# ===================== 反馈 API =====================
@csrf_exempt
def feedback_api(request):
    """收集用户反馈：like / dislike / undo"""
    if request.method != "POST":
        return JsonResponse({"error": "POST required"}, status=405)

    try:
        from webtest.models import UserProfile, UserFeedback
        from webtest.profile_analyzer import rebuild_profile

        data = json.loads(request.body)
        feedback = data.get("feedback", "")
        track_id = data.get("track_id", "")

        if feedback not in ("like", "dislike", "undo"):
            return JsonResponse({"error": "feedback 必须是 like/dislike/undo"}, status=400)

        if not track_id:
            return JsonResponse({"error": "track_id 不能为空"}, status=400)

        user = request.user

        if feedback == "undo":
            UserFeedback.objects.filter(user=user, track_id=track_id).delete()
        else:
            UserFeedback.objects.update_or_create(
                user=user,
                track_id=track_id,
                defaults={
                    "track_name": data.get("track_name", ""),
                    "artist_name": data.get("artist_name", ""),
                    "artist_genres": data.get("artist_genres", ""),
                    "feedback": feedback,
                },
            )

        # 重算档案
        profile, _ = UserProfile.objects.get_or_create(user=user)
        profile_data = rebuild_profile(user)
        profile.profile_json = json.dumps(profile_data, ensure_ascii=False)
        profile.save()

        return JsonResponse({"status": "ok"})
    except Exception as e:
        return JsonResponse({"error": str(e)}, status=400)


def feedback_list_api(request):
    """查询当前用户的反馈数据"""
    from webtest.models import UserFeedback, UserProfile

    feedback_type = request.GET.get("feedback", "").strip()
    limit = int(request.GET.get("limit", 100))

    user = request.user
    queryset = UserFeedback.objects.filter(user=user).order_by("-created_at")

    if feedback_type in ("like", "dislike"):
        queryset = queryset.filter(feedback=feedback_type)

    total = queryset.count()
    items = list(queryset[:limit])

    results = []
    for fb in items:
        results.append({
            "id": fb.id,
            "user_id": fb.user.username,
            "track_name": fb.track_name,
            "artist_name": fb.artist_name,
            "artist_genres": fb.artist_genres,
            "feedback": fb.feedback,
            "created_at": fb.created_at.strftime("%Y-%m-%d %H:%M:%S"),
        })

    profile_data = None
    profile = UserProfile.objects.filter(user=user).first()
    if profile:
        try:
            profile_data = json.loads(profile.profile_json or "{}")
        except Exception:
            profile_data = None

    return JsonResponse({
        "total": total,
        "results": results,
        "profile": profile_data,
    })


# ===================== 数据浏览 API =====================
def data_api(request):
    limit = int(request.GET.get("limit", 20))
    cursor = request.GET.get("cursor", "").strip()
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

    # 总数
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

    # 翻页
    scroll_payload = {
        "limit": limit,
        "with_payload": True,
    }
    if cursor:
        scroll_payload["offset"] = cursor
    if must_conditions:
        scroll_payload["filter"] = {"must": must_conditions}

    try:
        resp = requests.post(
            f"{QDRANT_URL}/collections/{COLLECTION_NAME}/points/scroll",
            json=scroll_payload,
            timeout=30,
        )
        resp.raise_for_status()
        result = resp.json().get("result", {})
        points = result.get("points", [])
        next_cursor = result.get("next_page_offset")
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
        "cursor": cursor,
        "next_cursor": next_cursor,
        "limit": limit,
    })


def seed_candidates_api(request):
    q = request.GET.get("q", "").strip()
    artist = request.GET.get("artist", "").strip()

    if not q:
        return JsonResponse({"candidates": []})

    candidates = embeat.find_candidates(track_name=q, artist_name=artist, limit=10)
    return JsonResponse({"candidates": candidates})
