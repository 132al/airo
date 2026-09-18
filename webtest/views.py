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
from webtest.meting_client import search_netease
from webtest.forms import RegisterForm, LoginForm
from webtest.tag_retriever import multi_recall, select_tags
from webtest.track_retriever import find_tracks_by_tags
from webtest import profile_service
from webtest import daily_recommender
from webtest.ranking import diversity_rerank
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


def safe_int(value, default=20, lo=1, hi=50):
    """
    安全地把请求参数转成受约束的整数。

    修复：原先直接 int(request.GET.get("top_k", 20))，
    遇到 ?top_k=abc 会抛 ValueError 导致 HTTP 500。
    """
    try:
        n = int(value)
    except (TypeError, ValueError):
        return default
    return max(lo, min(hi, n))


# ===================== 推荐 API =====================
def recommend_api(request):
    track_name = request.GET.get("q", "").strip()
    artist_name = request.GET.get("artist", "").strip()
    top_k = safe_int(request.GET.get("top_k"), default=20, lo=1, hi=50)

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
        # 【不再在此处补网易云链接】
        # 原同步补链会逐条查询网易云（Meting 串行），实测单请求最高 5.9 秒。
        # 现改为：列表页不查询，用户点开详情卡片时前端再按需调
        # POST /api/netease_link/ 获取这一首的链接。
        for r in result["results"]:
            r["netease_url"] = ""
            r["netease_id"] = ""

        # 画像：统一走 profile_service（画像独立化后三条路线共用）
        # ensure_llm=False：推荐路径不需要 LLM 画像摘要（前端不展示），
        # 避免为它白白多跑一次模型推理。
        profile = profile_service.get(user, ensure_llm=False)
        if profile.get("stage") != "cold":
            result["results"] = profile_service.apply_personalize_bonus(
                result["results"], profile
            )

            # 多样性重排：避免同一歌手/同一子流派刷屏
            result["results"] = diversity_rerank(
                result["results"],
                top_k=top_k,
                lam=0.85,
                max_per_artist=2,
                max_per_genre_idx=8,
                score_key="_final_score",
                vector_key="_vector",
            )
            # 前端展示用相似度（保留原始声学相似度，不掺杂画像加分）
            for r in result["results"]:
                r["similarity"] = r.get("similarity") or 0.0

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
    """
    兼容包装：画像重排逻辑已迁移到 profile_service。

    保留此函数是为了不破坏可能的外部调用；新代码请直接用
    profile_service.apply_personalize_bonus()。

    注意：若传入的是旧的扁平格式画像（preferred_genres 为 dict），
    这里会先归一化再处理。
    """
    profile = _normalize_profile(profile_data)
    return profile_service.apply_personalize_bonus(results, profile)


def _normalize_profile(profile_data):
    """把旧格式画像（dict 计数）转成新格式，避免直接调用时报错"""
    if not profile_data:
        return profile_service.empty_profile()
    if profile_data.get("schema_version") == profile_service.PROFILE_SCHEMA_VERSION:
        return profile_data

    def _to_list(d, idx_getter):
        if not isinstance(d, dict):
            return []
        out = []
        for name, weight in d.items():
            out.append({"idx": idx_getter(name), "name": name, "weight": float(weight)})
        out.sort(key=lambda x: -x["weight"])
        return out

    return {
        "schema_version": profile_service.PROFILE_SCHEMA_VERSION,
        "stage": "hot",
        "signal_count": profile_data.get("signal_count", 0),
        "liked_count": len(profile_data.get("liked_tracks", [])),
        "preferred_genres": _to_list(profile_data.get("preferred_genres", {}),
                                     profile_service.genre_idx_of),
        "disliked_genres": _to_list(profile_data.get("disliked_genres", {}),
                                    profile_service.genre_idx_of),
        "preferred_artists": [
            {"name": n, "weight": float(w)}
            for n, w in (profile_data.get("preferred_artists", {}) or {}).items()
        ],
        "centroid_available": False,
        "query_text": "",
        "llm_summary": profile_data.get("llm_summary", ""),
        "liked_tracks": profile_data.get("liked_tracks", []),
        "disliked_tracks": profile_data.get("disliked_tracks", []),
    }


# ===================== 网易云链接 API（按需获取） =====================
@csrf_exempt
def netease_link_api(request):
    """
    按需获取单首歌的网易云跳转链接。

    【为什么改成按需】
    原先三条推荐路线都会同步调用 enrich_results_with_netease()，
    对整批结果逐条查询网易云。由于 Meting MCP 客户端是"全局单例 +
    全局锁"，所有查询实际串行执行，导致：
      - 歌名找相似：单请求最高 5.9 秒
      - 每日推荐：P50 约 1.7 秒
      - 用户真正点进详情的可能只有 1-2 首，其余查询纯属浪费

    现在改为：**列表页不查询，只有打开歌曲详情卡片时才查这一首**。
    收益：
      - 推荐接口耗时下降一个数量级
      - 网易云查询量下降到原来的 1/20
    """
    if request.method != "POST":
        return JsonResponse({"error": "POST required"}, status=405)

    try:
        data = json.loads(request.body)
        track_name = (data.get("track_name") or "").strip()
        artist_name = (data.get("artist_name") or "").strip()

        if not track_name:
            return JsonResponse({"error": "track_name 不能为空"}, status=400)

        from webtest.meting_client import search_netease
        info = search_netease(track_name, artist_name)

        if info:
            return JsonResponse({
                "netease_url": info.get("netease_url", ""),
                "netease_id": info.get("netease_id", ""),
            })
        return JsonResponse({"netease_url": "", "netease_id": "", "found": False})
    except Exception as e:
        return JsonResponse({"error": str(e)}, status=400)


# ===================== 理由 API =====================
@csrf_exempt
def reason_api(request):
    """为单首歌生成理由"""
    if request.method != "POST":
        return JsonResponse({"error": "POST required"}, status=405)

    try:
        data = json.loads(request.body)

        # 推荐来源决定提示词组织方式：
        #   seed    -> 歌名找相似（有种子歌）
        #   profile -> 每日推荐（无种子歌，讲口味契合）
        #   intent  -> 描述找歌（无种子歌，讲与描述的匹配）
        mode = data.get("mode", "seed")
        if mode not in ("seed", "profile", "intent"):
            mode = "seed"

        seed_track = data.get("seed_track", "")
        seed_artist = data.get("seed_artist", "")
        seed_genres = data.get("seed_genres", "")
        intent_query = data.get("intent_query", "")

        rec_track = data.get("track_name", "")
        rec_artist = data.get("artist_name", "")
        rec_genres = data.get("artist_genres", "")
        sim = float(data.get("similarity") or 0)

        # 画像：走 profile_service 统一结构。
        # ensure_llm=False：理由生成不需要画像摘要（避免多跑一次 LLM），
        # 但需要 preferred_genres / preferred_artists 作为事实依据。
        prof = profile_service.get(request.user, ensure_llm=False)
        preferred_genres = prof.get("preferred_genres", [])
        liked_artists = [a.get("name") for a in prof.get("preferred_artists", [])]

        from webtest.reason_generator import generate_song_reason
        reason = generate_song_reason(
            rec_track=rec_track,
            rec_artist=rec_artist,
            rec_genres=rec_genres,
            similarity=sim,
            mode=mode,
            seed_track=seed_track,
            seed_artist=seed_artist,
            seed_genres=seed_genres,
            intent_query=intent_query,
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
        from webtest.models import UserFeedback

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
                    "artist_idx": int(data.get("artist_idx") or 0),
                    "artist_genre_idx": int(data.get("artist_genre_idx") or 0),
                },
            )

        # 重算画像：纯统计 + 时间衰减，毫秒级，不调 LLM
        # （LLM 摘要由 profile_service.get(ensure_llm=True) 惰性补生成）
        profile = profile_service.rebuild_profile(user)

        return JsonResponse({
            "status": "ok",
            "stage": profile.get("stage", "cold"),
            "signal_count": profile.get("signal_count", 0),
        })
    except Exception as e:
        return JsonResponse({"error": str(e)}, status=400)


def feedback_list_api(request):
    """查询当前用户的反馈数据"""
    from webtest.models import UserFeedback

    feedback_type = request.GET.get("feedback", "").strip()
    limit = safe_int(request.GET.get("limit"), default=100, lo=1, hi=1000)

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
            "track_id": fb.track_id,
            "track_name": fb.track_name,
            "artist_name": fb.artist_name,
            "artist_genres": fb.artist_genres,
            "feedback": fb.feedback,
            "created_at": fb.created_at.strftime("%Y-%m-%d %H:%M:%S"),
        })

    # 画像：统一走 profile_service（含流派 idx 与冷启动分档）
    #
    # 【性能】ensure_llm=False —— 本页不再同步调用 LLM。
    # 原因：原来 ensure_llm=True 时，若 llm_summary 为空且 LLM 不可用，
    # 会在请求中阻塞最长 3×120s，导致"页面一直加载中"，还会堵死
    # 单线程的 dev server。现在改为：
    #   - 页面加载：只读已缓存的摘要，恒定毫秒级
    #   - 需要生成：前端点「生成摘要」按钮 → POST /api/profile/summary/
    profile = profile_service.get(user, ensure_llm=False)

    return JsonResponse({
        "total": total,
        "results": results,
        "profile": profile,
        "llm_available": profile_service.llm_available(),
    })


@csrf_exempt
def profile_summary_api(request):
    """
    手动触发 LLM 画像摘要生成。

    从反馈页的「生成摘要」按钮调用。因为这是用户**主动**发起的操作，
    可以接受等待几十秒（前端会显示 loading）。
    """
    if request.method != "POST":
        return JsonResponse({"error": "POST required"}, status=405)

    try:
        user = request.user
        profile = profile_service.get(user, ensure_llm=True, force_llm=True)
        return JsonResponse({
            "status": "ok",
            "llm_summary": profile.get("llm_summary", ""),
            "llm_failed": bool(profile.get("llm_failed")),
        })
    except Exception as e:
        return JsonResponse({"error": str(e)}, status=400)


# ===================== 数据浏览 API =====================
def data_api(request):
    limit = safe_int(request.GET.get("limit"), default=20, lo=1, hi=100)
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


# ===================== 意图推荐 API =====================
@csrf_exempt
def recommend_by_intent_api(request):
    """意图推荐：标签召回 → 选标签 → 过滤歌曲 → 直接返回"""
    if request.method != "POST":
        return JsonResponse({"error": "POST required"}, status=405)

    try:
        data = json.loads(request.body)
        query = data.get("query", "").strip()
        if not query:
            return JsonResponse({"error": "query 不能为空"}, status=400)

        # 1. 多路召回标签
        candidates = multi_recall(query, per_path=15)
        print(f"[意图推荐] 召回标签: {len(candidates)} 个")
        if not candidates:
            return JsonResponse({"error": "未召回到标签"}, status=404)

        # 2. LLM 选标签
        selected = select_tags(query, candidates, top_n=5)
        print(f"[意图推荐] 选中标签: {selected}")
        if not selected:
            return JsonResponse({"error": "未选中标签"}, status=404)

        # 3. 标签过滤歌曲
        tracks = find_tracks_by_tags(selected, limit=20)
        print(f"[意图推荐] 候选歌曲: {len(tracks)} 首")
        if not tracks:
            return JsonResponse({"error": "未找到匹配的歌曲"}, status=404)

        # 4. 网易云链接改为按需获取（点开详情卡片时才查这一首），
        #    避免在这里同步逐条查询（Meting 串行，最多耗时数秒）
        for t in tracks:
            t["netease_url"] = ""
            t["netease_id"] = ""

        # 5. 画像融合：给候选打上画像加分并重排
        #    （修复：原先路线 B 完全不读画像）
        # ensure_llm=False：不需要 LLM 画像摘要，避免多余的模型推理
        profile = profile_service.get(request.user, ensure_llm=False)
        if profile.get("stage") != "cold":
            score_map = {t.get("track_id"): float(t.get("hit_count") or 0) for t in tracks}
            tracks = profile_service.apply_personalize_bonus(
                tracks, profile, score_getter=lambda r: score_map.get(r.get("track_id"), 0.0)
            )
            # 归一化相似度，让前端百分比有意义（原先硬编码为 0）
            top = tracks[0].get("_final_score") or 1.0
            for t in tracks:
                t["similarity"] = round((t.get("_final_score") or 0.0) / top, 4) if top else 0.0
        else:
            for t in tracks:
                t.setdefault("similarity", 0)

        # 6. 多样性重排：避免同歌手/同流派刷屏
        tracks = diversity_rerank(
            tracks,
            top_k=len(tracks),
            lam=0.8,
            max_per_artist=2,
            max_per_genre_idx=8,
            score_key="_final_score" if profile.get("stage") != "cold" else "hit_count",
            vector_key="_vector",
        )

        # 7. 附加反馈状态
        attach_user_feedback(tracks, request.user)

        # 8. 构造 seed（用于前端显示）
        seed = {
            "track_name": query,
            "artist_name": "意图推荐",
            "artist_genres": ", ".join(selected),
        }

        return JsonResponse({
            "seed": seed,
            "results": tracks,
            "reason": "",
            "selected_tags": selected,
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return JsonResponse({"error": str(e)}, status=500)


# ===================== 一键推荐 API（画像 → 歌曲） =====================
def recommend_for_me_api(request):
    """
    一键推荐：不需要任何输入，直接根据用户画像生成推荐。

    - 每次调用结果都不同（偏置随机取样），类似"每日推荐"刷新
    - 冷启动用户（无反馈）退化为"热门精选 + 多样性"，同样可用
    - 支持 ?refresh=0 拿到确定性的排序结果（便于调试/评测）
    """
    top_k = safe_int(request.GET.get("top_k"), default=20, lo=1, hi=50)

    refresh = request.GET.get("refresh", "1") not in ("0", "false", "False")

    try:
        user = request.user
        # ensure_llm=False：一键推荐不需要 LLM 画像摘要（前端不展示）
        profile = profile_service.get(user, ensure_llm=False)

        result = daily_recommender.recommend(profile, top_k=top_k, refresh=refresh)

        tracks = result.get("results", [])
        if tracks:
            # 网易云链接改为按需获取（点开详情卡片时才查），
            # 避免同步逐条查询拖慢响应（Meting 串行，实测曾达数秒）
            for t in tracks:
                t["netease_url"] = ""
                t["netease_id"] = ""
            attach_user_feedback(tracks, user)

        seed = daily_recommender.build_seed(profile)

        return JsonResponse({
            "seed": seed,
            "results": tracks,
            "reason": "",
            "stage": result.get("stage", "cold"),
            "explore": result.get("explore", False),
            "channels": result.get("channels", {}),
            "profile": {
                "stage": profile.get("stage"),
                "signal_count": profile.get("signal_count", 0),
                "preferred_genres": profile_service.to_genre_names(profile, top_n=5),
            },
        })
    except Exception as e:
        import traceback
        traceback.print_exc()
        return JsonResponse({"error": str(e)}, status=500)