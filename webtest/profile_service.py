# webtest/profile_service.py
"""
用户画像服务 —— 画像的唯一读写入口。

设计目标
--------
1. 画像"独立化"：三条推荐路线（item→item / 意图→歌曲 / 画像→歌曲）
   全部平等地通过本模块拿画像，不再各自 json.loads 猜字段。
2. 零数据库迁移：画像仍存在 UserProfile.profile_json，但内部结构
   升级为带 schema_version 的规范化格式。
3. 新增"向量化"能力：把喜欢的歌在 Qdrant 里的 64 维声学向量求加权质心，
   得到"用户口味向量"，用于 user→item 检索。
4. 新增"流派索引化"能力：借助 data/artist_genre_map.json 把画像里的
   流派映射成 artist_genre_idx，从而走 Qdrant 索引做毫秒级召回
   （artist_genres 文本字段没有索引，全文过滤是全表扫描）。
"""

import json
import math
import os
import threading
import time
from datetime import datetime

from django.utils import timezone

from webtest.ollama_client import client

# ============================================================
# 常量
# ============================================================
PROFILE_SCHEMA_VERSION = 2

# 半衰期：30 天前的一次反馈，权重降到 0.5
HALF_LIFE_DAYS = 30.0

# 画像里保留的历史歌曲条数上限（防止 profile_json 无限膨胀）
MAX_TRACKS_KEPT = 50

# 求质心所需的最少喜欢歌曲数
MIN_LIKED_FOR_CENTROID = 3

# 质心最多用多少首喜欢的歌
MAX_CENTROID_SEEDS = 30

# 冷启动分档阈值
HOT_THRESHOLD = 5       # >= 5 条有效反馈 → hot，否则 warm

# LLM 摘要生成失败后，多久才允许自动重试（小时）
# 目的：避免"每次打开反馈页都等 45 秒"的死循环
LLM_RETRY_AFTER_HOURS = 6

# 路径（与 tag_retriever.TAGS_FILE、build_tag_index.INPUT 保持同一风格）
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GENRE_MAP_FILE = os.path.join(BASE_DIR, "data", "artist_genre_map.json")
TAGS_FILE = os.path.join(BASE_DIR, "tags_full.json")
# 预生成的精简别名缓存（见 build_alias_cache），避免每次解析 2.8MB tags_full.json
ALIAS_CACHE_FILE = os.path.join(BASE_DIR, "data", "genre_alias.json")

# ============================================================
# artist_genre_map.json 懒加载
# 结构：{"0": "<UNK>", "1": "pop", "2": "rap", ...}（idx 字符串 → 流派名）
# ============================================================
_genre_map = None           # {int idx: str name}
_genre_name_to_idx = None   # {str name_lower: int idx}
_genre_alias_to_idx = None  # {str alias_lower: int idx}  含 tags_full.json 的别名


def _get_genre_map():
    """加载 genre_idx → genre_name 映射（进程内单例）"""
    global _genre_map, _genre_name_to_idx
    if _genre_map is None:
        if os.path.exists(GENRE_MAP_FILE):
            with open(GENRE_MAP_FILE, "r", encoding="utf-8") as f:
                raw = json.load(f)
            _genre_map = {int(k): v for k, v in raw.items()}
        else:
            print(f"[画像] 警告：未找到流派映射表 {GENRE_MAP_FILE}")
            _genre_map = {}
        _genre_name_to_idx = {}
        for idx, name in _genre_map.items():
            if not name:
                continue
            key = name.strip().lower()
            # 同名时保留最小的 idx（idx 越小代表越主流）
            if key not in _genre_name_to_idx:
                _genre_name_to_idx[key] = idx
        print(f"[画像] 加载 {len(_genre_map)} 个流派映射")
    return _genre_map


def genre_name(idx):
    """流派索引 → 流派名"""
    return _get_genre_map().get(int(idx or 0), "")


def genre_idx_of(name):
    """流派名 → 流派索引（找不到返回 0）"""
    _get_genre_map()
    if not name:
        return 0
    return _genre_name_to_idx.get(str(name).strip().lower(), 0)


def _get_genre_alias_to_idx():
    """
    别名表：tags_full.json 里的 tag + tag_retriever.KEYWORD_TAG_MAP 的标签
    统一映射到 genre_idx。

    这样画像既能消化"用户反馈里的 artist_genres 文本"，
    也能消化"意图推荐选中的标签"。

    【性能】优先读预生成的精简缓存（data/genre_alias.json，约 100KB，
    加载 <20ms）。读不到才回退到解析 tags_full.json（2.8MB，约 900ms）。
    缓存可用 `python manage.py build_alias_cache` 生成。
    """
    global _genre_alias_to_idx
    if _genre_alias_to_idx is not None:
        return _genre_alias_to_idx

    _get_genre_map()

    # 快速路径：预生成缓存
    cached = _load_alias_cache()
    if cached is not None:
        _genre_alias_to_idx = cached
        print(f"[画像] 别名表 {len(cached)} 条（读缓存）")
        return _genre_alias_to_idx

    # 慢速路径：现场解析（首次运行 / 缓存缺失）
    _genre_alias_to_idx = dict(_genre_name_to_idx or {})

    if os.path.exists(TAGS_FILE):
        try:
            with open(TAGS_FILE, "r", encoding="utf-8") as f:
                for item in json.load(f):
                    tag = (item.get("tag") or "").strip().lower()
                    if not tag or tag in _genre_alias_to_idx:
                        continue
                    idx = _genre_name_to_idx.get(tag, 0)
                    if idx:
                        _genre_alias_to_idx[tag] = idx
        except Exception as e:
            print(f"[画像] 读取 tags_full.json 失败: {e}")

    try:
        from webtest.tag_retriever import KEYWORD_TAG_MAP
        for tag_weights in KEYWORD_TAG_MAP.values():
            for tag, _w in tag_weights:
                tag = tag.strip().lower()
                if tag in _genre_alias_to_idx:
                    continue
                idx = _genre_name_to_idx.get(tag, 0)
                if idx:
                    _genre_alias_to_idx[tag] = idx
    except Exception:
        pass

    print(f"[画像] 别名表 {len(_genre_alias_to_idx)} 条（现场构建，建议跑 "
          f"build_alias_cache 生成缓存）")
    return _genre_alias_to_idx


def _load_alias_cache():
    """读取预生成的别名叫缓存；不存在或损坏时返回 None"""
    if not os.path.exists(ALIAS_CACHE_FILE):
        return None
    try:
        with open(ALIAS_CACHE_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict) or not data:
            return None
        # key 统一为小写字符串，value 为 int
        return {str(k).lower(): int(v) for k, v in data.items() if v}
    except Exception as e:
        print(f"[画像] 别名缓存读取失败: {type(e).__name__}: {e}")
        return None


def build_alias_cache(verbose=True):
    """
    生成精简别名缓存文件。

    把原本每次都要解析的 2.8MB tags_full.json + KEYWORD_TAG_MAP，
    压缩成一个 {name: idx} 的扁平字典（约 100KB），加载时间从 ~900ms
    降到 ~15ms。

    用法：
        python manage.py shell -c "from webtest import profile_service as p; p.build_alias_cache()"
    """
    _get_genre_map()
    alias = dict(_genre_name_to_idx or {})

    if os.path.exists(TAGS_FILE):
        with open(TAGS_FILE, "r", encoding="utf-8") as f:
            for item in json.load(f):
                tag = (item.get("tag") or "").strip().lower()
                if tag and tag not in alias:
                    idx = _genre_name_to_idx.get(tag, 0)
                    if idx:
                        alias[tag] = idx

    try:
        from webtest.tag_retriever import KEYWORD_TAG_MAP
        for tag_weights in KEYWORD_TAG_MAP.values():
            for tag, _w in tag_weights:
                tag = tag.strip().lower()
                if tag and tag not in alias:
                    idx = _genre_name_to_idx.get(tag, 0)
                    if idx:
                        alias[tag] = idx
    except Exception:
        pass

    os.makedirs(os.path.dirname(ALIAS_CACHE_FILE), exist_ok=True)
    with open(ALIAS_CACHE_FILE, "w", encoding="utf-8") as f:
        json.dump(alias, f, ensure_ascii=False, separators=(",", ":"))

    size_kb = os.path.getsize(ALIAS_CACHE_FILE) / 1024
    if verbose:
        print(f"[画像] 已写入别名缓存: {ALIAS_CACHE_FILE}")
        print(f"        {len(alias)} 条，{size_kb:.1f} KB")
    return len(alias)


def resolve_genre_idx(genre_text, fallback_idx=0):
    """
    把流派文本解析成 genre_idx。

    优先级：
    1. 调用方已知的 idx（最可靠）
    2. 单流派名精确匹配
    3. 多流派逐个尝试，取 idx 最小（最主流）的那个
    """
    if fallback_idx:
        return int(fallback_idx)
    if not genre_text:
        return 0

    alias = _get_genre_alias_to_idx()
    parts = [g.strip().lower() for g in str(genre_text).split(",") if g.strip()]

    if len(parts) == 1 and parts[0] in alias:
        return alias[parts[0]]

    hits = [alias[p] for p in parts if p in alias]
    if hits:
        return min(hits)
    return 0


def _genre_idxs_in_text(genre_text):
    """从 'uk pop, pop' 里提取所有能映射上的 genre_idx 集合"""
    if not genre_text:
        return set()
    alias = _get_genre_alias_to_idx()
    idxs = set()
    for p in str(genre_text).split(","):
        p = p.strip().lower()
        if not p:
            continue
        idx = alias.get(p, 0)
        if idx:
            idxs.add(idx)
    if not idxs:
        idx = resolve_genre_idx(genre_text)
        if idx:
            idxs.add(idx)
    return idxs


# ============================================================
# 记忆衰减
# ============================================================
def _decay(created_at, now=None):
    """按 30 天半衰期计算时间衰减权重"""
    if created_at is None:
        return 1.0
    now = now or timezone.now()
    try:
        days = (now - created_at).total_seconds() / 86400.0
    except Exception:
        return 1.0
    if days <= 0:
        return 1.0
    return math.pow(0.5, days / HALF_LIFE_DAYS)


# ============================================================
# 画像重建
# ============================================================
def rebuild(user, with_llm=False, keep_summary=True):
    """
    根据该用户当前所有反馈重建画像。

    Args:
        user: Django User
        with_llm: 是否同步生成 LLM 摘要。反馈接口应传 False（保持接口快速）。
        keep_summary: 是否保留已有的 llm_summary / llm_attempted_at。
                      默认 True —— 否则每次点反馈都会把摘要清空，
                      导致下次打开反馈页又触发一次 LLM 调用（死循环）。
    """
    from webtest.models import UserFeedback

    # 先读出旧画像里需要保留的字段
    preserved = {}
    if keep_summary:
        from webtest.models import UserProfile
        obj = UserProfile.objects.filter(user=user).first()
        if obj:
            try:
                old = json.loads(obj.profile_json or "{}")
                for k in ("llm_summary", "llm_attempted_at", "llm_failed"):
                    if old.get(k):
                        preserved[k] = old[k]
            except Exception:
                pass

    rows = list(UserFeedback.objects.filter(user=user).order_by("-created_at"))
    now = timezone.now()

    genre_scores = {}    # idx -> 净权重（正=喜欢，负=不喜欢）
    artist_scores = {}   # artist_name -> 权重
    liked_artist_idxs = set()   # 喜欢的歌手的 artist_idx（给 related_artist 召回用）
    liked_tracks = []
    disliked_tracks = []

    for r in rows:
        if r.feedback not in ("like", "dislike"):
            continue

        w = _decay(r.created_at, now=now)
        rec = {
            "track_id": r.track_id,
            "track_name": r.track_name,
            "artist_name": r.artist_name,
            "artist_genres": r.artist_genres,
        }
        if r.feedback == "like":
            liked_tracks.append(rec)
        else:
            disliked_tracks.append(rec)

        sign = 1.0 if r.feedback == "like" else -1.0

        # --- 流派：优先用记录时就存下的 artist_genre_idx（最可靠），
        #     退化到从 artist_genres 文本解析 ---
        gidxs = set()
        gidx_stored = int(getattr(r, "artist_genre_idx", 0) or 0)
        if gidx_stored:
            gidxs.add(gidx_stored)
        gidxs |= _genre_idxs_in_text(r.artist_genres)
        for idx in gidxs:
            genre_scores[idx] = genre_scores.get(idx, 0.0) + sign * w

        # --- 歌手 ---
        artist = (r.artist_name or "").strip()
        if artist:
            artist_scores[artist] = artist_scores.get(artist, 0.0) + sign * w
        aidx = int(getattr(r, "artist_idx", 0) or 0)
        if aidx and sign > 0:
            liked_artist_idxs.add(aidx)

    # ---- 拆正负 ----
    preferred_genres = []
    disliked_genres = []
    for idx, score in genre_scores.items():
        if score > 0:
            preferred_genres.append({"idx": idx, "name": genre_name(idx), "weight": round(score, 4)})
        elif score < 0:
            disliked_genres.append({"idx": idx, "name": genre_name(idx), "weight": round(-score, 4)})

    preferred_genres.sort(key=lambda x: -x["weight"])
    disliked_genres.sort(key=lambda x: -x["weight"])

    preferred_artists = [
        {"name": name, "weight": round(score, 4)}
        for name, score in artist_scores.items() if score > 0
    ]
    preferred_artists.sort(key=lambda x: -x["weight"])
    preferred_artists = preferred_artists[:10]

    # ---- 冷启动分档 ----
    signal_count = len(liked_tracks) + len(disliked_tracks)
    if signal_count == 0:
        stage = "cold"
    elif signal_count < HOT_THRESHOLD:
        stage = "warm"
    else:
        stage = "hot"

    liked_count = len(liked_tracks)
    profile = {
        "schema_version": PROFILE_SCHEMA_VERSION,
        "updated_at": now.isoformat(),
        "stage": stage,
        "signal_count": signal_count,
        "liked_count": liked_count,
        "preferred_genres": preferred_genres,
        "disliked_genres": disliked_genres,
        "preferred_artists": preferred_artists,
        "liked_artist_idxs": sorted(liked_artist_idxs),
        "centroid_available": liked_count >= MIN_LIKED_FOR_CENTROID,
        "query_text": ", ".join(g["name"] for g in preferred_genres[:5] if g["name"]),
        "llm_summary": "",
        "liked_tracks": liked_tracks[:MAX_TRACKS_KEPT],
        "disliked_tracks": disliked_tracks[:MAX_TRACKS_KEPT],
    }
    # 保留已生成的 LLM 摘要与"已尝试"标记，
    # 避免点一次反馈就把摘要清空、下次打开页面又触发 45 秒推理。
    profile.update(preserved)

    if with_llm and not profile.get("llm_summary") and signal_count > 0:
        summary = analyze_profile(profile)
        profile["llm_attempted_at"] = now.isoformat()
        if summary:
            profile["llm_summary"] = summary
        else:
            profile["llm_failed"] = True

    return profile


# ============================================================
# 画像读取 / 写回 / 惰性补 LLM 摘要
# ============================================================
def empty_profile():
    """冷启动用户的空画像"""
    return {
        "schema_version": PROFILE_SCHEMA_VERSION,
        "updated_at": timezone.now().isoformat(),
        "stage": "cold",
        "signal_count": 0,
        "liked_count": 0,
        "preferred_genres": [],
        "disliked_genres": [],
        "preferred_artists": [],
        "liked_artist_idxs": [],
        "centroid_available": False,
        "query_text": "",
        "llm_summary": "",
        "liked_tracks": [],
        "disliked_tracks": [],
    }


def get(user, ensure_llm=True, auto_build=True, force_llm=False):
    """
    读取用户画像。

    - 画像不存在或 schema 过期 → 自动重建（毫秒级，不调 LLM）
    - ensure_llm=True 且 llm_summary 为空且有行为信号 → 惰性补生成一次
    - force_llm=True  → 忽略"已尝试"标记，强制重新生成（供手动刷新用）

    【失败也要记账】
    LLM 生成失败时，会写入 llm_attempted_at 标记。这样下次调用会跳过，
    避免"每次打开页面都重试 45 秒"的死循环。
    只有超过 LLM_RETRY_AFTER_HOURS 小时、或用户主动 force_llm 时才重试。

    Returns: profile dict（永不返回 None）
    """
    from webtest.models import UserProfile

    profile = None
    obj = UserProfile.objects.filter(user=user).first()
    if obj:
        try:
            data = json.loads(obj.profile_json or "{}")
            if isinstance(data, dict) and data.get("schema_version") == PROFILE_SCHEMA_VERSION:
                profile = data
        except Exception:
            profile = None

    # schema 升级：自动重建一次（含从旧格式迁移）
    if profile is None and auto_build:
        profile = rebuild(user, with_llm=False)
        save(user, profile)

    if profile is None:
        profile = empty_profile()

    # 惰性补 LLM 摘要
    if ensure_llm and not profile.get("llm_summary") and profile.get("signal_count", 0) > 0:
        if force_llm or _should_retry_llm(profile):
            summary = analyze_profile(profile)
            # 无论成功失败都记时间，避免反复重试
            profile["llm_attempted_at"] = timezone.now().isoformat()
            if summary:
                profile["llm_summary"] = summary
                profile.pop("llm_failed", None)
            else:
                profile["llm_failed"] = True
            save(user, profile)

    return profile


def _should_retry_llm(profile):
    """
    判断是否值得再试一次 LLM 摘要生成。

    规则：从未尝试过 → 试；上次尝试已超过 LLM_RETRY_AFTER_HOURS 小时 → 再试；
    否则跳过（这就是避免"每次刷新都等 45 秒"的关键）。
    """
    attempted = profile.get("llm_attempted_at")
    if not attempted:
        return True
    try:
        last = datetime.fromisoformat(attempted)
        if timezone.is_naive(last):
            last = timezone.make_aware(last)
    except Exception:
        return True
    elapsed = (timezone.now() - last).total_seconds()
    return elapsed > LLM_RETRY_AFTER_HOURS * 3600


def save(user, profile):
    """画像写回数据库"""
    from webtest.models import UserProfile
    from webtest.db_retry import with_retry

    # get_or_create + save 是"读-改-写"，多用户并发时会撞 SQLite 写锁
    # （详见 webtest/db_retry.py），故包一层重试
    @with_retry
    def _write():
        obj, _ = UserProfile.objects.get_or_create(user=user)
        obj.profile_json = json.dumps(profile, ensure_ascii=False)
        obj.save()

    _write()
    return profile



# ============================================================
# 画像 → 检索可用的表示
# ============================================================
def to_genre_idxs(profile, top_n=5):
    """
    画像流派 → genre_idx 列表（按权重降序，已剔除不喜欢的）。

    这些 idx 有 Qdrant integer 索引，可以直接 MatchValue 过滤，
    比 artist_genres 全文过滤快几个数量级。
    """
    if not profile:
        return []
    disliked = {g["idx"] for g in profile.get("disliked_genres", [])}
    return [
        g["idx"] for g in profile.get("preferred_genres", [])[:top_n]
        if g.get("idx") and g["idx"] not in disliked
    ]


def to_genre_names(profile, top_n=5):
    if not profile:
        return []
    return [g["name"] for g in profile.get("preferred_genres", [])[:top_n] if g.get("name")]


def to_disliked_genre_idxs(profile, top_n=5):
    """不喜欢的流派 idx（用于 Qdrant must_not）"""
    if not profile:
        return []
    return [g["idx"] for g in profile.get("disliked_genres", [])[:top_n] if g.get("idx")]


def to_disliked_track_ids(profile):
    if not profile:
        return []
    return [t["track_id"] for t in profile.get("disliked_tracks", []) if t.get("track_id")]


def to_liked_track_ids(profile):
    if not profile:
        return []
    return [t["track_id"] for t in profile.get("liked_tracks", []) if t.get("track_id")]


def to_seen_track_ids(profile):
    """已交互（喜欢或不喜欢）的全部 track_id —— 用于排除已听"""
    return list(set(to_liked_track_ids(profile)) | set(to_disliked_track_ids(profile)))


def to_centroid(profile, qdrant_client, collection_name="spotify_tracks",
                max_seeds=MAX_CENTROID_SEEDS, limit=60):
    """
    画像 → 64 维口味向量（加权质心，L2 归一化）。

    【为什么不用用户喜欢的那些歌本身】
    spotify_tracks 的 track_id **没有 payload 索引**，按 track_id 反查
    会在 1238 万点上全表扫描（实测超时）。所以这里改为"流派锚定"策略：

      取画像 top-N 流派 → 用 artist_genre_idx（有索引）+ popularity
      在该流派内取最热的一批歌 → 用它们的声学向量按流派权重加权平均。

    语义上这更稳：得到的是"用户偏好流派的主流听感质心"，而不是
    "恰好被点过的那几首歌的均值"，抗单曲噪声、且恒定有数据可算
    （只要有 1 个流派就能算，不依赖凑够 3 首喜欢的歌）。

    Returns: list[float] 或 None
    """
    if not profile:
        return None

    # 支持两种画像：新版（preferred_genres 为 [{idx,name,weight}]）
    # 与旧版（{name: count}）
    weights = {}
    pg = profile.get("preferred_genres") or []
    if isinstance(pg, dict):
        max_w = max(pg.values()) if pg else 1.0
        for name, w in pg.items():
            idx = genre_idx_of(name)
            if idx:
                weights[idx] = w / max_w
    else:
        max_w = max((g.get("weight", 0) for g in pg), default=1.0) or 1.0
        for g in pg:
            idx = g.get("idx") or genre_idx_of(g.get("name", ""))
            if idx:
                weights[idx] = g.get("weight", 0) / max_w

    if not weights:
        return None

    # 取权重最高的若干流派
    top = sorted(weights.items(), key=lambda x: -x[1])[:3]

    acc = None
    weight_sum = 0.0
    for idx, gw in top:
        try:
            from qdrant_client.http import models as qmodels
            records, _ = qdrant_client.scroll(
                collection_name=collection_name,
                scroll_filter=qmodels.Filter(must=[
                    qmodels.FieldCondition(
                        key="artist_genre_idx",
                        match=qmodels.MatchValue(value=int(idx)),
                    ),
                    qmodels.FieldCondition(
                        key="popularity",
                        range=qmodels.Range(gte=0.5),
                    ),
                ]),
                limit=max_seeds,
                with_payload=False,
                with_vectors=True,
            )
        except Exception as e:
            print(f"[画像] 流派 {idx} 锚定向量失败: {e}")
            continue

        for rec in records:
            vec = rec.vector
            if vec is None:
                continue
            try:
                vec = [float(v) for v in vec]
            except (TypeError, ValueError):
                continue
            if not vec:
                continue
            if acc is None:
                acc = [0.0] * len(vec)
            if len(vec) != len(acc):
                continue
            for i, v in enumerate(vec):
                acc[i] += v * gw
            weight_sum += gw

    if acc is None or weight_sum <= 0:
        return None

    acc = [v / weight_sum for v in acc]
    norm = math.sqrt(sum(v * v for v in acc))
    if norm <= 0:
        return None
    return [v / norm for v in acc]


# ============================================================
# 画像排序加成（三条路线共用）
# ============================================================
def score_bonus(candidate, profile, genre_weight=0.06, artist_weight=0.10, cap=0.25):
    """
    单条候选的画像加成。

    相对旧 personalize_results 的改动：
    - 旧实现按 max 归一化后"逐流派累加"，流派多的歌会获得叠加优势；
      这里改为"取最强的一个正向流派 + 最强的一个负向流派 + 歌手加成"，
      并设总上限，消除该偏差。
    - 候选流派同时认 artist_genre_idx（索引）和 artist_genres（文本）两条路径。
    """
    if not profile:
        return 0.0

    bonus = 0.0

    pref = {g["idx"]: g["weight"] for g in profile.get("preferred_genres", []) if g.get("idx")}
    bad = {g["idx"]: g["weight"] for g in profile.get("disliked_genres", []) if g.get("idx")}
    max_pref = max(pref.values()) if pref else 1.0
    max_bad = max(bad.values()) if bad else 1.0

    cand_idxs = set()
    gidx = int(candidate.get("artist_genre_idx") or 0)
    if gidx:
        cand_idxs.add(gidx)
    cand_idxs |= _genre_idxs_in_text(candidate.get("artist_genres"))

    best_pos, best_neg = 0.0, 0.0
    for i in cand_idxs:
        if i in pref:
            best_pos = max(best_pos, pref[i] / max_pref)
        if i in bad:
            best_neg = max(best_neg, bad[i] / max_bad)
    bonus += best_pos * genre_weight
    bonus -= best_neg * genre_weight

    # ---- 歌手 ----
    artists = {a["name"]: a["weight"] for a in profile.get("preferred_artists", [])}
    if artists:
        max_artist = max(artists.values())
        cand_artist = (candidate.get("artist_name") or "").strip()
        if cand_artist in artists:
            bonus += (artists[cand_artist] / max_artist) * artist_weight

    return round(max(-cap, min(cap, bonus)), 4)


def apply_personalize_bonus(results, profile, score_getter=None):
    """
    对一批结果施加画像加成并按最终分排序。

    score_getter(r) -> float：取"基础分"的函数。默认依次尝试
    final_score → similarity → _rrf_score，与旧 personalize_results 契约一致。
    """
    def _default_base(r):
        for key in ("final_score", "similarity", "_rrf_score"):
            v = r.get(key)
            if v is not None:
                return float(v)
        return 0.0

    getter = score_getter or _default_base

    for r in results:
        bonus = score_bonus(r, profile)
        r["_personal_bonus"] = bonus
        r["_final_score"] = round(getter(r) + bonus, 6)

    results.sort(key=lambda x: x["_final_score"], reverse=True)
    return results


# ============================================================
# LLM 摘要（保持原有能力，改为惰性触发）
# ============================================================
def analyze_profile(profile_data: dict) -> str:
    """用 LLM 分析用户画像，生成偏好摘要"""
    liked = profile_data.get("liked_tracks", [])[:10]
    disliked = profile_data.get("disliked_tracks", [])[:10]
    top_genres = [(g["name"], g["weight"]) for g in profile_data.get("preferred_genres", [])[:5]]
    top_artists = [(a["name"], a["weight"]) for a in profile_data.get("preferred_artists", [])[:5]]
    bad_genres = [(g["name"], g["weight"]) for g in profile_data.get("disliked_genres", [])[:5]]

    if not liked and not disliked and not top_genres:
        return ""

    liked_brief = [{"track_name": t["track_name"], "artist_name": t["artist_name"]} for t in liked]
    disliked_brief = [{"track_name": t["track_name"], "artist_name": t["artist_name"]} for t in disliked]

    prompt = f"""根据以下用户音乐行为数据，总结这个用户的音乐偏好：

喜欢的歌曲：
{json.dumps(liked_brief, ensure_ascii=False)}

不喜欢的歌曲：
{json.dumps(disliked_brief, ensure_ascii=False)}

偏好流派（按喜好程度排序）：
{top_genres}

偏好艺术家：
{top_artists}

不喜欢的流派：
{bad_genres}

请用一段简洁的中文描述这个用户的音乐品味，包括他喜欢的风格特点、不喜欢的东西。不要列举具体歌名。"""

    try:
        # read_timeout 显式给一个较短的超时：
        # 画像摘要是"可选增强"，不值得让用户等 90 秒。
        # 失败时上层会保存"已尝试"标记，避免每次刷新都重试。
        summary = client.generate(
            prompt=prompt,
            max_tokens=500,
            temperature=0.7,
            read_timeout=45,
        )
        if summary:
            return summary.strip()
    except Exception as e:
        print(f"[画像分析] 失败: {type(e).__name__}: {e}")
    return ""


# ============================================================
# LLM 可用性（供前端决定是否显示「生成摘要」按钮）
# ============================================================
# 探测结果缓存：TTL 内复用，避免每次页面加载都去探测
_llm_avail_cache = {"at": 0.0, "ok": False}
_LLM_AVAIL_TTL = 30          # 秒
_LLM_PROBE_TIMEOUT = 1.5     # 秒 —— 后台线程探测，可以稍宽松


def llm_available(timeout=_LLM_PROBE_TIMEOUT):
    """
    快速探测 LLM 是否可用（**非阻塞**）。

    【为什么要后台探测】
    冷启动兜底的 llm_available() 会同步 `GET /api/version`，Ollama 未加载
    模型时该请求会挂起直到超时，让反馈页首屏多等 1-2 秒。

    现在改为：
    - 首次调用立刻返回 False 并**在后台线程里探测**，同时填充缓存
    - 后续调用直接读缓存（TTL 30 秒）
    - 页面永不因探测而阻塞；最坏情况只是首屏少显示一个按钮，
      30 秒内的后续访问就能拿到真实结果

    这个信息只用于前端"是否显示生成摘要按钮"，对核心功能无影响，
    所以允许首屏短暂不准确。
    """
    now = time.time()
    age = now - _llm_avail_cache["at"]

    if age < _LLM_AVAIL_TTL:
        return _llm_avail_cache["ok"]

    # 缓存过期：后台刷新，本次先用旧值（首次为 False）
    _probe_llm_async(timeout)
    return _llm_avail_cache["ok"]


def _probe_llm_async(timeout):
    """在后台线程里探测 LLM，完成后写入缓存（防重入）"""
    if _llm_avail_cache.get("probing"):
        return

    # 先占位，避免同一时刻起多个线程；同时把 at 提前，
    # 防止失败时每个请求都重新起线程
    _llm_avail_cache["probing"] = True
    _llm_avail_cache["at"] = time.time()

    def _worker():
        ok = False
        try:
            import requests
            base = getattr(client, "base_url", "http://localhost:11434")
            r = requests.get(f"{base}/api/version", timeout=timeout)
            ok = r.status_code == 200
        except Exception:
            ok = False
        finally:
            _llm_avail_cache["ok"] = ok
            _llm_avail_cache["at"] = time.time()
            _llm_avail_cache["probing"] = False

    t = threading.Thread(target=_worker, daemon=True)
    t.start()


# ============================================================
# 兼容层：旧的 profile_analyzer.rebuild_profile 语义
# ============================================================
def rebuild_profile(user):
    """兼容旧调用点（feedback_api）：重建并落库"""
    profile = rebuild(user, with_llm=False)
    return save(user, profile)
