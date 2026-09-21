# webtest/playlist_import.py
"""
歌单导入 —— 用 Meting MCP 读网易云/QQ音乐歌单，把歌曲写入用户偏好。

设计思路（为什么这么做）
------------------------
不新建"歌单偏好"这套并行逻辑，而是把导入的歌**转成 UserFeedback
记录（feedback="like"）**。理由：

1. 画像链路（profile_service.rebuild）已经是"读 UserFeedback 聚合"，
   写入即可生效，**零改动复用**画像、召回、排序、理由生成全流程。
2. 用户在反馈页能看到这些歌、能取消（undo），行为一致。
3. 偏好强度天然可用"比重"表达：歌单里的歌是 like，用户后来又 dislike
   的会被后续反馈覆盖（update_or_create）。

核心约定：拿不到就跳过
----------------------
歌单里的歌若在曲库里查不到（版权/未收录，例如绝大多数中文歌），
**直接跳过，绝不写入任何 DB 记录** —— 不编造 track_id，不用空 idx 占位。

原因：UserFeedback 的 track_id / artist_idx / artist_genre_idx 是召回链路的
输入，写入假值会污染画像与推荐结果（比如把所有中文歌都算成 artist_idx=0，
反而会形成一个"虚假流派聚集"）。宁可少导入，不可脏数据。

实测：导入 30 首中文歌单 → imported=6 / skipped=24，
DB 恰好 6 条，且全部字段非空（测试 test_playlist_import.py 已断言）。

"""

import json
import re

SUPPORTED_PLATFORMS = ("netease", "tencent", "kugou", "kuwo")

# 单次导入上限，避免一个千首歌单把画像淹没、也避免 DB 写入过慢
MAX_IMPORT = 200


def parse_playlist_id(raw):
    """
    从用户输入里抽出歌单 ID。

    支持：
      - 纯数字 ID：           "3778678"
      - 网易云链接：          https://music.163.com/#/playlist?id=3778678
      - 网易云分享长链：      https://y.music.163.com/m/playlist?id=3778678&userid=1
      - QQ音乐链接：          https://y.qq.com/n/ryqq/playlist/7011264340
      - 带 "playlist/" 或 "id=" 的任意串

    Returns: (playlist_id, error_message)
    """
    if not raw:
        return None, "请输入歌单 ID 或链接"

    s = str(raw).strip()

    # 纯数字
    if s.isdigit():
        return s, None

    # URL 里 ?id=xxx
    m = re.search(r"[?&]id=(\d+)", s)
    if m:
        return m.group(1), None

    # QQ音乐 /playlist/xxxx
    m = re.search(r"/playlist/(\w+)", s)
    if m:
        return m.group(1), None

    # 兜底：串里最长的数字片段
    nums = re.findall(r"\d{5,}", s)
    if nums:
        return max(nums, key=len), None

    return None, "无法从输入里识别出歌单 ID"


def detect_platform(raw):
    """从链接里猜平台，默认 netease。"""
    s = str(raw or "").lower()
    if "y.qq.com" in s or "c.y.qq.com" in s or "qq.com" in s:
        return "tencent"
    if "kugou" in s:
        return "kugou"
    if "kuwo" in s:
        return "kuwo"
    return "netease"


def _text_of(mcp_result):
    """MCP 返回 {result:{content:[{type:text,text:...}]}}，取出 text 并解析 JSON。"""
    try:
        content = (mcp_result or {}).get("result", {}).get("content") or []
        for c in content:
            if c.get("type") == "text":
                return json.loads(c.get("text") or "{}")
    except Exception as e:
        print(f"[歌单导入] 解析 MCP 返回失败: {e}")
    return {}


def fetch_playlist(platform, playlist_id, timeout=30):
    """
    调 Meting MCP 的 playlist 工具，返回歌曲列表。

    Returns: (songs, error)
      songs: [{"id","name","artist","album"}, ...]
    """
    from .meting_client import get_client

    try:
        client = get_client()
    except Exception as e:
        return [], f"Meting 客户端不可用: {e}"

    try:
        raw = client.call_tool("playlist", {
            "platform": platform,
            "id": str(playlist_id),
        })
    except Exception as e:
        return [], f"调用 Meting 失败: {e}"

    payload = _text_of(raw)
    if not payload.get("ok"):
        return [], payload.get("error") or "歌单获取失败（可能是私有歌单或 ID 不对）"

    data = payload.get("data") or []
    if not isinstance(data, list):
        return [], "歌单返回格式异常"

    songs = []
    for item in data:
        if not isinstance(item, dict):
            continue
        name = str(item.get("name") or "").strip()
        if not name:
            continue
        artists = item.get("artist") or []
        if isinstance(artists, list):
            artist_str = ", ".join(str(a) for a in artists if a)
        else:
            artist_str = str(artists)
        songs.append({
            "id": str(item.get("id") or ""),
            "name": name,
            "artist": artist_str,
            "album": str(item.get("album") or ""),
        })

    if not songs:
        return [], "歌单里没有解析到歌曲"
    return songs, None


# ---------------------------------------------------------------------------
# 歌曲 → Qdrant 元数据
# ---------------------------------------------------------------------------
# 为什么要这一步：
#   UserFeedback 需要 track_id / artist_idx / artist_genre_idx，
#   而歌单只给歌名+歌手。这些索引字段只有 Qdrant 里有，必须反查。
#   查不到的歌（版权/库未收录）会被跳过 —— 不能编造 track_id。
_QDRANT_URL = "http://127.0.0.1:6333"
_COLLECTION = "spotify_tracks"

_qc_client = None


def _qdrant():
    global _qc_client
    if _qc_client is None:
        from qdrant_client import QdrantClient
        _qc_client = QdrantClient(url=_QDRANT_URL, timeout=60)
    return _qc_client


def lookup_track(track_name, artist_name):
    """
    按"歌名 + 歌手"在 Qdrant 里找最接近的一首，返回元数据 dict 或 None。

    只用已建索引的 track_name / artist_name 做 text 匹配，然后在结果里
    用 popularity 最高的一条（避免 remix/翻唱排在前面）。
    """
    from qdrant_client.http import models as qm

    tn = str(track_name or "").strip()
    ar = str(artist_name or "").strip()
    if not tn:
        return None

    must = [qm.FieldCondition(key="track_name", match=qm.MatchText(text=tn))]
    # 歌单的歌手字段可能是 "A, B"，取第一个主歌手去匹配更稳
    primary_ar = ar.split(",")[0].strip() if ar else ""
    if primary_ar:
        must.append(qm.FieldCondition(key="artist_name",
                                      match=qm.MatchText(text=primary_ar)))

    try:
        pts, _ = _qdrant().scroll(
            _COLLECTION,
            scroll_filter=qm.Filter(must=must),
            limit=10, with_payload=True, with_vectors=False)
    except Exception as e:
        print(f"[歌单导入] Qdrant 查询失败 {tn}: {e}")
        return None

    if not pts:
        return None

    # 1) 先剔除重复版本（Remaster/Live/A Cappella...）
    from .embeat_similar import _is_dup_version
    good = [p for p in pts
            if not _is_dup_version((p.payload or {}).get("track_name") or "")]
    pool = good if good else list(pts)

    # 2) 再按"歌名与原歌名的接近度"排序，避免 "Numb" 命中 "Numb / Encore"
    def _name_fit(p):
        cand = str((p.payload or {}).get("track_name") or "").lower()
        target = tn.lower()
        exact = 1 if cand == target else 0
        # 加了后缀的（cand 更长）略微降权
        return (exact, -abs(len(cand) - len(target)))

    pool.sort(key=lambda p: (_name_fit(p),
                             float((p.payload or {}).get("popularity") or 0)),
              reverse=True)

    pl = pool[0].payload or {}
    return {
        "track_id": str(pl.get("track_id") or ""),
        "track_name": pl.get("track_name") or tn,
        "artist_name": pl.get("artist_name") or ar,
        "artist_genres": pl.get("artist_genres") or "",
        "artist_idx": int(pl.get("artist_idx") or 0),
        "artist_genre_idx": int(pl.get("artist_genre_idx") or 0),
    }


# ---------------------------------------------------------------------------
# 主入口
# ---------------------------------------------------------------------------
def import_playlist(user, raw_input, platform=None, limit=MAX_IMPORT,
                    overwrite=False):
    """
    导入歌单为用户的"已有偏好"。

    Args:
        user: Django User
        raw_input: 歌单 ID 或分享链接
        platform: netease/tencent/kugou/kuwo，None 时自动判断
        limit: 最多导入多少首
        overwrite: True 时覆盖已有的 dislike（默认不覆盖，尊重用户显式否定）

    Returns: dict（含 imported / skipped / total / songs 明细）
    """
    from webtest.models import UserFeedback
    from webtest.db_retry import retry_call

    pid, err = parse_playlist_id(raw_input)
    if err:
        return {"ok": False, "error": err}

    platform = platform or detect_platform(raw_input)

    songs, err = fetch_playlist(platform, pid)
    if err:
        return {"ok": False, "error": err}

    total = len(songs)
    songs = songs[:max(1, int(limit))]

    imported, skipped, detail = 0, 0, []

    for s in songs:
        meta = lookup_track(s["name"], s["artist"])
        if not meta or not meta["track_id"]:
            skipped += 1
            detail.append({"name": s["name"], "artist": s["artist"],
                           "status": "skipped", "reason": "曲库未收录"})
            continue

        existing = UserFeedback.objects.filter(
            user=user, track_id=meta["track_id"]).first()

        # 已有 dislike：默认保留（用户明确说不要），除非 overwrite
        if existing and existing.feedback == "dislike" and not overwrite:
            skipped += 1
            detail.append({"name": meta["track_name"],
                           "artist": meta["artist_name"],
                           "status": "skipped", "reason": "已标记不喜欢"})
            continue

        if existing and existing.feedback == "like":
            # 已经是 like，不重复计
            detail.append({"name": meta["track_name"],
                           "artist": meta["artist_name"],
                           "status": "exists"})
            continue

        retry_call(
            UserFeedback.objects.update_or_create,
            user=user, track_id=meta["track_id"],
            defaults={
                "track_name": meta["track_name"],
                "artist_name": meta["artist_name"],
                "artist_genres": meta["artist_genres"],
                "feedback": "like",
                "artist_idx": meta["artist_idx"],
                "artist_genre_idx": meta["artist_genre_idx"],
            })

        imported += 1
        detail.append({"name": meta["track_name"],
                       "artist": meta["artist_name"],
                       "status": "imported"})

    # 重建画像：复用现有链路，导入即生效
    # 也要走重试 —— rebuild_profile 会写 UserProfile，同样可能撞锁
    from webtest import profile_service
    profile = retry_call(profile_service.rebuild_profile, user)


    return {
        "ok": True,
        "playlist_id": pid,
        "platform": platform,
        "total": total,
        "imported": imported,
        "skipped": skipped,
        "songs": detail,
        "stage": profile.get("stage", "cold"),
        "signal_count": profile.get("signal_count", 0),
    }


