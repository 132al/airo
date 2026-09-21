# ============================================================================
# 歌单导入 · 回归测试
#
# 覆盖：ID/链接解析、平台识别、曲库匹配、端到端导入、接口边界。
# 注意：会调用 Meting MCP（需联网）；拿不到歌单时跳过网络用例而非判失败。
#
# 用法：python test_playlist_import.py
# ============================================================================

import json
import os
import sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import runtime_env  # noqa: F401
import django
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "web_project.settings")
django.setup()

from django.test import Client
from django.contrib.auth.models import User
from webtest.models import UserFeedback, UserProfile
from webtest.playlist_import import (parse_playlist_id, detect_platform,
                                     lookup_track, fetch_playlist)

U = "__plit__"
PASS, FAIL, SKIP = [], [], []


def check(name, cond, detail=""):
    (PASS if cond else FAIL).append(name)
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}"
          + (f"  {detail}" if detail else ""))


def skip(name, why=""):
    SKIP.append(name)
    print(f"  [SKIP] {name}  {why}")


def cleanup():
    UserFeedback.objects.filter(user__username=U).delete()
    UserProfile.objects.filter(user__username=U).delete()
    User.objects.filter(username=U).delete()


def main():
    print("#" * 74)
    print("#  歌单导入 · 回归测试")
    print("#" * 74)

    # ---------- 1. 解析 ----------
    print()
    print("1 · 歌单 ID / 链接解析")
    cases = [
        ("3778678", "3778678"),
        ("https://music.163.com/#/playlist?id=3778678", "3778678"),
        ("https://y.music.163.com/m/playlist?id=3778678&userid=1", "3778678"),
        ("https://y.qq.com/n/ryqq/playlist/7011264340", "7011264340"),
        ("  https://music.163.com/playlist?id=123456  ", "123456"),
    ]
    for raw, exp in cases:
        got, _ = parse_playlist_id(raw)
        check(f"解析 {raw.strip()[:44]}", got == exp, f"got={got}")

    for raw in ("", "abc", None):
        got, err = parse_playlist_id(raw)
        check(f"非法输入 {raw!r} 应报错", got is None and bool(err))

    print()
    print("2 · 平台识别")
    check("纯 ID 默认 netease", detect_platform("3778678") == "netease")
    check("QQ音乐链接 -> tencent",
          detect_platform("https://y.qq.com/n/ryqq/playlist/1") == "tencent")
    check("酷狗链接 -> kugou", detect_platform("https://kugou.com/x") == "kugou")
    check("酷我链接 -> kuwo", detect_platform("https://kuwo.cn/x") == "kuwo")
    check("网易云链接 -> netease",
          detect_platform("https://y.music.163.com/m/playlist?id=1") == "netease")

    # ---------- 3. 曲库匹配 ----------
    print()
    print("3 · 曲库匹配（lookup_track）")
    en_cases = [("Shape of You", "Ed Sheeran"), ("Bohemian Rhapsody", "Queen"),
                ("Take Five", "The Dave Brubeck Quartet"),
                ("Rolling in the Deep", "Adele"), ("Numb", "Linkin Park")]
    miss = [tn for tn, ar in en_cases if not lookup_track(tn, ar)]
    check("英文歌能匹配到曲库", not miss, f"未命中: {miss}" if miss else "5/5")

    m = lookup_track("Numb", "Linkin Park")
    if m:
        check("匹配优先排除重复版本（不选 Numb / Encore）",
              "encore" not in m["track_name"].lower(), f"got={m['track_name']}")
    check("不存在的歌返回 None", lookup_track("ZZZ_不存在_999", "Nobody") is None)
    check("空歌名返回 None", lookup_track("", "X") is None)

    # ---------- 4. 端到端 ----------
    print()
    print("4 · 端到端导入")
    cleanup()
    u = User.objects.create_user(username=U, password="x")
    c = Client()
    c.login(username=U, password="x")

    before = c.get("/api/recommend_for_me/?top_k=10").json()
    check("导入前画像是 cold", before.get("stage") == "cold",
          f"stage={before.get('stage')}")

    songs, err = fetch_playlist("netease", "3778678")
    if err or not songs:
        skip("端到端导入", f"Meting 不可用: {err}")
    else:
        check("Meting 能取到歌单", len(songs) > 0, f"n={len(songs)}")
        check("歌单项含 name/artist", all(s.get("name") for s in songs[:5]))

        r = c.post("/api/playlist/import/",
                   data=json.dumps({"playlist": "3778678", "limit": 30}),
                   content_type="application/json")
        check("导入接口 200", r.status_code == 200, f"HTTP {r.status_code}")
        d = r.json()
        check("导入返回 ok", d.get("ok") is True, str(d.get("error") or ""))
        check("有歌曲被导入", (d.get("imported") or 0) > 0,
              f"imported={d.get('imported')} skipped={d.get('skipped')}")

        n_db = UserFeedback.objects.filter(user=u, feedback="like").count()
        check("DB 里写入 like 记录", n_db > 0, f"n={n_db}")

        sample = UserFeedback.objects.filter(user=u).first()
        if sample:
            check("artist_idx 反查成功（非 0）", sample.artist_idx != 0,
                  f"idx={sample.artist_idx}")
            check("artist_genre_idx 反查成功（非 0）",
                  sample.artist_genre_idx != 0, f"gidx={sample.artist_genre_idx}")

        after = c.get("/api/recommend_for_me/?top_k=10").json()
        check("导入后画像脱离 cold", after.get("stage") != "cold",
              f"stage={after.get('stage')}")

        n_before = UserFeedback.objects.filter(user=u).count()
        c.post("/api/playlist/import/",
               data=json.dumps({"playlist": "3778678", "limit": 30}),
               content_type="application/json")
        n_after = UserFeedback.objects.filter(user=u).count()
        check("重复导入幂等（不产生重复记录）", n_after == n_before,
              f"{n_before} -> {n_after}")

        # ---- 核心约定：拿不到就跳过，绝不写脏数据 ----
        rows = list(UserFeedback.objects.filter(user=u))
        check("DB 记录数 == imported（跳过的不入库）",
              len(rows) == (d.get("imported") or 0),
              f"db={len(rows)} imported={d.get('imported')}")
        check("全部记录 feedback=like",
              all(r.feedback == "like" for r in rows))
        check("全部记录 track_id 非空",
              all(bool(r.track_id) for r in rows))
        check("全部记录 track_name 非空",
              all(bool(r.track_name) for r in rows))
        check("全部记录 artist_idx 非 0（不编造元数据）",
              all(r.artist_idx != 0 for r in rows))
        check("全部记录 artist_genre_idx 非 0",
              all(r.artist_genre_idx != 0 for r in rows))

        skipped_names = [s["name"] for s in (d.get("songs") or [])
                         if s.get("status") == "skipped"]
        db_names = [r.track_name for r in rows]
        leaked = [n for n in skipped_names if n in db_names]
        check("跳过的歌没有混入 DB", not leaked,
              f"泄漏: {leaked[:3]}" if leaked else f"skipped={len(skipped_names)}")


    # ---------- 5. 边界 ----------
    print()
    print("5 · 接口边界")
    r = c.get("/api/playlist/import/")
    check("GET 返回 405", r.status_code == 405, f"HTTP {r.status_code}")

    r = c.post("/api/playlist/import/", data=json.dumps({}),
               content_type="application/json")
    check("缺 playlist 返回 400", r.status_code == 400, f"HTTP {r.status_code}")

    r = c.post("/api/playlist/import/", data=json.dumps({"playlist": "abc"}),
               content_type="application/json")
    check("无法解析的 ID 返回 400", r.status_code == 400, f"HTTP {r.status_code}")

    r = c.post("/api/playlist/import/",
               data=json.dumps({"playlist": "1", "platform": "spotify"}),
               content_type="application/json")
    check("不支持的平台返回 400", r.status_code == 400, f"HTTP {r.status_code}")

    anon = Client()
    r = anon.post("/api/playlist/import/", data=json.dumps({"playlist": "1"}),
                  content_type="application/json")
    check("未登录被拦截（302/403）", r.status_code in (302, 403),
          f"HTTP {r.status_code}")

    cleanup()

    # ---------- 汇总 ----------
    print()
    print("=" * 74)
    print(f"  通过 {len(PASS)}  失败 {len(FAIL)}  跳过 {len(SKIP)}")
    if FAIL:
        print("  失败项:")
        for f in FAIL:
            print(f"    - {f}")
    print("=" * 74)
    return 0 if not FAIL else 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    finally:
        cleanup()

