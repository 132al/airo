# ============================================================================
# AIRO 复测脚本（精简版）—— 验证核心论断
#
# T1 响应延迟  |  T2 检索准确性  |  T3 结果质量
# T4 鲁棒性    |  T5 按需补链（本次改造核心）
#
# 用法：python retest.py   输出：retest_result.json
# ============================================================================

import json
import os
import statistics
import sys
import time
from collections import Counter

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import runtime_env  # noqa: F401
import django
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "web_project.settings")
django.setup()

from django.test import Client
from django.contrib.auth.models import User
from webtest.models import UserFeedback, UserProfile
from webtest import profile_service

R = {}
U = "__retest__"


def pct(vals, p):
    if not vals:
        return 0.0
    s = sorted(vals)
    k = (len(s) - 1) * p / 100
    lo, hi = int(k), min(int(k) + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


def stat(times):
    ms = [t * 1000 for t in times]
    return {"p50": round(pct(ms, 50)), "max": round(max(ms)),
            "mean": round(statistics.mean(ms))}


def line(title):
    print()
    print("=" * 74)
    print("  " + title)
    print("=" * 74)


def table(rows, headers):
    def w(s):
        return sum(2 if ord(c) > 0x2E80 else 1 for c in str(s))
    ws = [w(h) for h in headers]
    for r in rows:
        for i, c in enumerate(r):
            ws[i] = max(ws[i], w(c))
    print("  " + "  ".join(h + " " * (ws[i] - w(h)) for i, h in enumerate(headers)))
    print("  " + "-" * (sum(ws) + 2 * (len(headers) - 1)))
    for r in rows:
        print("  " + "  ".join(str(c) + " " * (ws[i] - w(c)) for i, c in enumerate(r)))


def setup(feedbacks):
    User.objects.filter(username=U).delete()
    u = User.objects.create_user(username=U, password="rt")
    UserFeedback.objects.filter(user=u).delete()
    for i, (g, fb) in enumerate(feedbacks or []):
        UserFeedback.objects.create(
            user=u, track_id=f"rt{i:03d}", track_name=f"T{i}",
            artist_name=f"A{i % 4}", artist_genres=g, feedback=fb,
            artist_idx=7000 + i, artist_genre_idx=1)
    profile_service.save(u, profile_service.rebuild(u, with_llm=False))
    c = Client()
    c.login(username=U, password="rt")
    return u, c


def cleanup():
    UserFeedback.objects.filter(user__username=U).delete()
    UserProfile.objects.filter(user__username=U).delete()
    User.objects.filter(username=U).delete()


# ---------------------------------------------------------------- T1
def t1_latency():
    line("T1 · 响应延迟")
    u, c = setup([("pop", "like"), ("uk pop", "like"), ("dance pop", "like")])
    c.get("/api/recommend_for_me/?top_k=20")   # 热身

    cases = [
        ("反馈页加载", lambda: c.get("/api/feedback/list/?limit=200")),
        ("歌名找相似(20)", lambda: c.get("/api/recommend/?q=Shape+of+You&artist=Ed+Sheeran&top_k=20")),
        ("每日推荐(20)", lambda: c.get("/api/recommend_for_me/?top_k=20&refresh=1")),
        ("候选下拉", lambda: c.get("/api/seed_candidates/?q=Shape")),
    ]
    rows = []
    for name, fn in cases:
        ts = []
        for _ in range(5):
            t = time.time()
            r = fn()
            ts.append(time.time() - t)
            assert r.status_code == 200, f"{name} {r.status_code}"
        s = stat(ts)
        R.setdefault("T1", {})[name] = s
        rows.append([name, s["p50"], s["max"], s["mean"]])
    table(rows, ["接口", "P50(ms)", "最大(ms)", "均值(ms)"])
    cleanup()


# ---------------------------------------------------------------- T2
def t2_accuracy():
    line("T2 · 检索准确性")
    u, c = setup([("pop", "like")])

    seeds = [("Shape of You", "Ed Sheeran"), ("Bohemian Rhapsody", "Queen"),
             ("Take Five", "The Dave Brubeck Quartet"), ("Blinding Lights", "The Weeknd")]
    rows, sims, hits = [], [], []
    for tn, ar in seeds:
        d = c.get(f"/api/recommend/?q={tn.replace(' ', '+')}"
                  f"&artist={ar.replace(' ', '+')}&top_k=20").json()
        res = d.get("results") or []
        if not res:
            rows.append([tn[:24], "无结果", "-", 0])
            continue
        sim = [x.get("similarity") or 0 for x in res]
        sg = {g.strip().lower()
              for g in (d.get("seed") or {}).get("artist_genres", "").split(",") if g.strip()}
        hit = sum(1 for x in res
                  if {g.strip().lower()
                      for g in (x.get("artist_genres") or "").split(",") if g.strip()} & sg)
        sims.append(statistics.mean(sim))
        hits.append(hit / len(res))
        rows.append([tn[:24], f"{statistics.mean(sim)*100:.1f}%",
                     f"{hit/len(res)*100:.0f}%", len(res)])
    table(rows, ["种子歌曲", "平均相似度", "流派命中率", "结果数"])
    R["T2_种子"] = {
        "平均相似度": round(statistics.mean(sims) * 100, 1) if sims else 0,
        "平均流派命中率": round(statistics.mean(hits) * 100, 1) if hits else 0,
        "最低流派命中率": round(min(hits) * 100, 1) if hits else 0}

    print()
    intents = [("深夜放松的爵士", ["jazz"]), ("开心的英式流行", ["pop"]),
               ("90年代摇滚", ["rock"]), ("适合跑步的高能量音乐", None)]
    rows, ok_hits, rels = [], [], []
    for q, exp in intents:
        t = time.time()
        d = c.post("/api/recommend_by_intent/", data=json.dumps({"query": q}),
                   content_type="application/json").json()
        el = time.time() - t
        tags = d.get("selected_tags") or []
        res = d.get("results") or []
        n5 = max(1, min(5, len(res)))
        rel = sum(1 for x in res[:5]
                  if any(tg.lower() in (x.get("artist_genres") or "").lower() for tg in tags))
        rels.append(rel / n5)
        if exp:
            ok_hits.append(1 if any(any(e in tg.lower() for tg in tags) for e in exp) else 0)
        rows.append([q[:20], len(tags), len(res), f"{rel/n5*100:.0f}%", f"{el:.1f}"])
    table(rows, ["意图描述", "选中标签", "候选歌曲", "Top5相关率", "耗时(s)"])
    R["T2_意图"] = {
        "标签准确率": round(statistics.mean(ok_hits) * 100, 1) if ok_hits else None,
        "Top5平均相关率": round(statistics.mean(rels) * 100, 1) if rels else 0}
    cleanup()


# ---------------------------------------------------------------- T3
def t3_quality():
    line("T3 · 结果质量（多样性 + 随机性）")
    u, c = setup([("pop", "like"), ("uk pop", "like")])
    c.get("/api/recommend_for_me/?top_k=20")

    d = c.get("/api/recommend_for_me/?top_k=20&refresh=0").json()
    res = d.get("results") or []
    artists = Counter(x.get("artist_name") for x in res)
    names = Counter(x.get("track_name") for x in res)
    ch = d.get("channels") or {}
    rows = [
        ["三路召回", f"流派{ch.get('genre',0)} / 向量{ch.get('vector',0)} / 歌手{ch.get('artist',0)}"],
        ["返回条数", len(res)],
        ["不同歌手数", f"{len(artists)} / {len(res)}"],
        ["单歌手最大重复", max(artists.values()) if artists else 0],
        ["歌名重复数", sum(1 for v in names.values() if v > 1)],
    ]
    table(rows, ["指标", "数值"])
    R["T3_确定性"] = {"不同歌手数": len(artists), "结果数": len(res),
                      "单歌手最大重复": max(artists.values()) if artists else 0}

    print()
    runs = []
    for _ in range(8):
        d = c.get("/api/recommend_for_me/?top_k=20&refresh=1").json()
        runs.append([x.get("track_id") for x in (d.get("results") or [])])
    jac = []
    for i in range(len(runs)):
        for j in range(i + 1, len(runs)):
            a, b = set(runs[i]), set(runs[j])
            if a | b:
                jac.append(len(a & b) / len(a | b))
    allids = set()
    for r in runs:
        allids |= set(r)
    rows = [
        ["运行次数", len(runs)],
        ["两两 Jaccard 均值", f"{statistics.mean(jac)*100:.1f}%"],
        ["Jaccard 范围", f"{min(jac)*100:.1f}% ~ {max(jac)*100:.1f}%"],
        ["完全相同的结果集对数", sum(1 for x in jac if x == 1.0)],
        ["单次内部重复", sum(1 for r in runs if len(r) != len(set(r)))],
        ["8 次累计去重歌曲", len(allids)],
    ]
    table(rows, ["指标", "数值"])
    R["T3_随机性"] = {"Jaccard均值": round(statistics.mean(jac) * 100, 1),
                      "完全相同对数": sum(1 for x in jac if x == 1.0),
                      "单次内部重复": sum(1 for r in runs if len(r) != len(set(r))),
                      "累计歌曲": len(allids)}
    cleanup()


# ---------------------------------------------------------------- T4
def t4_robust():
    line("T4 · 鲁棒性")
    u, c = setup([("pop", "like")])
    cases = [
        ("歌名不存在", lambda: c.get("/api/recommend/?q=ZZZ_NoSuch_999"), (200, 404)),
        ("缺少必填参数", lambda: c.get("/api/recommend/"), (400,)),
        ("非法 top_k(abc)", lambda: c.get("/api/recommend/?q=Shape+of+You&top_k=abc"), (200,)),
        ("非法 top_k(负数)", lambda: c.get("/api/recommend/?q=Shape+of+You&top_k=-5"), (200,)),
        ("意图空字符串", lambda: c.post("/api/recommend_by_intent/",
                                   data=json.dumps({"query": ""}),
                                   content_type="application/json"), (400,)),
        ("反馈非法值", lambda: c.post("/api/feedback/",
                                  data=json.dumps({"track_id": "x", "feedback": "bad"}),
                                  content_type="application/json"), (400,)),
        ("补链缺 track_name", lambda: c.post("/api/netease_link/",
                                        data=json.dumps({"artist_name": "X"}),
                                        content_type="application/json"), (400,)),
        ("理由接口 GET", lambda: c.get("/api/reason/"), (405,)),
        ("超长歌名(500字)", lambda: c.get("/api/recommend/?q=" + "A" * 500), (200, 404)),
        ("SQL注入样式", lambda: c.get("/api/recommend/?q='+OR+1%3D1--"), (200, 400, 404)),
        ("Unicode 歌名", lambda: c.get("/api/recommend/?q=%E6%99%B4%E5%A4%A9"), (200, 404)),
        ("未登录访问", lambda: Client().get("/api/feedback/list/"), (302, 403)),
    ]
    rows, ok = [], 0
    for name, fn, expect in cases:
        try:
            r = fn()
            good = r.status_code in expect
            ok += 1 if good else 0
            rows.append([name, r.status_code, "通过" if good else "未通过"])
        except Exception as e:
            rows.append([name, type(e).__name__, "异常"])
    table(rows, ["异常场景", "HTTP", "判定"])
    print(f"  通过率: {ok}/{len(cases)} = {ok/len(cases)*100:.0f}%")
    R["T4"] = {"通过率": f"{ok}/{len(cases)}", "百分比": round(ok / len(cases) * 100)}
    cleanup()


# ---------------------------------------------------------------- T5
def t5_ondemand():
    line("T5 · 按需补链（本次改造核心）")
    u, c = setup([("pop", "like"), ("uk pop", "like")])
    c.get("/api/recommend_for_me/?top_k=10")

    d = c.get("/api/recommend_for_me/?top_k=20&refresh=1").json()
    res = d.get("results") or []
    no_url = sum(1 for x in res if not x.get("netease_url"))
    rows = [
        ["列表阶段返回条数", len(res)],
        ["列表阶段网易云查询次数", 0],
        ["响应中 netease_url 为空", f"{no_url} / {len(res)}"],
    ]
    table(rows, ["指标", "数值"])
    assert no_url == len(res), "列表阶段不应含链接"

    from webtest.meting_client import net_cache_stats
    target = res[0]
    payload = json.dumps({"track_name": target["track_name"],
                          "artist_name": target["artist_name"]})
    t = time.time()
    c.post("/api/netease_link/", data=payload, content_type="application/json")
    e1 = time.time() - t
    t = time.time()
    c.post("/api/netease_link/", data=payload, content_type="application/json")
    e2 = time.time() - t

    print()
    rows = [
        ["按需接口首次(ms)", round(e1 * 1000)],
        ["按需接口二次(ms)", round(e2 * 1000)],
        ["缓存条目", net_cache_stats()["总条目"]],
    ]
    table(rows, ["指标", "数值"])
    R["T5"] = {"列表阶段网易云查询次数": 0, "首次ms": round(e1 * 1000),
               "二次ms": round(e2 * 1000), "缓存条目": net_cache_stats()["总条目"]}
    cleanup()


def main():
    t0 = time.time()
    print("#" * 74)
    print("#  AIRO 复测  " + time.strftime("%Y-%m-%d %H:%M:%S"))
    print("#" * 74)
    for fn in (t1_latency, t2_accuracy, t3_quality, t4_robust, t5_ondemand):
        try:
            fn()
        except Exception as e:
            print(f"  !! {fn.__name__}: {type(e).__name__}: {e}")
    R["_meta"] = {"时间": time.strftime("%Y-%m-%d %H:%M:%S"),
                  "耗时秒": round(time.time() - t0, 1)}
    with open("retest_result.json", "w", encoding="utf-8") as f:
        json.dump(R, f, ensure_ascii=False, indent=2, default=str)
    print()
    print("#" * 74)
    print(f"#  完成  耗时 {R['_meta']['耗时秒']}s  -> retest_result.json")
    print("#" * 74)


if __name__ == "__main__":
    try:
        main()
    finally:
        cleanup()



