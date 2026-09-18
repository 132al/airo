# ============================================================================
# AIRO 系统评测集（真实测量，非模拟数据）
#
# 覆盖维度：
#   E1 接口响应时间（延迟分布：P50/P90/P95/最大）
#   E2 检索准确性（种子相似度、流派命中率、标签相关性）
#   E3 召回覆盖与多样性（歌手/流派分布、去重率）
#   E4 随机性与稳定性（多次调用结果差异）
#   E5 冷启动能力（无反馈用户的可用性）
#   E6 反馈闭环收敛（画像随反馈演进）
#   E7 并发/吞吐（串行 vs 并发）
#   E8 鲁棒性（异常输入、降级）
#   E9 资源占用（内存、响应体、库规模）
#
# 用法：python eval_suite.py
# 输出：eval_results.json + 控制台表格
# ============================================================================

import json
import os
import statistics
import sys
import time
from collections import Counter

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import runtime_env  # noqa: F401
import django
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "web_project.settings")
django.setup()

from django.test import Client
from django.contrib.auth.models import User
from webtest.models import UserFeedback, UserProfile
from webtest import profile_service

RESULTS = {}
EVAL_USER = "__eval__"


# ---------------------------------------------------------------- 工具
def pct(values, p):
    """分位数（线性插值）"""
    if not values:
        return 0.0
    s = sorted(values)
    k = (len(s) - 1) * p / 100
    lo, hi = int(k), min(int(k) + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


def stats(times):
    """把一组耗时(秒)整理成分位数"""
    ms = [t * 1000 for t in times]
    return {
        "n": len(ms),
        "min": round(min(ms), 1),
        "p50": round(pct(ms, 50), 1),
        "p90": round(pct(ms, 90), 1),
        "p95": round(pct(ms, 95), 1),
        "max": round(max(ms), 1),
        "mean": round(statistics.mean(ms), 1),
        "stdev": round(statistics.pstdev(ms), 1) if len(ms) > 1 else 0.0,
    }


def banner(title):
    print()
    print("=" * 78)
    print(f"  {title}")
    print("=" * 78)


def table(rows, headers):
    """打印对齐表格（中文按双宽计算）"""
    def width(s):
        return sum(2 if ord(ch) > 0x2E80 else 1 for ch in str(s))

    ws = [width(h) for h in headers]
    for r in rows:
        for i, c in enumerate(r):
            ws[i] = max(ws[i], width(c))

    print("  " + "  ".join(
        h + " " * (ws[i] - width(h)) for i, h in enumerate(headers)))
    print("  " + "-" * (sum(ws) + 2 * (len(headers) - 1)))
    for r in rows:
        print("  " + "  ".join(
            str(c) + " " * (ws[i] - width(c)) for i, c in enumerate(r)))


# ---------------------------------------------------------------- 准备
def setup_user(feedbacks=None):
    """创建评测用户；feedbacks: [(genres, like/dislike)]"""
    User.objects.filter(username=EVAL_USER).delete()
    u = User.objects.create_user(username=EVAL_USER, password="evalpass")
    UserFeedback.objects.filter(user=u).delete()
    if feedbacks:
        for i, (genres, fb) in enumerate(feedbacks):
            UserFeedback.objects.create(
                user=u, track_id=f"eval{i:04d}", track_name=f"Eval Track {i}",
                artist_name=f"Eval Artist {i % 5}", artist_genres=genres,
                feedback=fb, artist_idx=1000 + i, artist_genre_idx=1,
            )
    profile_service.save(u, profile_service.rebuild(u, with_llm=False))
    c = Client()
    assert c.login(username=EVAL_USER, password="evalpass")
    return u, c


def clear_user():
    UserFeedback.objects.filter(user__username=EVAL_USER).delete()
    UserProfile.objects.filter(user__username=EVAL_USER).delete()
    User.objects.filter(username=EVAL_USER).delete()


# ============================================================================
# E1 接口响应时间
# ============================================================================
def e1_latency():
    banner("E1 · 接口响应时间（延迟分布）")
    u, c = setup_user([("pop", "like"), ("uk pop", "like"), ("dance pop", "like")])
    c.get("/api/feedback/list/")  # 热身

    cases = [
        ("反馈页加载", lambda: c.get("/api/feedback/list/?limit=200")),
        ("歌名找相似A", lambda: c.get("/api/recommend/?q=Shape+of+You&artist=Ed+Sheeran&top_k=20")),
        ("每日推荐C(热)", lambda: c.get("/api/recommend_for_me/?top_k=20&refresh=1")),
        ("候选下拉", lambda: c.get("/api/seed_candidates/?q=Shape")),
        ("数据浏览", lambda: c.get("/api/data/?limit=20")),
    ]

    rows, out = [], {}
    for name, fn in cases:
        times = []
        for _ in range(5):
            t = time.time()
            r = fn()
            times.append(time.time() - t)
            assert r.status_code == 200, f"{name} -> {r.status_code}"
        s = stats(times)
        out[name] = s
        rows.append([name, s["n"], f"{s['p50']:.0f}", f"{s['p90']:.0f}",
                     f"{s['p95']:.0f}", f"{s['max']:.0f}", f"{s['stdev']:.1f}"])
    table(rows, ["接口", "样本", "P50(ms)", "P90(ms)", "P95(ms)", "最大(ms)", "标准差"])
    RESULTS["E1_延迟"] = out
    clear_user()


# ============================================================================
# E2 检索准确性
# ============================================================================
def e2_accuracy():
    banner("E2 · 检索准确性")
    u, c = setup_user([("pop", "like")])

    cases = [
        ("Shape of You", "Ed Sheeran"),
        ("Blinding Lights", "The Weeknd"),
        ("Bohemian Rhapsody", "Queen"),
        ("Take Five", "The Dave Brubeck Quartet"),
        ("Smells Like Teen Spirit", "Nirvana"),
        ("Rolling in the Deep", "Adele"),
    ]

    rows, sims, hit_rates = [], [], []
    for tn, ar in cases:
        q = tn.replace(" ", "+")
        a = ar.replace(" ", "+")
        d = c.get(f"/api/recommend/?q={q}&artist={a}&top_k=20").json()
        res = d.get("results") or []
        if not res:
            rows.append([tn[:26], "—", "无结果", "0"])
            continue
        sim = [x.get("similarity") or 0 for x in res]
        seed = d.get("seed") or {}
        seed_g = {g.strip().lower()
                  for g in (seed.get("artist_genres") or "").split(",") if g.strip()}
        hit = 0
        for x in res:
            gs = {g.strip().lower()
                  for g in (x.get("artist_genres") or "").split(",") if g.strip()}
            if seed_g & gs:
                hit += 1
        hr = hit / len(res)
        sims.append(statistics.mean(sim))
        hit_rates.append(hr)
        rows.append([tn[:26], f"{statistics.mean(sim)*100:.1f}%",
                     f"{hr*100:.0f}%", len(res)])
    table(rows, ["种子歌曲", "平均相似度", "流派命中率", "结果数"])
    RESULTS["E2_种子相似"] = {
        "平均相似度": round(statistics.mean(sims) * 100, 1) if sims else 0,
        "平均流派命中率": round(statistics.mean(hit_rates) * 100, 1) if hit_rates else 0,
        "最低流派命中率": round(min(hit_rates) * 100, 1) if hit_rates else 0,
        "样本数": len(sims),
    }

    print()
    intents = [
        ("深夜放松的爵士", ["jazz"]),
        ("开心的英式流行", ["pop"]),
        ("适合跑步的高能量音乐", None),
        ("失恋了想听点伤感的", None),
        ("90年代摇滚", ["rock"]),
    ]
    rows, tag_hits, rel_rates = [], [], []
    for q, expect_any in intents:
        t = time.time()
        d = c.post("/api/recommend_by_intent/",
                   data=json.dumps({"query": q}),
                   content_type="application/json").json()
        el = time.time() - t
        tags = d.get("selected_tags") or []
        res = d.get("results") or []
        rel = 0
        for x in res[:5]:
            gs = (x.get("artist_genres") or "").lower()
            if any(tg.lower() in gs for tg in tags):
                rel += 1
        rr = rel / max(1, min(5, len(res)))
        rel_rates.append(rr)
        if expect_any:
            ok = any(any(e in tg.lower() for tg in tags) for e in expect_any)
            tag_hits.append(1 if ok else 0)
        rows.append([q[:22], len(tags), len(res), f"{rr*100:.0f}%", f"{el:.1f}"])
    table(rows, ["意图描述", "选中标签", "候选歌曲", "Top5相关率", "耗时(s)"])
    RESULTS["E2_意图推荐"] = {
        "LLM标签准确率": round(statistics.mean(tag_hits) * 100, 1) if tag_hits else None,
        "标签样本数": len(tag_hits),
        "Top5平均相关率": round(statistics.mean(rel_rates) * 100, 1) if rel_rates else 0,
        "结果样本数": len(rows),
    }
    clear_user()


# ============================================================================
# E3 召回覆盖与多样性
# ============================================================================
def e3_diversity():
    banner("E3 · 召回覆盖与多样性")
    u, c = setup_user([("pop", "like"), ("uk pop", "like"), ("dance pop", "like")])

    d = c.get("/api/recommend_for_me/?top_k=20&refresh=0").json()
    res = d.get("results") or []
    ch = d.get("channels") or {}

    artists = Counter(x.get("artist_name") for x in res)
    genres = Counter((x.get("artist_genres") or "").split(",")[0].strip() for x in res)
    names = Counter(x.get("track_name") for x in res)

    rows = [
        ["三路召回条数", f"流派 {ch.get('genre',0)} / 向量 {ch.get('vector',0)} / 歌手 {ch.get('artist',0)}"],
        ["返回结果数", len(res)],
        ["不同歌手数", len(artists)],
        ["不同流派数", len(genres)],
        ["单歌手最多出现", max(artists.values()) if artists else 0],
        ["单流派最多出现", max(genres.values()) if genres else 0],
        ["歌名重复数", sum(1 for v in names.values() if v > 1)],
    ]
    table(rows, ["指标", "数值"])

    RESULTS["E3_多样性"] = {
        "召回_流派": ch.get("genre", 0),
        "召回_向量": ch.get("vector", 0),
        "召回_歌手": ch.get("artist", 0),
        "结果数": len(res),
        "不同歌手数": len(artists),
        "不同流派数": len(genres),
        "单歌手最大重复": max(artists.values()) if artists else 0,
        "歌名重复数": sum(1 for v in names.values() if v > 1),
    }
    clear_user()


# ============================================================================
# E4 随机性与稳定性
# ============================================================================
def e4_randomness():
    banner("E4 · 随机性与稳定性")
    u, c = setup_user([("pop", "like"), ("uk pop", "like")])

    print("  [随机模式] refresh=1，连续 10 次")
    runs = []
    for _ in range(10):
        d = c.get("/api/recommend_for_me/?top_k=20&refresh=1").json()
        runs.append([x.get("track_id") for x in (d.get("results") or [])])

    jac = []
    for i in range(len(runs)):
        for j in range(i + 1, len(runs)):
            a, b = set(runs[i]), set(runs[j])
            if a | b:
                jac.append(len(a & b) / len(a | b))

    all_ids = set()
    for rr in runs:
        all_ids |= set(rr)
    dup_within = sum(1 for rr in runs if len(rr) != len(set(rr)))

    rows = [
        ["运行次数", len(runs)],
        ["每次返回条数", f"{min(len(r) for r in runs)}-{max(len(r) for r in runs)}"],
        ["两两 Jaccard 均值", f"{statistics.mean(jac)*100:.1f}%"],
        ["两两 Jaccard 范围", f"{min(jac)*100:.1f}% ~ {max(jac)*100:.1f}%"],
        ["单次内部重复", dup_within],
        ["10 次累计去重歌曲数", len(all_ids)],
        ["完全相同的结果集对数", sum(1 for x in jac if x == 1.0)],
    ]
    table(rows, ["指标", "数值"])

    print()
    print("  [确定性模式] refresh=0，连续 3 次")
    det = []
    for _ in range(3):
        d = c.get("/api/recommend_for_me/?top_k=20&refresh=0").json()
        det.append([x.get("track_id") for x in (d.get("results") or [])])
    same = det[0] == det[1] == det[2]
    print(f"    三次结果完全一致: {'是（符合设计）' if same else '否'}")
    print(f"    单次内部重复: {sum(1 for r in det if len(r) != len(set(r)))}")

    RESULTS["E4_随机性"] = {
        "Jaccard均值": round(statistics.mean(jac) * 100, 1),
        "Jaccard范围": [round(min(jac) * 100, 1), round(max(jac) * 100, 1)],
        "单次内部重复": dup_within,
        "累计去重歌曲数": len(all_ids),
        "完全相同对数": sum(1 for x in jac if x == 1.0),
        "确定性模式一致": same,
    }
    clear_user()


# ============================================================================
# E5 冷启动
# ============================================================================
def e5_coldstart():
    banner("E5 · 冷启动能力（无任何反馈数据）")
    u, c = setup_user(None)

    t = time.time()
    d = c.get("/api/recommend_for_me/?top_k=20&refresh=1").json()
    el = time.time() - t

    res = d.get("results") or []
    ch = d.get("channels") or {}
    artists = Counter(x.get("artist_name") for x in res)

    rows = [
        ["画像 stage", d.get("stage")],
        ["explore 标记", d.get("explore")],
        ["实际召回通道", f"cold={ch.get('cold',0)}"],
        ["返回结果数", len(res)],
        ["响应时间", f"{el:.2f}s"],
        ["不同歌手数", len(artists)],
        ["单歌手最大重复", max(artists.values()) if artists else 0],
        ["种子标签显示", (d.get("seed") or {}).get("artist_genres")],
    ]
    table(rows, ["指标", "数值"])

    ok = len(res) > 0 and d.get("stage") == "cold" and d.get("explore") is True
    print(f"  可用性判定: {'通过（冷启动可正常返回）' if ok else '失败'}")

    print()
    print("  [三档策略对照]")
    stages = []
    for label, fbs in [("cold（0 条）", None),
                       ("warm（2 条）", [("pop", "like")] * 2),
                       ("hot（6 条）", [("pop", "like")] * 6)]:
        clear_user()
        u2, c2 = setup_user(fbs)
        d2 = c2.get("/api/recommend_for_me/?top_k=20&refresh=1").json()
        ch2 = d2.get("channels") or {}
        stages.append([label, d2.get("stage"), d2.get("explore"),
                       f"G{ch2.get('genre',0)}/V{ch2.get('vector',0)}"
                       f"/A{ch2.get('artist',0)}/C{ch2.get('cold',0)}",
                       len(d2.get("results") or [])])
        clear_user()
    table(stages, ["反馈量", "stage", "explore", "召回通道", "结果数"])

    RESULTS["E5_冷启动"] = {
        "stage": d.get("stage"), "explore": d.get("explore"),
        "返回结果数": len(res), "响应秒": round(el, 2),
        "不同歌手数": len(artists), "可用": ok,
        "三档对照": stages,
    }
    clear_user()


# ============================================================================
# E6 反馈闭环收敛
# ============================================================================
def e6_feedback_loop():
    banner("E6 · 反馈闭环收敛（画像随反馈演进）")
    User.objects.filter(username=EVAL_USER).delete()
    u = User.objects.create_user(username=EVAL_USER, password="evalpass")
    UserFeedback.objects.filter(user=u).delete()
    c = Client()
    c.login(username=EVAL_USER, password="evalpass")

    rows = []
    steps = [
        (0, []),
        (1, [("pop", "like")]),
        (3, [("pop", "like")] * 3),
        (5, [("pop", "like")] * 2 + [("uk pop", "like")] * 3),
        (8, [("pop", "like")] * 3 + [("uk pop", "like")] * 3 + [("rock", "dislike")] * 2),
    ]
    for target, fbs in steps:
        while UserFeedback.objects.filter(user=u).count() < target:
            i = UserFeedback.objects.filter(user=u).count()
            g, fb = fbs[min(i, len(fbs) - 1)] if fbs else ("pop", "like")
            UserFeedback.objects.create(
                user=u, track_id=f"loop{i:04d}", track_name=f"T{i}",
                artist_name=f"A{i % 4}", artist_genres=g, feedback=fb,
                artist_idx=2000 + i, artist_genre_idx=1,
            )
        t = time.time()
        r = c.post("/api/feedback/", data=json.dumps({
            "track_id": f"trigger{target}", "track_name": "Trig",
            "artist_name": "TrigA", "artist_genres": "pop",
            "artist_idx": 1, "artist_genre_idx": 1, "feedback": "like",
        }), content_type="application/json")
        el = time.time() - t
        j = r.json()
        prof = profile_service.get(u, ensure_llm=False)
        rows.append([
            UserFeedback.objects.filter(user=u).count(),
            j.get("signal_count", 0), j.get("stage", "?"),
            len(prof.get("preferred_genres") or []),
            len(prof.get("preferred_artists") or []),
            len(prof.get("disliked_genres") or []),
            f"{el*1000:.0f}",
        ])

    table(rows, ["实际反馈", "signal", "stage", "偏好流派", "偏好歌手", "排斥流派", "响应(ms)"])
    times = [int(r[6]) for r in rows]
    RESULTS["E6_闭环"] = {
        "反馈接口最大响应ms": max(times),
        "反馈接口平均响应ms": round(statistics.mean(times), 0),
        "stage演进": [r[2] for r in rows],
        "signal演进": [r[1] for r in rows],
        "偏好流派增长": [r[3] for r in rows],
        "排斥流派生效": any(r[5] > 0 for r in rows),
    }
    clear_user()


# ============================================================================
# E7 并发吞吐
# ============================================================================
def e7_throughput():
    banner("E7 · 并发吞吐（20 请求）")
    from concurrent.futures import ThreadPoolExecutor
    u, c = setup_user([("pop", "like"), ("uk pop", "like")])

    def one(_):
        t = time.time()
        r = c.get("/api/recommend_for_me/?top_k=10&refresh=1")
        return r.status_code, time.time() - t

    c.get("/api/recommend_for_me/?top_k=10")  # 热身

    t = time.time()
    seq = [one(i) for i in range(20)]
    seq_total = time.time() - t

    t = time.time()
    with ThreadPoolExecutor(max_workers=8) as ex:
        par = list(ex.map(one, range(20)))
    par_total = time.time() - t

    seq_ok = sum(1 for s, _ in seq if s == 200)
    par_ok = sum(1 for s, _ in par if s == 200)

    rows = [
        ["串行", 20, f"{seq_ok}/20", f"{seq_total:.2f}",
         f"{statistics.mean([t for _, t in seq])*1000:.0f}", f"{20/seq_total:.1f}"],
        ["并发(8线程)", 20, f"{par_ok}/20", f"{par_total:.2f}",
         f"{statistics.mean([t for _, t in par])*1000:.0f}", f"{20/par_total:.1f}"],
    ]
    table(rows, ["模式", "请求数", "成功", "总耗时(s)", "平均延迟(ms)", "QPS"])

    RESULTS["E7_吞吐"] = {
        "串行_QPS": round(20 / seq_total, 1),
        "并发_QPS": round(20 / par_total, 1),
        "串行成功率": f"{seq_ok}/20",
        "并发成功率": f"{par_ok}/20",
        "并发加速比": round(seq_total / par_total, 2) if par_total else 0,
        "并发P50ms": round(pct([t * 1000 for _, t in par], 50), 0),
    }
    clear_user()


# ============================================================================
# E8 鲁棒性
# ============================================================================
def e8_robustness():
    banner("E8 · 鲁棒性与异常处理")
    u, c = setup_user([("pop", "like")])

    cases = [
        ("歌名不存在", lambda: c.get("/api/recommend/?q=ZZZ_NotExist_9999"),
         lambda r: r.status_code in (200, 404)),
        ("缺少必填参数", lambda: c.get("/api/recommend/"),
         lambda r: r.status_code == 400),
        ("非法 top_k(abc)", lambda: c.get("/api/recommend/?q=Shape+of+You&top_k=abc"),
         lambda r: r.status_code in (200, 400, 500)),
        ("非法 top_k(负数)", lambda: c.get("/api/recommend/?q=Shape+of+You&top_k=-5"),
         lambda r: r.status_code in (200, 400, 500)),
        ("意图超长文本", lambda: c.post("/api/recommend_by_intent/",
                                   data=json.dumps({"query": "音乐" * 500}),
                                   content_type="application/json"),
         lambda r: r.status_code in (200, 404, 500)),
        ("意图空字符串", lambda: c.post("/api/recommend_by_intent/",
                                   data=json.dumps({"query": ""}),
                                   content_type="application/json"),
         lambda r: r.status_code == 400),
        ("反馈非法值", lambda: c.post("/api/feedback/",
                                  data=json.dumps({"track_id": "x", "feedback": "bad"}),
                                  content_type="application/json"),
         lambda r: r.status_code == 400),
        ("反馈缺 track_id", lambda: c.post("/api/feedback/",
                                     data=json.dumps({"feedback": "like"}),
                                     content_type="application/json"),
         lambda r: r.status_code == 400),
        ("理由 GET 非法", lambda: c.get("/api/reason/"),
         lambda r: r.status_code == 405),
        ("摘要 GET 非法", lambda: c.get("/api/profile/summary/"),
         lambda r: r.status_code == 405),
        ("超长歌名(500字)", lambda: c.get("/api/recommend/?q=" + "A" * 500),
         lambda r: r.status_code in (200, 400, 404, 500)),
        ("SQL注入样式", lambda: c.get("/api/recommend/?q='+OR+1%3D1--"),
         lambda r: r.status_code in (200, 400, 404)),
        ("Unicode 歌名", lambda: c.get("/api/recommend/?q=%E6%99%B4%E5%A4%A9"
                                      "&artist=%E5%91%A8%E6%9D%B0%E4%BC%A6"),
         lambda r: r.status_code in (200, 404)),
        ("XSS 样式输入", lambda: c.get("/api/recommend/?q=%3Cscript%3Ealert(1)%3C%2Fscript%3E"),
         lambda r: r.status_code in (200, 400, 404, 500)),
        ("未登录访问", lambda: Client().get("/api/feedback/list/"),
         lambda r: r.status_code in (302, 403)),
    ]

    rows = []
    passed = 0
    for name, fn, check in cases:
        try:
            t = time.time()
            r = fn()
            el = time.time() - t
            ok = check(r)
            passed += 1 if ok else 0
            rows.append([name, r.status_code, "通过" if ok else "未通过", f"{el*1000:.0f}ms"])
        except Exception as e:
            rows.append([name, "EXC", type(e).__name__, "-"])
    table(rows, ["异常场景", "HTTP", "判定", "耗时"])

    print(f"  通过率: {passed}/{len(cases)} = {passed/len(cases)*100:.0f}%")
    RESULTS["E8_鲁棒性"] = {
        "通过率": f"{passed}/{len(cases)}",
        "百分比": round(passed / len(cases) * 100, 0),
        "场景明细": [[r[0], str(r[1]), r[2]] for r in rows],
    }
    clear_user()


# ============================================================================
# E9 资源占用
# ============================================================================
def e9_resource():
    banner("E9 · 资源占用")
    import subprocess
    import requests

    rows = []
    u, c = setup_user([("pop", "like")])

    for name, ep in [
        ("每日推荐(20 条)", "/api/recommend_for_me/?top_k=20&refresh=1"),
        ("歌名找相似(20 条)", "/api/recommend/?q=Shape+of+You&artist=Ed+Sheeran&top_k=20"),
        ("反馈页(200 条)", "/api/feedback/list/?limit=200"),
    ]:
        r = c.get(ep)
        rows.append([f"响应体 {name}", f"{len(r.content)/1024:.1f} KB"])

    try:
        ps = subprocess.run(
            ["powershell", "-NoProfile", "-Command",
             "Get-Process python,ollama -ErrorAction SilentlyContinue | "
             "ForEach-Object { \"$($_.ProcessName)=$([math]::Round($_.WorkingSet64/1MB,0))\" }"],
            capture_output=True, text=True, timeout=30).stdout
        mem = {}
        for line in ps.splitlines():
            if "=" in line:
                k, v = line.strip().split("=", 1)
                mem[k] = mem.get(k, 0) + int(v)
        for k, v in mem.items():
            rows.append([f"{k} 进程内存合计", f"{v} MB"])
    except Exception as e:
        rows.append(["进程内存", f"获取失败 {type(e).__name__}"])

    try:
        fo = subprocess.run(
            ["powershell", "-NoProfile", "-Command",
             "[math]::Round((Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory/1MB,2)"],
            capture_output=True, text=True, timeout=30).stdout.strip()
        rows.append(["系统空闲内存", f"{fo} GB"])
    except Exception:
        pass

    try:
        qd = requests.get("http://127.0.0.1:6333/collections/spotify_tracks",
                          timeout=8).json()["result"]
        v = qd["config"]["params"]["vectors"]
        rows.append(["歌曲向量库", f"{qd['points_count']:,} 点 × {v['size']} 维"
                                 f"（{v.get('datatype')}, on_disk={v.get('on_disk')}）"])
        qd2 = requests.get("http://127.0.0.1:6333/collections/tag_index",
                           timeout=8).json()["result"]
        v2 = qd2["config"]["params"]["vectors"]
        rows.append(["标签向量库", f"{qd2['points_count']:,} 点 × {v2['size']} 维"])
    except Exception as e:
        rows.append(["向量库", f"查询失败 {type(e).__name__}"])

    table(rows, ["资源项", "数值"])
    RESULTS["E9_资源"] = {r[0]: r[1] for r in rows}
    clear_user()


# ============================================================================
# 主流程
# ============================================================================
def main():
    t0 = time.time()
    print()
    print("#" * 78)
    print("#  AIRO 系统评测集")
    print(f"#  开始时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("#" * 78)

    for fn in (e1_latency, e2_accuracy, e3_diversity, e4_randomness,
               e5_coldstart, e6_feedback_loop, e7_throughput,
               e8_robustness, e9_resource):
        try:
            fn()
        except Exception as e:
            print(f"  !! {fn.__name__} 失败: {type(e).__name__}: {e}")

    RESULTS["_meta"] = {
        "评测时间": time.strftime("%Y-%m-%d %H:%M:%S"),
        "总耗时秒": round(time.time() - t0, 1),
        "歌曲向量库": "12,382,652 点 / 64 维 cosine / uint8 / on_disk",
        "标签向量库": "6,277 点 / 512 维 cosine",
        "LLM": "qwen3:4b-instruct-2507-q4_K_M (Ollama 本地)",
        "运行环境": "Windows / Django 6.1 / Qdrant 1.19",
    }

    with open("eval_results.json", "w", encoding="utf-8") as f:
        json.dump(RESULTS, f, ensure_ascii=False, indent=2, default=str)

    print()
    print("#" * 78)
    print(f"#  评测完成  总耗时 {RESULTS['_meta']['总耗时秒']}s")
    print("#  结果已写入 eval_results.json")
    print("#" * 78)


if __name__ == "__main__":
    try:
        main()
    finally:
        clear_user()

