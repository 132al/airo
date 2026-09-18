# ============================================================================
# T2 重构版 · 检索准确性（脱离"流派字符串命中"判断）
#
# 为什么换掉旧方法
# ----------------
# 旧报告用 "候选的 artist_genres 是否包含种子的 artist_genres" 判准确率，
# 得到 100%。但系统内部 filter_candidates() 本身就做了同样的硬过滤：
#     common = seed_genres_set & payload_genres_set
#     if not common: continue      # 无交集直接淘汰
# 所以"流派命中率 100%"是自证（tautology），不构成准确性证据。
#
# 新方法：只看向量与第三方信号
# ----------------------------
# A. 声学距离     候选与种子的 64 维余弦（直接用 API 的 similarity 字段，
#                 它就是 Qdrant 的 candidate.score，非流派）
# B. 随机基线     从全库随机抽 30 首，算同一种子的余弦 → 量化"检索 vs 瞎猜"
# C. 结果自洽     结果集内部两两余弦 → 是否成簇、有没有混进无关歌
# D. 歌手隔离     结果里有几首是种子本人 → 排除"只是同歌手"的解释
# E. 关联歌手     related_artist_idxs（Spotify 官方关联表，与流派无关）
# F. 意图共识     多个语义不同但指向相近的 query 是否召回重叠的结果
#
# 用法：python t2_acoustic.py  →  t2_acoustic_result.json
# ============================================================================

import json
import math
import os
import random
import statistics
import sys
import time

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import runtime_env  # noqa: F401
import django
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "web_project.settings")
django.setup()

from django.test import Client
from django.contrib.auth.models import User
from qdrant_client import QdrantClient
from qdrant_client.http import models as qm
from webtest.models import UserFeedback, UserProfile
from webtest import profile_service

R = {}
U = "__acoust__"
COLL = "spotify_tracks"
QC = QdrantClient(url="http://127.0.0.1:6333", timeout=120)
random.seed(20260918)

SEEDS = [
    ("Shape of You", "Ed Sheeran"),
    ("Bohemian Rhapsody", "Queen"),
    ("Take Five", "The Dave Brubeck Quartet"),
    ("Blinding Lights", "The Weeknd"),
    ("Lose Yourself", "Eminem"),
]


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


def line(t):
    print()
    print("=" * 80)
    print("  " + t)
    print("=" * 80)


def cleanup():
    UserFeedback.objects.filter(user__username=U).delete()
    UserProfile.objects.filter(user__username=U).delete()
    User.objects.filter(username=U).delete()


def setup():
    cleanup()
    u = User.objects.create_user(username=U, password="ac")
    for i, g in enumerate(["pop", "uk pop", "dance pop"]):
        UserFeedback.objects.create(
            user=u, track_id=f"ac{i:03d}", track_name=f"T{i}", artist_name=f"A{i}",
            artist_genres=g, feedback="like", artist_idx=7000 + i, artist_genre_idx=1)
    profile_service.save(u, profile_service.rebuild(u, with_llm=False))
    c = Client()
    c.login(username=U, password="ac")
    return u, c


def rand_vecs(n):
    """从全库随机抽 n 个向量（用 offset 随机跳，避免全表扫描）。"""
    out = []
    for _ in range(n * 3):
        if len(out) >= n:
            break
        try:
            pts, _ = QC.scroll(COLL, limit=1, offset=random.randrange(12_300_000),
                               with_vectors=True, with_payload=False)
        except Exception:
            continue
        if pts and pts[0].vector:
            out.append([float(x) for x in pts[0].vector])
    return out


def vec_of_seed(track_name, artist_name):
    """按歌名+歌手取种子向量（track_name/artist_name 都有 payload 索引）。"""
    try:
        pts, _ = QC.scroll(
            COLL,
            scroll_filter=qm.Filter(must=[
                qm.FieldCondition(key="track_name", match=qm.MatchText(text=track_name)),
                qm.FieldCondition(key="artist_name", match=qm.MatchText(text=artist_name)),
            ]),
            limit=5, with_vectors=True, with_payload=True)
    except Exception as e:
        print(f"  [种子向量失败] {e}")
        return None, None
    best = None
    for p in pts:
        if not p.vector:
            continue
        nm = str((p.payload or {}).get("track_name", "")).lower()
        if "remix" in nm:
            continue
        best = p
        break
    if best is None:
        return None, None
    return [float(x) for x in best.vector], (best.payload or {})


def part_a_acoustic():
    line("A · 种子相似 —— 64 维声学余弦（与流派字段无关）")
    print("  候选↔种子余弦直接取自 API 返回的 similarity 字段")
    print("  （即 Qdrant candidate.score，是真实声学距离，不是流派匹配）")
    print()
    u, c = setup()
    c.get("/api/recommend_for_me/?top_k=10")

    rows = []
    all_sim, all_base, all_pair = [], [], []
    for tn, ar in SEEDS:
        d = c.get(f"/api/recommend/?q={tn.replace(' ', '+')}"
                  f"&artist={ar.replace(' ', '+')}&top_k=20").json()
        res = d.get("results") or []
        seed = d.get("seed") or {}
        if not res:
            rows.append([tn[:20], "-", "-", "-", "-", "-"])
            continue

        sim = [float(x.get("similarity") or 0) for x in res]
        sv, spay = vec_of_seed(tn, ar)

        # 随机基线：从全库随机抽 30 首，算它们与种子的声学余弦
        rv = rand_vecs(30)
        base = []
        if sv:
            for v in rv:
                dot = sum(a * b for a, b in zip(v, sv))
                na = math.sqrt(sum(a * a for a in v))
                nb = math.sqrt(sum(b * b for b in sv))
                base.append(dot / (na * nb) if na and nb else 0)

        # 结果自洽：结果之间声学是否成簇（用 similarity 的离散度代理）
        pair = [max(0.0, 1.0 - abs(sim[i] - sim[j]))
                for i in range(len(sim)) for j in range(i + 1, len(sim))]

        self_cnt = sum(1 for x in res
                       if str(x.get("artist_name", "")).lower().strip()
                       == str(seed.get("artist_name", "")).lower().strip())

        all_sim += sim
        all_base += base
        all_pair += pair
        rows.append([tn[:20], f"{statistics.mean(sim):.3f}",
                     f"{min(sim):.3f}",
                     f"{statistics.mean(base):.3f}" if base else "-",
                     f"{statistics.mean(pair):.3f}", f"{self_cnt}/{len(res)}"])

    table(rows, ["种子", "候选↔种子cos 均值", "最低", "随机基线cos",
                 "结果自洽", "种子本人"])

    if all_sim and all_base:
        lift = statistics.mean(all_sim) - statistics.mean(all_base)
        print()
        print(f"  候选↔种子 平均余弦 : {statistics.mean(all_sim):.3f}  (最低 {min(all_sim):.3f})")
        print(f"  随机基线   平均余弦 : {statistics.mean(all_base):.3f}")
        print(f"  >>> 相对随机提升     : +{lift:.3f}   (提升 {lift/statistics.mean(all_base)*100:.0f}%)")
        R["A_声学"] = {
            "候选↔种子cos均值": round(statistics.mean(all_sim), 4),
            "最低cos": round(min(all_sim), 4),
            "随机基线cos": round(statistics.mean(all_base), 4),
            "相对提升": round(lift, 4),
            "提升倍数": round(1 + lift / statistics.mean(all_base), 3)}
    cleanup()


def part_e_related():
    line("E · 关联歌手命中 —— 独立于流派的第三方信号")
    print("  related_artist_idxs 是 Spotify 官方关联歌手表（谁和谁相似）。")
    print("  它和 artist_genres 是两个独立字段：一个歌手可以流派不同但被官方判为关联。")
    print("  用 'artist_idx 是否落在种子的关联表' 检验 —— 这条通路不经过流派过滤。")
    print()
    u, c = setup()
    rows, tot_hit, tot_n = [], 0, 0
    for tn, ar in SEEDS:
        d = c.get(f"/api/recommend/?q={tn.replace(' ', '+')}"
                  f"&artist={ar.replace(' ', '+')}&top_k=20").json()
        res = d.get("results") or []
        sv, spay = vec_of_seed(tn, ar)
        if not res or not spay:
            rows.append([tn[:20], "-", "-", "-"])
            continue
        rel = set(int(x) for x in (spay.get("related_artist_idxs") or []))
        if not rel:
            rows.append([tn[:20], "无关联表", "-", "-"])
            continue
        # 候选的 artist_idx 不在 API 响应里 → 用歌名+歌手反查
        hit = 0
        for x in res:
            _, cpay = vec_of_seed(x.get("track_name", ""), x.get("artist_name", ""))
            if cpay and int(cpay.get("artist_idx") or -1) in rel:
                hit += 1
        tot_hit += hit
        tot_n += len(res)
        rows.append([tn[:20], len(rel), f"{hit}/{len(res)}", f"{hit/len(res)*100:.0f}%"])
    table(rows, ["种子", "官方关联歌手数", "命中", "命中率"])
    if tot_n:
        print()
        print(f"  >>> 关联歌手平均命中率: {tot_hit/tot_n*100:.1f}%  ({tot_hit}/{tot_n})")
        R["E_关联歌手"] = {"命中率": round(tot_hit / tot_n * 100, 1),
                        "命中": tot_hit, "总数": tot_n}
    cleanup()


def part_f_intent():
    line("F · 意图理解 —— 语义等价/近似 query 的结果重叠")
    print("  不查流派。改用'同一意图的不同说法是否召回相似结果'来判断。")
    print("  若系统真正理解语义，'深夜放松的爵士' 与 '晚上一个人听的爵士' 应显著重叠。")
    print()
    u, c = setup()

    groups = [
        ("爵士放松", ["深夜放松的爵士", "晚上一个人听的爵士", "安静的爵士乐"]),
        ("摇滚", ["90年代摇滚", "老摇滚", "经典摇滚乐"]),
        ("跑步", ["适合跑步的高能量音乐", "跑步时听的快节奏歌", "运动健身音乐"]),
    ]
    rows, grp_scores = [], []
    for gname, queries in groups:
        sets = []
        for q in queries:
            d = c.post("/api/recommend_by_intent/",
                       data=json.dumps({"query": q}),
                       content_type="application/json").json()
            ids = [x.get("track_id") for x in (d.get("results") or [])[:20]]
            sets.append(set(i for i in ids if i))
        jac = []
        for i in range(len(sets)):
            for j in range(i + 1, len(sets)):
                if sets[i] | sets[j]:
                    jac.append(len(sets[i] & sets[j]) / len(sets[i] | sets[j]))
        m = statistics.mean(jac) if jac else 0
        grp_scores.append(m)
        rows.append([gname, len(queries), f"{m*100:.0f}%",
                     f"{min(jac)*100:.0f}%" if jac else "-",
                     f"{max(jac)*100:.0f}%" if jac else "-"])
    table(rows, ["意图组", "问法数", "平均 Jaccard", "最低", "最高"])
    print()
    print("  对照：不同意图组之间（应显著更低）")
    # 跨组对照
    allsets = {}
    for gname, queries in groups:
        d = c.post("/api/recommend_by_intent/",
                   data=json.dumps({"query": queries[0]}),
                   content_type="application/json").json()
        allsets[gname] = set(x.get("track_id") for x in (d.get("results") or [])[:20])
    names = list(allsets)
    cross = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = allsets[names[i]], allsets[names[j]]
            if a | b:
                cross.append(len(a & b) / len(a | b))
    print(f"  组内平均 Jaccard : {statistics.mean(grp_scores)*100:.0f}%")
    print(f"  组间平均 Jaccard : {statistics.mean(cross)*100:.0f}%")
    denom = statistics.mean(cross) if cross else 0.0
    ratio = (statistics.mean(grp_scores) / denom) if denom > 0 else None
    if ratio:
        print(f"  >>> 组内/组间 倍数 : {ratio:.2f}x")
    else:
        print("  >>> 组间重叠为 0 —— 组内/组间不可比（组间无任何共享歌曲）")
    R["F_意图语义"] = {
        "组内Jaccard": round(statistics.mean(grp_scores) * 100, 1),
        "组间Jaccard": round(denom * 100, 1),
        "组内组间倍数": round(ratio, 2) if ratio else None}
    cleanup()



if __name__ == "__main__":
    t0 = time.time()
    print("#" * 80)
    print("#  T2 声学版 · 检索准确性（不用流派字段判断）  " + time.strftime("%Y-%m-%d %H:%M"))
    print("#" * 80)
    for fn in (part_a_acoustic, part_e_related, part_f_intent):
        try:
            fn()
        except Exception as e:
            import traceback
            print(f"  !! {fn.__name__}: {type(e).__name__}: {e}")
            traceback.print_exc()
    R["_meta"] = {"时间": time.strftime("%Y-%m-%d %H:%M:%S"),
                  "耗时秒": round(time.time() - t0, 1)}
    with open("t2_acoustic_result.json", "w", encoding="utf-8") as f:
        json.dump(R, f, ensure_ascii=False, indent=2, default=str)
    print()
    print("=" * 80)
    print(f"  完成  耗时 {R['_meta']['耗时秒']}s  ->  t2_acoustic_result.json")
    print("=" * 80)


