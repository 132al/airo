# ============================================================================
# 向量相似度校准 · 回归测试
#
# 背景（这是本测试要守护的核心事实）：
#   spotify_tracks 的 64 维向量每维都在 128 附近且全为正数，
#   导致任意两首的余弦被"共同基线"抬到 ~0.986：
#       随机对  mean=0.9862  sd=0.0049   ← 几乎是常数
#   因此 similarity=0.997 显示为"99.7% 相似"是几何伪影，随机抽也有 98.6%。
#
# 修复：去均值（centering）后再算余弦 -> cal_similarity
#       随机基线回到 ≈0，推荐落在 0.2~0.8，于是数字才有刻度意义。
#
# 用法：python test_calibration.py
# ============================================================================

import math
import os
import statistics
import sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import runtime_env  # noqa: F401
import django
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "web_project.settings")
django.setup()

from django.test import Client
from django.contrib.auth.models import User
from qdrant_client import QdrantClient
from webtest.models import UserFeedback, UserProfile
from webtest import vector_calibration as vc

PASS, FAIL = [], []


def check(name, cond, detail=""):
    (PASS if cond else FAIL).append(name)
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f"  {detail}" if detail else ""))


def cleanup():
    UserFeedback.objects.filter(user__username="__cal__").delete()
    UserProfile.objects.filter(user__username="__cal__").delete()
    User.objects.filter(username="__cal__").delete()


def main():
    print("#" * 76)
    print("#  向量相似度校准 · 回归测试")
    print("#" * 76)

    # ---------- 1. 均值向量可用 ----------
    print()
    print("1 · 均值向量")
    mu = vc.get_mean_vec()
    check("均值向量长度 = 64", len(mu) == 64, f"len={len(mu)}")
    m = statistics.mean(mu)
    check("均值向量均值在 100~160（符合 128 基线假设）",
          100 < m < 160, f"mean={m:.2f}")

    # ---------- 2. 去均值确实把随机基线拉回 0 ----------
    print()
    print("2 · 去均值效果（核心断言）")
    qc = QdrantClient(url="http://127.0.0.1:6333", timeout=300)
    V = []
    off = None
    for _ in range(220):
        pts, nxt = qc.scroll("spotify_tracks", limit=1, offset=off,
                             with_vectors=True, with_payload=False)
        if not pts:
            break
        if pts[0].vector:
            V.append([float(x) for x in pts[0].vector])
        off = nxt
    check("取到 >=100 条向量", len(V) >= 100, f"n={len(V)}")

    raw, cal = [], []
    for i in range(0, min(200, len(V)), 2):
        for j in range(i + 1, min(200, len(V)), 2):
            raw.append(vc.cos(V[i], V[j]))
            cal.append(vc.calibrated_cos(V[i], V[j], mu))

    raw_sd = statistics.pstdev(raw)
    cal_sd = statistics.pstdev(cal)
    raw_mean = statistics.mean(raw)
    cal_mean = statistics.mean(cal)

    print(f"    原始余弦 : mean={raw_mean:.4f}  sd={raw_sd:.4f}")
    print(f"    去均值后 : mean={cal_mean:.4f}  sd={cal_sd:.4f}")

    check("原始余弦均值 > 0.95（证实'被常数占领'）",
          raw_mean > 0.95, f"{raw_mean:.4f}")
    check("去均值后均值 |mean| < 0.05（基线归零）",
          abs(cal_mean) < 0.05, f"{cal_mean:.4f}")
    check("去均值后区分度提升 >= 20x",
          cal_sd / raw_sd >= 20, f"x{cal_sd/raw_sd:.1f}")

    # ---------- 3. API 暴露 cal_similarity / sim_percent ----------
    print()
    print("3 · API 契约")
    cleanup()
    u = User.objects.create_user(username="__cal__", password="x")
    c = Client()
    c.login(username="__cal__", password="x")
    d = c.get("/api/recommend/?q=Shape+of+You&artist=Ed+Sheeran&top_k=20").json()
    res = d.get("results") or []
    check("返回结果非空", len(res) > 0, f"n={len(res)}")

    if res:
        has_cal = all(r.get("cal_similarity") is not None for r in res)
        has_pct = all(r.get("sim_percent") is not None for r in res)
        check("每条结果都有 cal_similarity", has_cal)
        check("每条结果都有 sim_percent（冷启动也要有）", has_pct)

        raw_v = [r["similarity"] for r in res]
        cal_v = [r["cal_similarity"] for r in res if r.get("cal_similarity") is not None]
        pct_v = [r["sim_percent"] for r in res if r.get("sim_percent") is not None]

        if cal_v:
            check("校准值落在 (-1, 1) 合法区间",
                  all(-1 <= x <= 1 for x in cal_v))
            check("校准值区分度 (span) 远大于原始余弦",
                  (max(cal_v) - min(cal_v)) > (max(raw_v) - min(raw_v)) * 10,
                  f"cal span={max(cal_v)-min(cal_v):.4f} vs raw span={max(raw_v)-min(raw_v):.4f}")
            check("推荐结果校准均值显著高于 0（确实相关）",
                  statistics.mean(cal_v) > 0.1, f"mean={statistics.mean(cal_v):.4f}")
        if pct_v:
            check("sim_percent 在 0~100 区间",
                  all(0 <= x <= 100 for x in pct_v),
                  f"range=[{min(pct_v)}, {max(pct_v)}]")

    cleanup()

    # ---------- 4. 边界 ----------
    print()
    print("4 · 边界与降级")
    check("center(None) 返回 None", vc.center(None) is None)
    check("cos 长度不等返回 0", vc.cos([1, 2], [1, 2, 3]) == 0.0)
    check("cos 空向量返回 0", vc.cos([], []) == 0.0)
    z = [0.0] * 64
    check("cos 零向量返回 0", vc.cos(z, z) == 0.0)
    check("to_percent(None) = 0", vc.to_percent(None) == 0.0)
    check("to_percent 下限钳制为 0", vc.to_percent(-9.0) == 0.0)
    check("to_percent 上限钳制为 100", vc.to_percent(9.0) == 100.0)

    # ---------- 5. 去重与同歌手限流（本次顺带修的 bug） ----------
    print()
    print("5 · 重复版本过滤")
    import webtest.embeat_similar as es_
    dup_cases = [
        ("Bohemian Rhapsody - Remastered 2011", True),
        ("Song (Live)", True),
        ("X [Deluxe Edition]", True),
        ("Track - Radio Edit", True),
        ("A (Remix)", True),
        ("Shape of You", False),
        ("Live and Let Die", False),      # 歌名本身含 Live，不能误伤
        ("Version of Me", False),         # 歌名本身含 Version，不能误伤
        ("Bohemian Rhapsody", False),
    ]
    bad = [s for s, exp in dup_cases if es_._is_dup_version(s) != exp]
    check("_is_dup_version 判定全部正确（含防误伤用例）", not bad,
          f"错误: {bad}" if bad else f"{len(dup_cases)}/{len(dup_cases)}")

    print()
    print("6 · 推荐结果质量（限流生效）")
    cleanup()
    u = User.objects.create_user(username="__cal__", password="x")
    c = Client()
    c.login(username="__cal__", password="x")
    for tn, ar in [("Bohemian Rhapsody", "Queen"),
                   ("Shape of You", "Ed Sheeran"),
                   ("Take Five", "The Dave Brubeck Quartet")]:
        d = c.get(f"/api/recommend/?q={tn.replace(' ', '+')}"
                  f"&artist={ar.replace(' ', '+')}&top_k=20").json()
        res = d.get("results") or []
        if not res:
            continue
        arts = [x.get("artist_name") for x in res]
        mx = max(arts.count(a) for a in set(arts))
        dupv = sum(1 for x in res if es_._is_dup_version(str(x.get("track_name", "")).lower()))
        check(f"{tn[:18]} 无重复版本", dupv == 0, f"dup={dupv}")
        check(f"{tn[:18]} 单歌手 <= 3", mx <= 3, f"max={mx}")

    cleanup()

    # ---------- 汇总 ----------
    print()
    print("=" * 76)
    print(f"  通过 {len(PASS)} / {len(PASS) + len(FAIL)}")
    if FAIL:
        print("  失败项:")
        for f in FAIL:
            print(f"    - {f}")
    print("=" * 76)
    return 0 if not FAIL else 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    finally:
        cleanup()
