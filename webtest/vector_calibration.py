# webtest/vector_calibration.py
"""
向量相似度校准 —— 把"被常数占领的余弦"还原成可解释的相似度刻度。

问题
----
本库的 64 维声学向量每维都在 128 附近波动且**全为正数**
（实测：逐维均值 127.6，范围 80~165）。这导致任意两条向量的余弦
被"共同基线"抬到极高：

    随机两首 余弦 mean = 0.9862, sd = 0.0049   ← 几乎是常数

后果：
- "相似度 99.7%" 看起来很高，但随机抽也有 98.6% → 数字无刻度意义
- 真实音频差异信号只体现在小数点后第三位（sd 0.0049）
- 增加小数位数**无济于事**，因为信号本身就这么小

解法：去均值（centering）
-------------------------
对每条向量减去全库逐维均值 mu，再算余弦：

    cos_centered(a, b) = cos(a - mu, b - mu)

效果（实测）：
    随机对 sd     0.0049  ->  0.3401   （区分度提升 69.8x）
    随机基线均值  0.9862  ->  -0.0001
    推荐结果      0.9975  ->  mean 0.688, 范围 0.199~0.799

于是 0.69 才真正可读作"约 69% 相似"，且随机基线为 0，有明确零点。

mu 的来源
---------
优先用 build_alias_cache 同款思路：预计算一次存盘（快）。
没有缓存时现场抽样估计（准，但要几百次 scroll）。
"""

import json
import math
import os
import statistics

MU_CACHE_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "vector_mean_64.json")

DIM = 64
_QDRANT_URL = "http://127.0.0.1:6333"
_COLLECTION = "spotify_tracks"

_mu_cache = None


def get_mean_vec(force_rebuild=False, sample=400, verbose=False):
    """返回 64 维全局均值向量（带文件缓存）。"""
    global _mu_cache
    if _mu_cache is not None and not force_rebuild:
        return _mu_cache

    if not force_rebuild and os.path.exists(MU_CACHE_PATH):
        try:
            with open(MU_CACHE_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
            mu = [float(x) for x in data["mu"]]
            if len(mu) == DIM:
                _mu_cache = mu
                return mu
        except Exception:
            pass

    mu = estimate_mean_vec(sample=sample, verbose=verbose)
    if mu:
        _mu_cache = mu
        try:
            with open(MU_CACHE_PATH, "w", encoding="utf-8") as f:
                json.dump({"mu": mu, "dim": DIM, "sample": sample,
                           "note": "向量去均值校准用；由 vector_calibration.py 生成"},
                          f, ensure_ascii=False, indent=1)
        except Exception as e:
            print(f"[校准] 写缓存失败: {e}")
    return mu or [0.0] * DIM


def estimate_mean_vec(sample=400, verbose=False):
    """现场抽样估计全局逐维均值（不依赖 Qdrant 端聚合）。"""
    try:
        from qdrant_client import QdrantClient
    except Exception:
        return None

    qc = QdrantClient(url=_QDRANT_URL, timeout=300)
    vecs = []
    offset = None
    for _ in range(sample):
        try:
            pts, nxt = qc.scroll(_COLLECTION, limit=1, offset=offset,
                                 with_vectors=True, with_payload=False)
        except Exception:
            break
        if not pts:
            break
        v = pts[0].vector
        if v:
            vecs.append([float(x) for x in v])
        offset = nxt
    if len(vecs) < 20:
        return None
    mu = [statistics.mean(v[d] for v in vecs) for d in range(DIM)]
    if verbose:
        print(f"[校准] 从 {len(vecs)} 条估计均值: 平均={statistics.mean(mu):.2f} "
              f"逐维sd={statistics.pstdev(mu):.2f}")
    return mu


def center(vec, mu=None):
    """去均值。"""
    if vec is None:
        return None
    if mu is None:
        mu = get_mean_vec()
    return [float(vec[d]) - mu[d] for d in range(min(len(vec), len(mu)))]


def cos(a, b):
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = na = nb = 0.0
    for x, y in zip(a, b):
        x, y = float(x), float(y)
        dot += x * y
        na += x * x
        nb += y * y
    if na <= 0 or nb <= 0:
        return 0.0
    return dot / (math.sqrt(na) * math.sqrt(nb))


def calibrated_cos(a, b, mu=None):
    """去均值后的余弦 —— 这才是可读作"相似度"的数。"""
    return cos(center(a, mu), center(b, mu))


# ---------------------------------------------------------------- 映射到前端展示
# 实测：去均值后
#   随机基线      mean -0.000, 范围 [-0.71, 0.73]
#   推荐结果 top20 mean  0.688, 范围 [ 0.199, 0.799]
# 因此把 [-0.05, 0.80] 线性映射到 [0, 100] 作为"相似度%"，比裸余弦诚实得多。
CAL_LO = -0.05
CAL_HI = 0.80


def to_percent(c, lo=CAL_LO, hi=CAL_HI):
    """去均值余弦 -> 0~100 的可读百分比。"""
    if c is None:
        return 0.0
    p = (float(c) - lo) / (hi - lo) * 100.0
    return round(max(0.0, min(100.0, p)), 1)
