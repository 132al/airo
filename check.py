import requests
import numpy as np
import json
from collections import Counter, defaultdict

QDRANT_URL = "http://localhost:6333"
COLLECTION_NAME = "spotify_songs_test"
SAMPLE_SIZE = 5000

# ===== 特征配置（与 views.py 保持一致） =====
FEATURE_NAMES = [
    'danceability', 'energy', 'key', 'mode', 'loudness',
    'speechiness', 'acousticness', 'instrumentalness',
    'liveness', 'valence', 'tempo', 'time_signature'
]

# 当前使用的权重
FEATURE_WEIGHTS = [
    1.0,   # danceability
    0.8,   # energy
    0.5,   # key
    0.5,   # mode
    0.8,   # loudness
    0.3,   # speechiness
    0.7,   # acousticness
    0.4,   # instrumentalness
    0.5,   # liveness
    0.05,  # valence
    0.4,   # tempo
    0.5,   # time_signature
]


def get_random_sample(sample_size=5000):
    """从 Qdrant 随机采样数据"""
    # 获取总点数
    try:
        resp = requests.get(f"{QDRANT_URL}/collections/{COLLECTION_NAME}")
        if resp.status_code != 200:
            print(f"❌ 集合不存在: {resp.status_code}")
            return []
        total = resp.json()['result']['points_count']
        print(f"📊 总点数: {total:,}")
    except Exception as e:
        print(f"❌ 连接失败: {e}")
        return []
    
    if total == 0:
        print("❌ 集合为空")
        return []
    
    # 随机采样
    import random
    random.seed(42)
    sample_size = min(sample_size, total)
    offsets = sorted(random.sample(range(total), sample_size))
    
    sampled_points = []
    
    print(f"📊 采样 {sample_size:,} 条...")
    for i, offset in enumerate(offsets):
        try:
            resp = requests.post(
                f"{QDRANT_URL}/collections/{COLLECTION_NAME}/points/scroll",
                json={"limit": 1, "offset": offset, "with_payload": True, "with_vector": True},
                timeout=10
            )
            if resp.status_code == 200:
                points = resp.json().get('result', {}).get('points', [])
                if points:
                    p = points[0]
                    sampled_points.append({
                        'id': p.get('id'),
                        'vector': p.get('vector'),
                        'payload': p.get('payload', {})
                    })
            
            if (i + 1) % 1000 == 0:
                print(f"  进度: {i+1}/{len(offsets)}")
                
        except Exception:
            pass
    
    print(f"✅ 采样完成: {len(sampled_points)} 条")
    return sampled_points


def analyze_vectors(points):
    """分析采样数据的向量质量"""
    n = len(points)
    if n == 0:
        print("❌ 无数据")
        return
    
    vectors = [p['vector'] for p in points if p['vector']]
    if not vectors:
        print("❌ 所有向量均为空")
        return
    
    n_vec = len(vectors)
    print(f"\n{'='*70}")
    print(f"📊 向量质量分析 (采样 {n_vec:,} 条)")
    print(f"{'='*70}")
    
    # 1. 向量维度
    vec_len = len(vectors[0])
    print(f"\n📐 向量维度: {vec_len}")
    
    # 2. 范数检查
    norms = [np.linalg.norm(v) for v in vectors]
    print(f"\n📊 范数统计:")
    print(f"  平均值: {np.mean(norms):.4f}")
    print(f"  中位数: {np.median(norms):.4f}")
    print(f"  最小值: {np.min(norms):.4f}")
    print(f"  最大值: {np.max(norms):.4f}")
    print(f"  标准差: {np.std(norms):.4f}")
    
    # 范数异常（应该在 1 左右）
    norm_anomaly = sum(1 for n_ in norms if n_ < 0.5 or n_ > 1.5)
    if norm_anomaly > 0:
        print(f"  ⚠️ 范数异常 (<0.5 或 >1.5): {norm_anomaly} 条 ({norm_anomaly/n*100:.1f}%)")
    else:
        print(f"  ✅ 所有范数正常 (0.5-1.5)")
    
    # 3. 各维度统计
    print(f"\n📊 各维度统计:")
    print(f"  {'维度':<18} {'平均值':>10} {'标准差':>10} {'最小值':>10} {'最大值':>10} {'主导占比':>12}")
    print(f"  {'-'*18} {'-'*10} {'-'*10} {'-'*10} {'-'*10} {'-'*12}")
    
    for i, name in enumerate(FEATURE_NAMES):
        vals = [v[i] for v in vectors]
        mean_val = np.mean(vals)
        std_val = np.std(vals)
        min_val = np.min(vals)
        max_val = np.max(vals)
        
        # 计算该维度在向量中的平均占比
        dominance = np.mean([abs(v[i]) / (np.sum(np.abs(v)) + 1e-10) for v in vectors]) * 100
        
        # 标记异常维度（标准差太小或均值异常）
        if std_val < 0.01:
            status = " ⚠️ 方差过小"
        elif dominance > 30:
            status = " ⚠️ 主导"
        else:
            status = ""
        
        print(f"  {name:<18} {mean_val:>10.4f} {std_val:>10.4f} {min_val:>10.4f} {max_val:>10.4f} {dominance:>11.1f}%{status}")
    
    # 4. 维度主导分析
    print(f"\n📊 主导维度分析（每个向量中占比最大的维度）:")
    dominant_dims = []
    for v in vectors:
        abs_v = np.abs(v)
        total = np.sum(abs_v)
        if total > 0:
            dominant_idx = np.argmax(abs_v)
            dominant_dims.append((FEATURE_NAMES[dominant_idx], abs_v[dominant_idx]/total*100))
    
    if dominant_dims:
        dim_counter = Counter([d[0] for d in dominant_dims])
        print(f"  主导维度分布 (Top 5):")
        for dim, count in dim_counter.most_common(5):
            pct = count / len(dominant_dims) * 100
            avg_dominance = np.mean([d[1] for d in dominant_dims if d[0] == dim])
            print(f"    {dim}: {count} 条 ({pct:.1f}%) 平均占比 {avg_dominance:.1f}%")
        
        # 检查最严重的问题维度
        most_common = dim_counter.most_common(1)[0]
        if most_common[1] / len(dominant_dims) > 0.5:
            print(f"\n  ⚠️ 问题: {most_common[0]} 占绝对主导 ({most_common[1]/len(dominant_dims)*100:.1f}%)")
            print(f"     建议: 降低 {most_common[0]} 的权重")


def analyze_payloads(points):
    """分析采样数据的 payload 质量"""
    n = len(points)
    if n == 0:
        return
    
    print(f"\n{'='*70}")
    print(f"📦 Payload 质量分析 (采样 {n:,} 条)")
    print(f"{'='*70}")
    
    # 1. 字段覆盖率
    field_count = defaultdict(int)
    for p in points:
        for key in p['payload'].keys():
            field_count[key] += 1
    
    print(f"\n📊 字段覆盖率:")
    for key, count in sorted(field_count.items()):
        pct = count / n * 100
        print(f"  {key}: {count:,} 条 ({pct:.1f}%)")
    
    # 2. 热度分布
    popularities = [p['payload'].get('popularity', 0) for p in points]
    print(f"\n🔥 热度分布 (popularity):")
    print(f"  平均值: {np.mean(popularities):.4f}")
    print(f"  中位数: {np.median(popularities):.4f}")
    print(f"  最小值: {np.min(popularities):.4f}")
    print(f"  最大值: {np.max(popularities):.4f}")
    
    # 热度分段
    bins = [(0, 0.02), (0.02, 0.05), (0.05, 0.10), (0.10, 0.20), (0.20, 0.30), (0.30, 0.40), (0.40, 0.50), (0.50, 0.60), (0.60, 1.0)]
    print(f"\n  热度分段:")
    for lo, hi in bins:
        cnt = sum(1 for p in popularities if lo <= p < hi)
        pct = cnt / n * 100
        bar = '█' * int(pct / 2)
        print(f"    [{lo:.2f}, {hi:.2f}): {cnt:>4} 条 ({pct:>5.1f}%) {bar}")
    
    # 3. 年份分布
    years = [p['payload'].get('release_year', 0) for p in points if p['payload'].get('release_year', 0) > 0]
    if years:
        print(f"\n📅 年份分布:")
        print(f"  范围: {min(years)} - {max(years)}")
        print(f"  平均值: {np.mean(years):.1f}")
        
        # 年代分布
        decades = defaultdict(int)
        for y in years:
            if y >= 1960:
                decades[y // 10 * 10] += 1
        print(f"  年代分布:")
        for d in sorted(decades.keys()):
            pct = decades[d] / len(years) * 100
            print(f"    {d}s: {decades[d]} 条 ({pct:.1f}%)")
    
    # 4. 流派多样性
    genres = [p['payload'].get('artist_genres', '') for p in points if p['payload'].get('artist_genres', '')]
    if genres:
        print(f"\n🎵 流派多样性:")
        all_genres = []
        for g in genres:
            for genre in g.split(','):
                genre = genre.strip()
                if genre:
                    all_genres.append(genre)
        
        genre_counter = Counter(all_genres)
        print(f"  总流派数: {len(genre_counter)}")
        print(f"  Top 10 流派:")
        for genre, count in genre_counter.most_common(10):
            pct = count / len(all_genres) * 100
            print(f"    {genre[:25]}: {count} 次 ({pct:.1f}%)")


def main():
    print("="*70)
    print("📊 Qdrant 数据质量检查")
    print("="*70)
    
    # 1. 采样
    points = get_random_sample(SAMPLE_SIZE)
    if not points:
        return
    
    # 2. 分析向量
    analyze_vectors(points)
    
    # 3. 分析 payload
    analyze_payloads(points)
    
    # 4. 总结
    print("\n" + "="*70)
    print("💡 总结与建议")
    print("="*70)
    
    # 检查范数
    vectors = [p['vector'] for p in points if p['vector']]
    if vectors:
        norms = [np.linalg.norm(v) for v in vectors]
        avg_norm = np.mean(norms)
        if 0.9 < avg_norm < 1.1:
            print("✅ L2 归一化正常 (平均范数 ≈ 1.0)")
        else:
            print(f"⚠️ L2 归一化异常 (平均范数 = {avg_norm:.4f})")
    
    # 检查主导维度
    all_v = np.array(vectors)
    if len(all_v) > 0:
        # 计算每个维度的平均方差
        stds = np.std(all_v, axis=0)
        for i, name in enumerate(FEATURE_NAMES):
            if stds[i] < 0.01:
                print(f"⚠️ {name} 方差极小 ({stds[i]:.4f})，该维度可能没有区分度")


if __name__ == "__main__":
    main()