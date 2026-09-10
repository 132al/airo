import requests
import numpy as np
from collections import Counter, defaultdict
import json

QDRANT_URL = "http://localhost:6333"
COLLECTION_NAME = "spotify_songs_test"  # 测试集合
SAMPLE_SIZE = 5000

# 特征名称
FEATURE_NAMES = [
    'danceability', 'energy', 'loudness', 'speechiness',
    'acousticness', 'instrumentalness', 'liveness', 'valence',
    'tempo', 'key', 'mode', 'time_signature'
]


def get_collection_info():
    """获取集合基本信息"""
    try:
        resp = requests.get(f"{QDRANT_URL}/collections/{COLLECTION_NAME}")
        if resp.status_code != 200:
            print(f"❌ 集合不存在: {resp.status_code}")
            return None
        info = resp.json()['result']
        return info
    except Exception as e:
        print(f"❌ 连接失败: {e}")
        return None


def random_sample(sample_size=5000):
    """随机采样数据"""
    info = get_collection_info()
    if not info:
        return []
    
    total = info['points_count']
    print(f"📊 总点数: {total:,}")
    
    if total == 0:
        print("❌ 集合为空")
        return []
    
    # 随机采样
    import random
    random.seed(42)
    sample_size = min(sample_size, total)
    offsets = sorted(random.sample(range(total), sample_size))
    
    sampled = []
    print(f"📊 采样 {sample_size:,} 条...")
    
    for i, offset in enumerate(offsets):
        try:
            resp = requests.post(
                f"{QDRANT_URL}/collections/{COLLECTION_NAME}/points/scroll",
                json={
                    "limit": 1,
                    "offset": offset,
                    "with_payload": True,
                    "with_vector": True
                },
                timeout=10
            )
            if resp.status_code == 200:
                points = resp.json().get('result', {}).get('points', [])
                if points:
                    p = points[0]
                    sampled.append({
                        'id': p.get('id'),
                        'vector': p.get('vector'),
                        'payload': p.get('payload', {})
                    })
            
            if (i + 1) % 1000 == 0:
                print(f"  进度: {i+1}/{len(offsets)}")
        except Exception:
            pass
    
    print(f"✅ 采样完成: {len(sampled)} 条")
    return sampled


def analyze_vectors(points):
    """分析向量质量"""
    n = len(points)
    if n == 0:
        return
    
    vectors = [p['vector'] for p in points if p['vector'] and len(p['vector']) > 0]
    if not vectors:
        print("❌ 无有效向量")
        return
    
    n_vec = len(vectors)
    print("\n" + "=" * 70)
    print(f"📐 向量质量分析 (采样 {n_vec:,} 条)")
    print("=" * 70)
    
    # 1. 维度
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
    
    # 范数异常
    norm_anomaly = sum(1 for n_ in norms if n_ < 0.5 or n_ > 1.5)
    if norm_anomaly == 0:
        print("  ✅ 所有范数正常 (0.5-1.5)")
    else:
        print(f"  ⚠️ 范数异常: {norm_anomaly} 条 ({norm_anomaly/n*100:.1f}%)")
    
    # 3. 各维度统计
    print(f"\n📊 各维度统计:")
    print(f"  {'维度':<18} {'平均值':>10} {'标准差':>10} {'最小值':>10} {'最大值':>10} {'主导占比':>12}")
    print(f"  {'-'*18} {'-'*10} {'-'*10} {'-'*10} {'-'*10} {'-'*12}")
    
    all_vectors = np.array(vectors)
    
    for i, name in enumerate(FEATURE_NAMES):
        vals = all_vectors[:, i]
        mean_val = np.mean(vals)
        std_val = np.std(vals)
        min_val = np.min(vals)
        max_val = np.max(vals)
        
        # 该维度在向量中的平均占比
        dominance = np.mean([abs(v[i]) / (np.sum(np.abs(v)) + 1e-10) for v in vectors]) * 100
        
        # 判断
        if std_val < 0.01:
            status = " ⚠️ 方差过小"
        elif dominance > 30:
            status = " ⚠️ 主导"
        else:
            status = " ✅"
        
        print(f"  {name:<18} {mean_val:>10.4f} {std_val:>10.4f} {min_val:>10.4f} {max_val:>10.4f} {dominance:>11.1f}%{status}")
    
    # 4. 主导维度
    print(f"\n📊 主导维度 (每个向量中占比最大的):")
    dominant_dims = []
    for v in vectors:
        abs_v = np.abs(v)
        total = np.sum(abs_v)
        if total > 0:
            dominant_idx = np.argmax(abs_v)
            dominant_dims.append(FEATURE_NAMES[dominant_idx])
    
    dim_counter = Counter(dominant_dims)
    for dim, count in dim_counter.most_common(5):
        pct = count / len(dominant_dims) * 100
        print(f"  {dim}: {count} 条 ({pct:.1f}%)")
    
    # 5. 判断是否有问题
    most_common = dim_counter.most_common(1)
    if most_common and most_common[0][1] / len(dominant_dims) > 0.5:
        print(f"\n  ⚠️ 问题: {most_common[0][0]} 占绝对主导 ({most_common[0][1]/len(dominant_dims)*100:.1f}%)")
    else:
        print("\n  ✅ 各维度贡献均衡")


def analyze_payloads(points):
    """分析 payload 质量"""
    n = len(points)
    if n == 0:
        return
    
    print("\n" + "=" * 70)
    print(f"📦 Payload 分析 (采样 {n:,} 条)")
    print("=" * 70)
    
    # 1. 字段覆盖率
    field_count = defaultdict(int)
    for p in points:
        for key in p['payload'].keys():
            field_count[key] += 1
    
    print(f"\n📊 字段覆盖率:")
    for key, count in sorted(field_count.items()):
        print(f"  {key}: {count:,} 条 ({count/n*100:.1f}%)")
    
    # 2. 热度分布
    popularities = [p['payload'].get('popularity', 0) for p in points]
    print(f"\n🔥 热度分布:")
    print(f"  平均值: {np.mean(popularities):.4f}")
    print(f"  中位数: {np.median(popularities):.4f}")
    print(f"  最小值: {np.min(popularities):.4f}")
    print(f"  最大值: {np.max(popularities):.4f}")
    
    # 热度分段
    bins = [
        (0, 0.02), (0.02, 0.05), (0.05, 0.10), (0.10, 0.20),
        (0.20, 0.30), (0.30, 0.40), (0.40, 0.50), (0.50, 0.60), (0.60, 1.0)
    ]
    print(f"\n  热度分段:")
    for lo, hi in bins:
        cnt = sum(1 for p in popularities if lo <= p < hi)
        pct = cnt / n * 100
        bar = '█' * int(pct / 2) if pct > 0 else ''
        print(f"    [{lo:.2f}, {hi:.2f}): {cnt:>4} 条 ({pct:>5.1f}%) {bar}")
    
    # 3. 年份
    years = [p['payload'].get('release_year', 0) for p in points if p['payload'].get('release_year', 0) > 0]
    if years:
        print(f"\n📅 年份分布:")
        print(f"  范围: {min(years)} - {max(years)}")
        print(f"  平均值: {np.mean(years):.1f}")
        
        decades = defaultdict(int)
        for y in years:
            if y >= 1960:
                decades[y // 10 * 10] += 1
        print(f"  年代分布:")
        for d in sorted(decades.keys()):
            pct = decades[d] / len(years) * 100
            print(f"    {d}s: {decades[d]} 条 ({pct:.1f}%)")
    
    # 4. 流派
    genres = [p['payload'].get('artist_genres', '') for p in points if p['payload'].get('artist_genres', '')]
    if genres:
        print(f"\n🎵 流派 Top 10:")
        all_genres = []
        for g in genres:
            for genre in g.split(','):
                genre = genre.strip()
                if genre:
                    all_genres.append(genre)
        
        genre_counter = Counter(all_genres)
        print(f"  总流派数: {len(genre_counter)}")
        for genre, count in genre_counter.most_common(10):
            print(f"    {genre[:25]}: {count} 次")


def check_sample_data(points):
    """检查几条样本数据"""
    if not points:
        return
    
    print("\n" + "=" * 70)
    print("📋 样本数据 (前 3 条)")
    print("=" * 70)
    
    for i, p in enumerate(points[:3]):
        print(f"\n--- 样本 {i+1} ---")
        print(f"  ID: {p['id']}")
        
        vector = p['vector']
        if vector:
            print(f"  向量范数: {np.linalg.norm(vector):.4f}")
            print(f"  向量前5维: {vector[:5]}")
        
        payload = p['payload']
        print(f"  track_name: {payload.get('track_name', 'N/A')}")
        print(f"  artist_name: {payload.get('artist_name', 'N/A')}")
        print(f"  popularity: {payload.get('popularity', 0)}")
        print(f"  artist_genres: {payload.get('artist_genres', '')[:50]}")
        print(f"  release_year: {payload.get('release_year', 0)}")


def main():
    print("=" * 70)
    print(f"📊 解析测试集合: {COLLECTION_NAME}")
    print("=" * 70)
    
    # 1. 检查集合是否存在
    info = get_collection_info()
    if not info:
        print(f"❌ 集合 '{COLLECTION_NAME}' 不存在或为空")
        print("   请先运行 import_test_1m.py 导入数据")
        return
    
    print(f"📊 集合状态:")
    print(f"  总点数: {info['points_count']:,}")
    print(f"  向量维度: {info['config']['params']['vectors']['size']}")
    print(f"  距离: {info['config']['params']['vectors']['distance']}")
    
    # 2. 采样
    points = random_sample(SAMPLE_SIZE)
    if not points:
        return
    
    # 3. 分析
    analyze_vectors(points)
    analyze_payloads(points)
    check_sample_data(points)
    
    # 4. 总结
    print("\n" + "=" * 70)
    print("💡 总结")
    print("=" * 70)
    
    vectors = [p['vector'] for p in points if p['vector']]
    if vectors:
        norms = [np.linalg.norm(v) for v in vectors]
        avg_norm = np.mean(norms)
        if 0.9 < avg_norm < 1.1:
            print("✅ L2 归一化正常")
        else:
            print(f"⚠️ L2 归一化异常: {avg_norm:.4f}")
        
        # 检查各维度标准差
        all_v = np.array(vectors)
        stds = np.std(all_v, axis=0)
        bad_dims = [FEATURE_NAMES[i] for i, s in enumerate(stds) if s < 0.1 or s > 2.0]
        if bad_dims:
            print(f"⚠️ 标准差异常维度: {', '.join(bad_dims)}")
        else:
            print("✅ 各维度标准差正常")
    
    print("=" * 70)


if __name__ == "__main__":
    main()