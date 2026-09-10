import polars as pl
import numpy as np
import gc
import time
import os
from sklearn.preprocessing import StandardScaler, normalize
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, VectorParams, Distance
import joblib
import pyarrow.parquet as pq

# ==================== 配置 ====================
client = QdrantClient(host="localhost", port=6333)
COLLECTION_NAME = "spotify_songs_test"  # 用测试集合，不干扰主数据

# 向量特征（12维）
FEATURE_COLS = [
    'danceability', 'energy', 'loudness', 'speechiness',
    'acousticness', 'instrumentalness', 'liveness', 'valence',
    'tempo', 'key', 'mode', 'time_signature'
]
VECTOR_SIZE = len(FEATURE_COLS)

# Payload 字段
PAYLOAD_COLS = [
    'track_name', 'artist_name', 'popularity', 'artist_genres',
    'release_year', 'explicit', 'duration_ms'
]
ALL_COLS = FEATURE_COLS + PAYLOAD_COLS

# 文件路径
DATA_PATH = "F:/airo/aipro/data/embeat_45m.parquet"
SCALER_PATH = "F:/airo/aipro/data/scaler.pkl"

# ===== 测试配置 =====
TARGET_IMPORT = 1000000  # 只导入 100 万条
BATCH_SIZE = 20000       # 每批读取 2 万条
SUB_BUILD_SIZE = 500     # 每批构建 500 条


def drop_and_create_collection():
    """删除并创建测试集合"""
    try:
        if client.collection_exists(COLLECTION_NAME):
            print(f"🗑️ 删除旧测试集合: {COLLECTION_NAME}")
            client.delete_collection(COLLECTION_NAME)
    except Exception as e:
        print(f"⚠️ 删除集合时出错: {e}")
    
    print(f"📊 创建测试集合: {COLLECTION_NAME}")
    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
    )
    print(f"✅ 集合创建成功 (维度: {VECTOR_SIZE})")


def get_scaler():
    """加载 Scaler"""
    if os.path.exists(SCALER_PATH):
        scaler = joblib.load(SCALER_PATH)
        print(f"✅ 加载 Scaler")
        return scaler
    
    # 如果不存在，训练一个
    print("📊 训练 StandardScaler...")
    df_sample = pl.scan_parquet(DATA_PATH).select(FEATURE_COLS).limit(50000).collect()
    sample_features = df_sample.to_numpy().astype('float32')
    sample_features = np.nan_to_num(sample_features, nan=0.0)
    
    scaler = StandardScaler()
    scaler.fit(sample_features)
    joblib.dump(scaler, SCALER_PATH)
    print(f"✅ Scaler 训练完成")
    return scaler


def clean_value(val, col_name):
    """清理字段值"""
    if val is None:
        return 0.0 if col_name in FEATURE_COLS else ''
    if isinstance(val, str):
        return val.strip()[:200] if val.strip() else ''
    if isinstance(val, float):
        return float(val) if not np.isnan(val) else 0.0
    if isinstance(val, int):
        return int(val)
    if isinstance(val, bool):
        return bool(val)
    return str(val)[:200]


def process_batch(batch_df, scaler, start_idx):
    """处理一个批次"""
    if batch_df.height == 0:
        return 0
    
    # 1. 特征处理
    features = batch_df.select(FEATURE_COLS).to_numpy().astype('float32')
    features = np.nan_to_num(features, nan=0.0)
    
    # 2. 标准化
    features_scaled = scaler.transform(features)
    
    # 3. L2 归一化
    features_normalized = normalize(features_scaled, norm='l2')
    
    # 4. 构建 points
    rows = batch_df.rows(named=True)
    points = []
    
    for idx, (row, vec) in enumerate(zip(rows, features_normalized)):
        payload = {}
        for col in PAYLOAD_COLS:
            payload[col] = clean_value(row.get(col), col)
        
        points.append(PointStruct(
            id=start_idx + idx,
            vector=vec.tolist(),
            payload=payload
        ))
    
    # 5. 分批导入
    total = len(points)
    for i in range(0, total, SUB_BUILD_SIZE):
        sub_points = points[i:i+SUB_BUILD_SIZE]
        client.upsert(collection_name=COLLECTION_NAME, points=sub_points)
    
    return total


def import_data():
    print("=" * 70)
    print("🚀 测试导入 (100 万条)")
    print("=" * 70)
    
    # 1. 创建测试集合
    drop_and_create_collection()
    
    # 2. 加载 Scaler
    scaler = get_scaler()
    
    # 3. 准备数据
    print(f"\n📊 开始导入...")
    print(f"📊 目标: {TARGET_IMPORT:,} 条")
    print(f"📊 批次大小: {BATCH_SIZE:,} 条")
    print("=" * 70)
    
    start_time = time.time()
    processed = 0
    
    # 4. 流式读取 Parquet
    parquet_file = pq.ParquetFile(DATA_PATH)
    
    for batch in parquet_file.iter_batches(batch_size=BATCH_SIZE, columns=ALL_COLS):
        if processed >= TARGET_IMPORT:
            break
        
        # 转为 Polars DataFrame
        batch_df = pl.from_arrow(batch)
        
        # 处理
        imported = process_batch(batch_df, scaler, processed)
        processed += imported
        
        # 进度
        elapsed = time.time() - start_time
        speed = processed / elapsed if elapsed > 0 else 0
        progress = processed / TARGET_IMPORT * 100
        
        print(f"📊 进度: {progress:.1f}% | 已导入: {processed:,} | "
              f"速度: {speed/1000:.1f}k/s")
        
        # 清理
        del batch_df
        gc.collect()
    
    # 5. 完成
    elapsed = time.time() - start_time
    print("=" * 70)
    print(f"✅ 导入完成！")
    print(f"📊 共导入: {processed:,} 条")
    print(f"⏱️ 总耗时: {elapsed/60:.1f} 分钟")
    print(f"📈 平均速度: {processed/elapsed/1000:.1f} k/s")
    print("=" * 70)


if __name__ == "__main__":
    import_data()