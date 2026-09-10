import pyarrow.parquet as pq
import faiss
import numpy as np
import joblib
import time
import gc
from datetime import datetime, timedelta

# 1. 加载 Scaler
scaler = joblib.load("F:/airo/aipro/data/scaler.pkl")
print("✅ Scaler 加载完成")

# 2. 打开 Parquet 文件
parquet_file = pq.ParquetFile("F:/airo/aipro/data/embeat_45m.parquet")
feature_cols = [
    'danceability', 'energy', 'key', 'mode', 'loudness',
    'speechiness', 'acousticness', 'instrumentalness',
    'liveness', 'valence', 'tempo', 'time_signature'
]

total_rows = parquet_file.metadata.num_rows
print(f"📊 总数据量: {total_rows:,} 条")

# 3. 加载或创建索引
index_path = "F:/airo/aipro/data/embeat_45m_ivf.index"
try:
    index = faiss.read_index(index_path)
    print(f"✅ 从已有索引继续，当前 {index.ntotal:,} 条")
except:
    # 如果索引不存在，重新创建
    print("创建新索引...")
    dim = len(feature_cols)
    nlist = 4096
    quantizer = faiss.IndexFlatL2(dim)
    index = faiss.IndexIVFFlat(quantizer, dim, nlist)
    
    # 训练（用 10 万条数据）
    print("读取训练样本...")
    train_features_list = []
    train_batch_size = 50000
    train_samples_needed = 100000
    collected = 0
    for batch in parquet_file.iter_batches(batch_size=train_batch_size, columns=feature_cols):
        df_batch = batch.to_pandas()
        features = df_batch[feature_cols].values.astype('float32')
        train_features_list.append(features)
        collected += len(features)
        if collected >= train_samples_needed:
            break
    train_features = np.vstack(train_features_list)[:train_samples_needed]
    train_features_scaled = scaler.transform(train_features)
    index.train(train_features_scaled)
    index.nprobe = 16
    print("✅ 索引创建完成")

# 4. 分批添加（从当前进度继续）
batch_size = 50000  # 更小批次
total_processed = index.ntotal
start_time = time.time()
log_interval = 5000000

print(f"开始从 {total_processed:,} 条继续构建...")
for batch in parquet_file.iter_batches(batch_size=batch_size, columns=feature_cols):
    df_batch = batch.to_pandas()
    features = df_batch[feature_cols].values.astype('float32')
    features_scaled = scaler.transform(features)
    index.add(features_scaled)
    
    total_processed += len(df_batch)
    
    # 每 500 万条打印进度
    if total_processed % log_interval == 0 or total_processed >= total_rows:
        elapsed = time.time() - start_time
        progress = total_processed / total_rows * 100
        speed = total_processed / elapsed if elapsed > 0 else 0
        remaining_rows = total_rows - total_processed
        eta_seconds = remaining_rows / speed if speed > 0 else 0
        eta_str = str(timedelta(seconds=int(eta_seconds)))
        print(f"📊 已处理: {total_processed:,} / {total_rows:,} ({progress:.1f}%) "
              f"| 耗时: {timedelta(seconds=int(elapsed))} "
              f"| 速度: {speed/1000:.0f}k 条/秒 "
              f"| 预计剩余: {eta_str}")
    
    # 强制回收内存
    df_batch = None
    features = None
    features_scaled = None
    if total_processed % 1000000 == 0:
        gc.collect()

# 5. 保存索引
print("💾 正在保存索引...")
faiss.write_index(index, index_path)
print(f"✅ 索引构建完成！共 {index.ntotal:,} 条")