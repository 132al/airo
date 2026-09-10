from sklearn.preprocessing import StandardScaler
import joblib
import numpy as np

# 1. 把刚才读到的 12 个特征列名写清楚
feature_cols = [
    'danceability', 'energy', 'key', 'mode', 'loudness',
    'speechiness', 'acousticness', 'instrumentalness',
    'liveness', 'valence', 'tempo', 'time_signature'
]

# 2. 用你刚才已经打开的文件，读取前 100000 行

import pyarrow.parquet as pq
parquet_file = pq.ParquetFile("F:/airo/aipro/data/embeat_45m.parquet")

train_df = parquet_file.read(columns=feature_cols, use_threads=False).to_pandas().head(100000)

# 3. 把所有特征都转成 float32（整数特征也统一转一下）
train_features = train_df[feature_cols].values.astype('float32')

# 4. 创建并拟合 Scaler
scaler = StandardScaler()
scaler.fit(train_features)
print("✅ Scaler 拟合完成！")

# 5. 保存 Scaler
joblib.dump(scaler, "F:/airo/aipro/data/scaler.pkl")
print("✅ Scaler 已保存到 F:/airo/aipro/data/scaler.pkl")