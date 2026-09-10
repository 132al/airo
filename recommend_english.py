# recommend_english.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns

# 1. 读取数据
print("正在加载数据...")
df = pd.read_parquet('data/spotify_tracks_full.parquet')

# 2. 筛选英文歌曲（排除中文歌名）
english_df = df[~df['track_name'].str.contains('[\u4e00-\u9fa5]', na=False)]
english_df = english_df.reset_index(drop=True)
print(f"英文歌曲数量: {len(english_df)} 首")

# 3. 选择音频特征
feature_cols = [
    'danceability', 'energy', 'valence', 
    'acousticness', 'instrumentalness', 
    'liveness', 'speechiness', 'tempo'
]

print(f"使用的特征: {feature_cols}")

# 4. 数据清洗（去除缺失值）
english_df = english_df.dropna(subset=feature_cols)
print(f"清洗后剩余: {len(english_df)} 首")

# 5. 特征归一化
scaler = MinMaxScaler()
features_scaled = scaler.fit_transform(english_df[feature_cols])
print(f"特征归一化完成，维度: {features_scaled.shape}")

# 6. 计算相似度矩阵
print("正在计算相似度矩阵（可能需要几秒钟）...")
similarity_matrix = cosine_similarity(features_scaled)
print("相似度矩阵计算完成！")

# 7. 推荐函数
def recommend_songs(track_name, n=5):
    """根据歌曲名推荐相似歌曲"""
    # 查找歌曲
    matches = english_df[english_df['track_name'].str.contains(track_name, case=False, na=False)]
    
    if len(matches) == 0:
        return f"❌ 找不到歌曲: {track_name}"
    
    idx = matches.index[0]
    
    # 获取相似度分数
    sim_scores = list(enumerate(similarity_matrix[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:n+1]
    
    recommendations = []
    for i, score in sim_scores:
        recommendations.append({
            '歌曲名': english_df.iloc[i]['track_name'],
            '歌手': english_df.iloc[i]['artists'],
            '专辑': english_df.iloc[i]['album_name'],
            '流派': english_df.iloc[i]['track_genre'],
            '热度': english_df.iloc[i]['popularity'],
            '相似度': f"{score:.2%}"
        })
    
    return pd.DataFrame(recommendations)

def recommend_by_artist(artist_name, n=5):
    """根据歌手推荐该歌手的其他热门歌曲"""
    artist_songs = english_df[english_df['artists'].str.contains(artist_name, case=False, na=False)]
    
    if len(artist_songs) == 0:
        return f"❌ 找不到歌手: {artist_name}"
    
    return artist_songs.nlargest(n, 'popularity')[['track_name', 'album_name', 'track_genre', 'popularity']]

def recommend_by_genre(genre, n=10):
    """根据流派推荐热门歌曲"""
    genre_songs = english_df[english_df['track_genre'].str.contains(genre, case=False, na=False)]
    
    if len(genre_songs) == 0:
        return f"❌ 找不到流派: {genre}"
    
    return genre_songs.nlargest(n, 'popularity')[['track_name', 'artists', 'album_name', 'popularity']]

# 8. 测试推荐
print("\n" + "="*50)
print("🎵 英文歌曲推荐测试")
print("="*50)

# 测试1：按歌曲推荐
print("\n【测试1】推荐与 'Shape of You' 相似的歌曲:")
print(recommend_songs('Shape of You', 5))

# 测试2：按歌手推荐
print("\n【测试2】Ed Sheeran 的热门歌曲:")
print(recommend_by_artist('Ed Sheeran', 5))

# 测试3：按流派推荐
print("\n【测试3】Electronic 流派热门歌曲:")
print(recommend_by_genre('electronic', 5))

# 9. 数据可视化：流派分布
print("\n" + "="*50)
print("📊 数据可视化")
print("="*50)

# 流派分布（前10）
print("\n流派分布（前10）:")
print(english_df['track_genre'].value_counts().head(10))

# 10. 保存推荐模型（供Django使用）
print("\n正在保存推荐模型...")
import pickle
model_data = {
    'df': english_df,
    'similarity_matrix': similarity_matrix,
    'scaler': scaler,
    'feature_cols': feature_cols
}

with open('recommendation_model.pkl', 'wb') as f:
    pickle.dump(model_data, f)

print("✅ 推荐模型已保存为 recommendation_model.pkl")
print("💡 在 Django 中加载: model = pickle.load(open('recommendation_model.pkl', 'rb'))")