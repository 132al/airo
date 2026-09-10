# collaborative_filtering_load.py
import pandas as pd
import numpy as np
from pathlib import Path
import pickle
from scipy.sparse import csr_matrix

# =============================================
# 设置路径
# =============================================
DATA_DIR = Path("F:/airo/aipro/data")

print("=" * 60)
print("📥 加载已训练的 ALS 模型...")
print("=" * 60)

# =============================================
# 加载模型
# =============================================
with open(DATA_DIR / "als_model.pkl", 'rb') as f:
    saved = pickle.load(f)
    model = saved['model']
    user_to_idx = saved['user_map']
    item_to_idx = saved['item_map']

print(f"✅ 模型加载成功！")
print(f"   用户数: {len(user_to_idx):,}")
print(f"   歌曲数: {len(item_to_idx):,}")
print(f"   因子维度: {model.factors}")

# =============================================
# 加载原始数据
# =============================================
print("\n📥 加载收听数据...")
listens = pd.read_parquet(DATA_DIR / "listens.parquet")
print(f"   收听记录数: {len(listens):,}")

# 构建用户-物品交互矩阵（用于推荐时过滤已听歌曲）
# 1. 用 map 获取索引，并验证没有空值
row_ind = listens['uid'].map(user_to_idx).fillna(-1).astype(int).values
col_ind = listens['item_id'].map(item_to_idx).fillna(-1).astype(int).values

# 2. 检查是否有 -1
if (row_ind == -1).any() or (col_ind == -1).any():
    print("⚠️ 发现未映射的索引，请检查数据")
    # 过滤掉 -1
    valid_mask = (row_ind != -1) & (col_ind != -1)
    row_ind = row_ind[valid_mask]
    col_ind = col_ind[valid_mask]

# 3. 构建矩阵
data = np.ones(len(row_ind), dtype=np.float32)
user_item_matrix = csr_matrix((data, (row_ind, col_ind)), 
                              shape=(len(user_to_idx), len(item_to_idx)))

print(f"✅ 矩阵重建完成: {user_item_matrix.shape}")
print(f"  非零元素: {user_item_matrix.nnz:,}")

# ============================================================
# 为用户生成推荐（修复版）
# ============================================================
print("\n" + "=" * 60)
print("🎯 为用户生成推荐...")
print("=" * 60)

def recommend_for_user(user_id, model, user_to_idx, item_to_idx, user_item_matrix, n=10):
    """为用户生成 Top-N 推荐"""
    if user_id not in user_to_idx:
        return []
    
    user_idx = user_to_idx[user_id]
    
    # 获取该用户的交互矩阵行（CSR 格式）
    user_row = user_item_matrix[[user_idx], :]
    
    # 生成推荐（user_row 作为已交互物品传入）
    item_indices, scores = model.recommend(
        user_idx,
        user_row,
        N=n
    )
    
    reversed_item_map = {v: k for k, v in item_to_idx.items()}
    recommendations = []
    
    for idx, score in zip(item_indices, scores):
        item_id = reversed_item_map.get(idx)
        if item_id:
            recommendations.append({
                'item_id': item_id,
                'score': float(score)
            })
    
    return recommendations

# 测试几个用户
sample_users = listens['uid'].drop_duplicates().sample(min(5, len(listens['uid'].unique()))).tolist()
for user in sample_users:
    recs = recommend_for_user(user, model, user_to_idx, item_to_idx, user_item_matrix, n=10)
    print(f"\n🎯 用户 {user} 的推荐结果:")
    if not recs:
        print("   无推荐结果")
    else:
        for i, r in enumerate(recs[:5], 1):
            print(f"   {i}. 歌曲ID: {r['item_id']} (得分: {r['score']:.4f})")

# ============================================================
# 离线评估 (Recall@K)
# ============================================================
print("\n" + "=" * 60)
print("📊 离线评估 (Recall@10)...")
print("=" * 60)

def evaluate_recall(model, user_to_idx, item_to_idx, user_item_matrix, test_ratio=0.1, n=10, max_users=50):
    """计算平均 Recall@K"""
    users = list(user_to_idx.keys())
    test_users = np.random.choice(users, size=min(max_users, int(len(users) * test_ratio)), replace=False)
    
    recalls = []
    hit_counts = []
    
    for user_id in test_users:
        if user_id not in user_to_idx:
            continue
        
        user_idx = user_to_idx[user_id]
        user_items = set(listens[listens['uid'] == user_id]['item_id'])
        
        if len(user_items) < 5:
            continue
        
        # 随机取 20% 作为测试集
        test_items = set(np.random.choice(list(user_items), size=max(1, int(len(user_items) * 0.2)), replace=False))
        
        # 生成推荐
        user_row = user_item_matrix[[user_idx], :]
        item_indices, _ = model.recommend(user_idx, user_row, N=n)
        reversed_item_map = {v: k for k, v in item_to_idx.items()}
        rec_items = set()
        for idx in item_indices:
            rec_items.add(reversed_item_map.get(idx))
        
        hit = len(rec_items & test_items)
        recall = hit / len(test_items) if test_items else 0
        recalls.append(recall)
        hit_counts.append(hit)
    
    avg_recall = np.mean(recalls) if recalls else 0
    avg_hits = np.mean(hit_counts) if hit_counts else 0
    print(f"✅ 评估完成，平均 Recall@{n}: {avg_recall:.4f}")
    print(f"   平均命中数: {avg_hits:.2f} / {n}")
    print(f"   基于 {len(recalls)} 个用户")
    return avg_recall

avg_recall = evaluate_recall(model, user_to_idx, item_to_idx, user_item_matrix, 
                             test_ratio=0.1, n=10, max_users=50)

print("\n" + "=" * 60)
print("✅ 所有任务完成！")
print("=" * 60)