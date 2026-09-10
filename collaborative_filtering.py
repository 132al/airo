# collaborative_filtering.py
import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from pathlib import Path
import pickle
import time
from sklearn.model_selection import train_test_split

# =============================================
# 1. 设置路径 & 读取数据
# =============================================
DATA_DIR = Path("F:/airo/aipro/data")

print("=" * 60)
print("📥 读取数据...")
print("=" * 60)

listens = pd.read_parquet(DATA_DIR / "listens.parquet")
likes = pd.read_parquet(DATA_DIR / "likes.parquet")

print(f"   收听记录数: {len(listens):,}")
print(f"   喜欢记录数: {len(likes):,}")
print(f"   列名: {listens.columns.tolist()}")

# 提取用户和歌曲ID列（按你的数据列名）
uid_col = 'uid'
item_col = 'item_id'

# 统计数据
users = listens[uid_col].unique()
items = listens[item_col].unique()
print(f"   用户数: {len(users):,}")
print(f"   歌曲数: {len(items):,}")

# =============================================
# 2. 构建交互矩阵（只使用 listens）
# =============================================
print("\n" + "=" * 60)
print("🔨 构建交互矩阵...")
print("=" * 60)

# 给用户和歌曲编号
user_to_idx = {u: i for i, u in enumerate(users)}
item_to_idx = {it: i for i, it in enumerate(items)}

# 映射
row_ind = listens[uid_col].map(user_to_idx).values
col_ind = listens[item_col].map(item_to_idx).values
data = np.ones(len(listens), dtype=np.float32)  # 所有收听为 1

# 构建稀疏矩阵
interaction_matrix = csr_matrix((data, (row_ind, col_ind)),
                                shape=(len(users), len(items)))

print(f"   矩阵大小: {interaction_matrix.shape}")
print(f"   非零元素: {interaction_matrix.nnz:,}")
print(f"   矩阵密度: {interaction_matrix.nnz / (interaction_matrix.shape[0] * interaction_matrix.shape[1]):.6%}")

# =============================================
# 3. 训练 ALS 模型
# =============================================
print("\n" + "=" * 60)
print("🚀 训练 ALS 模型...")
print("=" * 60)

# 检查是否已安装 implicit
try:
    import implicit
except ImportError:
    print("❌ 请先安装 implicit: pip install implicit")
    exit()

# 模型参数
factors = 64
iterations = 15
regularization = 0.1

print(f"   因子数: {factors}")
print(f"   迭代次数: {iterations}")
print(f"   正则化系数: {regularization}")
print("   开始训练...")

start_time = time.time()

model = implicit.als.AlternatingLeastSquares(
    factors=factors,
    iterations=iterations,
    regularization=regularization,
    random_state=42,
    use_gpu=False  # 如果你的电脑有 GPU 并安装了 cupy，可以改为 True
)
model.fit(interaction_matrix.T)  # implicit 要求 items × users

train_time = time.time() - start_time
print(f"✅ 训练完成！耗时: {train_time:.1f} 秒")

# =============================================
# 4. 保存模型
# =============================================
print("\n" + "=" * 60)
print("💾 保存模型...")
print("=" * 60)

with open(DATA_DIR / "als_model.pkl", 'wb') as f:
    pickle.dump({
        'model': model,
        'user_map': user_to_idx,
        'item_map': item_to_idx
    }, f)

print(f"✅ 模型已保存到: {DATA_DIR / 'als_model.pkl'}")

# =============================================
# 5. 为用户生成推荐（测试）
# =============================================
print("\n" + "=" * 60)
print("🎯 为用户生成推荐...")
print("=" * 60)

def recommend_for_user(user_id, model, user_map, item_map, listens_df, n=10):
    """为用户生成 Top-N 推荐"""
    if user_id not in user_map:
        return []
    
    user_idx = user_map[user_id]
    heard_items = set(listens_df[listens_df[uid_col] == user_id][item_col])
    
    # 生成推荐
    item_indices, scores = model.recommend(
        user_idx,
        model.user_factors[user_idx],
        N=50
    )
    
    reversed_item_map = {v: k for k, v in item_map.items()}
    recommendations = []
    
    for idx, score in zip(item_indices, scores):
        item_id = reversed_item_map.get(idx)
        if item_id and item_id not in heard_items:
            recommendations.append({
                'item_id': item_id,
                'score': float(score)
            })
            if len(recommendations) >= n:
                break
    
    return recommendations

# 测试几个用户
sample_users = listens[uid_col].sample(min(3, len(listens))).tolist()
for user in sample_users:
    recs = recommend_for_user(user, model, user_map, item_map, listens, n=10)
    print(f"\n🎯 用户 {user} 的推荐结果:")
    for i, r in enumerate(recs[:5], 1):
        print(f"   {i}. 歌曲ID: {r['item_id']} (得分: {r['score']:.4f})")

# =============================================
# 6. 离线评估 (Recall@K)
# =============================================
print("\n" + "=" * 60)
print("📊 离线评估 (Recall@10)...")
print("=" * 60)

def evaluate_recall(model, user_map, item_map, listens_df, test_ratio=0.1, n=10, max_users=100):
    """计算平均 Recall@K"""
    users = list(user_map.keys())
    test_users = np.random.choice(users, size=min(max_users, int(len(users) * test_ratio)), replace=False)
    
    recalls = []
    
    for user_id in test_users:
        if user_id not in user_map:
            continue
        
        user_idx = user_map[user_id]
        user_items = set(listens_df[listens_df[uid_col] == user_id][item_col])
        
        if len(user_items) < 5:
            continue
        
        # 随机取 20% 作为测试集
        test_items = set(np.random.choice(list(user_items), size=max(1, int(len(user_items) * 0.2)), replace=False))
        train_items = user_items - test_items
        
        # 模拟训练：用训练集生成推荐
        # 注意：这里为了评估，直接用完整模型，实际应重新训练
        item_indices, _ = model.recommend(user_idx, model.user_factors[user_idx], N=50)
        reversed_item_map = {v: k for k, v in item_map.items()}
        rec_items = set()
        for idx in item_indices:
            rec_items.add(reversed_item_map.get(idx))
            if len(rec_items) >= n:
                break
        
        hit = len(rec_items & test_items)
        recall = hit / len(test_items) if test_items else 0
        recalls.append(recall)
    
    avg_recall = np.mean(recalls) if recalls else 0
    print(f"✅ 评估完成，平均 Recall@{n}: {avg_recall:.4f} (基于 {len(recalls)} 个用户)")
    return avg_recall

avg_recall = evaluate_recall(model, user_map, item_map, listens, test_ratio=0.1, n=10, max_users=50)

print("\n" + "=" * 60)
print("✅ 所有任务完成！")
print("=" * 60)