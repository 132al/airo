# build_tag_index.py
import json
from qdrant_client import QdrantClient
from qdrant_client.http import models
from sentence_transformers import SentenceTransformer

QDRANT_URL = "http://127.0.0.1:6333"
COLLECTION = "tag_index"
INPUT = "tags_full.json"
MODEL_NAME = "BAAI/bge-small-zh-v1.5"
BATCH_SIZE = 128


def main():
    # 1. 读数据（JSON 数组格式）
    print(f"读取 {INPUT}...")
    with open(INPUT, "r", encoding="utf-8") as f:
        data = json.load(f)
    print(f"共 {len(data)} 个标签")

    tags = [item["tag"] for item in data]
    embed_texts = [item["embed_text"] for item in data]

    # 2. 加载模型
    print(f"加载模型 {MODEL_NAME}...")
    model = SentenceTransformer(MODEL_NAME)

    # 3. 编码 embed_text
    print("编码中...")
    embeddings = model.encode(
        embed_texts,
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        normalize_embeddings=True,
    )
    print(f"编码完成，维度：{embeddings.shape}")

    # 4. 建 collection
    client = QdrantClient(url=QDRANT_URL)

    if client.collection_exists(COLLECTION):
        info = client.get_collection(COLLECTION)
        print(f"删除已有 collection: {COLLECTION}（{info.points_count} 个点）")
        client.delete_collection(COLLECTION)

    print(f"创建 collection: {COLLECTION}")
    client.create_collection(
        collection_name=COLLECTION,
        vectors_config=models.VectorParams(
            size=embeddings.shape[1],
            distance=models.Distance.COSINE,
        ),
    )

    # 5. 写入
    print("写入 Qdrant...")
    points = []
    for i in range(len(data)):
        points.append(
            models.PointStruct(
                id=i,
                vector=embeddings[i].tolist(),
                payload={"tag": tags[i]},
            )
        )
        if len(points) >= BATCH_SIZE:
            client.upsert(collection_name=COLLECTION, points=points)
            points = []
    if points:
        client.upsert(collection_name=COLLECTION, points=points)

    # 6. 验证
    info = client.get_collection(COLLECTION)
    print(f"完成，collection 中共 {info.points_count} 个点")


if __name__ == "__main__":
    main()