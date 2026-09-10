# show_top_song.py
import json
import requests

QDRANT_URL = "http://localhost:6333"
COLLECTION_NAME = "spotify_tracks"

# 想看的歌名，随便改
TRACK_NAME = "Shape of You"
OUTPUT_FILE = "one_song.json"


def show_top(track_name):
    # 多取几条，按 popularity 排序，取最火的
    resp = requests.post(
        f"{QDRANT_URL}/collections/{COLLECTION_NAME}/points/scroll",
        json={
            "filter": {"must": [{"key": "track_name", "match": {"text": track_name}}]},
            "limit": 50,
            "with_payload": True,
            "with_vector": True,
        },
        timeout=10,
    )
    points = resp.json()["result"]["points"]
    if not points:
        print(f"❌ 没找到: {track_name}")
        return

    # 按 popularity 降序
    points_sorted = sorted(
        points,
        key=lambda x: float(x.get("payload", {}).get("popularity", 0) or 0),
        reverse=True,
    )
    p = points_sorted[0]

    # 保存
    data = {
        "id": p.get("id"),
        "payload": p.get("payload", {}),
        "vector": p.get("vector"),
    }

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"✅ 已保存到 {OUTPUT_FILE}")
    print(f"   歌曲: {data['payload'].get('track_name')} - {data['payload'].get('artist_name')}")
    print(f"   Popularity: {data['payload'].get('popularity')}")
    print(f"   Payload 字段数: {len(data['payload'])}")
    print(f"   向量维度: {len(data['vector']) if data['vector'] else 0}")


if __name__ == "__main__":
    show_top(TRACK_NAME)