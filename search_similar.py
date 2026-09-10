# check_id_92.py
import requests

QDRANT_URL = "http://localhost:6333"
COLLECTION_NAME = "spotify_songs_test"

# 直接用 Qdrant 查询 ID 92
resp = requests.post(
    f"{QDRANT_URL}/collections/{COLLECTION_NAME}/points/scroll",
    json={
        "filter": {"must": [{"key": "id", "match": {"value": 92}}]},
        "limit": 1,
        "with_payload": True,
        "with_vector": True,
    }
)

print(f"状态码: {resp.status_code}")
print(f"返回内容: {resp.json()}")

if resp.status_code == 200:
    points = resp.json().get("result", {}).get("points", [])
    if points:
        p = points[0]
        print(f"\n✅ ID 92 存在")
        print(f"  歌曲: {p['payload'].get('track_name')}")
        print(f"  歌手: {p['payload'].get('artist_name')}")
        print(f"  流派: {p['payload'].get('artist_genres')}")
    else:
        print("\n❌ ID 92 不存在")
else:
    print("❌ 查询失败")