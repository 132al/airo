# webtest/services/recommend.py
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition

client = QdrantClient(host="localhost", port=6333)
COLLECTION_NAME = "spotify_songs"

def search_similar(track_name: str, top_k: int = 5):
    # 1. 查找歌曲
    scroll_result = client.scroll(
        collection_name=COLLECTION_NAME,
        scroll_filter=Filter(
            must=[FieldCondition(key="track_name", match=track_name)]
        ),
        limit=1,
        with_vectors=True,
        with_payload=True
    )
    
    points = scroll_result[0]
    if not points:
        return None
    
    query_vector = points[0].vector
    seed_name = points[0].payload.get('track_name', track_name)
    
    # 2. 相似检索
    search_results = client.search(
        collection_name=COLLECTION_NAME,
        query_vector=query_vector,
        limit=top_k + 1,
        with_payload=True
    )
    
    output = []
    for hit in search_results:
        if hit.payload.get('track_name') == seed_name:
            continue
        output.append({
            'track_name': hit.payload.get('track_name', '未知'),
            'artist_name': hit.payload.get('artist_name', '未知'),
            'similarity': round(hit.score, 4)
        })
        if len(output) >= top_k:
            break
    
    return output