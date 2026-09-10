import requests
import json
from collections import Counter

QDRANT_URL = "http://localhost:6333"
COLLECTION_NAME = "spotify_songs_test"

def analyze_genres():
    print("=" * 70)
    print("🔍 分析 artist_genres 字段的所有值")
    print("=" * 70)
    
    # 获取所有数据（分批滚动）
    all_genres = []
    offset = 0
    limit = 1000
    total_processed = 0
    
    print("\n📥 正在扫描数据...")
    
    while True:
        scroll_payload = {
            "limit": limit,
            "offset": offset,
            "with_payload": True,
            "with_vector": False
        }
        
        try:
            resp = requests.post(
                f"{QDRANT_URL}/collections/{COLLECTION_NAME}/points/scroll",
                json=scroll_payload,
                timeout=60
            )
            resp.raise_for_status()
            data = resp.json()
            points = data.get('result', {}).get('points', [])
            
            if not points:
                break
                
            for point in points:
                payload = point.get('payload', {})
                genre = payload.get('artist_genres', '')
                if genre:
                    all_genres.append(genre)
                    
            total_processed += len(points)
            offset += limit
            print(f"  已处理: {total_processed} 条...")
            
            if len(points) < limit:
                break
                
        except Exception as e:
            print(f"  ❌ 错误: {e}")
            break
    
    print(f"\n✅ 共扫描 {total_processed} 条记录")
    print(f"✅ 包含流派信息的记录: {len(all_genres)} 条")
    
    # ===== 统计分析 =====
    print("\n" + "=" * 70)
    print("📊 1. 流派字符串统计")
    print("=" * 70)
    
    genre_counter = Counter(all_genres)
    
    print(f"\n  不同流派组合数: {len(genre_counter)}")
    print(f"\n  Top 20 最多的流派组合:")
    print("  " + "-" * 60)
    
    for idx, (genre_str, count) in enumerate(genre_counter.most_common(20), 1):
        print(f"  {idx:2d}. {genre_str:50s} → {count:5d} 首")
    
    # ===== 拆分成独立标签 =====
    print("\n" + "=" * 70)
    print("📊 2. 拆分后的独立流派标签 (按逗号分割)")
    print("=" * 70)
    
    all_tags = []
    for genre_str in all_genres:
        tags = [tag.strip().lower() for tag in genre_str.split(',')]
        all_tags.extend(tags)
    
    tag_counter = Counter(all_tags)
    
    print(f"\n  不同标签数: {len(tag_counter)}")
    print(f"\n  Top 30 最多的标签:")
    print("  " + "-" * 60)
    
    for idx, (tag, count) in enumerate(tag_counter.most_common(30), 1):
        print(f"  {idx:2d}. {tag:40s} → {count:5d} 首")
    
    # ===== 查看 Shape of You 的流派 =====
    print("\n" + "=" * 70)
    print("🎵 3. 查找 'Shape of You' 的流派")
    print("=" * 70)
    
    search_payload = {
        "filter": {
            "must": [
                {"key": "track_name", "match": {"text": "Shape of You"}}
            ]
        },
        "limit": 10,
        "with_payload": True,
        "with_vector": False
    }
    
    try:
        resp = requests.post(
            f"{QDRANT_URL}/collections/{COLLECTION_NAME}/points/scroll",
            json=search_payload,
            timeout=30
        )
        resp.raise_for_status()
        data = resp.json()
        points = data.get('result', {}).get('points', [])
        
        if points:
            print(f"\n  找到 {len(points)} 条相关记录:\n")
            for idx, point in enumerate(points, 1):
                payload = point.get('payload', {})
                print(f"  {idx}. track_name: {payload.get('track_name')}")
                print(f"     artist_name: {payload.get('artist_name')}")
                print(f"     artist_genres: {payload.get('artist_genres')}")
                print()
        else:
            print("  ❌ 未找到 'Shape of You'")
            
    except Exception as e:
        print(f"  ❌ 搜索失败: {e}")
    
    # ===== 查看 Die With A Smile 的流派 =====
    print("\n" + "=" * 70)
    print("🎵 4. 查找 'Die With A Smile' 的流派")
    print("=" * 70)
    
    search_payload = {
        "filter": {
            "must": [
                {"key": "track_name", "match": {"text": "Die With A Smile"}}
            ]
        },
        "limit": 10,
        "with_payload": True,
        "with_vector": False
    }
    
    try:
        resp = requests.post(
            f"{QDRANT_URL}/collections/{COLLECTION_NAME}/points/scroll",
            json=search_payload,
            timeout=30
        )
        resp.raise_for_status()
        data = resp.json()
        points = data.get('result', {}).get('points', [])
        
        if points:
            print(f"\n  找到 {len(points)} 条相关记录:\n")
            for idx, point in enumerate(points, 1):
                payload = point.get('payload', {})
                print(f"  {idx}. track_name: {payload.get('track_name')}")
                print(f"     artist_name: {payload.get('artist_name')}")
                print(f"     artist_genres: {payload.get('artist_genres')}")
                print()
        else:
            print("  ❌ 未找到 'Die With A Smile'")
            
    except Exception as e:
        print(f"  ❌ 搜索失败: {e}")
    
    # ===== 按标签统计最相似的歌曲 =====
    print("\n" + "=" * 70)
    print("🔗 5. 与 'Shape of You' 共用标签的歌曲数量")
    print("=" * 70)
    
    shape_of_you_genres = []
    for point in points:
        genre = point.get('payload', {}).get('artist_genres', '')
        if genre:
            shape_of_you_genres = [g.strip().lower() for g in genre.split(',')]
            break
    
    if shape_of_you_genres:
        print(f"\n  'Shape of You' 的流派标签: {shape_of_you_genres}\n")
        
        for tag in shape_of_you_genres:
            count = tag_counter.get(tag, 0)
            print(f"  包含 '{tag}' 的歌曲数: {count}")
            
            # 显示前5个示例
            examples = []
            for genre_str in all_genres:
                if tag in genre_str.lower():
                    examples.append(genre_str)
                    if len(examples) >= 5:
                        break
            
            if examples:
                print(f"    示例组合: {', '.join(examples[:3])}")
            print()
    
    # ===== 推荐优化建议 =====
    print("\n" + "=" * 70)
    print("💡 6. 优化建议")
    print("=" * 70)
    
    if 'pop' in tag_counter and tag_counter['pop'] > 1000:
        print("  ⚠️  'pop' 标签过于普遍，匹配时应该排除或降低权重")
    
    if 'dance pop' in tag_counter:
        print(f"  ✅ 'dance pop' 有 {tag_counter['dance pop']} 首，可作为 'Shape of You' 的替代标签")
    
    if 'reggaeton' in tag_counter:
        print(f"  ✅ 'reggaeton' 有 {tag_counter['reggaeton']} 首")
    
    if 'trap latino' in tag_counter:
        print(f"  ✅ 'trap latino' 有 {tag_counter['trap latino']} 首")
    
    print("\n" + "=" * 70)

if __name__ == "__main__":
    analyze_genres()