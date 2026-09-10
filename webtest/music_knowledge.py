# webtest/music_knowledge.py

from webtest.models import MusicKnowledge


def query_artist(artist_name, max_results=2):
    """查询歌手的维基百科描述"""
    if not artist_name:
        return []
    try:
        # 1. 先精确匹配 entity
        results = list(
            MusicKnowledge.objects.using('music')
            .filter(entity__iexact=artist_name)
            .order_by('-text')
            .values('entity', 'text', 'url')[:max_results]
        )

        # 2. 精确匹配不到，退而求其次：entity 中包含艺术家名
        if not results:
            results = list(
                MusicKnowledge.objects.using('music')
                .filter(entity__icontains=artist_name)
                .order_by('-text')
                .values('entity', 'text', 'url')[:max_results]
            )

        # 3. 还查不到，尝试在 text 中搜索
        if not results:
            results = list(
                MusicKnowledge.objects.using('music')
                .filter(text__icontains=artist_name)
                .order_by('-text')
                .values('entity', 'text', 'url')[:max_results]
            )

        # 去重
        seen = set()
        unique = []
        for r in results:
            if r['entity'] not in seen:
                seen.add(r['entity'])
                unique.append(r)
        return unique[:max_results]
    except Exception as e:
        print(f"[知识库] 查询歌手失败 {artist_name}: {e}")
        return []


def query_genre(genre_name, max_results=1):
    """查询流派的维基百科描述"""
    if not genre_name:
        return []
    try:
        results = list(
            MusicKnowledge.objects.using('music')
            .filter(entity__icontains=genre_name)
            .exclude(text__isnull=True)
            .order_by('-text')
            .values('entity', 'text', 'url')[:max_results]
        )

        # 去重
        seen = set()
        unique = []
        for r in results:
            if r['entity'] not in seen:
                seen.add(r['entity'])
                unique.append(r)
        return unique[:max_results]
    except Exception as e:
        print(f"[知识库] 查询流派失败 {genre_name}: {e}")
        return []


def build_knowledge_context(seed: dict, results: list) -> str:
    """为种子歌曲和推荐结果构建知识库上下文"""
    context_parts = []

    # 1. 种子歌手的资料
    seed_artist = seed.get("artist_name", "")
    if seed_artist:
        artist_info = query_artist(seed_artist, max_results=1)
        if artist_info:
            context_parts.append(
                f"【种子歌手：{seed_artist}】\n{artist_info[0]['text']}"
            )

    # 2. 种子流派的资料
    seed_genres = seed.get("artist_genres", "")
    if seed_genres:
        main_genre = seed_genres.split(",")[0].strip()
        if main_genre:
            genre_info = query_genre(main_genre, max_results=1)
            if genre_info:
                context_parts.append(
                    f"【流派：{main_genre}】\n{genre_info[0]['text']}"
                )

    # 3. 推荐歌曲的歌手资料（最多取前 3 个不同歌手）
    seen_artists = set()
    for r in results[:10]:
        artist = r.get("artist_name", "")
        if not artist or artist in seen_artists:
            continue
        if len(seen_artists) >= 3:
            break
        seen_artists.add(artist)

        artist_info = query_artist(artist, max_results=1)
        if artist_info:
            context_parts.append(
                f"【推荐歌手：{artist}】\n{artist_info[0]['text'][:400]}"
            )

    return "\n\n".join(context_parts)