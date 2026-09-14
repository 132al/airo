# webtest/reason_generator.py

from webtest.ollama_client import client

def generate_song_reason(
    seed_track: str,
    seed_artist: str,
    seed_genres: str,
    rec_track: str,
    rec_artist: str,
    rec_genres: str,
    similarity: float,
    user_summary: str = "",
    preferred_genres: dict = None,
    liked_artists: dict = None,
) -> str:
    """为单首推荐歌曲生成理由"""

    seed_set = {g.strip().lower() for g in (seed_genres or "").split(",") if g.strip()}
    rec_set = {g.strip().lower() for g in (rec_genres or "").split(",") if g.strip()}
    common = seed_set & rec_set
    common_str = ", ".join(sorted(common)) if common else "无"

    user_block = ""
    if user_summary:
        user_block += f"用户画像：{user_summary}\n"
    if preferred_genres:
        top = sorted(preferred_genres.items(), key=lambda x: -x[1])[:5]
        user_block += f"用户偏好流派：{', '.join(g for g, _ in top)}\n"
    if liked_artists and rec_artist in liked_artists:
        user_block += f"用户此前喜欢过 {rec_artist} 的歌\n"

    prompt = f"""{user_block}
种子歌曲：《{seed_track}》- {seed_artist}（流派：{seed_genres}）
推荐歌曲：《{rec_track}》- {rec_artist}（流派：{rec_genres}，相似度：{similarity:.1%}）
两首歌的共同流派：{common_str}

请用一段具体、自然的中文说明为什么推荐这首歌，要求：
1. 指出两首歌在音乐上的具体共同点（流派、编曲、节奏、情绪等）
2. 如果用户画像或偏好与这首歌相关，自然地提到
3. 不要只说「相似」「风格接近」，要说具体特征
4. 不要重复歌名
5. 用中文回答，流派名可以保留英文

直接输出理由，不要加前缀。"""

    try:
        reason = client.generate(prompt=prompt, max_tokens=1500, temperature=0.9)
        if reason:
            return reason.strip()
    except Exception as e:
        print(f"[单曲理由] 生成失败: {e}")
    return ""



def generate_recommend_reason(seed: dict, results: list) -> str:
    if not results:
        return ""

    seed_track = seed.get("track_name", "")
    seed_artist = seed.get("artist_name", "")
    seed_genres = seed.get("artist_genres", "")

    top_results = results[:8]
    result_lines = []
    for i, r in enumerate(top_results, 1):
        sim = r.get("similarity", 0) * 100
        genres = r.get("artist_genres", "")
        result_lines.append(
            f"{i}. 《{r.get('track_name', '')}》- {r.get('artist_name', '')}"
            f"（流派：{genres or '未知'}，相似度：{sim:.1f}%）"
        )
    results_text = "\n".join(result_lines)

    prompt = f"""你是一个音乐推荐助手。用户听了《{seed_track}》- {seed_artist}（流派：{seed_genres}），系统根据声学特征推荐了以下歌曲：

{results_text}

请用一段简洁自然的中文（80-120字），概括性地解释这些推荐歌曲与种子歌曲的共同音乐特点。不要逐首介绍，也不要重复歌名，直接说明推荐逻辑。用中文回答，流派名可以保留英文。"""

    try:
        reason = client.generate(prompt=prompt, max_tokens=500, temperature=0.7)
        if reason:
            return reason.strip()
    except Exception as e:
        print(f"[推荐理由] 生成失败: {e}")
    return ""