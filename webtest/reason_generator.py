# webtest/reason_generator.py

from webtest.ollama_client import client


# 流派中文映射（可选，让理由更自然）
GENRE_CN = {
    "pop": "流行",
    "rock": "摇滚",
    "hip hop": "嘻哈",
    "rap": "说唱",
    "jazz": "爵士",
    "electronic": "电子",
    "edm": "电子舞曲",
    "dance pop": "舞曲流行",
    "indie": "独立",
    "folk": "民谣",
    "r&b": "节奏布鲁斯",
    "soul": "灵魂乐",
    "metal": "金属",
    "punk": "朋克",
    "classical": "古典",
    "country": "乡村",
    "blues": "布鲁斯",
    "reggae": "雷鬼",
    "latin": "拉丁",
    "k-pop": "韩流",
    "j-pop": "日系流行",
    "anime": "动漫",
    "j-rock": "日系摇滚",
    "uk pop": "英式流行",
    "brostep": "回响贝斯",
    "dubstep": "回响贝斯",
}


def _translate_genres(genres: str) -> str:
    """把英文流派翻译成中文（部分翻译，保留原文）"""
    if not genres:
        return ""
    parts = [g.strip() for g in genres.split(",") if g.strip()]
    translated = []
    for p in parts[:3]:
        cn = GENRE_CN.get(p.lower(), "")
        if cn:
            translated.append(f"{cn}({p})")
        else:
            translated.append(p)
    return "、".join(translated)


def generate_recommend_reason(seed: dict, results: list) -> str:
    """
    用大模型生成推荐理由

    Args:
        seed: 种子歌曲 dict（含 track_name, artist_name, artist_genres）
        results: 推荐结果列表（含 track_name, artist_name, artist_genres, similarity）

    Returns:
        推荐理由字符串
    """
    if not results:
        return ""

    seed_track = seed.get("track_name", "")
    seed_artist = seed.get("artist_name", "")
    seed_genres = _translate_genres(seed.get("artist_genres", ""))

    # 只取前 8 首，避免 prompt 太长
    top_results = results[:8]
    result_lines = []
    for i, r in enumerate(top_results, 1):
        genres = _translate_genres(r.get("artist_genres", ""))
        sim = r.get("similarity", 0) * 100
        result_lines.append(
            f"{i}. 《{r.get('track_name', '')}》- {r.get('artist_name', '')}"
            f"（流派：{genres or '未知'}，相似度：{sim:.1f}%）"
        )
    results_text = "\n".join(result_lines)

    prompt = f"""你是一个音乐推荐助手。用户听了《{seed_track}》- {seed_artist}（流派：{seed_genres}），系统根据声学特征推荐了以下歌曲：

{results_text}

请用一段简洁自然的中文（80-120字），概括性地解释这些推荐歌曲与种子歌曲的共同音乐特点。不要逐首介绍，也不要重复歌名，直接说明推荐逻辑。"""

    try:
        reason = client.generate(
            prompt=prompt,
            max_tokens=200,
            temperature=0.7,
        )
        if reason:
            return reason.strip()
    except Exception as e:
        print(f"[推荐理由] 生成失败: {e}")

    return ""