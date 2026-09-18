# webtest/reason_generator.py

from webtest.ollama_client import client


def _fmt_list(value, limit=5):
    """把 [{name,weight}] / {name:count} / [str] 统一成逗号分隔的名字串"""
    if not value:
        return ""
    if isinstance(value, dict):
        top = sorted(value.items(), key=lambda x: -x[1])[:limit]
        names = [k for k, _ in top]
    else:
        names = [
            (v.get("name") if isinstance(v, dict) else str(v))
            for v in list(value)[:limit]
        ]
    return ", ".join(n for n in names if n)


def generate_song_reason(
    rec_track: str,
    rec_artist: str,
    rec_genres: str,
    similarity: float = 0.0,
    mode: str = "seed",
    seed_track: str = "",
    seed_artist: str = "",
    seed_genres: str = "",
    intent_query: str = "",
    user_summary: str = "",
    preferred_genres=None,
    liked_artists=None,
) -> str:
    """
    为单首推荐歌曲生成理由。

    Args:
        mode: 推荐来源，决定提示词的组织方式
              - "seed"    歌名找相似：有明确种子歌，强调两首歌的共同点
              - "profile" 每日推荐：**没有**种子歌，强调与用户口味的契合
              - "intent"  描述找歌：**没有**种子歌，强调与用户描述/选中标签的匹配
        intent_query: mode="intent" 时的用户原始描述

    Returns: 理由文本（失败时返回基于事实的兜底文案）
    """
    pref_str = _fmt_list(preferred_genres)
    liked_str = _fmt_list(liked_artists, limit=20)

    # 这首歌是否命中"用户以前喜欢过的歌手"
    liked_hit = bool(liked_str) and bool(rec_artist) and (rec_artist in liked_str)

    if mode == "profile":
        return _reason_from_profile(
            rec_track, rec_artist, rec_genres, pref_str, liked_hit
        )
    if mode == "intent":
        return _reason_from_intent(
            rec_track, rec_artist, rec_genres, intent_query,
            seed_genres, pref_str, liked_hit
        )
    return _reason_from_seed(
        rec_track, rec_artist, rec_genres, similarity,
        seed_track, seed_artist, seed_genres, pref_str, liked_hit
    )


def _safe_generate(prompt, max_tokens=1500, temperature=0.9):
    try:
        out = client.generate(prompt=prompt, max_tokens=max_tokens, temperature=temperature)
        return out.strip() if out else ""
    except Exception as e:
        print(f"[单曲理由] 生成失败: {type(e).__name__}: {e}")
        return ""


def _reason_from_seed(rec_track, rec_artist, rec_genres, similarity,
                      seed_track, seed_artist, seed_genres, pref_str, liked_hit):
    """模式一：有种子歌，解释两首歌为什么相似"""
    seed_set = {g.strip().lower() for g in (seed_genres or "").split(",") if g.strip()}
    rec_set = {g.strip().lower() for g in (rec_genres or "").split(",") if g.strip()}
    common = seed_set & rec_set
    common_str = ", ".join(sorted(common)) if common else "无"

    user_block = ""
    if pref_str:
        user_block += f"用户偏好流派：{pref_str}\n"
    if liked_hit:
        user_block += f"用户此前喜欢过 {rec_artist} 的歌\n"

    prompt = f"""{user_block}种子歌曲：《{seed_track}》- {seed_artist}（流派：{seed_genres}）
推荐歌曲：《{rec_track}》- {rec_artist}（流派：{rec_genres}，相似度：{similarity:.1%}）
两首歌的共同流派：{common_str}

请用一段具体、自然的中文说明为什么推荐这首歌，要求：
1. 指出两首歌在音乐上的具体共同点（流派、编曲、节奏、情绪等）
2. 如果用户偏好与这首歌相关，自然地提到
3. 不要只说「相似」「风格接近」，要说具体特征
4. 不要重复歌名
5. 用中文回答，流派名可以保留英文

直接输出理由，不要加前缀。"""

    reason = _safe_generate(prompt)
    if reason:
        return reason
    if common:
        return (f"这首歌与种子歌曲同属于 {common_str} 风格，"
                f"在节奏、配器与整体听感上较为接近，因此推荐给你。")
    return ""


def _reason_from_profile(rec_track, rec_artist, rec_genres, pref_str, liked_hit):
    """
    模式二：每日推荐（没有种子歌）。
    解释"这首歌为什么符合你的口味"。
    """
    taste = pref_str or "（暂无明确偏好记录）"
    prompt = f"""你是一位音乐推荐助手，正在向用户解释推荐理由。

这位用户的口味偏好：{taste}
{"该歌手是用户以前喜欢过的，可自然提及。" if liked_hit else ""}

本次推荐的歌曲：《{rec_track}》- {rec_artist}（流派：{rec_genres}）

请用一段具体、自然的中文说明为什么推荐这首歌，要求：
1. 结合上面提到的用户偏好流派，说明这首歌在风格、编曲、节奏或情绪上如何契合
2. 不要只说「符合你的口味」这种空话，要指出具体的音乐特征
3. 如果该歌手的作品与用户偏好方向一致，可以点出来
4. 不要重复歌名
5. 用中文回答，流派名可以保留英文

直接输出理由，不要加前缀。"""

    reason = _safe_generate(prompt)
    if reason:
        return reason
    if pref_str:
        return (f"这首歌属于 {rec_genres or '你常听的风格'}，"
                f"与你偏好的 {pref_str} 方向一致，因此推荐给你。")
    return ""


def _reason_from_intent(rec_track, rec_artist, rec_genres, intent_query,
                        matched_tags, pref_str, liked_hit):
    """
    模式三：描述找歌（没有种子歌）。
    解释"这首歌为什么符合你的描述"。
    """
    desc = intent_query or "（未提供描述）"
    tags = matched_tags or ""
    prompt = f"""你是一位音乐推荐助手，正在向用户解释推荐理由。

用户的需求描述：「{desc}」
系统理解出的音乐标签：{tags or "（无）"}
{"用户偏好流派：" + pref_str if pref_str else ""}
{"该歌手是用户以前喜欢过的，可自然提及。" if liked_hit else ""}

本次推荐的歌曲：《{rec_track}》- {rec_artist}（流派：{rec_genres}）

请用一段具体、自然的中文说明为什么推荐这首歌，要求：
1. 结合用户的需求描述，说明这首歌在风格、场景、情绪或节奏上如何对应
2. 不要只说「符合你的需求」这种空话，要指出具体的音乐特征
3. 不要重复用户的描述原文，也不要重复歌名
4. 用中文回答，流派名可以保留英文

直接输出理由，不要加前缀。"""

    reason = _safe_generate(prompt)
    if reason:
        return reason
    if rec_genres:
        return (f"这首歌属于 {rec_genres}，"
                f"在你的需求「{desc}」对应的方向上比较契合，因此推荐给你。")
    return ""