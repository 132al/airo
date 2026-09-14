# webtest/profile_analyzer.py

import json
from webtest.ollama_client import client


def rebuild_profile(user) -> dict:
    """
    根据该用户当前所有反馈，重算档案。
    一个用户一首歌只有一条反馈，所以计数就是行数。
    """
    from webtest.models import UserFeedback

    rows = UserFeedback.objects.filter(user=user)

    liked_tracks = []
    disliked_tracks = []
    preferred_genres = {}
    disliked_genres = {}
    preferred_artists = {}

    for r in rows:
        genres = [g.strip().lower() for g in r.artist_genres.split(",") if g.strip()]

        if r.feedback == "like":
            liked_tracks.append({"track_name": r.track_name, "artist_name": r.artist_name})
            for g in genres:
                preferred_genres[g] = preferred_genres.get(g, 0) + 1
            preferred_artists[r.artist_name] = preferred_artists.get(r.artist_name, 0) + 1
        elif r.feedback == "dislike":
            disliked_tracks.append({"track_name": r.track_name, "artist_name": r.artist_name})
            for g in genres:
                disliked_genres[g] = disliked_genres.get(g, 0) + 1

    profile_data = {
        "liked_tracks": liked_tracks,
        "disliked_tracks": disliked_tracks,
        "preferred_genres": preferred_genres,
        "disliked_genres": disliked_genres,
        "preferred_artists": preferred_artists,
    }

    # 生成/更新 LLM 摘要
    summary = analyze_profile(profile_data)
    if summary:
        profile_data["llm_summary"] = summary

    return profile_data


def analyze_profile(profile_data: dict) -> str:
    """用 LLM 分析用户档案，生成偏好摘要"""
    liked = profile_data.get("liked_tracks", [])[-10:]
    disliked = profile_data.get("disliked_tracks", [])[-10:]
    preferred_genres = profile_data.get("preferred_genres", {})
    disliked_genres = profile_data.get("disliked_genres", {})
    preferred_artists = profile_data.get("preferred_artists", {})

    top_genres = sorted(preferred_genres.items(), key=lambda x: -x[1])[:5]
    top_artists = sorted(preferred_artists.items(), key=lambda x: -x[1])[:5]
    bad_genres = sorted(disliked_genres.items(), key=lambda x: -x[1])[:5]

    if not liked and not disliked and not top_genres:
        return ""

    prompt = f"""根据以下用户音乐行为数据，总结这个用户的音乐偏好）：

喜欢的歌曲：
{json.dumps(liked, ensure_ascii=False)}

不喜欢的歌曲：
{json.dumps(disliked, ensure_ascii=False)}

偏好流派（按喜好程度排序）：
{top_genres}

偏好艺术家：
{top_artists}

不喜欢的流派：
{bad_genres}

请用一段简洁的中文描述这个用户的音乐品味，包括他喜欢的风格特点、不喜欢的东西。不要列举具体歌名。"""

    try:
        summary = client.generate(prompt=prompt, max_tokens=800, temperature=0.7)
        if summary:
            return summary.strip()
    except Exception as e:
        print(f"[档案分析] 失败: {e}")
    return ""