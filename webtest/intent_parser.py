# webtest/intent_parser.py
import json
from .ollama_client import client


class IntentParser:
    """把用户自然语言输入解析成结构化意图"""

    VALID_TYPES = {"seed_similar", "filter_only", "artist", "hot"}

    def parse(self, user_input, verbose=False):
        prompt = f"""分析用户的音乐请求，只输出 JSON。

用户说："{user_input}"

输出格式：
{{
  "type": "seed_similar" | "filter_only" | "artist" | "hot",
  "seed_track": "歌名或null",
  "seed_artist": "歌手或null",
  "filters": {{
    "genre": "英文流派或null",
    "mood": "情绪或null",
    "decade": "年代或null"
  }}
}}

类型说明：
- seed_similar：用户提到了具体歌名，想找相似的歌
- filter_only：用户描述了流派、情绪、年代，没有具体歌名
- artist：用户只提到了歌手名
- hot：用户想要热门推荐、随便听听

示例：
"Shape of You" → {{"type": "seed_similar", "seed_track": "Shape of You", "seed_artist": null, "filters": {{}}}}
"周杰伦" → {{"type": "artist", "seed_track": null, "seed_artist": "周杰伦", "filters": {{}}}}
"摇滚" → {{"type": "filter_only", "seed_track": null, "seed_artist": null, "filters": {{"genre": "rock"}}}}
"开心的英式流行" → {{"type": "filter_only", "seed_track": null, "seed_artist": null, "filters": {{"genre": "uk pop", "mood": "happy"}}}}
"90年代摇滚" → {{"type": "filter_only", "seed_track": null, "seed_artist": null, "filters": {{"genre": "rock", "decade": "1990s"}}}}
"推荐几首好听的" → {{"type": "hot", "seed_track": null, "seed_artist": null, "filters": {{}}}}

只输出 JSON，不要解释。"""

        response = client.generate(prompt, max_tokens=200, temperature=0.1, format="json")

        if verbose:
            print(f"[意图] 输入: {user_input}")
            print(f"[意图] 原始: {response}")

        try:
            result = json.loads(response)
        except json.JSONDecodeError:
            print(f"[意图] JSON 解析失败: {response}")
            return self._default()

        return self._validate(result)

    def _validate(self, result):
        """校验并补全字段"""
        if not isinstance(result, dict):
            return self._default()

        t = result.get("type", "")
        if t not in self.VALID_TYPES:
            return self._default()

        filters = result.get("filters") or {}
        if not isinstance(filters, dict):
            filters = {}

        return {
            "type": t,
            "seed_track": result.get("seed_track") or None,
            "seed_artist": result.get("seed_artist") or None,
            "filters": {
                "genre": filters.get("genre") or None,
                "mood": filters.get("mood") or None,
            },
        }

    def _default(self):
        return {
            "type": "hot",
            "seed_track": None,
            "seed_artist": None,
            "filters": {"genre": None, "mood": None},
        }


intent_parser = IntentParser()