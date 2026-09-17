# webtest/intent_parser_v2.py
import json
from .ollama_client import client


PROMPT_TEMPLATE = """分析用户的音乐需求，输出结构化意图。

用户说："{user_input}"

输出 JSON，字段：
- genres: 音乐流派，英文，1-3 个。只能是流派名（pop, rock, jazz, folk, electronic, metal 等），不要放情绪词。
- moods: 情绪，英文，0-2 个。只能是情绪（happy, sad, chill, energetic, calm, aggressive 等），不要放流派名。
- scenes: 场景，英文，0-2 个。只能是场景（running, driving, sleep, party, study, coffee shop 等）。
- keywords: 用于全文检索的关键词，0-3 个。
- avoid: 要排除的关键词，英文，0-3 个。
- scenes: 场景，只在用户明确提到场景时才填（running, driving, sleep, party, study, coffee shop）
- 如果没有明确场景，留空
- 不要从情绪推断场
示例：

"工业摇滚" → {{"genres": ["industrial rock"], "moods": ["intense"], "scenes": [], "keywords": ["industrial"], "avoid": []}}

"开心的流行" → {{"genres": ["pop"], "moods": ["happy", "upbeat"], "scenes": [], "keywords": ["happy pop"], "avoid": ["sad", "emo"]}}

"适合跑步听的歌" → {{"genres": ["electronic", "pop"], "moods": ["energetic", "upbeat"], "scenes": ["running"], "keywords": ["running"], "avoid": ["slow", "sad"]}}

"深夜放松的爵士" → {{"genres": ["jazz"], "moods": ["chill", "relaxed"], "scenes": ["late night"], "keywords": ["jazz"], "avoid": ["loud", "energetic"]}}

"失恋了想听点伤感的" → {{"genres": ["pop", "folk"], "moods": ["sad", "melancholy"], "scenes": ["alone"], "keywords": ["sad", "heartbreak"], "avoid": ["happy", "upbeat"]}}

"适合睡前放松的纯音乐" → {{"genres": ["ambient", "classical"], "moods": ["calm", "relaxed"], "scenes": ["sleep"], "keywords": ["sleep", "relax"], "avoid": ["loud", "energetic"]}}

"适合咖啡馆放的背景音乐" → {{"genres": ["jazz"], "moods": ["chill", "calm"], "scenes": ["coffee shop"], "keywords": ["background", "chill"], "avoid": ["loud", "energetic"]}}

"想听点暴躁的重金属" → {{"genres": ["metal"], "moods": ["aggressive", "intense"], "scenes": [], "keywords": ["metal", "aggressive"], "avoid": ["calm", "soft", "acoustic"]}}

"最近流行的抖音神曲" → {{"genres": ["pop"], "moods": ["happy", "upbeat"], "scenes": [], "keywords": ["viral", "tiktok"], "avoid": ["classical", "jazz"]}}
优先选大众、主流的标签，尽量不要选冷门子类
错误示例（不要这样）：
"开心的流行" → {{"genres": ["pop", "happy"]}}   ← happy 是 mood，不该在 genres
"跑步" → {{"genres": ["pop", "running"]}}        ← running 是 scene，不该在 genres

现在请处理：

用户说："{user_input}"

只输出 JSON，不要解释。"""


def parse_intent(user_input):
    prompt = PROMPT_TEMPLATE.format(user_input=user_input)
    resp = client.generate(prompt, max_tokens=300, temperature=0.1, format="json")
    try:
        data = json.loads(resp)
        return {
            "genres": (data.get("genres") or [])[:3],
            "moods": (data.get("moods") or [])[:2],
            "scenes": (data.get("scenes") or [])[:2],
            "keywords": (data.get("keywords") or [])[:3],
            "avoid": (data.get("avoid") or [])[:3],
            "_raw_query": user_input,   # 新增
        }
    except Exception as e:
        print(f"[意图解析] 失败: {e}, 原始: {resp}")
        return {
            "genres": [], "moods": [], "scenes": [],
            "keywords": [user_input], "avoid": [],
            "_raw_query": user_input,
        }