# scripts/extract_scene_mood_tags.py
import json

KEYWORDS = [
    # 场景
    "sleep", "study", "workout", "running", "driving", "party",
    "coffee", "background", "focus", "relax", "meditation",
    "dinner", "lounge", "gym", "exercise", "work", "reading",
    "travel", "road", "night", "morning",
    # 情绪
    "happy", "sad", "calm", "energetic", "melancholy", "chill",
    "upbeat", "dark", "emotional", "peaceful", "angry",
]

with open("tags_full.json", "r", encoding="utf-8") as f:
    data = json.load(f)

matched = []
seen = set()
for item in data:
    tag = item["tag"].lower()
    if tag in seen:
        continue
    if any(kw in tag for kw in KEYWORDS):
        matched.append(item["tag"])
        seen.add(tag)

matched.sort()
print(f"共 {len(matched)} 个匹配\n")

with open("tags_scene_mood.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(matched))

print("已保存到 tags_scene_mood.txt")
print("\n前 50 个：")
for t in matched[:50]:
    print(f"  {t}")