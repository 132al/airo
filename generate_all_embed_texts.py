# scripts/generate_all_embed_texts.py
import json
import time
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed

DEEPSEEK_API_KEY = "sk-de5ec64f40af445d87e66efbe1efdacd"
DEEPSEEK_URL = "https://api.deepseek.com/v1/chat/completions"
MODEL = "deepseek-chat"

INPUT = "all_tags.json"
OUTPUT = "tags_full.json"

MAX_WORKERS = 10
SAVE_EVERY = 50


PROMPT_TEMPLATE = """你是一个音乐知识助手。请为给定的音乐流派标签生成一段用于向量检索的 embed_text。

输出格式：一段自然语言，不要用 |、换行、JSON 等结构化符号。整段写成连贯的话。

必须包含：
1. 标签名（英文原文）
2. 中文名（有通用译法就用，没有就保留英文，不要编造）
3. 用户可能说的口语说法（3-5 个，真实的口语，不要生造）
4. 英文解释（1-2 句，说明这个流派是什么）
5. 和相近标签的区别（1 句，明确说和谁比、区别在哪；如果没有相近标签就省略这句）

长度：80-150 字。

不要：列举具体艺术家或歌曲名。

示例：

标签：industrial rock

输出：
industrial rock（工业摇滚）是一种融合工业音乐与摇滚的流派，用户可能说工业摇滚、工业摇、工业化摇滚。它以扭曲吉他、程序化鼓机和粗粝音景为特征，同时保留摇滚的歌曲结构。和 industrial metal 的区别在于它更偏摇滚结构，没有那么重。

现在请处理：

标签：{tag}

输出："""


def call_deepseek(tag):
    prompt = PROMPT_TEMPLATE.format(tag=tag)
    for attempt in range(3):
        try:
            body = json.dumps({
                "model": MODEL,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.3,
                "max_tokens": 400,
            }, ensure_ascii=False)

            resp = requests.post(
                DEEPSEEK_URL,
                headers={
                    "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
                    "Content-Type": "application/json; charset=utf-8",
                },
                data=body.encode("utf-8"),
                timeout=60,
            )
            if resp.status_code == 200:
                data = resp.json()
                text = data["choices"][0]["message"]["content"].strip()
                if text:
                    return tag, text
            else:
                print(f"[失败] {tag}: HTTP {resp.status_code} {resp.text[:200]}")
        except Exception as e:
            print(f"[异常] {tag}: {e}")
        time.sleep(2)
    return tag, ""


def main():
    with open(INPUT, "r", encoding="utf-8") as f:
        data = json.load(f)

    tags = [item["tag"] for item in data]
    print(f"共 {len(tags)} 个标签")

    results = []
    failed = []

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(call_deepseek, tag): tag for tag in tags}
        for i, future in enumerate(as_completed(futures), 1):
            tag, embed_text = future.result()
            if embed_text:
                results.append({"tag": tag, "embed_text": embed_text})
            else:
                failed.append(tag)
                results.append({"tag": tag, "embed_text": f"{tag} music genre"})

            if i % SAVE_EVERY == 0:
                print(f"进度：{i}/{len(tags)}，成功 {len(results)-len(failed)}，失败 {len(failed)}")
                with open(OUTPUT, "w", encoding="utf-8") as f:
                    json.dump(results, f, ensure_ascii=False, indent=2)

    with open(OUTPUT, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\n完成：{len(results)} 个")
    print(f"失败：{len(failed)} 个")
    if failed:
        print(f"失败列表（前 20）：{failed[:20]}")


if __name__ == "__main__":
    main()