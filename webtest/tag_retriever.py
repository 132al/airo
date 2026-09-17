# webtest/tag_retriever.py
import json
import os
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
from .ollama_client import client
QDRANT_URL = "http://127.0.0.1:6333"
COLLECTION = "tag_index"
MODEL_NAME = "BAAI/bge-small-zh-v1.5"
TAGS_FILE = "tags_full.json"

_model = None
_client = None
_tag_set = None

# ============================================================
# 关键词 → 标签映射表
# 统一：流派 / 地域 / 场景 / 情绪 / 概念
# 每个 tag 带权重，权重越高越优先
# ============================================================
KEYWORD_TAG_MAP = {
    # ===== 流派大类 =====
    "摇滚": [("rock", 1.0), ("classic rock", 0.9), ("hard rock", 0.8), ("alternative rock", 0.8)],
    "流行": [("pop", 1.0), ("dance pop", 0.9), ("pop rock", 0.8), ("synthpop", 0.8)],
    "爵士": [("jazz", 1.0), ("smooth jazz", 0.9), ("vocal jazz", 0.8), ("bossa nova", 0.7)],
    "金属": [("metal", 1.0), ("heavy metal", 0.9), ("nu metal", 0.8), ("thrash metal", 0.8)],
    "电音": [("edm", 1.0), ("house", 0.8), ("techno", 0.8), ("trance", 0.8)],
    "说唱": [("rap", 1.0), ("hip hop", 0.9), ("trap", 0.8), ("pop rap", 0.7)],
    "嘻哈": [("hip hop", 1.0), ("rap", 0.9), ("trap", 0.8), ("pop rap", 0.7)],
    "民谣": [("folk", 1.0), ("folk rock", 0.8), ("indie folk", 0.8)],
    "古典": [("classical", 1.0), ("baroque", 0.8), ("classical era", 0.8)],
    "乡村": [("country", 1.0), ("country rock", 0.8), ("contemporary country", 0.8)],
    "朋克": [("punk", 1.0), ("pop punk", 0.8), ("hardcore punk", 0.8)],
    "雷鬼": [("reggae", 1.0), ("roots reggae", 0.8), ("dancehall", 0.8)],

    # ===== 地域 =====
    "印度": [
        ("filmi", 1.0),
        ("tollywood", 0.8),
        ("kollywood", 0.8),
        ("mollywood", 0.8),
        ("indian classical", 0.7),
    ],
    "宝莱坞": [("filmi", 1.0), ("modern bollywood", 0.9), ("classic bollywood", 0.8)],
    "韩国": [("k-pop", 1.0), ("korean pop", 0.9), ("korean r&b", 0.8)],
    "韩流": [("k-pop", 1.0), ("korean pop", 0.9), ("k-pop boy group", 0.8), ("k-pop girl group", 0.8)],
    "日本": [("j-pop", 1.0), ("j-rock", 0.9), ("city pop", 0.8), ("anime", 0.8)],
    "日系": [("j-pop", 1.0), ("j-rock", 0.9), ("city pop", 0.8), ("anime", 0.8)],
    "华语": [("mandopop", 1.0), ("c-pop", 0.9), ("cantopop", 0.8)],
    "中文": [("mandopop", 1.0), ("c-pop", 0.9), ("cantopop", 0.8)],
    "拉丁": [("latin pop", 1.0), ("reggaeton", 0.9), ("urbano latino", 0.8)],
    "法语": [("french pop", 1.0), ("french hip hop", 0.9), ("chanson", 0.8)],
    "德语": [("german pop", 1.0), ("german hip hop", 0.9), ("schlager", 0.8)],
    "巴西": [("sertanejo", 1.0), ("mpb", 0.9), ("bossa nova", 0.8), ("samba", 0.8)],
    "墨西哥": [("norteno", 1.0), ("banda", 0.9), ("corrido", 0.9), ("ranchera", 0.8)],

    # ===== 场景 =====
    "跑步": [("workout product", 1.0), ("gym hardstyle", 0.8), ("gym phonk", 0.7)],
    "健身": [("gym hardstyle", 1.0), ("gym phonk", 0.9), ("gymcore", 0.9), ("workout product", 0.8)],
    "开车": [("country road", 1.0), ("truck-driving country", 0.9), ("nightrun", 0.8)],
    "公路": [("country road", 1.0), ("truck-driving country", 0.9)],
    "咖啡馆": [("background jazz", 1.0), ("dinner jazz", 0.9), ("lounge", 0.9), ("bossa nova", 0.8)],
    "学习": [("lo-fi study", 1.0), ("study beats", 0.9), ("focus", 0.9), ("focus beats", 0.9)],
    "工作": [("focus", 1.0), ("background music", 0.9), ("lo-fi study", 0.8)],
    "睡前": [("sleep", 1.0), ("lo-fi sleep", 0.9), ("relaxative", 0.9), ("calming instrumental", 0.9)],
    "助眠": [("sleep", 1.0), ("lo-fi sleep", 0.9), ("relaxative", 0.9), ("white noise", 0.7)],
    "冥想": [("meditation", 1.0), ("guided meditation", 0.9), ("world meditation", 0.9)],
    "派对": [("partyschlager", 1.0), ("dark clubbing", 0.9), ("edm", 0.8), ("dance pop", 0.8)],
    "阅读": [("reading", 1.0), ("background piano", 0.9), ("lo-fi study", 0.8)],

    # ===== 情绪 =====
    "伤感": [("melancholia", 1.0), ("sad lo-fi", 0.9), ("indie triste", 0.9), ("sad rap", 0.8)],
    "失恋": [("melancholia", 1.0), ("indie triste", 0.9), ("sad lo-fi", 0.9)],
    "伤心": [("melancholia", 1.0), ("sad lo-fi", 0.9), ("indie triste", 0.9)],
    "开心": [("happy hardcore", 1.0), ("dance pop", 0.8), ("disco", 0.8)],
    "放松": [("relaxative", 1.0), ("chill out", 0.9), ("chill lounge", 0.9), ("ambient", 0.8)],
    "慵懒": [("lounge", 1.0), ("chill lounge", 0.9), ("bossa nova", 0.9), ("dinner jazz", 0.8)],
    "暴躁": [("aggressive phonk", 1.0), ("metal", 0.9), ("hardcore", 0.8), ("punk", 0.8)],
    "浪漫": [("pop romantico", 1.0), ("romantico", 0.9), ("r&b", 0.8), ("soul", 0.8)],
    "怀旧": [("classic rock", 1.0), ("mellow gold", 0.9), ("adult standards", 0.8)],

    # ===== 概念 =====
    "动漫": [("anime", 1.0), ("j-pop", 0.8), ("j-rock", 0.8), ("otacore", 0.7)],
    "二次元": [("anime", 1.0), ("j-pop", 0.8), ("j-rock", 0.8), ("otacore", 0.7)],
    "游戏": [("video game music", 1.0), ("japanese vgm", 0.9), ("chiptune", 0.8)],
    "电影": [("soundtrack", 1.0), ("orchestral soundtrack", 0.9)],
    "纯音乐": [("instrumental", 1.0), ("calming instrumental", 0.9), ("background piano", 0.8)],
}


def _get_model():
    global _model
    if _model is None:
        _model = SentenceTransformer(MODEL_NAME)
    return _model


def _get_client():
    global _client
    if _client is None:
        _client = QdrantClient(url=QDRANT_URL)
    return _client


def _get_tag_set():
    global _tag_set
    if _tag_set is None:
        if os.path.exists(TAGS_FILE):
            with open(TAGS_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
            _tag_set = set(item["tag"] for item in data)
        else:
            _tag_set = set()
        print(f"[tag_retriever] 加载 {len(_tag_set)} 个标签")
    return _tag_set


def search_by_text(text, limit=10):
    """纯向量检索"""
    if not text or not text.strip():
        return []
    model = _get_model()
    qdrant = _get_client()
    vec = model.encode(text.strip(), normalize_embeddings=True).tolist()
    results = qdrant.query_points(
        collection_name=COLLECTION,
        query=vec,
        limit=limit,
    ).points
    return [
        {"tag": r.payload["tag"], "score": r.score}
        for r in results
    ]


def keyword_recall(query, tag_set):
    """
    关键词表匹配：query 含关键词 → 加对应标签
    多个关键词命中同一标签时，取最大权重（不累加）
    """
    hits = {}
    for kw, tag_weights in KEYWORD_TAG_MAP.items():
        if kw in query:
            for tag, weight in tag_weights:
                if tag in tag_set:
                    if tag not in hits or hits[tag] < weight:
                        hits[tag] = weight
    return hits


def multi_recall(query, per_path=15):
    """
    关键词表 + 向量，合并召回。

    优先级：
    1. 关键词表命中（权重高，1.0 + weight * 1.0）
    2. 向量召回（兜底，score * 1.0）

    返回：按 weighted_score 排序的候选列表
    """
    candidates = {}

    def add(tag, score, source):
        if tag not in candidates or candidates[tag]["weighted_score"] < score:
            candidates[tag] = {
                "tag": tag,
                "score": score,
                "weighted_score": score,
                "source": source,
            }

    tag_set = _get_tag_set()

    # 1. 关键词表
    kw_hits = keyword_recall(query, tag_set)
    for tag, weight in kw_hits.items():
        # 关键词命中：1.0 + weight * 1.0（最大 2.0）
        add(tag, 1.0 + weight, "keyword")

    # 2. 向量召回
    for r in search_by_text(query, limit=per_path):
        add(r["tag"], r["score"], "vector")

    result = sorted(candidates.values(), key=lambda x: x["weighted_score"], reverse=True)
    return result


def select_tags(user_input, candidates, top_n=5):
    """LLM 从候选里选标签"""
    if not candidates:
        return []

    # 候选只给 Top-20，减少干扰
    lines = []
    for c in candidates[:20]:
        lines.append(f"- {c['tag']}  (source: {c.get('source', '')})")
    candidates_text = "\n".join(lines)

    prompt = f"""用户说：「{user_input}」

从以下候选标签里选出最匹配的 1-{top_n} 个：

{candidates_text}

要求：
- 只选和用户意图直接相关的标签
- 不要选太宽泛的大类（如 pop、rock、soundtrack），除非用户明确说了
- 不要选地域无关的标签
- 优先选 source 为 keyword 的标签
- 只输出候选列表里有的标签
- 输出 JSON

输出格式：
{{"tags": ["tag1", "tag2"]}}

只输出 JSON。"""

    resp = client.generate(prompt, max_tokens=150, temperature=0.1, format="json")
    try:
        data = json.loads(resp)
        tags = data.get("tags", [])
        valid = {c["tag"] for c in candidates}
        selected = [t for t in tags if t in valid][:top_n]
        if selected:
            return selected
    except Exception as e:
        print(f"[标签选择] 失败: {e}")
    # fallback
    return [c["tag"] for c in candidates[:top_n]]