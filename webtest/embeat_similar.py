# webtest/embeat_similar.py

import re

import requests
import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.http import models as qdrant_models


DUP_VERSION_MARKERS = (
    "remix", "remaster", "remastered", "re-recorded", "rerecorded",
    "live", "deluxe", "anniversary", "edition", "version",
    "radio edit", "single edit", "album edit", "extended mix",
    "instrumental", "acoustic version", "demo", "reprise",
    "sped up", "slowed", "tiktok", "bonus track",
)


def _is_dup_version(track_name_lower):
    """
    判断这是否为"同一首歌的重复版本"（Remaster/Live/Deluxe/Edit...）。

    为什么要挡：库里同一首歌有大量副本，且**向量完全相同**
    （实测 Bohemian Rhapsody 的 - Remaster 副本 raw cos = 1.000000），
    不挡的话推荐位会被同一首歌的多个版本刷屏。
    注意用括号/短横线限定的形式匹配，避免误伤歌名本身就含这些词的原创曲
    （如 "Live and Let Die"）——所以只匹配出现在括号或 " - " 之后的情形。
    """
    if not track_name_lower:
        return False
    name = str(track_name_lower).lower()

    # 只在括号内 / " - " 之后 / 结尾处找标记，降低误伤
    segments = []
    segments += re.findall(r"\(([^)]*)\)", name)      # (Remastered 2011)
    segments += re.findall(r"\[([^\]]*)\]", name)     # [Live]
    if " - " in name:
        segments.append(name.split(" - ", 1)[1])      # Xxx - Remaster
    for seg in segments:
        for m in DUP_VERSION_MARKERS:
            if m in seg:
                return True
    return False


class EmbeatSimilar:

    """只实现 Embeat 的声学相似一路召回"""

    def __init__(
        self,
        qdrant_url: str = "http://127.0.0.1:6333",
        collection_name: str = "spotify_tracks",
        qdrant_timeout: int = 30,
    ):
        self.qdrant_url = qdrant_url.rstrip("/")
        self.collection_name = collection_name
        self.qdrant_timeout = qdrant_timeout
        self.client = QdrantClient(
            url=self.qdrant_url,
            port=6333,
            timeout=qdrant_timeout,
        )
        # 去均值校准用的全库均值向量（懒加载，失败则置 None 表示不校凇）
        self._mu_vec = None
        self._calibration_ready = False

    def _ensure_calibration(self):
        """首次调用时加载均值向量；失败不抛错，只是关闭校准。"""
        if self._calibration_ready:
            return
        self._calibration_ready = True
        try:
            from .vector_calibration import get_mean_vec
            self._mu_vec = get_mean_vec()
        except Exception as e:
            print(f"[校准] 不可用，跳过: {e}")
            self._mu_vec = None

    # ========== 1. 查找种子歌曲 ==========
    def find_seed(self, track_id="", isrc="", track_name="", artist_name=""):
        must_conditions = []

        if track_id:
            must_conditions.append(
                qdrant_models.FieldCondition(
                    key="track_id",
                    match=qdrant_models.MatchValue(value=track_id),
                )
            )
        elif isrc:
            must_conditions.append(
                qdrant_models.FieldCondition(
                    key="isrc",
                    match=qdrant_models.MatchValue(value=isrc.upper()),
                )
            )
        elif track_name and artist_name:
            must_conditions.append(
                qdrant_models.FieldCondition(
                    key="track_name",
                    match=qdrant_models.MatchText(text=track_name),
                )
            )
            must_conditions.append(
                qdrant_models.FieldCondition(
                    key="artist_name",
                    match=qdrant_models.MatchText(text=artist_name),
                )
            )
        else:
            return None

        try:
            records, _ = self.client.scroll(
                collection_name=self.collection_name,
                scroll_filter=qdrant_models.Filter(must=must_conditions),
                limit=20,
                with_payload=True,
                with_vectors=True,
            )
        except Exception as e:
            print(f"[错误] 查找种子失败: {e}")
            return None

        if not records:
            print(f"[调试] 未找到: track_name={track_name}, artist_name={artist_name}")
            return None

        # 优先返回：不带 remix、流行度最高的版本
        def score_record(r):
            payload = r.payload or {}
            name = str(payload.get("track_name") or "").lower()
            pop = float(payload.get("popularity") or 0.0)
            # 不带 remix 的优先
            is_remix = 1 if "remix" in name else 0
            # 歌名越短越接近原版
            name_penalty = len(name)
            return (is_remix, -pop, name_penalty)

        records_sorted = sorted(records, key=score_record)
        best = records_sorted[0]

        p = best.payload
        print(f"[种子] {p.get('track_name')} - {p.get('artist_name')} "
            f"(popularity={p.get('popularity')}, genre_idx={p.get('artist_genre_idx')})")

        return best

    # ========== 1.5 候选列表（宽松匹配，供用户选择） ==========
    def find_candidates(self, track_name="", artist_name="", limit=10):
        """
        宽松匹配候选：只按 track_name 匹配，artist_name 作为排序权重。
        返回多条，供用户选择。
        """
        if not track_name:
            return []

        must_conditions = [
            qdrant_models.FieldCondition(
                key="track_name",
                match=qdrant_models.MatchText(text=track_name),
            )
        ]

        try:
            records, _ = self.client.scroll(
                collection_name=self.collection_name,
                scroll_filter=qdrant_models.Filter(must=must_conditions),
                limit=50,
                with_payload=True,
                with_vectors=False,
            )
        except Exception as e:
            print(f"[候选] 查询失败: {e}")
            return []

        if not records:
            return []

        input_artists = self._split_artists(artist_name)

        def artist_score(payload):
            cand_artists = self._split_artists(payload.get("artist_name") or "")
            if not input_artists:
                return 0
            if input_artists == cand_artists:
                return 2
            if input_artists.issubset(cand_artists):
                return 1
            if input_artists & cand_artists:
                return 0
            return -1

        def score_record(r):
            payload = r.payload or {}
            name = str(payload.get("track_name") or "").lower()
            pop = float(payload.get("popularity") or 0.0)
            is_remix = 1 if "remix" in name else 0
            a_score = artist_score(payload)
            name_penalty = len(name)
            return (is_remix, -a_score, -pop, name_penalty)

        records_sorted = sorted(records, key=score_record)

        results = []
        seen = set()
        for r in records_sorted:
            p = r.payload or {}
            key = (p.get("track_name", ""), p.get("artist_name", ""))
            if key in seen:
                continue
            seen.add(key)
            results.append({
                "track_id": p.get("track_id", ""),
                "track_name": p.get("track_name", ""),
                "artist_name": p.get("artist_name", ""),
                "artist_genres": p.get("artist_genres", ""),
                "popularity": p.get("popularity", 0.0),
            })
            if len(results) >= limit:
                break

        return results

    def _split_artists(self, s):
        """把 'Ed Sheeran & Beyoncé' 拆成 {'ed sheeran', 'beyoncé'}"""
        s = str(s or "").lower().strip()
        for sep in [" feat. ", " featuring ", " with ", " & ", " / ", ", ", " x "]:
            s = s.replace(sep, "|")
        return {a.strip() for a in s.split("|") if a.strip()}

    # ========== 2. 声学相似召回 ==========
    def search_similar(
        self,
        seed_vector: list,
        candidate_limit: int = 200,
        artist_genre_idx: int = 0,
    ):
        """
        声学相似召回（对应官方 search_vector_similar_record）

        Args:
            seed_vector: 种子向量
            candidate_limit: 召回数量
            artist_genre_idx: 流派索引（>0 时过滤同流派）
        """
        query_filter = None
        if artist_genre_idx > 0:
            query_filter = qdrant_models.Filter(
                must=[
                    qdrant_models.FieldCondition(
                        key="artist_genre_idx",
                        match=qdrant_models.MatchValue(value=artist_genre_idx),
                    )
                ]
            )

        try:
            response = self.client.query_points(
                collection_name=self.collection_name,
                query=seed_vector,
                query_filter=query_filter,
                limit=candidate_limit,
                with_payload=True,
                with_vectors=True,   # 需要向量才能算去均值校准相似度
            )
        except Exception as e:
            print(f"[错误] 向量搜索失败: {e}")
            return []

        # 过滤掉没有流派的歌曲（官方逻辑）
        results = [r for r in response.points if r.payload.get("artist_genres", "") != ""]
        return results

    # ========== 3. 过滤候选 ==========
    def _genres_set(self, genre_str):
        """把 'uk pop, pop' 转成 {'uk pop', 'pop'}"""
        if not genre_str:
            return set()
        return {g.strip().lower() for g in genre_str.split(",") if g.strip()}

    def filter_candidates(self, seed_payload, candidates, top_k=20, seed_center=None):

        """
        过滤候选：
        - 种子有流派：要求候选流派有交集，idx 相同加高分
        - 种子无流派：不做流派过滤，不做流派加分
        """
        seed_track_name = str(seed_payload.get("track_name") or "").lower().split(" (")[0].split(" - ")[0].strip()
        seed_artist_name = str(seed_payload.get("artist_name") or "").lower().strip()
        seed_genre_str = seed_payload.get("artist_genres", "")
        seed_genre_idx = int(seed_payload.get("artist_genre_idx") or 0)
        seed_popularity = float(seed_payload.get("popularity") or 0.0)
        seed_track_id = str(seed_payload.get("track_id") or "")

        seed_genres_set = self._genres_set(seed_genre_str)
        # 种子流派为空的标志
        seed_has_genre = bool(seed_genres_set) or seed_genre_idx > 0

        print(f"[过滤] 种子流派: {seed_genre_str or '（空）'}")
        print(f"[过滤] 种子流派 idx: {seed_genre_idx}")
        print(f"[过滤] 是否启用流派过滤: {seed_has_genre}")

        min_popularity = min(seed_popularity, 0.1)
        same_artist_ratio = 0.15
        max_same_artist = max(1, int(top_k * same_artist_ratio))
        same_artist_counter = 0

        candidates_scored = []

        for candidate in candidates:
            payload = candidate.payload or {}
            payload_track_id = str(payload.get("track_id") or "").strip()
            payload_track_name = str(payload.get("track_name") or "").lower().split(" (")[0].split(" - ")[0].strip()
            payload_artist_name = str(payload.get("artist_name") or "").lower().strip()
            payload_genre_str = payload.get("artist_genres", "")
            payload_genre_idx = int(payload.get("artist_genre_idx") or 0)
            payload_popularity = float(payload.get("popularity") or 0.0)
            payload_similarity = float(candidate.score)

            # 原始余弦被"全正向量常数基线"抬到 ~0.986（随机也有这么高），
            # 不能读作相似度。这里额外算一个去均值后的校准值 _cal_sim，
            # 它才是可解释的（随机≈0，推荐≈0.2~0.8）。用途见 vector_calibration.py
            _cal_sim = None
            if (self._mu_vec is not None and seed_center is not None
                    and getattr(candidate, "vector", None)):
                try:
                    from .vector_calibration import calibrated_cos
                    _cal_sim = calibrated_cos(candidate.vector, seed_center,
                                              self._mu_vec)
                except Exception:
                    _cal_sim = None



            # --- 基础过滤 ---
            if not payload_track_id or not payload_track_name:
                continue
            if payload_track_id == seed_track_id:
                continue
            if payload_track_name == seed_track_name and payload_artist_name == seed_artist_name:
                continue
            # 过滤重复版本（原实现只挡 remix，导致 Remaster/Live/Deluxe
            # 等副本挤占推荐位。实测 Bohemian Rhapsody 前 15 个候选全是
            # 它自己的 - Remaster 副本，向量完全相同）
            # 注意：必须用**原始**歌名判断 —— 下一行的 payload_track_name
            # 会把 " - xxx" 和 "(xxx)" 截掉，截完就认不出 Remaster 了。
            if _is_dup_version(payload.get("track_name") or ""):
                continue
            if payload_popularity < min_popularity:
                continue

            # ===== 流派交集入围（仅当种子有流派时才启用） =====
            payload_genres_set = self._genres_set(payload_genre_str)

            if seed_has_genre:
                # 种子有流派：要求候选必须有交集
                if payload_genres_set:
                    common = seed_genres_set & payload_genres_set
                    if not common:
                        continue  # 无交集，直接淘汰
                else:
                    # 候选流派为空，且种子有流派 → 淘汰
                    continue

            # ===== 流派加分（仅当种子有流派时才启用） =====
            genre_bonus = 0.0
            if seed_has_genre:
                if seed_genre_idx > 0 and payload_genre_idx == seed_genre_idx:
                    # idx 完全相同，+15%
                    genre_bonus += 0.15
                elif seed_genres_set and payload_genres_set:
                    # 按交集数量加分（每个 +3%，最多 3 个）
                    common_count = len(seed_genres_set & payload_genres_set)
                    genre_bonus += min(common_count, 3) * 0.03

            final_score = payload_similarity + genre_bonus

            candidates_scored.append({
                "track_id": payload_track_id,
                "track_name": payload.get("track_name", ""),
                "artist_name": payload.get("artist_name", ""),
                "album_name": payload.get("album_name", ""),
                "artist_genres": payload_genre_str,
                "artist_genre_idx": payload_genre_idx,
                "popularity": payload_popularity,
                "similarity": round(payload_similarity, 4),
                "cal_similarity": round(_cal_sim, 4) if _cal_sim is not None else None,

                "genre_bonus": round(genre_bonus, 4),
                "final_score": round(final_score, 4),
                "payload_track_name": payload_track_name,
                "payload_artist_name": payload_artist_name,
                # 供 ranking.diversity_rerank 的 MMR 算相似度惩罚项。
                # 原实现没带这个字段，导致 MMR 的 penalty 恒为 0（退化成
                # "只有硬约束的贪婪选择"）；带上后 MMR 才真正生效。
                "_vector": ([float(x) for x in candidate.vector]
                            if getattr(candidate, "vector", None) else None),
            })


        # 按最终分数排序
        candidates_scored.sort(key=lambda x: x["final_score"], reverse=True)

        # 去重 + 同歌手限制
        # 原实现只对"种子歌手本人"限流（`== seed_artist_name`），其它歌手
        # 可无限重复，导致如 Bohemian Rhapsody 的结果里 Queen/David Bowie
        # 各占 3~4 席。这里改为对**所有歌手**一视同仁限流。
        result = []
        result_track_names = set()
        artist_seen = {}

        for item in candidates_scored:
            if item["payload_track_name"] in result_track_names:
                continue

            artist = item["payload_artist_name"]
            if artist_seen.get(artist, 0) >= max_same_artist:
                continue

            result_track_names.add(item["payload_track_name"])
            artist_seen[artist] = artist_seen.get(artist, 0) + 1
            result.append(item)
            if len(result) >= top_k:
                break

        return result

    # ========== 4. 主入口 ==========
    def recommend(
         self,
        track_id: str = "",
        isrc: str = "",
        track_name: str = "",
        artist_name: str = "",
        top_k: int = 20,
        candidate_limit: int = None,
    ):
        """主入口：只做声学相似一路召回"""
        # 1. 查找种子
        seed = self.find_seed(
            track_id=track_id,
            isrc=isrc,
            track_name=track_name,
            artist_name=artist_name,
        )
        if seed is None:
            return {"error": "未找到种子歌曲"}

        seed_payload = seed.payload or {}
        seed_vector = seed.vector

        if seed_vector is None:
            return {"error": "种子向量为空"}

        # 校准：加载均值向量，并算出种子的"去均值中心向量"
        self._ensure_calibration()
        _seed_center = None
        if self._mu_vec is not None:
            try:
                from .vector_calibration import center as _center
                _seed_center = _center(seed_vector, self._mu_vec)
            except Exception:
                _seed_center = None

        # 2. 确定流派
        seed_artist_genre_idx = int(seed_payload.get("artist_genre_idx") or 0)
  
        # 3. 声学相似召回
        candidate_limit = max(1, min(int(top_k * 10), 512))
        candidates = self.search_similar(
            seed_vector=seed_vector,
            candidate_limit=candidate_limit,
            artist_genre_idx=seed_artist_genre_idx,
        )

        # 4. 过滤候选
        results = self.filter_candidates(
            seed_payload=seed_payload,
            candidates=candidates,
            top_k=top_k,
            seed_center=_seed_center,
        )


        return {
            "seed": {
                "track_id": seed_payload.get("track_id", ""),
                "track_name": seed_payload.get("track_name", ""),
                "artist_name": seed_payload.get("artist_name", ""),
                "artist_genres": seed_payload.get("artist_genres", ""),
                "artist_genre_idx": seed_artist_genre_idx,
                "popularity": seed_payload.get("popularity", 0.0),
            },
            "results": results,
        }