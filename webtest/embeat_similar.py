# webtest/embeat_similar.py

import requests
import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.http import models as qdrant_models


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
                with_vectors=False,
            )
        except Exception as e:
            print(f"[错误] 向量搜索失败: {e}")
            return []

        # 过滤掉没有流派的歌曲（官方逻辑）
        results = [r for r in response.points if r.payload.get("artist_genres", "") != ""]
        return results

    # ========== 3. 过滤候选 ==========
    def filter_candidates(
        self,
        seed_payload: dict,
        candidates: list,
        top_k: int = 20,
    ):
        """
        过滤候选（对应官方 filter_similar_candidates）
        
        过滤规则：
        - 跳过自己
        - 跳过同名同歌手
        - 跳过重复歌名
        - 跳过 remix
        - 跳过低流行度
        - 跳过流派不匹配
        - 同歌手最多 15%
        """
        result = []
        result_track_names = []

        seed_track_name = str(seed_payload.get("track_name") or "").lower().split(" (")[0].split(" - ")[0].strip()
        seed_artist_name = str(seed_payload.get("artist_name") or "").lower().strip()
        seed_artist_genre_idx = int(seed_payload.get("artist_genre_idx") or 0)
        seed_popularity = float(seed_payload.get("popularity") or 0.0)
        seed_track_id = str(seed_payload.get("track_id") or "")

        # 最小流行度（官方逻辑：min(种子流行度, 0.1)）
        min_popularity = min(seed_popularity, 0.1)

        # 同歌手最大数量（15%）
        same_artist_ratio = 0.15
        max_same_artist = max(1, int(top_k * same_artist_ratio))
        same_artist_counter = 0

        prev_similarity = 1.0
        similarity_eps = 1e-5

        for candidate in candidates:
            payload = candidate.payload or {}

            payload_track_id = str(payload.get("track_id") or "").strip()
            payload_track_name = str(payload.get("track_name") or "").lower().split(" (")[0].split(" - ")[0].strip()
            payload_artist_name = str(payload.get("artist_name") or "").lower().strip()
            payload_artist_genre_idx = int(payload.get("artist_genre_idx") or 0)
            payload_popularity = float(payload.get("popularity") or 0.0)
            payload_similarity = float(candidate.score)

            # --- 过滤规则 ---
            if not payload_track_id or not payload_track_name:
                continue
            if payload_track_id == seed_track_id:
                continue
            if payload_track_name == seed_track_name and payload_artist_name == seed_artist_name:
                continue
            if payload_track_name in result_track_names:
                continue
            if f"{seed_track_name} " in payload_track_name:
                continue
            if "remix" in payload_track_name:
                continue
            if payload_popularity < min_popularity:
                continue

            # 流派过滤
            if seed_artist_genre_idx > 0 and payload_artist_genre_idx != seed_artist_genre_idx:
                continue
            # 跳过未知流派
            if payload_artist_genre_idx == 0:
                continue

            # 同歌手限制
            if same_artist_counter >= max_same_artist and payload_artist_name == seed_artist_name:
                continue

            # 相似度去重（避免连续相同分数）
            if abs(payload_similarity - prev_similarity) < similarity_eps:
                if result and payload_popularity > float(result[-1].get("popularity") or 0.0):
                    result.pop(-1)
                else:
                    continue

            result.append({
                "track_id": payload_track_id,
                "track_name": payload.get("track_name", ""),
                "artist_name": payload.get("artist_name", ""),
                "album_name": payload.get("album_name", ""),
                "artist_genres": payload.get("artist_genres", ""),
                "popularity": payload_popularity,
                "similarity": round(payload_similarity, 4),
            })
            result_track_names.append(payload_track_name)

            if payload_artist_name == seed_artist_name:
                same_artist_counter += 1

            prev_similarity = payload_similarity

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