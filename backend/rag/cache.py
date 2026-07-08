"""
rag/cache.py — Semantic similarity cache backed by Qdrant.

How it works:
  1. Incoming query is embedded with OpenAI text-embedding-3-small.
  2. The embedding is compared (cosine similarity) against cached embeddings in Qdrant.
  3. If the best match exceeds the threshold, the cached answer is returned immediately.
  4. On a miss, the caller runs the full RAG chain and stores the result here.

Performance: O(log n) using Qdrant's HNSW index for scalable cache lookup.
"""
from __future__ import annotations

import logging
import time
import uuid
from dataclasses import dataclass, field

import numpy as np

from ..config import settings
from .vectorstore import get_embeddings, EMBEDDING_DIM
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, HnswConfigDiff

logger = logging.getLogger(__name__)

CACHE_COLLECTION = "semantic_cache"


@dataclass
class CacheResult:
    hit: bool
    answer: str | None
    similarity: float
    latency_ms: float
    source: str  # "CACHE HIT" | "LLM GENERATED"
    cached_query: str | None = field(default=None)


class SemanticCache:
    """Qdrant-backed semantic cache using cosine similarity on query embeddings."""

    def __init__(self) -> None:
        self._client = QdrantClient(
            url=settings.qdrant_url,
            timeout=30,
        )
        self._embeddings = get_embeddings()
        self.threshold = settings.similarity_threshold
        self._ensure_cache_collection()

    def _ensure_cache_collection(self) -> None:
        """Create the cache collection if it doesn't exist."""
        try:
            existing = {c.name for c in self._client.get_collections().collections}
            if CACHE_COLLECTION not in existing:
                self._client.create_collection(
                    collection_name=CACHE_COLLECTION,
                    vectors_config=VectorParams(
                        size=EMBEDDING_DIM,
                        distance=Distance.COSINE,
                    ),
                    hnsw_config=HnswConfigDiff(
                        m=16,
                        ef_construct=100,
                    ),
                )
                logger.info("Created semantic cache collection: %s", CACHE_COLLECTION)
        except Exception as exc:
            logger.warning("Could not create cache collection: %s", exc)

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _embed(self, text: str) -> list[float]:
        return self._embeddings.embed_query(text)

    @staticmethod
    def _cosine(v1: list[float], v2: list[float]) -> float:
        a, b = np.array(v1, dtype=np.float32), np.array(v2, dtype=np.float32)
        norm = np.linalg.norm(a) * np.linalg.norm(b)
        if norm == 0:
            return 0.0
        return float(np.dot(a, b) / norm)

    # ── Public API ────────────────────────────────────────────────────────────

    def lookup(self, query: str) -> CacheResult:
        """Check the cache for a semantically similar query."""
        t0 = time.perf_counter()
        query_vec = self._embed(query)

        try:
            # Use Qdrant's HNSW for O(log n) similarity search
            results = self._client.search(
                collection_name=CACHE_COLLECTION,
                query_vector=query_vec,
                limit=1,
                with_payload=True,
                score_threshold=self.threshold,
            )

            if results:
                best = results[0]
                elapsed_ms = (time.perf_counter() - t0) * 1_000
                logger.info("Cache HIT — similarity=%.4f query=%r", best.score, query)
                return CacheResult(
                    hit=True,
                    answer=best.payload.get("answer"),
                    similarity=best.score,
                    latency_ms=elapsed_ms,
                    source="CACHE HIT",
                    cached_query=best.payload.get("query"),
                )
        except Exception as exc:
            logger.warning("Cache lookup error: %s", exc)

        elapsed_ms = (time.perf_counter() - t0) * 1_000
        logger.info("Cache MISS — query=%r", query)
        return CacheResult(
            hit=False,
            answer=None,
            similarity=0.0,
            latency_ms=elapsed_ms,
            source="LLM GENERATED",
        )

    def store(self, query: str, answer: str, ttl: int | None = None) -> None:
        """Embed and store a query-answer pair with an optional TTL (seconds)."""
        try:
            embedding = self._embed(query)
            point_id = str(uuid.uuid4())
            
            self._client.upsert(
                collection_name=CACHE_COLLECTION,
                points=[{
                    "id": point_id,
                    "vector": embedding,
                    "payload": {
                        "query": query,
                        "answer": answer,
                        "created_at": time.time(),
                    },
                }],
            )
            logger.info("Stored in cache — query=%r", query)
        except Exception as exc:
            logger.warning("Cache store error (non-fatal): %s", exc)

    def flush(self) -> int:
        """Delete all cache entries. Returns number of points deleted."""
        try:
            info = self._client.get_collection(CACHE_COLLECTION)
            count = info.points_count or 0
            self._client.delete_collection(CACHE_COLLECTION)
            self._ensure_cache_collection()
            return count
        except Exception as exc:
            logger.warning("Cache flush error: %s", exc)
            return 0

    def stats(self) -> dict:
        """Return basic cache stats for the /health endpoint."""
        try:
            info = self._client.get_collection(CACHE_COLLECTION)
            return {"cached_entries": info.points_count or 0, "threshold": self.threshold}
        except Exception:
            return {"cached_entries": 0, "threshold": self.threshold}


# Module-level singleton — imported by main.py and tasks
semantic_cache = SemanticCache()