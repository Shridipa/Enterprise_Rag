"""
rag/vectorstore.py — Qdrant vector store wrapper.
Handles collection creation and provides a shared vectorstore instance.
"""
from functools import lru_cache
import hashlib
import inspect
import logging
import os

from dotenv import load_dotenv
from langchain_core.embeddings import Embeddings
from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, HnswConfigDiff
import numpy as np

from ..config import settings

logger = logging.getLogger(__name__)

# ── Embedding model ───────────────────────────────────────────────────────────
EMBEDDING_DIM = 1536  # text-embedding-3-small has 1536 dims


class LocalFallbackEmbeddings(Embeddings):
    """Deterministic embedding fallback used when OpenAI is unavailable."""

    def __init__(self, dim: int = EMBEDDING_DIM):
        self.dim = dim

    @property
    def embedding_dim(self) -> int:
        return self.dim

    def _embed_text(self, text: str) -> list[float]:
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        ints = np.frombuffer(digest, dtype=np.uint8).astype(np.float32)
        if len(ints) < self.dim:
            ints = np.pad(ints, (0, self.dim - len(ints)))
        else:
            ints = ints[: self.dim]
        return (ints / 255.0).tolist()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed_text(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed_text(text)


def get_embeddings():
    """Create embeddings using OpenAI when available, otherwise a deterministic local fallback."""
    try:
        load_dotenv(override=True)
    except TypeError:
        try:
            load_dotenv()
        except Exception:
            pass
    except Exception:
        pass

    api_key = os.getenv("OPENAI_API_KEY") or settings.openai_api_key
    if not api_key or api_key in {"sk-placeholder", "sk-test-key"}:
        logger.warning("Using local embedding fallback because OpenAI API key is not configured for live calls.")
        return LocalFallbackEmbeddings()

    try:
        return OpenAIEmbeddings(
            model="text-embedding-3-small",
            openai_api_key=api_key,
            max_retries=3,
            request_timeout=60,
        )
    except Exception as exc:
        logger.warning("Falling back to local embeddings because OpenAI initialization failed: %s", exc)
        return LocalFallbackEmbeddings()


# ── Qdrant client (singleton via lru_cache) ───────────────────────────────────
@lru_cache(maxsize=1)
def get_qdrant_client() -> QdrantClient:
    client_kwargs = {
        "url": settings.qdrant_url,
        "timeout": 30,
    }
    if "prefer_gzip" in inspect.signature(QdrantClient.__init__).parameters:
        client_kwargs["prefer_gzip"] = True
    return QdrantClient(**client_kwargs)


def init_collection() -> None:
    """Create the Qdrant collection if it does not already exist."""
    client = get_qdrant_client()
    existing = {c.name for c in client.get_collections().collections}
    if settings.qdrant_collection not in existing:
        client.create_collection(
            collection_name=settings.qdrant_collection,
            vectors_config=VectorParams(
                size=EMBEDDING_DIM,
                distance=Distance.COSINE,
            ),
            hnsw_config=HnswConfigDiff(
                m=16,  # Number of connections per layer
                ef_construct=100,  # Construction quality
            ),
        )


def get_vectorstore() -> QdrantVectorStore:
    """Return a LangChain-compatible Qdrant vector store, creating the collection if needed."""
    init_collection()
    return QdrantVectorStore(
        client=get_qdrant_client(),
        collection_name=settings.qdrant_collection,
        embedding=get_embeddings(),
    )


def collection_info() -> dict:
    """Return basic stats about the Qdrant collection (for the /health endpoint)."""
    try:
        client = get_qdrant_client()
        info = client.get_collection(settings.qdrant_collection)
        return {
            "status": str(info.status),
            "vectors_count": getattr(info, "vectors_count", None),
            "points_count": getattr(info, "points_count", None),
        }
    except Exception as exc:
        return {"error": str(exc)}