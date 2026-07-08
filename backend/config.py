"""
config.py — All application settings loaded from environment variables.
Centralises configuration so every module imports from one place.
Validates all settings on startup.
"""
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
from functools import lru_cache


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file = ".env",
        env_file_encoding = "utf-8",
        extra = "ignore",
        env_prefix = "",
    )

    # OpenAI
    openai_api_key: str = Field(..., description="OpenAI API key (required)")
    
    @field_validator("openai_api_key")
    @classmethod
    def validate_openai_key(cls, v: str) -> str:
        if not v or v == "sk-placeholder" or v.startswith("sk-or-v1-"):
            raise ValueError("OPENAI_API_KEY must be a valid OpenAI API key, not a placeholder")
        if not v.startswith("sk-"):
            raise ValueError("OPENAI_API_KEY must start with 'sk-'")
        return v

    # Redis (message broker + result backend + semantic cache)
    redis_url: str = Field(default="redis://localhost:6379/0", description="Redis connection URL")
    
    @field_validator("redis_url")
    @classmethod
    def validate_redis_url(cls, v: str) -> str:
        if not v.startswith("redis://"):
            raise ValueError("REDIS_URL must start with 'redis://'")
        return v

    # Qdrant vector store
    qdrant_url: str = Field(default="http://localhost:6333", description="Qdrant REST URL")
    
    @field_validator("qdrant_url")
    @classmethod
    def validate_qdrant_url(cls, v: str) -> str:
        if not v.startswith(("http://", "https://")):
            raise ValueError("QDRANT_URL must be a valid HTTP URL")
        return v

    qdrant_collection: str = Field(default="enterprise_docs", description="Collection name")

    # RAG tuning
    similarity_threshold: float = Field(default=0.90, ge=0.0, le=1.0, description="Cache hit threshold")
    chunk_size: int = Field(default=512, gt=0, le=8192, description="Characters per chunk")
    chunk_overlap: int = Field(default=64, ge=0, description="Overlap between chunks")
    top_k_chunks: int = Field(default=4, gt=0, le=100, description="Chunks retrieved per query")

    # Cache TTL (seconds)
    cache_ttl: int = Field(default=3600, gt=0, description="Cache entry TTL in seconds")
    
    # File upload limits
    max_file_size_mb: int = Field(default=50, gt=0, description="Max upload file size in MB")
    
    # Rate limiting
    rate_limit_requests: int = Field(default=100, gt=0, description="Requests per minute per user")
    
    # API settings
    api_version: str = Field(default="v1", description="API version prefix")


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()


# Convenience singleton used throughout the app
settings = get_settings()