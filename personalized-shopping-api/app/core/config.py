"""
Application configuration using Pydantic Settings

Environment-based configuration with validation
"""

from typing import List, Optional, Literal, Any
from pydantic import Field, field_validator, AnyHttpUrl
from pydantic_settings import BaseSettings, SettingsConfigDict
from pathlib import Path

class Settings(BaseSettings):
    """
    Application settings loaded from environment variables

    Environment files:
    - .env (default)
    - .env.test (testing)
    - .env.production (production)
    """

    # ===== APPLICATION =====
    PROJECT_NAME: str = "Personalized Shopping Assistant API"
    VERSION: str = "1.0.0"
    ENVIRONMENT: Literal["development", "staging", "production"] = "development"
    DEBUG: bool = False
    API_V1_PREFIX: str = "/api/v1"

    # ===== SERVER =====
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    RELOAD: bool = True
    WORKERS: int = 1

    # ===== CORS =====
    BACKEND_CORS_ORIGINS: List[str] = Field(
        default=["http://localhost:3000", "http://localhost:8000"]
    )

    @field_validator("BACKEND_CORS_ORIGINS", mode="before")
    @classmethod
    def assemble_cors_origins(cls, v: Any) -> List[str]:
        if isinstance(v, str):
            return [i.strip() for i in v.split(",")]
        return v

    # ===== SECURITY =====
    SECRET_KEY: str = Field(default="dev-secret-key-change-in-production")
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60 * 24 * 7
    ALGORITHM: str = "HS256"

    # ===== DATABASE (Optional) =====
    DATABASE_URL: Optional[str] = None
    DB_POOL_SIZE: int = 5
    DB_MAX_OVERFLOW: int = 10
    DB_ECHO: bool = False

    # ===== REDIS (Optional) =====
    REDIS_URL: Optional[str] = None
    CACHE_TTL_SECONDS: int = 3600
    ENABLE_CACHE: bool = True

    # ===== DATA PATHS =====
    DATA_DIR: Path = Path("./data")
    PURCHASE_DATA_PATH: Path = Path("./data/raw/customer_purchase_data.csv")
    REVIEW_DATA_PATH: Path = Path("./data/raw/customer_reviews_data.csv")
    EMBEDDINGS_DIR: Path = Path("./data/embeddings")
    VECTOR_INDEX_PATH: Path = Path("./data/embeddings/customer_index.faiss")

    # ===== LLM CONFIGURATION =====
    OLLAMA_BASE_URL: str = "http://localhost:11434"

    # Model aliases for compatibility
    @property
    def profiling_model(self) -> str:
        return self.PROFILING_MODEL

    @property
    def sentiment_model(self) -> str:
        return self.SENTIMENT_MODEL

    @property
    def recommendation_model(self) -> str:
        return self.RECOMMENDATION_MODEL

    @property
    def response_model(self) -> str:
        return self.RESPONSE_MODEL

    @property
    def ollama_base_url(self) -> str:
        return self.OLLAMA_BASE_URL

    @property
    def embedding_model(self) -> str:
        return self.EMBEDDING_MODEL

    @property
    def normalize_embeddings(self) -> bool:
        return self.NORMALIZE_EMBEDDINGS

    @property
    def temperature(self) -> float:
        return self.TEMPERATURE

    @property
    def max_tokens(self) -> int:
        return self.MAX_TOKENS

    @property
    def request_timeout_seconds(self) -> int:
        return self.REQUEST_TIMEOUT

    @property
    def min_reviews_for_inclusion(self) -> int:
        return self.MIN_REVIEWS_FOR_INCLUSION

    PROFILING_MODEL: str = "llama3.2:3b"
    SENTIMENT_MODEL: str = "llama3.1:8b"
    RECOMMENDATION_MODEL: str = "llama3.1:8b"
    RESPONSE_MODEL: str = "llama3.1:8b"

    MAX_TOKENS: int = 2048
    TEMPERATURE: float = 0.1
    REQUEST_TIMEOUT: int = 30

    # ===== EMBEDDINGS =====
    EMBEDDING_MODEL: str = "BAAI/bge-base-en-v1.5"
    EMBEDDING_DIMENSION: int = 768
    NORMALIZE_EMBEDDINGS: bool = True

    # ===== VECTOR DATABASE =====
    VECTOR_DB_TYPE: Literal["faiss", "chroma"] = "faiss"
    SIMILARITY_THRESHOLD: float = 0.75
    SIMILARITY_TOP_K: int = 20
    FAISS_INDEX_TYPE: Literal["flat", "ivf", "hnsw"] = "ivf"
    FAISS_NLIST: int = 100
    FAISS_NPROBE: int = 10

    # ===== AGENT CONFIGURATION =====
    MIN_PURCHASES_FOR_PROFILE: int = 1
    MIN_REVIEWS_FOR_INCLUSION: int = 1
    SENTIMENT_THRESHOLD: float = 0.6
    RECOMMENDATION_TOP_N: int = 5
    COLLABORATIVE_WEIGHT: float = 0.6
    CATEGORY_AFFINITY_WEIGHT: float = 0.4

    # ===== PERFORMANCE =====
    MAX_WORKERS: int = 4
    ENABLE_MODEL_CASCADING: bool = True
    ENABLE_SEMANTIC_CACHING: bool = True
    CACHE_SIMILARITY_THRESHOLD: float = 0.95

    # ===== OBSERVABILITY =====
    LOG_LEVEL: str = "INFO"
    LOG_FORMAT: str = "json"
    ENABLE_METRICS: bool = True
    ENABLE_TRACING: bool = False

    # LangSmith (Legacy - consider migrating to LangFuse)
    LANGSMITH_API_KEY: Optional[str] = None
    LANGSMITH_PROJECT: str = "shopping-assistant-api"

    # LangFuse (Recommended for LLM observability)
    LANGFUSE_ENABLED: bool = False
    LANGFUSE_PUBLIC_KEY: Optional[str] = None
    LANGFUSE_SECRET_KEY: Optional[str] = None
    LANGFUSE_HOST: str = "https://cloud.langfuse.com"
    LANGFUSE_RELEASE: Optional[str] = None
    LANGFUSE_ENVIRONMENT: str = "development"
    LANGFUSE_DEBUG: bool = False

    # ===== RATE LIMITING =====
    RATE_LIMIT_ENABLED: bool = True
    RATE_LIMIT_PER_MINUTE: int = 100

    # ===== TESTING =====
    TESTING: bool = False

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore",
    )

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Create directories if they don't exist
        self.DATA_DIR.mkdir(parents=True, exist_ok=True)
        self.EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True)

# Singleton instance
settings = Settings()
