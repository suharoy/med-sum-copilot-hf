from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field

class Settings(BaseSettings):
    STORAGE_DIR: str = Field(default="data")

    # Providers stay local (HF only)
    EMBED_PROVIDER: str = Field(default="hf")
    GENERATION_PROVIDER: str = Field(default="hf")

    # Models (CPU-friendly)
    HF_EMBED_MODEL: str = Field(default="sentence-transformers/all-MiniLM-L6-v2")
    HF_SUMMARIZER_MODEL: str = Field(default="t5-base")  # tighter than BART

    # Retrieval knobs
    BM25_WEIGHT: float = 0.5
    COSINE_WEIGHT: float = 0.5
    MAX_SENT_PER_CHUNK: int = 12
    MMR_LAMBDA: float = 0.6           # 0..1 (higher = more relevance, lower = more diversity)
    MAX_SENTENCES_FOR_ANSWER: int = 6

    model_config = SettingsConfigDict(env_file=".env", extra="ignore", case_sensitive=False)

settings = Settings()