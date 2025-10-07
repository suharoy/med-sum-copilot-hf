from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field

class Settings(BaseSettings):
    STORAGE_DIR: str = Field(default="data")
    EMBED_PROVIDER: str = Field(default="hf")
    GENERATION_PROVIDER: str = Field(default="hf")
    HF_EMBED_MODEL: str = Field(default="sentence-transformers/all-MiniLM-L6-v2")
    HF_SUMMARIZER_MODEL: str = Field(default="facebook/bart-large-cnn")
    model_config = SettingsConfigDict(env_file=".env", extra="ignore", case_sensitive=False)
settings = Settings()
