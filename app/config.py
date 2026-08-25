from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT_DIR = Path(__file__).resolve().parents[1]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    openai_api_key: str = ""
    openai_model: str = "gpt-4o-mini"
    openai_base_url: str = ""
    temperature: float = 0.0

    diwan_api_key: str = ""
    cors_origins: str = "http://localhost:8000"
    rate_limit_per_minute: int = 30
    max_query_length: int = 4000
    max_upload_bytes: int = 2_097_152
    max_extra_text_chars: int = 24_000
    enable_safety_filter: bool = True

    index_dir: Path = Field(default=ROOT_DIR / "data" / "index")
    database_path: Path = Field(default=ROOT_DIR / "data" / "diwan.db")
    knowledge_dir: Path = Field(default=ROOT_DIR / "data" / "knowledge")

    chunk_size: int = 900
    chunk_overlap: int = 140
    top_k: int = 6
    retrieval_threshold: float = 0.06

    input_cost_per_1m: float = 0.15
    output_cost_per_1m: float = 0.60

    log_level: str = "INFO"

    @property
    def demo_mode(self) -> bool:
        key = self.openai_api_key.strip()
        return not key or key.startswith("sk-your-key")

    @property
    def allowed_origins(self) -> list[str]:
        raw = [part.strip() for part in self.cors_origins.split(",") if part.strip()]
        return raw or ["http://localhost:8000"]


@lru_cache
def get_settings() -> Settings:
    return Settings()
