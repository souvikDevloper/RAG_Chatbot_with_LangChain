from pydantic import BaseModel
from pathlib import Path
import os

class Settings(BaseModel):
    data_dir: Path = Path(os.getenv("DATA_DIR", "./data"))
    use_queue: bool = os.getenv("USE_QUEUE", "1") == "1"
    redis_url: str = os.getenv("REDIS_URL", "redis://localhost:6379/0")

    default_llm_base_url: str = os.getenv("DEFAULT_LLM_BASE_URL", "https://api.openai.com/v1")
    default_llm_model: str = os.getenv("DEFAULT_LLM_MODEL", "gpt-4o-mini")
    default_embed_base_url: str = os.getenv("DEFAULT_EMBED_BASE_URL", "https://api.openai.com/v1")
    default_embed_model: str = os.getenv("DEFAULT_EMBED_MODEL", "text-embedding-3-small")

    default_top_k: int = int(os.getenv("DEFAULT_TOP_K", "12"))

    @property
    def sqlite_path(self) -> Path:
        return self.data_dir / "sqlite" / "app.db"

    @property
    def chroma_path(self) -> Path:
        return self.data_dir / "chroma"

    @property
    def blob_path(self) -> Path:
        return self.data_dir / "blobs"

settings = Settings()
