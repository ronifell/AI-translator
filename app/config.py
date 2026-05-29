from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    openai_api_key: str = ""
    openai_base_url: str | None = None
    openai_model: str = "gpt-4o-mini"
    max_chunk_chars: int = 6000
    # Maximum number of AI requests in flight at once across the whole document
    # (covers all texts, titles, chunks, qumran sections, comentarios, etc.).
    # Tune this to your OpenAI tier RPM/TPM limits. Default is conservative.
    ai_max_concurrency: int = 12
    # Kept for backward compatibility with any external code; the asyncio path
    # uses ai_max_concurrency for all parallelism (item-level + chunk-level).
    chunk_parallel_workers: int = 4
    ai_cache_enabled: bool = True
    ai_cache_size: int = 20000


settings = Settings()
