from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    openai_api_key: str = ""
    openai_base_url: str | None = None
    openai_model: str = "gpt-4o-mini"
    max_chunk_chars: int = 6000
    chunk_parallel_workers: int = 4
    ai_cache_enabled: bool = True
    ai_cache_size: int = 20000


settings = Settings()
