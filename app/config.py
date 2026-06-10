from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    openai_api_key: str = ""
    openai_base_url: str | None = None
    openai_model: str = "gpt-4o-mini"
    # Number of automatic retries the OpenAI SDK performs on transient errors
    # (HTTP 429 rate limits, request timeouts, connection errors, 5xx). The SDK
    # applies exponential backoff and honors any Retry-After header. The default
    # of 2 is too low for very large documents that sustain heavy request load,
    # where per-minute rate limits would otherwise fail the whole job.
    openai_max_retries: int = 8
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
    # --- Resilience / never-fail processing ---
    # Cap for exponential backoff between retries on transient API errors.
    ai_retry_max_backoff_seconds: int = 60
    # Max attempts for *unclassified* errors before giving up. Known-transient
    # errors (rate limits, timeouts, connection drops, 5xx) retry indefinitely
    # so large jobs are never lost; only unknown errors are bounded to avoid
    # spinning forever on a genuine bug.
    ai_unknown_error_max_attempts: int = 6
    # When a daily request limit is hit, pause and auto-resume after the quota
    # resets (assumed at the next UTC midnight). Applies to every file format.
    ai_daily_limit_pause_enabled: bool = True
    # If a single unit hits a non-retryable error (e.g. content rejected), keep
    # the original text and continue instead of failing the whole document.
    ai_isolate_fatal_units: bool = True


settings = Settings()
