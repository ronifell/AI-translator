from __future__ import annotations

import logging
from collections import OrderedDict
from threading import Lock
from typing import Callable

from openai import OpenAI

from app.config import settings

logger = logging.getLogger(__name__)
_cache_lock = Lock()
_cache: OrderedDict[str, str] = OrderedDict()

SYSTEM_PROMPT = """You are a linguistic correction tool for sensitive religious and scholarly texts.

Rules (strict):
- Only fix spelling, grammar, and syntax.
- Do NOT rewrite, paraphrase, or reinterpret meaning.
- Preserve meaning and sacred phrasing exactly.
- Keep structure, line breaks, and formatting unchanged unless a clear typo requires a minimal fix.
- If a short passage is clearly in the wrong language for its context, translate it minimally to the target language. Keep such changes under 1% of the total text unless the wrong-language segment is longer.

Return ONLY the corrected text with no quotes, no preamble, and no explanation."""


def _client() -> OpenAI:
    kwargs: dict = {"api_key": settings.openai_api_key or None}
    if settings.openai_base_url:
        kwargs["base_url"] = settings.openai_base_url
    return OpenAI(**kwargs)


def _cache_get(key: str) -> str | None:
    if not settings.ai_cache_enabled:
        return None
    with _cache_lock:
        val = _cache.get(key)
        if val is None:
            return None
        _cache.move_to_end(key)
        return val


def _cache_set(key: str, value: str) -> None:
    if not settings.ai_cache_enabled:
        return
    with _cache_lock:
        _cache[key] = value
        _cache.move_to_end(key)
        while len(_cache) > settings.ai_cache_size:
            _cache.popitem(last=False)


def correct_text(
    text: str,
    *,
    target_language_hint: str | None = None,
    on_progress: Callable[[str], None] | None = None,
) -> str:
    if not text.strip():
        return text
    if not settings.openai_api_key:
        raise ValueError("OPENAI_API_KEY is not set. Add it to backend/.env")

    lang = target_language_hint or "the document's primary language"
    cache_key = f"correct|{settings.openai_model}|{lang}|{text}"
    cached = _cache_get(cache_key)
    if cached is not None:
        if on_progress:
            on_progress("ai_cache_hit")
        return cached

    user_content = f"""Target language context (for wrong-language fixes): {lang}

Text to correct:
\"\"\"
{text}
\"\"\"

Return ONLY the corrected text."""

    if on_progress:
        on_progress("ai_request")

    resp = _client().chat.completions.create(
        model=settings.openai_model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ],
        temperature=0.1,
    )
    out = (resp.choices[0].message.content or "").strip()
    if on_progress:
        on_progress("ai_done")
    final = out if out else text
    _cache_set(cache_key, final)
    return final


def generate_title(text: str, *, on_progress: Callable[[str], None] | None = None) -> str:
    if not settings.openai_api_key:
        raise ValueError("OPENAI_API_KEY is not set. Add it to backend/.env")
    cache_key = f"title|{settings.openai_model}|{text[:8000]}"
    cached = _cache_get(cache_key)
    if cached is not None:
        if on_progress:
            on_progress("ai_cache_hit")
        return cached
    prompt = f"""Based on the commentary or excerpt below, produce ONE concise section title (max ~12 words).
Use the same language as the text. No quotes. No trailing period unless part of an abbreviation.

Text:
\"\"\"
{text[:8000]}
\"\"\"

Return ONLY the title line."""

    if on_progress:
        on_progress("ai_title")

    resp = _client().chat.completions.create(
        model=settings.openai_model,
        messages=[
            {
                "role": "system",
                "content": "You write short, neutral scholarly titles. Output only the title.",
            },
            {"role": "user", "content": prompt},
        ],
        temperature=0.2,
    )
    title = (resp.choices[0].message.content or "").strip().split("\n")[0].strip()
    if on_progress:
        on_progress("ai_title_done")
    final = title if title else "Untitled"
    _cache_set(cache_key, final)
    return final
