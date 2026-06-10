from __future__ import annotations

import asyncio
import copy
import hashlib
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Awaitable, Callable

import httpx
from openai import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    AuthenticationError,
    BadRequestError,
    InternalServerError,
    PermissionDeniedError,
    RateLimitError,
)

from app.services import ai_service
from app.config import settings
from app.utils.diff_tracker import DiffTracker

logger = logging.getLogger(__name__)


@dataclass
class TraverseState:
    apocryphal_zone: bool = False


@dataclass
class ProcessOptions:
    target_language: str = "pt-BR"
    treat_biblical_texto_as_google: bool = False


@dataclass
class ProcessContext:
    """Per-document caches and per-call concurrency primitives.

    Caches (`reviewed_text_cache`, `generated_title_cache`) survive across
    multiple `process_json_document` calls so identical texts are never
    sent to the AI more than once.

    Concurrency primitives (`ai_semaphore`, futures dicts) are bound to a
    specific event loop and are reset/created at the start of each
    `process_json_document` call.
    """

    reviewed_text_cache: dict[str, str] = field(default_factory=dict)
    generated_title_cache: dict[str, str] = field(default_factory=dict)
    reviewed_text_futures: dict[str, asyncio.Future] = field(default_factory=dict)
    generated_title_futures: dict[str, asyncio.Future] = field(default_factory=dict)
    ai_semaphore: asyncio.Semaphore | None = None
    # Optional durable, content-addressed result store. When provided, every
    # successfully corrected chunk / generated title is persisted immediately so
    # that a restart or retry re-uses it instead of calling the API again. This
    # is what makes resume work for *any* JSON structure (not just `livros`).
    durable_get: Callable[[str], str | None] | None = None
    durable_put: Callable[[str, str], None] | None = None


ProgressCb = Callable[[str, str | None], None]
ShouldCancelCb = Callable[[], bool]


class ProcessingCancelledError(RuntimeError):
    pass


def _raise_if_cancelled(should_cancel: ShouldCancelCb | None) -> None:
    if should_cancel and should_cancel():
        raise ProcessingCancelledError("Processing cancelled by user")


def _is_daily_rate_limit_error(exc: BaseException) -> bool:
    """True when a rate-limit error specifically concerns the *daily* quota."""
    msg = str(exc).lower()
    has_rate_limit_signal = any(
        s in msg
        for s in ("rate limit", "too many requests", "error code: 429", "status code 429")
    )
    has_daily_signal = any(
        s in msg
        for s in ("requests per day", "rpd", "max_requests_per_1_day", "per_1_day", " per day")
    )
    return has_rate_limit_signal and has_daily_signal


def _classify_error(exc: BaseException) -> str:
    """Bucket an exception so the resilience loop knows how to react.

    Returns one of: "auth" (stop the job, config problem), "daily" (pause until
    quota reset), "transient" (retry indefinitely with backoff), "fatal" (skip
    this single unit), or "unknown" (retry a bounded number of times).
    """
    if isinstance(exc, (AuthenticationError, PermissionDeniedError)):
        return "auth"
    # Configuration problems (e.g. missing OPENAI_API_KEY) surface as ValueError
    # from ai_service; retrying cannot help, so stop the job with a clear message.
    if isinstance(exc, ValueError):
        return "auth"
    if _is_daily_rate_limit_error(exc):
        return "daily"
    if isinstance(exc, (RateLimitError, APITimeoutError, APIConnectionError, InternalServerError)):
        return "transient"
    if isinstance(exc, APIStatusError):
        code = getattr(exc, "status_code", None)
        if isinstance(code, int) and (code == 429 or code >= 500):
            return "transient"
        return "fatal"
    if isinstance(exc, (httpx.TimeoutException, httpx.TransportError)):
        return "transient"
    if isinstance(exc, BadRequestError):
        return "fatal"
    return "unknown"


def _extract_retry_hint_seconds(exc: BaseException) -> int | None:
    msg = str(exc)
    m = re.search(r"try again in\s+(\d+(?:\.\d+)?)s", msg, flags=re.I)
    if m:
        return max(1, int(float(m.group(1))))
    m = re.search(r"try again in\s+(\d+)m(\d+)s", msg, flags=re.I)
    if m:
        return int(m.group(1)) * 60 + int(m.group(2))
    return None


def _next_utc_midnight_iso() -> str:
    now = datetime.now(timezone.utc)
    next_day = (now + timedelta(days=1)).date()
    return datetime.combine(next_day, datetime.min.time(), tzinfo=timezone.utc).isoformat()


def _seconds_until_next_utc_midnight() -> int:
    now = datetime.now(timezone.utc)
    next_day = (now + timedelta(days=1)).date()
    midnight = datetime.combine(next_day, datetime.min.time(), tzinfo=timezone.utc)
    return max(int((midnight - now).total_seconds()), 0)


async def _sleep_with_cancel(seconds: float, should_cancel: ShouldCancelCb | None) -> None:
    """Sleep in short slices so cancellation is honored promptly."""
    remaining = seconds
    while remaining > 0:
        _raise_if_cancelled(should_cancel)
        step = min(remaining, 5.0)
        await asyncio.sleep(step)
        remaining -= step
    _raise_if_cancelled(should_cancel)


async def _wait_until_next_utc_midnight(should_cancel: ShouldCancelCb | None) -> None:
    while True:
        _raise_if_cancelled(should_cancel)
        remaining = _seconds_until_next_utc_midnight()
        if remaining <= 0:
            return
        await _sleep_with_cancel(min(remaining, 60), should_cancel)


def _durable_key(kind: str, model: str, lang: str, text: str) -> str:
    """Stable content-addressed key for the durable result store."""
    h = hashlib.sha256(f"{kind}\x00{model}\x00{lang}\x00{text}".encode("utf-8")).hexdigest()
    return f"{kind}:{h}"


# Sentinel returned by the resilience wrapper when a unit could not be processed
# and must keep its original text (fatal, non-retryable error on that unit).
_SKIP_UNIT = object()


async def _call_ai_resilient(
    coro_factory: Callable[[], Awaitable[str]],
    *,
    path: str | None,
    progress: ProgressCb | None,
    should_cancel: ShouldCancelCb | None,
) -> Any:
    """Run an AI call, riding through rate limits and transient failures.

    - daily limit  -> pause until UTC midnight, then retry (never fails)
    - transient    -> exponential backoff, retry indefinitely (never fails)
    - unknown      -> bounded retries, then re-raise
    - auth         -> re-raise immediately (config problem; whole job stops)
    - fatal/unit   -> return `_SKIP_UNIT` so the caller keeps the original text
    """
    attempt = 0
    backoff = 1.0
    max_backoff = float(max(1, settings.ai_retry_max_backoff_seconds))
    while True:
        _raise_if_cancelled(should_cancel)
        try:
            return await coro_factory()
        except (ProcessingCancelledError, asyncio.CancelledError, KeyboardInterrupt, SystemExit):
            raise
        except BaseException as exc:  # noqa: BLE001 - classified below
            kind = _classify_error(exc)
            if kind == "auth":
                raise
            if kind == "daily":
                if not settings.ai_daily_limit_pause_enabled:
                    raise
                logger.warning("Daily rate limit hit; pausing until UTC midnight.")
                if progress:
                    progress("paused", _next_utc_midnight_iso())
                await _wait_until_next_utc_midnight(should_cancel)
                if progress:
                    progress("resumed", path)
                continue
            if kind == "fatal":
                if not settings.ai_isolate_fatal_units:
                    raise
                logger.warning("Skipping unit %s after non-retryable error: %s", path, exc)
                if progress:
                    progress("skipped", path)
                return _SKIP_UNIT
            # transient or unknown
            attempt += 1
            if kind == "unknown" and attempt >= settings.ai_unknown_error_max_attempts:
                raise
            hint = _extract_retry_hint_seconds(exc)
            delay = float(hint) if hint is not None else min(backoff, max_backoff)
            logger.info(
                "Transient AI error on %s (attempt %d, retrying in %.0fs): %s",
                path, attempt, delay, exc,
            )
            await _sleep_with_cancel(delay, should_cancel)
            backoff = min(backoff * 2, max_backoff)
            continue


def _path_join(base: str, key: str | int) -> str:
    if isinstance(key, int):
        return f"{base}[{key}]" if base else f"[{key}]"
    return f"{base}.{key}" if base else key


def _normalize_apocryphal_id(val: Any) -> bool:
    if val is None:
        return False
    s = str(val).strip().lower().replace(" ", "")
    return s in ("1.esra", "1esra", "i.esra")


def _parent_suggests_google(parent: dict[str, Any], options: ProcessOptions) -> bool:
    if options.treat_biblical_texto_as_google:
        return True
    bool_keys = (
        "traduzido_google",
        "traducao_google",
        "google_translate",
        "via_google",
        "googleTranslated",
    )
    for k in bool_keys:
        if parent.get(k) in (True, "true", "yes", 1, "1", "sim"):
            return True
    if str(parent.get("fonte_traducao", "")).lower() == "google":
        return True
    if str(parent.get("fonte", "")).lower() == "google":
        return True
    if str(parent.get("origem_traducao", "")).lower() == "google":
        return True
    return False


def _should_review_biblical_texto(
    parent: dict[str, Any],
    state: TraverseState,
    options: ProcessOptions,
) -> bool:
    if state.apocryphal_zone:
        return True
    return _parent_suggests_google(parent, options)


def _chunk_string(text: str, max_chars: int) -> list[str]:
    if len(text) <= max_chars:
        return [text]
    parts: list[str] = []
    start = 0
    while start < len(text):
        end = min(start + max_chars, len(text))
        if end < len(text):
            split_at = text.rfind("\n\n", start, end)
            if split_at == -1 or split_at <= start:
                split_at = text.rfind(" ", start, end)
            if split_at > start:
                end = split_at + 1
        parts.append(text[start:end])
        start = end
    return parts


async def _correct_chunk(
    chunk: str,
    options: ProcessOptions,
    ctx: ProcessContext,
    progress_subpath: str | None,
    progress: ProgressCb | None,
    should_cancel: ShouldCancelCb | None,
) -> str:
    """Correct one chunk; deduplicates concurrent calls for identical chunks."""
    _raise_if_cancelled(should_cancel)

    cached = ctx.reviewed_text_cache.get(chunk)
    if cached is not None:
        return cached

    dkey: str | None = None
    if ctx.durable_get is not None or ctx.durable_put is not None:
        dkey = _durable_key("correct", settings.openai_model, options.target_language, chunk)
    if ctx.durable_get is not None and dkey is not None:
        dval = ctx.durable_get(dkey)
        if dval is not None:
            ctx.reviewed_text_cache[chunk] = dval
            return dval

    pending = ctx.reviewed_text_futures.get(chunk)
    if pending is not None:
        return await pending

    fut: asyncio.Future = asyncio.get_running_loop().create_future()
    ctx.reviewed_text_futures[chunk] = fut
    try:
        result = await _call_ai_resilient(
            lambda: ai_service.correct_text(
                chunk,
                target_language_hint=options.target_language,
                on_progress=(
                    (lambda _ev, sp=progress_subpath: progress("ai", sp))
                    if progress
                    else None
                ),
                semaphore=ctx.ai_semaphore,
            ),
            path=progress_subpath,
            progress=progress,
            should_cancel=should_cancel,
        )
        corrected = chunk if result is _SKIP_UNIT else result
        ctx.reviewed_text_cache[chunk] = corrected
        if ctx.durable_put is not None and dkey is not None and result is not _SKIP_UNIT:
            ctx.durable_put(dkey, corrected)
        if not fut.done():
            fut.set_result(corrected)
        return corrected
    except BaseException as e:
        if not fut.done():
            fut.set_exception(e)
        ctx.reviewed_text_futures.pop(chunk, None)
        raise


async def _review_body_text(
    original: str,
    path: str,
    tracker: DiffTracker,
    options: ProcessOptions,
    ctx: ProcessContext,
    progress: ProgressCb | None,
    should_cancel: ShouldCancelCb | None,
) -> str:
    _raise_if_cancelled(should_cancel)

    chunks = _chunk_string(original, settings.max_chunk_chars)
    n = len(chunks)

    full_cached = ctx.reviewed_text_cache.get(original)
    if full_cached is not None:
        if progress:
            for j in range(n):
                _raise_if_cancelled(should_cancel)
                subpath = f"{path}#part{j}" if n > 1 else path
                progress("memo_hit", subpath)
        tracker.record(path, original, full_cached)
        return full_cached

    if progress:
        for i in range(n):
            _raise_if_cancelled(should_cancel)
            subpath = f"{path}#part{i}" if n > 1 else path
            progress("chunk", subpath)

    coros = [
        _correct_chunk(
            ch,
            options,
            ctx,
            f"{path}#part{i}" if n > 1 else path,
            progress,
            should_cancel,
        )
        for i, ch in enumerate(chunks)
    ]
    out = await asyncio.gather(*coros)
    merged = "".join(out)
    if n > 1:
        ctx.reviewed_text_cache[original] = merged
    tracker.record(path, original, merged)
    return merged


async def _generate_title_for(
    body: str,
    titulo_path: str,
    options: ProcessOptions,
    ctx: ProcessContext,
    progress: ProgressCb | None,
    should_cancel: ShouldCancelCb | None,
) -> str:
    _raise_if_cancelled(should_cancel)

    cached = ctx.generated_title_cache.get(body)
    if cached is not None:
        if progress:
            progress("memo_hit", titulo_path)
        return cached

    dkey: str | None = None
    if ctx.durable_get is not None or ctx.durable_put is not None:
        dkey = _durable_key("title", settings.openai_model, "", body[:8000])
    if ctx.durable_get is not None and dkey is not None:
        dval = ctx.durable_get(dkey)
        if dval is not None:
            ctx.generated_title_cache[body] = dval
            if progress:
                progress("memo_hit", titulo_path)
            return dval

    pending = ctx.generated_title_futures.get(body)
    if pending is not None:
        return await pending

    fut: asyncio.Future = asyncio.get_running_loop().create_future()
    ctx.generated_title_futures[body] = fut
    try:
        if progress:
            progress("title_gen", titulo_path)
        result = await _call_ai_resilient(
            lambda: ai_service.generate_title(
                body,
                on_progress=(
                    (lambda _ev, sp=titulo_path: progress("ai", sp))
                    if progress
                    else None
                ),
                semaphore=ctx.ai_semaphore,
            ),
            path=titulo_path,
            progress=progress,
            should_cancel=should_cancel,
        )
        title = "" if result is _SKIP_UNIT else result
        ctx.generated_title_cache[body] = title
        if ctx.durable_put is not None and dkey is not None and result is not _SKIP_UNIT:
            ctx.durable_put(dkey, title)
        if not fut.done():
            fut.set_result(title)
        return title
    except BaseException as e:
        if not fut.done():
            fut.set_exception(e)
        ctx.generated_title_futures.pop(body, None)
        raise


async def _review_or_generate_title(
    item: dict[str, Any],
    titulo_path: str,
    tracker: DiffTracker,
    options: ProcessOptions,
    ctx: ProcessContext,
    progress: ProgressCb | None,
    should_cancel: ShouldCancelCb | None,
) -> None:
    _raise_if_cancelled(should_cancel)
    raw = item.get("titulo")
    body = item.get("texto") if isinstance(item.get("texto"), str) else ""

    if raw is None or (isinstance(raw, str) and not raw.strip()):
        if not (isinstance(body, str) and body.strip()):
            return
        title = await _generate_title_for(
            body, titulo_path, options, ctx, progress, should_cancel
        )
        item["titulo"] = title
        tracker.record(titulo_path, raw if isinstance(raw, str) else "", title)
        return

    if isinstance(raw, str) and raw.strip():
        item["titulo"] = await _review_body_text(
            raw, titulo_path, tracker, options, ctx, progress, should_cancel
        )


async def _walk_qumran(
    parent: Any,
    key: Any,
    path: str,
    tracker: DiffTracker,
    options: ProcessOptions,
    ctx: ProcessContext,
    progress: ProgressCb | None,
    should_cancel: ShouldCancelCb | None,
) -> None:
    """Process node at parent[key] in place, parallelizing across siblings."""
    _raise_if_cancelled(should_cancel)
    node = parent[key]

    if isinstance(node, dict):
        coros = []
        for k in list(node.keys()):
            v = node[k]
            cp = _path_join(path, k)
            if k == "texto_ingles":
                continue
            if isinstance(v, str):
                coros.append(
                    _qumran_set_string(node, k, v, cp, tracker, options, ctx, progress, should_cancel)
                )
            else:
                coros.append(
                    _walk_qumran(node, k, cp, tracker, options, ctx, progress, should_cancel)
                )
        if coros:
            await asyncio.gather(*coros)
        return

    if isinstance(node, list):
        coros = []
        for i in range(len(node)):
            item = node[i]
            cp = _path_join(path, i)
            if isinstance(item, str):
                coros.append(
                    _qumran_set_string(node, i, item, cp, tracker, options, ctx, progress, should_cancel)
                )
            else:
                coros.append(
                    _walk_qumran(node, i, cp, tracker, options, ctx, progress, should_cancel)
                )
        if coros:
            await asyncio.gather(*coros)
        return

    if isinstance(node, str):
        parent[key] = await _review_body_text(
            node, path, tracker, options, ctx, progress, should_cancel
        )


async def _qumran_set_string(
    container: Any,
    idx: Any,
    val: str,
    cur: str,
    tracker: DiffTracker,
    options: ProcessOptions,
    ctx: ProcessContext,
    progress: ProgressCb | None,
    should_cancel: ShouldCancelCb | None,
) -> None:
    container[idx] = await _review_body_text(
        val, cur, tracker, options, ctx, progress, should_cancel
    )


async def _set_corrected_in_place(
    node: dict[str, Any],
    key: str,
    val: str,
    cur: str,
    tracker: DiffTracker,
    options: ProcessOptions,
    ctx: ProcessContext,
    progress: ProgressCb | None,
    should_cancel: ShouldCancelCb | None,
) -> None:
    node[key] = await _review_body_text(
        val, cur, tracker, options, ctx, progress, should_cancel
    )


async def _process_one_comentario(
    item: Any,
    p: str,
    state: TraverseState,
    options: ProcessOptions,
    tracker: DiffTracker,
    ctx: ProcessContext,
    progress: ProgressCb | None,
    should_cancel: ShouldCancelCb | None,
) -> None:
    if not isinstance(item, dict):
        return

    # Body text first; title generation may depend on the corrected body.
    t = item.get("texto")
    if isinstance(t, str) and t.strip():
        item["texto"] = await _review_body_text(
            t, _path_join(p, "texto"), tracker, options, ctx, progress, should_cancel
        )

    await _review_or_generate_title(
        item, _path_join(p, "titulo"), tracker, options, ctx, progress, should_cancel
    )

    # Walk other keys in parallel; each child has its own state copy because
    # state mutations there should not affect siblings within this comentario.
    other_coros = []
    for k, v in list(item.items()):
        if k in ("texto", "titulo"):
            continue
        child_state = TraverseState(apocryphal_zone=state.apocryphal_zone)
        other_coros.append(
            _walk(v, _path_join(p, k), child_state, options, tracker, ctx, progress, should_cancel)
        )
    if other_coros:
        await asyncio.gather(*other_coros)


async def _process_comentarios(
    node: Any,
    path: str,
    state: TraverseState,
    options: ProcessOptions,
    tracker: DiffTracker,
    ctx: ProcessContext,
    progress: ProgressCb | None,
    should_cancel: ShouldCancelCb | None,
) -> None:
    if isinstance(node, list):
        coros = []
        for i, item in enumerate(node):
            _raise_if_cancelled(should_cancel)
            p = _path_join(path, i)
            coros.append(
                _process_one_comentario(
                    item, p, state, options, tracker, ctx, progress, should_cancel
                )
            )
        if coros:
            await asyncio.gather(*coros)
    elif isinstance(node, dict):
        await _walk(node, path, state, options, tracker, ctx, progress, should_cancel)


async def _walk(
    node: Any,
    path: str,
    state: TraverseState,
    options: ProcessOptions,
    tracker: DiffTracker,
    ctx: ProcessContext,
    progress: ProgressCb | None,
    should_cancel: ShouldCancelCb | None,
) -> None:
    _raise_if_cancelled(should_cancel)
    if isinstance(node, dict):
        if _normalize_apocryphal_id(node.get("__id__")):
            state.apocryphal_zone = True

        coros = []

        if "qumran" in node:
            qpath = _path_join(path, "qumran")
            coros.append(
                _walk_qumran(node, "qumran", qpath, tracker, options, ctx, progress, should_cancel)
            )

        if "comentarios" in node:
            coros.append(
                _process_comentarios(
                    node["comentarios"],
                    _path_join(path, "comentarios"),
                    state,
                    options,
                    tracker,
                    ctx,
                    progress,
                    should_cancel,
                )
            )

        for key in list(node.keys()):
            if key in ("qumran", "comentarios"):
                continue
            val = node[key]
            cur = _path_join(path, key)
            if key == "texto_ingles":
                continue
            if key == "texto" and isinstance(val, str):
                if _should_review_biblical_texto(node, state, options):
                    coros.append(
                        _set_corrected_in_place(
                            node, key, val, cur, tracker, options, ctx, progress, should_cancel
                        )
                    )
                continue
            if key == "titulo" and isinstance(val, str):
                coros.append(
                    _set_corrected_in_place(
                        node, key, val, cur, tracker, options, ctx, progress, should_cancel
                    )
                )
                continue
            coros.append(
                _walk(val, cur, state, options, tracker, ctx, progress, should_cancel)
            )

        if coros:
            await asyncio.gather(*coros)
        return

    if isinstance(node, list):
        if not node:
            return
        # Pre-compute apocryphal_zone for each item so list items can be
        # processed in parallel while preserving the "from `1. Esra` onward"
        # semantics that the original sequential walk produced.
        flags: list[bool] = []
        current = state.apocryphal_zone
        for item in node:
            flags.append(current)
            if isinstance(item, dict) and _normalize_apocryphal_id(item.get("__id__")):
                current = True

        coros = []
        for i, item in enumerate(node):
            child_state = TraverseState(apocryphal_zone=flags[i])
            coros.append(
                _walk(
                    item,
                    _path_join(path, i),
                    child_state,
                    options,
                    tracker,
                    ctx,
                    progress,
                    should_cancel,
                )
            )
        await asyncio.gather(*coros)
        # Propagate "we have entered the apocryphal zone" upward so that
        # later siblings of this list (same parent dict) keep the flag.
        state.apocryphal_zone = current


async def _process_async(
    root: Any,
    options: ProcessOptions,
    tracker: DiffTracker,
    ctx: ProcessContext,
    progress: ProgressCb | None,
    should_cancel: ShouldCancelCb | None,
) -> Any:
    state = TraverseState()
    # Bind concurrency primitives to the current event loop. Reset on every
    # call because asyncio.Future / Semaphore objects are loop-local and
    # `process_json_document` may be invoked once per book.
    ctx.ai_semaphore = asyncio.Semaphore(max(1, settings.ai_max_concurrency))
    ctx.reviewed_text_futures = {}
    ctx.generated_title_futures = {}
    await _walk(root, "", state, options, tracker, ctx, progress, should_cancel)
    return root


async def process_json_document_async(
    data: Any,
    options: ProcessOptions,
    tracker: DiffTracker,
    progress: ProgressCb | None = None,
    ctx: ProcessContext | None = None,
    should_cancel: ShouldCancelCb | None = None,
    *,
    in_place: bool = False,
) -> Any:
    """Async entry point for the processing pipeline.

    Use this from inside a running event loop (for example, from an
    `async def` FastAPI route). Use the synchronous `process_json_document`
    wrapper from regular sync code (CLI, threadpool-backed handlers,
    `BackgroundTasks`).

    `in_place=True` skips the protective deep copy at the boundary; pass it
    only when the caller already owns a fresh copy of the data.
    """
    root = data if in_place else copy.deepcopy(data)
    if ctx is None:
        ctx = ProcessContext()
    return await _process_async(root, options, tracker, ctx, progress, should_cancel)


def process_json_document(
    data: Any,
    options: ProcessOptions,
    tracker: DiffTracker,
    progress: ProgressCb | None = None,
    ctx: ProcessContext | None = None,
    should_cancel: ShouldCancelCb | None = None,
    *,
    in_place: bool = False,
) -> Any:
    """Synchronous entry point. Internally runs the async pipeline.

    Safe to call from regular synchronous code (CLI, threadpool-backed
    handlers, BackgroundTasks). NOT safe to call from inside a running
    event loop — use `process_json_document_async` instead.

    `in_place=True` skips the protective deep copy at the boundary; pass it
    only when the caller already owns a fresh copy of the data.
    """
    return asyncio.run(
        process_json_document_async(
            data,
            options,
            tracker,
            progress,
            ctx,
            should_cancel,
            in_place=in_place,
        )
    )


def estimate_progress_units(data: Any, options: ProcessOptions) -> int:
    state = TraverseState()
    total = 0

    def count_text_units(text: str) -> int:
        if not text.strip():
            return 0
        return len(_chunk_string(text, settings.max_chunk_chars))

    def walk(node: Any) -> None:
        nonlocal total
        if isinstance(node, dict):
            if _normalize_apocryphal_id(node.get("__id__")):
                state.apocryphal_zone = True

            if "qumran" in node:
                walk_qumran(node["qumran"])

            if "comentarios" in node:
                process_comentarios(node["comentarios"])

            for key, val in node.items():
                if key in ("qumran", "comentarios", "texto_ingles"):
                    continue
                if key == "texto" and isinstance(val, str):
                    if _should_review_biblical_texto(node, state, options):
                        total += count_text_units(val)
                    continue
                if key == "titulo" and isinstance(val, str):
                    total += count_text_units(val)
                    continue
                walk(val)
        elif isinstance(node, list):
            for item in node:
                walk(item)

    def process_comentarios(node: Any) -> None:
        nonlocal total
        if isinstance(node, list):
            for item in node:
                if not isinstance(item, dict):
                    continue
                t = item.get("texto")
                if isinstance(t, str) and t.strip():
                    total += count_text_units(t)
                raw = item.get("titulo")
                body = item.get("texto") if isinstance(item.get("texto"), str) else ""
                if raw is None or (isinstance(raw, str) and not raw.strip()):
                    if isinstance(body, str) and body.strip():
                        total += 1
                elif isinstance(raw, str) and raw.strip():
                    total += count_text_units(raw)
                for k, v in item.items():
                    if k in ("texto", "titulo"):
                        continue
                    walk(v)
        elif isinstance(node, dict):
            walk(node)

    def walk_qumran(node: Any) -> None:
        nonlocal total
        if isinstance(node, dict):
            for k, v in node.items():
                if k == "texto_ingles":
                    continue
                if isinstance(v, str):
                    total += count_text_units(v)
                else:
                    walk_qumran(v)
        elif isinstance(node, list):
            for item in node:
                walk_qumran(item)
        elif isinstance(node, str):
            total += count_text_units(node)

    walk(data)
    return max(total, 1)
