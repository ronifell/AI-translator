from __future__ import annotations

import copy
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Callable

from app.services import ai_service
from app.config import settings
from app.utils.diff_tracker import DiffTracker


@dataclass
class TraverseState:
    apocryphal_zone: bool = False


@dataclass
class ProcessOptions:
    target_language: str = "pt-BR"
    treat_biblical_texto_as_google: bool = False


@dataclass
class ProcessContext:
    # Per-document memoization:
    # identical source texts are reviewed once, then reused across all paths.
    reviewed_text_cache: dict[str, str]
    generated_title_cache: dict[str, str]


ProgressCb = Callable[[str, str | None], None]


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


def _review_body_text(
    original: str,
    path: str,
    tracker: DiffTracker,
    options: ProcessOptions,
    ctx: ProcessContext,
    progress: ProgressCb | None,
) -> str:
    cached = ctx.reviewed_text_cache.get(original)
    if cached is not None:
        if progress:
            # Match estimate_progress_units: each chunk slot for this occurrence counts as one unit.
            n = len(_chunk_string(original, settings.max_chunk_chars))
            for j in range(n):
                subpath = f"{path}#part{j}" if n > 1 else path
                progress("memo_hit", subpath)
        tracker.record(path, original, cached)
        return cached

    chunks = _chunk_string(original, settings.max_chunk_chars)
    out: list[str] = []
    use_parallel = len(chunks) > 1 and settings.chunk_parallel_workers > 1
    if use_parallel:
        tasks: list[tuple[int, str, str]] = []
        for i, ch in enumerate(chunks):
            subpath = f"{path}#part{i}" if len(chunks) > 1 else path
            if progress:
                progress("chunk", subpath)
            tasks.append((i, ch, subpath))
        with ThreadPoolExecutor(max_workers=settings.chunk_parallel_workers) as pool:
            futures = [
                pool.submit(
                    ai_service.correct_text,
                    ch,
                    target_language_hint=options.target_language,
                    on_progress=(lambda _ev, sp=subpath: progress("ai", sp) if progress else None),
                )
                for (_i, ch, subpath) in tasks
            ]
            out = [""] * len(chunks)
            for (i, _ch, _subpath), fut in zip(tasks, futures):
                out[i] = fut.result()
    else:
        for i, ch in enumerate(chunks):
            subpath = f"{path}#part{i}" if len(chunks) > 1 else path
            if progress:
                progress("chunk", subpath)
            corrected = ai_service.correct_text(
                ch,
                target_language_hint=options.target_language,
                on_progress=lambda _: progress("ai", subpath) if progress else None,
            )
            out.append(corrected)
    merged = "".join(out)
    ctx.reviewed_text_cache[original] = merged
    tracker.record(path, original, merged)
    return merged


def _review_or_generate_title(
    item: dict[str, Any],
    titulo_path: str,
    tracker: DiffTracker,
    options: ProcessOptions,
    ctx: ProcessContext,
    progress: ProgressCb | None,
) -> None:
    raw = item.get("titulo")
    body = item.get("texto") if isinstance(item.get("texto"), str) else ""

    if raw is None or (isinstance(raw, str) and not raw.strip()):
        if not (isinstance(body, str) and body.strip()):
            return
        title = ctx.generated_title_cache.get(body)
        if title is None:
            if progress:
                progress("title_gen", titulo_path)
            title = ai_service.generate_title(
                body,
                on_progress=lambda _: progress("ai", titulo_path) if progress else None,
            )
            ctx.generated_title_cache[body] = title
        elif progress:
            progress("memo_hit", titulo_path)
        item["titulo"] = title
        tracker.record(titulo_path, raw if isinstance(raw, str) else "", title)
        return

    if isinstance(raw, str) and raw.strip():
        item["titulo"] = _review_body_text(raw, titulo_path, tracker, options, ctx, progress)


def _walk_qumran(
    node: Any,
    path: str,
    tracker: DiffTracker,
    options: ProcessOptions,
    ctx: ProcessContext,
    progress: ProgressCb | None,
) -> Any:
    if isinstance(node, dict):
        out: dict[str, Any] = {}
        for k, v in node.items():
            cp = _path_join(path, k)
            if k == "texto_ingles":
                out[k] = v
            elif isinstance(v, str):
                out[k] = _review_body_text(v, cp, tracker, options, ctx, progress)
            else:
                out[k] = _walk_qumran(v, cp, tracker, options, ctx, progress)
        return out
    if isinstance(node, list):
        return [
            _walk_qumran(item, _path_join(path, i), tracker, options, ctx, progress)
            for i, item in enumerate(node)
        ]
    if isinstance(node, str):
        return _review_body_text(node, path, tracker, options, ctx, progress)
    return node


def _process_comentarios(
    node: Any,
    path: str,
    state: TraverseState,
    options: ProcessOptions,
    tracker: DiffTracker,
    ctx: ProcessContext,
    progress: ProgressCb | None,
) -> None:
    if isinstance(node, list):
        for i, item in enumerate(node):
            p = _path_join(path, i)
            if not isinstance(item, dict):
                continue
            t = item.get("texto")
            if isinstance(t, str) and t.strip():
                item["texto"] = _review_body_text(
                    t, _path_join(p, "texto"), tracker, options, ctx, progress
                )
            _review_or_generate_title(
                item, _path_join(p, "titulo"), tracker, options, ctx, progress
            )
            for k, v in item.items():
                if k in ("texto", "titulo"):
                    continue
                _walk(v, _path_join(p, k), state, options, tracker, ctx, progress)
    elif isinstance(node, dict):
        _walk(node, path, state, options, tracker, ctx, progress)


def _walk(
    node: Any,
    path: str,
    state: TraverseState,
    options: ProcessOptions,
    tracker: DiffTracker,
    ctx: ProcessContext,
    progress: ProgressCb | None,
) -> None:
    if isinstance(node, dict):
        if _normalize_apocryphal_id(node.get("__id__")):
            state.apocryphal_zone = True

        if "qumran" in node:
            qpath = _path_join(path, "qumran")
            node["qumran"] = _walk_qumran(
                node["qumran"], qpath, tracker, options, ctx, progress
            )

        if "comentarios" in node:
            _process_comentarios(
                node["comentarios"],
                _path_join(path, "comentarios"),
                state,
                options,
                tracker,
                ctx,
                progress,
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
                    node[key] = _review_body_text(val, cur, tracker, options, ctx, progress)
                continue
            if key == "titulo" and isinstance(val, str):
                node[key] = _review_body_text(val, cur, tracker, options, ctx, progress)
                continue
            _walk(val, cur, state, options, tracker, ctx, progress)

    elif isinstance(node, list):
        for i, item in enumerate(node):
            _walk(item, _path_join(path, i), state, options, tracker, ctx, progress)


def process_json_document(
    data: Any,
    options: ProcessOptions,
    tracker: DiffTracker,
    progress: ProgressCb | None = None,
    ctx: ProcessContext | None = None,
) -> Any:
    root = copy.deepcopy(data)
    state = TraverseState()
    if ctx is None:
        ctx = ProcessContext(reviewed_text_cache={}, generated_title_cache={})
    _walk(root, "", state, options, tracker, ctx, progress)
    return root


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
