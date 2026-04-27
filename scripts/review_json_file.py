"""Run process_json_document on a JSON file (CLI helper for large local reviews)."""
from __future__ import annotations

import argparse
import copy
import json
import re
import shutil
import sys
import time
from pathlib import Path
from typing import Any

# Run from backend/: PYTHONPATH=. python scripts/review_json_file.py ...
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.services.json_processor import (  # noqa: E402
    ProcessContext,
    ProcessOptions,
    estimate_progress_units,
    process_json_document,
)
from app.utils.diff_tracker import DiffTracker  # noqa: E402


def _is_daily_rate_limit_error(exc: BaseException) -> bool:
    msg = str(exc).lower()
    patterns = (
        "requests per day",
        "rpd",
        "rate limit reached",
        "used",
        "limit",
    )
    return all(p in msg for p in ("rate limit", "day")) or (
        "429" in msg and all(p in msg for p in patterns)
    )


def _extract_retry_hint_seconds(exc: BaseException) -> int | None:
    msg = str(exc)
    # Common variants:
    # "Please try again in 8.64s."
    # "Try again in 2m30s."
    sec_match = re.search(r"try again in\s+(\d+(?:\.\d+)?)s", msg, flags=re.I)
    if sec_match:
        return max(1, int(float(sec_match.group(1))))
    min_sec_match = re.search(r"try again in\s+(\d+)m(\d+)s", msg, flags=re.I)
    if min_sec_match:
        return int(min_sec_match.group(1)) * 60 + int(min_sec_match.group(2))
    return None


def _checkpoint_path_for_output(out_path: Path) -> Path:
    return out_path.with_suffix(out_path.suffix + ".resume.json")


def _save_checkpoint(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _load_checkpoint(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise SystemExit(f"Resume checkpoint not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    # When stdout is piped/captured, Python may block-buffer; keep logs timely.
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except (AttributeError, OSError):
        pass

    ap = argparse.ArgumentParser()
    ap.add_argument("input", type=Path, help="Path to input JSON")
    ap.add_argument(
        "--in-place",
        action="store_true",
        help="Write result to INPUT (creates .bak backup first)",
    )
    ap.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output JSON path (default: INPUT.reviewed.json)",
    )
    ap.add_argument("--target-language", default="de")
    ap.add_argument(
        "--treat-biblical-texto-as-google",
        action="store_true",
        help="Review all biblical 'texto' fields, not only Google-tagged / apocryphal.",
    )
    ap.add_argument(
        "--by-livro",
        action="store_true",
        help=(
            "If the root object has a top-level 'livros' array, process each book separately "
            "with one shared memo cache, rewrite the output file after each book (safer for huge files)."
        ),
    )
    ap.add_argument(
        "--livro-start",
        type=int,
        default=0,
        help="With --by-livro: first index to process (default 0).",
    )
    ap.add_argument(
        "--livro-end",
        type=int,
        default=None,
        help="With --by-livro: stop before this index (default: len(livros)).",
    )
    ap.add_argument(
        "--resume-from",
        type=Path,
        default=None,
        help=(
            "Resume from a checkpoint JSON created by this script "
            "(default checkpoint path: OUTPUT.resume.json)."
        ),
    )
    args = ap.parse_args()

    inp: Path = args.input
    if not inp.is_file():
        raise SystemExit(f"Not a file: {inp}")

    out: Path
    if args.in_place:
        bak = inp.with_suffix(inp.suffix + ".bak")
        print(f"Backup -> {bak}", flush=True)
        shutil.copy2(inp, bak)
        out = inp
    else:
        out = args.output or inp.with_name(inp.stem + ".reviewed.json")
    resume_path = args.resume_from or _checkpoint_path_for_output(out)

    if args.resume_from and not args.by_livro:
        raise SystemExit("--resume-from currently requires --by-livro.")

    print(f"Loading {inp} ...", flush=True)
    t0 = time.time()
    data = json.loads(inp.read_text(encoding="utf-8"))
    print(f"Loaded in {time.time() - t0:.2f}s", flush=True)

    opts = ProcessOptions(
        target_language=args.target_language,
        treat_biblical_texto_as_google=args.treat_biblical_texto_as_google,
    )
    total = estimate_progress_units(data, opts)
    print(f"Estimated progress units: {total}", flush=True)

    tracker = DiffTracker()
    done = [0]
    last_log = [0.0]

    def on_progress(event: str, path: str | None) -> None:
        if event in ("chunk", "title_gen", "memo_hit"):
            done[0] += 1
            n = done[0]
            now = time.time()
            if (
                n == 1
                or n >= total
                or n % 25 == 0
                or now - last_log[0] >= 30.0
            ):
                last_log[0] = now
                pct = round(100.0 * min(n, total) / max(total, 1), 2)
                print(f"  progress {n}/{total} ({pct}%) {event} {path!r}", flush=True)

    out.parent.mkdir(parents=True, exist_ok=True)

    t1 = time.time()
    if args.by_livro and isinstance(data, dict) and isinstance(data.get("livros"), list):
        root = copy.deepcopy(data)
        livros = root["livros"]
        end = args.livro_end if args.livro_end is not None else len(livros)
        end = min(max(end, 0), len(livros))
        start = min(max(args.livro_start, 0), len(livros))
        if start >= end:
            raise SystemExit(f"Invalid livro range: start={start} end={end}")

        # Resume support:
        # - If checkpoint exists, continue from saved state.
        # - Otherwise start fresh and create output skeleton.
        if resume_path.is_file():
            cp = _load_checkpoint(resume_path)
            if cp.get("input_path") != str(inp.resolve()):
                raise SystemExit(
                    f"Checkpoint input mismatch:\n  checkpoint={cp.get('input_path')}\n  current={inp.resolve()}"
                )
            checkpoint_out = Path(cp.get("output_path", out))
            if checkpoint_out.resolve() != out.resolve():
                raise SystemExit(
                    f"Checkpoint output mismatch:\n  checkpoint={checkpoint_out}\n  current={out.resolve()}"
                )
            next_index = int(cp.get("next_livro_index", start))
            start = min(max(next_index, start), end)
            if out.is_file():
                root = json.loads(out.read_text(encoding="utf-8"))
                livros = root["livros"]
                print(f"Resuming from existing output {out}", flush=True)
            else:
                print("Checkpoint found but output file missing; rebuilding from input.", flush=True)
                out.write_text(
                    json.dumps(root, ensure_ascii=False, indent=4),
                    encoding="utf-8",
                )
            print(f"Resuming at livros[{start}] from {resume_path}", flush=True)
        else:
            out.write_text(
                json.dumps(root, ensure_ascii=False, indent=4),
                encoding="utf-8",
            )
            print(f"Wrote initial copy -> {out}", flush=True)
            _save_checkpoint(
                resume_path,
                {
                    "input_path": str(inp.resolve()),
                    "output_path": str(out.resolve()),
                    "next_livro_index": start,
                    "livro_end": end,
                    "updated_at": int(time.time()),
                },
            )
            print(f"Created checkpoint -> {resume_path}", flush=True)

        ctx = ProcessContext(reviewed_text_cache={}, generated_title_cache={})
        for i in range(start, end):
            bid = livros[i].get("__id__", i) if isinstance(livros[i], dict) else i
            print(f"--- livros[{i}] ({bid!r}) ---", flush=True)
            wrapped = {"livros": [copy.deepcopy(livros[i])]}
            try:
                processed = process_json_document(
                    wrapped, opts, tracker, progress=on_progress, ctx=ctx
                )
            except Exception as exc:
                # Keep checkpoint at current index so rerun continues from the same livro.
                _save_checkpoint(
                    resume_path,
                    {
                        "input_path": str(inp.resolve()),
                        "output_path": str(out.resolve()),
                        "next_livro_index": i,
                        "livro_end": end,
                        "updated_at": int(time.time()),
                        "last_error": str(exc),
                    },
                )
                if _is_daily_rate_limit_error(exc):
                    print(
                        f"\nDaily request cap reached near livros[{i}]. "
                        f"Checkpoint saved to {resume_path}",
                        flush=True,
                    )
                    print(
                        "Increase RPD/quota or rerun after reset with the same command; "
                        "it will resume automatically.",
                        flush=True,
                    )
                else:
                    hint = _extract_retry_hint_seconds(exc)
                    if hint is not None:
                        print(
                            f"\nRate-limited near livros[{i}] (retry hint: ~{hint}s). "
                            f"Checkpoint saved to {resume_path}",
                            flush=True,
                        )
                raise
            livros[i] = processed["livros"][0]
            out.write_text(
                json.dumps(root, ensure_ascii=False, indent=4),
                encoding="utf-8",
            )
            _save_checkpoint(
                resume_path,
                {
                    "input_path": str(inp.resolve()),
                    "output_path": str(out.resolve()),
                    "next_livro_index": i + 1,
                    "livro_end": end,
                    "updated_at": int(time.time()),
                },
            )
            print(f"Saved after livros[{i}] ({time.time() - t1:.0f}s elapsed)", flush=True)

        result = root
    else:
        if args.by_livro:
            print("No top-level 'livros' array; falling back to whole-document pass.", flush=True)
        result = process_json_document(data, opts, tracker, progress=on_progress)

    print(f"Processing wall time: {time.time() - t1:.2f}s", flush=True)

    out.write_text(
        json.dumps(result, ensure_ascii=False, indent=4),
        encoding="utf-8",
    )
    changes_path = out.with_suffix(out.suffix + ".changes.json")
    changes_path.write_text(
        json.dumps(tracker.to_json(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Wrote {out}", flush=True)
    print(f"Wrote {changes_path} ({len(tracker.changes)} entries)", flush=True)
    if resume_path.is_file():
        try:
            resume_path.unlink()
            print(f"Removed checkpoint {resume_path}", flush=True)
        except OSError:
            print(f"Finished, but could not remove checkpoint {resume_path}", flush=True)


if __name__ == "__main__":
    main()
