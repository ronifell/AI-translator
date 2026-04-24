"""Run process_json_document on a JSON file (CLI helper for large local reviews)."""
from __future__ import annotations

import argparse
import copy
import json
import shutil
import sys
import time
from pathlib import Path

# Run from backend/: PYTHONPATH=. python scripts/review_json_file.py ...
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.services.json_processor import (  # noqa: E402
    ProcessContext,
    ProcessOptions,
    estimate_progress_units,
    process_json_document,
)
from app.utils.diff_tracker import DiffTracker  # noqa: E402


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

        # Same rules as About.md / the web UI: write a full document early, then refresh per book.
        out.write_text(
            json.dumps(root, ensure_ascii=False, indent=4),
            encoding="utf-8",
        )
        print(f"Wrote initial copy -> {out}", flush=True)

        ctx = ProcessContext(reviewed_text_cache={}, generated_title_cache={})
        for i in range(start, end):
            bid = livros[i].get("__id__", i) if isinstance(livros[i], dict) else i
            print(f"--- livros[{i}] ({bid!r}) ---", flush=True)
            wrapped = {"livros": [copy.deepcopy(livros[i])]}
            processed = process_json_document(
                wrapped, opts, tracker, progress=on_progress, ctx=ctx
            )
            livros[i] = processed["livros"][0]
            out.write_text(
                json.dumps(root, ensure_ascii=False, indent=4),
                encoding="utf-8",
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


if __name__ == "__main__":
    main()
