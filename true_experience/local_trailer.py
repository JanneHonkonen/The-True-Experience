"""Evaluate a user-provided local copy of the selected trailer.

No video downloading, authentication or GitHub media uploads. This outputs a
sanitized, timestamp-linked report; transient ASR text stays in process memory.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from .pipeline import analyze
from .trailer_review import make_review
from .context import build_context


def positive(value: str) -> float:
    result = float(value)
    if not math.isfinite(result) or result <= 0:
        raise argparse.ArgumentTypeError("Must be a positive finite number")
    return result


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Evaluate local Iron Man trailer with AI")
    p.add_argument("video", type=Path, help="Legally obtained video file; never uploaded by this program")
    p.add_argument("-o", "--output", type=Path, default=Path("iron_man_review.json"))
    p.add_argument("--video-id", default="8ugaeA-nMTc")
    p.add_argument("--whisper-model", default="tiny")
    p.add_argument("--language", default="en")
    p.add_argument("--max-caption-frames", type=int, default=36)
    p.add_argument("--caption-spacing", type=positive, default=6.0)
    p.add_argument("--interval", type=positive, default=5.0)
    p.add_argument("--context-model", help="Opt-in local Ollama vision model, e.g. gemma3:4b")
    p.add_argument("--context-output", type=Path, default=Path("iron_man_context_private.json"),
                   help="Private model-hypothesis analysis; do not publish without review")
    p.add_argument("--context-window", type=positive, default=12.0,
                   help="Length of contextual analysis window in seconds (not a scene)")
    p.add_argument("--max-context-images", type=int, default=3,
                   help="Maximum real video frames seen by Ollama in each window (1..8)")
    p.add_argument("--context-timeout", type=positive, default=300.0,
                   help="Timeout per local Ollama model inference, in seconds")
    args = p.parse_args(argv)
    try:
        if not args.video.is_file():
            raise FileNotFoundError(f"Video file not found: {args.video}")
        result = analyze(
            args.video, interval=args.interval, transcribe=True,
            whisper_model=args.whisper_model, language=args.language,
            caption_frames=True, max_caption_frames=args.max_caption_frames,
            caption_spacing=args.caption_spacing)
        public_report = make_review(result, args.video_id)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(public_report, ensure_ascii=False, indent=2, allow_nan=False)+"\n",
            encoding="utf-8")
        if args.context_model:
            # Checkpoint after each window so an interrupted long model run does
            # not discard earlier reasoning. This report may contain ASR paraphrases.
            def checkpoint(context):
                target = args.context_output
                target.parent.mkdir(parents=True, exist_ok=True)
                tmp = target.with_name(target.name + ".tmp")
                tmp.write_text(json.dumps(context, ensure_ascii=False, indent=2,
                                          allow_nan=False)+"\n", encoding="utf-8")
                tmp.replace(target)
            context = build_context(
                result, args.video, model=args.context_model,
                window_seconds=args.context_window,
                max_images=args.max_context_images,
                timeout=args.context_timeout,
                on_window=checkpoint)
            checkpoint(context)
            print(f"Saved {len(context['windows'])} rolling-context windows to "
                  f"{args.context_output} (PRIVATE model hypotheses)")
        print(f"Saved {public_report['visual']['total_captions']} timestamped frame captions "
              f"and {public_report['speech']['segment_count']} ASR segment counts to {args.output}")
        print("Caption accuracy and transcription WER remain unmeasured.")
        return 0
    except (OSError, RuntimeError, ValueError, TypeError, KeyError) as exc:
        print(f"Evaluation failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
