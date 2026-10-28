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
        print(f"Saved {public_report['visual']['total_captions']} timestamped frame captions "
              f"and {public_report['speech']['segment_count']} ASR segment counts to {args.output}")
        print("Caption accuracy and transcription WER remain unmeasured.")
        return 0
    except (OSError, RuntimeError, ValueError, TypeError, KeyError) as exc:
        print(f"Evaluation failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
