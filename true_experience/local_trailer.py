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
from .evaluation import validate_report
from .shot_archive import create_archive

def _cache_signature(video, options):
    stat=video.resolve().stat()
    return {"absolute_video_path":str(video.resolve()),
            "video_bytes":stat.st_size,"modified_ns":stat.st_mtime_ns,
            "options":options}

def _write_json_atomic(target, data):
    target.parent.mkdir(parents=True,exist_ok=True)
    tmp=target.with_name(target.name+".tmp")
    tmp.write_text(json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False)+"\n",
                   encoding="utf-8")
    tmp.replace(target)


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
    p.add_argument("--analysis-cache",type=Path,
                   default=Path("iron_man_analysis_private.json"),
                   help="PRIVATE full ASR/BLIP cache used when context-model is enabled")
    p.add_argument("--reuse-analysis",action="store_true",
                   help="Reuse validated PRIVATE source analysis; skip Whisper and BLIP")
    p.add_argument("--resume-context",action="store_true",
                   help="Resume previously checkpointed context windows (requires cache reuse)")
    thumbs = p.add_mutually_exclusive_group()
    thumbs.add_argument("--thumbnails",action="store_true",default=None,
                        help="Create local ZIP of shot thumbnails + exact model evidence frames")
    thumbs.add_argument("--no-thumbnails",action="store_false",dest="thumbnails",
                        help="Never prompt or create thumbnail ZIP")
    p.add_argument("--thumbnails-output",type=Path,
                   help="ZIP destination; defaults to <public_report_stem>_shots.zip")
    args = p.parse_args(argv)
    try:
        if not args.video.is_file():
            raise FileNotFoundError(f"Video file not found: {args.video}")
        if args.reuse_analysis and not args.context_model:
            raise ValueError("--reuse-analysis requires --context-model")
        if args.resume_context and not args.reuse_analysis:
            raise ValueError("--resume-context also requires --reuse-analysis")
        destinations = [args.output.resolve()]
        if args.context_model:
            destinations.extend([args.context_output.resolve(),args.analysis_cache.resolve()])
        thumbnail_target = (args.thumbnails_output if args.thumbnails_output
                            else args.output.with_name(args.output.stem+"_shots.zip"))
        if thumbnail_target.resolve() in destinations or thumbnail_target.resolve()==args.video.resolve():
            raise ValueError("Thumbnail ZIP must be distinct from all source and JSON files")
        if len(destinations)!=len(set(destinations)):
            raise ValueError("Public, analysis cache and context output files must be different")
        pipeline_options={"interval":args.interval, "whisper_model":args.whisper_model,
                          "language":args.language,"caption_frame_cap":args.max_caption_frames,
                          "caption_spacing":args.caption_spacing}
        signature=_cache_signature(args.video,pipeline_options) if args.context_model else None
        if args.reuse_analysis:
            if not args.analysis_cache.is_file():
                raise ValueError(f"No analysis cache at {args.analysis_cache}. Run once without --reuse-analysis")
            cached=json.loads(args.analysis_cache.read_text(encoding="utf-8"))
            if not isinstance(cached,dict) or cached.get("signature")!=signature:
                raise ValueError("Analysis cache does not match current video or inference settings")
            result=cached.get("report")
            if not isinstance(result,dict):
                raise ValueError("Analysis cache contains no valid report")
            validation=validate_report(result)
            if not validation["valid"]:
                raise ValueError("Cached report failed structural validation")
            print("Reused private analysis cache; skipped Whisper and BLIP.",file=sys.stderr)
        else:
            result = analyze(
                args.video, interval=args.interval, transcribe=True,
                whisper_model=args.whisper_model, language=args.language,
                caption_frames=True, max_caption_frames=args.max_caption_frames,
                caption_spacing=args.caption_spacing)
            if args.context_model:
                _write_json_atomic(args.analysis_cache,{
                    "cache_version":1,"signature":signature,"report":result,
                    "privacy":"PRIVATE: contains raw speech transcription; never publish"})
                print(f"Private analysis cache saved at {args.analysis_cache}",file=sys.stderr)
        public_report = make_review(result, args.video_id)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(public_report, ensure_ascii=False, indent=2, allow_nan=False)+"\n",
            encoding="utf-8")
        # Ask ONLY in an interactive terminal, and default to no in CI.
        # Archive runs after analysis but before any expensive context LLM calls.
        do_thumbnails = args.thumbnails
        if do_thumbnails is None:
            if sys.stdin.isatty():
                try:
                    answer=input(f"Create shot/evidence thumbnail ZIP at {thumbnail_target}? [y/N]: ")
                except EOFError:
                    answer=""
                do_thumbnails=answer.strip().casefold() in ("y","yes")
            else:
                do_thumbnails=False
        archive_error=False
        if do_thumbnails:
            try:
                plan=((args.context_window,args.max_context_images)
                      if args.context_model else None)
                manifest=create_archive(args.video,result,thumbnail_target,context_plan=plan)
                print(f"Saved {manifest['shot_count']} shot images and "
                      f"{manifest['visual_sample_count']} exact visual evidence images "
                      f"to {thumbnail_target} (LOCAL; not uploaded)")
            except (OSError,RuntimeError,ValueError) as exc:
                archive_error=True
                print(f"Thumbnail archive failed; continuing context analysis: {exc}",
                      file=sys.stderr)
        if args.context_model:
            # Context can include ASR excerpts. Keep it separate from public output.
            def checkpoint(context):
                _write_json_atomic(args.context_output,context)
            resume=None
            if args.resume_context:
                if not args.context_output.is_file():
                    raise ValueError(f"No existing context checkpoint at {args.context_output}")
                resume=json.loads(args.context_output.read_text(encoding="utf-8"))
            context = build_context(
                result, args.video, model=args.context_model,
                window_seconds=args.context_window,
                max_images=args.max_context_images,
                timeout=args.context_timeout,
                on_window=checkpoint,
                resume=resume)
            checkpoint(context)
            print(f"Saved {len(context['windows'])} rolling-context windows to "
                  f"{args.context_output} (PRIVATE model hypotheses)")
        print(f"Saved {public_report['visual']['total_captions']} timestamped frame captions "
              f"and {public_report['speech']['segment_count']} ASR segment counts to {args.output}")
        print("Caption accuracy and transcription WER remain unmeasured.")
        return 1 if archive_error else 0
    except (OSError, RuntimeError, ValueError, TypeError, KeyError) as exc:
        print(f"Evaluation failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
