"""Sanitized evidence-linked review without redistributing video or ASR text."""
from __future__ import annotations
import argparse
import json
import math
import re
from pathlib import Path
from .evaluation import validate_report

VIDEO_ID = re.compile(r"^[A-Za-z0-9_-]{11}$")


def make_review(report: dict, video_id: str, *, max_captions: int = 60) -> dict:
    if not isinstance(video_id, str) or not VIDEO_ID.fullmatch(video_id):
        raise ValueError("Invalid YouTube video ID")
    if isinstance(max_captions, bool) or not isinstance(max_captions, int) or max_captions < 1:
        raise ValueError("max_captions must be a positive integer")
    validation = validate_report(report)
    if not validation["valid"]:
        raise ValueError("Invalid analysis report: " + "; ".join(validation["errors"][:5]))
    meta = report["metadata"]
    duration = float(meta["duration_seconds"])
    observations = report.get("visual_observations", [])
    speech = report.get("speech_segments", [])
    frames = []
    for obs in sorted(observations, key=lambda o: o["frame_time_seconds"])[:max_captions]:
        t = float(obs["frame_time_seconds"])
        if not math.isfinite(t) or t < 0 or t > duration:
            raise ValueError("Invalid caption time")
        frames.append({
            "time_seconds": round(t, 3),
            "shot_id": obs["shot_id"],
            "caption_hypothesis": obs["description"],
            "model": obs.get("model"),
            "view_source_at_time": f"https://www.youtube.com/watch?v={video_id}&t={int(t)}s",
            "human_groundedness_review": "not_reviewed",
        })
    spoken_duration = sum(max(0.0, float(s["end_seconds"])-float(s["start_seconds"])) for s in speech)
    word_count = sum(len(s.get("text", "").split()) for s in speech)
    unique_captions = len({c["caption_hypothesis"].strip().casefold() for c in frames})
    return {
        "evaluation_type": "unannotated_real_video",
        "source": {
            "youtube_video_id": video_id,
            "url": f"https://www.youtube.com/watch?v={video_id}",
            "source_title": "Iron Man (2008) Trailer #1 | Movieclips Classic Trailers",
            "original_video_included": False,
            "raw_transcript_included": False,
            "frames_included": False,
        },
        "structure": {
            "valid": True,
            "duration_seconds": duration,
            "shots_detected": len(report.get("shots", [])),
            "timeline_intervals": len(report.get("timeline", [])),
            "audio_stream_present": bool(meta.get("has_audio")),
            "source_frame_rate": meta.get("frame_rate"),
        },
        "speech": {
            "asr_model": (meta.get("perception") or {}).get("speech_model"),
            "segment_count": len(speech),
            "word_count": word_count,
            "sum_of_segment_durations_seconds": round(spoken_duration, 2),
            "word_error_rate": None,
            "word_error_rate_reason": "No independently annotated reference transcript",
            "speaker_identification": "not_performed",
            "copyright_note": "ASR text excluded from shared report",
        },
        "visual": {
            "model": (meta.get("perception") or {}).get("caption_model"),
            "total_captions": len(observations),
            "caption_examples_in_report": len(frames),
            "distinct_caption_texts_in_report": unique_captions,
            "sampling": (meta.get("perception") or {}).get("visual_sampling"),
            "caption_accuracy": None,
            "caption_accuracy_reason": "Requires frame-by-frame human comparison",
            "caption_hypotheses": frames,
        },
        "conclusions": [
            "Completing inference and passing structure checks do not establish film understanding.",
            "Captions are model hypotheses, not verified descriptions of shots.",
            "Cut boundaries are not necessarily semantic scene boundaries.",
            "Speech word counts and segments do not measure transcription accuracy.",
            "No emotion, character identity, event continuity, or narrative accuracy was evaluated.",
            "Review timestamped captions against the original video before judging accuracy.",
        ],
    }


def main(argv=None) -> int:
    p=argparse.ArgumentParser(description="Create source-linked sanitized review report")
    p.add_argument("analysis",type=Path)
    p.add_argument("--video-id",required=True)
    p.add_argument("--max-captions",type=int,default=60)
    p.add_argument("-o","--output",type=Path,required=True)
    args=p.parse_args(argv)
    try:
        data=json.loads(args.analysis.read_text(encoding="utf-8"))
        result=make_review(data,args.video_id,max_captions=args.max_captions)
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+"\n",encoding="utf-8")
        print(f"Sanitized report: {result['structure']['shots_detected']} shots, "
              f"{result['speech']['segment_count']} speech segments, "
              f"{result['visual']['total_captions']} frame hypotheses")
        return 0
    except (ValueError, KeyError, TypeError, OSError) as exc:
        p.exit(1,f"Review report failed: {exc}\n")


if __name__=="__main__":
    raise SystemExit(main())
