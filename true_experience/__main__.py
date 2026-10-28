"""CLI: python -m true_experience video.mp4 -o analysis.json"""
import argparse
import json
import math
import sys
from pathlib import Path
from .pipeline import analyze

def positive(value):
    n = float(value)
    if not math.isfinite(n) or n <= 0:
        raise argparse.ArgumentTypeError("Expected a positive finite number")
    return n

def main(argv=None):
    p = argparse.ArgumentParser(description="The True Experience: evidence-first video timeline")
    p.add_argument("video",help="Local video file")
    p.add_argument("-o","--output",default="analysis_output.json")
    p.add_argument("--interval",type=positive,default=5.0)
    p.add_argument("--scene-threshold",type=float,default=0.3)
    p.add_argument("--silence-db",type=float,default=-35.0)
    p.add_argument("--silence-min-duration",type=positive,default=0.25)
    p.add_argument("--subtitles",help="Optional local .srt file")
    p.add_argument("--transcribe",action="store_true",help="Run local Whisper speech recognition")
    p.add_argument("--whisper-model",default="tiny",help="faster-whisper model ID")
    p.add_argument("--language",help="Optional language code (e.g. fi); auto-detect if omitted")
    p.add_argument("--caption-frames",action="store_true",help="Run local BLIP frame captioning")
    p.add_argument("--caption-model",default="Salesforce/blip-image-captioning-base")
    p.add_argument("--max-caption-frames",type=int,default=120,help="Maximum sampled frames to caption")
    p.add_argument("--caption-spacing",type=positive,default=10.,help="Target maximum seconds per sample in each shot (budget permitting)")
    a = p.parse_args(argv)
    try:
        result = analyze(a.video,interval=a.interval,scene_threshold=a.scene_threshold,
                         silence_db=a.silence_db,silence_min_duration=a.silence_min_duration,
                         subtitles_path=a.subtitles,transcribe=a.transcribe,
                         whisper_model=a.whisper_model,language=a.language,
                         caption_frames=a.caption_frames,caption_model=a.caption_model,
                         max_caption_frames=a.max_caption_frames,caption_spacing=a.caption_spacing)
        path = Path(a.output)
        path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(json.dumps(result,indent=2,ensure_ascii=False,allow_nan=False)+"\n",encoding="utf-8")
        print(f"Saved {len(result['shots'])} shots and {len(result['timeline'])} intervals to {path}")
        return 0
    except (ValueError,OSError,RuntimeError) as exc:
        print(f"Analysis failed: {exc}",file=sys.stderr)
        return 1

if __name__ == "__main__":
    raise SystemExit(main())
