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
    a = p.parse_args(argv)
    try:
        result = analyze(a.video,interval=a.interval,scene_threshold=a.scene_threshold,
                         silence_db=a.silence_db,silence_min_duration=a.silence_min_duration,
                         subtitles_path=a.subtitles)
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
