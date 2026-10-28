"""Fixture-specific descriptive score for the manual synthetic real-model smoke.

This assesses only the red/blue/green solid-color fixture created by the
repository's workflow, not general film caption truth or hallucination rates.
"""
import argparse
import json
import math
import re
from pathlib import Path

EXPECTED = ((5.0, "red"), (15.0, "blue"), (25.0, "green"))
COLORS = {"red", "blue", "green", "white", "black", "yellow", "orange", "pink", "purple"}


def assess_synthetic_fixture(report):
    observations=report.get("visual_observations",[])
    if not isinstance(observations,list):
        raise ValueError("visual_observations must be a list")
    details=[]
    for t,color in EXPECTED:
        candidates=[obs for obs in observations
                    if isinstance(obs,dict) and
                    isinstance(obs.get("frame_time_seconds"),(float,int)) and
                    not isinstance(obs.get("frame_time_seconds"),bool) and
                    math.isfinite(obs["frame_time_seconds"]) and
                    abs(obs["frame_time_seconds"]-t)<0.2]
        if len(candidates)>1:
            raise ValueError(f"Ambiguous frame samples at {t}s")
        obs=candidates[0] if candidates else None
        caption=obs.get("description","") if obs else ""
        if not isinstance(caption,str):
            raise ValueError("Caption must be text")
        words=set(re.findall(r"[a-z]+",caption.casefold()))
        wrong_colors=sorted((words & COLORS) - {color})
        details.append({
            "frame_time_seconds":t, "expected_color":color, "caption":caption or None,
            "color_mentioned":color in words if obs else None,
            "unsupported_border_claim":("border" in words) if obs else None,
            "other_color_words":wrong_colors,
        })
    present=[x for x in details if x["caption"] is not None]
    return {
        "fixture":"ffmpeg_solid_red_blue_green_with_espeak",
        "scope":"Only synthetic fixture color words and border mentions; not movie understanding",
        "expected_sample_count":len(EXPECTED),
        "evaluated_caption_count":len(present),
        "correct_color_count":sum(x["color_mentioned"] is True for x in present),
        "unsupported_border_claim_count":sum(x["unsupported_border_claim"] is True for x in present),
        "wrong_color_word_count":sum(len(x["other_color_words"]) for x in present),
        "details":details,
    }


def main(argv=None):
    p=argparse.ArgumentParser(description="Score captions against known synthetic fixture")
    p.add_argument("report",type=Path)
    p.add_argument("-o","--output",type=Path)
    args=p.parse_args(argv)
    try:
        report=json.loads(args.report.read_text(encoding="utf-8"))
        result=assess_synthetic_fixture(report)
        payload=json.dumps(result,indent=2,ensure_ascii=False,allow_nan=False)+"\n"
        if args.output:
            args.output.write_text(payload,encoding="utf-8")
        else:
            print(payload,end="")
        return 0
    except (ValueError,OSError,KeyError,TypeError) as exc:
        p.exit(1,f"Fixture scoring failed: {exc}\n")


if __name__=="__main__":
    raise SystemExit(main())
