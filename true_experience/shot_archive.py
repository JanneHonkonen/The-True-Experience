"""Private ZIP of timestamped shot thumbnails and exact model-evidence frames.

Works entirely offline with FFmpeg/Pillow. The archive contains sampled images
from the input video (which may be copyrighted), so it is created ONLY after an
explicit opt-in and never sent to GitHub automatically.
"""
from __future__ import annotations

import argparse
import io
import json
import math
import sys
import zipfile
from collections import defaultdict
from pathlib import Path

from .evaluation import validate_report
from .perception import _extract_image

MANIFEST_NAME = "manifest.json"
MAX_SHOTS = 10000
MAX_EVIDENCE_FRAMES = 10000


def timestamp_name(t: float) -> str:
    """Filename-safe HH-MM-SS.ffffff; JSON retains exact source float values."""
    if not isinstance(t, (int, float)) or isinstance(t, bool) or not math.isfinite(t) or t < 0:
        raise ValueError(f"Invalid timestamp: {t!r}")
    us = round(t * 1_000_000)
    h, rem = divmod(us, 3_600_000_000)
    m, rem = divmod(rem, 60_000_000)
    sec, us = divmod(rem, 1_000_000)
    return f"{h:02d}-{m:02d}-{sec:02d}.{us:06d}"


def _planned_context_ids(report, window_seconds, max_images):
    """Map model-visible evidence IDs to intended context windows."""
    from .context import choose_frames, make_windows
    mapping = defaultdict(list)
    for w in make_windows(report, window_seconds):
        for frame in choose_frames(w["frames"], max_images):
            mapping[frame["id"]].append(w["window_id"])
    return mapping


def _actual_context_ids(context):
    mapping=defaultdict(list)
    for window in context.get("windows", []):
        for eid in window.get("evidence_ids", []):
            if isinstance(eid, str) and eid.startswith("v"):
                mapping[eid].append(window["window_id"])
    return mapping


def _jpeg_bytes(video, time, loader, quality):
    frame = loader(video, time)
    buffer = io.BytesIO()
    frame.save(buffer, format="JPEG", quality=quality)
    data = buffer.getvalue()
    if not data.startswith(b"\xff\xd8"):
        raise RuntimeError(f"JPEG encode failed at timestamp {time}")
    return data


def create_archive(video, report, output, *, context=None, context_plan=None,
                   frame_loader=None, quality=80):
    """Create ZIP of each shot midpoint and every exact caption-source frame.

    When context is given, link evidence IDs to windows that actually cited
    them. If context_plan is set instead, links are marked planned (not
    completed). A map links shot IDs, JSON times, VLM evidence IDs and images.
    No raw ASR transcript or full video/audio is included.
    """
    video = Path(video)
    output = Path(output)
    if not video.is_file():
        raise FileNotFoundError(f"Video does not exist: {video}")
    if output.resolve() == video.resolve():
        raise ValueError("Archive output cannot overwrite source video")
    if context is not None and context_plan is not None:
        raise ValueError("Use actual context OR a proposed context plan, not both")
    if not isinstance(quality, int) or isinstance(quality, bool) or not 1 <= quality <= 95:
        raise ValueError("JPEG quality must be between 1 and 95")
    valid=validate_report(report)
    if not valid["valid"]:
        raise ValueError("Analysis is structurally invalid: "+"; ".join(valid["errors"][:3]))
    shots=report["shots"]
    samples=report.get("visual_observations",[])
    if len(shots)>MAX_SHOTS or len(samples)>MAX_EVIDENCE_FRAMES:
        raise ValueError("Too many frames for one archive; narrow the input")
    source = frame_loader or _extract_image
    if context is not None:
        links = _actual_context_ids(context)
        link_status = "actual_context_evidence_ids"
    elif context_plan is not None:
        links = _planned_context_ids(report, context_plan[0], context_plan[1])
        link_status = "planned_context_frames_not_verified_run"
    else:
        links = {}
        link_status = "not_provided"

    samples = sorted(samples,key=lambda s:(s["frame_time_seconds"],s["shot_id"]))
    image_manifest = {"schema_version":"1.0",
        "source_video_name":video.name,
        "video_duration_seconds":report["metadata"]["duration_seconds"],
        "shot_count":len(shots),"visual_sample_count":len(samples),
        "context_links_status":link_status,
        "shots":[],"visual_evidence":[],
        "privacy":{
            "contains_copyrighted_sampled_frames":True,
            "contains_full_video":False,"contains_audio":False,
            "contains_raw_asr_dialogue":False,
            "automatic_upload":False
        },
        "notes":[
            "Image filenames use Windows-safe timestamps (- instead of :), 6 fractional digits.",
            "start_seconds/end_seconds and frame_time_seconds in this manifest match analysis JSON.",
            "Shot thumbnails sample a single midpoint, not all motion in a shot.",
            "Visual evidence images match the precise timestamps of v0001, v0002, etc.",
            "Frame captions are unverified. Only users can determine whether an image supports a claim."
        ]}
    evidence_by_shot=defaultdict(list)
    for i,sample in enumerate(samples,1):
        evidence_by_shot[sample["shot_id"]].append(f"v{i:04d}")

    output.parent.mkdir(parents=True,exist_ok=True)
    tmp=output.with_name(output.name+".tmp")
    try:
        with zipfile.ZipFile(tmp,"w",compression=zipfile.ZIP_STORED) as archive:
            for shot in shots:
                a,b=shot["start_seconds"],shot["end_seconds"]
                frame_t=round((a+b)/2,6)
                name=(f"shots/shot_{shot['shot_id']:04d}_"
                      f"{timestamp_name(a)}_to_{timestamp_name(b)}.jpg")
                archive.writestr(name,_jpeg_bytes(video,frame_t,source,quality))
                image_manifest["shots"].append({
                    "shot_id":shot["shot_id"],"start_seconds":a,"end_seconds":b,
                    "start":shot.get("start"),"end":shot.get("end"),
                    "frame_time_seconds":frame_t,"image":name,
                    "visual_evidence_ids":evidence_by_shot.get(shot["shot_id"],[])
                })
            for i,sample in enumerate(samples,1):
                eid=f"v{i:04d}"
                t=sample["frame_time_seconds"]
                name=(f"evidence/{eid}_shot_{sample['shot_id']:04d}_"
                      f"at_{timestamp_name(t)}.jpg")
                archive.writestr(name,_jpeg_bytes(video,t,source,quality))
                item={
                    "evidence_id":eid,"shot_id":sample["shot_id"],
                    "frame_time_seconds":t,"image":name,
                    "caption_hypothesis":sample.get("description"),
                    "source_model":sample.get("model"),
                    "context_windows":links.get(eid,[])
                }
                image_manifest["visual_evidence"].append(item)
            archive.writestr(MANIFEST_NAME,json.dumps(image_manifest,
                               ensure_ascii=False,indent=2,allow_nan=False)+"\n")
            archive.writestr("README.txt",
                "LOCAL VIDEO EVALUATION FRAMES\n\n"
                "shots/: one still image sampled at the midpoint of every detected shot.\n"
                "evidence/: exact sampled visual-observation timestamps, named v0001, v0002, etc.\n"
                "manifest.json: authoritative mapping to shot IDs, JSON timestamps, frame captions and model evidence IDs.\n"
                "Times are represented to microsecond precision in filenames and as numbers in manifest.json.\n"
                "These are user-provided video frames; review sharing rights before uploading.\n"
                "No transcript, video or audio has been copied into this ZIP.\n")
        tmp.replace(output)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    return image_manifest


def _read_analysis(file,video):
    data=json.loads(Path(file).read_text(encoding="utf-8"))
    if "report" in data:
        signature=data.get("signature",{})
        stat=Path(video).resolve().stat()
        if not isinstance(signature,dict) or any((
            signature.get("absolute_video_path")!=str(Path(video).resolve()),
            signature.get("video_bytes")!=stat.st_size,
            signature.get("modified_ns")!=stat.st_mtime_ns,
        )):
            raise ValueError("Cached analysis does not match the video path/size/mtime")
        return data["report"]
    return data


def main(argv=None):
    p=argparse.ArgumentParser(description="Archive shot thumbnails and actual sampled evidence frames without AI model inference")
    p.add_argument("video",type=Path)
    p.add_argument("--analysis",type=Path,required=True,
                   help="Unredacted analysis JSON or the PRIVATE analysis cache")
    p.add_argument("--context",type=Path,help="Optional PRIVATE context JSON to map model-used evidence IDs")
    p.add_argument("-o","--output",type=Path,default=Path("iron_man_shots.zip"))
    args=p.parse_args(argv)
    try:
        report=_read_analysis(args.analysis,args.video)
        context=json.loads(args.context.read_text(encoding="utf-8")) if args.context else None
        manifest=create_archive(args.video,report,args.output,context=context)
        print(f"Created {args.output}: {manifest['shot_count']} shot thumbnails and "
              f"{manifest['visual_sample_count']} exact visual evidence frames")
        print("This ZIP contains source-video frames; review rights before sharing.")
        return 0
    except (ValueError,OSError,KeyError,TypeError,RuntimeError,zipfile.BadZipFile) as exc:
        print(f"Thumbnail archive failed: {exc}",file=sys.stderr)
        return 1


if __name__=="__main__":
    raise SystemExit(main())
