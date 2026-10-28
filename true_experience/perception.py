"""Optional local speech recognition and frame captioning with timestamps and provenance.

Model outputs are *hypotheses*, not established facts. Heavy dependencies load
only when explicitly requested, never in baseline FFmpeg-only mode.
"""
from __future__ import annotations
import io
import math
import subprocess
from pathlib import Path
from .sampling import plan_frame_samples


def _sample_shots(shots: list[dict], limit: int) -> list[dict]:
    """Pick capped, distributed shots; one actual frame per chosen shot."""
    if not isinstance(limit,int) or isinstance(limit,bool) or limit<1:
        raise ValueError("max_caption_frames must be a positive integer")
    n=len(shots)
    if n<=limit:
        return shots[:]
    if limit==1:
        return [shots[n//2]]
    indices=sorted({round(i*(n-1)/(limit-1)) for i in range(limit)})
    return [shots[i] for i in indices]


def transcribe_audio(video_path: Path, duration: float, *, model_name: str="tiny",
                     language: str|None=None, backend=None) -> list[dict]:
    """Timestamped local ASR; no invented speaker labels or confidence scores."""
    if backend is None:
        try:
            from faster_whisper import WhisperModel
        except ImportError as exc:
            raise RuntimeError("Speech recognition requires: pip install '.[speech]'") from exc
        backend=WhisperModel(model_name,device="auto",compute_type="default")
    segments,info=backend.transcribe(str(video_path),language=language,beam_size=5,vad_filter=True)
    lang=getattr(info,"language",None)
    results=[]
    for seg in segments:
        try:
            raw_start,raw_end=float(seg.start),float(seg.end)
        except (TypeError,ValueError) as exc:
            raise RuntimeError("ASR returned invalid timestamps") from exc
        if not all(map(math.isfinite,(raw_start,raw_end))):
            raise RuntimeError("ASR returned nonfinite timestamps")
        start=max(0.0,raw_start)
        end=min(duration,raw_end)
        text=str(seg.text).strip()
        if start>=end or not text:
            continue
        results.append({"start_seconds":round(start,6),"end_seconds":round(end,6),
                        "text":text,"source":"faster_whisper","model":model_name,
                        "language":lang,"speaker":None,"confidence":None})
    return sorted(results,key=lambda x:(x["start_seconds"],x["end_seconds"]))


def _extract_image(video_path: Path,timestamp: float):
    """Decode one frame using FFmpeg, bounding image size, not buffering the movie."""
    try:
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError("Image captioning requires: pip install '.[vision]'") from exc
    command=["ffmpeg","-hide_banner","-nostdin","-v","error","-ss",
             f"{timestamp:.6f}","-i",str(video_path),"-map","0:v:0","-frames:v","1",
             "-vf","scale=640:-2","-f","image2pipe","-vcodec","png","pipe:1"]
    result=subprocess.run(command,capture_output=True)
    if result.returncode or not result.stdout:
        detail=result.stderr.decode("utf-8",errors="replace")
        raise RuntimeError(f"Could not extract frame at {timestamp:.3f}s: {detail[-400:]}")
    with Image.open(io.BytesIO(result.stdout)) as decoded:
        return decoded.convert("RGB")


def caption_shots(video_path: Path,shots: list[dict], *,
                  model_name: str="Salesforce/blip-image-captioning-base",
                  max_frames: int=120,sample_spacing: float=10.0,
                  backend=None,frame_loader=None) -> list[dict]:
    """Caption stratified frame instants, never imply full-scene observations."""
    selected=plan_frame_samples(shots,max_frames,sample_spacing)
    if not selected:
        return []
    if backend is None:
        try:
            from transformers import pipeline
        except ImportError as exc:
            raise RuntimeError("Image captioning requires: pip install '.[vision]'") from exc
        backend=pipeline("image-to-text",model=model_name,device=-1)
    frame_loader=frame_loader or _extract_image
    output=[]
    for sample in selected:
        timestamp=sample["frame_time_seconds"]
        frame=frame_loader(video_path,timestamp)
        response=backend(frame,max_new_tokens=64)
        caption=response[0].get("generated_text","").strip() if response else ""
        if not caption:
            raise RuntimeError(f"Caption model produced no description at {timestamp:.3f}s")
        output.append({"frame_time_seconds":round(timestamp,6),"shot_id":sample["shot_id"],
                       "description":caption,"source":"vision_language_model",
                       "model":model_name,"confidence":None})
    return output


def align_observations(timeline: list[dict],speech: list[dict],visuals: list[dict]) -> None:
    """Align speech intervals and sampled frame timestamps without extrapolation."""
    for item in timeline:
        a,b=item["start_seconds"],item["end_seconds"]
        item["speech"]=[dict(seg) for seg in speech
                        if max(a,seg["start_seconds"])<min(b,seg["end_seconds"])]
        item["visual_observations"]=[dict(obs) for obs in visuals
                                     if a<=obs["frame_time_seconds"]<b]
        if item["visual_observations"]:
            item["visual_description"]="; ".join(
                obs["description"] for obs in item["visual_observations"])
