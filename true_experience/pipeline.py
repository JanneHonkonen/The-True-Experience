"""FFmpeg-based temporal evidence extraction. No fabricated semantic labels."""
import json
import math
import re
import shutil
import subprocess
from fractions import Fraction
from pathlib import Path
from .srt import load_srt

PTS = re.compile(r"\bpts_time:\s*([0-9.]+)")
SILENCE = re.compile(r"silence_(start|end):\s*([0-9.]+)")

def run(args):
    p = subprocess.run(args,capture_output=True,text=True,encoding="utf-8",errors="replace")
    if p.returncode:
        raise RuntimeError(f"{args[0]} failed ({p.returncode}): {p.stderr[-1200:]}")
    return p

def probe(path):
    data=json.loads(run(["ffprobe","-v","error","-show_streams","-show_format","-of","json",str(path)]).stdout)
    video=next((s for s in data.get("streams",[]) if s.get("codec_type")=="video"),None)
    if not video:
        raise ValueError("No video stream found")
    duration=None
    for value in (data.get("format",{}).get("duration"),video.get("duration")):
        try:
            d=float(value)
            if math.isfinite(d) and d>0:
                duration=round(d,6)
                break
        except (TypeError,ValueError):
            pass
    if duration is None:
        raise ValueError("Valid duration unavailable")
    rate=None
    for field in ("avg_frame_rate","r_frame_rate"):
        try:
            n=float(Fraction(video.get(field,"0")))
            if n>0 and math.isfinite(n):
                rate=round(n,6)
                break
        except (ValueError,ZeroDivisionError):
            pass
    return {"duration_seconds":duration,"frame_rate":rate,
            "width":video.get("width"),"height":video.get("height"),
            "has_audio":any(s.get("codec_type")=="audio" for s in data.get("streams",[]))}

def detect_cuts(path,duration,threshold):
    stderr=run(["ffmpeg","-hide_banner","-nostdin","-v","info","-i",str(path),
                "-map","0:v:0","-vf",f"select=gt(scene\\,{threshold}),showinfo",
                "-an","-sn","-dn","-f","null","-"]).stderr
    cuts=set()
    for line in stderr.splitlines():
        if "showinfo" in line and (m:=PTS.search(line)):
            t=round(float(m.group(1)),6)
            if 0<t<duration:
                cuts.add(t)
    result=[]
    for t in sorted(cuts):
        if not result or t-result[-1]>.05:
            result.append(t)
    return result

def detect_silences(path,duration,db,min_length):
    stderr=run(["ffmpeg","-hide_banner","-nostdin","-v","info","-i",str(path),
                "-map","0:a:0","-af",f"silencedetect=noise={db}dB:d={min_length}",
                "-vn","-sn","-dn","-f","null","-"]).stderr
    start=None
    spans=[]
    for line in stderr.splitlines():
        if m:=SILENCE.search(line):
            t=float(m.group(2))
            if m.group(1)=="start":
                start=max(0.,t)
            elif start is not None:
                end=min(duration,t)
                if end>start:
                    spans.append((round(start,6),round(end,6)))
                start=None
    if start is not None and duration>start:
        spans.append((round(start,6),duration))
    return spans

def format_time(t):
    ms=round(t*1000)
    h,rem=divmod(ms,3600000)
    m,rem=divmod(rem,60000)
    s,ms=divmod(rem,1000)
    return f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}"

def overlap(a,b,x,y):
    return max(0.,min(b,y)-max(a,x))

def create_shots(cuts,duration):
    boundaries=[0.]+sorted({c for c in cuts if 0<c<duration})+[duration]
    return [{"shot_id":i+1,"start_seconds":a,"end_seconds":b,
             "start":format_time(a),"end":format_time(b)}
            for i,(a,b) in enumerate(zip(boundaries,boundaries[1:])) if b>a]

def build_timeline(duration,interval,shots,silences,has_audio,subtitles):
    if not math.isfinite(interval) or interval<=0:
        raise ValueError("Interval must be positive and finite")
    count=math.ceil(duration/interval)
    if count>100000:
        raise ValueError("Too many timeline intervals")
    result=[]
    for i in range(count):
        a,b=min(duration,i*interval),min(duration,(i+1)*interval)
        if b<=a:
            continue
        fraction=min(1.,sum(overlap(a,b,x,y) for x,y in silences)/(b-a))
        state=("no_audio" if not has_audio else "silent" if fraction>=.999
               else "not_detected_silent" if fraction<=.001 else "mixed")
        result.append({
            "start_seconds":round(a,6),"end_seconds":round(b,6),
            "start":format_time(a),"end":format_time(b),
            "shot_ids":[s["shot_id"] for s in shots if overlap(a,b,s["start_seconds"],s["end_seconds"])>0],
            "cut_times_seconds":[s["start_seconds"] for s in shots[1:] if a<=s["start_seconds"]<b],
            "audio":{"state":state,"silence_fraction":round(fraction,4) if has_audio else None},
            "dialogue":[c.copy() for c in subtitles if overlap(a,b,c["start_seconds"],c["end_seconds"])>0],
            "visual_description":None,"emotion":None,"narrative_function":None})
    return result

def analyze(video_path,*,interval=5.,scene_threshold=.3,silence_db=-35.,
            silence_min_duration=.25,subtitles_path=None):
    path=Path(video_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Video file not found: {path}")
    if not math.isfinite(scene_threshold) or not 0<scene_threshold<1:
        raise ValueError("Scene threshold must be in (0,1)")
    if not math.isfinite(silence_db) or not -100<=silence_db<=0:
        raise ValueError("Silence dB must be in [-100,0]")
    if not math.isfinite(silence_min_duration) or silence_min_duration<=0:
        raise ValueError("Minimum silence duration must be positive")
    if not math.isfinite(interval) or interval<=0:
        raise ValueError("Interval must be positive")
    for name in ("ffmpeg","ffprobe"):
        if not shutil.which(name):
            raise RuntimeError(f"{name} is required on PATH")
    info=probe(path)
    duration=info["duration_seconds"]
    cuts=detect_cuts(path,duration,scene_threshold)
    silences=detect_silences(path,duration,silence_db,silence_min_duration) if info["has_audio"] else []
    subtitles=load_srt(subtitles_path) if subtitles_path else []
    shots=create_shots(cuts,duration)
    return {"schema_version":"0.1.0","metadata":{
        "source_file":path.name,**info,"interval_seconds":interval,
        "scene_change_threshold":scene_threshold,"silence_threshold_db":silence_db,
        "subtitle_source":Path(subtitles_path).name if subtitles_path else None,
        "methods":["ffprobe","ffmpeg_scene_change","ffmpeg_silencedetect"]+
                  (["srt_subtitles"] if subtitles_path else []),
        "limitations":["Shot cuts are not semantic scenes.",
                       "Audio activity does not identify speech, sound effects or music.",
                       "Subtitles are externally supplied, not verified transcription.",
                       "Emotion, character identities and narrative function are not inferred."]},
        "shots":shots,
        "silence_intervals":[{"start_seconds":a,"end_seconds":b} for a,b in silences],
        "timeline":build_timeline(duration,interval,shots,silences,info["has_audio"],subtitles)}
