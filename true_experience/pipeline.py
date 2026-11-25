"""FFmpeg-based temporal evidence extraction. No fabricated semantic labels."""
import json
import math
import re
import shutil
import subprocess
from fractions import Fraction
from pathlib import Path
from .srt import load_srt
from .perception import align_observations, caption_shots, transcribe_audio
from .sampling import describe_coverage

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
            silence_min_duration=.25,subtitles_path=None,
            transcribe=False,whisper_model="tiny",language=None,
            caption_frames=False,caption_model="Salesforce/blip-image-captioning-base",
            max_caption_frames=120,caption_spacing=10.0,feeling_signals=False,
            visual_mood_model=None,visual_mood_frames=6,
            voice=False,voice_emotion_model="superb/wav2vec2-base-superb-er",
            characters=False,character_roles_model=None,character_fps=1.0,
            separate_audio=False,separation_model="htdemucs",
            voice_speaker_model="microsoft/wavlm-base-plus-sv",
            scenes=False,story_model=None,narration=True):
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
    if not isinstance(max_caption_frames,int) or isinstance(max_caption_frames,bool) or max_caption_frames<1:
        raise ValueError("max_caption_frames must be a positive integer")
    if (not isinstance(caption_spacing,(int,float)) or isinstance(caption_spacing,bool) or
        not math.isfinite(caption_spacing) or caption_spacing<=0):
        raise ValueError("caption_spacing must be positive and finite")
    if transcribe and (not isinstance(whisper_model,str) or not whisper_model.strip()):
        raise ValueError("whisper_model must be a nonempty model ID")
    if caption_frames and (not isinstance(caption_model,str) or not caption_model.strip()):
        raise ValueError("caption_model must be a nonempty model ID")
    for name in ("ffmpeg","ffprobe"):
        if not shutil.which(name):
            raise RuntimeError(f"{name} is required on PATH")
    import time
    timing={}
    clock=[time.perf_counter()]
    def lap(name):
        now=time.perf_counter()
        timing[name]=round(now-clock[0],2)
        clock[0]=now
    started=clock[0]
    info=probe(path)
    duration=info["duration_seconds"]
    cuts=detect_cuts(path,duration,scene_threshold)
    silences=detect_silences(path,duration,silence_db,silence_min_duration) if info["has_audio"] else []
    subtitles=load_srt(subtitles_path) if subtitles_path else []
    shots=create_shots(cuts,duration)
    lap("probe_cuts_silence")
    if transcribe and not info["has_audio"]:
        raise ValueError("Cannot transcribe: no audio stream in video")
    separation=None
    if separate_audio and info["has_audio"]:
        from .separation import separate_audio as run_separation
        separation=run_separation(path,model=separation_model)
        lap("separation")
    vocals=separation["vocals_path"] if separation else path
    music=separation["music_path"] if separation else None
    speech=transcribe_audio(vocals,duration,model_name=whisper_model,language=language) if transcribe else []
    if transcribe: lap("speech")
    voice_settings=None
    if voice and speech:
        from .voice import analyze_voice
        voice_settings=analyze_voice(vocals,speech,emotion_model=voice_emotion_model or None,
                                     speaker_model=voice_speaker_model or None)
        lap("voice")
    visuals=caption_shots(path,shots,model_name=caption_model,max_frames=max_caption_frames,sample_spacing=caption_spacing) if caption_frames else []
    if caption_frames: lap("frame_captions")
    timeline=build_timeline(duration,interval,shots,silences,info["has_audio"],subtitles)
    character_result=None
    if characters or character_roles_model:
        if not math.isfinite(character_fps) or not 0.1<=character_fps<=5:
            raise ValueError("character_fps must be between 0.1 and 5")
        from .characters import analyze_characters
        # Before align_observations: windows copy speech segments, which gain "character".
        character_result=analyze_characters(path,info,shots,timeline,speech,fps=character_fps)
        lap("characters")
    align_observations(timeline,speech,visuals)
    signal_settings=None
    if feeling_signals:
        from .signals import compute_signals
        signal_settings=compute_signals(path,info,shots,timeline,music_path=music)
        lap("feeling_signals")
    visual_mood_settings=None
    if feeling_signals and visual_mood_model:
        from .visual_mood import analyze_visual_mood
        visual_mood_settings=analyze_visual_mood(path,timeline,model=visual_mood_model,
                                                 frames_per_window=visual_mood_frames)
        lap("visual_mood")
    character_roles=None
    if character_result and character_roles_model:
        from .characterhood import interpret_characters
        character_roles=interpret_characters(path,info,character_result["characters"],speech,timeline,
                                             model=character_roles_model)
        lap("character_roles")
    feeling_curve=None
    if feeling_signals:
        from .fusion import fuse_timeline
        feeling_curve=fuse_timeline(timeline)
        lap("fusion")
    scene_list=None; scene_settings=None; story=None; story_settings=None
    if scenes or story_model:
        from .scenes import detect_scenes
        scene_list,scene_settings=detect_scenes(
            path,info,shots,timeline,speech,silences,
            characters=character_result["characters"] if character_result else None,
            character_tracks=character_result["character_tracks"] if character_result else None)
        lap("scenes")
    if scene_list and story_model:
        from .story import interpret_story
        story,story_settings=interpret_story(
            path,scene_list,timeline,speech,
            character_result["characters"] if character_result else [],feeling_curve,
            model=story_model)
        lap("story")
    naming_settings=story_settings.pop("naming",None) if story_settings else None
    if character_result:
        from .naming import add_profiles,name_characters
        if naming_settings is None:
            # No story pass: names from self-introductions in the transcript only.
            _,naming_settings=name_characters(character_result["characters"],speech)
        add_profiles(character_result["characters"])
    narration_list=None; narration_settings=None
    if scene_list and story_model and narration:
        from .narration import narrate
        narration_list,narration_settings=narrate(
            path,scene_list,speech,character_result["characters"] if character_result else [],
            timeline,story,model=story_model)
        lap("narration")
    timing["total"]=round(time.perf_counter()-started,2)
    timing["seconds_per_video_minute"]=round(timing["total"]/(duration/60),2)
    return {"schema_version":"0.3.0","metadata":{
        "source_file":path.name,**info,"interval_seconds":interval,
        "scene_change_threshold":scene_threshold,"silence_threshold_db":silence_db,
        "subtitle_source":Path(subtitles_path).name if subtitles_path else None,
        "methods":(["ffprobe","ffmpeg_scene_change"]+
                   (["ffmpeg_silencedetect"] if info["has_audio"] else [])+
                   (["srt_subtitles"] if subtitles_path else [])+
                   (["faster_whisper"] if transcribe else [])+
                   (["blip_frame_captioning"] if caption_frames else [])+
                   (["cpu_feeling_signals"] if feeling_signals else [])+
                   (["ollama_visual_mood"] if visual_mood_settings else [])+
                   (["voice_prosody_and_emotion"] if voice_settings else [])+
                   (["demucs_separation"] if separation else [])+
                   (["character_tracking"] if character_result else [])+
                   (["scene_grouping"] if scene_list is not None else [])+
                   (["ollama_story_reading"] if story_settings else [])+
                   (["ollama_narration"] if narration_settings else [])+
                   (["ollama_character_roles"] if character_roles else [])),
        "feeling_signals":signal_settings,
        "visual_mood":visual_mood_settings,
        "voice":voice_settings,
        "separation":separation["settings"] if separation else None,
        "characters":character_result["settings"] if character_result else None,
        "character_roles":character_roles,
        "scenes":scene_settings,
        "story":story_settings,
        "naming":naming_settings,
        "narration":narration_settings,
        "timing_seconds":timing,
        "perception":{"speech_enabled":transcribe,
                      "speech_model":whisper_model if transcribe else None,
                      "captioning_enabled":caption_frames,
                      "caption_model":caption_model if caption_frames else None,
                      "caption_sample_count":len(visuals),
                      "caption_frame_cap":max_caption_frames if caption_frames else None,
                      "visual_sampling": ({"target_spacing_seconds":caption_spacing,
                          **describe_coverage(shots,visuals)} if caption_frames else None)},
        "limitations":["Shot cuts are not semantic scenes.",
                       "Audio activity does not identify speech, sound effects or music.",
                       "Subtitles are externally supplied, not verified transcription.",
                       "Model transcripts and captions may be wrong or hallucinated.",
                       "Captured frames are samples, not descriptions of whole scenes.",
                       "Speaker identities, emotion, and narrative function are not inferred."]+
                      (["Feeling signals are measurements (light, motion, loudness, tempo, key), not emotions.",
                        "Music features are computed on the full audio mix, not a separated music track.",
                        "Fused feeling (valence, arousal, tension, emotion label) is a heuristic estimate "
                        "from measured signals, not yet validated against human ratings."]
                       if feeling_signals else [])+
                      (["Characters are grouped within this video only. Names come only from the film's "
                        "dialogue and on-screen text, never from face recognition of actors.",
                        "Physical descriptions are apparent (as seen on screen) and may be wrong.",
                        "Face and clothing grouping can split one person or merge look-alikes "
                        "(same uniform, profile faces, dark scenes).",
                        "Character roles are model judgements of the fictional role from visible cues "
                        "and dialogue, not facts."] if character_result else [])},
        "shots":shots,
        "silence_intervals":[{"start_seconds":a,"end_seconds":b} for a,b in silences],
        "speech_segments":speech,"visual_observations":visuals,"timeline":timeline,
        **({"feeling_curve":feeling_curve} if feeling_curve is not None else {}),
        **({"scenes":scene_list} if scene_list is not None else {}),
        **({"story":story} if story_settings else {}),
        **({"narration":narration_list} if narration_list is not None else {}),
        **({"characters":character_result["characters"],
            "character_tracks":character_result["character_tracks"]} if character_result else {})}
