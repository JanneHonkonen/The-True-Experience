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

def write_outputs(result,path,details=True):
    """The compact experience file, the full details and the narration text. Returns the paths."""
    from .compact import experience
    from .narration import render_text
    dump = lambda data: json.dumps(data,indent=2,ensure_ascii=False,allow_nan=False)+"\n"
    saved = []
    view = experience(result)
    if details:
        detail_path = path.with_name(path.stem+"_details"+path.suffix)
        detail_path.write_text(dump(result),encoding="utf-8")
        view["about"]["details_file"] = detail_path.name
        saved.append(detail_path)
    path.write_text(dump(view),encoding="utf-8")
    saved.insert(0,path)
    if result.get("narration"):
        text_path = path.with_name(path.stem+"_narration.md")
        text_path.write_text(render_text(Path(result["metadata"]["source_file"]).stem,
                                         result["narration"],result.get("story")),encoding="utf-8")
        saved.append(text_path)
    return saved

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
    p.add_argument("--feeling-signals",action="store_true",
                   help="Measure pacing, look, loudness, tempo and key per interval and fuse them into an emotional timeline (CPU, needs NumPy)")
    p.add_argument("--visual-mood-model",metavar="MODEL",
                   help="Local Ollama vision model for visual mood, e.g. qwen3.5:9b; implies --feeling-signals")
    p.add_argument("--visual-mood-frames",type=int,default=6,
                   help="Frames per interval sent to the visual mood model (1-8)")
    p.add_argument("--voice",action="store_true",
                   help="Analyze how lines are spoken (pitch, rate, loudness, voice groups, voice emotion); "
                        "implies --transcribe and --feeling-signals")
    p.add_argument("--voice-emotion-model",default="superb/wav2vec2-base-superb-er",
                   help="Hugging Face audio-classification model for voice emotion; 'none' to skip")
    p.add_argument("--characters",action="store_true",
                   help="Find people, follow them by face and by clothing, and group them into characters C1, C2 ... "
                        "(CPU, OpenCV; downloads about 180 MB of models once)")
    p.add_argument("--character-roles",metavar="MODEL",
                   help="Local Ollama vision model that judges each character's role (protagonist, villain, "
                        "journalist ...), e.g. qwen3.5:9b; implies --characters")
    p.add_argument("--character-fps",type=float,default=1.0,
                   help="Frames per second sampled for character tracking (0.1-5)")
    p.add_argument("--scenes",action="store_true",
                   help="Group shots into scenes by look, people, dialogue and silence (CPU)")
    p.add_argument("--story",metavar="MODEL",
                   help="Local Ollama vision model that reads each scene and the whole story arc, "
                        "e.g. qwen3.5:9b; implies --scenes")
    p.add_argument("--listen",action="store_true",
                   help="Listen to the music (style, instruments, mood) and sound effects (explosions, engines, "
                        "crowds ...) with CLAP (GPU if available, about 600 MB once); implies --feeling-signals")
    p.add_argument("--listen-model",default="laion/clap-htsat-unfused",help="Hugging Face CLAP model")
    p.add_argument("--narration",action=argparse.BooleanOptionalAction,default=True,
                   help="With --story, write timed passages per scene that describe what is seen, heard and "
                        "felt, quoting the real lines (on by default; --no-narration skips it)")
    p.add_argument("--details",action=argparse.BooleanOptionalAction,default=True,
                   help="Also write the full analysis with every measurement to <output>_details.json "
                        "(on by default; the main output is the compact experience file)")
    p.add_argument("--speaker-model",default="microsoft/wavlm-base-plus-sv",
                   help="Hugging Face speaker-embedding model for voice groups; 'none' uses rough MFCC fingerprints")
    p.add_argument("--separate-audio",action=argparse.BooleanOptionalAction,default=None,
                   help="Split dialogue from music with Demucs (GPU if available) before speech, voice "
                        "and music analysis. On by default with --voice; --no-separate-audio turns it off")
    p.add_argument("--separation-model",default="htdemucs",help="Demucs model name")
    a = p.parse_args(argv)
    try:
        result = analyze(a.video,interval=a.interval,scene_threshold=a.scene_threshold,
                         silence_db=a.silence_db,silence_min_duration=a.silence_min_duration,
                         subtitles_path=a.subtitles,transcribe=a.transcribe or a.voice,
                         whisper_model=a.whisper_model,language=a.language,
                         caption_frames=a.caption_frames,caption_model=a.caption_model,
                         max_caption_frames=a.max_caption_frames,caption_spacing=a.caption_spacing,
                         feeling_signals=a.feeling_signals or bool(a.visual_mood_model) or a.voice or a.listen,
                         listen=a.listen,listen_model=a.listen_model,
                         visual_mood_model=a.visual_mood_model,visual_mood_frames=a.visual_mood_frames,
                         voice=a.voice,
                         characters=a.characters,character_roles_model=a.character_roles,
                         character_fps=a.character_fps,
                         separate_audio=a.voice if a.separate_audio is None else a.separate_audio,
                         separation_model=a.separation_model,
                         scenes=a.scenes,story_model=a.story,narration=a.narration,
                         voice_speaker_model=None if a.speaker_model.lower()=="none" else a.speaker_model,
                         voice_emotion_model=None if a.voice_emotion_model.lower()=="none" else a.voice_emotion_model)
        path = Path(a.output)
        path.parent.mkdir(parents=True,exist_ok=True)
        saved = write_outputs(result,path,details=a.details)
        print(f"Saved {len(result['shots'])} shots and {len(result['timeline'])} intervals: "
              + ", ".join(str(p) for p in saved))
        timing=result["metadata"].get("timing_seconds") or {}
        if timing:
            print("Timing (s): "+", ".join(f"{k}={v}" for k,v in timing.items()))
        return 0
    except (ValueError,OSError,RuntimeError) as exc:
        print(f"Analysis failed: {exc}",file=sys.stderr)
        return 1

if __name__ == "__main__":
    raise SystemExit(main())
