"""Stage 7: visual mood from short frame sequences with a local VLM (Ollama).

Default model: Qwen3.5 9B (``qwen3.5:9b``, about 7 GB at Q4_K_M), chosen to
fit a 12 GB GPU. Each timeline window gets a sequence of frames (default 6)
plus the window's measured signals, and the model fills fixed feeling fields
under a JSON schema. Replaces the earlier BLIP-caption-plus-three-stills
approach. Frames go only to the local Ollama listener on 127.0.0.1.

The model is unloaded when the stage ends (``keep_alive: 0``) so later GPU
stages get the whole card.
"""
from __future__ import annotations

import base64
import io
import json
import math
import subprocess
import sys
import time
import urllib.error
import urllib.request


OLLAMA_CHAT = "http://127.0.0.1:11434/api/chat"
OLLAMA_GENERATE = "http://127.0.0.1:11434/api/generate"
DEFAULT_MODEL = "qwen3.5:9b"
FRAME_WIDTH = 512

LIGHTING = ("bright", "normal", "dim", "dark", "mixed")
EXPRESSIONS = ("neutral", "happy", "sad", "angry", "afraid", "surprised", "disgusted",
               "tense", "unclear")
INTENSITY = ("still", "low", "medium", "high", "frantic")
CAMERA = ("static", "slow_move", "fast_move", "handheld", "unclear")

SCHEMA = {
    "type": "object", "additionalProperties": False,
    "properties": {
        "lighting": {"type": "string", "enum": list(LIGHTING)},
        "colour_mood": {"type": "string", "maxLength": 120},
        "people_visible": {"type": "integer", "minimum": 0, "maximum": 50},
        "facial_expressions": {"type": "array", "maxItems": 6, "items": {
            "type": "object", "additionalProperties": False,
            "properties": {"person": {"type": "string", "maxLength": 20},
                           "expression": {"type": "string", "enum": list(EXPRESSIONS)}},
            "required": ["person", "expression"]}},
        "body_language": {"type": "string", "maxLength": 200},
        "action_intensity": {"type": "string", "enum": list(INTENSITY)},
        "camera": {"type": "string", "enum": list(CAMERA)},
        "mood_words": {"type": "array", "maxItems": 3,
                       "items": {"type": "string", "maxLength": 30}},
        "visual_valence": {"type": "number", "minimum": -1, "maximum": 1},
        "visual_arousal": {"type": "number", "minimum": -1, "maximum": 1},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        "evidence": {"type": "string", "maxLength": 300},
    },
    "required": ["lighting", "colour_mood", "people_visible", "facial_expressions",
                 "body_language", "action_intensity", "camera", "mood_words",
                 "visual_valence", "visual_arousal", "confidence", "evidence"],
}

SYSTEM_PROMPT = """You watch a few seconds of a film as a short sequence of frames, in order.
Describe how this moment LOOKS AND FEELS visually, as a viewer would perceive it.
Rules:
- Use only what is visible in these frames. Do not name actors, characters or the film.
- Frames are in time order; you may describe change or movement between them.
- People are P1, P2 ... by position in the first frame they appear in.
- visual_valence: -1 very unpleasant/grim ... +1 very pleasant/joyful.
- visual_arousal: -1 very calm/still ... +1 very intense/agitated.
- confidence: how clearly the frames show the mood (0..1). Dark or ambiguous frames -> lower.
- "measured_signals" are computed measurements for this window (light, motion, sound);
  use them as context, but trust what you see.
- evidence: one short sentence naming the visible details behind your judgement.
Answer with JSON matching the schema only."""


def _clip(x, lo, hi):
    return max(lo, min(hi, x))


def frame_times(start, end, count):
    """Evenly spaced bin centres inside the window."""
    if count < 1 or end <= start:
        return []
    step = (end - start) / count
    return [round(start + (i + 0.5) * step, 3) for i in range(count)]


def _jpeg(image):
    if image.width > FRAME_WIDTH:
        image = image.resize((FRAME_WIDTH, max(2, round(image.height * FRAME_WIDTH / image.width))))
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=85)
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def extract_window_jpegs(video, start, end, count, width=FRAME_WIDTH):
    """Decode ``count`` frames at the bin centres of [start, end) with ONE FFmpeg call.

    Returns base64 JPEG strings. Much faster than one seek per frame.
    """
    step = (end - start) / count
    cmd = ["ffmpeg", "-hide_banner", "-nostdin", "-v", "error",
           "-ss", f"{start + step / 2:.6f}", "-i", str(video), "-map", "0:v:0",
           "-vf", f"fps={1 / step:.6f},scale={width}:-2", "-frames:v", str(count),
           "-q:v", "3", "-f", "image2pipe", "-vcodec", "mjpeg", "pipe:1"]
    result = subprocess.run(cmd, capture_output=True)
    if result.returncode:
        raise RuntimeError("Frame extraction failed at "
                           f"{start:.2f}s: {result.stderr.decode('utf-8', 'replace')[-300:]}")
    images = [chunk + b"\xff\xd9" for chunk in result.stdout.split(b"\xff\xd9") if chunk.startswith(b"\xff\xd8")]
    if not images:
        raise RuntimeError(f"No frames decoded at {start:.2f}s")
    return [base64.b64encode(img).decode("ascii") for img in images[:count]]


def _measured(signals):
    look = (signals or {}).get("look") or {}
    audio = (signals or {}).get("audio") or {}
    pacing = (signals or {}).get("pacing") or {}
    return {k: v for k, v in {
        "brightness_0_1": look.get("brightness"), "motion_0_1": look.get("motion"),
        "warmth_-1_1": look.get("warmth"), "cuts_per_minute": pacing.get("cuts_per_minute"),
        "loudness_dbfs": audio.get("rms_dbfs"), "sudden_loud_sound": audio.get("startle"),
        "music_mode_estimate": audio.get("mode"),
    }.items() if v is not None}


def ollama_chat(payload, timeout):
    req = urllib.request.Request(OLLAMA_CHAT, data=json.dumps(payload).encode("utf-8"),
                                 headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            body = json.load(response)
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise RuntimeError("Local Ollama unavailable. Install Ollama, run "
                           f"'ollama pull {payload['model']}' and keep it running. {exc}") from exc
    try:
        return json.loads(body["message"]["content"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("Ollama did not return JSON content") from exc


def unload_model(model, timeout=30):
    """Ask Ollama to free VRAM now (best effort)."""
    req = urllib.request.Request(OLLAMA_GENERATE,
                                 data=json.dumps({"model": model, "keep_alive": 0}).encode("utf-8"),
                                 headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout):
            pass
    except (urllib.error.URLError, TimeoutError, OSError):
        pass


def validate(output):
    """Check enums and ranges; clamp numbers. Raises ValueError on bad output."""
    if not isinstance(output, dict):
        raise ValueError("response is not an object")
    for key, allowed in (("lighting", LIGHTING), ("action_intensity", INTENSITY), ("camera", CAMERA)):
        if output.get(key) not in allowed:
            raise ValueError(f"{key} must be one of {allowed}")
    numbers = {}
    for key, lo, hi in (("visual_valence", -1, 1), ("visual_arousal", -1, 1), ("confidence", 0, 1)):
        value = output.get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError(f"{key} must be a number")
        numbers[key] = round(_clip(float(value), lo, hi), 3)
    faces = output.get("facial_expressions")
    if not isinstance(faces, list) or any(
            not isinstance(f, dict) or f.get("expression") not in EXPRESSIONS for f in faces):
        raise ValueError("facial_expressions invalid")
    people = output.get("people_visible")
    if isinstance(people, bool) or not isinstance(people, int) or people < 0:
        raise ValueError("people_visible must be a non-negative integer")
    words = output.get("mood_words")
    if not isinstance(words, list) or any(not isinstance(w, str) for w in words):
        raise ValueError("mood_words must be strings")

    def text(key, limit):
        value = output.get(key)
        if not isinstance(value, str):
            raise ValueError(f"{key} must be text")
        return value.strip()[:limit]

    return {"lighting": output["lighting"], "colour_mood": text("colour_mood", 120),
            "people_visible": people,
            "facial_expressions": [{"person": str(f.get("person", ""))[:20],
                                    "expression": f["expression"]} for f in faces[:6]],
            "body_language": text("body_language", 200),
            "action_intensity": output["action_intensity"], "camera": output["camera"],
            "mood_words": [w.strip()[:30] for w in words[:3]],
            **numbers, "evidence": text("evidence", 300)}


def analyze_visual_mood(video, timeline, *, model=DEFAULT_MODEL, frames_per_window=6,
                        timeout=300, backend=None, frame_loader=None, unload=None,
                        on_window=None):
    """Add ``feeling_signals.visual_mood`` to each window. Returns stage metadata."""
    if not isinstance(model, str) or not model.strip():
        raise ValueError("visual mood model must be a non-empty name")
    if isinstance(frames_per_window, bool) or not isinstance(frames_per_window, int) \
            or not 1 <= frames_per_window <= 8:
        raise ValueError("frames per window must be between 1 and 8")
    backend = backend or ollama_chat
    unload = unload or unload_model
    abstained = 0
    frame_seconds = model_seconds = 0.0
    try:
        for index, window in enumerate(timeline):
            times = frame_times(window["start_seconds"], window["end_seconds"], frames_per_window)
            signals = window.setdefault("feeling_signals", {})
            began = time.perf_counter()
            if frame_loader is None:
                images = extract_window_jpegs(video, window["start_seconds"],
                                              window["end_seconds"], frames_per_window)
            else:
                images = [_jpeg(frame_loader(video, t)) for t in times]
            frame_seconds += time.perf_counter() - began
            payload = {"model": model, "stream": False, "think": False, "format": SCHEMA,
                       "keep_alive": "10m", "options": {"temperature": 0, "num_ctx": 8192},
                       "messages": [
                           {"role": "system", "content": SYSTEM_PROMPT},
                           {"role": "user", "images": images, "content": json.dumps({
                               "window_seconds": [window["start_seconds"], window["end_seconds"]],
                               "frame_times_seconds": times,
                               "measured_signals": _measured(signals)})}]}
            result = None
            began = time.perf_counter()
            for attempt in range(2):
                try:
                    result = validate(backend(payload, timeout))
                    break
                except ValueError as exc:
                    error = str(exc)
                    payload["messages"] = payload["messages"][:2] + [
                        {"role": "user", "content": f"Invalid answer ({error}). Return JSON "
                                                    "matching the schema exactly."}]
            model_seconds += time.perf_counter() - began
            if result is None:
                abstained += 1
                print(f"WARNING: visual mood abstained at {window['start_seconds']:.1f}s ({error})",
                      file=sys.stderr)
                signals["visual_mood"] = None
            else:
                signals["visual_mood"] = {**result, "model": model, "frame_times_seconds": times}
            if on_window:
                on_window(index, window)
            print(f"Visual mood {index + 1}/{len(timeline)}", file=sys.stderr)
    finally:
        unload(model)
    return {"model": model, "frames_per_window": frames_per_window,
            "frame_width": FRAME_WIDTH, "abstained_windows": abstained,
            "frame_extraction_seconds": round(frame_seconds, 2),
            "model_seconds": round(model_seconds, 2)}
