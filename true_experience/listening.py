"""Stage 5b: listen to the music and the sound effects (CLAP, GPU if available).

Tempo, key and loudness say little about what music *feels* like. Viewers
hear a heroic brass theme, a tense string ostinato, a rock track, a choir, a
sudden silence; and effects such as explosions, gunfire, engines or a crowd
carry as much of the mood as the picture.

This stage scores short pieces of the soundtrack against text descriptions
with CLAP (``laion/clap-htsat-unfused``, Apache-2.0, about 600 MB, about 1 GB of
VRAM, freed after the stage):

- **music** per 5 s window: style, instruments, mood, and whether music plays,
- **sounds** per half window (2.5 s): the effects heard, with a time,
- **moments** over the whole film: music starting and stopping, builds,
  drops and hits, from the music stem's loudness and the CLAP scores.

It listens to the Demucs music stem (drums + bass + other, which also holds
the effects) when separation ran, so dialogue does not mask the score.
Otherwise it listens to the full mix. Labels are zero-shot matches against
text, so they are reported with scores and kept only above a margin.
"""
from __future__ import annotations

import sys
import time

DEFAULT_MODEL = "laion/clap-htsat-unfused"
RATE = 48000

STYLES = {
    "orchestral film score": "orchestral film score",
    "epic trailer music": "epic trailer music with choir and big drums",
    "rock": "rock music with electric guitars and drums",
    "heavy metal": "heavy metal music",
    "electronic": "electronic music with synthesizers",
    "hip hop": "hip hop beat",
    "pop": "pop song",
    "jazz": "jazz",
    "solo piano": "solo piano",
    "ambient drone": "dark ambient drone",
    "acoustic folk": "acoustic folk guitar",
    "classical strings": "classical string quartet",
}
INSTRUMENTS = {
    "electric guitar": "electric guitar", "drums": "loud drums", "brass": "brass section",
    "strings": "string section", "piano": "piano", "choir": "choir singing",
    "synthesizer": "synthesizer", "bass": "deep bass", "percussion hits": "big percussion hits",
    "acoustic guitar": "acoustic guitar",
}
MOODS = {
    "heroic": ("heroic, triumphant music", 0.6, 0.6),
    "tense": ("tense, suspenseful music", -0.4, 0.6),
    "ominous": ("ominous, dark, threatening music", -0.6, 0.4),
    "sad": ("sad, melancholic music", -0.6, -0.4),
    "playful": ("playful, light-hearted music", 0.6, 0.3),
    "calm": ("calm, peaceful music", 0.3, -0.5),
    "aggressive": ("aggressive, energetic music", -0.2, 0.8),
    "romantic": ("romantic, tender music", 0.5, -0.1),
    "mysterious": ("mysterious music", -0.2, 0.0),
    "exciting": ("exciting, driving action music", 0.4, 0.8),
}
SOUNDS = {
    "explosion": "an explosion", "gunfire": "gunfire", "car engine": "a car engine revving",
    "jet": "a jet aircraft flying past", "helicopter": "a helicopter", "crowd cheering": "a crowd cheering",
    "crowd chatter": "people talking in a crowd", "camera shutters": "camera shutter clicks",
    "metal clanking": "metal clanking and hammering", "whoosh": "a whoosh", "impact": "a heavy impact hit",
    "glass shattering": "glass shattering", "siren": "an alarm or siren", "footsteps": "footsteps",
    "wind": "wind blowing", "machinery": "electrical machinery humming and sparking",
    "rumble": "a deep rumble",
}
NO_MUSIC = "no music, only sound effects or silence"
MUSIC = "music playing"
QUIET = "silence"

MUSIC_MIN = 0.55       # share of "music playing" against "no music" to call it music
LABEL_MARGIN = 1.6     # a label must score this many times the uniform share
SOUND_MARGIN = 2.5
SILENT_DB = -45.0


def _np():
    try:
        import numpy
    except ImportError as exc:
        raise RuntimeError("Listening requires NumPy: python -m pip install numpy") from exc
    return numpy


def load_clap(model=DEFAULT_MODEL):
    """Returns embed(texts or audio list) helpers on the GPU when available."""
    try:
        import torch
        from transformers import ClapModel, ClapProcessor
    except ImportError as exc:
        raise RuntimeError("Listening needs transformers and torch (python -m pip install -r requirements.txt). "
                           f"Import error: {exc}") from exc
    np = _np()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    net = ClapModel.from_pretrained(model).to(device).eval()
    proc = ClapProcessor.from_pretrained(model)

    def text(labels):
        with torch.no_grad():
            inputs = proc(text=list(labels), return_tensors="pt", padding=True).to(device)
            emb = net.get_text_features(**inputs)
        return torch.nn.functional.normalize(emb, dim=-1).cpu().numpy()

    def audio(clips):
        with torch.no_grad():
            arrays = [np.asarray(c, dtype=np.float32) for c in clips]
            try:  # newer transformers call it "audio", older ones "audios"
                inputs = proc(audio=arrays, sampling_rate=RATE, return_tensors="pt")
            except (TypeError, ValueError):
                inputs = proc(audios=arrays, sampling_rate=RATE, return_tensors="pt")
            inputs = inputs.to(device)
            emb = net.get_audio_features(**inputs)
        return torch.nn.functional.normalize(emb, dim=-1).cpu().numpy()

    def close():
        nonlocal net
        net = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    scale = float(net.logit_scale_a.exp().item()) if hasattr(net, "logit_scale_a") else 100.0
    return {"text": text, "audio": audio, "close": close, "device": device, "scale": scale}


def _softmax(x, scale):
    np = _np()
    z = np.asarray(x, dtype=np.float64) * scale
    z = np.exp(z - z.max())
    return z / z.sum()


def _top(names, probs, margin, k):
    uniform = 1.0 / len(names)
    ranked = sorted(zip(names, probs), key=lambda kv: -kv[1])
    return [{"label": n, "score": round(float(p), 3)} for n, p in ranked[:k] if p >= margin * uniform]


class Listener:
    def __init__(self, backend):
        self.backend = backend
        self.scale = backend.get("scale", 100.0)
        t = backend["text"]
        self.style = t([STYLES[k] for k in STYLES])
        self.instr = t([INSTRUMENTS[k] for k in INSTRUMENTS])
        self.mood = t([MOODS[k][0] for k in MOODS])
        self.sound = t([SOUNDS[k] for k in SOUNDS] + [QUIET])
        self.music = t([MUSIC, NO_MUSIC])

    def window(self, clip, halves, start, level_db):
        """Music description of one window and the sounds in each half."""
        a, *parts = self.backend["audio"]([clip] + halves)
        music_p = float(_softmax(self.music @ a, self.scale)[0])
        present = level_db is not None and level_db > SILENT_DB and music_p >= MUSIC_MIN
        music = {"present": present, "music_score": round(music_p, 3)}
        if present:
            mood = _softmax(self.mood @ a, self.scale)
            music.update({
                "style": _top(list(STYLES), _softmax(self.style @ a, self.scale), LABEL_MARGIN, 2),
                "instruments": _top(list(INSTRUMENTS), _softmax(self.instr @ a, self.scale), LABEL_MARGIN, 3),
                "mood": _top(list(MOODS), mood, LABEL_MARGIN, 2),
                "mood_valence": round(float(sum(p * MOODS[k][1] for k, p in zip(MOODS, mood))), 3),
                "mood_arousal": round(float(sum(p * MOODS[k][2] for k, p in zip(MOODS, mood))), 3),
                "mood_confidence": round(float(mood.max()), 3)})
        sounds = []
        names = list(SOUNDS) + ["silence"]
        half = (len(clip) / RATE) / max(1, len(halves))
        for i, emb in enumerate(parts):
            for hit in _top(names, _softmax(self.sound @ emb, self.scale), SOUND_MARGIN, 2):
                if hit["label"] != "silence":
                    sounds.append({**hit, "t": round(start + i * half, 2)})
        return music, sounds


def music_moments(timeline, min_jump_db=8.0):
    """Music starts, stops, builds, drops and hits over the film, from the stem level and CLAP."""
    moments = []
    prev = None
    levels = []
    for w in timeline:
        signals = w.get("feeling_signals") or {}
        stem = signals.get("music") or signals.get("audio") or {}
        heard = ((signals.get("listening") or {}).get("music") or {}).get("present")
        level = stem.get("rms_dbfs")
        t = w["start_seconds"]
        if prev is not None and heard is not None and prev["heard"] is not None and heard != prev["heard"]:
            moments.append({"t": t, "kind": "music_starts" if heard else "music_stops"})
        if prev is not None and level is not None and prev["level"] is not None and heard and prev["heard"]:
            jump = level - prev["level"]
            if jump >= min_jump_db:
                moments.append({"t": t, "kind": "music_surges", "db": round(jump, 1)})
            elif jump <= -min_jump_db and heard:
                moments.append({"t": t, "kind": "music_drops", "db": round(jump, 1)})
        for hit in stem.get("startle_times_seconds") or []:
            moments.append({"t": hit, "kind": "hit"})
        levels.append((t, level if heard else None))
        prev = {"heard": heard, "level": level}
    # A build: three or more windows in a row getting louder by 6 dB or more in total.
    run = []
    for t, level in levels + [(None, None)]:
        if level is not None and run and level > run[-1][1] + 0.5:
            run.append((t, level))
            continue
        if len(run) >= 3 and run[-1][1] - run[0][1] >= 6.0:
            moments.append({"t": run[0][0], "kind": "music_builds", "until": run[-1][0],
                            "db": round(run[-1][1] - run[0][1], 1)})
        run = [(t, level)] if level is not None else []
    return sorted(moments, key=lambda m: m["t"])


def describe_music(listening):
    """A few words for a window's music: 'heroic, exciting rock; electric guitar, drums'."""
    m = (listening or {}).get("music") or {}
    if not m:
        return None
    if not m.get("present"):
        return "no music"
    mood = ", ".join(x["label"] for x in m.get("mood") or [])
    style = " / ".join(x["label"] for x in m.get("style") or [])
    head = " ".join(x for x in (mood, style) if x) or "music"
    instr = ", ".join(x["label"] for x in m.get("instruments") or [])
    return f"{head}; {instr}" if instr else head


def listen(source, timeline, *, model=DEFAULT_MODEL, backend=None, reader=None):
    """Add ``feeling_signals.listening`` to every window and return (moments, metadata)."""
    from .signals import stream_audio_windows
    began = time.perf_counter()
    backend = backend or load_clap(model)
    np = _np()
    bounds = [(w["start_seconds"], w["end_seconds"]) for w in timeline]
    windows = reader(bounds) if reader else stream_audio_windows(source, bounds, rate=RATE)
    try:
        listener = Listener(backend)
        for w, clip in zip(timeline, windows):
            clip = np.asarray(clip, dtype=np.float32)
            signals = w.setdefault("feeling_signals", {})
            level = (signals.get("music") or signals.get("audio") or {}).get("rms_dbfs")
            if len(clip) < RATE // 2:
                signals["listening"] = None
                continue
            mid = len(clip) // 2
            music, sounds = listener.window(clip, [clip[:mid], clip[mid:]], w["start_seconds"], level)
            signals["listening"] = {"music": music, "sounds": sounds, "description": None}
            signals["listening"]["description"] = describe_music(signals["listening"])
    finally:
        if hasattr(windows, "close"):
            windows.close()
        close = backend.get("close")
        if close:
            close()
    moments = music_moments(timeline)
    print(f"Listening: {len(timeline)} windows, {len(moments)} music moments", file=sys.stderr)
    return moments, {"model": model, "device": backend.get("device"), "rate": RATE,
                     "source": str(source.name) if hasattr(source, "name") else None,
                     "music_min": MUSIC_MIN, "label_margin": LABEL_MARGIN, "sound_margin": SOUND_MARGIN,
                     "seconds": round(time.perf_counter() - began, 2)}
