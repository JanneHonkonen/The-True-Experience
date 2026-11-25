"""Fusion v1: turn measured feeling signals into an emotional timeline.

Stage 8 in docs/FEELING_PIPELINE.md. Each modality gives a *reading*
(valence -1..1, arousal -1..1, confidence 0..1, and the features it used).
Readings are combined with the specification's priority weights, carried over
between windows so feeling lingers, and checked for disagreement (irony,
contrast). All numbers here are transparent heuristics, not trained values;
they are meant to be calibrated against annotated clips.
"""
from __future__ import annotations

import math

VERSION = "fusion_v1_signals"

# Priority from the specification: dialogue > music > visual tone.
# "visual" is the VLM's reading of the frames; "look" is the low-level light/colour
# measurement. Together they carry the visual-tone weight.
WEIGHTS = {"voice": 1.0, "music": 0.8, "visual": 0.6, "look": 0.4, "sound": 0.5, "pacing": 0.4}
CONFLICT_MIN_CONFIDENCE = 0.3
CONFLICT_VALENCE_GAP = 0.8
CARRY_HALF_LIFE_SECONDS = 5.0


def _clip(x, lo=-1.0, hi=1.0):
    return max(lo, min(hi, x))


def _r(x):
    return None if x is None else round(float(x), 3)


def _reading(valence, arousal, confidence, basis, **extra):
    return {"valence": _r(valence), "arousal": _r(arousal),
            "confidence": _r(_clip(confidence, 0.0, 1.0)), "basis": basis, **extra}


# ----------------------------------------------------------- readings ----

# Viewers adapt: a trailer that is always fast and loud is not equally intense
# everywhere. Pacing and loudness are judged partly against the film's own
# typical level (median and spread), once there are enough windows.
RELATIVE_SHARE = 0.6
MIN_WINDOWS_FOR_BASELINE = 6


def baseline(values):
    """Median and half the 10-90 percentile spread, or None with too little data."""
    vals = sorted(v for v in values if v is not None)
    if len(vals) < MIN_WINDOWS_FOR_BASELINE:
        return None
    def pct(q):
        i = q * (len(vals) - 1)
        lo, hi = int(math.floor(i)), int(math.ceil(i))
        return vals[lo] + (vals[hi] - vals[lo]) * (i - lo)
    return {"median": pct(0.5), "spread": max((pct(0.9) - pct(0.1)) / 2, 1e-6)}


def _blend(absolute, value, base):
    if base is None:
        return absolute
    relative = _clip((value - base["median"]) / base["spread"])
    return _clip((1 - RELATIVE_SHARE) * absolute + RELATIVE_SHARE * relative)


def read_pacing(pacing, base=None):
    cpm = (pacing or {}).get("cuts_per_minute")
    if cpm is None:
        return None
    # 10 cuts/min is neutral; 30+ is frantic; 0 is calm.
    arousal = _blend(_clip((cpm - 10.0) / 20.0), cpm, base)
    return _reading(None, arousal, 0.5, ["cuts_per_minute"] + (["film_baseline"] if base else []))


def read_look(look):
    if not look:
        return None
    b, c = look["brightness"], look["contrast"]
    s, w, m = look["saturation"], look["warmth"], look.get("motion")
    valence = _clip(0.8 * (b - 0.4) + 0.6 * w + 0.4 * (s - 0.3))
    arousal = _clip((c - 0.2) * 1.5 + ((m * 8.0 - 0.4) if m is not None else 0.0))
    basis = ["brightness", "contrast", "saturation", "warmth"] + (["motion"] if m is not None else [])
    return _reading(valence, arousal, 0.35 if m is not None else 0.25, basis,
                    darkness=_r(_clip((0.35 - b) / 0.35, 0.0, 1.0)))


def read_voice(speech, start, end, loud_base=None):
    """How lines are delivered in this window: prosody, plus voice emotion if measured."""
    parts = []
    for seg in speech or []:
        voice = seg.get("voice")
        overlap = min(end, seg["end_seconds"]) - max(start, seg["start_seconds"])
        if voice and overlap > 0:
            parts.append((overlap, voice))
    if not parts:
        return None
    total = sum(w for w, _ in parts)

    def avg(key, sub=None):
        vals = [(w, (v.get(sub) or {}).get(key) if sub else v.get(key)) for w, v in parts]
        vals = [(w, x) for w, x in vals if x is not None]
        return sum(w * x for w, x in vals) / sum(w for w, _ in vals) if vals else None

    variability, rate, loud = avg("pitch_variability_semitones"), avg("words_per_second"), avg("loudness_dbfs")
    terms = []
    if variability is not None:
        terms.append(_clip((variability - 3.0) / 2.0))
    if rate is not None:
        terms.append(_clip((rate - 2.5) / 1.5))
    if loud is not None:
        terms.append(_blend(_clip((loud + 30.0) / 15.0), loud, loud_base))
    arousal = sum(terms) / len(terms) if terms else None
    emotion_v, emotion_a, emotion_score = avg("valence", "emotion"), avg("arousal", "emotion"), avg("score", "emotion")
    coverage = min(1.0, total / max(1e-6, (end - start) * 0.5))
    if emotion_v is not None:
        arousal = emotion_a if arousal is None else 0.5 * arousal + 0.5 * emotion_a
        confidence = (0.3 + 0.5 * (emotion_score or 0.0)) * coverage
    else:
        confidence = 0.35 * coverage
    if arousal is None and emotion_v is None:
        return None
    labels = sorted({(v.get("emotion") or {}).get("label") for _, v in parts} - {None})
    speakers = sorted({seg.get("speaker") for seg in speech or [] if seg.get("speaker")})
    return _reading(emotion_v, arousal, confidence,
                    [b for b, x in (("voice_emotion_model", emotion_v), ("pitch_variability", variability),
                                    ("speaking_rate", rate), ("voice_loudness", loud)) if x is not None],
                    emotions=labels, speakers=speakers, speech_seconds=_r(total))


def read_visual(mood):
    """The visual-mood model's own valence/arousal, scaled by its stated confidence."""
    if not mood:
        return None
    return _reading(mood["visual_valence"], mood["visual_arousal"], 0.8 * mood["confidence"],
                    ["visual_mood_model"], lighting=mood.get("lighting"),
                    mood_words=mood.get("mood_words", []))


def read_music(audio, separated=False, listening=None):
    """Major/minor colour and rhythmic drive; trusted more from a separated music stem.

    When the listening stage heard the music (CLAP), its mood (heroic, ominous,
    sad ...) leads and the measured mode and tempo only colour it.
    """
    heard = (listening or {}).get("music") or {}
    if heard and not heard.get("present"):
        return None  # the listening model hears no music here
    if heard.get("mood_valence") is not None:
        base = read_music(audio, separated) if audio else None
        mv, ma = heard["mood_valence"], heard["mood_arousal"]
        if base:
            mv = 0.75 * mv + 0.25 * (base["valence"] if base["valence"] is not None else mv)
            ma = 0.75 * ma + 0.25 * (base["arousal"] if base["arousal"] is not None else ma)
        moods = [m["label"] for m in heard.get("mood") or []]
        return _reading(mv, ma, 0.4 + 0.5 * (heard.get("mood_confidence") or 0.0),
                        ["listening_mood"] + (base["basis"] if base else []),
                        mode=(base or {}).get("mode"), moods=moods,
                        source="music_stem" if separated else "full_mix")
    if not audio:
        return None
    mode, mode_conf = audio.get("mode"), audio.get("mode_confidence") or 0.0
    onsets = audio.get("onset_rate_per_second") or 0.0
    tempo = audio.get("tempo_bpm")
    valence = None
    if mode in ("major", "minor") and mode_conf > 0:
        valence = (0.3 + 0.5 * mode_conf) * (1 if mode == "major" else -1)
    drive = _clip((onsets - 1.0) / 3.0)
    if tempo is not None:
        drive = _clip(0.5 * drive + 0.5 * (tempo - 100.0) / 60.0)
    if valence is None and tempo is None:
        return None
    # On the full mix, dialogue and effects make key and tempo flip between
    # windows, so music counts for little there. The separated stem is cleaner.
    factor = 0.7 if separated else 0.35
    if separated and (audio.get("rms_dbfs") is not None and audio["rms_dbfs"] < -45):
        return None  # no music playing
    confidence = factor * max(mode_conf, (audio.get("tempo_confidence") or 0.0))
    return _reading(valence, drive, confidence,
                    [k for k, v in (("mode", valence), ("tempo_bpm", tempo)) if v is not None] +
                    ["onset_rate_per_second"], mode=mode if valence is not None else None,
                    source="music_stem" if separated else "full_mix")


def read_sound(audio, silence_state, base=None):
    if silence_state == "silent":
        return _reading(None, -0.6, 0.5, ["silence"], startle=False)
    if not audio or audio.get("rms_dbfs") is None:
        return None
    loud = _blend(_clip((audio["rms_dbfs"] + 35.0) / 20.0), audio["rms_dbfs"], base)
    startle = bool(audio.get("startle"))
    arousal = _clip(loud + (0.6 if startle else 0.0))
    return _reading(-0.3 if startle else None, arousal, 0.5 if startle else 0.4,
                    ["rms_dbfs"] + (["startle"] if startle else []), startle=startle)


# -------------------------------------------------------------- labels ----

def label_for(valence, arousal, tension, startle):
    if startle:
        return "shock"
    if tension >= 0.7:
        return "tension"
    if arousal > 0.4:
        return "tension" if valence < -0.3 else "excitement" if valence > 0.3 else "intensity"
    if arousal < -0.3:
        return "melancholy" if valence < -0.3 else "calm" if valence > 0.3 else "stillness"
    return "unease" if valence < -0.3 else "warmth" if valence > 0.3 else "neutral"


def _weighted(readings, key):
    total = weight = 0.0
    for name, r in readings.items():
        if r[key] is None:
            continue
        w = WEIGHTS[name] * r["confidence"]
        total += w * r[key]
        weight += w
    return (total / weight, weight) if weight > 0 else (None, 0.0)


def find_conflicts(readings):
    confident = [(n, r) for n, r in readings.items()
                 if r["valence"] is not None and r["confidence"] >= CONFLICT_MIN_CONFIDENCE]
    conflicts = []
    for i, (a, ra) in enumerate(confident):
        for b, rb in confident[i + 1:]:
            if abs(ra["valence"] - rb["valence"]) > CONFLICT_VALENCE_GAP:
                conflicts.append({"between": [a, b], "valence": [ra["valence"], rb["valence"]]})
    music = readings.get("music")
    irony = bool(music and music["valence"] is not None and music["valence"] > 0.3 and
                 music["confidence"] >= CONFLICT_MIN_CONFIDENCE and
                 any(readings.get(o) and readings[o]["valence"] is not None and
                     readings[o]["valence"] < -0.3 and
                     readings[o]["confidence"] >= CONFLICT_MIN_CONFIDENCE
                     for o in ("look", "visual", "voice")))
    return conflicts, irony


# --------------------------------------------------------------- fusion ----

def fuse_timeline(timeline):
    """Add ``feeling`` to each window and set ``emotion``; return the film-level curve."""
    state_v = state_a = None
    previous_peak = None
    curve = {"start_seconds": [], "valence": [], "arousal": [], "tension": [], "label": []}
    windows = [w for w in timeline if w.get("feeling_signals")]
    pace_base = baseline([(w["feeling_signals"].get("pacing") or {}).get("cuts_per_minute")
                          for w in windows])
    loud_base = baseline([(w["feeling_signals"].get("audio") or {}).get("rms_dbfs")
                          for w in windows
                          if (w.get("audio") or {}).get("state") != "silent"])
    voice_base = baseline([(seg.get("voice") or {}).get("loudness_dbfs")
                           for w in windows for seg in (w.get("speech") or [])])
    for window in timeline:
        signals = window.get("feeling_signals")
        if not signals:
            continue
        length = window["end_seconds"] - window["start_seconds"]
        silence_state = (window.get("audio") or {}).get("state")
        readings = {k: v for k, v in {
            "pacing": read_pacing(signals.get("pacing"), pace_base),
            "look": read_look(signals.get("look")),
            "visual": read_visual(signals.get("visual_mood")),
            "voice": read_voice(window.get("speech"), window["start_seconds"],
                                window["end_seconds"], voice_base),
            "music": (read_music(signals["music"], separated=True, listening=signals.get("listening"))
                      if signals.get("music") is not None
                      else read_music(signals.get("audio"), listening=signals.get("listening"))),
            "sound": read_sound(signals.get("audio"), silence_state, loud_base),
        }.items() if v is not None}
        v_now, v_weight = _weighted(readings, "valence")
        a_now, a_weight = _weighted(readings, "arousal")
        carry = 0.5 ** (length / CARRY_HALF_LIFE_SECONDS)
        if v_now is not None:
            state_v = v_now if state_v is None else carry * state_v + (1 - carry) * v_now
        if a_now is not None:
            smoothed = a_now if state_a is None else carry * state_a + (1 - carry) * a_now
            # A shock is felt at once and dominates the moment; calming down takes time.
            sound = readings.get("sound") or {}
            state_a = max(smoothed, sound["arousal"]) if sound.get("startle") else smoothed
        valence, arousal = state_v or 0.0, state_a or 0.0
        startle = bool(readings.get("sound", {}).get("startle"))
        music = readings.get("music") or {}
        look = readings.get("look") or {}
        audio = signals.get("audio") or {}
        hush_after_loud = (silence_state == "silent" and previous_peak is not None
                           and previous_peak > -20.0)
        tension = _clip(0.5 * max(arousal, 0.0) + 0.25 * max(-valence, 0.0)
                        + 0.15 * (look.get("darkness") or 0.0)
                        + (0.1 if music.get("mode") == "minor" else 0.0)
                        + (0.1 if set(music.get("moods") or []) & {"tense", "ominous"} else 0.0)
                        + (0.2 if hush_after_loud else 0.0), 0.0, 1.0)
        previous_peak = audio.get("peak_short_term_dbfs")
        conflicts, irony = find_conflicts(readings)
        possible = sum(WEIGHTS.values())
        confidence = _clip((v_weight + a_weight) / (2 * possible), 0.0, 1.0)
        label = label_for(valence, arousal, tension, startle)
        window["feeling"] = {
            **readings,
            "fused": {"valence": _r(valence), "arousal": _r(arousal), "tension": _r(tension),
                      "label": label, "confidence": _r(confidence),
                      "conflicts": conflicts, "possible_irony": irony,
                      "method": VERSION}}
        window["emotion"] = label
        curve["start_seconds"].append(window["start_seconds"])
        for key, value in (("valence", valence), ("arousal", arousal), ("tension", tension)):
            curve[key].append(_r(value))
        curve["label"].append(label)
    curve["baselines"] = {"cuts_per_minute": pace_base, "rms_dbfs": loud_base}
    curve["turning_points"] = turning_points(curve)
    curve["method"] = VERSION
    return curve


def turning_points(curve, min_change=0.3, peak_tension=0.6):
    """Valence sign changes with a clear swing, and tension peaks."""
    points = []
    v, t, times = curve["valence"], curve["tension"], curve["start_seconds"]
    for i in range(1, len(v)):
        if v[i - 1] * v[i] < 0 and abs(v[i] - v[i - 1]) >= min_change:
            points.append({"time_seconds": times[i], "kind": "mood_turn",
                           "direction": "brighter" if v[i] > v[i - 1] else "darker"})
    for i in range(len(t)):
        left = t[i - 1] if i > 0 else -math.inf
        right = t[i + 1] if i + 1 < len(t) else -math.inf
        if t[i] >= peak_tension and t[i] > left and t[i] >= right:
            points.append({"time_seconds": times[i], "kind": "tension_peak", "tension": t[i]})
    return sorted(points, key=lambda p: p["time_seconds"])
