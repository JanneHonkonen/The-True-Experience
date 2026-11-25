"""CPU feeling signals: pacing, look, loudness and basic music features.

Stage 2 and the CPU part of stage 5 in docs/FEELING_PIPELINE.md. These are
measurements, not emotions: they are later mapped to valence/arousal by the
fusion step. Audio and frames are streamed from FFmpeg one timeline window at a
time, so memory does not grow with film length. Needs NumPy only.

Audio features are computed on the full mix (no source separation yet), so
music values can be disturbed by dialogue and effects.
"""
from __future__ import annotations

import math
import subprocess
from pathlib import Path

AUDIO_RATE = 22050
FRAME_FPS = 2.0
FRAME_WIDTH = 160
N_FFT = 2048
HOP = 512
STARTLE_JUMP_DB = 15.0
STARTLE_MIN_DBFS = -30.0

# Krumhansl-Kessler key profiles, index 0 = tonic.
MAJOR_PROFILE = (6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88)
MINOR_PROFILE = (6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17)
PITCH_NAMES = ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")


def _np():
    try:
        import numpy
    except ImportError as exc:
        raise RuntimeError("Feeling signals require NumPy: python -m pip install numpy") from exc
    return numpy


def _round(value, digits=4):
    return None if value is None else round(float(value), digits)


# ---------------------------------------------------------------- look ----

def frame_look(frame):
    """Brightness, contrast, saturation and warmth of one RGB uint8 frame (0..1 scales)."""
    np = _np()
    rgb = frame.astype(np.float32) / 255.0
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    luma = 0.2126 * r + 0.7152 * g + 0.0722 * b
    mx, mn = rgb.max(axis=-1), rgb.min(axis=-1)
    saturation = np.where(mx > 1e-6, (mx - mn) / np.maximum(mx, 1e-6), 0.0)
    return {"brightness": float(luma.mean()), "contrast": float(luma.std()),
            "saturation": float(saturation.mean()), "warmth": float((r - b).mean()),
            "luma": luma}


def summarize_look(frames, times, cut_times):
    """Average look over a window's frames; motion skips frame pairs that span a cut."""
    np = _np()
    if not frames:
        return None
    looks = [frame_look(f) for f in frames]
    motion = []
    for i in range(1, len(looks)):
        a, b = times[i - 1], times[i]
        if any(a < c <= b for c in cut_times):
            continue
        motion.append(float(np.abs(looks[i]["luma"] - looks[i - 1]["luma"]).mean()))
    return {"frame_count": len(frames),
            **{k: _round(sum(l[k] for l in looks) / len(looks))
               for k in ("brightness", "contrast", "saturation", "warmth")},
            "motion": _round(sum(motion) / len(motion)) if motion else None}


# --------------------------------------------------------------- audio ----

def _spectrogram(samples, rate):
    np = _np()
    if len(samples) < N_FFT:
        samples = np.pad(samples, (0, N_FFT - len(samples)))
    count = 1 + (len(samples) - N_FFT) // HOP
    idx = np.arange(N_FFT)[None, :] + HOP * np.arange(count)[:, None]
    window = np.hanning(N_FFT).astype(np.float32)
    mag = np.abs(np.fft.rfft(samples[idx] * window, axis=1))
    freqs = np.fft.rfftfreq(N_FFT, 1.0 / rate)
    return mag, freqs


def onset_envelope(mag):
    """Spectral flux on log magnitude: how much new energy appears per frame."""
    np = _np()
    logmag = np.log1p(100.0 * mag)
    flux = np.maximum(0.0, np.diff(logmag, axis=0)).sum(axis=1)
    return np.concatenate([[0.0], flux])


def estimate_tempo(envelope, rate, min_bpm=60.0, max_bpm=180.0):
    """Autocorrelation tempo. Returns (bpm, confidence 0..1) or (None, 0)."""
    np = _np()
    env = envelope - envelope.mean()
    if len(env) < 8 or not np.any(env):
        return None, 0.0
    # Light smoothing so beats that fall between analysis frames still line up.
    kernel = np.array([0.25, 0.5, 1.0, 0.5, 0.25])
    env = np.convolve(env, kernel / kernel.sum(), mode="same")
    ac = np.correlate(env, env, mode="full")[len(env) - 1:]
    if ac[0] <= 0:
        return None, 0.0
    frames_per_second = rate / HOP
    lo = max(1, int(math.floor(60.0 * frames_per_second / max_bpm)))
    hi = min(len(ac) - 1, int(math.ceil(60.0 * frames_per_second / min_bpm)))
    if hi <= lo:
        return None, 0.0
    lags = np.arange(lo, hi + 1)
    # Prefer tempi near 120 BPM when multiples score alike (half/double-time ambiguity).
    prior = np.exp(-0.5 * (np.log2(60.0 * frames_per_second / lags / 120.0)) ** 2)
    lag = int(lags[np.argmax(ac[lo:hi + 1] * prior)])
    confidence = float(ac[lag] / ac[0])
    # Parabolic interpolation for sub-frame lag precision.
    if 0 < lag < len(ac) - 1:
        a, b, c = ac[lag - 1], ac[lag], ac[lag + 1]
        denom = a - 2 * b + c
        if denom:
            lag = lag + 0.5 * (a - c) / denom
    if confidence < 0.1:
        return None, max(0.0, confidence)
    return 60.0 * frames_per_second / lag, min(1.0, confidence)


def onset_rate(envelope, rate):
    """Onsets per second: local peaks clearly above the background, at least 0.1 s apart.

    A peak must exceed mean + 1.5 std *and* 1.5x the median, so steady tones
    and noise floors do not count as onsets.
    """
    np = _np()
    if len(envelope) < 3:
        return 0.0
    threshold = max(envelope.mean() + 1.5 * envelope.std(), 1.5 * float(np.median(envelope)))
    min_gap = max(1, int(0.1 * rate / HOP))
    peaks, last = 0, -min_gap
    for i in range(1, len(envelope) - 1):
        if (envelope[i] > threshold and envelope[i] >= envelope[i - 1]
                and envelope[i] >= envelope[i + 1] and i - last >= min_gap):
            peaks += 1
            last = i
    return peaks / (len(envelope) * HOP / rate)


def chroma(mag, freqs):
    """12-bin pitch-class energy (C=0) from 55 Hz to 5 kHz."""
    np = _np()
    band = (freqs >= 55.0) & (freqs <= 5000.0)
    pcs = (np.round(12 * np.log2(freqs[band] / 440.0)).astype(int) + 9) % 12
    energy = (mag[:, band] ** 2).sum(axis=0)
    result = np.zeros(12)
    np.add.at(result, pcs, energy)
    return result


def estimate_key(chroma_vector):
    """Best major/minor key by profile correlation.

    Returns (key name, mode, confidence 0..1). Confidence is the gap between the
    best key of the chosen mode and the best key of the other mode; weak or
    atonal input gives mode 'unclear'.
    """
    np = _np()
    if chroma_vector.sum() <= 0 or np.allclose(chroma_vector, chroma_vector[0]):
        return None, "unclear", 0.0
    best = {}
    for mode, profile in (("major", MAJOR_PROFILE), ("minor", MINOR_PROFILE)):
        scores = [float(np.corrcoef(chroma_vector, np.roll(profile, tonic))[0, 1])
                  for tonic in range(12)]
        tonic = int(np.argmax(scores))
        best[mode] = (scores[tonic], tonic)
    mode = max(best, key=lambda m: best[m][0])
    other = "minor" if mode == "major" else "major"
    score, tonic = best[mode]
    confidence = min(1.0, max(0.0, (score - best[other][0]) * 4))
    if score < 0.5 or confidence < 0.05:
        return None, "unclear", 0.0
    return f"{PITCH_NAMES[tonic]} {mode}", mode, confidence


def short_term_db(samples, rate, frame_seconds=0.05):
    np = _np()
    size = max(1, int(rate * frame_seconds))
    count = len(samples) // size
    if count == 0:
        return np.array([])
    blocks = samples[:count * size].reshape(count, size)
    rms = np.sqrt((blocks.astype(np.float64) ** 2).mean(axis=1))
    return 20 * np.log10(np.maximum(rms, 1e-10))


def detect_startles(levels_db, frame_seconds, history_frames, offset_seconds):
    """Times where loudness jumps >= STARTLE_JUMP_DB above the previous second's median.

    ``history_frames`` levels at the start belong to the previous window and are
    used only as context.
    """
    np = _np()
    times = []
    context = max(1, int(round(1.0 / frame_seconds)))
    last = -1e9
    for i in range(history_frames, len(levels_db)):
        start = max(0, i - context)
        if i - start < context // 2:
            continue
        previous = float(np.median(levels_db[start:i]))
        t = offset_seconds + (i - history_frames) * frame_seconds
        if (levels_db[i] >= STARTLE_MIN_DBFS and levels_db[i] - previous >= STARTLE_JUMP_DB
                and t - last >= 1.0):
            times.append(round(t, 3))
            last = t
    return times


def summarize_audio(samples, rate, history, offset_seconds):
    """Loudness, startle, onset rate, brightness, tempo and key for one window."""
    np = _np()
    if len(samples) == 0:
        return None
    levels = short_term_db(np.concatenate([history, samples]), rate)
    history_frames = len(history) // max(1, int(rate * 0.05))
    own = levels[history_frames:]
    rms = float(np.sqrt((samples.astype(np.float64) ** 2).mean()))
    mag, freqs = _spectrogram(samples, rate)
    power = mag.sum(axis=1)
    centroid = float((mag * freqs).sum() / power.sum()) if power.sum() > 0 else None
    env = onset_envelope(mag)
    rate_per_second = onset_rate(env, rate)
    tempo, tempo_conf = estimate_tempo(env, rate)
    if rate_per_second * len(samples) / rate < 3:
        tempo, tempo_conf = None, 0.0  # too few real onsets to speak of a beat
    key, mode, mode_conf = estimate_key(chroma(mag, freqs))
    startles = detect_startles(levels, 0.05, history_frames, offset_seconds)
    return {"source": "full_mix",
            "rms_dbfs": _round(20 * math.log10(max(rms, 1e-10)), 2),
            "peak_short_term_dbfs": _round(own.max(), 2) if len(own) else None,
            "startle": bool(startles), "startle_times_seconds": startles,
            "onset_rate_per_second": _round(rate_per_second, 3),
            "spectral_centroid_hz": _round(centroid, 1),
            "tempo_bpm": _round(tempo, 1), "tempo_confidence": _round(tempo_conf, 3),
            "key": key, "mode": mode, "mode_confidence": _round(mode_conf, 3)}


# ------------------------------------------------------------- pacing ----

def summarize_pacing(start, end, shots):
    cuts = [s["start_seconds"] for s in shots[1:] if start <= s["start_seconds"] < end]
    overlapping = [s["end_seconds"] - s["start_seconds"] for s in shots
                   if min(end, s["end_seconds"]) > max(start, s["start_seconds"])]
    length = end - start
    return {"cuts_in_window": len(cuts),
            "cuts_per_minute": _round(len(cuts) * 60.0 / length, 2) if length > 0 else None,
            "mean_overlapping_shot_seconds":
                _round(sum(overlapping) / len(overlapping), 3) if overlapping else None}


# ------------------------------------------------------------ streaming ----

def _even(value):
    return max(2, int(round(value / 2)) * 2)


def stream_audio_windows(video, bounds, rate=AUDIO_RATE):
    """Yield one float32 mono array per (start, end) window, read sequentially."""
    np = _np()
    cmd = ["ffmpeg", "-hide_banner", "-nostdin", "-v", "error", "-i", str(video),
           "-map", "0:a:0", "-vn", "-ac", "1", "-ar", str(rate), "-f", "f32le", "pipe:1"]
    with subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL) as proc:
        consumed = 0
        for start, end in bounds:
            target = int(round(end * rate))
            need = max(0, target - consumed)
            data = proc.stdout.read(need * 4) if need else b""
            consumed += len(data) // 4
            yield np.frombuffer(data[:len(data) // 4 * 4], dtype=np.float32)
        proc.stdout.close()
        proc.wait()


def stream_frame_windows(video, bounds, width, height, fps=FRAME_FPS, frame_width=FRAME_WIDTH):
    """Yield (frames, times) per window from a low-resolution fixed-rate decode."""
    np = _np()
    w = _even(frame_width)
    h = _even(frame_width * height / width) if width and height else _even(frame_width * 9 / 16)
    size = w * h * 3
    cmd = ["ffmpeg", "-hide_banner", "-nostdin", "-v", "error", "-i", str(video),
           "-map", "0:v:0", "-an", "-sn", "-vf", f"fps={fps},scale={w}:{h}",
           "-pix_fmt", "rgb24", "-f", "rawvideo", "pipe:1"]
    with subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL) as proc:
        index = 0
        pending = None
        for start, end in bounds:
            frames, times = [], []
            while True:
                if pending is None:
                    raw = proc.stdout.read(size)
                    if len(raw) < size:
                        break
                    pending = (np.frombuffer(raw, dtype=np.uint8).reshape(h, w, 3), index / fps)
                    index += 1
                frame, t = pending
                if t >= end:
                    break
                if t >= start:
                    frames.append(frame)
                    times.append(t)
                pending = None
            yield frames, times
        proc.stdout.close()
        proc.wait()


def compute_signals(video, info, shots, timeline, music_path=None):
    """Add ``feeling_signals`` to every timeline window, in place."""
    video = Path(video)
    bounds = [(w["start_seconds"], w["end_seconds"]) for w in timeline]
    cut_times = [s["start_seconds"] for s in shots[1:]]
    np = _np()
    audio_iter = stream_audio_windows(video, bounds) if info.get("has_audio") else None
    frame_iter = stream_frame_windows(video, bounds, info.get("width"), info.get("height"))
    music_iter = stream_audio_windows(music_path, bounds) if music_path else None
    history = np.zeros(0, dtype=np.float32)
    music_history = np.zeros(0, dtype=np.float32)
    for window, (start, end) in zip(timeline, bounds):
        frames, times = next(frame_iter, ([], []))
        audio = None
        if audio_iter is not None:
            samples = next(audio_iter, np.zeros(0, dtype=np.float32))
            audio = summarize_audio(samples, AUDIO_RATE, history, start)
            history = np.concatenate([history, samples])[-AUDIO_RATE:]
        music = None
        if music_iter is not None:
            samples = next(music_iter, np.zeros(0, dtype=np.float32))
            music = summarize_audio(samples, AUDIO_RATE, music_history, start)
            music_history = np.concatenate([music_history, samples])[-AUDIO_RATE:]
        window["feeling_signals"] = {
            "pacing": summarize_pacing(start, end, shots),
            "look": summarize_look(frames, times, cut_times),
            "audio": audio,
            **({"music": music} if music_iter is not None else {})}
    for it in (audio_iter, frame_iter, music_iter):
        if it is not None:
            it.close()
    return {"audio_rate": AUDIO_RATE if info.get("has_audio") else None,
            "frame_fps": FRAME_FPS, "frame_width": _even(FRAME_WIDTH),
            "audio_source": "full_mix" if info.get("has_audio") else None,
            "music_source": "separated_music_stem" if music_path else None}
