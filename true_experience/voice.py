"""Stage 4: how things are said. Prosody, speaker grouping and voice emotion.

Runs on the speech segments that faster-whisper found. For each segment:

- **Prosody (CPU, NumPy):** pitch level and variability, loudness, speaking
  rate, pause share.
- **Speaker grouping (CPU, NumPy):** a simple voice fingerprint (MFCC
  statistics plus pitch) clustered into S1, S2 ... Rough, with no model
  download; good enough to tell clearly different voices apart.
- **Voice emotion (optional, GPU if available):** a small audio-classification
  model (default ``superb/wav2vec2-base-superb-er``, about 0.4 GB, classes
  neutral/happy/angry/sad). Loaded once, used for all segments, then freed.

Audio is streamed from FFmpeg in time order, so memory stays flat.
"""
from __future__ import annotations

import math
import subprocess
import sys

RATE = 16000
FRAME = 640          # 40 ms
HOP = 160            # 10 ms
F0_MIN, F0_MAX = 70.0, 400.0
DEFAULT_EMOTION_MODEL = "superb/wav2vec2-base-superb-er"

# Valence/arousal for common speech-emotion labels.
EMOTION_VA = {
    "neu": (0.0, 0.0), "neutral": (0.0, 0.0), "calm": (0.2, -0.4),
    "hap": (0.7, 0.5), "happy": (0.7, 0.5), "joy": (0.7, 0.5), "surprised": (0.2, 0.7),
    "ang": (-0.6, 0.8), "angry": (-0.6, 0.8), "anger": (-0.6, 0.8),
    "sad": (-0.6, -0.4), "sadness": (-0.6, -0.4),
    "fear": (-0.7, 0.6), "fearful": (-0.7, 0.6), "disgust": (-0.6, 0.3),
}


def _np():
    try:
        import numpy
    except ImportError as exc:
        raise RuntimeError("Voice analysis requires NumPy: python -m pip install numpy") from exc
    return numpy


def _r(x, d=3):
    return None if x is None else round(float(x), d)


# -------------------------------------------------------------- prosody ----

def _frames(samples):
    np = _np()
    if len(samples) < FRAME:
        samples = np.pad(samples, (0, FRAME - len(samples)))
    count = 1 + (len(samples) - FRAME) // HOP
    idx = np.arange(FRAME)[None, :] + HOP * np.arange(count)[:, None]
    return samples[idx]


def pitch_track(samples, rate=RATE):
    """F0 per 10 ms frame by normalised autocorrelation; NaN where unvoiced."""
    np = _np()
    frames = _frames(samples.astype(np.float64))
    frames = frames - frames.mean(axis=1, keepdims=True)
    energy = (frames ** 2).sum(axis=1)
    spec = np.fft.rfft(frames, n=2 * FRAME, axis=1)
    ac = np.fft.irfft(np.abs(spec) ** 2, axis=1)[:, :FRAME]
    lo, hi = int(rate / F0_MAX), int(rate / F0_MIN)
    lags = lo + np.argmax(ac[:, lo:hi], axis=1)
    peak = ac[np.arange(len(ac)), lags] / np.maximum(ac[:, 0], 1e-12)
    loud_enough = energy > 1e-6 * FRAME  # about -60 dBFS
    voiced = (peak > 0.5) & loud_enough
    f0 = np.where(voiced, rate / lags, np.nan)
    return f0


def prosody(samples, text, rate=RATE):
    """Pitch, loudness, rate and pauses for one speech segment."""
    np = _np()
    seconds = len(samples) / rate
    if seconds <= 0:
        return None
    f0 = pitch_track(samples, rate)
    voiced = f0[~np.isnan(f0)]
    semis = 12 * np.log2(voiced / 100.0) if len(voiced) else voiced
    frame_db = 20 * np.log10(np.maximum(np.sqrt((_frames(samples) ** 2).mean(axis=1)), 1e-10))
    loud = float(20 * math.log10(max(float(np.sqrt((samples.astype(np.float64) ** 2).mean())), 1e-10)))
    pause_share = float((frame_db < loud - 15).mean()) if len(frame_db) else 0.0
    words = len(text.split())
    return {"pitch_median_hz": _r(np.median(voiced), 1) if len(voiced) else None,
            "pitch_variability_semitones": _r(np.std(semis), 2) if len(voiced) > 5 else None,
            "voiced_share": _r(len(voiced) / max(1, len(f0))),
            "loudness_dbfs": _r(loud, 2),
            "words_per_second": _r(words / seconds, 2),
            "pause_share": _r(pause_share)}


# ------------------------------------------------------ speaker grouping ----

def _mel_filterbank(n_fft, rate, n_mels=26):
    np = _np()
    def hz_to_mel(f): return 2595 * np.log10(1 + f / 700)
    def mel_to_hz(m): return 700 * (10 ** (m / 2595) - 1)
    mels = np.linspace(hz_to_mel(60), hz_to_mel(rate / 2 * 0.95), n_mels + 2)
    bins = np.floor((n_fft + 1) * mel_to_hz(mels) / rate).astype(int)
    bank = np.zeros((n_mels, n_fft // 2 + 1))
    for m in range(1, n_mels + 1):
        a, b, c = bins[m - 1], bins[m], bins[m + 1]
        for k in range(a, b):
            bank[m - 1, k] = (k - a) / max(1, b - a)
        for k in range(b, c):
            bank[m - 1, k] = (c - k) / max(1, c - b)
    return bank


def voice_fingerprint(samples, rate=RATE, pitch_hz=None):
    """MFCC 1-12 mean and std, plus log pitch: a 25-number voice summary."""
    np = _np()
    frames = _frames(samples.astype(np.float64)) * np.hamming(FRAME)
    power = np.abs(np.fft.rfft(frames, n=1024, axis=1)) ** 2
    mel = np.log(power @ _mel_filterbank(1024, rate).T + 1e-10)
    n = mel.shape[1]
    dct = np.cos(np.pi / n * (np.arange(n)[None, :] + 0.5) * np.arange(13)[:, None])
    mfcc = mel @ dct.T
    voiced = mfcc[(power.sum(axis=1) > np.percentile(power.sum(axis=1), 40))] if len(mfcc) > 4 else mfcc
    body = voiced[:, 1:13]
    pitch = math.log2(pitch_hz / 100.0) * 4 if pitch_hz else 0.0
    return np.concatenate([body.mean(axis=0), body.std(axis=0), [pitch]])


def cluster_speakers(fingerprints, threshold=0.7):
    """Average-linkage clustering of voice fingerprints. Returns labels S1, S2...

    Features are centred and divided by their spread, but never by less than 1,
    so near-constant features cannot blow small noise up into fake speakers.
    The threshold is a first guess to be tuned on real dialogue.
    """
    np = _np()
    if not fingerprints:
        return []
    X = np.array(fingerprints, dtype=np.float64)
    if len(X) == 1:
        return ["S1"]
    X = (X - X.mean(axis=0)) / np.maximum(X.std(axis=0), 1.0)
    clusters = [[i] for i in range(len(X))]
    dist = lambda a, b: float(np.mean([np.linalg.norm(X[i] - X[j]) / math.sqrt(X.shape[1])
                                       for i in a for j in b]))
    while len(clusters) > 1:
        best = min(((dist(clusters[i], clusters[j]), i, j)
                    for i in range(len(clusters)) for j in range(i + 1, len(clusters))))
        if best[0] > threshold:
            break
        _, i, j = best
        clusters[i] += clusters.pop(j)
    order = sorted(clusters, key=min)
    labels = [None] * len(X)
    for n, members in enumerate(order, 1):
        for m in members:
            labels[m] = f"S{n}"
    return labels


# --------------------------------------------------------- voice emotion ----

def load_emotion_backend(model=DEFAULT_EMOTION_MODEL):
    """Transformers audio-classification pipeline on GPU when available."""
    try:
        import torch
        from transformers import pipeline
    except ImportError as exc:
        raise RuntimeError("Voice emotion needs transformers and torch "
                           "(python -m pip install -r requirements.txt)") from exc
    device = 0 if torch.cuda.is_available() else -1
    clf = pipeline("audio-classification", model=model, device=device)

    def classify(samples, rate):
        return clf({"raw": samples, "sampling_rate": rate}, top_k=None)

    def close():
        nonlocal clf
        clf = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    classify.close = close
    return classify


def emotion_reading(scores):
    """Turn [{label, score}] into the top label plus expected valence/arousal."""
    total = v = a = 0.0
    for item in scores:
        key = str(item["label"]).strip().lower()
        if key in EMOTION_VA:
            ev, ea = EMOTION_VA[key]
            v += item["score"] * ev
            a += item["score"] * ea
            total += item["score"]
    if total <= 0:
        return None
    top = max(scores, key=lambda s: s["score"])
    return {"label": str(top["label"]).lower(), "score": _r(top["score"]),
            "valence": _r(v / total), "arousal": _r(a / total)}


# ------------------------------------------------------------ streaming ----

def stream_segments(video, segments, rate=RATE):
    """Yield float32 audio for time-ordered segments from one FFmpeg decode."""
    np = _np()
    cmd = ["ffmpeg", "-hide_banner", "-nostdin", "-v", "error", "-i", str(video),
           "-map", "0:a:0", "-vn", "-ac", "1", "-ar", str(rate), "-f", "f32le", "pipe:1"]
    with subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL) as proc:
        buffer = np.zeros(0, dtype=np.float32)
        buffer_start = 0  # sample index of buffer[0]
        for seg in segments:
            a = max(int(seg["start_seconds"] * rate), buffer_start)
            b = max(a, int(seg["end_seconds"] * rate))
            missing = b - (buffer_start + len(buffer))
            if missing > 0:
                data = proc.stdout.read(missing * 4)
                buffer = np.concatenate([buffer, np.frombuffer(data[:len(data) // 4 * 4], np.float32)])
            yield buffer[a - buffer_start:b - buffer_start].copy()
            keep_from = max(buffer_start, b - 2 * rate)  # keep 2 s for overlapping segments
            buffer = buffer[keep_from - buffer_start:]
            buffer_start = keep_from
        proc.stdout.close()
        proc.wait()


def analyze_voice(video, speech, *, emotion_model=DEFAULT_EMOTION_MODEL,
                  emotion_backend=None, audio_loader=None):
    """Add ``voice`` to each speech segment in place. Returns stage metadata."""
    segments = sorted(speech, key=lambda s: (s["start_seconds"], s["end_seconds"]))
    if not segments:
        return {"segments": 0, "speakers": 0, "emotion_model": None}
    if emotion_backend is None and emotion_model:
        emotion_backend = load_emotion_backend(emotion_model)
    loader = audio_loader or (lambda: stream_segments(video, segments))
    fingerprints = []
    try:
        for seg, samples in zip(segments, loader()):
            info = prosody(samples, seg.get("text", "")) or {}
            emotion = None
            if emotion_backend is not None and len(samples) >= RATE // 2:
                emotion = emotion_reading(emotion_backend(samples, RATE))
            seg["voice"] = {**info, "emotion": emotion}
            fingerprints.append(voice_fingerprint(samples, pitch_hz=info.get("pitch_median_hz"))
                                if len(samples) >= RATE // 2 else None)
    finally:
        close = getattr(emotion_backend, "close", None)
        if close:
            close()
    usable = [i for i, f in enumerate(fingerprints) if f is not None]
    labels = cluster_speakers([fingerprints[i] for i in usable])
    for i, label in zip(usable, labels):
        segments[i]["speaker"] = label
        segments[i]["speaker_source"] = "voice_fingerprint_cluster"
    print(f"Voice: {len(segments)} segments, {len(set(labels))} voice groups", file=sys.stderr)
    return {"segments": len(segments), "speakers": len(set(labels)),
            "emotion_model": emotion_model if emotion_backend is not None else None,
            "speaker_method": "mfcc_pitch_average_linkage"}
