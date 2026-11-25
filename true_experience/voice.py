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
DEFAULT_SPEAKER_MODEL = "microsoft/wavlm-base-plus-sv"
SPEAKER_THRESHOLD = 0.80     # cosine similarity; the model card suggests 0.86 for clean audio
MIN_SPEAKER_SECONDS = 1.0

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


def pitch_track(samples, rate=RATE, threshold=0.15):
    """F0 per 10 ms frame with the YIN difference function; NaN where unvoiced.

    YIN's cumulative-mean normalisation avoids most octave errors that plain
    autocorrelation makes on real (and separated) voices.
    """
    np = _np()
    frames = _frames(samples.astype(np.float64))
    frames = frames - frames.mean(axis=1, keepdims=True)
    energy = (frames ** 2).sum(axis=1)
    spec = np.fft.rfft(frames, n=2 * FRAME, axis=1)
    ac = np.fft.irfft(np.abs(spec) ** 2, axis=1)[:, :FRAME]
    lo, hi = int(rate / F0_MAX), int(rate / F0_MIN)
    diff = 2.0 * (ac[:, :1] - ac[:, :hi + 1])            # d(tau), tau = 0..hi
    diff[:, 0] = 0.0
    cum = np.cumsum(diff[:, 1:], axis=1) / np.arange(1, hi + 1)
    cmnd = np.ones_like(diff)
    cmnd[:, 1:] = diff[:, 1:] / np.maximum(cum, 1e-12)
    band = cmnd[:, lo:hi + 1]
    below = band < threshold
    # First dip under the threshold, then walk to its local minimum.
    first = np.where(below.any(axis=1), below.argmax(axis=1), band.argmin(axis=1))
    idx = first.copy()
    for _ in range(8):
        step = np.minimum(idx + 1, band.shape[1] - 1)
        better = band[np.arange(len(band)), step] < band[np.arange(len(band)), idx]
        idx = np.where(better, step, idx)
    best = band[np.arange(len(band)), idx]
    lags = (lo + idx).astype(np.float64)
    # Parabolic interpolation around the minimum for sub-sample precision.
    left = band[np.arange(len(band)), np.maximum(idx - 1, 0)]
    right = band[np.arange(len(band)), np.minimum(idx + 1, band.shape[1] - 1)]
    denom = left - 2 * best + right
    with np.errstate(divide="ignore", invalid="ignore"):
        shift = np.where(np.abs(denom) > 1e-12, 0.5 * (left - right) / denom, 0.0)
    lags = lags + np.clip(np.nan_to_num(shift), -1.0, 1.0)
    loud_enough = energy > 1e-6 * FRAME  # about -60 dBFS
    voiced = (best < 0.35) & loud_enough
    return np.where(voiced, rate / lags, np.nan)


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


def load_speaker_backend(model=DEFAULT_SPEAKER_MODEL):
    """Speaker embedding (x-vector) model from transformers; GPU when available."""
    try:
        import torch
        from transformers import AutoFeatureExtractor, WavLMForXVector
    except ImportError as exc:
        raise RuntimeError("Speaker embeddings need transformers and torch "
                           "(python -m pip install -r requirements.txt)") from exc
    np = _np()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    extractor = AutoFeatureExtractor.from_pretrained(model)
    net = WavLMForXVector.from_pretrained(model).to(device).eval()

    def embed(samples, rate):
        inputs = extractor([samples], sampling_rate=rate, return_tensors="pt", padding=True)
        with torch.no_grad():
            vec = net(**{k: v.to(device) for k, v in inputs.items()}).embeddings[0]
        vec = vec.cpu().numpy().astype(np.float64)
        return vec / max(1e-9, float(np.linalg.norm(vec)))

    def close():
        nonlocal net
        net = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    embed.close = close
    return embed


def cluster_embeddings(vectors, threshold=SPEAKER_THRESHOLD):
    """Average-linkage clustering of unit speaker embeddings by cosine similarity."""
    np = _np()
    if not vectors:
        return []
    X = np.array(vectors, dtype=np.float64)
    sim = X @ X.T
    clusters = [[i] for i in range(len(X))]
    while len(clusters) > 1:
        best = None
        for a in range(len(clusters)):
            for b in range(a + 1, len(clusters)):
                s = float(sim[np.ix_(clusters[a], clusters[b])].mean())
                if best is None or s > best[0]:
                    best = (s, a, b)
        if best[0] < threshold:
            break
        _, a, b = best
        clusters[a] += clusters.pop(b)
    labels = [None] * len(X)
    for n, members in enumerate(sorted(clusters, key=min), 1):
        for m in members:
            labels[m] = f"S{n}"
    return labels


def relative_emotions(raw):
    """Re-read each line's emotion against the film's own average.

    A classifier that leans one way on a whole soundtrack (separated vocals
    came out "angry" for 22 of 25 trailer lines) then still shows which lines
    are angrier, happier or sadder than usual. ``raw`` holds score lists or None.
    """
    labels = sorted({str(i["label"]).strip().lower() for r in raw if r for i in r})
    if not labels:
        return [None] * len(raw)
    rows = [None if r is None else {str(i["label"]).strip().lower(): float(i["score"]) for i in r}
            for r in raw]
    present = [r for r in rows if r]
    mean = {k: sum(r.get(k, 0.0) for r in present) / len(present) for k in labels}
    out = []
    for r in rows:
        if r is None:
            out.append(None)
            continue
        lift = {k: r.get(k, 0.0) - mean[k] for k in labels}
        positive = [{"label": k, "score": v} for k, v in lift.items() if v > 0]
        reading = emotion_reading(positive) if positive else None
        top_raw = max(r.items(), key=lambda kv: kv[1])
        top = max(lift.items(), key=lambda kv: kv[1])
        if reading is None or top[1] < 0.1:
            reading = {"label": "neu", "valence": 0.0, "arousal": 0.0}
        out.append({"label": reading["label"] if top[1] >= 0.1 else "neu",
                    "score": _r(r.get(top[0], 0.0)), "lift": _r(top[1]),
                    "valence": reading["valence"], "arousal": reading["arousal"],
                    "model_label": top_raw[0], "model_score": _r(top_raw[1]),
                    "relative_to": "film_average"})
    return out


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
                  emotion_backend=None, audio_loader=None,
                  speaker_model=DEFAULT_SPEAKER_MODEL, speaker_backend=None):
    """Add ``voice`` and ``speaker`` to each speech segment in place. Returns stage metadata."""
    np = _np()
    segments = sorted(speech, key=lambda s: (s["start_seconds"], s["end_seconds"]))
    if not segments:
        return {"segments": 0, "speakers": 0, "emotion_model": None}
    if emotion_backend is None and emotion_model:
        emotion_backend = load_emotion_backend(emotion_model)
    if speaker_backend is None and speaker_model:
        try:
            speaker_backend = load_speaker_backend(speaker_model)
        except Exception as exc:  # model download or import problems: fall back, say so
            print(f"WARNING: speaker model unavailable ({exc}); using MFCC voice fingerprints",
                  file=sys.stderr)
            speaker_backend = None
    loader = audio_loader or (lambda: stream_segments(video, segments))
    fingerprints, embeddings, raw_emotions = [], [], []
    try:
        for seg, samples in zip(segments, loader()):
            info = prosody(samples, seg.get("text", "")) or {}
            scores = None
            if emotion_backend is not None and len(samples) >= RATE // 2:
                scores = emotion_backend(samples, RATE)
            raw_emotions.append(scores)
            seg["voice"] = {**info, "emotion": None}
            long_enough = len(samples) >= RATE // 2
            fingerprints.append(voice_fingerprint(samples, pitch_hz=info.get("pitch_median_hz"))
                                if long_enough else None)
            embeddings.append(speaker_backend(samples, RATE)
                              if speaker_backend is not None and long_enough else None)
    finally:
        for backend in (emotion_backend, speaker_backend):
            close = getattr(backend, "close", None)
            if close:
                close()
    for seg, reading in zip(segments, relative_emotions(raw_emotions)):
        seg["voice"]["emotion"] = reading
    if speaker_backend is not None:
        seconds = [s["end_seconds"] - s["start_seconds"] for s in segments]
        core = [i for i, e in enumerate(embeddings) if e is not None and seconds[i] >= MIN_SPEAKER_SECONDS]
        labels = dict(zip(core, cluster_embeddings([embeddings[i] for i in core])))
        # Short lines join the closest group if they are close enough.
        groups = {}
        for i, label in labels.items():
            groups.setdefault(label, []).append(embeddings[i])
        centres = {k: np.mean(v, axis=0) / max(1e-9, float(np.linalg.norm(np.mean(v, axis=0))))
                   for k, v in groups.items()}
        for i, e in enumerate(embeddings):
            if e is None or i in labels or not centres:
                continue
            label, sim = max(((k, float(c @ e)) for k, c in centres.items()), key=lambda kv: kv[1])
            if sim >= SPEAKER_THRESHOLD:
                labels[i] = label
        method, source = "wavlm_xvector_average_linkage", "speaker_embedding_cluster"
    else:
        usable = [i for i, f in enumerate(fingerprints) if f is not None]
        labels = dict(zip(usable, cluster_speakers([fingerprints[i] for i in usable])))
        method, source = "mfcc_pitch_average_linkage", "voice_fingerprint_cluster"
    for i, label in labels.items():
        segments[i]["speaker"] = label
        segments[i]["speaker_source"] = source
    groups = set(labels.values())
    print(f"Voice: {len(segments)} segments, {len(groups)} voice groups", file=sys.stderr)
    return {"segments": len(segments), "speakers": len(groups),
            "emotion_model": emotion_model if emotion_backend is not None else None,
            "emotion_scale": "relative_to_film_average",
            "speaker_model": speaker_model if speaker_backend is not None else None,
            "speaker_method": method,
            "speaker_threshold": SPEAKER_THRESHOLD if speaker_backend is not None else None}
