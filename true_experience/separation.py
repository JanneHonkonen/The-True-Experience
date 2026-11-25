"""Stage 3: split the soundtrack into dialogue and music (Demucs, GPU if available).

Trailers and films put music and effects under almost every line. Measured on
the full mix, a single speaker's pitch jumped between 110 and 400 Hz, voice
grouping found 24 voices in 25 lines, and Whisper turned 20 s of score into an
invented sentence. Separating the vocals first fixes the input of every later
voice stage.

- Model: ``htdemucs`` (Hybrid Transformer Demucs, MIT licence, about 80 MB,
  downloaded once by the ``demucs`` package). About 1-2 GB of VRAM; freed after
  the stage.
- The soundtrack is decoded by FFmpeg and separated in 60 s chunks with a
  short overlap on each side, so memory stays flat for a full-length film.
- Writes two WAV files: ``vocals`` (16 kHz mono, for Whisper and voice) and
  ``music`` (22.05 kHz mono, drums + bass + other, for tempo, key and
  harmony). Loudness and startles still use the full mix, since a jump scare
  is felt in the full mix.
- Stems are cached per video (by size and modification time), so repeated
  runs while tuning skip this stage.
"""
from __future__ import annotations

import hashlib
import subprocess
import sys
import time
from pathlib import Path

RATE = 44100
CHUNK_SECONDS = 60.0
PAD_SECONDS = 2.0
VOCALS_RATE = 16000
MUSIC_RATE = 22050
DEFAULT_MODEL = "htdemucs"


def _np():
    try:
        import numpy
    except ImportError as exc:
        raise RuntimeError("Audio separation requires NumPy: python -m pip install numpy") from exc
    return numpy


def cache_dir(video, root=None):
    """A folder per video version; reused while the file is unchanged."""
    video = Path(video)
    stat = video.stat()
    key = hashlib.sha1(f"{video.resolve()}|{stat.st_size}|{int(stat.st_mtime)}".encode()).hexdigest()[:12]
    base = Path(root) if root else Path.home() / ".cache" / "true_experience" / "stems"
    return base / f"{video.stem[:40]}_{key}"


def load_demucs(model=DEFAULT_MODEL):
    """Returns separate(stereo[2, n] float32 at 44.1 kHz) -> (vocals[n], music[n])."""
    try:
        import torch
        from demucs.apply import apply_model
        from demucs.pretrained import get_model
    except ImportError as exc:
        raise RuntimeError("Audio separation needs demucs and torch "
                           "(python -m pip install -r requirements.txt). "
                           f"Python: {sys.executable}. Import error: {exc}") from exc
    np = _np()
    net = get_model(model)
    net.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if getattr(net, "samplerate", RATE) != RATE:
        raise RuntimeError(f"{model} expects {net.samplerate} Hz audio")
    vocal = net.sources.index("vocals")

    def separate(stereo):
        wav = torch.from_numpy(np.ascontiguousarray(stereo, dtype=np.float32))
        ref = wav.mean(0)
        mean, std = ref.mean(), ref.std() + 1e-8
        with torch.no_grad():
            out = apply_model(net, ((wav - mean) / std)[None], device=device, shifts=0,
                              split=True, overlap=0.25, progress=False)[0]
        out = (out * std + mean).mean(1)  # sources x samples, mono
        vocals = out[vocal]
        music = out.sum(0) - vocals
        return vocals.cpu().numpy(), music.cpu().numpy()

    def close():
        nonlocal net
        net = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    separate.close = close
    separate.device = device
    return separate


def _reader(video):
    cmd = ["ffmpeg", "-hide_banner", "-nostdin", "-v", "error", "-i", str(video),
           "-map", "0:a:0", "-vn", "-ac", "2", "-ar", str(RATE), "-f", "f32le", "pipe:1"]
    return subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)


def _writer(path, rate):
    cmd = ["ffmpeg", "-hide_banner", "-nostdin", "-v", "error", "-y", "-f", "f32le", "-ar", str(RATE),
           "-ac", "1", "-i", "pipe:0", "-ar", str(rate), "-c:a", "pcm_s16le", str(path)]
    return subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.DEVNULL)


def stream_chunks(proc, chunk=CHUNK_SECONDS, pad=PAD_SECONDS):
    """Yield (stereo with context, samples of context before, samples of the chunk itself)."""
    np = _np()
    n_chunk, n_pad = int(chunk * RATE), int(pad * RATE)
    frame = 8  # bytes per stereo float32 sample

    def read(n):
        data = proc.stdout.read(n * frame)
        return np.frombuffer(data[:len(data) // frame * frame], np.float32).reshape(-1, 2).T

    before = np.zeros((2, 0), np.float32)
    body = read(n_chunk)
    while body.shape[1]:
        after = read(n_chunk)
        context = np.concatenate([before, body, after[:, :n_pad]], axis=1)
        yield context, before.shape[1], body.shape[1]
        before = body[:, -n_pad:]
        body = after


def separate_audio(video, *, model=DEFAULT_MODEL, cache_root=None, backend=None, reuse=True):
    """Write vocals and music stems for ``video``. Returns paths and stage metadata."""
    out = cache_dir(video, cache_root)
    vocals_path, music_path = out / "vocals.wav", out / "music.wav"
    done = out / "complete.txt"
    if reuse and done.exists() and vocals_path.exists() and music_path.exists():
        print(f"Separation: reusing stems in {out}", file=sys.stderr)
        return {"vocals_path": vocals_path, "music_path": music_path,
                "settings": {"model": done.read_text(encoding="utf-8").strip(), "reused": True,
                             "vocals_rate": VOCALS_RATE, "music_rate": MUSIC_RATE}}
    out.mkdir(parents=True, exist_ok=True)
    done.unlink(missing_ok=True)
    separate = backend or load_demucs(model)
    began = time.perf_counter()
    seconds = 0.0
    reader = _reader(video)
    writers = [_writer(vocals_path, VOCALS_RATE), _writer(music_path, MUSIC_RATE)]
    try:
        for context, start, length in stream_chunks(reader):
            vocals, music = separate(context)
            for w, stem in zip(writers, (vocals, music)):
                w.stdin.write(stem[start:start + length].astype("<f4").tobytes())
            seconds += length / RATE
            print(f"Separation: {seconds / 60:.1f} min done", file=sys.stderr)
    finally:
        reader.stdout.close()
        reader.wait()
        for w in writers:
            w.stdin.close()
            w.wait()
        close = getattr(separate, "close", None)
        if close:
            close()
    if seconds <= 0:
        raise RuntimeError("Separation got no audio from the video")
    if any(w.returncode for w in writers):
        raise RuntimeError("Writing the separated stems failed")
    done.write_text(model, encoding="utf-8")
    return {"vocals_path": vocals_path, "music_path": music_path,
            "settings": {"model": model, "reused": False, "audio_seconds": round(seconds, 2),
                         "device": getattr(separate, "device", None),
                         "separation_seconds": round(time.perf_counter() - began, 2),
                         "vocals_rate": VOCALS_RATE, "music_rate": MUSIC_RATE,
                         "chunk_seconds": CHUNK_SECONDS, "pad_seconds": PAD_SECONDS}}
