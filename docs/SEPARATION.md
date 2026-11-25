# Audio separation (stage 3)

Films and trailers put music and effects under almost every line. On the full
mix of a 2:29 trailer, one speaker's measured pitch jumped between 110 and
400 Hz, voice grouping found 24 voices in 25 lines, and Whisper turned 20 s of
score into an invented sentence. Separating the soundtrack first fixes the
input of every voice stage.

```powershell
python -m true_experience "C:\Videos\clip.mp4" --voice --visual-mood-model qwen3.5:9b -o clip_feeling.json
```

`--voice` turns separation on. Use `--separate-audio` on its own to get the
cleaner music reading without voice analysis, or `--no-separate-audio` to
skip it.

## What it does

- **Model:** `htdemucs` (Hybrid Transformer Demucs, MIT licence, from the
  `demucs` package). The weights are about 80 MB and download once. It needs
  about 1-2 GB of VRAM, runs on the GPU when PyTorch sees one, and is freed
  after the stage.
- **Streaming:** FFmpeg decodes the soundtrack, which is separated in 60 s
  chunks with 2 s of context on each side. Memory stays flat for a full film.
- **Stems:**
  - `vocals` (16 kHz mono) goes to Whisper and to voice analysis (pitch,
    loudness, voice groups, voice emotion).
  - `music` (drums + bass + other, 22.05 kHz mono) gives tempo, key, mode and
    onsets per interval in `feeling_signals.music`. Fusion trusts it twice as
    much as the full-mix guess, and reads a near-silent stem (below -45 dBFS)
    as "no music".
  - Loudness and startles still come from the full mix, because a jump scare
    is felt in the full mix.
- **Cache:** stems are kept in `~/.cache/true_experience/stems/` per video
  (keyed by path, size and modification time), so repeated runs while tuning
  skip this stage. Delete the folder to free the space (about 3 MB per minute
  of video).

`metadata.separation` records the model, device, audio length and time taken.

## Known limits

- Demucs separates *vocals*, not *dialogue*: sung vocals in the score land in
  the vocals stem, and some effects (screams, crowd noise) do too.
- "Other" mixes ambient effects into the music stem, so loud effects can
  still nudge the music reading.
- On CPU it is slow (several times real time). Use a GPU for full films.
