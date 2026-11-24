# CPU feeling signals (stage 2 and CPU part of stage 5)

```bash
python -m true_experience video.mp4 --feeling-signals -o analysis.json
```

Adds `feeling_signals` to every timeline interval. Runs on the CPU with NumPy
and FFmpeg only; audio and low-resolution frames are streamed one interval at a
time, so memory stays flat for full-length films. A generated 10-minute 720p
test video took about 25 seconds on a cloud CPU.

| Group | Fields | What it tells the later fusion step |
|---|---|---|
| `pacing` | `cuts_in_window`, `cuts_per_minute`, `mean_overlapping_shot_seconds` | Fast cutting raises arousal |
| `look` | `brightness`, `contrast`, `saturation`, `warmth` (red minus blue), `motion` (mean frame difference, ignoring pairs across cuts), `frame_count` | Dark, contrasty images lean tense; warm, saturated ones lean pleasant (weak cue) |
| `audio` | `rms_dbfs`, `peak_short_term_dbfs`, `startle` + `startle_times_seconds` (a jump of 15 dB or more above the previous second), `onset_rate_per_second`, `spectral_centroid_hz`, `tempo_bpm` + `tempo_confidence`, `key`, `mode` (`major`, `minor` or `unclear`) + `mode_confidence` | Loudness, shocks, rhythm, brightness of sound, major/minor colour |

These are **measurements, not emotions**. The fusion step (next PR) maps them
to valence, arousal and tension with confidences.

Known limits:
- Audio features use the full mix. Dialogue and effects disturb tempo and key
  until source separation (Demucs) lands.
- Tempo needs at least three clear onsets in the interval; otherwise it is `null`.
- Key/mode is a short-window estimate; steady tones or noise give `unclear`.

## Fusion v1: the emotional timeline

With `--feeling-signals` the signals are also fused (see
`true_experience/fusion.py`). Each interval gets:

- `feeling.<modality>`: a reading from `pacing`, `look`, `music`, `sound`
  with `valence` (-1..1), `arousal` (-1..1), `confidence` and the `basis`
  features it used.
- `feeling.fused`: `valence`, `arousal`, `tension` (0..1), `label`,
  `confidence`, `conflicts` and `possible_irony`.
- `emotion`: the fused label, from a controlled vocabulary: `shock`, `tension`,
  `excitement`, `intensity`, `melancholy`, `calm`, `stillness`, `unease`,
  `warmth`, `neutral`.

The report also gets a top-level `feeling_curve` (valence, arousal, tension and
label per interval) with `turning_points`: mood turns (valence changes sign
with a clear swing) and tension peaks.

How it works:
- Weights follow the specification's priority: voice 1.0 (not yet measured),
  music 0.8, look 0.6, sound 0.5, pacing 0.4 (arousal only), each multiplied
  by the reading's confidence.
- Feeling carries over with a 5-second half-life, so a loud scene still
  colours the next quiet seconds. A startle is felt at once.
- Tension rises with arousal, negative valence, darkness, minor mode, and a
  sudden hush after a loud interval.
- A conflict is recorded when two confident modalities disagree on valence by
  more than 0.8; major-key music over clearly negative visuals is flagged as
  `possible_irony`.

All constants are first guesses, chosen to be easy to read and tune. They will
be calibrated against hand-annotated clips. Confidence stays low until voice
and visual-mood models are added.
