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
