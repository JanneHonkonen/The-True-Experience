# Dialogue and voice (stage 4)

How lines are spoken carries much of a scene's feeling. `--voice` adds this to
the emotional timeline.

```powershell
python -m true_experience "C:\Videos\clip.mp4" --voice --visual-mood-model qwen3.5:9b --whisper-model large-v3-turbo -o clip_feeling.json
```

`--voice` implies `--transcribe` and `--feeling-signals`. Stages run one after
another, so only one model is on the GPU at a time: Whisper (about 2-3 GB for
`large-v3-turbo`), then the voice-emotion model (about 0.4 GB, freed after
use), then Qwen3.5 (about 7-8 GB).

For every speech segment, `speech_segments[i]` gains:

| Field | Meaning |
|---|---|
| `voice.pitch_median_hz` | typical pitch |
| `voice.pitch_variability_semitones` | how much the pitch moves; flat ~1-2, animated 4+ |
| `voice.loudness_dbfs` | level of the segment (full mix) |
| `voice.words_per_second` | speaking rate from the transcript |
| `voice.pause_share`, `voice.voiced_share` | pauses and voiced frames |
| `voice.emotion` | `label`, `score`, `valence`, `arousal` from the voice-emotion model |
| `speaker` | voice group `S1`, `S2` ... (`speaker_source: voice_fingerprint_cluster`) |

Fusion adds a `voice` reading per interval (weight 1.0, the highest, as the
specification asks), with the emotions heard and the voice groups present.
Arousal mixes pitch movement, speaking rate and loudness (relative to the
film's typical dialogue level) with the model's arousal; valence comes from the
voice-emotion model.

Options:
- `--voice-emotion-model none` skips the model (prosody and voice groups only).
- `--voice-emotion-model <hf id>` picks another audio-classification model.
  The default, `superb/wav2vec2-base-superb-er`, knows neutral, happy, angry
  and sad. Check the licence of any replacement.

Known limits:
- With `--voice`, analysis runs on the Demucs vocals stem (see
  [SEPARATION.md](SEPARATION.md)). With `--no-separate-audio` it falls back to
  the full mix, where music and explosions disturb pitch and emotion: on a
  trailer, one speaker's pitch read anywhere from 110 to 400 Hz.
- Voice groups are a rough fingerprint (MFCC statistics plus pitch), not true
  speaker recognition. Good for clearly different voices, unreliable for
  similar ones. A proper speaker-embedding model can replace it later.
- The default emotion model was trained on acted English speech.
