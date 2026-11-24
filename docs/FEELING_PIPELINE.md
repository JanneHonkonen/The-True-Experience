# Feeling-first pipeline (design, v0.4 target)

Status: **design**. Implemented step by step in follow-up PRs; this file is
updated as stages land.

## Why feeling first

The concept asks for AI that experiences a film as a *living timeline of moods,
rhythms, intentions and feelings*. Versions up to 0.3 built a careful evidence
layer (shots, silence, speech, frame captions, cited window notes), but the
timeline's `emotion` and `narrative_function` fields are always `null`.
Story structure (scenes, characters, arc) depends on knowing how each moment
*feels*, so the emotional layer comes next.

## Hardware target

**Minimum and reference hardware: one NVIDIA RTX 5080 with 12 GB VRAM.**
Every design choice below follows from that:

- At most **one GPU model is loaded at a time**. A stage loads its model,
  processes the whole video, writes results to disk, then releases VRAM
  (`torch.cuda.empty_cache()`, Ollama `keep_alive: 0`) before the next stage.
- No single model may need more than about **7 GB** of VRAM, leaving headroom
  for the desktop, CUDA context and activations.
- Quantized weights where available (int8 for Whisper, Q4 GGUF for Ollama
  models).
- Long audio is processed in chunks so memory does not grow with film length.
- Every stage is cached and resumable, keyed by the video fingerprint and
  stage settings, so a failure late in the run does not redo earlier stages.
- Every GPU stage also has a CPU fallback or can be skipped; the pipeline must
  still produce a (less rich) feeling timeline with no GPU at all.

## Core idea: per-modality readings, fused over time

Each modality produces, per 5-second window on **one shared grid**:

| Field | Meaning |
|---|---|
| `valence` | -1 (unpleasant) .. +1 (pleasant) |
| `arousal` | -1 (calm) .. +1 (intense) |
| `tags` | words from a controlled vocabulary (e.g. `tense`, `warm`, `eerie`) |
| `confidence` | 0..1, how much this reading should count |
| `evidence` | what was measured (feature values, model, timestamps) |

A fusion step combines the modalities into a single felt state, lets feeling
carry over between windows, and flags disagreement between senses.
Language models are used **last**, to put the measured state into words; they
do not create the signals.

## Stages

| # | Stage | Measures | Tool / model | VRAM (approx.) |
|---|---|---|---|---|
| 1 | Decode | 16 kHz mono and 44.1 kHz audio; 2 fps frames at 384 px | FFmpeg | CPU |
| 2 | Rhythm and look | cuts per minute, mean shot length, motion energy (frame difference), brightness, contrast, saturation, colour warmth | NumPy | CPU |
| 3 | Separation | dialogue stem vs. music + effects stem | Demucs `htdemucs`, chunked | ~2 GB |
| 4 | Speech and voice | transcript; speaker turns; pitch, loudness, speaking rate; voice arousal/valence | faster-whisper `large-v3-turbo` int8; speaker embeddings + clustering; small speech-emotion model | ~3 GB |
| 5 | Music | music present, tempo, major/minor mode, energy, onset density, mood tags | librosa (CPU) + CLAP zero-shot prompts | ~1 GB |
| 6 | Sound events | explosions, gunshots, crowd, rain, footsteps; sudden loud onsets ("startle") | PANNs CNN14 (AudioSet labels) | <1 GB |
| 7 | Visual mood | lighting, colour mood, facial expression, action intensity per window | Qwen3-VL 8B Q4 via Ollama | ~6-7 GB |
| 8 | Fusion | fused valence, arousal, tension, label, conflicts, irony | rules (later a small trained model) | CPU |
| 9 | Interpretation | plain-language mood per window, turning points, whole-film emotional arc | Qwen3 8B (text) Q4 via Ollama, reading only the fused timeline | ~5-6 GB |

Model choices are starting points; any replacement must fit the same budget.
Check each model's licence before use (some speech-emotion models are
non-commercial).

### Stage 2 and 5 first: signals that need no GPU

Classical signal processing already says a lot about feeling and runs anywhere:

- **Pacing:** cuts per minute and shot length are strong arousal cues.
- **Motion energy:** mean absolute frame difference.
- **Look:** low brightness and high contrast lean toward tension; warm,
  saturated colour leans toward positive valence (weak cue, low confidence).
- **Loudness:** RMS level and sudden jumps (startle).
- **Music basics:** tempo, onset density, spectral brightness, major/minor
  estimate (key profile correlation).

These form the first implementation step so that a feeling timeline exists
before any extra model is installed.

## Fusion rules (first version)

1. Each modality reading has `valence`, `arousal`, `confidence`.
2. Fused valence/arousal = confidence-weighted mean with priority weights from
   the specification: dialogue/voice 1.0, music 0.8, visual 0.6, sound 0.5,
   pacing 0.4 (pacing informs arousal only).
3. **Carry-over:** felt state decays toward the new reading
   (`state = a * previous + (1 - a) * current`, `a` about 0.5 per 5 s), so a
   shock still colours the following quiet seconds.
4. **Tension** rises with arousal, minor or dissonant music, low light, fast
   cutting, and silence right after loud events.
5. **Conflict flag** when two modalities with confidence above 0.5 disagree in
   valence by more than 0.8. If music is clearly positive while visuals or
   dialogue are clearly negative, mark `possible_irony`.
6. The label comes from a controlled vocabulary mapped from the
   valence/arousal plane (e.g. high arousal + negative valence: `fear`,
   `anger`, `tension`), refined by tags.

## Output (added to the existing JSON)

```json
"timeline": [{
  "start_seconds": 190.0, "end_seconds": 195.0,
  "feeling": {
    "pacing":  {"cuts_per_minute": 24, "arousal": 0.6, "confidence": 0.7},
    "look":    {"brightness": 0.18, "warmth": -0.3, "valence": -0.3, "arousal": 0.1, "confidence": 0.4},
    "music":   {"present": true, "tempo_bpm": 70, "mode": "minor", "tags": ["eerie"], "valence": -0.5, "arousal": 0.2, "confidence": 0.6},
    "sound":   {"events": ["wind"], "startle": false, "valence": -0.1, "arousal": 0.0, "confidence": 0.3},
    "voice":   {"speakers": ["S1"], "arousal": 0.5, "valence": -0.4, "confidence": 0.5},
    "fused":   {"valence": -0.4, "arousal": 0.4, "tension": 0.7, "label": "tension",
                "confidence": 0.55, "conflicts": [], "possible_irony": false}
  }
}],
"feeling_curve": {"window_seconds": 5, "valence": [], "arousal": [], "tension": []}
```

## Implementation order

1. Cleanup: general `experience` command, no hardcoded source labels, this plan. *(done)*
2. Stages 1, 2 and CPU part of 5: pacing, look, loudness and music basics.
3. Stage 8: fusion, feeling timeline, tension curve, irony flags.
4. GPU backends one at a time, each optional with a fake backend in tests:
   Demucs, voice emotion, CLAP music mood, PANNs sound events, Qwen3-VL visual mood.
5. Stage 9: interpretation and whole-film emotional arc.
6. Evaluation: hand-annotate 3-5 short, rights-cleared clips (felt emotion per
   window, music mood, startle moments) and score each stage against them.
7. Record real runtimes and peak VRAM on the RTX 5080 for a trailer and a
   full-length film.
