# Visual mood with a local vision model (stage 7)

Replaces the earlier approach (BLIP captions plus three stills per 12 s sent
to `gemma3:4b`). Each 5-second interval now gets a **sequence of frames**
(default 6, 512 px wide) and the interval's measured signals, and the model
fills fixed feeling fields under a JSON schema.

## Setup (12 GB GPU, e.g. RTX 5080)

1. Install Ollama (https://ollama.com/download) and keep it running.
2. `ollama pull qwen3.5:9b` (about 7 GB at Q4_K_M; Apache 2.0).
3. From the activated venv in the repository:

```powershell
python -m true_experience "C:\Videos\clip.mp4" --visual-mood-model qwen3.5:9b -o clip_feeling.json
```

`--visual-mood-model` implies `--feeling-signals`. Use
`--visual-mood-frames 1..8` to trade detail for speed. Fallback model:
`qwen3-vl:8b`. Worth comparing: `gemma4:e4b`.

Frames are sent only to Ollama on 127.0.0.1. The model is unloaded when the
stage finishes so later GPU stages get the whole card.

## Output

`timeline[i].feeling_signals.visual_mood`:

| Field | Values |
|---|---|
| `lighting` | bright, normal, dim, dark, mixed |
| `colour_mood` | short text |
| `people_visible` | integer |
| `facial_expressions` | `[{person: "P1", expression}]`, expression in neutral, happy, sad, angry, afraid, surprised, disgusted, tense, unclear |
| `body_language` | short text |
| `action_intensity` | still, low, medium, high, frantic |
| `camera` | static, slow_move, fast_move, handheld, unclear |
| `mood_words` | up to 3 words |
| `visual_valence`, `visual_arousal` | -1..1 |
| `confidence` | 0..1 |
| `evidence` | one sentence naming the visible details |
| `model`, `frame_times_seconds` | provenance |

Invalid answers are retried once, then the interval abstains (`null`) instead
of guessing. Fusion uses the reading as the `visual` modality (weight 0.6 times
the model's confidence x 0.8) next to the low-level `look` measurements, and it
counts toward `possible_irony`.

## Not yet measured

Speed and peak VRAM on the RTX 5080, and whether the moods match what a viewer
feels. Both need a run on real clips; please share timings from the first run.
