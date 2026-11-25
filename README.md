# The True Experience

AI that experiences a film the way a viewer does: what is seen, what is heard,
how lines are spoken, how the music and the editing move, and how all of that
feels from moment to moment. Everything runs locally on one 12 GB GPU and
produces a timestamped JSON timeline that a language model can reason over.

## Where the project is now

Development is **feeling first**: the system builds an emotional timeline of
the film before it tries to follow the story. The full design is in
[docs/FEELING_PIPELINE.md](docs/FEELING_PIPELINE.md).

| Layer | Status | Runs on |
|---|---|---|
| Shots, silence, 5 s intervals (FFmpeg) | working | CPU |
| Speech to text (faster-whisper) | working | GPU or CPU |
| Pacing, brightness, colour, motion, loudness, startles, tempo, key | working | CPU |
| Visual mood from frame sequences (Qwen3.5 9B through Ollama) | working | GPU, about 7-8 GB |
| Voice: pitch, rate, loudness, voice groups, voice emotion | working, needs tuning on real dialogue | CPU + small GPU model |
| Fusion into one emotional timeline (valence, arousal, tension, label, conflicts, irony, turning points) | working, first version | CPU |
| Separating dialogue from music (Demucs), feeding speech, voice and music | working, first version | GPU, about 1-2 GB |
| Music mood and sound events (CLAP, PANNs) | planned | GPU |
| Characters: face identity, clothing or suit identity, one ID per character, voice link | working, first version | CPU |
| Character roles: main character, hero, villain, journalist, business person ... | working, first version | GPU (same vision model) |
| Scenes and narrative arc | later | |
| "Talk about the film" interpretation | later | |

Measured on an RTX 5080 (12 GB) with a 2:29 trailer: about 105 s in total,
peak VRAM 8.7 GB, roughly 40-45 s of processing per minute of video. Only one
model is on the GPU at a time and each is unloaded after its stage.

All readings are model and signal **estimates**, not facts about the film.

## Install

You need **Python 3.11**, plus **FFmpeg and FFprobe** on your `PATH`. No
editable install (`-e`) is needed; everything runs with `python -m` from the
repository folder.

**Windows PowerShell:**

```powershell
git clone https://github.com/JanneHonkonen/The-True-Experience.git
cd The-True-Experience
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

**Linux / macOS:**

```bash
git clone https://github.com/JanneHonkonen/The-True-Experience.git
cd The-True-Experience
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

For visual mood, install [Ollama](https://ollama.com/download) and pull the
vision model once:

```powershell
ollama pull qwen3.5:9b
```

The first run also downloads the Whisper and voice-emotion weights, and
`--characters` downloads about 180 MB of OpenCV models once, and separation
downloads the Demucs weights (about 80 MB) once.

## Run the feeling timeline

The full feeling run on a local video:

```powershell
python -m true_experience "C:\Videos\clip.mp4" --voice --whisper-model large-v3-turbo --visual-mood-model qwen3.5:9b --character-roles qwen3.5:9b -o clip_feeling.json
```

Lighter variants:

| Command adds | What you get | GPU |
|---|---|---|
| `--feeling-signals` | pacing, look and sound signals fused into a timeline | none |
| `--visual-mood-model qwen3.5:9b` | adds visual mood per interval (implies `--feeling-signals`) | about 7-8 GB |
| `--voice` | adds transcript, prosody, voice groups and voice emotion (implies `--transcribe`, `--feeling-signals` and `--separate-audio`) | about 3 GB, then freed |
| `--separate-audio` | splits dialogue from music with Demucs first, so speech, voice and music are measured on clean stems (`--no-separate-audio` turns it off) | about 1-2 GB, then freed |
| `--voice-emotion-model none` | keeps prosody and voice groups, skips the emotion model | |
| `--characters` | finds people, follows them by face and by clothing, and gives each character one ID | none |
| `--character-roles qwen3.5:9b` | adds each character's role and apparent occupation (implies `--characters`) | the vision model again |
| `--visual-mood-frames N` | frames per interval sent to the vision model (1-8, default 6) | |

Run `python -m true_experience --help` for every option. The command prints a
timing line per stage, and the same numbers are saved in
`metadata.timing_seconds`.

### What the output contains

- `timeline[]`: one entry per 5 s interval, with shots, speech, measured
  `feeling_signals` and, when the stages ran, `feeling` and `emotion`.
  - `feeling.pacing`, `.look`, `.visual`, `.voice`, `.music`, `.sound`: one
    reading per modality, each with valence, arousal, confidence and what it
    was based on.
  - `feeling.fused`: the combined valence, arousal, tension, confidence and an
    emotion label (for example tension, excitement, melancholy, calm, shock),
    plus `conflicts` where modalities disagree and `possible_irony` (bright
    music over a dark scene).
- `speech_segments[]`: transcript lines with `voice` details and a `speaker`
  voice group (S1, S2 ...).
- `characters[]`: each character `C1`, `C2` ... with their face IDs, outfit IDs
  and colours, screen time, voice groups, prominence, `main_character` and,
  with `--character-roles`, their `role`. `character_tracks[]` lists every
  appearance, and each interval has `characters_present`.
- `feeling_curve`: valence, arousal, tension and label over time, the film's
  baselines and its turning points.

Stage guides: [feeling signals](docs/FEELING_SIGNALS.md),
[visual mood](docs/VISUAL_MOOD.md), [voice](docs/VOICE.md),
[characters](docs/CHARACTERS.md), [separation](docs/SEPARATION.md).

## Other tools in the repository

These come from the earlier evidence-first work and still work.

- **Review report with rolling context**
  (`python -m true_experience.experience video.mp4 --context-model gemma3:4b`):
  a shareable review JSON without the transcript, plus a private file where a
  vision model reads frames in 12 s windows with short-term memory and cites
  its evidence. Output names default to `<video name>_review.json`; add
  `--title` and `--video-id` (YouTube) only if you want them in the report.
  See [docs/CONTEXT_REASONING.md](docs/CONTEXT_REASONING.md).
- **Shot image archive** (`python -m true_experience.shot_archive`): a ZIP of
  timestamped JPEGs for every shot, to check model claims against the real
  frames. See [docs/SHOT_THUMBNAILS.md](docs/SHOT_THUMBNAILS.md).
- **Per-shot vision review** (`python -m true_experience.vision_review shots.zip`):
  detailed descriptions of the archived shots with a local Ollama model, or
  the OpenAI API only with explicit `--confirm-upload`. See
  [docs/VISION_REVIEW_BACKENDS.md](docs/VISION_REVIEW_BACKENDS.md).
- **Evaluation helpers**: see [docs/EVALUATION.md](docs/EVALUATION.md).

The older BLIP frame captions (`--caption-frames`) are still available but are
being replaced by the visual mood stage.

## Privacy and rights

Videos stay on your computer. The local Ollama and Hugging Face models do not
upload anything. Characters are grouped within one video only: nobody is
named or looked up, and face data is never written to the output. Only the
optional OpenAI review backend sends frames, and only after you confirm. Files ending in `_private.json` can contain dialogue and are
excluded from Git. Use only media you are allowed to process.

## Troubleshooting

- **"Speech recognition requires ..."**: the message names the real import
  error and the Python that ran. Activate the `.venv` and run
  `python -m pip install -r requirements.txt` again.
- **`open() got an unexpected keyword argument 'metadata_errors'`**: PyAV is
  too new for faster-whisper. Reinstalling from `requirements.txt` pins it
  below 19 (`python -m pip show av` should say 18.x or earlier).
- **`'.[all]' is not a valid editable requirement`**: install with
  `python -m pip install -r requirements.txt` instead.
- **Visual mood fails to connect**: start Ollama and check that
  `ollama list` shows `qwen3.5:9b`.

## The concept: AI that watches film like a human

### Introduction
Cinema is one of humanity’s richest forms of storytelling. A film is not a mere sequence of images and sounds; it is a carefully constructed emotional and narrative experience. Humans perceive it as a living timeline of moods, rhythms, intentions, and feelings.  Traditional AI systems do not “watch” films. They parse subtitles, detect objects, or classify scenes in isolation. These approaches miss the evolving emotional structure that makes cinema meaningful. The True Experience is a conceptual and technical framework designed to close this gap.  
Its goal is to enable AI systems to perceive and interpret video content in a way that more closely mirrors human film comprehension - through synchronized analysis of visuals, sound, rhythm, and emotion over time.

### Core Philosophy
Human film experience emerges from the fusion of multiple sensory and emotional layers:
- Visual storytelling through framing, light, and movement.  
- Emotional cues in dialogue, music, silence, and ambient sound.  
- Temporal rhythm: pacing, editing structure, buildup, release.  
- Contextual synthesis: irony, contrast, suspense, revelation.  

AI must be able to synthesize these layers dynamically to construct an internal emotional and narrative model of the film - not just detect objects or words.

### Concept Goals
- True multimodal perception. AI should simultaneously process visual, auditory, structural, and emotional signals.
- Timeline-based emotional mapping. The system must generate evolving emotional states across time, not isolated tags.
- Narrative structure understanding. Identify key beats: setup, buildup, conflict, climax, resolution.
- Character, dialogue, and sound comprehension. Not only what is said, but how it is said, where, and what surrounds it.
- Structured representation for LLMs. Convert film experience into clean, timestamped JSON suitable for reasoning, summarization, and deeper analysis.

### Vision
Concept defines AI system that can:
- Generate rich emotional and narrative summaries of any film.  
- Recognize subtle cinematic techniques, such as ironic music or visual foreshadowing.  
- Map entire story arcs for educational, analytical, or accessibility purposes.  
- Serve as true "film watcher" not transcript readers.  

The True Experience is not a single program but a framework: a structured methodology that developers, researchers, and tool creators can implement to give AI a more human-like grasp of moving images.

### Output Concept
The target output is a detailed, timestamped JSON timeline representing:
- Visual mood, composition, and scene types  
- Dialogue content, speaker tone, character presence  
- Music and ambient sound emotional cues  
- Sound effects and narrative triggers  
- Combined emotional state of each segment  
- Narrative function and structural position

This structured output acts as a bridge between cinematic expression and language models, enabling higher-level reasoning and analysis.

### Potential Applications
- Film analysis tools: Emotional and structural breakdown for critics or educators.  
- Filmmaking support: Directors can run emotional audits on their cuts.  
- Accessibility: Emotion-aware descriptions for visually impaired audiences.  
- Cultural archiving: Systematic emotional maps of films for research.  
- Adaptive AI systems: Robots or agents that perceive audiovisual media with contextual depth.

## License

This project is released under a **Non-Commercial License** for research, personal, and educational use.  
**Commercial use requires a separate commercial license** from the author.

- Free use, modification, and redistribution are allowed for non-commercial purposes.  
- Commercial use, including integration into paid products or services, is prohibited without explicit permission.  
- This license text must be included with any derivative works based on the concept or codebase.  
- Commercial users must obtain a license directly from the author.

**Contact for commercial licensing:**  
- X / Mastodon / Bluesky: [@JanneHonkonen](https://bsky.app/profile/jannehonkonen.com)  
- LinkedIn or email: *first name* at owlkeep dot com

© 2025 B.Eng. Janne Honkonen. All rights reserved.

## Author
B.Eng. Janne Honkonen  
- [Homepage](https://www.jannehonkonen.com)  
- [GitHub](https://github.com/JanneHonkonen)

## Closing Note
AI should not just parse cinema. It should feel it, follow it, and understand it. This concept outlines a path toward that ability - moving from data processing toward genuine emotional and narrative comprehension.
