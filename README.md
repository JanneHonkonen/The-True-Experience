# The True Experience  

## Quick start — virtual environment (recommended)

From the repository root, use **Python 3.11** and an isolated environment. FFmpeg and FFprobe must also be installed and available on your system PATH. The project has a plain `requirements.txt`; **no editable installation (`-e`) is needed** when running via `python -m`.

**Linux / macOS:**

```bash
git clone https://github.com/JanneHonkonen/The-True-Experience.git
cd The-True-Experience
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m true_experience.local_trailer "/path/to/iron-man-trailer.mp4" -o iron_man_review.json
```

**Windows PowerShell:**

```powershell
git clone https://github.com/JanneHonkonen/The-True-Experience.git
cd The-True-Experience
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m true_experience.local_trailer "C:\Videos\iron-man-trailer.mp4" -o iron_man_review.json
```

If the trailer command fails with `open() got an unexpected keyword argument 'metadata_errors'`, run `python -m pip install -r requirements.txt` again. This project now constrains PyAV below 19 for faster-whisper compatibility. Check with `python -m pip show av` (expected PyAV 18.x or earlier).

### Experimental: time-linked understanding with local visual reasoning

For **multiple actual frames per time window, short-term memory, cited
visual/audio evidence and tentative continuity**, install [Ollama](https://ollama.com/download),
pull a vision-capable model such as `ollama pull gemma3:4b`, then run:

```powershell
python -m true_experience.local_trailer "C:\winstuff\ai\iron_man_trailer.mp4" --whisper-model small --max-caption-frames 60 --caption-spacing 4 --context-model gemma3:4b --context-window 12 -o iron_man_review.json --context-output iron_man_context_private.json
```

The model reads sampled frame images and fallible ASR, but its narrative output
remains **unverified hypotheses**, not proven character identity or causality.
The private context file is **excluded from Git** and may paraphrase dialogue.
It is never included in the sanitized public report. See
[rolling context documentation](docs/CONTEXT_REASONING.md).

The first model run downloads Whisper and BLIP weights; allow sufficient disk space and network access. To see available options, run `python -m true_experience.local_trailer --help`. The video remains local. Use only media you're authorized to process; the report contains no source video or full transcript. To exit the environment, use `deactivate`.

If you see **`'.[all]' is not a valid editable requirement`**, use `python -m pip install -r requirements.txt` instead. `'.[all]'` is a *package extras selector*, not a filename; editable installation depends on being in the correct project directory and on a working build setup. The requirements-file command installs dependencies directly.

### Concept Framework for AI to Watch and Understand Film Like a Human

## Introduction
Cinema is one of humanity’s richest forms of storytelling. A film is not a mere sequence of images and sounds; it is a carefully constructed emotional and narrative experience. Humans perceive it as a living timeline of moods, rhythms, intentions, and feelings.  Traditional AI systems do not “watch” films. They parse subtitles, detect objects, or classify scenes in isolation. These approaches miss the evolving emotional structure that makes cinema meaningful. The True Experience is a conceptual and technical framework designed to close this gap.  
Its goal is to enable AI systems to perceive and interpret video content in a way that more closely mirrors human film comprehension - through synchronized analysis of visuals, sound, rhythm, and emotion over time.

## Core Philosophy
Human film experience emerges from the fusion of multiple sensory and emotional layers:
- Visual storytelling through framing, light, and movement.  
- Emotional cues in dialogue, music, silence, and ambient sound.  
- Temporal rhythm: pacing, editing structure, buildup, release.  
- Contextual synthesis: irony, contrast, suspense, revelation.  

AI must be able to synthesize these layers dynamically to construct an internal emotional and narrative model of the film - not just detect objects or words.

## Concept Goals
- True multimodal perception. AI should simultaneously process visual, auditory, structural, and emotional signals.
- Timeline-based emotional mapping. The system must generate evolving emotional states across time, not isolated tags.
- Narrative structure understanding. Identify key beats: setup, buildup, conflict, climax, resolution.
- Character, dialogue, and sound comprehension. Not only what is said, but how it is said, where, and what surrounds it.
- Structured representation for LLMs. Convert film experience into clean, timestamped JSON suitable for reasoning, summarization, and deeper analysis.

## Vision
Concept defines AI system that can:
- Generate rich emotional and narrative summaries of any film.  
- Recognize subtle cinematic techniques, such as ironic music or visual foreshadowing.  
- Map entire story arcs for educational, analytical, or accessibility purposes.  
- Serve as true "film watcher" not transcript readers.  

The True Experience is not a single program but a framework: a structured methodology that developers, researchers, and tool creators can implement to give AI a more human-like grasp of moving images.

## Output Concept
The target output is a detailed, timestamped JSON timeline representing:
- Visual mood, composition, and scene types  
- Dialogue content, speaker tone, character presence  
- Music and ambient sound emotional cues  
- Sound effects and narrative triggers  
- Combined emotional state of each segment  
- Narrative function and structural position

This structured output acts as a bridge between cinematic expression and language models, enabling higher-level reasoning and analysis.

## Potential Applications
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
