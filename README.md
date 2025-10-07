```markdown
# The True Experience – AI Film Analysis System

## Overview

**The True Experience** is a modular AI framework that processes a single video file and produces a **fully timestamped JSON summary** of all visual events, dialogue, characters, music, emotional content, and narrative structure.

Unlike traditional transcript parsers or metadata extractors, this system performs **multimodal analysis**—combining computer vision, audio processing, and emotional inference—to represent films in a way that large language models can meaningfully understand.

The output is designed for:
- LLM ingestion and reasoning
- Narrative and tone analysis
- Film education and critique tools
- Emotion-aware recommendation systems
- Advanced accessibility applications

---

## Core Features

- **Scene & Shot Detection**  
  Automatic segmentation using visual change metrics and shot boundary algorithms.

- **Vision Analysis**  
  Frame sampling, object detection, character presence, environment description, and cinematographic classification.

- **Audio Decomposition**  
  Source separation into dialogue, sound effects, and music/ambience.

- **Dialogue Extraction**  
  Speech-to-text transcription, speaker detection, and emotional prosody analysis.

- **Music & Ambience Classification**  
  Mood, tempo, key, genre, and environmental sound detection.

- **Multimodal Fusion**  
  Alignment of visual and audio data into unified emotional and narrative labels.

- **Emotional Timeline Construction**  
  Segment-level summaries (5–10 s granularity) describing the evolving mood, structure, and events.

- **Structured JSON Output**  
  Timestamped data suitable for direct ingestion by LLMs or downstream tools.

---

## System Architecture

```

Video Input
│
├── 1. Vision Module ── Scene & shot segmentation, visual summaries
│
├── 2. Audio Module
│      ├── Dialogue (STT + emotion)
│      ├── Sound effects (classification)
│      └── Music/Ambience (mood analysis)
│
├── 3. Multimodal Fusion ── emotional + narrative synthesis
│
├── 4. Emotional Timeline ── structured segment summaries
│
└── 5. Output JSON ── machine-readable narrative map

````

---

## Output Format

All analysis results are exported as a single JSON file.

Example:

```json
{
  "metadata": {
    "video_duration": 5400,
    "frame_rate": 24,
    "audio_rate": 44100
  },
  "timeline": [
    {
      "start": "00:03:10",
      "end": "00:03:20",
      "visual_mood": "cold and dim",
      "audio_mood": "minor strings, wind",
      "emotion": "loneliness",
      "characters": ["John", "Unknown Figure"],
      "dialogue": [
        {"speaker": "John", "text": "Who's there?", "emotion": "nervous"}
      ],
      "sound_effects": [
        {"type": "door_creak", "intensity": "medium"}
      ],
      "music": {"mood": "eerie", "genre": "orchestral"},
      "narrative_function": "tension buildup"
    }
  ]
}
````

---

## Key Components

| Component               | Task                                                       | Example Tools                 |
| ----------------------- | ---------------------------------------------------------- | ----------------------------- |
| **Video Processing**    | Scene detection, visual analysis                           | OpenCV, CLIP, PySceneDetect   |
| **Speech-to-Text**      | Dialogue transcription                                     | Whisper                       |
| **Audio Separation**    | Isolate dialogue, SFX, music                               | Demucs                        |
| **Emotion Recognition** | Prosody & music mood analysis                              | Custom classifiers, RL models |
| **Fusion & Timeline**   | Temporal alignment, emotional inference, structured output | Custom logic / ML             |

---

## Performance Guidelines

* Downsample video frames for speed; retain audio fidelity.
* Process vision and audio modules in parallel where possible.
* Use GPU acceleration for frame analysis; CPU can handle transcription.
* Target alignment tolerance: **±50 ms**.

---

## Error Handling

* Silence is labeled explicitly.
* Missing data → empty arrays with confidence scores.
* Corrupted segments are skipped but flagged.
* Each module provides confidence levels to aid downstream filtering.

---

## Roadmap

* [ ] Define and publish full JSON Schema for validation.
* [ ] Add multilingual dialogue support.
* [ ] Expand emotion taxonomy for cinematic use cases.
* [ ] Implement reinforcement learning for adaptive film literacy.
* [ ] Create benchmarking dataset for validation.

---

## License

This project is currently released under the "No commericial usage, no forks, no derivated works, without written permission" license.

License will be updated later on for practical reasons.

---

## Author

**B.Eng. Janne Honkonen**

* [Homepage](https://www.jannehonkonen.com)
* [GitHub](https://github.com/JanneHonkonen)

---

## Citation

If you use this framework in academic work or tools, please cite:

> Honkonen, J. (2025). *The True Experience – AI Film Analysis System: A Multimodal Emotional and Structural Understanding Framework for Video and Film.*
