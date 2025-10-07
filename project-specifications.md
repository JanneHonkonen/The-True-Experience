## **PROJECT SPECIFICATION**

**Title:** *The True Experience – AI Film Analysis System*
**Goal:** Process one video file and generate a fully detailed, timestamped JSON summary of all visual events, dialogue, characters, sound, music, emotional content, and narrative structure, suitable for LLM understanding and further processing.

---

### **1. INPUT HANDLING**

* **Input:** One video file (local or stream).
* **Preprocessing:**

  * Extract video frames at base framerate or downsample to fixed interval (e.g., 4–6 fps).
  * Extract audio track in full fidelity (e.g., 44.1 kHz).
* **Output:** Synced video and audio streams with shared timeline reference.

---

### **2. VIDEO SEGMENTATION & ANALYSIS**

**2.1 Scene & Shot Segmentation**

* Detect shot boundaries using a combination of:

  * Histogram delta > threshold
  * SSIM drop
  * Optional: PySceneDetect for standardized shot detection
* Merge shots into *scenes* based on continuity and time proximity.
* Store:

  * `start_time`, `end_time`
  * `scene_id`, `shot_ids`
  * Visual change metrics

**2.2 Visual Content Analysis per Segment**

* Run vision-language model (e.g., CLIP, BLIP) to generate:

  * Environment description (e.g., “dimly lit hallway”)
  * Detected objects
  * Character presence (with bounding boxes / IDs)
* Detect cinematographic elements:

  * Camera movement (static, pan, zoom, handheld)
  * Color palette / brightness
  * Visual rhythm (shot length, cuts/minute)
* Classify segment type: `dialogue`, `action`, `establishing`, `montage`, etc.

---

### **3. AUDIO ANALYSIS**

Split audio into **dialogue**, **sound effects**, **music/ambience** via source separation (e.g., Demucs).

**3.1 Dialogue Processing**

* Apply speech-to-text (Whisper or similar).
* Detect speaker turns, overlaps, interruptions.
* Tag each utterance with:

  * `speaker_id` (consistent per character)
  * Text transcript
  * Prosody/emotion tags (pitch, stress, urgency, tone)
* Align utterances with timestamps.

**3.2 Sound Effects**

* Identify common foley/diegetic categories (footsteps, impacts, mechanical, etc.).
* Record each with:

  * `timestamp`
  * `effect_type`
  * `volume/intensity`
  * `narrative_function` (background realism, surprise, emphasis)

**3.3 Music and Ambient**

* Isolate background score + ambient noise.
* Analyze:

  * Musical mood (tense, warm, ominous, etc.)
  * Tempo, key, chord changes
  * Style/genre classification
  * Ambient environment cues (rain, traffic, silence)
* Timestamp mood shifts and musical changes.

---

### **4. MULTIMODAL FUSION**

**Goal:** Combine vision + dialogue + sound + music for each aligned segment.

**Method:**

* Use shared timestamps to align data.
* Merge segment data into unified entries.
* Apply rule-based or ML fusion to derive:

  * **Combined emotional label** (from controlled vocabulary; e.g., Ekman + cinematic tone)
  * **Narrative function** (e.g., conflict buildup, revelation, comedic relief)
  * **Visual–audio contrast detection** (e.g., upbeat music over tragic visuals → irony)

**Conflict Resolution:**

* If emotion labels conflict:

  * Priority order: strong dialogue emotion > music > visual tone
  * Mark conflict flag for later refinement

---

### **5. EMOTIONAL & STRUCTURAL TIMELINE**

**Granularity:** 5–10 s fixed intervals across full video duration.

For each interval:

* Primary visual mood
* Primary audio mood
* Dominant emotion (controlled vocabulary)
* Narrative state (setup, development, climax, resolution, etc.)
* Detected characters present
* Key events summarized in one line

---

### **6. OUTPUT SCHEMA**

**Format:** JSON file
**Structure Example:**

```json
{
  "metadata": {
    "video_duration": 5400,
    "frame_rate": 24,
    "audio_rate": 44100
  },
  "scenes": [
    {
      "scene_id": 1,
      "start": "00:00:00",
      "end": "00:02:34",
      "shots": [...],
      "visual_summary": "...",
      "audio_summary": "...",
      "emotion": "tension",
      "narrative_function": "introduction"
    }
  ],
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
```

---

### **7. ERROR HANDLING**

* Silence → Mark audio mood as `"silence"` explicitly.
* No detected dialogue → Leave `dialogue` array empty.
* Low confidence outputs → Include `confidence` fields per module.
* Corrupted segments → Skip with error flag.

---

### **8. PERFORMANCE GUIDELINES**

* Downsample frames when possible; keep audio full quality.
* Process in parallel batches where possible (vision/audio modules independent).
* GPU recommended for frame analysis, CPU for transcription.

---

### **9. EVALUATION**

* Validate shot detection on reference clips.
* Check alignment accuracy (±50 ms tolerance).
* Benchmark emotion labeling against manually annotated short scenes.
* Ensure JSON passes schema validation and is temporally ordered.

