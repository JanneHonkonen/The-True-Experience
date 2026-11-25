"""Stage 9d: timed narration, as a viewer sinking into the film (local VLM).

The earlier stages measure and label: who is on screen, what they say and how,
the music, the light, the feeling curve, what each scene is. This stage turns
that into what a person would tell you after watching: short timed passages
in the present tense that describe what is seen and heard *and* how it feels,
with the real lines quoted.

One call per scene, with the same Ollama model as the scene readings, after
names and the arc are known. Each call gets:

- up to 6 frames across the scene,
- the cast as names or plain descriptions (never IDs),
- the dialogue with times, who says it and how it is delivered (calm, sharp,
  fast, flat ...) from the voice stage,
- the music in words (tempo, energy, loudness, major or minor) and the
  measured feeling at the start and end,
- the scene reading, the logline, and the last passage before this scene so
  the narration flows.

Quotes are checked against the transcript: a passage that quotes a line nobody
said is sent back once, then replaced by the scene summary.
"""
from __future__ import annotations

import difflib
import json
import re
import statistics
import sys
import time

from .visual_mood import DEFAULT_MODEL, extract_window_jpegs, ollama_chat, unload_model

SCHEMA = {
    "type": "object", "additionalProperties": False,
    "properties": {
        "passages": {"type": "array", "minItems": 1, "maxItems": 4, "items": {
            "type": "object", "additionalProperties": False,
            "properties": {"t": {"type": "number"},
                           "text": {"type": "string", "maxLength": 700}},
            "required": ["t", "text"]}},
    },
    "required": ["passages"],
}

PROMPT = """You are watching a film and telling a friend what it is like, moment by moment.
You get one scene: frames in time order and facts measured from it. Write 1 to 4 short timed
passages (t = seconds where the passage starts, inside the scene) that let the reader sink into
the scene: what is seen, what is heard, and above all how it feels.
Style:
- Present tense, plain vivid language, like a person describing it, not a report.
- Weave in the mood: the light, the pace, the music's energy, the tension, people's expressions
  and how lines are delivered ("she asks, almost accusing"; "he answers, deadpan").
- Quote dialogue exactly as given, in double quotes, with who says it. Quote only given lines,
  shorten by leaving words out with "...", never reword inside quotes.
- Call people by the given name or description. Never write IDs such as C1, S2 or SC3.
  When a speaker is unknown, describe them from the frames ("a blonde reporter").
- Do not invent plot, places or names that the frames and facts do not support. Never name
  actors or real people. Describe music by its feel, not by guessing a band or song.
- A title card, rating card or logo is said simply ("The green rating card: this is a trailer").
- First passage of a scene that continues the last one should flow on from it, not restart.
- About 25 to 90 words per passage; more passages for long or eventful scenes.
Answer with JSON matching the schema only."""

EMOTION_WORDS = {"ang": "sharp, edgy", "hap": "bright, warm", "sad": "low, sad", "neu": "even",
                 "fea": "afraid", "sur": "surprised", "dis": "disgusted"}


def mmss(seconds):
    seconds = max(0, int(round(seconds or 0)))
    return f"{seconds // 60:02d}:{seconds % 60:02d}"


def describe_person(character):
    """A name, or a short plain description from the role pass."""
    if not character:
        return None
    name = (character.get("name") or {}).get("name")
    if name:
        return name
    role = character.get("role") or {}
    body = role.get("physical") or {}
    gender = body.get("apparent_gender")
    noun = gender if gender in ("man", "woman", "boy", "girl") else "person"
    age = body.get("apparent_age_range")
    look = role.get("appearance")
    clause = ""
    if isinstance(look, str) and look.strip():
        clause = " ".join(re.split(r"[,;.]", look.strip())[0].split()[:12])
    clause = re.sub(r"^(?:a|an|the)\s+", "", clause, flags=re.I)
    clause = clause[:1].lower() + clause[1:]
    pronoun = {"man": "his", "boy": "his", "woman": "her", "girl": "her"}.get(noun, "their")
    head = f"{noun} in {pronoun} {age}" if age and age[0].isdigit() else noun
    if noun == "person" and clause:
        return clause  # the clothing sentence already names the person ("blonde woman in ...")
    return f"{head}, {clause}" if clause else head


def delivery(voice):
    """How a line is spoken, in words."""
    if not voice:
        return None
    words = []
    emotion = (voice.get("emotion") or {}).get("label")
    if emotion in EMOTION_WORDS and emotion != "neu":
        words.append(EMOTION_WORDS[emotion])
    rate = voice.get("words_per_second")
    if rate:
        words.append("fast" if rate >= 4.0 else "slow" if rate <= 2.0 else None)
    spread = voice.get("pitch_variability_semitones")
    if spread is not None:
        words.append("flat, deadpan" if spread < 1.0 else "lively" if spread > 4.0 else None)
    loud = voice.get("loudness_dbfs")
    if loud is not None:
        words.append("loud" if loud > -18 else "quiet" if loud < -34 else None)
    words = [w for w in words if w]
    return ", ".join(words) or "even"


def music_words(windows, film_rms):
    """The music of a scene in a few words, from the measured music features."""
    bits = []
    for w in windows:
        m = (w.get("feeling_signals") or {}).get("music") or {}
        rms = m.get("rms_dbfs")
        if rms is None or rms < -45:
            bits.append("no music")
            continue
        tempo = m.get("tempo_bpm") if (m.get("tempo_confidence") or 0) >= 0.3 else None
        words = ["loud" if film_rms is not None and rms > film_rms + 4 else
                 "soft" if film_rms is not None and rms < film_rms - 6 else None,
                 "driving" if (m.get("onset_rate_per_second") or 0) > 2.0 else
                 "sparse" if (m.get("onset_rate_per_second") or 0) < 0.6 else None,
                 f"about {int(tempo)} bpm" if tempo else None,
                 m.get("mode") if (m.get("mode_confidence") or 0) >= 0.4 else None,
                 "a sudden hit" if m.get("startle") else None]
        bits.append(", ".join(x for x in words if x) or "music")
    out = []
    for b in bits:  # keep order, drop repeats
        if b not in out:
            out.append(b)
    return out[:3]


def _film_music_rms(timeline):
    values = [((w.get("feeling_signals") or {}).get("music") or {}).get("rms_dbfs") for w in timeline]
    values = [v for v in values if v is not None and v >= -45]
    return statistics.median(values) if values else None


def scene_lines(scene, speech, by_id):
    a, b = scene["start_seconds"], scene["end_seconds"]
    out = []
    for s in speech or []:
        if s.get("speech_suspect") or not (a <= (s["start_seconds"] + s["end_seconds"]) / 2 < b):
            continue
        text = (s.get("text") or "").strip()
        if not text:
            continue
        who = describe_person(by_id.get(s.get("character")))
        if not who:
            voice = (s.get("speaker") or "").lstrip("S")
            who = f"unrecognised speaker (voice {voice})" if voice else "unrecognised speaker"
        out.append({"t": round(s["start_seconds"], 1), "time": mmss(s["start_seconds"]),
                    "who": who, "says": text, "delivery": delivery(s.get("voice"))})
    return out


def scene_facts(scene, speech, characters, timeline, story=None, previous=None, film_rms=None):
    by_id = {c["id"]: c for c in characters or []}
    a, b = scene["start_seconds"], scene["end_seconds"]
    windows = [w for w in timeline or [] if w["start_seconds"] < b and w["end_seconds"] > a]
    reading = scene.get("reading") or {}
    looks = []
    for w in windows:
        vm = (w.get("feeling_signals") or {}).get("visual_mood") or {}
        if vm.get("mood_words"):
            looks.extend(m for m in vm["mood_words"] if m not in looks)
    feeling = scene.get("feeling") or {}
    return {
        "scene": {"from": mmss(a), "to": mmss(b), "start_seconds": a, "end_seconds": b,
                  "shots": len(scene.get("shot_ids") or [])},
        "film": {"logline": (story or {}).get("logline"), "form": (story or {}).get("form")},
        "story_so_far": previous,
        "what_happens": reading.get("summary"), "place": reading.get("location"),
        "time_of_day": reading.get("time_of_day"), "scene_type": reading.get("scene_type"),
        "emotional_turn": reading.get("emotional_turn"),
        "people": [by_id[c].get("profile") or describe_person(by_id[c])
                   for c in scene.get("characters", [])[:5] if c in by_id],
        "dialogue": scene_lines(scene, speech, by_id)[:14],
        "music": music_words(windows, film_rms),
        "feeling": {"start": feeling.get("start_label"), "end": feeling.get("end_label"),
                    "tension": feeling.get("tension"), "peak_tension": feeling.get("peak_tension")},
        "visual_mood": looks[:6],
    }


def _norm(text):
    return re.sub(r"[^a-z0-9 ]+", "", text.lower().replace("’", "'")).split()


def quote_is_real(quote, lines):
    """True when every '...'-separated piece of the quote occurs in one of the lines."""
    pieces = [p for p in re.split(r"\.\.\.|…", quote) if _norm(p)]
    if not pieces:
        return True
    corpus = [" ".join(_norm(l)) for l in lines]
    for piece in pieces:
        p = " ".join(_norm(piece))
        if any(p in c for c in corpus):
            continue
        if not any(difflib.SequenceMatcher(None, p, c).find_longest_match(0, len(p), 0, len(c)).size
                   >= 0.85 * len(p) for c in corpus):
            return False
    return True


_QUOTE = re.compile(r"[\"“]([^\"”]{4,})[\"”]")
_ID = re.compile(r"\b(?:SC|C|S)\d{1,3}\b")


def validate(output, start, end, lines):
    if not isinstance(output, dict) or not isinstance(output.get("passages"), list) or not output["passages"]:
        raise ValueError("passages missing")
    out = []
    for p in output["passages"][:4]:
        if not isinstance(p, dict) or not isinstance(p.get("text"), str) or not p["text"].strip():
            raise ValueError("each passage needs text")
        t = p.get("t")
        if isinstance(t, bool) or not isinstance(t, (int, float)):
            raise ValueError("t must be seconds")
        text = " ".join(p["text"].split())[:700]
        for quote in _QUOTE.findall(text):
            if not quote_is_real(quote, lines):
                raise ValueError(f'quote not in the dialogue: "{quote[:60]}"; quote only given lines')
        if _ID.search(text):
            raise ValueError("passages must not contain IDs like C1 or S2; use names or descriptions")
        t = min(max(float(t), start), max(start, end - 0.5))
        out.append({"t": round(t, 2), "time": mmss(t), "text": text})
    out.sort(key=lambda p: p["t"])
    return out


def narrate(video, scenes, speech, characters, timeline, story=None, *, model=DEFAULT_MODEL,
            frames_per_scene=6, timeout=300, backend=None, frame_loader=None, unload=None):
    """Add ``narration`` to every scene. Returns (passages in time order, metadata)."""
    backend = backend or ollama_chat
    unload = unload or unload_model
    loader = frame_loader or (lambda a, b, n: extract_window_jpegs(video, a, b, n))
    film_rms = _film_music_rms(timeline or [])
    previous, fallbacks, began = None, 0, time.perf_counter()
    try:
        for n, scene in enumerate(scenes, 1):
            a, b = scene["start_seconds"], scene["end_seconds"]
            facts = scene_facts(scene, speech, characters, timeline, story, previous, film_rms)
            lines = [d["says"] for d in facts["dialogue"]]
            count = max(1, min(frames_per_scene, int(b - a) or 1))
            payload = {"model": model, "stream": False, "think": False, "format": SCHEMA,
                       "keep_alive": "10m", "options": {"temperature": 0.3, "num_ctx": 8192},
                       "messages": [{"role": "system", "content": PROMPT},
                                    {"role": "user", "images": loader(a, b, count),
                                     "content": json.dumps(facts, ensure_ascii=False)}]}
            passages, error = None, ""
            for _ in range(2):
                try:
                    passages = validate(backend(payload, timeout), a, b, lines)
                    break
                except ValueError as exc:
                    error = str(exc)
                    payload["messages"] = payload["messages"][:2] + [
                        {"role": "user", "content": f"Invalid answer ({error}). Return JSON matching the schema."}]
            if passages is None:
                fallbacks += 1
                summary = (scene.get("reading") or {}).get("summary")
                print(f"WARNING: narration fell back to the summary for {scene['scene_id']} ({error})",
                      file=sys.stderr)
                passages = [{"t": a, "time": mmss(a), "text": summary, "fallback": True}] if summary else []
            scene["narration"] = passages
            if passages:
                previous = passages[-1]["text"]
            print(f"Narration {n}/{len(scenes)}", file=sys.stderr)
    finally:
        unload(model)
    timeline_out = [{**p, "scene_id": s["scene_id"]} for s in scenes for p in s.get("narration") or []]
    return timeline_out, {"model": model, "scenes": len(scenes), "passages": len(timeline_out),
                          "fallbacks": fallbacks, "model_seconds": round(time.perf_counter() - began, 2)}


def render_text(title, narration, story=None):
    """The narration as a readable Markdown text."""
    out = [f"# {title}", ""]
    if story and story.get("logline"):
        out += [f"*{story['logline']}*", ""]
    for p in narration:
        out += [f"**[{p['time']}]** {p['text']}", ""]
    if story and story.get("emotional_journey"):
        out += ["---", "", story["emotional_journey"], ""]
    return "\n".join(out)
