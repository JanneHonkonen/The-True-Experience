"""Stage 9b: what happens in each scene, and the story it adds up to (local VLM).

Two passes with the same Ollama model used for visual mood (default
``qwen3.5:9b``), one model on the GPU at a time:

1. **Scene reading.** Per scene: up to 6 frames spread across it, plus the
   facts the earlier stages measured (characters present with their roles,
   dialogue with who speaks, the fused feeling at the start and end, music
   mode). The model returns a short summary, the place, the kind of scene,
   its story beat and the emotional turn.
2. **Story arc.** One text-only call over all scene readings and the feeling
   curve: a logline, the arc with its turning points, who drives the story
   and who opposes it, the emotional journey, and *hypotheses* that link
   characters the face and clothing stages could not (for example a masked
   figure who is probably the lead in a suit).

Everything the model says is labelled as interpretation; nothing overwrites
the measured data.
"""
from __future__ import annotations

import json
import math
import sys
import time

from .visual_mood import DEFAULT_MODEL, extract_window_jpegs, ollama_chat, unload_model

SCENE_TYPES = ("dialogue", "action", "fight", "chase", "montage", "establishing", "romance",
               "suspense", "comedy", "transition", "title_card", "other")
BEATS = ("setup", "inciting_incident", "rising_action", "confrontation", "crisis", "climax",
         "falling_action", "resolution", "unclear")
TIMES = ("day", "night", "dusk_dawn", "interior_unknown", "unclear")

SCENE_SCHEMA = {
    "type": "object", "additionalProperties": False,
    "properties": {
        "summary": {"type": "string", "maxLength": 300},
        "location": {"type": "string", "maxLength": 80},
        "time_of_day": {"type": "string", "enum": list(TIMES)},
        "scene_type": {"type": "string", "enum": list(SCENE_TYPES)},
        "story_beat": {"type": "string", "enum": list(BEATS)},
        "emotional_turn": {"type": "string", "maxLength": 160},
        "character_actions": {"type": "array", "maxItems": 6, "items": {
            "type": "object", "additionalProperties": False,
            "properties": {"id": {"type": "string", "maxLength": 8},
                           "doing": {"type": "string", "maxLength": 100}},
            "required": ["id", "doing"]}},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
    },
    "required": ["summary", "location", "time_of_day", "scene_type", "story_beat",
                 "emotional_turn", "character_actions", "confidence"],
}

SCENE_PROMPT = """You watch one scene of a film as frames in time order, with facts measured from it.
Say what happens, as a viewer would follow it.
Rules:
- Use what the frames show and the given dialogue; do not invent events.
- Refer to people by the given character IDs (C1, C2 ...). You may use a character name
  only if it is spoken in the dialogue. Never name actors or real people.
- summary: at most 40 words. location: a few words. emotional_turn: how the feeling
  changes from the start to the end of the scene, at most 20 words.
- story_beat: the scene's job in a story; "unclear" when a short clip gives no basis.
- character_actions: what each listed character does, at most 15 words each.
- confidence 0..1: lower it for dark, very short or ambiguous scenes.
Answer with JSON matching the schema only."""

ARC_SCHEMA = {
    "type": "object", "additionalProperties": False,
    "properties": {
        "logline": {"type": "string", "maxLength": 300},
        "genres": {"type": "array", "maxItems": 3, "items": {"type": "string", "maxLength": 30}},
        "form": {"type": "string", "enum": ["full_story", "trailer_or_montage", "excerpt", "unclear"]},
        "protagonist": {"type": "string", "maxLength": 8},
        "antagonist": {"type": "string", "maxLength": 8},
        "turning_points": {"type": "array", "maxItems": 8, "items": {
            "type": "object", "additionalProperties": False,
            "properties": {"scene_id": {"type": "string", "maxLength": 8},
                           "what_changes": {"type": "string", "maxLength": 160}},
            "required": ["scene_id", "what_changes"]}},
        "emotional_journey": {"type": "string", "maxLength": 400},
        "character_links": {"type": "array", "maxItems": 6, "items": {
            "type": "object", "additionalProperties": False,
            "properties": {"ids": {"type": "array", "minItems": 2, "maxItems": 4,
                                   "items": {"type": "string", "maxLength": 8}},
                           "hypothesis": {"type": "string", "maxLength": 160},
                           "confidence": {"type": "number", "minimum": 0, "maximum": 1}},
            "required": ["ids", "hypothesis", "confidence"]}},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
    },
    "required": ["logline", "genres", "form", "protagonist", "antagonist", "turning_points",
                 "emotional_journey", "character_links", "confidence"],
}

ARC_PROMPT = """You get the scene-by-scene reading of a film or clip, in order, with its measured
feeling curve and characters. Describe the story it tells and the emotional journey.
Rules:
- Base everything on the scene readings and facts. Refer to characters by ID.
- form: trailer_or_montage when scenes jump without continuity (then the arc is the arc of
  the cut, not of the film).
- protagonist / antagonist: a character ID, or "none".
- turning_points: scenes where the story or feeling clearly changes, with what changes.
- emotional_journey: at most 60 words, start to end.
- character_links: only when the story strongly suggests that different IDs are the same
  person (for example a masked or suited figure and a character seen building the suit);
  give a confidence. Empty list when there is no such evidence.
- logline at most 40 words. Never name actors or real people.
Answer with JSON matching the schema only."""


def _clip(x, lo, hi):
    return max(lo, min(hi, x))


def _number(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{name} must be a number")
    return round(_clip(float(value), 0.0, 1.0), 3)


def _text(output, key, limit):
    value = output.get(key)
    if not isinstance(value, str):
        raise ValueError(f"{key} must be text")
    return value.strip()[:limit]


def validate_scene(output, known_ids=()):
    if not isinstance(output, dict):
        raise ValueError("response is not an object")
    for key, allowed in (("time_of_day", TIMES), ("scene_type", SCENE_TYPES), ("story_beat", BEATS)):
        if output.get(key) not in allowed:
            raise ValueError(f"{key} must be one of {allowed}")
    actions = output.get("character_actions")
    if not isinstance(actions, list) or any(not isinstance(a, dict) for a in actions):
        raise ValueError("character_actions invalid")
    return {"summary": _text(output, "summary", 300), "location": _text(output, "location", 80),
            "time_of_day": output["time_of_day"], "scene_type": output["scene_type"],
            "story_beat": output["story_beat"],
            "emotional_turn": _text(output, "emotional_turn", 160),
            "character_actions": [{"id": str(a.get("id", ""))[:8], "doing": str(a.get("doing", ""))[:100]}
                                  for a in actions[:6]
                                  if not known_ids or str(a.get("id", "")) in known_ids],
            "confidence": _number(output.get("confidence"), "confidence")}


def validate_arc(output, scene_ids=(), character_ids=()):
    if not isinstance(output, dict):
        raise ValueError("response is not an object")
    if output.get("form") not in ("full_story", "trailer_or_montage", "excerpt", "unclear"):
        raise ValueError("form invalid")
    points = output.get("turning_points")
    links = output.get("character_links")
    genres = output.get("genres")
    if not isinstance(points, list) or not isinstance(links, list) or not isinstance(genres, list):
        raise ValueError("lists missing")
    who = lambda v: v if (v in character_ids or not character_ids) and v else "none"
    return {"logline": _text(output, "logline", 300),
            "genres": [str(g)[:30] for g in genres[:3]],
            "form": output["form"],
            "protagonist": who(str(output.get("protagonist", ""))),
            "antagonist": who(str(output.get("antagonist", ""))),
            "turning_points": [{"scene_id": str(p.get("scene_id")), "what_changes": str(p.get("what_changes", ""))[:160]}
                               for p in points[:8] if isinstance(p, dict)
                               and (not scene_ids or str(p.get("scene_id")) in scene_ids)],
            "emotional_journey": _text(output, "emotional_journey", 400),
            "character_links": [{"ids": [str(i) for i in l["ids"]][:4],
                                 "hypothesis": str(l.get("hypothesis", ""))[:160],
                                 "confidence": _number(l.get("confidence"), "link confidence")}
                                for l in links[:6] if isinstance(l, dict) and isinstance(l.get("ids"), list)
                                and len(l["ids"]) >= 2
                                and (not character_ids or all(str(i) in character_ids for i in l["ids"]))],
            "confidence": _number(output.get("confidence"), "confidence")}


def _ask(backend, payload, validate, timeout):
    error = ""
    for _ in range(2):
        try:
            return validate(backend(payload, timeout)), None
        except ValueError as exc:
            error = str(exc)
            payload["messages"] = payload["messages"][:2] + [
                {"role": "user", "content": f"Invalid answer ({error}). Return JSON matching the schema exactly."}]
    return None, error


def scene_facts(scene, timeline, speech, characters):
    by_id = {c["id"]: c for c in characters or []}
    a, b = scene["start_seconds"], scene["end_seconds"]
    lines = []
    for s in speech or []:
        if a <= (s["start_seconds"] + s["end_seconds"]) / 2 < b and not s.get("speech_suspect"):
            who = s.get("character") or s.get("speaker") or "unknown"
            lines.append(f"{who}: {s.get('text', '').strip()}"[:200])
    people = []
    for cid in scene["characters"][:6]:
        c = by_id.get(cid, {})
        role = c.get("role") or {}
        people.append({"id": cid, "main_character": c.get("main_character"),
                       "role": role.get("narrative_role"), "archetypes": role.get("archetypes"),
                       "appearance": role.get("appearance")})
    windows = [w for w in timeline if w["start_seconds"] < b and w["end_seconds"] > a]
    music = [((w.get("feeling") or {}).get("music") or {}).get("mode") for w in windows]
    return {"scene_id": scene["scene_id"], "seconds": [a, b], "shots": len(scene["shot_ids"]),
            "characters": people, "dialogue": lines[:12], "feeling": scene["feeling"],
            "music_mode": [m for m in music if m][:4]}


def read_scenes(video, scenes, timeline, speech, characters, *, model=DEFAULT_MODEL,
                frames_per_scene=6, timeout=300, backend=None, frame_loader=None):
    backend = backend or ollama_chat
    loader = frame_loader or (lambda a, b, n: extract_window_jpegs(video, a, b, n))
    abstained = 0
    for n, scene in enumerate(scenes, 1):
        a, b = scene["start_seconds"], scene["end_seconds"]
        count = max(1, min(frames_per_scene, int((b - a) / 1.0) or 1))
        payload = {"model": model, "stream": False, "think": False, "format": SCENE_SCHEMA,
                   "keep_alive": "10m", "options": {"temperature": 0, "num_ctx": 8192},
                   "messages": [{"role": "system", "content": SCENE_PROMPT},
                                {"role": "user", "images": loader(a, b, count),
                                 "content": json.dumps(scene_facts(scene, timeline, speech, characters))}]}
        known = set(scene["characters"])
        result, error = _ask(backend, payload, lambda o: validate_scene(o, known), timeout)
        if result is None:
            abstained += 1
            print(f"WARNING: scene reading abstained for {scene['scene_id']} ({error})", file=sys.stderr)
        scene["reading"] = None if result is None else {**result, "model": model}
        print(f"Scene readings {n}/{len(scenes)}", file=sys.stderr)
    return abstained


def read_arc(scenes, characters, curve, *, model=DEFAULT_MODEL, timeout=300, backend=None):
    backend = backend or ollama_chat
    compact = []
    for s in scenes:
        r = s.get("reading") or {}
        compact.append({"id": s["scene_id"], "t": [s["start_seconds"], s["end_seconds"]],
                        "who": s["characters"][:4], "type": r.get("scene_type"),
                        "beat": r.get("story_beat"), "what": " ".join(str(r.get("summary", "")).split()[:30]),
                        "turn": r.get("emotional_turn"),
                        "tension": s["feeling"].get("tension"), "valence": s["feeling"].get("valence")})
    cast = [{"id": c["id"], "screen_seconds": c["screen_seconds"], "main": c.get("main_character"),
             "identified_by": c.get("identified_by"),
             "role": (c.get("role") or {}).get("narrative_role"),
             "archetypes": (c.get("role") or {}).get("archetypes"),
             "appearance": (c.get("role") or {}).get("appearance")}
            for c in characters or [] if not c.get("background")]
    points = [{"t": p.get("time_seconds"), "kind": p.get("kind")} for p in (curve or {}).get("turning_points", [])][:12]
    payload = {"model": model, "stream": False, "think": False, "format": ARC_SCHEMA,
               "keep_alive": "10m", "options": {"temperature": 0, "num_ctx": 16384},
               "messages": [{"role": "system", "content": ARC_PROMPT},
                            {"role": "user", "content": json.dumps({
                                "scenes": compact, "characters": cast,
                                "measured_turning_points": points})}]}
    ids = {s["scene_id"] for s in scenes}
    cids = {c["id"] for c in characters or []}
    result, error = _ask(backend, payload, lambda o: validate_arc(o, ids, cids), timeout)
    if result is None:
        print(f"WARNING: story arc abstained ({error})", file=sys.stderr)
        return None
    return {**result, "model": model, "kind": "interpretation"}


def interpret_story(video, scenes, timeline, speech, characters, curve, *, model=DEFAULT_MODEL,
                    frames_per_scene=6, backend=None, frame_loader=None, unload=None):
    """Read every scene, then the arc. Returns (story, metadata)."""
    unload = unload or unload_model
    began = time.perf_counter()
    try:
        abstained = read_scenes(video, scenes, timeline, speech, characters, model=model,
                                frames_per_scene=frames_per_scene, backend=backend,
                                frame_loader=frame_loader)
        story = read_arc(scenes, characters, curve, model=model, backend=backend)
    finally:
        unload(model)
    return story, {"model": model, "scenes_read": len(scenes) - abstained, "abstained": abstained,
                   "frames_per_scene": frames_per_scene,
                   "model_seconds": round(time.perf_counter() - began, 2)}
