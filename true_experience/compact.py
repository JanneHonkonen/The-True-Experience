"""The experience file: what the film is like, without the measurements behind it.

The full analysis repeats a lot: every timeline window copies its speech
lines, each shot carries its own continuity scores, character tracks list
every appearance, and raw signals sit next to the feelings fused from them.
That detail is useful for tuning and is kept in ``<name>_details.json``.

The experience file keeps, once each:

- the film: length, logline, form, genres, emotional journey, turning points,
- the cast: name, profile and role per character (background extras left out),
- the scenes in order: time, place, the feeling from start to end, who is
  there, the narration passages and the lines spoken (with who and how),
- the feeling curve as columns (valence, arousal, tension, label per window).
"""
from __future__ import annotations

from .narration import describe_person, delivery, mmss


def _r(x, d=2):
    return None if x is None else round(float(x), d)


def _cast(characters):
    out = []
    for c in characters or []:
        if c.get("background"):
            continue
        role = c.get("role") or {}
        name = c.get("name") or {}
        out.append({k: v for k, v in {
            "id": c["id"], "name": name.get("name"), "also_called": name.get("also_called") or None,
            "profile": c.get("profile"), "main_character": c.get("main_character") or None,
            "screen_seconds": _r(c.get("screen_seconds"), 1),
            "role": role.get("narrative_role"),
            "archetypes": [a for a in role.get("archetypes") or [] if a != "unclear"] or None,
            "occupation": role.get("occupation") if role.get("occupation") not in (None, "unknown") else None,
            "identified_by": c.get("identified_by"),
        }.items() if v is not None})
    return out


def _lines(speech, by_id, a=None, b=None):
    out = []
    for s in speech or []:
        mid = (s["start_seconds"] + s["end_seconds"]) / 2
        if s.get("speech_suspect") or (a is not None and not a <= mid < b) or not (s.get("text") or "").strip():
            continue
        cid = s.get("character")
        out.append({k: v for k, v in {
            "time": mmss(s["start_seconds"]), "t": _r(s["start_seconds"], 1),
            "who": cid or s.get("speaker"), "name": describe_person(by_id.get(cid)) if cid else None,
            "text": s["text"].strip(), "delivery": delivery(s.get("voice"))}.items() if v is not None})
    return out


def _scene(scene, speech, by_id):
    reading = scene.get("reading") or {}
    feeling = scene.get("feeling") or {}
    a, b = scene["start_seconds"], scene["end_seconds"]
    return {k: v for k, v in {
        "scene_id": scene["scene_id"], "from": mmss(a), "to": mmss(b),
        "start_seconds": a, "end_seconds": b,
        "place": reading.get("location"), "time_of_day": reading.get("time_of_day"),
        "type": reading.get("scene_type"), "beat": reading.get("story_beat"),
        "feeling": {k: v for k, v in {
            "start": feeling.get("start_label"), "end": feeling.get("end_label"),
            "labels": feeling.get("labels"), "tension": feeling.get("tension"),
            "peak_tension": feeling.get("peak_tension"), "valence": feeling.get("valence"),
            "turn": reading.get("emotional_turn")}.items() if v is not None},
        "characters": scene.get("characters") or None,
        "narration": [{"time": p["time"], "text": p["text"]} for p in scene.get("narration") or []] or None,
        "summary": None if scene.get("narration") else reading.get("summary"),
        "dialogue": _lines(speech, by_id, a, b) or None,
    }.items() if v is not None}


def _curve(curve):
    if not curve:
        return None
    keep = ("start_seconds", "valence", "arousal", "tension", "label")
    out = {k: curve.get(k) for k in keep if k in curve}
    out["turning_points"] = [{"time": mmss(p.get("time_seconds")), **{k: v for k, v in p.items()}}
                             for p in curve.get("turning_points") or []]
    return out


def experience(result):
    """The compact experience view of a full analysis result."""
    meta = result.get("metadata") or {}
    characters = result.get("characters") or []
    by_id = {c["id"]: c for c in characters}
    speech = result.get("speech_segments") or []
    story = result.get("story") or {}
    scenes = result.get("scenes")
    names = lambda cid: describe_person(by_id.get(cid)) if cid in by_id else None
    film = {k: v for k, v in {
        "file": meta.get("source_file"), "duration": mmss(meta.get("duration_seconds")),
        "duration_seconds": meta.get("duration_seconds"),
        "logline": story.get("logline"), "form": story.get("form"), "genres": story.get("genres"),
        "protagonist": story.get("protagonist") if story.get("protagonist") not in (None, "none") else None,
        "protagonist_is": names(story.get("protagonist")),
        "antagonist": story.get("antagonist") if story.get("antagonist") not in (None, "none") else None,
        "antagonist_is": names(story.get("antagonist")),
        "emotional_journey": story.get("emotional_journey"),
        "turning_points": story.get("turning_points") or None,
        "character_links": story.get("character_links") or None,
        "names_not_on_screen": story.get("names_not_on_screen") or None,
        "interpretation_confidence": story.get("confidence"),
    }.items() if v is not None}
    out = {"format": "true-experience/1", "film": film, "characters": _cast(characters) or None}
    if scenes:
        out["scenes"] = [_scene(s, speech, by_id) for s in scenes]
    else:
        out["dialogue"] = _lines(speech, by_id) or None
    out["feeling_curve"] = _curve(result.get("feeling_curve"))
    out["about"] = {"details_file": None, "methods": meta.get("methods"),
                    "timing_seconds": meta.get("timing_seconds"),
                    "note": "Narration, readings, names and roles are model interpretation; "
                            "the measurements behind them are in the details file."}
    return {k: v for k, v in out.items() if v is not None}
