"""Stage 6b: what kind of character is this? (local VLM through Ollama)

For each prominent character found by :mod:`characters`, the vision model
sees a small set of crops (best face, each outfit, one full frame for
setting) plus what the character says and the mood of the scenes they are
in. It answers under a JSON schema: narrative role (protagonist, antagonist
...), archetypes (hero, villain, criminal, terrorist, journalist ...),
apparent occupation with the visible cues behind it, and a confidence.

The model judges the *fictional role* from costume, uniform, props, setting,
actions, framing and dialogue. It is told never to infer a role from
ethnicity, skin colour, religion, gender or age, never to name actors or real
people, and to answer "unclear" when the cues are not there.

It also describes the apparent look (gender and age range as they appear,
build, hair, facial hair, distinctive features) so :mod:`naming` can write a
profile. Ethnicity, skin colour and attractiveness are not described.
"""
from __future__ import annotations

import base64
import json
import math
import sys
import time

from .visual_mood import DEFAULT_MODEL, ollama_chat, unload_model

NARRATIVE_ROLES = ("protagonist", "antagonist", "supporting", "minor", "unclear")
ARCHETYPES = ("hero", "villain", "criminal", "terrorist", "victim", "mentor", "love_interest",
              "sidekick", "authority_figure", "henchman", "comic_relief", "civilian",
              "journalist", "business_person", "soldier", "law_enforcement", "scientist",
              "unclear")
GENDERS = ("man", "woman", "boy", "girl", "unclear")
AGES = ("child", "teen", "20s", "30s", "40s", "50s", "60s", "70+", "unclear")

SCHEMA = {
    "type": "object", "additionalProperties": False,
    "properties": {
        "narrative_role": {"type": "string", "enum": list(NARRATIVE_ROLES)},
        "archetypes": {"type": "array", "maxItems": 3,
                       "items": {"type": "string", "enum": list(ARCHETYPES)}},
        "occupation": {"type": "string", "maxLength": 40},
        "occupation_cues": {"type": "array", "maxItems": 4,
                            "items": {"type": "string", "maxLength": 60}},
        "appearance": {"type": "string", "maxLength": 200},
        "evidence": {"type": "string", "maxLength": 300},
        "apparent_gender": {"type": "string", "enum": list(GENDERS)},
        "apparent_age_range": {"type": "string", "enum": list(AGES)},
        "build": {"type": "string", "maxLength": 40},
        "hair": {"type": "string", "maxLength": 50},
        "facial_hair": {"type": "string", "maxLength": 50},
        "distinctive_features": {"type": "array", "maxItems": 3,
                                 "items": {"type": "string", "maxLength": 50}},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
    },
    "required": ["narrative_role", "archetypes", "occupation", "occupation_cues",
                 "appearance", "evidence", "apparent_gender", "apparent_age_range", "build",
                 "hair", "facial_hair", "distinctive_features", "confidence"],
}

SYSTEM_PROMPT = """You help a viewer follow a film. You get images of ONE character from it:
close-ups, their outfits, and a full frame showing the setting, plus facts measured from the film.
Judge what kind of character this is in the story.
Rules:
- Base every judgement on visible cues (costume, uniform, suit, armour, props, weapons, setting,
  actions, framing) and on the character's lines and screen time.
- Never infer a role from ethnicity, skin colour, religion, gender or age.
- Do not name actors, real people, characters or the film.
- narrative_role: protagonist (story centres on them), antagonist (opposes the protagonist),
  supporting, minor, or unclear. Large screen-time share and many lines suggest protagonist,
  but menace, weapons aimed at others or hostile lines suggest antagonist.
- archetypes: up to 3 that the cues clearly support; ["unclear"] if none.
- occupation: a short apparent occupation ("business executive", "news reporter", "soldier",
  "pilot") or "unknown"; occupation_cues: the visible or spoken cues behind it.
- appearance: one sentence on clothing and notable items, at most 25 words.
- evidence: one sentence of at most 35 words naming the cues behind the role.
- Physical description, only from what the images show, as a viewer would describe the
  character to someone who has not seen the film:
  apparent_gender and apparent_age_range as they appear ("unclear" when the face is hidden,
  masked or too small); build ("slim", "athletic", "heavyset" ...); hair (length, colour, style);
  facial_hair ("clean-shaven", "Van Dyke beard", "stubble" ...); distinctive_features: up to 3
  (glasses, scar, tattoo, eye patch, armour ...). Use "unclear" for anything not visible.
  Do not describe ethnicity, skin colour or attractiveness.
- confidence 0..1: lower it when the images are dark, small or ambiguous.
Answer with JSON matching the schema only."""


def _clip(x, lo, hi):
    return max(lo, min(hi, x))


def validate(output):
    if not isinstance(output, dict):
        raise ValueError("response is not an object")
    if output.get("narrative_role") not in NARRATIVE_ROLES:
        raise ValueError(f"narrative_role must be one of {NARRATIVE_ROLES}")
    archetypes = output.get("archetypes")
    if not isinstance(archetypes, list) or any(a not in ARCHETYPES for a in archetypes):
        raise ValueError("archetypes must come from the list")
    cues = output.get("occupation_cues")
    if not isinstance(cues, list) or any(not isinstance(c, str) for c in cues):
        raise ValueError("occupation_cues must be strings")
    conf = output.get("confidence")
    if isinstance(conf, bool) or not isinstance(conf, (int, float)) or not math.isfinite(conf):
        raise ValueError("confidence must be a number")
    for key in ("occupation", "appearance", "evidence"):
        if not isinstance(output.get(key), str):
            raise ValueError(f"{key} must be text")
    physical = validate_physical(output)
    archetypes = list(dict.fromkeys(archetypes))[:3]
    if len(archetypes) > 1 and "unclear" in archetypes:
        archetypes.remove("unclear")
    return {"narrative_role": output["narrative_role"], "archetypes": archetypes or ["unclear"],
            "occupation": output["occupation"].strip()[:40] or "unknown",
            "occupation_cues": [c.strip()[:60] for c in cues[:4]],
            "appearance": output["appearance"].strip()[:200],
            "evidence": output["evidence"].strip()[:300],
            "physical": physical,
            "confidence": round(_clip(float(conf), 0.0, 1.0), 3)}


def validate_physical(output):
    """Apparent look; missing or malformed parts become "unclear" rather than failing the role."""
    def word(key, allowed):
        return output.get(key) if output.get(key) in allowed else "unclear"

    def text(key, limit):
        value = output.get(key)
        return value.strip()[:limit] or "unclear" if isinstance(value, str) else "unclear"
    features = output.get("distinctive_features")
    return {"apparent_gender": word("apparent_gender", GENDERS),
            "apparent_age_range": word("apparent_age_range", AGES),
            "build": text("build", 40), "hair": text("hair", 50), "facial_hair": text("facial_hair", 50),
            "distinctive_features": [f.strip()[:50] for f in features[:3] if isinstance(f, str) and f.strip()]
            if isinstance(features, list) else []}


# ------------------------------------------------------------- material ----

def _crop(frame, box, margin, size):
    import cv2
    h, w = frame.shape[:2]
    x, y, bw, bh = box
    mx, my = bw * margin, bh * margin
    x0, y0 = int(max(0, x - mx)), int(max(0, y - my))
    x1, y1 = int(min(w, x + bw + mx)), int(min(h, y + bh + my))
    crop = frame[y0:y1, x0:x1]
    if crop.size == 0:
        return None
    scale = size / max(crop.shape[:2])
    if scale < 1:
        crop = cv2.resize(crop, (max(1, int(crop.shape[1] * scale)), max(1, int(crop.shape[0] * scale))))
    return crop


def _jpeg(image):
    import cv2
    ok, data = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 85])
    if not ok:
        raise RuntimeError("JPEG encoding failed")
    return base64.b64encode(data.tobytes()).decode("ascii")


def character_images(character, grab, max_outfits=2):
    """Best face, outfit crops and one setting frame, as base64 JPEGs."""
    images, notes = [], []
    cache = {}

    def frame(t):
        if t not in cache:
            cache[t] = grab(t)
        return cache[t]
    if character.get("best_face"):
        f = character["best_face"]
        crop = _crop(frame(f["t"]), f["box"], 0.6, 384)
        if crop is not None:
            images.append(_jpeg(crop))
            notes.append(f"close-up at {f['t']:.0f}s")
    for outfit in character.get("outfits", [])[:max_outfits]:
        s = outfit.get("sample")
        if not s:
            continue
        crop = _crop(frame(s["t"]), s["box"], 0.1, 512)
        if crop is not None:
            images.append(_jpeg(crop))
            notes.append(f"outfit {outfit['id']} ({', '.join(outfit['colours'])}) at {s['t']:.0f}s")
    body = character.get("best_body")
    if body:
        full = frame(body["t"])
        images.append(_jpeg(_crop(full, [0, 0, full.shape[1], full.shape[0]], 0.0, 640)))
        notes.append(f"full frame at {body['t']:.0f}s showing the setting")
    return images, notes


def character_facts(character, speech, timeline, total_screen, max_lines=10):
    lines = [s.get("text", "").strip() for s in speech
             if s.get("character") == character["id"] and s.get("text")]
    said, size = [], 0
    for line in lines[:max_lines]:
        if size + len(line) > 600:
            break
        said.append(line)
        size += len(line)
    moods = []
    for w in timeline:
        if any(p["id"] == character["id"] for p in w.get("characters_present") or []):
            vm = (w.get("feeling_signals") or {}).get("visual_mood")
            if vm and vm.get("evidence"):
                moods.append(vm["evidence"])
    step = max(1, len(moods) // 4)
    return {"character_id": character["id"],
            "screen_seconds": character["screen_seconds"],
            "share_of_all_screen_time": round(character["screen_seconds"] / max(1e-9, total_screen), 3),
            "prominence_rank": character.get("prominence_rank"),
            "scenes_seen": len(character.get("shots", [])),
            "face_seen": bool(character.get("face_ids")),
            "outfits": [{"id": o["id"], "colours": o["colours"]} for o in character.get("outfits", [])],
            "lines_spoken": said,
            "scene_descriptions": moods[::step][:4]}


def interpret_characters(video, info, characters, speech, timeline, *, model=DEFAULT_MODEL,
                         max_characters=8, min_screen_seconds=3.0, timeout=300,
                         backend=None, grab=None, unload=None):
    """Add ``role`` to the most prominent characters. Returns stage metadata."""
    from .characters import frame_at
    backend = backend or ollama_chat
    unload = unload or unload_model
    grab = grab or (lambda t: frame_at(video, info, t))
    total = sum(c["screen_seconds"] for c in characters)
    chosen = [c for c in sorted(characters, key=lambda c: c.get("prominence_rank") or 99)
              if c["screen_seconds"] >= min_screen_seconds][:max_characters]
    abstained, began = 0, time.perf_counter()
    try:
        for n, c in enumerate(chosen, 1):
            images, notes = character_images(c, grab)
            payload = {"model": model, "stream": False, "think": False, "format": SCHEMA,
                       "keep_alive": "10m", "options": {"temperature": 0, "num_ctx": 8192},
                       "messages": [
                           {"role": "system", "content": SYSTEM_PROMPT},
                           {"role": "user", "images": images, "content": json.dumps({
                               "images": notes,
                               "facts": character_facts(c, speech, timeline, total)})}]}
            result, error = None, ""
            for _ in range(2):
                try:
                    result = validate(backend(payload, timeout))
                    break
                except ValueError as exc:
                    error = str(exc)
                    payload["messages"] = payload["messages"][:2] + [
                        {"role": "user", "content": f"Invalid answer ({error}). Return JSON "
                                                    "matching the schema exactly."}]
            if result is None:
                abstained += 1
                print(f"WARNING: role abstained for {c['id']} ({error})", file=sys.stderr)
            c["role"] = None if result is None else {**result, "model": model}
            print(f"Character roles {n}/{len(chosen)}", file=sys.stderr)
    finally:
        unload(model)
    for c in characters:
        role = c.get("role") or {}
        c["main_character"] = bool(c.get("likely_main_character")) and \
            role.get("narrative_role") != "antagonist" or \
            (role.get("narrative_role") == "protagonist" and (c.get("prominence") or 0) >= 0.5)
    return {"model": model, "characters_interpreted": len(chosen), "abstained": abstained,
            "model_seconds": round(time.perf_counter() - began, 2)}
