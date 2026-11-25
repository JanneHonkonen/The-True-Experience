"""Stage 9c: character names and a one-line profile per character (CPU).

Names come only from the film itself: lines of dialogue and on-screen text.
Two sources vote for a name:

- **transcript**: a self-introduction ("I'm Tony Stark", "my name is ...",
  "call me ...") in a line already tied to a character by the voice and
  face stages;
- **scene readings**: the story stage asks the vision model, per scene, which
  names are heard or shown and whom each one belongs to (the speaker, the
  person addressed, a person on screen the line is about, or someone absent).

Forms of one name ("Tony Stark", "Mr. Stark", "Tony", "Stark") are merged and
shown under the fullest form. Votes are weighted: self-introduction 3,
on-screen caption 2, being addressed 2, being talked about while present 1.
A character gets a name when its best name reaches ``MIN_SCORE``; a name is
given to one character only.

The profile joins the name with the apparent physical description and the
role from :mod:`characterhood` into one readable sentence. Actor names are
never used, and the description leaves out ethnicity and judgements of looks.
"""
from __future__ import annotations

import re

WEIGHTS = {"self_introduction": 3.0, "on_screen_text": 2.0, "addressed": 2.0,
           "mentioned_present": 1.0, "mentioned_absent": 0.0}
MIN_SCORE = 2.0
TITLES = {"mr", "mrs", "ms", "miss", "mister", "dr", "doctor", "doc", "captain", "capt", "sir",
          "madam", "lady", "lord", "agent", "officer", "detective", "det", "professor", "prof",
          "colonel", "col", "general", "gen", "sergeant", "sgt", "lieutenant", "lt", "major",
          "commander", "director", "senator", "president", "judge", "father", "sister", "uncle",
          "aunt", "king", "queen", "prince", "princess"}
NOT_NAMES = {"i", "a", "the", "here", "there", "not", "just", "so", "sorry", "fine", "okay", "ok",
             "good", "back", "home", "done", "ready", "sure", "afraid", "glad", "going", "telling",
             "serious", "kidding", "god", "yes", "no", "what", "who", "this", "that", "it", "he",
             "she", "they", "we", "you", "one", "all", "an", "only", "still", "now", "out", "in",
             "on", "here", "listening", "coming", "leaving", "late", "right", "wrong", "busy"}
_INTRO = re.compile(r"\b(?i:my name is|my name's|name's|i am|i'm|call me)\s+"
                    r"((?:[A-Z][\w'\-]*\.?\s+){0,3}[A-Z][\w'\-]*)")


def _words(name):
    return [w for w in re.split(r"[\s.,]+", name.replace("'s ", " ")) if w]


def clean_name(name):
    """'Mr. Stark' -> 'Stark'; '' when nothing name-like remains."""
    words = [w.strip("\"'()-") for w in _words(str(name))]
    while words and words[0].lower().rstrip(".") in TITLES:
        words = words[1:]
    words = [w for w in words if w]
    if not words or len(words) > 4 or all(w.lower() in NOT_NAMES for w in words):
        return ""
    if any(not w[0].isalpha() for w in words):
        return ""
    return " ".join(w[0].upper() + w[1:] for w in words)


def transcript_mentions(speech):
    """Self-introductions in lines already tied to a character."""
    out = []
    for s in speech or []:
        cid, text = s.get("character"), s.get("text") or ""
        if not cid or s.get("speech_suspect"):
            continue
        for match in _INTRO.finditer(text):
            words = match.group(1).split()
            # "I'm Tony Stark and..." -> stop at the first word that is not a name.
            name = clean_name(" ".join(w for w in words if w.rstrip(".").lower() not in NOT_NAMES))
            if name and len(name.split()) <= 3:
                out.append({"name": name, "said_as": match.group(1).strip(), "refers_to": cid,
                            "cue": "self_introduction", "source": "transcript",
                            "time_seconds": s.get("start_seconds"), "line": text.strip()[:160]})
    return out


def scene_mentions(scenes):
    out = []
    for scene in scenes or []:
        for m in (scene.get("reading") or {}).get("name_mentions") or []:
            name = clean_name(m.get("name", ""))
            if name:
                out.append({"name": name, "said_as": m.get("name"), "refers_to": m.get("refers_to"),
                            "cue": m.get("cue"), "source": "scene_reading",
                            "scene_id": scene.get("scene_id"), "time_seconds": scene.get("start_seconds")})
    return out


def _same_person(a, b):
    """'Tony' ~ 'Tony Stark' ~ 'Stark': one is contained in the other, word for word."""
    wa, wb = set(a.lower().split()), set(b.lower().split())
    return wa <= wb or wb <= wa


def group_names(mentions):
    """Clusters of mentions that name one person: [(canonical, [mentions])]."""
    groups = []
    for m in sorted(mentions, key=lambda m: -len(m["name"].split())):
        for g in groups:
            if any(_same_person(m["name"], other["name"]) for other in g):
                g.append(m)
                break
        else:
            groups.append([m])
    out = []
    for g in groups:
        # Fullest form, ties broken by how often it is used.
        forms = [m["name"] for m in g]
        canonical = max(forms, key=lambda f: (len(f.split()), forms.count(f)))
        out.append((canonical, g))
    return out


def name_characters(characters, speech, scenes=None):
    """Set ``name`` on characters from dialogue and scene readings. Returns (unseen names, meta)."""
    mentions = transcript_mentions(speech) + scene_mentions(scenes)
    ids = {c["id"] for c in characters or []}
    candidates = []   # (score, cid, canonical, group)
    unseen = []
    for canonical, group in group_names(mentions):
        per = {}
        for m in group:
            if m["refers_to"] in ids:
                per.setdefault(m["refers_to"], []).append(m)
        for cid, ms in per.items():
            score = sum(WEIGHTS.get(m["cue"], 0.0) for m in ms)
            candidates.append((score, cid, canonical, group, ms))
        if not per and any(m["refers_to"] == "absent" for m in group):
            unseen.append({"name": canonical, "mentions": len(group),
                           "scene_ids": sorted({m["scene_id"] for m in group if m.get("scene_id")})})
    by_id = {c["id"]: c for c in characters or []}
    for c in characters or []:
        c["name"] = None
    taken = set()
    named = 0
    for score, cid, canonical, group, ms in sorted(candidates, key=lambda x: -x[0]):
        if score < MIN_SCORE or by_id[cid]["name"] or canonical.lower() in taken:
            continue
        rival = sum(x[0] for x in candidates if x[1] == cid and x[2] != canonical)
        also = sorted({str(m["said_as"]).strip() for m in group
                       if m["refers_to"] == cid and str(m["said_as"]).strip() != canonical})
        by_id[cid]["name"] = {
            "name": canonical, "also_called": also[:5],
            "confidence": round(min(1.0, score / 6.0) * score / (score + rival), 3),
            "evidence": [{k: m[k] for k in ("source", "cue", "scene_id", "time_seconds", "line", "said_as")
                          if m.get(k) is not None} for m in ms[:6]]}
        taken.add(canonical.lower())
        named += 1
    return unseen, {"mentions": len(mentions), "named": named, "unseen_names": len(unseen),
                    "min_score": MIN_SCORE, "weights": WEIGHTS,
                    "sources": sorted({m["source"] for m in mentions})}


# -------------------------------------------------------------- profile ----

_PRONOUN = {"man": "his", "woman": "her", "boy": "his", "girl": "her"}


def _known(value):
    return isinstance(value, str) and value.strip() and value.strip().lower() not in ("unclear", "unknown", "none")


def profile(character):
    """One sentence a reader can follow: name, apparent look, role."""
    role = character.get("role") or {}
    body = role.get("physical") or {}
    gender = body.get("apparent_gender") if _known(body.get("apparent_gender")) else None
    age = body.get("apparent_age_range") if _known(body.get("apparent_age_range")) else None
    noun = gender or "person"
    name = (character.get("name") or {}).get("name")
    head = f"{noun.capitalize()} named {name}" if name else noun.capitalize()
    also = (character.get("name") or {}).get("also_called") or []
    if also:
        head += " (" + ", ".join(f'"{a}"' for a in also[:2]) + ")"
    parts = []
    if age:
        if age in ("child", "teen"):
            parts.append(f"appears to be a {age}")
        elif age == "70+":
            parts.append("appears to be over 70")
        else:
            parts.append(f"appears to be in {_PRONOUN.get(gender, 'their')} {age}")
    for key in ("build", "hair", "facial_hair"):
        if _known(body.get(key)):
            parts.append(body[key].strip())
    parts += [f.strip() for f in body.get("distinctive_features") or [] if _known(f)][:2]
    if _known(role.get("appearance")):
        look = role["appearance"].strip().rstrip(".")
        parts.append(look[0].lower() + look[1:])
    kind = role.get("narrative_role")
    tags = [a.replace("_", " ") for a in role.get("archetypes") or [] if a != "unclear"]
    if _known(kind) or tags:
        parts.append((kind if _known(kind) else "character") + (f" ({', '.join(tags)})" if tags else ""))
    if not parts and not name:
        return None
    return head + ("; " + "; ".join(parts) if parts else "") + "."


def add_profiles(characters):
    for c in characters or []:
        if not c.get("background"):
            c["profile"] = profile(c)
