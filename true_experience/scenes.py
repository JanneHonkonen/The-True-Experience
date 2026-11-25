"""Stage 9a: group shots into scenes (CPU).

A scene is a run of shots in one place and moment. Cuts inside a scene keep
returning to the same look and the same people (shot / reverse shot), and
lines of dialogue run across them. A new scene changes place, people or
both, often with a pause in the sound.

For every cut the stage compares the incoming shot with the last few shots
of the current scene:

- **look**: colour histogram plus a coarse 3 x 3 layout of mean colours,
  from small frames decoded at 4 fps,
- **people**: overlap of the (non-background) characters on screen, when
  ``--characters`` ran,
- **speech**: a line of dialogue running across the cut,
- **silence**: a silent gap at the cut pushes towards a new scene.

Scenes shorter than ``MIN_SCENE_SECONDS`` are merged into the more similar
neighbour, so fast montages do not turn into dozens of one-shot scenes.
"""
from __future__ import annotations

import sys

SCENE_FPS = 4
FRAME_WIDTH = 160
LOOKBACK = 3
BREAK_BELOW = 0.45
MIN_SCENE_SECONDS = 4.0


def _np():
    try:
        import numpy
    except ImportError as exc:
        raise RuntimeError("Scene detection requires NumPy: python -m pip install numpy") from exc
    return numpy


def _r(x, d=3):
    return None if x is None else round(float(x), d)


def look_signature(frames):
    """Colour histogram (4x4x4 RGB bins) and 3x3 mean-colour layout of a shot."""
    np = _np()
    if not frames:
        return None
    stack = np.stack(frames).astype(np.float64)                  # n, h, w, 3
    q = np.clip((stack // 64).astype(int), 0, 3)
    idx = q[..., 0] * 16 + q[..., 1] * 4 + q[..., 2]
    hist = np.bincount(idx.ravel(), minlength=64).astype(np.float64)
    hist /= max(1.0, hist.sum())
    h, w = stack.shape[1:3]
    cells = [stack[:, i * h // 3:(i + 1) * h // 3, j * w // 3:(j + 1) * w // 3].mean(axis=(0, 1, 2))
             for i in range(3) for j in range(3)]
    return {"hist": hist, "layout": np.array(cells) / 255.0}


def look_similarity(a, b):
    """0..1: histogram intersection blended with layout closeness."""
    np = _np()
    if a is None or b is None:
        return 0.5
    hist = float(np.minimum(a["hist"], b["hist"]).sum())
    layout = 1.0 - min(1.0, float(np.abs(a["layout"] - b["layout"]).mean()) * 4.0)
    return 0.6 * hist + 0.4 * layout


def shot_signatures(video, info, shots):
    """One look signature per shot from a single low-resolution decode."""
    from .signals import stream_frame_windows
    bounds = [(s["start_seconds"], s["end_seconds"]) for s in shots]
    sigs = []
    it = stream_frame_windows(video, bounds, info.get("width"), info.get("height"),
                              fps=SCENE_FPS, frame_width=FRAME_WIDTH)
    try:
        for _ in bounds:
            frames, _times = next(it, ([], []))
            sigs.append(look_signature(frames))
    finally:
        it.close()
    # Very short shots may get no frame: borrow the previous shot's look.
    for i, s in enumerate(sigs):
        if s is None and i:
            sigs[i] = sigs[i - 1]
    return sigs


def shot_people(shots, character_tracks, characters):
    background = {c["id"] for c in characters or [] if c.get("background")}
    people = [set() for _ in shots]
    for row in character_tracks or []:
        if row["character"] not in background and 0 <= row["shot"] < len(shots):
            people[row["shot"]].add(row["character"])
    return people


def cut_scores(shots, signatures, people, speech, silences):
    """Continuity of each cut (between shot i-1 and i) with the scene so far: 0..1."""
    scores = [None]
    scene_start = 0
    for i in range(1, len(shots)):
        cut = shots[i]["start_seconds"]
        recent = range(max(scene_start, i - LOOKBACK), i)
        look = max(look_similarity(signatures[j], signatures[i]) for j in recent)
        before = set().union(*(people[j] for j in recent))
        if before and people[i]:
            together = len(before & people[i]) / len(before | people[i])
            person = 0.5 + 0.5 * together if together else 0.15
        else:
            person = 0.5  # nobody to compare
        spoken = any(s["start_seconds"] < cut - 0.2 and s["end_seconds"] > cut + 0.2
                     and not s.get("speech_suspect") for s in speech or [])
        hush = any(a <= cut + 0.3 and b >= cut - 0.3 and b - a >= 0.8 for a, b in silences or [])
        score = 0.55 * look + 0.30 * person + (0.25 if spoken else 0.0) - (0.15 if hush else 0.0)
        score = max(0.0, min(1.0, score))
        scores.append({"look": _r(look), "people": _r(person), "speech_across": spoken,
                       "silence_at_cut": hush, "continuity": _r(score)})
        if score < BREAK_BELOW:
            scene_start = i
    return scores


def group_scenes(shots, scores, min_seconds=MIN_SCENE_SECONDS):
    """Lists of shot indexes; short scenes join the neighbour they continue best."""
    groups = [[0]] if shots else []
    for i in range(1, len(shots)):
        if scores[i]["continuity"] < BREAK_BELOW:
            groups.append([i])
        else:
            groups[-1].append(i)
    length = lambda g: shots[g[-1]]["end_seconds"] - shots[g[0]]["start_seconds"]
    changed = True
    while changed and len(groups) > 1:
        changed = False
        for k, g in enumerate(groups):
            if length(g) >= min_seconds:
                continue
            left = scores[g[0]]["continuity"] if k > 0 else -1.0
            right = scores[groups[k + 1][0]]["continuity"] if k + 1 < len(groups) else -1.0
            if left >= right:
                groups[k - 1].extend(g)
            else:
                groups[k + 1][:0] = g
            groups.pop(k)
            changed = True
            break
    return groups


def _mean(values):
    values = [v for v in values if v is not None]
    return _r(sum(values) / len(values)) if values else None


def describe_scenes(shots, groups, timeline, speech, people):
    scenes = []
    for n, g in enumerate(groups, 1):
        a, b = shots[g[0]]["start_seconds"], shots[g[-1]]["end_seconds"]
        windows = [w for w in timeline if w["start_seconds"] < b and w["end_seconds"] > a]
        fused = [(w.get("feeling") or {}).get("fused") or {} for w in windows]
        labels = [f.get("label") for f in fused if f.get("label")]
        lines = [s for s in speech or [] if a <= (s["start_seconds"] + s["end_seconds"]) / 2 < b
                 and not s.get("speech_suspect")]
        seen = {}
        for i in g:
            for c in people[i]:
                seen[c] = seen.get(c, 0.0) + shots[i]["end_seconds"] - shots[i]["start_seconds"]
        scenes.append({
            "scene_id": f"SC{n}", "start_seconds": _r(a, 2), "end_seconds": _r(b, 2),
            "shot_ids": [shots[i]["shot_id"] for i in g],
            "characters": [c for c, _ in sorted(seen.items(), key=lambda kv: -kv[1])],
            "lines": len(lines),
            "speakers": sorted({s.get("character") or s.get("speaker") for s in lines
                                if s.get("character") or s.get("speaker")}),
            "feeling": {"valence": _mean(f.get("valence") for f in fused),
                        "arousal": _mean(f.get("arousal") for f in fused),
                        "tension": _mean(f.get("tension") for f in fused),
                        "peak_tension": _r(max((f.get("tension") or 0.0) for f in fused)) if fused else None,
                        "labels": sorted(set(labels), key=labels.index),
                        "start_label": labels[0] if labels else None,
                        "end_label": labels[-1] if labels else None}})
        for w in windows:
            w.setdefault("scene_ids", []).append(f"SC{n}")
    return scenes


def detect_scenes(video, info, shots, timeline, speech, silences, *, characters=None,
                  character_tracks=None, signatures=None):
    """Group shots into scenes. Adds ``scene_ids`` to timeline windows. Returns (scenes, meta)."""
    if not shots:
        return [], {"scenes": 0}
    sigs = signatures if signatures is not None else shot_signatures(video, info, shots)
    people = shot_people(shots, character_tracks, characters)
    scores = cut_scores(shots, sigs, people, speech, silences)
    groups = group_scenes(shots, scores)
    scenes = describe_scenes(shots, groups, timeline, speech, people)
    for shot, score in zip(shots, scores):
        if score:
            shot["continuity_with_scene"] = score
    print(f"Scenes: {len(shots)} shots -> {len(scenes)} scenes", file=sys.stderr)
    return scenes, {"scenes": len(scenes), "shots": len(shots),
                    "break_below": BREAK_BELOW, "min_scene_seconds": MIN_SCENE_SECONDS,
                    "lookback_shots": LOOKBACK, "uses_characters": bool(character_tracks),
                    "method": "look_people_speech_silence_v1"}
