"""Deterministic budgeted frame sampling for temporal visual evidence.

Plans positions only; no video decoding or ML dependencies. Frames are samples,
not claims about the content between their timestamps.
"""
from __future__ import annotations

import math


def _validate_samples(shots, limit, spacing_seconds):
    if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1:
        raise ValueError("max_caption_frames must be a positive integer")
    if (isinstance(spacing_seconds, bool) or
        not isinstance(spacing_seconds, (float, int)) or
        not math.isfinite(spacing_seconds) or spacing_seconds <= 0):
        raise ValueError("caption_spacing must be a positive finite number")
    last_end = None
    for shot in shots:
        a, b = shot["start_seconds"], shot["end_seconds"]
        if (not isinstance(a, (int, float)) or isinstance(a, bool) or
            not isinstance(b, (int, float)) or isinstance(b, bool) or
            not math.isfinite(a) or not math.isfinite(b) or b <= a or
            (last_end is not None and a < last_end - 1e-6)):
            raise ValueError("Shots must be positive-duration, finite and ordered")
        last_end = b


def plan_frame_samples(shots: list[dict], limit: int = 120,
                       spacing_seconds: float = 10.0) -> list[dict]:
    """Allocate samples across shots, then spread them through each shot.

    Each shot gets one frame if budget permits. Remaining slots are granted to
    the most undersampled long shots until the requested spacing or global cap.
    If there are more shots than slots, choose shots evenly by shot index.
    In every selected shot frames use equal-duration bins, with bin midpoints.
    This does not infer scenes and does not score sample visual importance.
    """
    _validate_samples(shots, limit, spacing_seconds)
    n = len(shots)
    if n == 0:
        return []
    if n <= limit:
        indices = list(range(n))
    elif limit == 1:
        indices = [n // 2]
    else:
        indices = [round(i * (n - 1) / (limit - 1)) for i in range(limit)]
    quota = {i: 1 for i in indices}
    target = {i: max(1, math.ceil((shots[i]["end_seconds"] - shots[i]["start_seconds"]) /
                                 spacing_seconds - 1e-9)) for i in indices}
    while sum(quota.values()) < limit:
        eligible = [i for i in indices if quota[i] < target[i]]
        if not eligible:
            break
        i = max(eligible, key=lambda x: (
            (shots[x]["end_seconds"] - shots[x]["start_seconds"]) / quota[x], -x))
        quota[i] += 1
    plan = []
    for i in indices:
        shot = shots[i]
        a, b = shot["start_seconds"], shot["end_seconds"]
        count = quota[i]
        for k in range(count):
            plan.append({"shot_id": shot["shot_id"],
                         "frame_time_seconds": a + (k + 0.5) * (b-a) / count})
    return sorted(plan, key=lambda x: x["frame_time_seconds"])


def describe_coverage(shots: list[dict], observations: list[dict]) -> dict:
    """Report sample gaps and omitted shots; never imply full visual coverage."""
    grouped = {}
    for obs in observations:
        grouped.setdefault(obs["shot_id"], []).append(obs["frame_time_seconds"])
    omitted_shots = 0
    omitted_duration = 0.0
    largest_gap = 0.0
    for shot in shots:
        a, b = shot["start_seconds"], shot["end_seconds"]
        samples = sorted(grouped.get(shot["shot_id"], []))
        if any((not isinstance(t, (float, int)) or not math.isfinite(t) or not a <= t <= b) for t in samples):
            raise ValueError("Observation time outside referenced shot")
        if not samples:
            omitted_shots += 1
            omitted_duration += b-a
            largest_gap = max(largest_gap, b-a)
            continue
        points = [a, *samples, b]
        largest_gap = max(largest_gap, *(v-u for u,v in zip(points, points[1:])))
    return {"sampled_shot_count": len(shots)-omitted_shots,
            "unsampled_shot_count": omitted_shots,
            "unsampled_shot_duration_seconds": round(omitted_duration, 3),
            "largest_gap_between_sample_points_seconds": round(largest_gap, 3)}
