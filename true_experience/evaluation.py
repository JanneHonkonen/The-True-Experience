"""Independent checks of timestamp alignment and optional ASR word error rate.

This does NOT validate the truth of frame captions, narrative, or emotion.
"""
import math
import re

EPS = 0.002


def seconds(item, key):
    value = item.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"Invalid {key}: {value!r}")
    return float(value)


def bounds(item, duration):
    a, b = seconds(item, "start_seconds"), seconds(item, "end_seconds")
    if a < -EPS or b > duration + EPS or b <= a:
        raise ValueError(f"Invalid interval: {a}..{b}")
    return a, b


def validate_report(report):
    errors, counts = [], {}
    try:
        duration = seconds(report["metadata"], "duration_seconds")
        if duration <= 0:
            raise ValueError("Video duration must be positive")
    except (KeyError, TypeError, AttributeError, ValueError) as exc:
        return {"valid": False, "errors": [f"Invalid metadata: {exc}"], "counts": {}}

    for name in ("shots", "timeline", "speech_segments", "visual_observations"):
        items = report.get(name, [])
        if not isinstance(items, list):
            errors.append(f"{name} must be a list")
            items = []
        counts[name] = len(items)
        previous_end, previous_start = None, None
        for i, item in enumerate(items):
            try:
                if name == "visual_observations":
                    t = seconds(item, "frame_time_seconds")
                    if not 0 <= t <= duration + EPS:
                        raise ValueError("Frame time outside video")
                    continue
                a, b = bounds(item, duration)
                if name in ("shots", "timeline"):
                    if previous_end is None and abs(a) > EPS:
                        raise ValueError("Does not begin at zero")
                    if previous_end is not None and abs(previous_end - a) > EPS:
                        raise ValueError("Gap or overlap in consecutive intervals")
                    previous_end = b
                else:
                    if previous_start is not None and a + EPS < previous_start:
                        raise ValueError("Speech timestamps out of order")
                    previous_start = a
                    if not item.get("source") or not item.get("model"):
                        raise ValueError("Missing ASR provenance")
            except (ValueError, KeyError, TypeError, AttributeError) as exc:
                errors.append(f"{name}[{i}]: {exc}")
        if name in ("shots", "timeline"):
            if not items:
                errors.append(f"{name} must not be empty")
            elif previous_end is not None and abs(previous_end - duration) > EPS:
                errors.append(f"{name} does not cover full video")

    shots = {s.get("shot_id"): s for s in report.get("shots", []) if isinstance(s, dict)}
    for i, obs in enumerate(report.get("visual_observations", [])):
        try:
            if not obs.get("source") or not obs.get("model"):
                raise ValueError("Missing caption provenance")
            shot = shots.get(obs.get("shot_id"))
            if shot is None:
                raise ValueError("References a nonexistent shot")
            t = seconds(obs, "frame_time_seconds")
            a, b = bounds(shot, duration)
            if not a - EPS <= t <= b + EPS:
                raise ValueError("Frame time outside referenced shot")
        except (ValueError, KeyError, TypeError, AttributeError) as exc:
            errors.append(f"visual_observations[{i}]: {exc}")

    for i, item in enumerate(report.get("timeline", [])):
        try:
            a, b = bounds(item, duration)
            for j, speech in enumerate(item.get("speech", [])):
                x, y = bounds(speech, duration)
                if max(x, a) >= min(y, b):
                    errors.append(f"timeline[{i}].speech[{j}] does not overlap interval")
            for j, obs in enumerate(item.get("visual_observations", [])):
                t = seconds(obs, "frame_time_seconds")
                if not a <= t < b:
                    errors.append(f"timeline[{i}].visual_observations[{j}] outside interval")
        except (ValueError, KeyError, TypeError, AttributeError) as exc:
            errors.append(f"timeline[{i}]: {exc}")
    return {"valid": not errors, "errors": errors, "counts": counts}


def word_error_rate(reference, hypothesis):
    def words(value):
        return re.findall(r"[^\W_]+(?:['’][^\W_]+)*", value.casefold(), flags=re.UNICODE)
    ref, hyp = words(reference), words(hypothesis)
    if not ref:
        raise ValueError("Reference transcript is empty")
    previous = list(range(len(hyp) + 1))
    for i, word in enumerate(ref, 1):
        current = [i]
        for j, found in enumerate(hyp, 1):
            current.append(min(current[-1] + 1, previous[j] + 1,
                               previous[j - 1] + (word != found)))
        previous = current
    return {"wer": round(previous[-1] / len(ref), 6),
            "edit_distance": previous[-1],
            "reference_words": len(ref), "hypothesis_words": len(hyp),
            "method": "unicode_casefold_levenshtein"}
