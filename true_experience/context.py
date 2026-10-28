"""Opt-in, local, bounded audiovisual-context reasoning.

A multimodal language model observes sampled frames, hears *unverified* ASR
through timestamps and receives a small rolling memory. Its outputs remain
explicit hypotheses; this never certifies identity, emotion, causality or plot.

The source movie and transcript are NEVER transmitted to a remote service:
the only supported endpoint is the local Ollama listener on 127.0.0.1.
"""
from __future__ import annotations

import base64
import hashlib
import io
import json
import math
import sys
import urllib.error
import urllib.request
from pathlib import Path

from .perception import _extract_image

LOCAL_API = "http://127.0.0.1:11434/api/chat"

# JSON mode only guarantees parsable JSON, not the fields we need. Ollama
# supports sending a JSON schema in 'format' for constrained generation.
CONTEXT_RESPONSE_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "summary": {"type": "string", "minLength": 1, "maxLength": 1200},
        "events": {
            "type": "array", "maxItems": 8,
            "items": {
                "type": "object", "additionalProperties": False,
                "properties": {
                    "claim": {"type": "string", "minLength": 1, "maxLength": 400},
                    "evidence_ids": {"type": "array", "minItems": 1, "items": {"type": "string"}},
                    "kind": {"type": "string", "enum": ["observation", "interpretation"]}
                },
                "required": ["claim", "evidence_ids", "kind"]
            }
        },
        "continuity": {"type": "string", "maxLength": 1200},
        "links_previous_windows": {"type": "array", "items": {"type": "string"}},
        "open_questions": {"type": "array", "maxItems": 8,
                           "items": {"type": "string", "maxLength": 300}}
    },
    "required": ["summary", "events", "continuity", "links_previous_windows", "open_questions"]
}


def analysis_fingerprint(report):
    """Stable fingerprint; do not emit full private analysis anywhere public."""
    return hashlib.sha256(json.dumps(report, sort_keys=True, ensure_ascii=False,
                                  separators=(",", ":"), allow_nan=False).encode("utf-8")).hexdigest()
SYSTEM_PROMPT = """You analyze audiovisual evidence from one brief time window.
Your information comes ONLY from attached sampled images, fallible frame-caption
hypotheses and fallible timestamped speech recognition.
Do not use knowledge of movies, actors, character names, or story plots unless
directly supported by current evidence. Do not treat prior AI summaries as facts.
Distinguish direct visible/audible observations from interpretation.
Never invent events between sampled frames, causal links, emotion or identity.
Use "unknown" for absent information. Do not reproduce dialogue verbatim.
Your answer MUST be a JSON object:
{"summary": "one grounded tentative summary",
 "events": [{"claim": "short paraphrased claim", "evidence_ids": ["v0001"], "kind": "observation"}],
 "continuity": "possible change or relationship to prior windows, clearly tentative",
 "links_previous_windows": ["w0001"],
 "open_questions": ["what evidence is missing?"]}
Use only supplied current evidence_ids for events. Each event must have one
or more evidence_ids. Kind must be "observation" or "interpretation".
links_previous_windows may ONLY include IDs actually supplied in memory.
If there is no visible or spoken evidence, say so; no invented narrative.
"""


def _positive(value, name):
    if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be positive and finite")


def make_windows(report, seconds=12.0):
    """Time bins are *analysis windows*, never claimed to be semantic scenes."""
    _positive(seconds, "context_window")
    duration = report["metadata"]["duration_seconds"]
    _positive(duration, "duration")
    count = math.ceil(duration/seconds)
    if count > 1000:
        raise ValueError("More than 1000 context windows; increase --context-window")
    obs = sorted(report.get("visual_observations",[]), key=lambda o:o["frame_time_seconds"])
    speech = sorted(report.get("speech_segments",[]), key=lambda s:s["start_seconds"])
    shots = report.get("shots",[])
    vids = [{"id":f"v{i:04d}","time_seconds":o["frame_time_seconds"],
             "shot_id":o["shot_id"],"caption_hypothesis":o.get("description","")}
            for i,o in enumerate(obs, 1)]
    audios = [{"id":f"a{i:04d}","start_seconds":s["start_seconds"],
               "end_seconds":s["end_seconds"],"text":s.get("text","")}
              for i,s in enumerate(speech, 1)]
    windows = []
    for i in range(count):
        a=round(seconds*i,6)
        b=round(min(duration,seconds*(i+1)),6)
        frames=[o for o in vids if a <= o["time_seconds"] < b or
                (i==count-1 and o["time_seconds"] == b)]
        words=[s for s in audios if s["start_seconds"] < b and s["end_seconds"] > a]
        overlapping=[s["shot_id"] for s in shots if s["start_seconds"]<b and s["end_seconds"]>a]
        windows.append({"window_id":f"w{i+1:04d}","start_seconds":a,"end_seconds":b,
                        "shot_ids":overlapping,"frames":frames,"speech":words})
    return windows


def choose_frames(frames, limit):
    if isinstance(limit,bool) or not isinstance(limit,int) or not 1<=limit<=8:
        raise ValueError("max_context_images must be between 1 and 8")
    if len(frames)<=limit:
        return frames
    if limit==1:
        return [frames[len(frames)//2]]
    return [frames[round(i*(len(frames)-1)/(limit-1))] for i in range(limit)]


def local_ollama(messages, model, timeout=300):
    """Single local HTTP request; no user-supplied remote URL is accepted."""
    data=json.dumps({"model":model,"messages":messages,"format":CONTEXT_RESPONSE_SCHEMA,
                     "stream":False,"options":{"temperature":0}},ensure_ascii=False).encode("utf-8")
    req=urllib.request.Request(LOCAL_API,data=data,headers={"Content-Type":"application/json"},method="POST")
    try:
        with urllib.request.urlopen(req,timeout=timeout) as response:
            body=json.load(response)
    except (urllib.error.URLError,TimeoutError,OSError) as exc:
        raise RuntimeError("Local Ollama unavailable or inference failed. Install Ollama, "
                           f"run 'ollama pull {model}' and ensure it is listening at "
                           "127.0.0.1:11434. "+str(exc)) from exc
    try:
        output=json.loads(body["message"]["content"])
    except (KeyError,TypeError,ValueError) as exc:
        raise RuntimeError("Ollama did not return a JSON object") from exc
    if not isinstance(output,dict):
        raise RuntimeError("Ollama output must be a JSON object")
    return output


def _jpeg_frame(video, timestamp, frame_loader):
    frame = frame_loader(video, timestamp)
    try:
        image=io.BytesIO()
        frame.save(image,format="JPEG",quality=75)
    except (ValueError,TypeError,AttributeError,OSError) as exc:
        raise RuntimeError(f"Unable to encode sampled frame at {timestamp}s") from exc
    return base64.b64encode(image.getvalue()).decode("ascii")


def _limited_string(value, field, limit=1200):
    if not isinstance(value,str) or len(value)>limit:
        raise ValueError(f"Model {field} must be a string no longer than {limit} chars")
    return value.strip()


def validate_hypothesis(output, current_ids, previous_ids):
    """Reject made-up source identifiers and malformed model JSON."""
    summary=_limited_string(output.get("summary"),"summary")
    if not summary:
        raise ValueError("Model summary cannot be empty")
    raw_events=output.get("events")
    if not isinstance(raw_events,list) or len(raw_events)>8:
        raise ValueError("Model events must be a list of <=8 entries")
    events=[]
    for event in raw_events:
        if not isinstance(event,dict):
            raise ValueError("Model event is not an object")
        claim=_limited_string(event.get("claim"),"claim",400)
        ids=event.get("evidence_ids")
        kind=event.get("kind")
        if not claim or not isinstance(ids,list) or not ids or any(
            not isinstance(e,str) or e not in current_ids for e in ids):
            raise ValueError("Event must cite only existing current-window evidence IDs")
        if kind not in ("observation","interpretation"):
            raise ValueError("Event kind must be observation or interpretation")
        events.append({"claim":claim,"evidence_ids":list(dict.fromkeys(ids)),"kind":kind,
                       "human_groundedness_review":"not_reviewed"})
    links=output.get("links_previous_windows")
    if not isinstance(links,list) or any(not isinstance(x,str) or x not in previous_ids for x in links):
        raise ValueError("Memory links must reference supplied previous windows")
    questions=output.get("open_questions")
    if not isinstance(questions,list) or len(questions)>8:
        raise ValueError("Open questions must be a list of <=8 items")
    return {"summary_hypothesis":summary,"events":events,
            "continuity_hypothesis":_limited_string(output.get("continuity"),"continuity"),
            "links_previous_windows":list(dict.fromkeys(links)),
            "open_questions":[_limited_string(q,"question",300) for q in questions],
            "validation":"schema_and_citations_only",
            "human_groundedness_review":"not_reviewed"}


def build_context(report,video,*,model="gemma3:4b",window_seconds=12.,
                  max_images=3,timeout=300,backend=None,frame_loader=None,
                  on_window=None, resume=None):
    """Sequential multimodal inference with capped memory and evidence citations.

    A model cannot assume adjacent windows depict a continuous scene.
    on_window allows callers to checkpoint completed windows.
    """
    if not isinstance(model,str) or not model.strip():
        raise ValueError("context model must be nonempty")
    _positive(timeout, "context timeout")
    windows=make_windows(report,window_seconds)
    choose_frames([],max_images)  # validate cap even when video is empty
    backend=backend or local_ollama
    frame_loader=frame_loader or _extract_image
    # Local-only audit map: trace event IDs to timestamps and original ASR
    # excerpts. Do NOT add this to the sanitized public trailer report.
    evidence_index={}
    for window in windows:
        for frame in window["frames"]:
            evidence_index[frame["id"]]={
                "source":"sampled_video_frame",
                "time_seconds":frame["time_seconds"],
                "shot_id":frame["shot_id"],
                "caption_hypothesis":frame["caption_hypothesis"]}
        for segment in window["speech"]:
            evidence_index[segment["id"]]={
                "source":"unverified_asr",
                "start_seconds":segment["start_seconds"],
                "end_seconds":segment["end_seconds"],
                "asr_excerpt":segment["text"][:400]}
    result={"schema_version":"0.1.0","context_model":model,
            "source_duration_seconds":report["metadata"]["duration_seconds"],
            "analysis_fingerprint":analysis_fingerprint(report),
            "max_context_images":max_images,
            "window_duration_seconds":window_seconds,
            "mode":"local_multimodal_hypotheses",
            "privacy":"PRIVATE: contains excerpts of ASR dialogue and model-generated hypotheses; never upload publicly",
            "evidence_index":evidence_index,
            "verified_accuracy":None,
            "warnings":["Frames are sparse samples; events between frames are unknown.",
                        "Input ASR and BLIP captions are fallible.",
                        "Earlier AI summaries are hypotheses, not known continuity.",
                        "Links are model hypotheses, not verified character or plot identities."],
            "windows":[]}
    if resume is not None:
        if not isinstance(resume,dict) or any((
            resume.get("analysis_fingerprint") != result["analysis_fingerprint"],
            resume.get("context_model") != model,
            resume.get("window_duration_seconds") != window_seconds,
            resume.get("max_context_images") != max_images,
        )):
            raise ValueError("Saved context is incompatible with this video, parameters or model")
        done = resume.get("windows")
        if not isinstance(done,list) or len(done)>len(windows):
            raise ValueError("Saved context has invalid window count")
        for i,item in enumerate(done):
            expected=windows[i]
            if (not isinstance(item,dict) or
                item.get("window_id")!=expected["window_id"] or
                item.get("start_seconds")!=expected["start_seconds"] or
                item.get("end_seconds")!=expected["end_seconds"]):
                raise ValueError(f"Saved context window {i+1} does not match video")
            if not isinstance(item.get("summary_hypothesis"),str):
                raise ValueError(f"Saved context window {i+1} lacks a summary")
            # Prevent resuming records that reference frames or ASR from
            # a different selection, even when the source report matches.
            selected=choose_frames(expected["frames"],max_images)
            ids={x["id"] for x in selected}
            ids.update(x["id"] for x in expected["speech"][:12])
            if item.get("evidence_ids")!=sorted(ids):
                raise ValueError(f"Saved context evidence mismatch in {expected['window_id']}")
        # A weak/invalid window must be retried, never cemented into memory.
        retry_at=next((i for i,w in enumerate(done)
                       if w.get("validation")=="model_response_invalid_abstained"),len(done))
        result["windows"] = done[:retry_at]
        if done:
            print(f"Resuming at context window {len(result['windows'])+1}/{len(windows)}",
                  file=sys.stderr)
    for window in windows[len(result["windows"]):]:
        selected=choose_frames(window["frames"],max_images)
        prior=result["windows"][-2:]
        prev=[{"window_id":w["window_id"],"summary_hypothesis":w["summary_hypothesis"],
               "evidence_ids":w["evidence_ids"]} for w in prior]
        evidence=[{"id":x["id"],"time_seconds":x["time_seconds"],
                   "caption_hypothesis":x["caption_hypothesis"]} for x in selected]
        evidence.extend({"id":x["id"],"start_seconds":x["start_seconds"],
                         "end_seconds":x["end_seconds"],"asr_hypothesis":x["text"][:400]}
                        for x in window["speech"][:12])
        current_ids={e["id"] for e in evidence}
        user_message={"window_id":window["window_id"],
                      "interval_seconds":[window["start_seconds"],window["end_seconds"]],
                      "shot_count":len(window["shot_ids"]),
                      "visual_frame_count":len(selected),
                      "visual_coverage_warning":("NO sampled frames in this interval" if not selected
                                                 else "Images are isolated instants, not motion"),
                      "current_evidence":evidence,
                      "unverified_previous_memory":prev}
        message={"role":"user","content":json.dumps(user_message,ensure_ascii=False)}
        if selected:
            message["images"]=[_jpeg_frame(video,x["time_seconds"],frame_loader) for x in selected]
        if not current_ids:
            # There is nothing to observe. Never ask a model to fill empty time.
            validated=validate_hypothesis({
                "summary":"No sampled visual or transcribed speech evidence in this interval.",
                "events":[],"continuity":"unknown",
                "links_previous_windows":[],"open_questions":["What occurs in this unsampled interval?"]
            },current_ids,set())
        else:
            messages=[{"role":"system","content":SYSTEM_PROMPT},message]
            for attempt in range(2):
                output=backend(messages,model,timeout)
                if not isinstance(output,dict):
                    raise RuntimeError("Context backend must return a JSON object")
                try:
                    validated=validate_hypothesis(output,current_ids,
                                                 {x["window_id"] for x in prior})
                    break
                except ValueError as exc:
                    if attempt:
                        # Keep the rest of the trailer moving. Invalid output
                        # must NOT be recast as a grounded observation.
                        validated=validate_hypothesis({
                            "summary":"Model response invalid; audiovisual interpretation unavailable.",
                            "events":[],"continuity":"unknown",
                            "links_previous_windows":[],
                            "open_questions":["Re-run this window or inspect source footage."]
                        }, current_ids, set())
                        validated["validation"]="model_response_invalid_abstained"
                        validated["model_error"]=str(exc)
                        print(f"WARNING: {window['window_id']} invalid model JSON; "
                              f"abstaining after retry ({exc})",file=sys.stderr)
                        break
                    messages.extend([
                        {"role":"assistant","content":json.dumps(output,ensure_ascii=False)},
                        {"role":"user","content":"Your JSON did not match the required schema: "
                         +str(exc)+". Return corrected JSON only, using ONLY the given "
                         "evidence IDs and previous window IDs."}
                    ])
        item={"window_id":window["window_id"],
              "start_seconds":window["start_seconds"],"end_seconds":window["end_seconds"],
              "evidence_ids":sorted(current_ids),
              "sampled_frame_times_seconds":[x["time_seconds"] for x in selected],
              "asr_segment_count":len(window["speech"]),
              "observed_frame_count":len(selected),
              "shot_ids":window["shot_ids"],**validated}
        result["windows"].append(item)
        if on_window:
            on_window(result)
        print(f"Context {len(result['windows'])}/{len(windows)}: "
              f"{window['start_seconds']:.1f}-{window['end_seconds']:.1f}s",file=sys.stderr)
    result["status"]=("complete_with_abstentions"
                      if any(w["validation"]=="model_response_invalid_abstained"
                             for w in result["windows"]) else "complete")
    return result
