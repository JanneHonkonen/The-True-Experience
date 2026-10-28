"""Portable, provider-neutral shot vision review contract.

Reviews describe individual stills; temporal motion, cause, speaker identity,
and character identity cannot be established from isolated frames.
"""
from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path

SHOT_FIELDS=("shot_id","visible_description","visible_objects","visible_action",
             "uncertainties")
ITEM_SCHEMA={
    "type":"object","additionalProperties":False,
    "properties":{
        "shot_id":{"type":"integer"},
        "visible_description":{"type":"string"},
        "visible_objects":{"type":"array","items":{"type":"string"}},
        "visible_action":{"type":"string"},
        "uncertainties":{"type":"array","items":{"type":"string"}}
    },"required":list(SHOT_FIELDS)
}
BATCH_SCHEMA={
    "type":"object","additionalProperties":False,
    "properties":{"shots":{"type":"array","items":ITEM_SCHEMA}},
    "required":["shots"]
}
INSTRUCTIONS=(
    "Analyze ONLY the supplied still images in the exact numbered order. "
    "Do not infer unseen movement, causal links, identities or story from the "
    "movie title or prior familiarity. Describe what is visually supported. "
    "If motion is not demonstrable in a still image, use 'unknown' as the "
    "visible_action. Use cautious descriptions, not actor/character names. "
    "Do not mistake a previous model caption for ground truth. "
    "Give one JSON entry per shot, using the provided exact shot_id values. "
    "uncertainties must explain genuinely ambiguous aspects. "
    "The response must conform to the supplied JSON Schema."
)


def canonical_digest(data):
    return hashlib.sha256(json.dumps(data,ensure_ascii=False,sort_keys=True,
                                      separators=(",",":"),allow_nan=False).encode()).hexdigest()


def load_archive(path):
    """Read an author-produced shot ZIP without extracting untrusted paths."""
    path=Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    with zipfile.ZipFile(path) as zipf:
        info=zipf.getinfo("manifest.json")
        if info.file_size>10_000_000:
            raise ValueError("Manifest exceeds 10 MB")
        manifest=json.loads(zipf.read(info))
        shots=manifest.get("shots")
        if not isinstance(shots,list) or not shots or len(shots)>10000:
            raise ValueError("Archive must have 1-10,000 shots")
        names=set(zipf.namelist())
        previous_id=0
        for item in shots:
            if not isinstance(item,dict):
                raise ValueError("Invalid shot entry")
            shot_id=item.get("shot_id")
            name=item.get("image")
            if isinstance(shot_id,bool) or not isinstance(shot_id,int) or shot_id<=previous_id:
                raise ValueError("Shot IDs must be ascending positive integers")
            if not isinstance(name,str) or not name.startswith("shots/") or \
                ".." in Path(name).parts or not name.lower().endswith((".jpg",".jpeg")) or name not in names:
                raise ValueError("Invalid shot image path")
            if zipf.getinfo(name).file_size>10_000_000:
                raise ValueError("Shot image larger than 10 MB")
            previous_id=shot_id
        return manifest


def selected_shots(manifest, limit=0):
    shots=manifest["shots"]
    if not isinstance(limit,int) or isinstance(limit,bool) or limit<0:
        raise ValueError("max_shots must be a nonnegative integer")
    return shots if limit==0 else shots[:limit]


def source_digest(shots):
    return canonical_digest([{"shot_id":s["shot_id"],"image":s["image"],
                              "start_seconds":s["start_seconds"],
                              "end_seconds":s["end_seconds"],
                              "frame_time_seconds":s["frame_time_seconds"]} for s in shots])


def validate_batch(output, batch):
    if not isinstance(output,dict) or set(output)!={"shots"} or not isinstance(output["shots"],list):
        raise ValueError("Expected a JSON object containing only a shots array")
    expected=[s["shot_id"] for s in batch]
    observed=[]
    for item in output["shots"]:
        if not isinstance(item,dict) or set(item)!=set(SHOT_FIELDS):
            raise ValueError("Invalid shot review keys")
        sid=item["shot_id"]
        if isinstance(sid,bool) or not isinstance(sid,int):
            raise ValueError("Invalid shot ID type")
        observed.append(sid)
        for key in ("visible_description","visible_action"):
            if not isinstance(item[key],str) or len(item[key])>2000:
                raise ValueError(f"Invalid {key}")
        for key in ("visible_objects","uncertainties"):
            val=item[key]
            if not isinstance(val,list) or len(val)>30 or any(
                not isinstance(x,str) or len(x)>300 for x in val
            ):
                raise ValueError(f"Invalid {key}")
    if observed!=expected:
        raise ValueError(f"Expected shot IDs {expected}; received {observed}")
    return output["shots"]


def new_review(shots,backend,model):
    return {"schema_version":"1.0","backend":backend,"model":model,
            "source_manifest_sha256":source_digest(shots),
            "source":"still_frame_visual_review",
            "limitations":["Still frames cannot establish motion, exact causality or speech.",
                           "Model-produced descriptions require human visual verification."],
            "shots":[]}


def validate_review(review,shots):
    if not isinstance(review,dict) or review.get("source_manifest_sha256")!=source_digest(shots):
        raise ValueError("Review source manifest hash does not match selected shots")
    actual=review.get("shots")
    if not isinstance(actual,list):
        raise ValueError("Review shots must be a list")
    validate_batch({"shots":actual},shots[:len(actual)])
    return review
