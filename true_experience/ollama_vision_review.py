"""Local Ollama shot-review backend for the same strict JSON format as OpenAI.

The user selects a vision-capable model (Qwen3-VL recommended for evaluation).
Images are sent only to Ollama's 127.0.0.1 interface, never to a cloud API.
No pretrained weights are downloaded by this Python module.
"""
from __future__ import annotations

import argparse
import base64
import json
import sys
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

from .vision_contract import (
    BATCH_SCHEMA, INSTRUCTIONS, load_archive, new_review,
    selected_shots, validate_batch, validate_review,
)

OLLAMA_API="http://127.0.0.1:11434/api/chat"


def request_payload(batch, image_data, model):
    """Require the model to return one grounded record per attached image."""
    if len(batch)!=len(image_data):
        raise ValueError("Image count differs from batch")
    if not batch or len(batch)>4:
        raise ValueError("Batch must contain between 1 and 4 shots")
    for data in image_data:
        if not isinstance(data,bytes) or not data.startswith(b"\xff\xd8"):
            raise ValueError("Archive contains a non-JPEG frame")
    instructions = (
        "Inspect each image in order; do not guess characters, scene continuity "
        "or movement from stills. Describe only visible evidence. "
        "Return the complete required JSON object with exactly these shot IDs: "
        +json.dumps([{"shot_id":s["shot_id"],
                      "frame_time_seconds":s["frame_time_seconds"]}
                     for s in batch])
    )
    return {
        "model":model,"stream":False,"format":BATCH_SCHEMA,
        "options":{"temperature":0},
        "messages":[
            {"role":"system","content":INSTRUCTIONS},
            {"role":"user","content":instructions,
             "images":[base64.b64encode(raw).decode("ascii") for raw in image_data]}
        ]
    }


def call_ollama(payload,timeout=300):
    """Fixed localhost endpoint; do not expose a configurable remote address."""
    if not isinstance(timeout,(float,int)) or timeout<=0:
        raise ValueError("Timeout must be positive")
    req=urllib.request.Request(
        OLLAMA_API,data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type":"application/json"},method="POST")
    try:
        with urllib.request.urlopen(req,timeout=timeout) as response:
            data=json.load(response)
    except urllib.error.HTTPError as exc:
        raise RuntimeError(f"Local Ollama HTTP {exc.code}; check model availability and "
                           "Ollama version. Qwen3-VL requires Ollama 0.12.7+.") from exc
    except (urllib.error.URLError,TimeoutError,OSError) as exc:
        raise RuntimeError("Cannot reach local Ollama at 127.0.0.1:11434; "
                           "start Ollama and pull the requested vision model.") from exc
    if not isinstance(data,dict):
        raise ValueError("Ollama returned an invalid response envelope")
    return data


def extract_response(data):
    try:
        content=data["message"]["content"]
        parsed=json.loads(content)
    except (KeyError,TypeError,ValueError) as exc:
        raise ValueError("Ollama did not return parseable review JSON") from exc
    if not isinstance(parsed,dict):
        raise ValueError("Ollama review must be a JSON object")
    return parsed


def _atomic_write(path,result):
    path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_name(path.name+".tmp")
    temp.write_text(json.dumps(result,indent=2,ensure_ascii=False,
                               allow_nan=False)+"\n",encoding="utf-8")
    temp.replace(path)


def review_archive(archive,output,*,model="qwen3-vl:8b",max_shots=12,
                   batch_size=1,timeout=300,resume=False,dry_run=False,
                   requester=None):
    manifest=load_archive(archive)
    shots=selected_shots(manifest,max_shots)
    if isinstance(batch_size,bool) or not isinstance(batch_size,int) or not 1<=batch_size<=4:
        raise ValueError("batch_size must be an integer from 1 to 4")
    if not isinstance(model,str) or not model.strip():
        raise ValueError("Model must be nonempty")
    batches=(len(shots)+batch_size-1)//batch_size
    if dry_run:
        return {"dry_run":True,"backend":"ollama","model":model,
                "shots":len(shots),"batch_size":batch_size,"requests":batches,
                "upload":"Local Ollama only (127.0.0.1); no OpenAI API"}
    output=Path(output)
    report=new_review(shots,"ollama",model)
    report["status"]="partial"
    report["batch_size"]=batch_size
    report["source_shot_count"]=len(manifest["shots"])
    if output.exists():
        if not resume:
            raise FileExistsError(f"{output} exists; use --resume, or a new output filename")
        existing=json.loads(output.read_text(encoding="utf-8"))
        validate_review(existing,shots)
        if (existing.get("backend")!="ollama" or existing.get("model")!=model
                or existing.get("batch_size")!=batch_size):
            raise ValueError("Existing review is incompatible with chosen local model or batch size")
        done=len(existing["shots"])
        if done%batch_size and done!=len(shots):
            raise ValueError("Partial output ends mid-batch; cannot resume safely")
        report=existing
    if len(report["shots"])==len(shots):
        return report
    sender=requester or call_ollama
    with zipfile.ZipFile(archive) as z:
        for start in range(len(report["shots"]),len(shots),batch_size):
            group=shots[start:start+batch_size]
            images=[z.read(s["image"]) for s in group]
            data=request_payload(group,images,model)
            response=sender(data) if requester is not None else sender(data,timeout)
            reviewed=validate_batch(extract_response(response),group)
            report["shots"].extend(reviewed)
            report["status"]="complete" if len(report["shots"])==len(shots) else "partial"
            _atomic_write(output,report)
            print(f"Local vision review: {len(report['shots'])}/{len(shots)} shots",
                  file=sys.stderr)
    return report


def main(argv=None):
    p=argparse.ArgumentParser(description="Analyze shot images with local Ollama vision model")
    p.add_argument("archive",type=Path,help="shot ZIP produced by true_experience.shot_archive")
    p.add_argument("-o","--output",type=Path,default=Path("ollama_shot_review.json"))
    p.add_argument("--model",default="qwen3-vl:8b",
                   help="Image-capable Ollama model, e.g. qwen3-vl:8b or gemma3:12b")
    p.add_argument("--max-shots",type=int,default=12,help="First N shots; 0 = all")
    p.add_argument("--batch-size",type=int,default=1,help="Frames per model request (1..4)")
    p.add_argument("--timeout",type=int,default=300)
    p.add_argument("--resume",action="store_true",help="Continue a checkpointed output")
    p.add_argument("--dry-run",action="store_true",help="Preview without model inference")
    a=p.parse_args(argv)
    try:
        result=review_archive(a.archive,a.output,model=a.model,
                              max_shots=a.max_shots,batch_size=a.batch_size,
                              timeout=a.timeout,resume=a.resume,dry_run=a.dry_run)
        if a.dry_run:
            print(json.dumps(result,indent=2))
        else:
            print(f"Saved {len(result['shots'])} local visual descriptions to {a.output}")
        return 0
    except (ValueError,OSError,RuntimeError,KeyError,TypeError,zipfile.BadZipFile) as exc:
        print(f"Local shot review failed: {exc}",file=sys.stderr)
        return 1


if __name__=="__main__":
    raise SystemExit(main())
