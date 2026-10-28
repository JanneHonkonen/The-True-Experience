"""Opt-in, billed OpenAI vision review of the SAME archived still images as the Ollama reviewer.

Do not call this from the normal video pipeline. User must set
OPENAI_API_KEY and --confirm-upload to transmit any source imagery.
No key, images or raw API replies are logged or committed.
"""
from __future__ import annotations
import argparse
import base64
import json
import os
import sys
import urllib.error
import urllib.request
import zipfile
from pathlib import Path
from .vision_contract import (
    BATCH_SCHEMA, INSTRUCTIONS, load_archive, selected_shots, new_review,
    validate_batch, validate_review,
)

ENDPOINT="https://api.openai.com/v1/responses"
DETAILS=("low","high","auto")


def request_payload(batch,image_data,model,detail):
    """OpenAI Responses API with current strict Structured Outputs format."""
    if len(image_data)!=len(batch):
        raise ValueError("Image count differs from batch")
    parts=[{"type":"input_text",
            "text":"Review exactly these chronological frames with the required shot IDs: "
                   +json.dumps([{"shot_id":s["shot_id"],
                                 "start_seconds":s["start_seconds"],
                                 "end_seconds":s["end_seconds"],
                                 "sample_time_seconds":s["frame_time_seconds"]}
                                for s in batch])}]
    for raw in image_data:
        if not raw.startswith(b"\xff\xd8"):
            raise ValueError("Archive contains a non-JPEG frame")
        parts.append({"type":"input_image",
                      "image_url":"data:image/jpeg;base64,"+
                                  base64.b64encode(raw).decode("ascii"),
                      "detail":detail})
    return {"model":model,
            "store":False,
            "input":[
                {"role":"system","content":INSTRUCTIONS},
                {"role":"user","content":parts}
            ],
            "text":{"format":{
                "type":"json_schema","name":"shot_visual_review",
                "strict":True,"schema":BATCH_SCHEMA
            }}
    }


def extract_response(data):
    """Read server JSON output without relying on SDK-only output_text property."""
    if not isinstance(data,dict) or data.get("status")=="incomplete":
        raise ValueError("API response was incomplete")
    contents=[]
    for item in data.get("output",[]):
        if not isinstance(item,dict):
            continue
        for part in item.get("content",[]):
            if part.get("type")=="refusal":
                raise ValueError("Model refused visual review")
            if part.get("type")=="output_text":
                contents.append(part.get("text",""))
    if not contents:
        raise ValueError("API returned no structured text output")
    try:
        return json.loads("".join(contents))
    except (TypeError,ValueError) as exc:
        raise ValueError("API returned invalid JSON") from exc


def call_openai(payload,*,timeout=180,key=None):
    key = key or os.environ.get("OPENAI_API_KEY")
    if not key:
        raise ValueError("OPENAI_API_KEY is missing (no upload performed)")
    req=urllib.request.Request(ENDPOINT,
        data=json.dumps(payload,ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type":"application/json",
                 "Authorization":"Bearer "+key},method="POST")
    try:
        with urllib.request.urlopen(req,timeout=timeout) as response:
            result=json.load(response)
    except urllib.error.HTTPError as exc:
        raise RuntimeError(f"OpenAI API HTTP {exc.code}; inspect account, model access, "
                           "limits and billing. No key or image data will be printed.") from exc
    except (urllib.error.URLError,TimeoutError,OSError) as exc:
        raise RuntimeError("OpenAI API connection failed; no successful review "
                           f"was received: {type(exc).__name__}") from exc
    if not isinstance(result,dict):
        raise RuntimeError("Unexpected API response envelope")
    return result


def _write_json_atomic(path,data):
    path=Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_name(path.name+".tmp")
    temp.write_text(json.dumps(data,indent=2,ensure_ascii=False,allow_nan=False)+"\n",
                    encoding="utf-8")
    temp.replace(path)


def review_archive(archive,output,*,model="gpt-5.4-mini",max_shots=12,
                   batch_size=4,detail="high",confirm_upload=False,
                   dry_run=False,resume=False,requester=None):
    manifest=load_archive(archive)
    shots=selected_shots(manifest,max_shots)
    if detail not in DETAILS:
        raise ValueError("Invalid detail")
    if isinstance(batch_size,bool) or not isinstance(batch_size,int) or not 1<=batch_size<=8:
        raise ValueError("batch_size must be 1..8")
    if not isinstance(model,str) or not model.strip():
        raise ValueError("model must be nonempty")
    n_batches=(len(shots)+batch_size-1)//batch_size
    plan={"source":str(archive),"model":model,"shots":len(shots),
          "batch_size":batch_size,"requests":n_batches,
          "image_detail":detail,
          "billing":"OpenAI API billed separately; actual cost depends on token usage"}
    if dry_run:
        return {"dry_run":True,**plan}
    if not confirm_upload:
        raise ValueError("Refusing to upload video frames without --confirm-upload")
    if requester is None and not os.environ.get("OPENAI_API_KEY"):
        raise ValueError("OPENAI_API_KEY is not set")
    output=Path(output)
    review=new_review(shots,"openai_api",model)
    review["image_detail"]=detail
    review["batch_size"]=batch_size
    review["source_shot_count"]=len(manifest["shots"])
    review["usage_batches"]=[]
    review["status"]="partial"
    if output.exists():
        if not resume:
            raise FileExistsError(f"{output} exists; use --resume to continue")
        existing=json.loads(output.read_text(encoding="utf-8"))
        validate_review(existing,shots)
        if any((existing.get("backend")!="openai_api",
                existing.get("model")!=model,
                existing.get("image_detail")!=detail,
                existing.get("batch_size")!=batch_size)):
            raise ValueError("Existing review uses incompatible API settings")
        if len(existing["shots"])%batch_size and len(existing["shots"])!=len(shots):
            raise ValueError("Existing review ends mid-batch; do not silently repeat billing")
        review=existing
    if len(review["shots"])==len(shots):
        return review

    done=len(review["shots"])
    request_fn=requester or call_openai
    with zipfile.ZipFile(archive) as z:
        for start in range(done,len(shots),batch_size):
            group=shots[start:start+batch_size]
            raw=[z.read(s["image"]) for s in group]
            payload=request_payload(group,raw,model,detail)
            response=request_fn(payload)
            model_output=extract_response(response)
            observations=validate_batch(model_output,group)
            review["shots"].extend(observations)
            usage=response.get("usage")
            if isinstance(usage,dict):
                review["usage_batches"].append({
                    "shot_ids":[s["shot_id"] for s in group],
                    "input_tokens":usage.get("input_tokens"),
                    "output_tokens":usage.get("output_tokens")})
            review["status"]="complete" if len(review["shots"])==len(shots) else "partial"
            _write_json_atomic(output,review)
            print(f"Reviewed {len(review['shots'])}/{len(shots)} stills "
                  "(API usage billed separately)",file=sys.stderr)
    return review


def main(argv=None):
    p=argparse.ArgumentParser(description="Opt-in paid OpenAI image review from shot ZIP")
    p.add_argument("archive",type=Path)
    p.add_argument("-o","--output",type=Path,default=Path("openai_shot_review.json"))
    p.add_argument("--model",default="gpt-5.4-mini",
                   help="Vision-capable structured-output model e.g. gpt-5.4")
    p.add_argument("--max-shots",type=int,default=12,
                   help="Review first N shots (0 = all); default pilot is only 12")
    p.add_argument("--batch-size",type=int,default=4)
    p.add_argument("--detail",choices=DETAILS,default="high")
    p.add_argument("--dry-run",action="store_true",help="No network, key or billed requests")
    p.add_argument("--confirm-upload",action="store_true",
                   help="Explicitly allow sending source frame images to OpenAI API")
    p.add_argument("--resume",action="store_true",
                   help="Resume validated existing partial output; avoid duplicate calls")
    a=p.parse_args(argv)
    try:
        result=review_archive(a.archive,a.output,model=a.model,max_shots=a.max_shots,
                              batch_size=a.batch_size,detail=a.detail,
                              confirm_upload=a.confirm_upload,dry_run=a.dry_run,
                              resume=a.resume)
        if a.dry_run:
            print(json.dumps(result,indent=2))
        else:
            print(f"Saved {len(result['shots'])} reviewed shots to {a.output}")
        return 0
    except (OSError,ValueError,TypeError,RuntimeError,KeyError,zipfile.BadZipFile) as exc:
        print(f"Vision review failed: {exc}",file=sys.stderr)
        return 1


if __name__=="__main__":
    raise SystemExit(main())
