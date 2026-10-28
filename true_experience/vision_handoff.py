"""Offline portable shot-review folder for Codex or another vision-capable AI.

Uses an existing shots ZIP; never calls AI, downloads models, or uploads.
"""
from __future__ import annotations
import argparse
import json
import shutil
import sys
import zipfile
from pathlib import Path
from .vision_contract import (
    BATCH_SCHEMA, load_archive, new_review, selected_shots, source_digest,
)

PROMPT = """# Shot-by-shot visual review task

You are reviewing INDIVIDUAL STILL FRAMES, not a video. No audio is provided.
Do not claim to know unseen movement, speakers, plot, characters, causal
relationships, or emotions unless directly visible.

Read manifest.json for selected shots and view the matching images in shots/
in their exact chronological order. shot_id and timestamps are authoritative.

For every shot, write an independent, evidence-grounded assessment:
- shot_id: exactly the supplied integer.
- visible_description: short concrete description of the visual evidence.
- visible_objects: visible objects and generic descriptions of people.
- visible_action: only clearly visually supported, otherwise 'unknown'.
- uncertainties: what cannot be established from the still.

If evidence/ images are present, they show extra precise sampled frames
with IDs such as v0001, linked through manifest.json. Use them only to
check what is actually visible, not to invent motion between frames.

Do not trust earlier model captions. Do not infer movie title, actor or
character names. Note ambiguity rather than guessing.

Create shot_review.json with metadata from shot_review.template.json,
filling its shots array in shot_id order. Use shot_review.schema.json.
Keep source_manifest_sha256 unchanged. Output valid JSON only. A complete
result includes one entry for EVERY selected shot.

Adjacent stills can be used for tentative continuity hypotheses, but
cannot establish unobserved movement or cause. No audio is available.
Never upload these images to an external service without permission.
"""

README = """# Portable The True Experience visual review

This folder can be opened with Codex or any image-capable AI tool.
Creation requires no OpenAI API key and makes no network requests.

1. Read TASK.md and manifest.json.
2. Inspect shot JPEGs in chronological order.
3. Write shot_review.json using shot_review.template.json.
4. Compare this JSON against results from other models; verify important
   disagreements against the actual source video.

Codex recognizes AGENTS.md but image inspection capabilities depend on
the Codex environment. API usage and Codex access are separate services.
Frames may be copyrighted; do not redistribute without permission.
"""


def create_handoff(archive,output,*,max_shots=0,overwrite=False):
    manifest=load_archive(archive)
    shots=selected_shots(manifest,max_shots)
    output=Path(output)
    if output.exists():
        if not overwrite:
            raise FileExistsError(f"{output} already exists; choose another folder")
        if not output.is_dir() or any(output.iterdir()):
            raise ValueError("Only an empty existing folder can be reused")
    output.mkdir(parents=True,exist_ok=True)
    (output/"shots").mkdir(exist_ok=True)
    (output/"evidence").mkdir(exist_ok=True)
    chosen_ids={s["shot_id"] for s in shots}
    visual_evidence=[{"evidence_id":v["evidence_id"],"shot_id":v["shot_id"],
                      "frame_time_seconds":v["frame_time_seconds"],"image":v["image"]}
                     for v in manifest.get("visual_evidence",[])
                     if v["shot_id"] in chosen_ids]
    subset={
        "schema_version":"1.0",
        "source_video_name":manifest.get("source_video_name"),
        "video_duration_seconds":manifest.get("video_duration_seconds"),
        "selection":{"selected_count":len(shots),
                     "source_shot_count":len(manifest["shots"])},
        "source_manifest_sha256":source_digest(shots),
        "shots":[{"shot_id":s["shot_id"],"image":s["image"],
                  "start_seconds":s["start_seconds"],
                  "end_seconds":s["end_seconds"],
                  "frame_time_seconds":s["frame_time_seconds"],
                  "visual_evidence_ids":s.get("visual_evidence_ids",[])} for s in shots],
        "visual_evidence":visual_evidence
    }
    try:
        with zipfile.ZipFile(archive) as z:
            for s in shots:
                name=s["image"]
                with z.open(name) as src, (output/name).open("wb") as dst:
                    shutil.copyfileobj(src,dst,64*1024)
        with zipfile.ZipFile(archive) as z:
            for v in visual_evidence:
                name=v["image"]
                if not isinstance(name,str) or not name.startswith("evidence/") or                    ".." in Path(name).parts or name not in z.namelist() or                    z.getinfo(name).file_size>10_000_000:
                    raise ValueError("Invalid evidence frame path")
                with z.open(name) as src, (output/name).open("wb") as dst:
                    shutil.copyfileobj(src,dst,64*1024)
        for filename, data in (
            ("manifest.json",subset),
            ("shot_review.schema.json",{
                "type":"object","additionalProperties":False,
                "properties":{
                    **{key:{"type":"string"} for key in (
                        "schema_version","backend","model",
                        "source_manifest_sha256","source")},
                    "limitations":{"type":"array","items":{"type":"string"}},
                    "shots":BATCH_SCHEMA["properties"]["shots"]
                },
                "required":["schema_version","backend","model",
                            "source_manifest_sha256","source",
                            "limitations","shots"]
            }),
            ("shot_review.template.json",new_review(shots,"codex_manual","unspecified")),
        ):
            (output/filename).write_text(json.dumps(data,ensure_ascii=False,indent=2)+"\n",
                                        encoding="utf-8")
        (output/"TASK.md").write_text(PROMPT,encoding="utf-8")
        (output/"AGENTS.md").write_text(
            "Follow TASK.md. Output shot_review.json. Never upload frames without consent.\n",
            encoding="utf-8")
        (output/"README.md").write_text(README,encoding="utf-8")
    except BaseException:
        (output/"INCOMPLETE.txt").write_text(
            "Package creation failed. Delete and regenerate this directory.\n")
        raise
    return {"path":str(output),"shot_count":len(shots),
            "source_manifest_sha256":subset["source_manifest_sha256"]}


def main(argv=None):
    p=argparse.ArgumentParser(description="Create portable visual-review folder from shot ZIP")
    p.add_argument("archive",type=Path)
    p.add_argument("-o","--output",type=Path,default=Path("shot_review_handoff"))
    p.add_argument("--max-shots",type=int,default=0,
                   help="First N chronological shots (0 = all)")
    args=p.parse_args(argv)
    try:
        result=create_handoff(args.archive,args.output,max_shots=args.max_shots)
        print(f"Created {result['shot_count']} shot images in {result['path']}. "
              "Read TASK.md; produce shot_review.json.")
        return 0
    except (ValueError,KeyError,TypeError,OSError,zipfile.BadZipFile) as exc:
        print(f"Handoff failed: {exc}",file=sys.stderr)
        return 1


if __name__=="__main__":
    raise SystemExit(main())
