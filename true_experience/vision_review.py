"""One user-facing CLI: local Ollama (default) or explicitly paid OpenAI API.

Both consume the same timestamped shot ZIP and emit the same JSON contract.
No Codex/manual packaging step is required.
"""
from __future__ import annotations
import argparse
import json
import sys
import zipfile
from pathlib import Path
from .ollama_vision_review import review_archive as local_review
from .openai_vision_review import review_archive as api_review


def main(argv=None):
    p=argparse.ArgumentParser(
        description="Review shot JPEGs with Ollama (default) or the optional OpenAI API")
    p.add_argument("archive",type=Path,help="Existing timestamped shots ZIP")
    p.add_argument("--backend",choices=("ollama","openai"),default="ollama")
    p.add_argument("--model",help="Ollama model e.g. qwen3-vl:8b, or OpenAI API model")
    p.add_argument("--max-shots",type=int,default=12,
                   help="Analyze the first N shots (0 = all); 12-shot pilot by default")
    p.add_argument("--batch-size",type=int,help="Default 1 for Ollama, 4 for OpenAI API")
    p.add_argument("-o","--output",type=Path,help="Output JSON; defaults per backend")
    p.add_argument("--resume",action="store_true",help="Resume partially completed output")
    p.add_argument("--dry-run",action="store_true",help="Preview without sending images")
    p.add_argument("--timeout",type=int,default=300,help="Ollama timeout per request")
    p.add_argument("--detail",choices=("low","high","auto"),default="high",
                   help="OpenAI API image detail; unused by Ollama")
    p.add_argument("--confirm-upload",action="store_true",
                   help="Explicit consent to send frames to OpenAI API")
    a=p.parse_args(argv)
    try:
        if a.backend=="ollama":
            if a.confirm_upload:
                raise ValueError("--confirm-upload applies only to the OpenAI API")
            if a.detail!="high":
                raise ValueError("--detail applies only to the OpenAI API")
            model=a.model or "qwen3-vl:8b"
            output=a.output or Path("ollama_shot_review.json")
            result=local_review(a.archive,output,model=model,max_shots=a.max_shots,
                                batch_size=a.batch_size if a.batch_size is not None else 1,
                                timeout=a.timeout,dry_run=a.dry_run,resume=a.resume)
        else:
            model=a.model or "gpt-5.4-mini"
            output=a.output or Path("openai_shot_review.json")
            result=api_review(a.archive,output,model=model,max_shots=a.max_shots,
                              batch_size=a.batch_size if a.batch_size is not None else 4,
                              detail=a.detail,confirm_upload=a.confirm_upload,
                              dry_run=a.dry_run,resume=a.resume)
        if a.dry_run:
            print(json.dumps(result,indent=2))
        else:
            print(f"Saved {len(result['shots'])} shot observations to {output}")
        return 0
    except (ValueError,OSError,TypeError,RuntimeError,KeyError,zipfile.BadZipFile) as exc:
        print(f"Shot review failed: {exc}",file=sys.stderr)
        return 1


if __name__=="__main__":
    raise SystemExit(main())
