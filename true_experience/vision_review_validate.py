"""Validate and optionally compare provider-neutral shot JSON outputs.

Different phrasings are disagreements for human review, NOT a measure of
accuracy without annotated ground-truth frames.
"""
import argparse
import json
import sys
from pathlib import Path
from .vision_contract import load_archive,selected_shots,validate_review


def compare_reviews(left,right):
    a={s["shot_id"]:s for s in left["shots"]}
    b={s["shot_id"]:s for s in right["shots"]}
    overlap=sorted(set(a)&set(b))
    changed=[{"shot_id":sid,
              "first_description":a[sid]["visible_description"],
              "second_description":b[sid]["visible_description"]}
             for sid in overlap if a[sid]["visible_description"].strip().casefold() !=
             b[sid]["visible_description"].strip().casefold()]
    return {"overlapping_shots":len(overlap),
            "different_wording_count":len(changed),
            "different_wording_examples":changed[:100],
            "accuracy":None,
            "accuracy_note":"Textual differences are NOT correctness or error rates."}


def main(argv=None):
    p=argparse.ArgumentParser(description="Validate visual review JSON; optionally compare another model")
    p.add_argument("archive",type=Path,help="Original shot ZIP")
    p.add_argument("review",type=Path,help="Completed API or manual JSON")
    p.add_argument("--compare",type=Path,help="Second review file")
    p.add_argument("--max-shots",type=int,default=0,
                   help="Number selected when reviews were prepared (0 = all)")
    args=p.parse_args(argv)
    try:
        manifest=load_archive(args.archive)
        shots=selected_shots(manifest,args.max_shots)
        first=json.loads(args.review.read_text(encoding="utf-8"))
        validate_review(first,shots)
        summary={"first_review":str(args.review),"valid":True,
                 "reviewed_shots":len(first["shots"]),
                 "expected_shots":len(shots),"complete":len(first["shots"])==len(shots)}
        if args.compare:
            second=json.loads(args.compare.read_text(encoding="utf-8"))
            validate_review(second,shots)
            summary["comparison"]=compare_reviews(first,second)
        print(json.dumps(summary,indent=2,ensure_ascii=False))
        return 0
    except (OSError,KeyError,ValueError,TypeError) as exc:
        print(f"Invalid review: {exc}",file=sys.stderr)
        return 1


if __name__=="__main__":
    raise SystemExit(main())
