"""CLI for report consistency and optional transcript WER checks."""
import argparse
import json
from pathlib import Path
from .evaluation import validate_report, word_error_rate
from .srt import load_srt


def main(argv=None):
    p = argparse.ArgumentParser(description="Evaluate True Experience report")
    p.add_argument("report", type=Path)
    p.add_argument("--reference-srt", type=Path,
                   help="Independent human transcript; not the generated ASR or input SRT")
    p.add_argument("-o", "--output", type=Path)
    args = p.parse_args(argv)
    try:
        data = json.loads(args.report.read_text(encoding="utf-8"))
        summary = {"structural": validate_report(data),
                   "asr_word_error_rate": None,
                   "caption_groundedness": "not_evaluated_requires_human_review"}
        if args.reference_srt:
            speech = data.get("speech_segments", [])
            if not speech:
                raise ValueError("No ASR speech segments; WER cannot be measured")
            reference = " ".join(c["text"] for c in load_srt(args.reference_srt))
            hypothesis = " ".join(c["text"] for c in speech)
            summary["asr_word_error_rate"] = word_error_rate(reference, hypothesis)
        payload = json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
        if args.output:
            args.output.write_text(payload, encoding="utf-8")
        else:
            print(payload, end="")
        return 0 if summary["structural"]["valid"] else 1
    except (ValueError, OSError, KeyError, TypeError) as exc:
        p.exit(1, f"Evaluation failed: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
