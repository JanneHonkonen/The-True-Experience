"""Parse local SubRip subtitles without external dependencies."""
import re
from pathlib import Path

STAMP = r"(\d{1,2}):(\d{2}):(\d{2})[,.](\d{1,3})"
ARROW = re.compile(rf"{STAMP}\s*-->\s*{STAMP}(?:\s+.*)?")

def seconds(parts):
    h, m, s, ms = parts
    if int(m) >= 60 or int(s) >= 60:
        raise ValueError("Invalid subtitle timestamp")
    return int(h)*3600 + int(m)*60 + int(s) + int(ms.ljust(3, "0"))/1000

def load_srt(path):
    raw = Path(path).read_text(encoding="utf-8-sig").replace("\r\n","\n").replace("\r","\n")
    cues = []
    for block in re.split(r"\n\s*\n", raw.strip()):
        if not block.strip():
            continue
        lines = block.strip().splitlines()
        if lines[0].strip().isdigit():
            lines = lines[1:]
        if not lines:
            continue
        match = ARROW.fullmatch(lines[0])
        if not match:
            raise ValueError(f"Invalid subtitle cue: {lines[0]}")
        start, end = seconds(match.group(*range(1,5))), seconds(match.group(*range(5,9)))
        if end <= start:
            raise ValueError("Subtitle end must follow start")
        if text := "\n".join(lines[1:]).strip():
            cues.append({"start_seconds":start,"end_seconds":end,"text":text,"source":"subtitle"})
    return sorted(cues,key=lambda c:(c["start_seconds"],c["end_seconds"]))
