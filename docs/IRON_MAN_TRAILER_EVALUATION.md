# Real trailer evaluation: Iron Man (2008), Trailer #1

Source: https://www.youtube.com/watch?v=8ugaeA-nMTc

## Access check (2026-10-28)

The repository attempted a one-time GitHub Actions inference trial:
https://github.com/JanneHonkonen/The-True-Experience/actions/runs/40728085455

The ordinary yt-dlp request reached YouTube but the platform returned
"Sign in to confirm you're not a bot." Therefore **no actual Iron Man
video was downloaded and no Whisper or BLIP inference occurred** during
this trial. The workflow was removed from the PR to prevent misleading
reruns or unnecessary use of runners. It did not fail because of the
film-analysis pipeline.

We will not request account cookies or circumvent authentication,
DRM, or restrictions. Respect media rights and platform conditions.

## Real evaluation with a local file

Obtain a local copy that you are authorized to process using an
appropriate lawful source, or use a different owned/licensed video.
Install FFmpeg, FFprobe and Python 3.11. Clone the repository, create a
virtual environment **in the repository root**, then install the direct
dependencies listed in `requirements.txt` (no editable `-e` install).

Linux / macOS:

```bash
git clone https://github.com/JanneHonkonen/The-True-Experience.git
cd The-True-Experience
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m true_experience.local_trailer "/path/to/trailer.mp4" \
  --language en --whisper-model tiny --caption-spacing 6 \
  --max-caption-frames 36 -o iron_man_review.json
```

Windows PowerShell, from the cloned project folder:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m true_experience.local_trailer "C:\Videos\trailer.mp4" -o iron_man_review.json
```

To leave the environment, run `deactivate`. The full installation includes
large AI dependencies and first inference downloads model weights. Running
`python -m ...` from the checkout requires **no** editable package install.

This command runs **real local AI inference** against the supplied
file but outputs only a **sanitized review**: media metadata, shot counts,
speech segment and word counts, frame captions, timestamped YouTube
review links, and sampling-gap stats. It does **not** include media,
sampled frames, or full ASR dialogue in the output. No upload is made.

The user can share iron_man_review.json for quantitative and
timestamp-linked review. Do not share footage unless you are
authorized to do so. The source link is an identification aid, not
proof of the local file's provenance: users must verify that the
video actually matches the specified YouTube source.

### Troubleshooting Windows decoding

If Whisper downloads its weights but fails with `open() got an unexpected keyword argument 'metadata_errors'`, reinstall dependencies **inside the activated venv**:

```powershell
git pull
python -m pip install -r requirements.txt
python -m pip show av faster-whisper
```

This project now pins PyAV below major version 19 because faster-whisper 1.2.1 calls a PyAV argument removed in version 19. If the current installation still contains PyAV 19, run `python -m pip install --force-reinstall "av>=11,<19"` and retry the local trailer command. The Hugging Face Xet package warning is unrelated and optional.

## Chaining audiovisual evidence across time

For a second pass that uses images **and** dialogue together, see
[the optional rolling-context guide](CONTEXT_REASONING.md). This uses
a locally installed Ollama vision model with previous-window memory.
It creates a separate private `iron_man_context_private.json` and does
not change the sanitized `iron_man_review.json`. Never confuse
model-proposed continuity with verified scene or character identity.

## Honest interpretation

- A successful model run only proves that perception works end-to-end.
- Shot boundaries are not narrative scene boundaries.
- BLIP outputs frame caption hypotheses; hallucinations are possible.
- WER remains unknown without an independently annotated transcript.
- Character recognition, speaker identification, actions between
  sampled frames, and emotion/narrative scoring remain unvalidated.
- Timestamp-linked captions allow human evaluation against the video.

Further evaluation work is tracked in:
https://github.com/JanneHonkonen/The-True-Experience/issues/4
