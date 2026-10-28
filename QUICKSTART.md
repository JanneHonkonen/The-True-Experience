# The True Experience — evidence-first audiovisual MVP v0.3

Runs on local video with FFmpeg, adding **optional local AI inference**. Captions and transcripts are model hypotheses, not verified facts. Emotion and narrative interpretation remain unimplemented.

## Installation: virtual environment (recommended)

Use **Python 3.11**, FFmpeg and FFprobe on `PATH`. From the repository root:

### Linux / macOS

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### Windows PowerShell

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Run the commands **inside the project directory** (the one containing `requirements.txt` and `true_experience/`). `python -m ...` runs from the source checkout; an editable package installation is not necessary.

If your old command `python -m pip install -e '.[all]'` failed with *not a valid editable requirement*, stop using that command and install from `requirements.txt` as shown above. This is a direct dependency list, not an editable install. `-e` is intended for installing a local Python package in editable mode, not for pointing pip at a requirements file.

**Note:** installing PyTorch can take substantial disk space. On **CPU-only Linux**, you may first install the CPU build of PyTorch with `python -m pip install torch --index-url https://download.pytorch.org/whl/cpu`, then run `python -m pip install -r requirements.txt` to install the remaining dependencies. First model inference also downloads model weights. The video is processed locally; no API key is required.

### Fix for Windows: `metadata_errors` when decoding video

If transcription fails with `open() got an unexpected keyword argument 'metadata_errors'`, an incompatible PyAV 19.x version was installed alongside faster-whisper 1.2.1. The new requirements file explicitly pins `av>=11,<19`. In your **activated venv**, run from the repository root:

```powershell
git pull
python -m pip install -r requirements.txt
python -m pip show av faster-whisper
```

If pip still reports PyAV 19, explicitly reinstall a compatible version:

```powershell
python -m pip install --force-reinstall "av>=11,<19"
```

The earlier message about Hugging Face Xet Storage simply indicates a slower download path; it is unrelated to the decoding error. Your downloaded model weights should remain cached. This workaround can be removed when a faster-whisper release supporting PyAV 19 becomes available.

## Video processing without AI models

You can skip Python dependency installation entirely for basic FFmpeg-only analysis:

```bash
python -m true_experience video.mp4 -o analysis_output.json
python -m true_experience video.mp4 --subtitles subtitles.srt --interval 5 -o analysis_output.json
```

## Local speech recognition and visual captions

```bash
python -m true_experience video.mp4 --transcribe --caption-frames --max-caption-frames 120 --caption-spacing 10 -o analysis.json
```

For Finnish transcription, try `--whisper-model small --language fi`. These models can make mistakes, especially with music and stylized dialogue.

## Iron Man trailer local-file evaluation

With an authorized copy already available on your computer:

```bash
python -m true_experience.local_trailer "/path/to/iron-man-trailer.mp4" -o iron_man_review.json
```

Upload the generated `iron_man_review.json` for discussion, not the source movie. See [Iron Man trailer evaluation](docs/IRON_MAN_TRAILER_EVALUATION.md).

## Experimental rolling audiovisual context

The baseline report only describes individual frames. For optional local
reasoning across time windows, install [Ollama](https://ollama.com/download),
start its local server, and pull the vision model:

```powershell
ollama pull gemma3:4b
```

Then in your activated venv run from the repository root:

```powershell
python -m true_experience.local_trailer "C:\winstuff\ai\iron_man_trailer.mp4" --whisper-model small --max-caption-frames 60 --caption-spacing 4 --context-model gemma3:4b --context-window 12 --max-context-images 3 -o iron_man_review.json --context-output iron_man_context_private.json
```

This adds **model-generated interpretations** grounded to named visual/audio
inputs, plus tentative links to two previous time windows. The context file
is private and can include paraphrases of speech, so inspect it before sharing.
The ordinary `iron_man_review.json` remains sanitized and does not include
the context text. The feature is not a film-understanding accuracy benchmark.

The local model now requests a strict JSON schema from Ollama, rather than
only requesting valid JSON. If the model nevertheless returns invalid data
twice, the affected window is marked unverified and skipped rather than
fabricating a story or aborting the whole film.

The command saves a PRIVATE `iron_man_analysis_private.json` with the raw
transcript and caption evidence. To retry failed context without running
Whisper/BLIP again, run the same command with `--reuse-analysis`; add
`--resume-context` to reuse already completed context windows. These
cache files are excluded from Git but are not encrypted.

The local model may take substantial RAM and compute time, especially on CPU.
The existing venv requirements remain sufficient; Ollama is a separate app.
See [context documentation](docs/CONTEXT_REASONING.md) for model constraints,
limitations, and how to evaluate unsupported character/causal links.

## Output fields and limitations

- `shots`: FFmpeg shot-cut candidates, **not** semantic scenes.
- `silence_intervals`: thresholded silence, not sound-event classification.
- `speech_segments`: optional ASR text with timestamps and model provenance, but no reliable speaker identity.
- `visual_observations`: optional captions from individual frames, each with sample time and shot ID.
- `timeline`: interval-aligned subtitles, ASR and visual samples; no extrapolation to unsampled intervals.
- `metadata.perception.visual_sampling`: sampled and unsampled shots, unsampled duration, and largest gap between sample positions.

Captions and transcription can hallucinate or miss details. Emotion, narrative and character identity are not inferred. The caption sampler favors coverage of long shots, subject to `--max-caption-frames` and `--caption-spacing`.

## Tests

```bash
python -m unittest discover -s tests -v
```

CI tests pipeline components and mocks inference backends; actual pretrained-model evaluation is a separate workflow. After testing, run `deactivate` to exit the virtual environment.

## License

See the author's non-commercial terms in README. Review model and dataset licenses before redistributing the combined system.
