# The True Experience — evidence-first audiovisual MVP v0.2

Runs on local video with FFmpeg, adding **optional local AI inference**. Captions and transcripts are model hypotheses, not verified facts. Emotion and narrative interpretation remain unimplemented.

## Requirements

Python 3.10+, FFmpeg and FFprobe on PATH. No API keys are needed. Baseline processing has no third-party Python dependencies.

## Run without models

```bash
python -m true_experience video.mp4 -o analysis_output.json
python -m true_experience video.mp4 --subtitles subtitles.srt --interval 5 -o analysis_output.json
```

## Real local speech recognition and frame captioning

```bash
python -m pip install -e '.[all]'
python -m true_experience video.mp4 --transcribe --caption-frames --max-caption-frames 120 -o analysis.json
```

Install speech or vision separately:

```bash
python -m pip install -e '.[speech]'
python -m true_experience video.mp4 --transcribe --whisper-model small --language fi -o speech.json

python -m pip install -e '.[vision]'
python -m true_experience video.mp4 --caption-frames --max-caption-frames 40 -o vision.json
```

First model use downloads weights (internet connection required); video and audio stay local. Image captioning runs on CPU. Faster-whisper automatically selects device. Model quality and runtime vary, and Finnish speech may need a larger model than `tiny`. Transformers v4 is required by the BLIP image-to-text pipeline, removed in v5.

## Output fields and limitations

- `shots`: FFmpeg cut candidates **not** semantically grouped scenes.
- `silence_intervals`: thresholded silence, not audio event classification.
- `speech_segments`: optional ASR transcripts with start/end, source, model and unknown speaker/confidence.
- `visual_observations`: optional captions of individual sampled frames, each with precise sample time and shot ID.
- `timeline`: interval-aligned SRT `dialogue`, ASR `speech`, sampled `visual_observations`; no extrapolation to unsampled intervals.
- `metadata.perception`: whether optional inference was actually enabled, chosen models and sample count.

SRT and ASR are separate evidence streams; they may disagree. Model-generated captions and transcripts can hallucinate. Emotion, narrative meaning, and character identities stay `null`, and a frame caption never claims to describe an entire scene.

Image captioning is capped by `--max-caption-frames` (default 120) and distributes samples over detected shots. CPU captioning and Whisper model inference may be slow for long films.

## Tests

```bash
python -m unittest discover -s tests -v
```

CI uses synthetic videos and mocks inference backends; it does **not** download or benchmark pretrained model weights. A separate real-model evaluation is required before claiming real-world accuracy.

## Next milestones

1. Evaluate Finnish/English ASR timestamps against human annotations.
2. Evaluate image captions and identify hallucinations.
3. Add real sound/music event recognition and timestamped fusion.
4. Experiment with scene continuity and narrative reasoning, keeping claims traceable to evidence.

## License

See the author's non-commercial terms in README. Review all model and dataset licenses before distributing the combined system.
