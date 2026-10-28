# The True Experience — runnable MVP v0.1

This is an evidence-first measurement pipeline, **not** yet an AI model that understands film emotions or narratives.

## Requirements

Python 3.10+, FFmpeg (both ffmpeg and ffprobe installed on PATH). The MVP needs no third-party Python packages, API keys, cloud services, or GPU.

## Run

```bash
python -m true_experience path/to/video.mp4 -o analysis_output.json
python -m true_experience path/to/video.mp4 --subtitles path/to/subtitles.srt --interval 5 -o analysis_output.json
```

Options: `--scene-threshold 0.3` (lower values find more visual cuts), `--silence-db -35`, `--silence-min-duration 0.25`. All media remains local.

## Output

`schema_version`, `metadata`, `shots`, `silence_intervals`, and `timeline`. The timeline has fixed intervals with a shorter final interval, overlapping shot IDs and subtitle cues, cut timestamps, and audio silence state.

**Measured:** video metadata, FFmpeg visual cuts (not semantic scenes), FFmpeg audio silence below a threshold (not identification of speech or music), and optionally user-provided subtitles (not ASR). **Not yet measured:** visual semantics, character identity, emotions, speaker identity, or narrative meaning. Those fields are explicitly `null`, not fabricated.

## Tests

```bash
python -m unittest discover -s tests -v
```

Integration tests generate synthetic videos with FFmpeg.

## Next milestones

1. Benchmark shot detection against manually labeled clips and add semantic scene grouping.
2. Add optional speech recognition with timing checks and provenance.
3. Add vision-language captioning with model uncertainty and frame sampling.
4. Add audio event/music models and confidence-aware multimodal fusion.
5. Evaluate emotional and narrative interpretations against multiple human annotators, never assuming any single emotion is objectively correct.

This MVP is separate from experimental PR #1; do not merge both approaches without reconciling their APIs and tests.

## Licensing

See the author's non-commercial terms in the repository README. Add a complete license file and review external model licenses before distributing more broadly.
