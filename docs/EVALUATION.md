# Evaluation gate for The True Experience

The regular CI suite tests deterministic audiovisual processing and **mocks** model inference. It cannot establish recognition quality.

## Step 1 — Structural evaluation (stdlib-only)

```bash
python -m true_experience your-video.mp4 -o analysis.json
python -m true_experience.evaluate analysis.json -o evaluation.json
```

This checks video duration, complete shot/timeline coverage, segment ordering, timestamps, sample-to-shot references, interval alignment and provenance. It does **not** check whether ASR text or captions are correct.

## Step 2 — Real pretrained model smoke test

In GitHub, open **Actions → Real model smoke (manual) → Run workflow** on `main` after this PR is merged. This opt-in action downloads open model weights and creates a synthetic, espeak-voiced 30-second video with three visual sections. It runs local faster-whisper tiny + BLIP base, then uploads JSON results. It uses CPU, consumes runner time and needs network access to Hugging Face. It deliberately does not run on every push or PR.

**Passing the smoke test means only that the pretrained models execute and produce structurally valid output.** A synthetic colored clip cannot establish understanding of a real movie.

## Step 3 — Human-annotated reference video

Use a 30–90 second owned/licensed clip with meaningful action, sound and dialogue. After running `--transcribe --caption-frames`, annotate its *actual* speech independently and write a reference `.srt` file. Compute word error rate:

```bash
python -m true_experience.evaluate analysis.json --reference-srt human-reference.srt -o evaluation.json
```

The reference SRT must not be generated from the ASR output; otherwise the metric is circular. WER ignores timing and gives a clip-level text metric; compare time boundaries manually/with independent annotations. Report the tested language and accent. For captions, manually rate each sampled image against its generated description for unsupported details, errors and missing coverage. Model confidence fields are intentionally null; don't substitute guesses. Record runtime, hardware, memory, hallucination counts and failures. Only then consider semantic audio models and emotion/narrative fusion.

See <https://github.com/JanneHonkonen/The-True-Experience/issues/4>.
