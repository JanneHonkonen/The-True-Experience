# Real trailer evaluation: Iron Man (2008), Trailer #1

Input: https://www.youtube.com/watch?v=8ugaeA-nMTc

This specific trailer includes action, speech, rapid cuts and music. It is
a more realistic test stimulus than the synthetic red/blue/green fixture.

The GitHub Actions workflow on the experiment branch attempts to retrieve
the exact linked video, process it temporarily, and delete the source.
The public artifact contains ONLY a sanitized JSON review report:
- shot boundaries and timeline counts
- speech segment counts and word counts (not the transcript)
- frame caption hypotheses with linked YouTube timestamps
- sample-gap and shot coverage measurements
- explicit unmeasured accuracy flags

No video, audio, sampled frames or full ASR text are published or committed.
Download availability is not guaranteed: YouTube can restrict automated
access. Do not bypass authentication, DRM, geographic restrictions, or
platform rules; only process media you are authorized to use.

Each visual caption is an unverified hypothesis from one frame. Review the
linked source moment to assess grounding. A green workflow demonstrates
functional inference and structural consistency, not story understanding.
No word-error-rate can be computed without an independent transcript.

Run settings: faster-whisper tiny (English), BLIP base,
maximum 36 frames, spacing target 6 seconds, 5-second timeline.
Video may have more than 36 shots, so sampling gaps are disclosed.
