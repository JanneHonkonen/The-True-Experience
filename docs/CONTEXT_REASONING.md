# Rolling audiovisual context — local experimental feature

The True Experience v0.3 originally detects shots, transcribes speech, and
generates isolated BLIP frame captions. A caption is not a scene. The first
151.37-second Iron Man trailer evaluation had **119 shot segments, 36 sampled
frames, 83 unsampled shots, and 110.57 seconds within unsampled shots**.
The captions also included movie-title hallucinations. These measurements
motivate the next stage; they **do not** tell us what happens in unseen shots.

## Design

1. Extract FFmpeg shot boundaries, Whisper ASR and BLIP sample captions.
2. Make fixed **context windows** (default 12 s), *not semantic scenes*.
3. For each window, pass up to three **actual decoded video frames** to a
   locally running vision-language model, alongside *fallible* caption
   hypotheses, *fallible* transcript snippets and their timestamps.
4. Pass a **bounded memory of the previous two windows**, labeled explicitly
   as unverified hypotheses, to propose continuity. **Previous summaries are
   not used as evidence to support new event claims.**
5. Require model events to reference current-window evidence IDs
   (e.g. v0003 for an actual sampled frame or a0007 for an ASR segment).
   Invalid IDs, malformed structured JSON and fake prior-window links fail.
6. Keep results separate from public summary:
   - iron_man_review.json: existing sanitized counts, captions and links
   - iron_man_context_private.json: sequential model hypotheses, citations,
     continuity suggestions, unanswered questions and explicit uncertainty,
     **plus a local evidence_index with timestamped ASR excerpts for auditing**
     (not included in the public report)
7. Checkpoint after each completed model window. Use `--resume-context` with
   the same model/settings and `--reuse-analysis` to skip prior completed
   inference. Resuming checks the cached input's fingerprint.
8. Ollama requests enforce a required-field JSON schema. If a model still
   produces malformed or unsupported claims twice, the window explicitly
   abstains rather than ending the entire analysis or inventing a summary.

**Important:** These windows are not semantic scenes. Visual samples are
instants, not verified motion; timestamps may not capture crucial actions.
No character/entity identity, causation, emotion, or narrative understanding
is *verified* by this feature. Citation IDs prove traceability to inputs, **not**
truth. Model output may hallucinate and may paraphrase dialogue, so the
private context JSON should NOT be published without reading it first.

## Setup on Windows (from activated Python venv)

Requirements from the repository root:

    git pull
    python -m pip install -r requirements.txt

Install a local Ollama server separately: https://ollama.com/download
Ollama must run on this PC at http://127.0.0.1:11434 (standard local API).
Pull a **vision-capable** model rather than a text-only variant:

    ollama pull gemma3:4b

Gemma 3 4B is a multi-image model; its download is several GB and
inference on CPU can be very slow. It needs Ollama 0.6 or later. No
model-server connection to external computers is supported by the code.

Run an initial test (moderate frame cap):

    python -m true_experience.local_trailer "C:\winstuff\ai\iron_man_trailer.mp4" --whisper-model small --max-caption-frames 60 --caption-spacing 4 --context-model gemma3:4b --context-window 12 --max-context-images 3 -o iron_man_review.json --context-output iron_man_context_private.json

To keep an expensive Whisper/BLIP run after a context failure, the program
now saves **iron_man_analysis_private.json** automatically when `--context-model`
is enabled. This includes raw ASR text and is excluded from Git. To reuse it,
rerun the same command with `--reuse-analysis`; to also keep previous
successful context windows add `--resume-context`. These options refuse
mismatched model settings or changes to the source video.

For denser visual evidence, increase --max-caption-frames to 120 and reduce
--caption-spacing if your hardware can handle it. Larger Whisper models
can improve ASR but are not guaranteed accurate. To check flags:

    python -m true_experience.local_trailer --help

The default mode (no --context-model) does **not** require Ollama or
generate any private context file. There is no added Python dependency
for the new logic; the vision model runs in the separately installed
Ollama app and the existing Pillow image decoder is reused.

## Output / evaluation

Private JSON windows contain:
- window_id/start/end, shot IDs, sampled timestamps and ASR segment count
- summary_hypothesis and event claims with current evidence IDs
- continuation hypothesis and optional previous window links
- open questions, unverified flags and model provenance
- human_groundedness_review = not_reviewed

A practical next evaluation is to annotate 20-30 of the frames and compare
each event to the *actual* image and audio before determining whether chains
are correct. Check what happens during unobserved shots separately. Count:
factually unsupported claims, incorrect recurrence links, unsupported
causality, and unknown or missing events. Do not score solely on fluent text.

## Security and limitations

The local tool sends sampled frame pixels and ASR snippets to Ollama
at **127.0.0.1:11434 only**. It never uploads the video to GitHub.
Both the analysis cache (including full ASR) and the context output (including
ASR excerpts) are saved to disk locally
and excluded from Git by .gitignore, but that does **not** encrypt it or
protect it if you share it.
Keep Ollama bound to localhost and treat your transcript data as private.
No remote endpoint can be selected using the CLI.
