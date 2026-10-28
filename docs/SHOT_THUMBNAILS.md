# Timestamped shot and visual-evidence archive

This is a human-auditing feature, not a model-accuracy improvement. A
model's cited vXXXX evidence ID is only useful if we can inspect the
**actual image** sampled at that time.

## Interactive option

After the trailer video has been processed, the command asks:

    Create shot/evidence thumbnail ZIP at iron_man_review_shots.zip? [y/N]:

Yes creates an archive before Ollama reasoning runs. Enter y or yes to
confirm; Enter, n or no skips it. There is no question in a noninteractive
process (CI, pipelines, redirected input), and **no ZIP by default**.

Override the prompt with --thumbnails or --no-thumbnails. Set a different
output using --thumbnails-output iron_man_shots.zip.

## Reuse an existing run (recommended for your Iron Man trailer)

In the activated Windows venv, from the repository root, run:

~~~powershell
git pull
python -m true_experience.shot_archive "C:\winstuff\ai\iron_man_trailer.mp4" --analysis iron_man_analysis_private.json --context iron_man_context_private.json -o iron_man_shots.zip
~~~

This loads the existing private analysis and context files and decodes
the JPEG frames directly from the local video using FFmpeg and Pillow.
It does **not** call Whisper, BLIP, or Gemma. The analysis cache must
match the source video's path, size and modification time. If your
context JSON is absent, omit the --context option.

You can pass a raw analysis JSON instead of the private cache, but raw
JSON lacks the file identity check used by the private cache.

## ZIP contents

- shots/: one actual JPEG at the midpoint of **every** detected shot,
  including shots missed by the caption model.
- evidence/: actual sampled JPEGs corresponding exactly to the visual
  observations v0001, v0002, etc. These are NOT necessarily the same
  timestamp as the shot-midpoint files.
- manifest.json: machine-readable shot ID, exact start/end timestamps,
  image filename, frame sample time, evidence ID, caption hypothesis and
  context-window IDs where available.
- README.txt: basic instructions.

Example image filenames:

    shots/shot_0001_00-00-00.000000_to_00-00-00.667333.jpg
    evidence/v0001_shot_0001_at_00-00-00.333666.jpg

Hyphens replace time colons because Windows forbids colons in filenames.
Image filenames display six decimal digits of seconds. The manifest
retains the original JSON timestamp values as numbers; use this mapping
for exact correspondence and automated checking.

In the interactive command, context-window links are planned links;
the final Ollama call may or may not succeed. In standalone mode with
--context, the links come from **actually cited** window IDs in the
private context JSON. The manifest says which kind of link is present.

## Human comparison workflow

1. Find a questionable event in iron_man_context_private.json and note
   the evidence ID, such as v0010.
2. Open evidence/v0010_...jpg in the ZIP. Confirm what is actually
   visible; do not rely only on the BLIP caption.
3. Open the matching shot_NNNN file and nearby shot images to inspect
   chronology. Review the full video for motion and causal relationships
   between stills.
4. Classify each claim as supported, unsupported, ambiguous or
   not observable; note whether the model introduced a different
   character, event, or movie title without evidence.

## Privacy and sharing

The ZIP includes *copyrighted frames* when you analyze a commercial
trailer. It has NO source video or audio and NO raw ASR transcript.
However, the photographs themselves are copyrighted material, so
review permissions before sharing or uploading them. Nothing is
automatically sent to GitHub or ChatGPT. The ZIP is ignored by Git
but that does not encrypt it or restrict access.

Requirements: FFmpeg and Pillow (in requirements.txt). Model
dependencies are not used by the standalone archive command.
