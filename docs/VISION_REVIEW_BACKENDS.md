# Visual review backends: local, OpenAI API, and Codex/manual

## Purpose and limitations

Do **not** replace FFmpeg cut detection with a general vision-language
model. FFmpeg detects frame changes; a vision model describes *one selected
still image*. Still frames cannot establish motion, which person is who
between cuts, dialogue source, causality, or the true narrative.

Your shot ZIP contains one midpoint thumbnail for **each detected shot**
and separate actual sample frames under evidence/. Both are useful, but
this visual-review experiment compares provider descriptions for the
**same midpoint images** so shot IDs and timestamps cannot drift.

The model output is a hypothesis, not validated truth. Keep BLIP, Ollama,
OpenAI API and Codex/manual reviews separate so disagreements can be
audited against the actual image.

## A. Existing local Ollama mode

Continue using the existing local_trailer command with --context-model
gemma3:4b. It sees multiple sampled frames and ASR, but the current
context reasoning still has known hallucination/continuity weaknesses.
It writes separate PRIVATE context output.

## B. Paid OpenAI API shot visual review

This mode reads the already-generated shots ZIP. It sends **only shot JPEGs
and shot IDs/timestamps** to the OpenAI Responses API and stores reviewed
per-shot JSON. It does not upload the full video, audio or private ASR.
It uses Structured Outputs and strict ID checking. It does not run
unless you explicitly allow the upload using --confirm-upload.

Requirements: Python 3.11; internet; an API account with separate billing;
OPENAI_API_KEY set securely in your environment. Do not paste API keys into
GitHub, JSON files or conversations. The Python standard library suffices,
so no additional pip install is required for this specific command.

Start by inspecting the plan (no network request, no key or charge):

~~~powershell
python -m true_experience.openai_vision_review iron_man_shots.zip --dry-run --max-shots 12
~~~

When you have chosen to transmit those 12 frames to OpenAI:

~~~powershell
python -m true_experience.openai_vision_review iron_man_shots.zip --model gpt-5.4-mini --max-shots 12 --batch-size 4 --confirm-upload -o openai_shot_review.json
~~~

For a potentially stronger but more expensive vision model, replace
--model gpt-5.4-mini with --model gpt-5.4. The full 119-shot run requires
--max-shots 0. DO NOT run the full set until you are comfortable with the
quality and token charges of a 12-shot pilot. A run checkpoints after each
completed batch; repeat with --resume to avoid paying for completed batches.

API charges are separate from ChatGPT/Codex subscriptions. Frames are
sent to a remote service only after explicit confirmation. The tool
requests store=false, which does not negate all service-side retention
or data processing policies. Review the applicable terms before sending
copyrighted footage.

OpenAI API docs:
- https://developers.openai.com/api/docs/guides/images-vision
- https://developers.openai.com/api/docs/guides/structured-outputs
- https://developers.openai.com/api/docs/models/gpt-5.4-mini

## C. Codex/manual handoff with no API key

Generate a **portable folder** from exactly the same source ZIP:

~~~powershell
python -m true_experience.vision_handoff iron_man_shots.zip -o iron_man_codex_handoff
~~~

Or start with the first 12 shots for a controlled experiment:

~~~powershell
python -m true_experience.vision_handoff iron_man_shots.zip -o iron_man_codex_pilot --max-shots 12
~~~

The folder has:
- shots/: midpoint JPEGs and original shot IDs/times
- evidence/: exact v0001 etc. frame samples, if present
- manifest.json: exact shot and evidence mapping
- TASK.md and AGENTS.md: explicit visual-review instructions
- shot_review.schema.json: required output structure
- shot_review.template.json: output metadata, initially empty array
- README.md: human workflow

Open the folder in a Codex environment that supports JPEG inspection,
ask it to follow TASK.md, and have it produce shot_review.json.
This is NOT an automatic call to Codex or a promise that every Codex
environment can view images. Codex access and API billing are distinct.

The generator makes no network calls and never uploads the images. A
human chooses whether to give the folder to another model or service.

## Comparing results without false accuracy scores

Both backend outputs use the **same per-shot fields**:
shot_id, visible_description, visible_objects, visible_action,
uncertainties.

Review their validity using the original archive and matching --max-shots:

~~~powershell
python -m true_experience.vision_review_validate iron_man_shots.zip openai_shot_review.json --max-shots 12
python -m true_experience.vision_review_validate iron_man_shots.zip openai_shot_review.json --max-shots 12 --compare iron_man_codex_pilot/shot_review.json
~~~

The comparison reports differing wording and common shot IDs. Different
phrases do NOT indicate that one backend is better. To evaluate precision,
a human must check claims against the actual JPEGs and, for motion or
narrative, the actual video/audio. Do not rank models by fluency.

## Privacy

- Still-image ZIP and portable folder may contain copyrighted frames.
- No media is automatically uploaded to GitHub or to this conversation.
- The API mode is explicitly opt-in; the manual folder works entirely
  offline until the user chooses where to process it.
- Raw audio/ASR and private context outputs are NOT included in the new
  visual-review API request or handoff folder.
- Do not commit extracted images, keys or generated reviews to the repo.
