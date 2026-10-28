# Simple visual recognition: local Ollama or optional OpenAI API

## Purpose

Our first BLIP results on the 151-second Iron Man trailer described many
frames incorrectly or labeled them with unrelated movie titles. The
12-shot manual review has much more concrete visual detail, but it is
**not validated ground truth**, and its model field is unspecified.

FFmpeg still detects cuts. The independent vision reviewer reads the
timestamped shot JPEGs from the existing iron_man_shots.zip and saves
structured per-shot hypotheses using the SAME IDs and output schema.
One still cannot establish actual motion, cause, speaker identity or
narrative continuity.

No Codex environment, folder preparation or API key is needed for the
default local mode.

## Local Ollama: recommended

1. Install or update Ollama from https://ollama.com/download.
   Qwen3-VL requires Ollama **0.12.7 or newer**.
2. Pull a model once:

       ollama pull qwen3-vl:8b

   It is roughly a 6.2GB download. If RAM is limited, use
   qwen3-vl:4b (around 3.3GB); another option is gemma3:12b
   (around 8.2GB). Larger models might help precision but also increase
   memory and inference time. Actual film accuracy has NOT been measured.
3. In your activated Python venv from the repository directory:

       python -m true_experience.vision_review iron_man_shots.zip --max-shots 12 -o qwen_shot_review.json

   Default backend is local Ollama and model qwen3-vl:8b. No API key
   or OpenAI endpoint is used. Each still is processed in a separate
   model request to prevent accidentally assigning a caption to another
   shot. Output is checkpointed after each image.

   Alternatives:
       --model qwen3-vl:4b
       --model gemma3:12b
       --max-shots 0            (all shots)
       --resume                 (continue saved partial JSON)
       --dry-run                (no model calls)

The output has visible_description, visible_objects, visible_action and
uncertainties for every reviewed shot. The model is asked to describe
only what is visible, NOT to guess motion or character names.

### Existing audiovisual context mode

The local_trailer tool is a separate experimental pass with actual
sampled video frames, Whisper ASR, rolling memory and optional Ollama.
You can already change its model without changing Python:

       ollama pull qwen3-vl:8b
       python -m true_experience.local_trailer VIDEO.mp4 --context-model qwen3-vl:8b --reuse-analysis --context-output iron_man_context_qwen_private.json

Reuse requires the original PRIVATE analysis cache and matching video
and inference settings. Context analysis still incorporates BLIP's
original captions, so a better model alone does not validate its
narrative output. The new shot-review JSON is intentionally a separate
experiment and is NOT silently substituted for BLIP observations.

## Optional OpenAI API: billed, explicit upload consent

No API call is made in the default mode. For API vision reviews, first
preview the number of images and requests:

       python -m true_experience.vision_review iron_man_shots.zip --backend openai --dry-run

Set OPENAI_API_KEY securely in your environment, and use:

       python -m true_experience.vision_review iron_man_shots.zip --backend openai --max-shots 12 --confirm-upload -o openai_shot_review.json

The API sends selected JPEGs and their shot IDs/timestamps, not the
full video, speech transcript or private context file. Images are
transmitted to OpenAI's remote API; the service is paid separately from
ChatGPT. Use --max-shots 0 only after evaluating cost and quality on
the pilot. Check permissions before transmitting copyrighted frames.
The API mode requests store=false, which is not a guarantee of no data
processing or retention under all policies.

## Validate or compare both modes

       python -m true_experience.vision_review_validate iron_man_shots.zip qwen_shot_review.json --max-shots 12

Once a second 12-shot JSON exists:

       python -m true_experience.vision_review_validate iron_man_shots.zip qwen_shot_review.json --max-shots 12 --compare openai_shot_review.json

You can also compare to an earlier pilot review when its manifest hash
matches the same selected shots. Different wording is NOT an
accuracy or hallucination score. Each claim still needs a comparison
with the actual JPEG or video frames.

## Privacy and limitations

The local Ollama model receives the individual JPEGs over localhost
127.0.0.1, and no API key is required. The optional OpenAI API sends
images only after --confirm-upload. Neither mode automatically sends
footage to GitHub or commits reviews. Avoid sharing source media or
dialogue without the appropriate permissions. The JSON files themselves
may contain descriptions of copyrighted imagery.

Qwen3-VL model reference: https://ollama.com/library/qwen3-vl
Gemma 3 model reference: https://ollama.com/library/gemma3
