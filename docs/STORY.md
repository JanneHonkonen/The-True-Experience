# Scenes and story (stage 9)

```powershell
python -m true_experience "C:\Videos\clip.mp4" --voice --visual-mood-model qwen3.5:9b --character-roles qwen3.5:9b --story qwen3.5:9b -o clip_feeling.json
```

`--scenes` groups shots into scenes on the CPU. `--story MODEL` also reads
each scene and the whole arc with the local vision model, and implies
`--scenes`. Run it together with `--characters` and the feeling stages: the
readings use everything they measured.

## Scenes (`--scenes`)

Each cut is scored for how well the incoming shot continues the scene so far,
by comparing it with the scene's last 3 shots, so shot / reverse-shot dialogue
stays one scene:

| Signal | Weight |
|---|---|
| look: colour histogram + 3 x 3 colour layout (4 fps, 160 px frames) | 0.55 |
| people: overlap of non-background characters (needs `--characters`) | 0.30 |
| a line of dialogue running across the cut | +0.25 |
| a silent gap at the cut | -0.15 |

A score under 0.45 starts a new scene. Scenes shorter than 4 s join the
neighbour they continue best, so a fast montage doesn't become dozens of
one-shot scenes. Each shot records its `continuity_with_scene` scores.

`scenes[]`: `scene_id` (SC1 ...), times, `shot_ids`, `characters` (by time
on screen), number of `lines`, `speakers`, and `feeling` (mean valence,
arousal, tension, peak tension, and the emotion labels in order).
Timeline intervals get `scene_ids`.

## Story (`--story MODEL`)

1. **Scene reading.** Each scene sends up to 6 frames (about one per second)
   to the model, plus measured facts: the characters with their roles and
   appearance, the dialogue labelled with who speaks, the scene's feeling and
   the music mode. `scenes[].reading` holds:
   - `summary`, `location`, `time_of_day`
   - `scene_type` (dialogue, action, fight, chase, montage, establishing,
     romance, suspense, comedy, transition, title_card, other)
   - `story_beat` (setup, inciting_incident, rising_action, confrontation,
     crisis, climax, falling_action, resolution, unclear)
   - `emotional_turn`, `character_actions`, `confidence`
   - `name_mentions`: names heard or shown in the scene, whom each belongs
     to and how (see [names](CHARACTERS.md#names-and-profiles))
2. **Names.** The name mentions of all scenes are merged into a `name` per
   character before the arc, so the arc can use them.
3. **Arc.** One text-only call over all scene readings, the cast and the
   measured turning points. `story` holds:
   - `logline`, `genres`
   - `form` (full_story, trailer_or_montage, excerpt, unclear)
   - `protagonist`, `antagonist`
   - `turning_points` (scene and what changes), `emotional_journey`
   - `character_links`: hypotheses that different IDs are the same person,
     for example a suited figure and the character who builds the suit, each
     with a confidence
   - `confidence`
   - `names_not_on_screen`: names that only ever refer to absent people

The model may use character names only if they are spoken in the dialogue,
and never names actors or real people. Unknown scene or character IDs in its
answers are dropped. All of it is marked `kind: "interpretation"`; the
measured data is never changed. A character link is a hypothesis, and the
character IDs stay separate.

Time: about 4-6 s per scene on the RTX 5080, so roughly 10-15 minutes for a
2-hour film with 150 scenes. The model is unloaded at the end.

## Narration (with `--story`, `--no-narration` to skip)

After the names and the arc, one more call per scene turns the measurements
into what a viewer would tell a friend: 1 to 4 timed passages per scene in
the present tense that describe what is seen and heard and how it feels.
Each call gets the scene's frames, the cast as names or plain descriptions,
the dialogue with times, speaker and delivery (calm, sharp, fast, flat ...
from the voice stage), the music in words (energy, tempo, major or minor,
loudness against the film's own level), the feeling at the start and end,
the visual mood words, the scene reading, the logline and the last passage
before the scene so the text flows.

Checks on every answer:
- every quoted line must occur in the transcript of that scene (pieces joined
  with "..." are checked one by one), so no invented dialogue;
- no IDs such as C1 or S2 in the text;
- each passage's time is kept inside its scene.

A failing answer is sent back once with the reason; if it fails again, the
scene's summary stands in (`fallback: true`). Passages are stored in
`scenes[].narration` and, in time order, in `narration`; the CLI writes them
to `<output>_narration.md`. Time: about one scene reading again (4-6 s per
scene on the RTX 5080).

## Known limits

- Trailers are montages, so scene grouping and the arc describe the cut, not
  the film. The model is asked to say so (`form: trailer_or_montage`).
- Look similarity can join two places that share a colour palette. The
  people and speech signals reduce this but don't remove it.
- With many scenes the arc prompt is compacted to about 30 words per scene.
