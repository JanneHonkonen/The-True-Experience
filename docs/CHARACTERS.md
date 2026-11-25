# Characters: who is who, and what role they play

Two flags add characters to the timeline:

```powershell
python -m true_experience "C:\Videos\clip.mp4" --characters -o clip_feeling.json
python -m true_experience "C:\Videos\clip.mp4" --voice --visual-mood-model qwen3.5:9b --character-roles qwen3.5:9b -o clip_feeling.json
```

`--characters` finds and follows people. `--character-roles MODEL` also asks
the local vision model what kind of character each one is, and implies
`--characters`.

## How people are followed

1. **Sampling.** About one frame per second (`--character-fps`), decoded by
   FFmpeg in one pass at 960 px width.
2. **Detection** on the CPU with OpenCV and four small models from the OpenCV
   model zoo, downloaded once (about 180 MB) to `~/.cache/true_experience/models`
   or to the folder in `TRUE_EXPERIENCE_MODELS`:
   - person boxes: YOLOX-S (Apache 2.0)
   - faces: YuNet (MIT)
   - face identity: SFace, 128 numbers per face (Apache 2.0)
   - clothing and body appearance: Youtu ReID, 768 numbers per person (Apache 2.0)
   Small background figures without a face (under 15% of the frame height)
   are ignored.
3. **Tracking.** Inside a shot, each person is followed from one sample to
   the next by position and appearance. One followed person in one shot is an
   *appearance* (`T1`, `T2` ...). A cut always starts new appearances.
4. **Linking across the film.**
   - **Face identity `F1, F2 ...`**: appearances with a usable face (at least
     36 px) are grouped by face similarity. A new haircut or new clothes do
     not matter here.
   - **Outfit identity `O1, O2 ...`**: all appearances are grouped by clothing
     appearance (the ReID model plus named colours).
   - **Character `C1, C2 ...`**: every face identity becomes a character.
     An appearance without a usable face (back to camera, helmet, mask,
     armour suit, too far away) joins the character that owns most of the
     face-visible appearances in the same outfit group. If an outfit group has
     no visible faces at all, it becomes a character of its own, marked
     `identified_by: "clothing"`. That is how a suited or masked figure who
     never shows a face still gets one ID through the whole video.
   - Two people seen in the same frame are never merged into one character.
5. **Voice.** With `--voice`, a voice group (`S1` ...) is tied to a character
   when that character is the only face on screen during most of that voice's
   lines. Lines then carry `character`.
6. **Prominence.** Screen time (50%), dialogue time (30%) and on-screen size
   (20%), relative to the most prominent character.
   `likely_main_character` is true for rank 1 and anyone within 80% of it.
7. **Background.** A figure seen in one shot only, for at most 2 s, with no
   lines and not in the top 3 is marked `background: true` (crowds, extras,
   quick cutaways). They keep their IDs so they can still be followed.

## Role interpretation (`--character-roles`)

For up to 8 characters with at least 3 s of screen time, the model gets:
- a face close-up, one crop per outfit (up to two) and one full frame for setting,
- their screen-time share, prominence rank and outfits,
- up to 10 of their lines and the visual mood descriptions of their scenes.

It answers under a fixed schema:

| Field | Values |
|---|---|
| `narrative_role` | protagonist, antagonist, supporting, minor, unclear |
| `archetypes` | up to 3 of: hero, villain, criminal, terrorist, victim, mentor, love_interest, sidekick, authority_figure, henchman, comic_relief, civilian, journalist, business_person, soldier, law_enforcement, scientist, unclear |
| `occupation` | short free text such as "business executive" or "news reporter", or "unknown" |
| `occupation_cues` | the visible or spoken cues behind it |
| `appearance`, `evidence` | one sentence each |
| `physical` | apparent look: `apparent_gender` (man, woman, boy, girl, unclear), `apparent_age_range` (child, teen, 20s ... 60s, 70+, unclear), `build`, `hair`, `facial_hair`, up to 3 `distinctive_features` |
| `confidence` | 0..1 |

The prompt tells the model to judge only from costume, uniform, props,
setting, actions, framing and dialogue, never from ethnicity, skin colour,
religion, gender or age, and never to name actors or real people. "unclear"
is always allowed. The physical description is only what the images show; it
leaves out ethnicity, skin colour and attractiveness, which cannot be read
reliably from a face and add nothing to following the story. A part the model
leaves out or gets wrong becomes "unclear" instead of failing the role.

## Names and profiles

Names come only from the film: what characters say and what is written on
screen. Nothing is matched against actors or any database.

- **Transcript.** A line already tied to a character that says "I'm Tony
  Stark", "my name is ..." or "call me ..." is a self-introduction.
- **Scene readings** (with `--story`). For each scene the model lists the
  names heard in the dialogue or shown on screen and whom each belongs to:
  the speaker (`self_introduction`), the person spoken to (`addressed`), a
  person in the scene the line is about (`mentioned_present`), someone not in
  the scene (`mentioned_absent`), or a caption (`on_screen_text`). A name that
  does not occur in the scene's dialogue is dropped unless it was on screen,
  so the model cannot fill in names it knows from elsewhere.

Forms of one name ("Tony Stark", "Mr. Stark", "Tony", "Stark") are merged
under the fullest form. Votes are weighted (self-introduction 3, caption 2,
addressed 2, mentioned while present 1); a character is named when the best
name reaches 2, and a name goes to one character only. Names that only ever
refer to absent people go to `story.names_not_on_screen`.

`characters[].name` is `{name, also_called, confidence, evidence}`, where
`evidence` lists the lines or scenes behind it. `characters[].profile` is one
sentence built from the name, the physical description, the clothing and the
role, for example:

> Man named Tony Stark ("Mr. Stark"); appears to be in his 40s; slim,
> athletic; short dark hair; Van Dyke beard; dark suit and tinted glasses;
> protagonist (hero, business person).

Without `--character-roles` the profile has only the name; without a name it
starts with "Man", "Woman" or "Person".

`main_character` combines both: the most prominent character (unless the model
calls them the antagonist), or a character the model calls the protagonist
with prominence of at least 0.5.

Runs after visual mood, with the same model, so nothing extra is loaded.
About one model call per character.

## Output

- `characters[]`: `id`, `identified_by` (face or clothing), `face_ids`,
  `outfit_ids`, `outfits` (named colours, screen seconds, a sample time and
  box), `screen_seconds`, `face_visible_seconds`, `shots`, `first_seconds`,
  `last_seconds`, `voice_groups`, `dialogue_seconds`, `prominence`,
  `prominence_rank`, `likely_main_character`, `main_character`, `name`,
  `profile` and, with `--character-roles`, `role`.
- `character_tracks[]`: every appearance with its character, face ID, outfit
  ID, how it was linked (`face` or `clothing`), times and shot.
- `timeline[].characters_present`: who is on screen in each interval, for
  how long, and whether their face was visible.
- `speech_segments[].character` with `character_source`
  (`voice_group` or `only_face_on_screen`).

Boxes are in pixels of the 960 px wide sampled frames.

## Privacy

People are grouped within one video only. Nobody is matched against any
database or looked up online; names come only from the film's own dialogue
and on-screen text. Face and body embeddings stay in memory and
are never written to the output. Run this only on video you have the right to
process.

## Known limits

- Profile faces and very dark or blurred faces often give no usable face
  match. Such appearances fall back to clothing.
- Identical uniforms or suits (soldiers, a team in matching kit) can pull
  different people into one outfit group. Faces still keep them apart when
  they are visible.
- One person in a quick costume change with no face visible becomes a
  separate clothing character.
- Similarity thresholds (`FACE_THRESHOLD` 0.40, `OUTFIT_THRESHOLD` 0.72) are
  first guesses to be tuned on real films.
- CPU time is about 0.15-0.25 s per sampled frame, roughly 10-25 minutes for a
  2-hour film at 1 frame per second. Use `--character-fps 0.5` to halve it.
