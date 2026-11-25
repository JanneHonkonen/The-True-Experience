# Listening: music and sound effects (`--listen`)

Tempo, key and loudness say little about what music feels like. A viewer
hears a heroic brass theme, a tense string ostinato, a rock track or a choir,
and notices when the music suddenly stops. Explosions, engines, gunfire and
crowds carry as much of the mood as the picture. This stage gives the rest of
the pipeline those sounds in words.

## Model

[CLAP](https://huggingface.co/laion/clap-htsat-unfused) (`laion/clap-htsat-unfused`,
Apache-2.0, about 600 MB, downloaded once by `transformers`). It places audio
and text in one space, so a piece of sound can be scored against text
descriptions such as "tense, suspenseful music" or "an explosion" without
training. It uses about 1 GB of VRAM and is freed before the vision model loads.
On the RTX 5080 it should take a few seconds for a trailer (to be confirmed on
a real run).

It listens to the Demucs **music stem** (drums, bass and other; the effects
end up in "other" too), so dialogue does not mask the score. Without
separation it listens to the full mix.

## What it adds

Per 5 s window, in `timeline[].feeling_signals.listening`:

| Field | What it holds |
|---|---|
| `music.present` | whether music plays (CLAP "music playing" against "no music", and the stem level above -45 dBFS) |
| `music.style` | up to 2 of: orchestral film score, epic trailer music, rock, heavy metal, electronic, hip hop, pop, jazz, solo piano, ambient drone, acoustic folk, classical strings |
| `music.instruments` | up to 3 of: electric guitar, drums, brass, strings, piano, choir, synthesizer, bass, percussion hits, acoustic guitar |
| `music.mood` | up to 2 of: heroic, tense, ominous, sad, playful, calm, aggressive, romantic, mysterious, exciting |
| `music.mood_valence`, `mood_arousal` | the mood scores mapped to valence and arousal |
| `sounds` | effects heard in each half window, with a time: explosion, gunfire, car engine, jet, helicopter, crowd cheering, crowd chatter, camera shutters, metal clanking, whoosh, impact, glass shattering, siren, footsteps, wind, machinery, rumble |
| `description` | a short line such as "heroic, exciting orchestral film score; brass, strings" |

A label is kept only when it scores well above an even share
(1.6 times for music labels, 2.5 times for sounds), with its score.

Over the whole film, `music_moments[]`: `music_starts`, `music_stops`,
`music_surges` and `music_drops` (8 dB or more between windows),
`music_builds` (three or more windows getting louder, 6 dB or more in total)
and `hit` (sudden loud onsets in the stem).

## Where it is used

- **Fusion:** when the music is heard, its mood leads the music reading (75%),
  and mode and tempo only colour it. Tense or ominous music adds to tension.
  Where CLAP hears no music, the music reading is left out instead of being
  guessed from the key of sound effects.
- **Narration:** each scene gets the soundtrack descriptions, the sound
  effects with times and the music moments. The prompt asks the model to let
  music and sound carry the mood, and to use only what was heard.
- **Experience file:** every scene lists `music`, `sounds` and `music_moments`.

## Known limits

- Zero-shot labels are matches against text, not trained classifiers. Close
  styles (orchestral score and epic trailer music) are easily swapped, and
  effects inside the music stem can be read as percussion.
- 2.5 s is the finest time step for sounds.
- Music style names are broad on purpose; the stage never names a song, band
  or composer.
