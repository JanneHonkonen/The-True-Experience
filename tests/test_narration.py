"""Timed narration, the compact experience file and the written outputs."""
import json
import tempfile
import unittest
from pathlib import Path

from true_experience.compact import experience
from true_experience.narration import (delivery, describe_person, mmss, narrate, quote_is_real,
                                       render_text, validate)

LINES = ["Mr. Stark, you've been called the Da Vinci of our time. What do you say to that?",
         "Absolutely ridiculous. I don't paint."]


def fixture():
    characters = [
        {"id": "C1", "screen_seconds": 30, "main_character": True, "name": {"name": "Tony Stark",
         "also_called": ["Mr. Stark"]}, "profile": "Man named Tony Stark; protagonist.",
         "role": {"narrative_role": "protagonist", "archetypes": ["hero"], "occupation": "executive",
                  "appearance": "Black suit.", "physical": {"apparent_gender": "man", "apparent_age_range": "40s"}}},
        {"id": "C2", "screen_seconds": 6, "role": {"appearance": "Blonde woman in a dark blazer, light shirt."}},
        {"id": "C9", "screen_seconds": 1, "background": True}]
    speech = [{"start_seconds": 6.96, "end_seconds": 10.76, "text": LINES[0], "speaker": "S1",
               "voice": {"emotion": {"label": "ang"}, "words_per_second": 4.5, "pitch_variability_semitones": 2.3}},
              {"start_seconds": 10.76, "end_seconds": 12.76, "text": LINES[1], "speaker": "S2", "character": "C1",
               "voice": {"emotion": {"label": "neu"}, "words_per_second": 2.5, "pitch_variability_semitones": 0.3}},
              {"start_seconds": 20, "end_seconds": 29, "text": "la la", "speech_suspect": True}]
    scenes = [{"scene_id": "SC1", "start_seconds": 0.0, "end_seconds": 4.0, "shot_ids": [1], "characters": [],
               "feeling": {"start_label": "stillness", "end_label": "stillness"},
               "reading": {"summary": "A green rating card.", "scene_type": "title_card"}},
              {"scene_id": "SC2", "start_seconds": 4.0, "end_seconds": 18.0, "shot_ids": [2, 3],
               "characters": ["C1", "C2"], "feeling": {"start_label": "stillness", "end_label": "shock"},
               "reading": {"summary": "Press at a gala.", "location": "gala entrance"}}]
    timeline = [{"start_seconds": a, "end_seconds": a + 5, "feeling_signals": {"music": {
        "rms_dbfs": -20.0, "onset_rate_per_second": 2.5, "tempo_bpm": 131.8, "tempo_confidence": 0.5,
        "mode": "major", "mode_confidence": 0.6}}} for a in range(0, 20, 5)]
    return characters, speech, scenes, timeline


class NarrationTests(unittest.TestCase):
    def test_helpers(self):
        self.assertEqual(mmss(66.4), "01:06")
        c, *_ = fixture()
        self.assertEqual(describe_person(c[0]), "Tony Stark")
        self.assertEqual(describe_person(c[1]), "blonde woman in a dark blazer")
        self.assertEqual(describe_person({"role": {"appearance": "Dark suit.", "physical": {
            "apparent_gender": "man", "apparent_age_range": "40s"}}}), "man in his 40s, dark suit")
        self.assertEqual(delivery({"emotion": {"label": "neu"}, "pitch_variability_semitones": 0.3}), "flat, deadpan")
        self.assertEqual(delivery({"emotion": {"label": "ang"}, "words_per_second": 4.5}), "sharp, edgy, fast")

    def test_quotes_must_be_real(self):
        self.assertTrue(quote_is_real("Absolutely ridiculous. I don't paint", LINES))
        self.assertTrue(quote_is_real("you've been called the Da Vinci ... What do you say", LINES))
        self.assertFalse(quote_is_real("I never paint anything", LINES))

    def test_validate(self):
        ok = validate({"passages": [{"t": 30, "text": 'He shrugs: "Absolutely ridiculous."'},
                                    {"t": 1, "text": "Flashes everywhere."}]}, 4.0, 18.0, LINES)
        self.assertEqual([p["t"] for p in ok], [4.0, 17.5])          # clamped into the scene, sorted
        self.assertEqual(ok[0]["time"], "00:04")
        with self.assertRaises(ValueError):
            validate({"passages": [{"t": 5, "text": 'He says "I love paint."'}]}, 4, 18, LINES)
        with self.assertRaises(ValueError):
            validate({"passages": [{"t": 5, "text": "C1 walks in."}]}, 4, 18, LINES)

    def test_narrate_with_fake_model(self):
        characters, speech, scenes, timeline = fixture()
        sent, unloaded = [], []
        answers = iter([
            {"passages": [{"t": 0, "text": "The green rating card: this is a trailer."}]},
            {"passages": [{"t": 6, "text": 'He says "I paint every day."'}]},            # invented quote
            {"passages": [{"t": 6, "text": "Flashes pop as Tony Stark arrives."},
                          {"t": 11, "text": 'Deadpan: "Absolutely ridiculous. I don\'t paint."'}]}])

        def backend(payload, timeout):
            sent.append(json.loads(json.dumps(payload)))
            return next(answers)
        story = {"logline": "An arms maker changes course.", "form": "trailer_or_montage"}
        flat, meta = narrate("x", scenes, speech, characters, timeline, story, backend=backend,
                             frame_loader=lambda a, b, n: ["img"] * n, unload=unloaded.append)
        self.assertEqual([p["time"] for p in flat], ["00:00", "00:06", "00:11"])
        self.assertEqual(flat[1]["scene_id"], "SC2")
        self.assertEqual(meta["fallbacks"], 0)
        self.assertEqual(unloaded, ["qwen3.5:9b"])
        facts = json.loads(sent[1]["messages"][1]["content"])
        self.assertEqual(facts["story_so_far"], "The green rating card: this is a trailer.")
        self.assertEqual([d["who"] for d in facts["dialogue"]], ["unrecognised speaker (voice 1)", "Tony Stark"])
        self.assertEqual(facts["dialogue"][1]["delivery"], "flat, deadpan")
        self.assertIn("driving", facts["music_measured"][0])
        self.assertIn("quote not in the dialogue", sent[2]["messages"][2]["content"])
        self.assertNotIn("C1", json.dumps(facts["people"]) + json.dumps(facts["dialogue"]))

    def test_fallback_to_summary(self):
        characters, speech, scenes, timeline = fixture()
        flat, meta = narrate("x", scenes[:1], speech, characters, timeline, backend=lambda p, t: {"passages": []},
                             frame_loader=lambda a, b, n: [], unload=lambda m: None)
        self.assertEqual(flat[0]["text"], "A green rating card.")
        self.assertEqual(meta["fallbacks"], 1)


class ExperienceTests(unittest.TestCase):
    def result(self):
        characters, speech, scenes, timeline = fixture()
        scenes[1]["narration"] = [{"t": 6, "time": "00:06", "text": "Flashes pop."}]
        timeline[1]["feeling_signals"]["listening"] = {
            "description": "exciting rock; electric guitar", "sounds": [{"label": "camera shutters", "t": 6.0}]}
        return {"metadata": {"source_file": "clip.mp4", "duration_seconds": 149.3, "methods": ["x"]},
                "shots": [{"shot_id": 1}], "timeline": timeline, "speech_segments": speech,
                "characters": characters, "character_tracks": [{"x": 1}] * 50, "scenes": scenes,
                "story": {"logline": "L", "protagonist": "C1", "antagonist": "none", "form": "trailer_or_montage"},
                "narration": [{"t": 6, "time": "00:06", "text": "Flashes pop.", "scene_id": "SC2"}],
                "feeling_curve": {"start_seconds": [0], "valence": [0.1], "arousal": [0.2], "tension": [0.3],
                                  "label": ["calm"], "baselines": {}, "turning_points": [{"time_seconds": 80.0}]}}

    def test_compact_view(self):
        view = experience(self.result())
        self.assertEqual(view["film"]["duration"], "02:29")
        self.assertEqual(view["film"]["protagonist_is"], "Tony Stark")
        self.assertNotIn("antagonist", view["film"])
        self.assertEqual([c["id"] for c in view["characters"]], ["C1", "C2"])   # extra left out
        sc2 = view["scenes"][1]
        self.assertEqual(sc2["narration"], [{"time": "00:06", "text": "Flashes pop."}])
        self.assertNotIn("summary", sc2)                                       # narration replaces it
        self.assertEqual([d["text"] for d in sc2["dialogue"]], LINES)
        self.assertEqual(sc2["dialogue"][1]["name"], "Tony Stark")
        self.assertEqual(view["scenes"][0]["summary"], "A green rating card.")
        self.assertEqual(sc2["music"], [{"time": "00:05", "music": "exciting rock; electric guitar"}])
        self.assertEqual(sc2["sounds"], [{"time": "00:06", "sound": "camera shutters"}])
        self.assertEqual(view["scenes"][1]["music_moments"] if "music_moments" in view["scenes"][1] else None, None)
        self.assertEqual(view["feeling_curve"]["turning_points"][0]["time"], "01:20")
        for key in ("shots", "timeline", "character_tracks", "speech_segments"):
            self.assertNotIn(key, view)

    def test_written_files(self):
        from true_experience.__main__ import write_outputs
        with tempfile.TemporaryDirectory() as tmp:
            saved = write_outputs(self.result(), Path(tmp) / "clip_feeling.json")
            self.assertEqual([p.name for p in saved],
                             ["clip_feeling.json", "clip_feeling_details.json", "clip_feeling_narration.md"])
            view = json.loads(saved[0].read_text(encoding="utf-8"))
            self.assertEqual(view["about"]["details_file"], "clip_feeling_details.json")
            self.assertIn("**[00:06]** Flashes pop.", saved[2].read_text(encoding="utf-8"))
            self.assertIn("character_tracks", json.loads(saved[1].read_text(encoding="utf-8")))
            self.assertEqual(len(write_outputs(self.result(), Path(tmp) / "b.json", details=False)), 2)

    def test_render_text(self):
        text = render_text("clip", [{"time": "00:00", "text": "Hi."}], {"logline": "L", "emotional_journey": "J"})
        self.assertEqual(text, "# clip\n\n*L*\n\n**[00:00]** Hi.\n\n---\n\nJ\n")


if __name__ == "__main__":
    unittest.main()
