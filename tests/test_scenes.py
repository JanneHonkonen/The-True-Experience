"""Scenes: grouping shots by look, people, speech and silence; story readings."""
import json
import unittest

try:
    import numpy as np
except ImportError:  # pragma: no cover
    np = None


def shots_of(lengths):
    out, t = [], 0.0
    for i, n in enumerate(lengths):
        out.append({"shot_id": i + 1, "start_seconds": t, "end_seconds": t + n})
        t += n
    return out


@unittest.skipIf(np is None, "NumPy not installed")
class SceneTests(unittest.TestCase):
    def sig(self, colour):
        from true_experience.scenes import look_signature
        frame = np.zeros((18, 32, 3), np.uint8)
        frame[:] = colour
        frame[:6, :10] = (255, 255, 255)
        return look_signature([frame])

    def test_shot_reverse_shot_stays_one_scene_and_a_new_place_breaks(self):
        from true_experience.scenes import detect_scenes
        office, office_b, desert = self.sig((90, 60, 40)), self.sig((80, 70, 50)), self.sig((230, 200, 120))
        # Office dialogue A/B/A/B, then the desert.
        shots = shots_of([3, 3, 3, 3, 5, 5])
        sigs = [office, office_b, office, office_b, desert, desert]
        tracks = [{"character": "C1", "shot": 0}, {"character": "C2", "shot": 1},
                  {"character": "C1", "shot": 2}, {"character": "C2", "shot": 3},
                  {"character": "C3", "shot": 4}, {"character": "C3", "shot": 5}]
        chars = [{"id": c, "background": False} for c in ("C1", "C2", "C3")]
        speech = [{"start_seconds": 2.0, "end_seconds": 4.0, "speaker": "S1", "character": "C1", "text": "x"}]
        timeline = [{"start_seconds": a, "end_seconds": a + 5,
                     "feeling": {"fused": {"valence": 0.1, "arousal": 0.2, "tension": 0.3, "label": "calm"}}}
                    for a in range(0, 25, 5)]
        scenes, meta = detect_scenes("x", {}, shots, timeline, speech, [], characters=chars,
                                     character_tracks=tracks, signatures=sigs)
        self.assertEqual([s["shot_ids"] for s in scenes], [[1, 2, 3, 4], [5, 6]])
        self.assertEqual(scenes[0]["characters"], ["C1", "C2"])
        self.assertEqual(scenes[0]["lines"], 1)
        self.assertEqual(scenes[1]["characters"], ["C3"])
        self.assertEqual(timeline[2]["scene_ids"], ["SC1", "SC2"])  # 10-15 s spans the change at 12 s
        self.assertTrue(shots[1]["continuity_with_scene"]["continuity"] >= 0.45)
        self.assertLess(shots[4]["continuity_with_scene"]["continuity"], 0.45)
        self.assertEqual(meta["scenes"], 2)

    def test_short_montage_scenes_are_merged(self):
        from true_experience.scenes import group_scenes
        shots = shots_of([1, 1, 1, 6])
        scores = [None] + [{"continuity": c} for c in (0.1, 0.3, 0.2)]
        groups = group_scenes(shots, scores, min_seconds=4.0)
        self.assertTrue(all(shots[g[-1]]["end_seconds"] - shots[g[0]]["start_seconds"] >= 4 for g in groups))
        self.assertEqual(sum(len(g) for g in groups), 4)

    def test_silence_at_cut_pushes_towards_a_break(self):
        from true_experience.scenes import cut_scores
        same = self.sig((90, 60, 40))
        shots = shots_of([4, 4])
        quiet = cut_scores(shots, [same, same], [set(), set()], [], [(3.5, 5.0)])
        loud = cut_scores(shots, [same, same], [set(), set()], [], [])
        self.assertLess(quiet[1]["continuity"], loud[1]["continuity"])
        self.assertTrue(quiet[1]["silence_at_cut"])


class StoryTests(unittest.TestCase):
    def scenes(self):
        return [{"scene_id": "SC1", "start_seconds": 0.0, "end_seconds": 10.0, "shot_ids": [1, 2],
                 "characters": ["C1", "C2"], "feeling": {"tension": 0.2, "valence": 0.1}},
                {"scene_id": "SC2", "start_seconds": 10.0, "end_seconds": 14.0, "shot_ids": [3],
                 "characters": ["C6"], "feeling": {"tension": 0.8, "valence": -0.3}}]

    def test_scene_and_arc_readings_with_fake_model(self):
        from true_experience.story import interpret_story
        scenes = self.scenes()
        chars = [{"id": "C1", "screen_seconds": 30, "main_character": True, "identified_by": "face",
                  "role": {"narrative_role": "protagonist"}},
                 {"id": "C2", "screen_seconds": 5}, {"id": "C6", "screen_seconds": 3, "identified_by": "clothing"}]
        speech = [{"start_seconds": 1, "end_seconds": 3, "text": "Mr. Stark, a question.", "speaker": "S1"},
                  {"start_seconds": 4, "end_seconds": 6, "text": "Go ahead.", "character": "C1"}]
        sent, unloaded = [], []

        def backend(payload, timeout):
            sent.append(payload)
            if payload["format"]["properties"].get("scene_type"):
                return {"summary": "A press interview.", "location": "hotel lobby", "time_of_day": "day",
                        "scene_type": "dialogue", "story_beat": "setup", "emotional_turn": "playful to tense",
                        "character_actions": [{"id": "C1", "doing": "answers"}, {"id": "C9", "doing": "x"}],
                        "confidence": 0.8}
            return {"logline": "An arms maker changes course.", "genres": ["action"], "form": "trailer_or_montage",
                    "protagonist": "C1", "antagonist": "C7", "emotional_journey": "From swagger to resolve.",
                    "turning_points": [{"scene_id": "SC2", "what_changes": "the suit appears"},
                                       {"scene_id": "SC9", "what_changes": "made up"}],
                    "character_links": [{"ids": ["C6", "C1"], "hypothesis": "C1 inside the suit", "confidence": 0.7}],
                    "confidence": 0.6}
        story, meta = interpret_story("x", scenes, [], speech, chars, {"turning_points": []},
                                      backend=backend, frame_loader=lambda a, b, n: ["img"] * n,
                                      unload=unloaded.append)
        self.assertEqual(meta["scenes_read"], 2)
        self.assertEqual(scenes[0]["reading"]["character_actions"], [{"id": "C1", "doing": "answers"}])
        facts = json.loads(sent[0]["messages"][1]["content"])
        self.assertIn("S1: Mr. Stark, a question.", facts["dialogue"])
        self.assertIn("C1: Go ahead.", facts["dialogue"])
        self.assertEqual(len(sent[0]["messages"][1]["images"]), 6)
        self.assertEqual(len(sent[1]["messages"][1]["images"]), 4)  # 4 s scene -> 4 frames
        self.assertNotIn("images", sent[2]["messages"][1])          # the arc is text only
        self.assertEqual(story["antagonist"], "none")               # unknown ID dropped
        self.assertEqual([p["scene_id"] for p in story["turning_points"]], ["SC2"])
        self.assertEqual(story["character_links"][0]["ids"], ["C6", "C1"])
        self.assertEqual(story["kind"], "interpretation")
        self.assertEqual(unloaded, ["qwen3.5:9b"])
        arc_cast = json.loads(sent[2]["messages"][1]["content"])["characters"]
        self.assertEqual(arc_cast[0]["name"], None)
        self.assertEqual(meta["naming"]["named"], 0)
        self.assertEqual(story["names_not_on_screen"], [])

    def test_invalid_scene_answer_abstains(self):
        from true_experience.story import interpret_story
        scenes = self.scenes()[:1]
        calls = []

        def backend(payload, timeout):
            calls.append(1)
            return {"summary": "x"}
        story, meta = interpret_story("x", scenes, [], [], [], None, backend=backend,
                                      frame_loader=lambda a, b, n: [], unload=lambda m: None)
        self.assertIsNone(scenes[0]["reading"])
        self.assertEqual(meta["abstained"], 1)
        self.assertIsNone(story)


if __name__ == "__main__":
    unittest.main()
