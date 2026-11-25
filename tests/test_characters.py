"""Characters: tracking, face and clothing identity, speech links, roles."""
import unittest

try:
    import numpy as np
except ImportError:  # pragma: no cover
    np = None
try:
    import cv2
except ImportError:  # pragma: no cover
    cv2 = None


def unit(seed, dims):
    v = np.random.default_rng(seed).normal(size=dims)
    return v / np.linalg.norm(v)


def colours(index):
    h = np.zeros(12)
    h[index] = 1.0
    return h


def person(x, face=None, outfit=None, colour=3, share=0.2):
    return {"box": [x, 50.0, 100.0, 300.0], "score": 0.9, "frame_share": share,
            "face_box": [x + 30, 60.0, 40.0, 40.0] if face is not None else None,
            "face_embedding": face, "body_embedding": outfit, "colours": colours(colour)}


@unittest.skipIf(np is None, "NumPy not installed")
class LinkingTests(unittest.TestCase):
    def setUp(self):
        self.face_a, self.face_b = unit(1, 128), unit(2, 128)
        self.suit_x, self.suit_y, self.armour = unit(3, 768), unit(4, 768), unit(5, 768)
        self.coat = unit(6, 768)

    def near(self, v, seed):
        n = v + 0.15 * unit(seed, len(v))
        return n / np.linalg.norm(n)

    def run_film(self, frames, shots, speech=None):
        from true_experience.characters import analyze_characters
        timeline = [{"start_seconds": a, "end_seconds": a + 5} for a in range(0, 30, 5)]
        speech = speech if speech is not None else []
        it = iter(frames)
        result = analyze_characters("x.mp4", {"width": 960, "height": 540}, shots, timeline, speech,
                                    backend=lambda frame: next(it), frame_loader=lambda: (
                                        (i + 0.5, None) for i in range(len(frames))))
        return result, timeline, speech

    def test_face_clothing_and_suit_identities(self):
        A, B = self.face_a, self.face_b
        frames = (
            # shot 0 (0-3 s): A, face visible, suit X
            [[person(100, self.near(A, i), self.near(self.suit_x, 10 + i))] for i in range(3)] +
            # shot 1 (3-6 s): A from behind, suit X, no face
            [[person(300, None, self.near(self.suit_x, 20 + i))] for i in range(3)] +
            # shot 2 (6-9 s): A in other clothes (Y), face visible; B in a coat beside them
            [[person(100, self.near(A, 30 + i), self.near(self.suit_y, 40 + i), colour=8),
              person(500, self.near(B, 50 + i), self.near(self.coat, 60 + i), colour=11)] for i in range(3)] +
            # shots 3 and 4: a faceless armoured figure, twice
            [[person(200, None, self.near(self.armour, 70 + i), colour=0)] for i in range(3)] +
            [[person(600, None, self.near(self.armour, 80 + i), colour=0)] for i in range(3)])
        shots = [{"start_seconds": 3 * k, "end_seconds": 3 * k + 3} for k in range(5)]
        result, timeline, _ = self.run_film(frames, shots)
        chars = {c["id"]: c for c in result["characters"]}
        rows = result["character_tracks"]
        a_id = rows[0]["character"]
        self.assertEqual(chars[a_id]["identified_by"], "face")
        self.assertEqual(len(chars[a_id]["outfit_ids"]), 2, "A wears two outfits")
        back = [r for r in rows if r["shot"] == 1][0]
        self.assertEqual(back["character"], a_id, "faceless back view linked by clothing")
        self.assertEqual(back["link"], "clothing")
        b_row = [r for r in rows if r["shot"] == 2 and r["face_id"] != rows[0]["face_id"]][0]
        self.assertNotEqual(b_row["character"], a_id)
        armour = {r["character"] for r in rows if r["shot"] in (3, 4)}
        self.assertEqual(len(armour), 1, "same suit in two shots is one character")
        (armour_id,) = armour
        self.assertEqual(chars[armour_id]["identified_by"], "clothing")
        self.assertEqual(chars[armour_id]["face_ids"], [])
        self.assertEqual(len(chars), 3)
        self.assertEqual(chars["C1"]["screen_seconds"], 9.0)
        self.assertIn("characters_present", timeline[0])
        self.assertNotIn("face_embedding", str(result))

    def test_people_seen_together_are_never_merged(self):
        A = self.face_a
        frames = [[person(100, self.near(A, i), self.near(self.suit_x, 10 + i)),
                   person(500, None, self.near(self.suit_x, 20 + i))] for i in range(3)]
        result, _, _ = self.run_film(frames, [{"start_seconds": 0, "end_seconds": 30}])
        self.assertEqual(len({r["character"] for r in result["character_tracks"]}), 2)

    def test_voice_group_links_lines_to_character(self):
        A, B = self.face_a, self.face_b
        frames = ([[person(100, self.near(A, i), self.near(self.suit_x, 10 + i))] for i in range(4)] +
                  [[person(100, self.near(B, 20 + i), self.near(self.suit_y, 30 + i), colour=8)]
                   for i in range(4)])
        shots = [{"start_seconds": 0, "end_seconds": 4}, {"start_seconds": 4, "end_seconds": 8}]
        speech = [{"start_seconds": 0.2, "end_seconds": 1.8, "speaker": "S1", "text": "a"},
                  {"start_seconds": 2.0, "end_seconds": 3.9, "speaker": "S1", "text": "b"},
                  {"start_seconds": 4.2, "end_seconds": 5.8, "speaker": "S2", "text": "c"},
                  {"start_seconds": 6.0, "end_seconds": 7.0, "speaker": "S1", "text": "d"}]
        result, _, speech = self.run_film(frames, shots, speech)
        chars = {c["id"]: c for c in result["characters"]}
        a_id = result["character_tracks"][0]["character"]
        self.assertEqual(chars[a_id]["voice_groups"], ["S1"])
        self.assertEqual(speech[3]["character"], a_id, "S1 heard over B's shot is still A")
        self.assertEqual(speech[3]["character_source"], "voice_group")
        self.assertEqual(speech[2]["character_source"], "only_face_on_screen")
        self.assertGreater(chars[a_id]["dialogue_seconds"], 4)
        self.assertTrue(chars[a_id]["likely_main_character"])


class RoleTests(unittest.TestCase):
    def test_validate(self):
        from true_experience.characterhood import validate
        good = {"narrative_role": "antagonist", "archetypes": ["terrorist", "unclear", "terrorist"],
                "occupation": "insurgent leader", "occupation_cues": ["rifle", "camp"],
                "appearance": "Dark scarf.", "evidence": "Holds a rifle on hostages.", "confidence": 3}
        r = validate(good)
        self.assertEqual(r["archetypes"], ["terrorist"])
        self.assertEqual(r["confidence"], 1.0)
        with self.assertRaises(ValueError):
            validate({**good, "narrative_role": "evil"})
        with self.assertRaises(ValueError):
            validate({**good, "archetypes": ["wizard"]})

    @unittest.skipIf(np is None or cv2 is None, "NumPy or OpenCV not installed")
    def test_interpret_with_fake_model(self):
        from true_experience.characterhood import interpret_characters
        chars = [{"id": "C1", "screen_seconds": 40.0, "prominence": 1.0, "prominence_rank": 1,
                  "likely_main_character": True, "shots": [0, 1], "face_ids": ["F1"],
                  "best_face": {"t": 1.5, "box": [100, 50, 60, 60]},
                  "best_body": {"t": 2.5, "box": [80, 40, 200, 400]},
                  "outfits": [{"id": "O1", "colours": ["grey"], "sample": {"t": 2.5, "box": [80, 40, 200, 400]}}]},
                 {"id": "C2", "screen_seconds": 1.0, "prominence": 0.1, "prominence_rank": 2,
                  "likely_main_character": False, "shots": [1], "face_ids": [], "outfits": []}]
        speech = [{"start_seconds": 0, "end_seconds": 2, "text": "I am a reporter.", "character": "C1"}]
        timeline = [{"start_seconds": 0, "end_seconds": 5, "characters_present": [{"id": "C1"}],
                     "feeling_signals": {"visual_mood": {"evidence": "A newsroom."}}}]
        seen, unloaded = [], []

        def backend(payload, timeout):
            seen.append(payload)
            return {"narrative_role": "protagonist", "archetypes": ["journalist", "hero"],
                    "occupation": "news reporter", "occupation_cues": ["press badge"],
                    "appearance": "Grey jacket.", "evidence": "Says she is a reporter.", "confidence": 0.7}
        meta = interpret_characters("x", {}, chars, speech, timeline, backend=backend,
                                    grab=lambda t: np.zeros((540, 960, 3), np.uint8),
                                    unload=unloaded.append)
        self.assertEqual(meta["characters_interpreted"], 1)
        self.assertEqual(len(seen[0]["messages"][1]["images"]), 3)
        facts = seen[0]["messages"][1]["content"]
        self.assertIn("I am a reporter.", facts)
        self.assertIn("A newsroom.", facts)
        self.assertEqual(chars[0]["role"]["archetypes"], ["journalist", "hero"])
        self.assertTrue(chars[0]["main_character"])
        self.assertFalse(chars[1]["main_character"])
        self.assertEqual(unloaded, ["qwen3.5:9b"])


@unittest.skipIf(np is None or cv2 is None, "NumPy or OpenCV not installed")
class ColourTests(unittest.TestCase):
    def test_named_colours(self):
        from true_experience.characters import colour_histogram, colour_words
        patch = np.zeros((20, 20, 3), np.uint8)
        patch[:, :10] = (0, 0, 220)      # red (BGR)
        patch[:, 10:] = (230, 230, 230)  # white
        self.assertEqual(sorted(colour_words(colour_histogram(patch))), ["red", "white"])
        self.assertEqual(colour_words(colour_histogram(np.zeros((8, 8, 3), np.uint8))), ["black"])


if __name__ == "__main__":
    unittest.main()
