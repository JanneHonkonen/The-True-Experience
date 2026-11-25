"""Listening: music style, instruments, mood, sound effects and music moments (fake CLAP)."""
import unittest

try:
    import numpy as np
except ImportError:  # pragma: no cover
    np = None


def fake_backend(plan):
    """Text labels get fixed random vectors; a clip filled with value k sounds like plan[k]."""
    rng = {}

    def vec(label):
        if label not in rng:
            r = np.random.default_rng(abs(hash(label)) % (2 ** 32)).normal(size=64)
            rng[label] = r / np.linalg.norm(r)
        return rng[label]

    def text(labels):
        return np.stack([vec(l) for l in labels])

    def audio(clips):
        out = []
        for c in clips:
            v = sum(vec(l) for l in plan[int(round(float(np.mean(c))))])
            out.append(v / np.linalg.norm(v))
        return np.stack(out)
    closed = []
    return {"text": text, "audio": audio, "close": lambda: closed.append(1), "scale": 30.0,
            "device": "cpu", "closed": closed}


@unittest.skipIf(np is None, "NumPy not installed")
class ListeningTests(unittest.TestCase):
    def test_windows_music_and_sounds(self):
        from true_experience.listening import INSTRUMENTS, MOODS, MUSIC, NO_MUSIC, SOUNDS, STYLES, listen
        plan = {1: [MUSIC, STYLES["rock"], INSTRUMENTS["electric guitar"], MOODS["exciting"][0]],
                2: [NO_MUSIC, SOUNDS["explosion"]],
                3: [MUSIC, STYLES["orchestral film score"], INSTRUMENTS["brass"], MOODS["heroic"][0]]}
        timeline = [{"start_seconds": a, "end_seconds": a + 5,
                     "feeling_signals": {"music": {"rms_dbfs": db}}}
                    for a, db in ((0, -20.0), (5, -38.0), (10, -18.0))]
        clips = [np.full(48000 * 5, k, np.float32) for k in (1, 2, 3)]
        backend = fake_backend(plan)
        moments, meta = listen("x", timeline, backend=backend, reader=lambda bounds: iter(clips))
        first = timeline[0]["feeling_signals"]["listening"]
        self.assertTrue(first["music"]["present"])
        self.assertEqual(first["music"]["style"][0]["label"], "rock")
        self.assertEqual(first["music"]["mood"][0]["label"], "exciting")
        self.assertIn("electric guitar", [i["label"] for i in first["music"]["instruments"]])
        self.assertTrue(first["description"].startswith("exciting"))
        second = timeline[1]["feeling_signals"]["listening"]
        self.assertFalse(second["music"]["present"])
        self.assertEqual(second["description"], "no music")
        self.assertEqual([s["label"] for s in second["sounds"]], ["explosion", "explosion"])
        self.assertEqual([s["t"] for s in second["sounds"]], [5.0, 7.5])
        third = timeline[2]["feeling_signals"]["listening"]["music"]
        self.assertEqual(third["mood"][0]["label"], "heroic")
        self.assertGreater(third["mood_valence"], 0.2)
        self.assertEqual([m["kind"] for m in moments], ["music_stops", "music_starts"])
        self.assertEqual(backend["closed"], [1])
        self.assertEqual(meta["device"], "cpu")

    def test_music_moments(self):
        from true_experience.listening import music_moments
        levels = [-30, -27, -24, -20, -35, -34]
        timeline = [{"start_seconds": i * 5, "feeling_signals": {
            "music": {"rms_dbfs": db, "startle_times_seconds": [12.4] if i == 2 else []},
            "listening": {"music": {"present": True}}}} for i, db in enumerate(levels)]
        kinds = [(m["kind"], m["t"]) for m in music_moments(timeline)]
        self.assertIn(("music_builds", 0), kinds)
        self.assertIn(("music_drops", 20), kinds)
        self.assertIn(("hit", 12.4), kinds)

    def test_fusion_uses_heard_mood(self):
        from true_experience.fusion import read_music
        stem = {"mode": "minor", "mode_confidence": 0.6, "tempo_bpm": 120, "tempo_confidence": 0.5,
                "onset_rate_per_second": 2.0, "rms_dbfs": -20}
        heard = {"music": {"present": True, "mood": [{"label": "heroic", "score": 0.7}],
                           "mood_valence": 0.5, "mood_arousal": 0.6, "mood_confidence": 0.7}}
        r = read_music(stem, separated=True, listening=heard)
        self.assertGreater(r["valence"], 0.2)          # heroic outweighs the minor mode
        self.assertEqual(r["moods"], ["heroic"])
        self.assertIsNone(read_music(stem, separated=True, listening={"music": {"present": False}}))
        self.assertLess(read_music(stem, separated=True)["valence"], 0)

    def test_describe_music(self):
        from true_experience.listening import describe_music
        self.assertEqual(describe_music({"music": {"present": True, "mood": [{"label": "tense"}],
                                                   "style": [{"label": "orchestral film score"}],
                                                   "instruments": [{"label": "strings"}]}}),
                         "tense orchestral film score; strings")
        self.assertIsNone(describe_music(None))


if __name__ == "__main__":
    unittest.main()
