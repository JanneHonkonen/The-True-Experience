"""Fusion v1: readings, carry-over, tension, conflicts, irony and turning points."""
import json
import unittest

from true_experience import fusion


def window(start, *, look=None, audio=None, cpm=10.0, state="not_detected_silent"):
    return {"start_seconds": start, "end_seconds": start + 5, "audio": {"state": state},
            "emotion": None,
            "feeling_signals": {"pacing": {"cuts_per_minute": cpm}, "look": look, "audio": audio}}


DARK = {"brightness": 0.08, "contrast": 0.35, "saturation": 0.2, "warmth": -0.3, "motion": 0.01}
BRIGHT = {"brightness": 0.7, "contrast": 0.2, "saturation": 0.6, "warmth": 0.4, "motion": 0.02}
CALM_AUDIO = {"rms_dbfs": -40.0, "peak_short_term_dbfs": -38.0, "startle": False,
              "onset_rate_per_second": 0.2, "tempo_bpm": None, "tempo_confidence": 0.0,
              "mode": "unclear", "mode_confidence": 0.0}


def audio(**changes):
    return {**CALM_AUDIO, **changes}


class ReadingTests(unittest.TestCase):
    def test_pacing_arousal_scale(self):
        self.assertEqual(fusion.read_pacing({"cuts_per_minute": 0})["arousal"], -0.5)
        self.assertEqual(fusion.read_pacing({"cuts_per_minute": 40})["arousal"], 1.0)
        self.assertIsNone(fusion.read_pacing({"cuts_per_minute": None}))

    def test_look_dark_vs_bright(self):
        dark, bright = fusion.read_look(DARK), fusion.read_look(BRIGHT)
        self.assertLess(dark["valence"], 0)
        self.assertGreater(bright["valence"], 0)
        self.assertGreater(dark["darkness"], bright["darkness"])

    def test_music_mode_sets_valence_sign(self):
        major = fusion.read_music(audio(mode="major", mode_confidence=0.8))
        minor = fusion.read_music(audio(mode="minor", mode_confidence=0.8))
        self.assertGreater(major["valence"], 0)
        self.assertLess(minor["valence"], 0)
        self.assertIsNone(fusion.read_music(audio()))

    def test_silence_is_calm(self):
        self.assertEqual(fusion.read_sound(None, "silent")["arousal"], -0.6)


class TimelineTests(unittest.TestCase):
    def test_calm_bright_scene_vs_dark_minor_scene(self):
        timeline = [window(0, look=BRIGHT, audio=audio(mode="major", mode_confidence=0.8), cpm=2)]
        fusion.fuse_timeline(timeline)
        calm = timeline[0]["feeling"]["fused"]
        timeline = [window(0, look=DARK, audio=audio(mode="minor", mode_confidence=0.8,
                                                      rms_dbfs=-15.0, onset_rate_per_second=3.0),
                           cpm=30)]
        fusion.fuse_timeline(timeline)
        tense = timeline[0]["feeling"]["fused"]
        self.assertGreater(calm["valence"], 0)
        self.assertLess(tense["valence"], 0)
        self.assertGreater(tense["tension"], calm["tension"])
        self.assertIn(calm["label"], ("calm", "warmth"))
        self.assertEqual(tense["label"], "tension")
        self.assertEqual(timeline[0]["emotion"], "tension")

    def test_startle_is_immediate_shock(self):
        timeline = [window(0, look=BRIGHT, audio=audio()),
                    window(5, look=BRIGHT, audio=audio(rms_dbfs=-12.0, startle=True))]
        curve = fusion.fuse_timeline(timeline)
        self.assertEqual(curve["label"][1], "shock")
        self.assertGreater(curve["arousal"][1], curve["arousal"][0] + 0.5)

    def test_feeling_lingers_after_loud_scene(self):
        loud = audio(rms_dbfs=-10.0, onset_rate_per_second=4.0)
        timeline = [window(0, look=DARK, audio=loud, cpm=30),
                    window(5, look=DARK, audio=loud, cpm=30),
                    window(10, look=BRIGHT, audio=audio(), cpm=2)]
        curve = fusion.fuse_timeline(timeline)
        alone = [window(0, look=BRIGHT, audio=audio(), cpm=2)]
        fusion.fuse_timeline(alone)
        self.assertGreater(curve["arousal"][2], alone[0]["feeling"]["fused"]["arousal"])

    def test_hush_after_loud_raises_tension(self):
        loud = audio(rms_dbfs=-12.0, peak_short_term_dbfs=-6.0)
        with_hush = [window(0, look=DARK, audio=loud), window(5, look=DARK, audio=audio(), state="silent")]
        quiet = [window(0, look=DARK, audio=audio()), window(5, look=DARK, audio=audio(), state="silent")]
        a = fusion.fuse_timeline(with_hush)["tension"][1]
        b = fusion.fuse_timeline(quiet)["tension"][1]
        self.assertGreater(a, b)

    def test_irony_when_happy_music_over_grim_visuals(self):
        grim = {**DARK, "brightness": 0.03, "warmth": -0.8, "saturation": 0.05}
        timeline = [window(0, look=grim, audio=audio(mode="major", mode_confidence=1.0,
                                                      tempo_confidence=0.6))]
        fusion.fuse_timeline(timeline)
        fused = timeline[0]["feeling"]["fused"]
        self.assertTrue(fused["possible_irony"])
        self.assertEqual(fused["conflicts"][0]["between"], ["look", "music"])

    def test_no_irony_when_senses_agree(self):
        timeline = [window(0, look=BRIGHT, audio=audio(mode="major", mode_confidence=1.0))]
        fusion.fuse_timeline(timeline)
        self.assertFalse(timeline[0]["feeling"]["fused"]["possible_irony"])
        self.assertEqual(timeline[0]["feeling"]["fused"]["conflicts"], [])

    def test_turning_points_and_json_safe(self):
        timeline = ([window(i * 5, look=BRIGHT, audio=audio(mode="major", mode_confidence=0.9), cpm=2)
                     for i in range(4)] +
                    [window(20 + i * 5, look=DARK,
                            audio=audio(mode="minor", mode_confidence=0.9, rms_dbfs=-12.0,
                                        onset_rate_per_second=4.0), cpm=35)
                     for i in range(4)] +
                    [window(40, look=BRIGHT, audio=audio(), cpm=2)])
        curve = fusion.fuse_timeline(timeline)
        kinds = {p["kind"] for p in curve["turning_points"]}
        self.assertIn("mood_turn", kinds)
        self.assertIn("tension_peak", kinds)
        darker = [p for p in curve["turning_points"] if p.get("direction") == "darker"]
        self.assertTrue(20 <= darker[0]["time_seconds"] <= 30)
        json.dumps(timeline, allow_nan=False)

    def test_windows_without_signals_are_skipped(self):
        timeline = [{"start_seconds": 0, "end_seconds": 5}]
        curve = fusion.fuse_timeline(timeline)
        self.assertEqual(curve["valence"], [])
        self.assertNotIn("feeling", timeline[0])


if __name__ == "__main__":
    unittest.main()
