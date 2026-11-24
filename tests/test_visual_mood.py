"""Visual mood stage with a fake Ollama backend (no model or GPU needed)."""
import base64
import unittest

try:
    from PIL import Image
except ImportError:  # pragma: no cover
    Image = None

from true_experience import fusion
from true_experience import visual_mood as vm


def good(**changes):
    answer = {"lighting": "dark", "colour_mood": "cold blue", "people_visible": 1,
              "facial_expressions": [{"person": "P1", "expression": "afraid"}],
              "body_language": "hunched, backing away", "action_intensity": "medium",
              "camera": "handheld", "mood_words": ["dread"], "visual_valence": -0.7,
              "visual_arousal": 0.5, "confidence": 0.8,
              "evidence": "Dim corridor, wide eyes, figure retreats across frames."}
    answer.update(changes)
    return answer


def timeline(count=2):
    return [{"start_seconds": i * 5.0, "end_seconds": i * 5.0 + 5,
             "audio": {"state": "not_detected_silent"},
             "feeling_signals": {"pacing": {"cuts_per_minute": 12},
                                 "look": {"brightness": 0.1, "contrast": 0.3, "saturation": 0.2,
                                          "warmth": -0.2, "motion": 0.03},
                                 "audio": None}} for i in range(count)]


@unittest.skipIf(Image is None, "Pillow not installed")
class VisualMoodTests(unittest.TestCase):
    def setUp(self):
        self.loaded = []
        self.unloaded = []

    def loader(self, video, t):
        self.loaded.append(t)
        return Image.new("RGB", (1280, 720), (20, 20, 40))

    def run_stage(self, backend, **kw):
        tl = timeline()
        meta = vm.analyze_visual_mood("x.mp4", tl, backend=backend, frame_loader=self.loader,
                                      unload=self.unloaded.append, **kw)
        return tl, meta

    def test_frame_sequence_payload_and_result(self):
        payloads = []

        def backend(payload, timeout):
            payloads.append(payload)
            return good()

        tl, meta = self.run_stage(backend, frames_per_window=4)
        first = payloads[0]
        self.assertEqual(first["model"], "qwen3.5:9b")
        self.assertFalse(first["think"])
        self.assertEqual(first["format"], vm.SCHEMA)
        user = first["messages"][1]
        self.assertEqual(len(user["images"]), 4)
        self.assertTrue(base64.b64decode(user["images"][0]).startswith(b"\xff\xd8"))
        self.assertIn("measured_signals", user["content"])
        self.assertEqual(self.loaded[:4], [0.625, 1.875, 3.125, 4.375])
        mood = tl[0]["feeling_signals"]["visual_mood"]
        self.assertEqual(mood["lighting"], "dark")
        self.assertEqual(mood["frame_times_seconds"], [0.625, 1.875, 3.125, 4.375])
        self.assertEqual(self.unloaded, ["qwen3.5:9b"])
        self.assertEqual(meta["abstained_windows"], 0)

    def test_retry_then_abstain_and_still_unloads(self):
        calls = []

        def backend(payload, timeout):
            calls.append(payload)
            return good(lighting="purple")

        tl, meta = self.run_stage(backend)
        self.assertEqual(len(calls), 4)  # two windows x two attempts
        self.assertIn("Invalid answer", calls[1]["messages"][-1]["content"])
        self.assertIsNone(tl[0]["feeling_signals"]["visual_mood"])
        self.assertEqual(meta["abstained_windows"], 2)
        self.assertEqual(self.unloaded, ["qwen3.5:9b"])

    def test_unload_even_when_ollama_down(self):
        def backend(payload, timeout):
            raise RuntimeError("Local Ollama unavailable")

        with self.assertRaises(RuntimeError):
            self.run_stage(backend)
        self.assertEqual(self.unloaded, ["qwen3.5:9b"])

    def test_values_clamped_and_validated(self):
        out = vm.validate(good(visual_valence=-3, confidence=2))
        self.assertEqual(out["visual_valence"], -1.0)
        self.assertEqual(out["confidence"], 1.0)
        for bad in (good(visual_arousal="high"), good(people_visible=-1),
                    good(facial_expressions=[{"person": "P1", "expression": "smug"}]),
                    good(camera=None)):
            with self.assertRaises(ValueError):
                vm.validate(bad)

    def test_bad_settings(self):
        with self.assertRaises(ValueError):
            vm.analyze_visual_mood("x", timeline(), frames_per_window=9, backend=lambda p, t: good(),
                                   frame_loader=self.loader, unload=self.unloaded.append)

    def test_fusion_uses_visual_mood(self):
        def backend(payload, timeout):
            return good()

        tl, _ = self.run_stage(backend)
        tl[0]["feeling_signals"]["audio"] = {
            "rms_dbfs": -30.0, "peak_short_term_dbfs": -28.0, "startle": False,
            "onset_rate_per_second": 0.5, "tempo_bpm": None, "tempo_confidence": 0.0,
            "mode": "major", "mode_confidence": 1.0}
        fusion.fuse_timeline(tl)
        feeling = tl[0]["feeling"]
        self.assertIn("visual", feeling)
        self.assertEqual(feeling["visual"]["mood_words"], ["dread"])
        self.assertLess(feeling["fused"]["valence"], 0.2)
        self.assertTrue(feeling["fused"]["possible_irony"])


if __name__ == "__main__":
    unittest.main()


class WindowExtractionTests(unittest.TestCase):
    def test_one_ffmpeg_call_gives_all_frames_in_order(self):
        import shutil, subprocess, tempfile
        from pathlib import Path
        if not shutil.which("ffmpeg"):
            self.skipTest("FFmpeg missing")
        with tempfile.TemporaryDirectory() as td:
            video = Path(td) / "v.mp4"
            # Brightness ramps up over 5 s so frame order is checkable.
            subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "lavfi",
                            "-i", "color=c=black:s=640x360:r=25:d=5",
                            "-vf", "format=yuv420p,geq=lum=T*50:cb=128:cr=128",
                            "-c:v", "mpeg4", str(video)], check=True)
            jpegs = vm.extract_window_jpegs(video, 0.0, 5.0, 6)
            self.assertEqual(len(jpegs), 6)
            raw = [base64.b64decode(j) for j in jpegs]
            self.assertTrue(all(r.startswith(b"\xff\xd8") and r.endswith(b"\xff\xd9") for r in raw))
            if Image is not None:
                import io
                imgs = [Image.open(io.BytesIO(r)).convert("L") for r in raw]
                self.assertEqual(imgs[0].width, 512)
                means = [sum(i.getdata()) / (i.width * i.height) for i in imgs]
                self.assertEqual(means, sorted(means))
                self.assertGreater(means[-1] - means[0], 100)
