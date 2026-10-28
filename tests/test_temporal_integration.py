"""Exercise frame sampling against real decoded video, without model downloads."""
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from true_experience.pipeline import analyze
from true_experience.perception import caption_shots


@unittest.skipUnless(shutil.which("ffmpeg") and shutil.which("ffprobe"), "FFmpeg required")
class TemporalIntegrationTests(unittest.TestCase):
    def test_continuous_shot_multiple_captions(self):
        try:
            import PIL
        except ImportError:
            self.skipTest("Pillow needed for decoding")
        with tempfile.TemporaryDirectory() as tmp:
            video = Path(tmp)/"static.mp4"
            subprocess.run(["ffmpeg", "-hide_banner", "-nostdin", "-loglevel", "error", "-y",
                            "-f", "lavfi", "-i", "color=c=blue:s=64x64:r=10:d=21",
                            "-c:v", "mpeg4", str(video)], check=True, capture_output=True)

            def caption(frame, **kwargs):
                r, g, b = frame.getpixel((10, 10))
                self.assertGreater(b, r)
                return [{"generated_text": "test blue"}]

            def real_frames_with_stub_model(path, shots, **kwargs):
                return caption_shots(path, shots, backend=caption, **kwargs)

            with patch("true_experience.pipeline.caption_shots", side_effect=real_frames_with_stub_model):
                report = analyze(video, caption_frames=True, caption_spacing=5, max_caption_frames=10)

            times = [o["frame_time_seconds"] for o in report["visual_observations"]]
            self.assertEqual(times, [2.1, 6.3, 10.5, 14.7, 18.9])
            self.assertEqual(report["metadata"]["perception"]["caption_sample_count"], 5)
            stats = report["metadata"]["perception"]["visual_sampling"]
            self.assertEqual(stats["unsampled_shot_count"], 0)
            self.assertAlmostEqual(stats["largest_gap_between_sample_points_seconds"], 4.2)
            self.assertEqual(sum(len(t["visual_observations"]) for t in report["timeline"]), 5)
            self.assertTrue(all(t["emotion"] is None for t in report["timeline"]))

    def test_budget_capped(self):
        from true_experience.sampling import plan_frame_samples, describe_coverage
        shots = [{"shot_id": 1, "start_seconds": 0, "end_seconds": 21}]
        plan = plan_frame_samples(shots, limit=2, spacing_seconds=5)
        self.assertEqual([x["frame_time_seconds"] for x in plan], [5.25,15.75])
        self.assertEqual(describe_coverage(shots, plan)["largest_gap_between_sample_points_seconds"],10.5)
