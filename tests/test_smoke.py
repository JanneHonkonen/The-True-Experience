import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path
from true_experience.pipeline import analyze

@unittest.skipUnless(shutil.which("ffmpeg") and shutil.which("ffprobe"),"FFmpeg required")
class SmokeTests(unittest.TestCase):
    def test_actual_video_cut(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/"cut.mp4"
            subprocess.run(["ffmpeg","-hide_banner","-nostdin","-v","error","-y",
                "-f","lavfi","-i","color=c=red:s=64x64:r=10:d=1",
                "-f","lavfi","-i","color=c=blue:s=64x64:r=10:d=1",
                "-filter_complex","[0:v][1:v]concat=n=2:v=1:a=0[v]",
                "-map","[v]","-c:v","mpeg4",str(path)],check=True,capture_output=True)
            report=analyze(path,interval=.75,scene_threshold=.01)
            self.assertAlmostEqual(report["metadata"]["duration_seconds"],2,places=1)
            self.assertEqual(len(report["timeline"]),3)
            self.assertGreaterEqual(len(report["shots"]),2)
            self.assertEqual(report["timeline"][0]["audio"]["state"],"no_audio")
            self.assertIsNone(report["timeline"][0]["visual_description"])

    def test_silent_audio_stream(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/"silent.mp4"
            subprocess.run(["ffmpeg","-hide_banner","-nostdin","-v","error","-y",
                "-f","lavfi","-i","color=c=black:s=64x64:r=10:d=1",
                "-f","lavfi","-i","anullsrc=r=16000:cl=mono",
                "-t","1","-c:v","mpeg4","-c:a","aac","-shortest",str(path)],check=True,capture_output=True)
            report=analyze(path,interval=1)
            self.assertTrue(report["metadata"]["has_audio"])
            self.assertTrue(report["silence_intervals"])
            self.assertEqual(report["timeline"][0]["audio"]["state"],"silent")
