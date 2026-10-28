"""Verify context reasoning receives genuine FFmpeg-decoded JPEG frames."""
import base64
import io
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

from true_experience.context import build_context
from true_experience.pipeline import create_shots, build_timeline


@unittest.skipUnless(shutil.which("ffmpeg") and shutil.which("ffprobe"), "FFmpeg required")
class RealFrameContextTests(unittest.TestCase):
    def test_real_video_frame_reaches_mocked_local_multimodal_backend(self):
        try:
            from PIL import Image
        except ImportError:
            self.skipTest("Pillow required for local frame decoding")
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/"green.mp4"
            subprocess.run([
                "ffmpeg","-hide_banner","-nostdin","-loglevel","error","-y",
                "-f","lavfi","-i","color=c=green:s=64x64:r=10:d=2",
                "-c:v","mpeg4",str(path)],check=True,capture_output=True)
            shots=create_shots([],2)
            report={
                "metadata":{"duration_seconds":2},
                "shots":shots,
                "timeline":build_timeline(2,1,shots,[],False,[]),
                "visual_observations":[{
                    "frame_time_seconds":1.0,"shot_id":1,
                    "description":"a green background","model":"stub",
                    "source":"vision_language_model"}],
                "speech_segments":[]}
            def backend(messages,model,timeout):
                images=messages[1]["images"]
                self.assertEqual(len(images),1)
                img=Image.open(io.BytesIO(base64.b64decode(images[0])))
                self.assertEqual(img.format,"JPEG")
                self.assertEqual(img.size,(640,640))
                r,g,b=img.getpixel((200,200))
                self.assertGreater(g,r)
                self.assertGreater(g,b)
                return {"summary":"A sampled frame may have a green background",
                        "events":[{"claim":"green color in the sampled frame",
                                   "kind":"observation","evidence_ids":["v0001"]}],
                        "continuity":"unknown","links_previous_windows":[],
                        "open_questions":["What occurs between frames?"]}
            context=build_context(report,path,model="stub",
                                  window_seconds=2,max_images=1,backend=backend)
            self.assertEqual(len(context["windows"]),1)
            self.assertEqual(context["windows"][0]["events"][0]["evidence_ids"],["v0001"])
