import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from true_experience.local_trailer import main
from true_experience.pipeline import create_shots, build_timeline


class LocalEvalTests(unittest.TestCase):
    def test_copyright_safe_report_with_model_output_mocked(self):
        shots=create_shots([],4)
        timeline=build_timeline(4,2,shots,[],True,[])
        raw={
            "metadata":{"duration_seconds":4,"has_audio":True,"frame_rate":25,
                        "perception":{"speech_model":"tiny","caption_model":"blip",
                                      "visual_sampling":{"unsampled_shot_count":0}}},
            "shots":shots,"timeline":timeline,
            "speech_segments":[{"start_seconds":.2,"end_seconds":1,
                                "text":"Copyrighted full dialogue should stay private",
                                "model":"tiny","source":"faster_whisper"}],
            "visual_observations":[{"frame_time_seconds":2,"shot_id":1,
                                    "description":"a gray suit","model":"blip",
                                    "source":"vision_language_model"}]}
        with tempfile.TemporaryDirectory() as td:
            video=Path(td)/"clip.mp4"
            video.write_bytes(b"stub")
            output=Path(td)/"review.json"
            with patch("true_experience.local_trailer.analyze",return_value=raw) as model:
                status=main([str(video),"-o",str(output)])
            self.assertEqual(status,0)
            self.assertTrue(model.called)
            text=output.read_text(encoding="utf-8")
            report=json.loads(text)
            self.assertNotIn("Copyrighted full dialogue",text)
            self.assertNotIn("speech_segments",text)
            self.assertEqual(report["visual"]["total_captions"],1)
            self.assertEqual(report["speech"]["segment_count"],1)

    def test_missing_video_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            self.assertEqual(main([str(Path(td)/"missing.mp4")]),1)
