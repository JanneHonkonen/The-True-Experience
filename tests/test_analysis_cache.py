"""Regression for cost-saving PRIVATE ASR/BLIP analysis caching."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from true_experience.local_trailer import main
from true_experience.pipeline import create_shots, build_timeline


def report():
    shots=create_shots([],4.)
    return {"metadata":{"duration_seconds":4.,"has_audio":True,
                        "perception":{"speech_model":"small","caption_model":"blip",
                                      "visual_sampling":{"unsampled_shot_count":0}}},
            "shots":shots,"timeline":build_timeline(4,2,shots,[],True,[]),
            "speech_segments":[{"start_seconds":.1,"end_seconds":1.8,
                                "text":"Private raw dialogue not in public report",
                                "model":"small","source":"faster_whisper"}],
            "visual_observations":[{"frame_time_seconds":2.,"shot_id":1,
                                    "model":"blip","source":"vision_language_model",
                                    "description":"a person"}]}


class AnalysisCacheTests(unittest.TestCase):
    def paths(self,tmp):
        video=Path(tmp)/"original.mp4"
        video.write_bytes(b"fake local video bytes")
        public=Path(tmp)/"public.json"
        private=Path(tmp)/"analysis.json"
        context=Path(tmp)/"context.json"
        base=[str(video),"-o",str(public),"--analysis-cache",str(private),
              "--context-output",str(context),"--context-model","gemma3:4b",
              "--whisper-model","small"]
        return video,public,private,context,base

    def test_private_cache_is_written_and_reused_without_models(self):
        with tempfile.TemporaryDirectory() as td:
            video,public,private,context,base=self.paths(td)
            def fake_context(report,video,**kwargs):
                result={"windows":[],"status":"complete","mock":True}
                kwargs["on_window"](result)
                return result
            with patch("true_experience.local_trailer.analyze",return_value=report()) as analyze, \
                 patch("true_experience.local_trailer.build_context",side_effect=fake_context):
                self.assertEqual(main(base),0)
                analyze.assert_called_once()
            self.assertTrue(private.is_file())
            self.assertTrue(context.is_file())
            self.assertIn("Private raw dialogue",private.read_text())
            self.assertNotIn("Private raw dialogue",public.read_text())
            with patch("true_experience.local_trailer.analyze") as skipped, \
                 patch("true_experience.local_trailer.build_context",side_effect=fake_context) as resumed:
                self.assertEqual(main(base+["--reuse-analysis","--resume-context"]),0)
                skipped.assert_not_called()
                self.assertEqual(resumed.call_args.kwargs["resume"]["mock"],True)

    def test_reject_modified_video_and_changed_inference_parameters(self):
        with tempfile.TemporaryDirectory() as td:
            video,public,private,context,base=self.paths(td)
            with patch("true_experience.local_trailer.analyze",return_value=report()), \
                 patch("true_experience.local_trailer.build_context",
                       return_value={"windows":[],"status":"complete"}):
                self.assertEqual(main(base),0)
            with patch("true_experience.local_trailer.analyze") as skipped:
                self.assertEqual(main(base+["--reuse-analysis","--max-caption-frames","120"]),1)
                video.write_bytes(b"changed video")
                self.assertEqual(main(base+["--reuse-analysis"]),1)
                skipped.assert_not_called()

    def test_resume_requires_cache_and_model(self):
        with tempfile.TemporaryDirectory() as td:
            video,public,private,context,base=self.paths(td)
            self.assertEqual(main(base+["--resume-context"]),1)
            self.assertEqual(main([str(video),"--reuse-analysis"]),1)

    def test_public_output_cannot_be_private_cache(self):
        with tempfile.TemporaryDirectory() as td:
            video,public,private,context,base=self.paths(td)
            self.assertEqual(main(base+["--analysis-cache",str(public)]),1)
