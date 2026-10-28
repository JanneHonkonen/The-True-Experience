import math
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from true_experience.perception import (
    _sample_shots,_extract_image,align_observations,caption_shots,transcribe_audio)
from true_experience.pipeline import analyze,build_timeline,create_shots


class StubSpeech:
    def transcribe(self,path,**kwargs):
        self.called_with=kwargs
        return iter([SimpleNamespace(start=.3,end=2.,text=" Hello "),
                     SimpleNamespace(start=9.,end=15.,text=" End "),
                     SimpleNamespace(start=3.,end=4.,text="  ")]),SimpleNamespace(language="fi")


class PerceptionTests(unittest.TestCase):
    def test_transcribe_alignment_and_provenance(self):
        backend=StubSpeech()
        speech=transcribe_audio(Path("a.mp4"),10.,backend=backend,language=None)
        self.assertEqual(len(speech),2)
        self.assertEqual(speech[0]["text"],"Hello")
        self.assertEqual(speech[0]["language"],"fi")
        self.assertEqual(speech[-1]["end_seconds"],10.)
        self.assertIsNone(speech[0]["speaker"])
        self.assertIsNone(speech[0]["confidence"])
        self.assertTrue(backend.called_with["vad_filter"])

    def test_uniform_sample_and_cap(self):
        shots=create_shots(list(range(1,10)),10)
        self.assertEqual([s["shot_id"] for s in _sample_shots(shots,3)],[1,5,10])
        self.assertEqual(_sample_shots(shots,1)[0]["shot_id"],6)
        with self.assertRaises(ValueError):
            _sample_shots(shots,0)

    def test_caption_provenance_and_timeline(self):
        shots=create_shots([5],10)
        samples=[]
        def loader(path,timestamp):
            samples.append(timestamp)
            return "mock-image"
        def backend(image,**kwargs):
            return [{"generated_text":" a blue door "}]
        visuals=caption_shots(Path("test.mp4"),shots,max_frames=2,
                              backend=backend,frame_loader=loader)
        self.assertEqual(samples,[2.5,7.5])
        self.assertEqual(visuals[0]["description"],"a blue door")
        self.assertIsNone(visuals[0]["confidence"])
        timeline=build_timeline(10,5,shots,[],False,[])
        align_observations(timeline,[],visuals)
        self.assertEqual([len(x["visual_observations"]) for x in timeline],[1,1])
        self.assertEqual(timeline[0]["visual_description"],"a blue door")
        self.assertIsNone(timeline[0]["emotion"])

    def test_no_spill_to_unsampled_intervals(self):
        timeline=build_timeline(12,4,create_shots([],12),[],False,[])
        align_observations(timeline,[],[{"frame_time_seconds":5.5,"description":"cat"}])
        self.assertEqual([len(x["visual_observations"]) for x in timeline],[0,1,0])
        self.assertIsNone(timeline[2]["visual_description"])

    def test_invalid_asr_timing(self):
        class Backend:
            def transcribe(self,*args,**kwargs):
                return [SimpleNamespace(start=math.nan,end=2,text="bad")],SimpleNamespace(language="en")
        with self.assertRaises(RuntimeError):
            transcribe_audio(Path("a.mp4"),10,backend=Backend())


@unittest.skipUnless(shutil.which("ffmpeg") and shutil.which("ffprobe"),"FFmpeg required")
class LiveFFmpegTests(unittest.TestCase):
    def test_decode_real_frame(self):
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/"red.mp4"
            subprocess.run(["ffmpeg","-hide_banner","-v","error","-y",
                            "-f","lavfi","-i","color=c=red:s=64x64:r=10:d=2",
                            "-c:v","mpeg4",str(path)],check=True,capture_output=True)
            try:
                import PIL
            except ImportError:
                self.skipTest("Pillow required for frame extraction test")
            frame=_extract_image(path,1)
            self.assertGreater(frame.getpixel((20,20))[0],180)
            self.assertLess(frame.getpixel((20,20))[2],100)


class IntegrationFlagsTests(unittest.TestCase):
    @unittest.skipUnless(shutil.which("ffmpeg") and shutil.which("ffprobe"),"FFmpeg required")
    def test_model_paths_called_and_aligned(self):
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/"small.mp4"
            subprocess.run(["ffmpeg","-hide_banner","-v","error","-y",
                            "-f","lavfi","-i","color=c=blue:s=64x64:r=10:d=1",
                            "-f","lavfi","-i","anullsrc=r=16000:cl=mono",
                            "-t","1","-c:v","mpeg4","-c:a","aac","-shortest",str(path)],
                           check=True,capture_output=True)
            speech=[{"start_seconds":.1,"end_seconds":.7,"text":"hello","source":"faster_whisper"}]
            visuals=[{"frame_time_seconds":.5,"description":"blue screen",
                      "shot_id":1,"source":"vision_language_model"}]
            with patch("true_experience.pipeline.transcribe_audio",return_value=speech) as stt,\
                 patch("true_experience.pipeline.caption_shots",return_value=visuals) as caption:
                result=analyze(path,interval=.5,transcribe=True,caption_frames=True)
            stt.assert_called_once()
            caption.assert_called_once()
            self.assertEqual(len(result["speech_segments"]),1)
            self.assertEqual(result["timeline"][0]["speech"][0]["text"],"hello")
            self.assertEqual(len(result["timeline"][1]["visual_observations"]),1)
            self.assertIsNone(result["timeline"][1]["emotion"])
            self.assertIn("faster_whisper",result["metadata"]["methods"])
            self.assertIn("blip_frame_captioning",result["metadata"]["methods"])

    @unittest.skipUnless(shutil.which("ffmpeg") and shutil.which("ffprobe"),"FFmpeg required")
    def test_no_audio_rejects_transcription(self):
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/"noaudio.mp4"
            subprocess.run(["ffmpeg","-hide_banner","-v","error","-y","-f","lavfi",
                            "-i","color=c=green:s=64x64:r=10:d=1","-c:v","mpeg4",
                            str(path)],check=True,capture_output=True)
            with self.assertRaisesRegex(ValueError,"no audio stream"):
                analyze(path,transcribe=True)
