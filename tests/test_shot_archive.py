"""Timestamped archive regression tests (no movie or pretrained models needed)."""
import io
import json
import shutil
import subprocess
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch
from PIL import Image

from true_experience.shot_archive import (
    create_archive, timestamp_name, _read_analysis, main as archive_main)
from true_experience.pipeline import build_timeline, create_shots
from true_experience.local_trailer import main as trailer_main


def fake_report():
    shots=create_shots([1],2)
    return {
        "metadata":{"duration_seconds":2.0,"has_audio":False,
                    "perception":{"speech_model":None,"caption_model":"blip",
                                  "visual_sampling":{}}},
        "shots":shots,
        "timeline":build_timeline(2,1,shots,[],False,[]),
        "speech_segments":[],
        "visual_observations":[
            {"shot_id":1,"frame_time_seconds":.4,"description":"red",
             "model":"stub","source":"vision_language_model"},
            {"shot_id":2,"frame_time_seconds":1.6,"description":"blue",
             "model":"stub","source":"vision_language_model"}]
    }


def solid_image(path,t):
    color = "red" if t < 1.0 else "blue"
    return Image.new("RGB",(64,48),color=color)


class ArchiveTests(unittest.TestCase):
    def test_timestamps_win_safe_and_precise(self):
        self.assertEqual(timestamp_name(0),"00-00-00.000000")
        self.assertEqual(timestamp_name(60.000001),"00-01-00.000001")
        self.assertEqual(timestamp_name(3600.123456),"01-00-00.123456")
        self.assertNotIn(":",timestamp_name(123.45))
        with self.assertRaises(ValueError):
            timestamp_name(-1)

    def test_zip_manifest_links_json_shots_and_context_evidence(self):
        with tempfile.TemporaryDirectory() as temp:
            video=Path(temp)/"clip.mp4"
            video.write_bytes(b"stub (loader is mocked)")
            archive=Path(temp)/"clip_shots.zip"
            context={"windows":[
                {"window_id":"w0001","evidence_ids":["v0001","a0001"]},
                {"window_id":"w0002","evidence_ids":["v0002"]}]}
            m=create_archive(video,fake_report(),archive,
                             frame_loader=solid_image,context=context)
            self.assertEqual(m["shot_count"],2)
            self.assertEqual(m["visual_sample_count"],2)
            self.assertEqual(m["context_links_status"],"actual_context_evidence_ids")
            self.assertEqual(m["visual_evidence"][0]["context_windows"],["w0001"])
            self.assertEqual(m["visual_evidence"][1]["context_windows"],["w0002"])
            self.assertEqual([x["visual_evidence_ids"] for x in m["shots"]],
                             [["v0001"],["v0002"]])
            with zipfile.ZipFile(archive) as z:
                names=z.namelist()
                self.assertEqual(len(names),6)  # 2 shots + 2 evidence + manifest + readme
                self.assertTrue(all(":" not in name for name in names))
                self.assertTrue(all(".." not in name for name in names))
                manifest=json.loads(z.read("manifest.json"))
                self.assertEqual(manifest["shots"][0]["frame_time_seconds"],.5)
                self.assertEqual(manifest["visual_evidence"][0]["frame_time_seconds"],.4)
                self.assertFalse(manifest["privacy"]["contains_raw_asr_dialogue"])
                for shot in manifest["shots"]:
                    self.assertIn(shot["image"],names)
                for evidence in manifest["visual_evidence"]:
                    self.assertIn(evidence["image"],names)
                im=Image.open(io.BytesIO(z.read(manifest["visual_evidence"][1]["image"])))
                r,g,b=im.getpixel((20,20))
                self.assertGreater(b,r)
                self.assertEqual(im.size,(64,48))

    def test_planned_context_windows_are_labeled_unverified(self):
        with tempfile.TemporaryDirectory() as temp:
            video=Path(temp)/"clip.mp4";video.write_bytes(b"x")
            archive=Path(temp)/"out.zip"
            m=create_archive(video,fake_report(),archive,frame_loader=solid_image,
                             context_plan=(1.0,3))
            self.assertEqual(m["context_links_status"],
                             "planned_context_frames_not_verified_run")
            self.assertEqual(m["visual_evidence"][0]["context_windows"],["w0001"])
            self.assertEqual(m["visual_evidence"][1]["context_windows"],["w0002"])
            with self.assertRaises(ValueError):
                create_archive(video,fake_report(),archive,context={},context_plan=(1,3))

    def test_cache_requires_matching_original_video_file(self):
        with tempfile.TemporaryDirectory() as td:
            video=Path(td)/"video.mp4";video.write_bytes(b"video")
            stat=video.resolve().stat()
            cache=Path(td)/"analysis-private.json"
            data={"signature":{"absolute_video_path":str(video.resolve()),
                               "video_bytes":stat.st_size,"modified_ns":stat.st_mtime_ns},
                  "report":fake_report()}
            cache.write_text(json.dumps(data))
            self.assertEqual(_read_analysis(cache,video)["shots"],fake_report()["shots"])
            video.write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError,"does not match"):
                _read_analysis(cache,video)

    def test_noninteractive_no_prompt_and_cli_explicit_flag(self):
        with tempfile.TemporaryDirectory() as td:
            video=Path(td)/"video.mp4";video.write_bytes(b"x")
            public=Path(td)/"report.json"
            with patch("true_experience.local_trailer.analyze",return_value=fake_report()), \
                 patch("true_experience.local_trailer.create_archive") as create, \
                 patch("true_experience.local_trailer.sys.stdin.isatty",return_value=False), \
                 patch("builtins.input",side_effect=AssertionError("must not prompt")):
                self.assertEqual(trailer_main([str(video),"-o",str(public)]),0)
                create.assert_not_called()
                self.assertEqual(trailer_main([str(video),"-o",str(public),
                                               "--thumbnails"]),0)
                create.assert_called_once()
                self.assertEqual(create.call_args.args[2],
                                 public.with_name("report_shots.zip"))

    def test_yes_no_prompt(self):
        with tempfile.TemporaryDirectory() as td:
            video=Path(td)/"video.mp4";video.write_bytes(b"x")
            public=Path(td)/"out.json"
            with patch("true_experience.local_trailer.analyze",return_value=fake_report()), \
                 patch("true_experience.local_trailer.create_archive") as create, \
                 patch("true_experience.local_trailer.sys.stdin.isatty",return_value=True), \
                 patch("builtins.input",side_effect=["n","YES"]):
                self.assertEqual(trailer_main([str(video),"-o",str(public)]),0)
                create.assert_not_called()
                self.assertEqual(trailer_main([str(video),"-o",str(public)]),0)
                create.assert_called_once()

    def test_standalone_cli_uses_cache_not_models(self):
        with tempfile.TemporaryDirectory() as td:
            video=Path(td)/"clip.mp4";video.write_bytes(b"x")
            stat=video.resolve().stat()
            analysis=Path(td)/"analysis.json"
            analysis.write_text(json.dumps({"signature":{
                "absolute_video_path":str(video.resolve()),
                "video_bytes":stat.st_size,"modified_ns":stat.st_mtime_ns},
                "report":fake_report()}),encoding="utf-8")
            zip_path=Path(td)/"archive.zip"
            with patch("true_experience.shot_archive._extract_image",side_effect=solid_image):
                self.assertEqual(archive_main([str(video),"--analysis",str(analysis),
                                               "-o",str(zip_path)]),0)
            with zipfile.ZipFile(zip_path) as z:
                m=json.loads(z.read("manifest.json"))
            self.assertEqual(len(m["shots"]),2)
            self.assertEqual(len(m["visual_evidence"]),2)

    def test_invalid_report_does_not_leave_partial_archive(self):
        with tempfile.TemporaryDirectory() as td:
            video=Path(td)/"clip.mp4";video.write_bytes(b"x")
            archive=Path(td)/"failed.zip"
            broken=fake_report()
            broken["timeline"]=[]
            with self.assertRaises(ValueError):
                create_archive(video,broken,archive,frame_loader=solid_image)
            self.assertFalse(archive.exists())
            self.assertFalse(Path(str(archive)+".tmp").exists())


@unittest.skipUnless(shutil.which("ffmpeg") and shutil.which("ffprobe"),"FFmpeg required")
class ArchiveFFmpegTests(unittest.TestCase):
    def test_real_decoded_video_shot_and_evidence_frames(self):
        with tempfile.TemporaryDirectory() as td:
            video=Path(td)/"two.mp4"
            subprocess.run([
                "ffmpeg","-hide_banner","-nostdin","-v","error","-y",
                "-f","lavfi","-i","color=c=red:s=80x60:r=10:d=1",
                "-f","lavfi","-i","color=c=blue:s=80x60:r=10:d=1",
                "-filter_complex","[0:v][1:v]concat=n=2:v=1:a=0[v]",
                "-map","[v]","-c:v","mpeg4",str(video)],
                capture_output=True,check=True)
            archive=Path(td)/"test.zip"
            manifest=create_archive(video,fake_report(),archive)
            with zipfile.ZipFile(archive) as z:
                for i,expected in enumerate(("red","blue")):
                    path=manifest["visual_evidence"][i]["image"]
                    im=Image.open(io.BytesIO(z.read(path)))
                    r,g,b=im.getpixel((100,100))
                    if expected=="red":
                        self.assertGreater(r,170)
                        self.assertLess(b,80)
                    else:
                        self.assertGreater(b,170)
                        self.assertLess(r,80)


if __name__=="__main__":
    unittest.main()
