"""Local Ollama vision backend and one-command selector regression tests."""
import json
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

from true_experience.ollama_vision_review import (
    request_payload,extract_response,review_archive,OLLAMA_API)
from true_experience.vision_review import main as unified_cli
from true_experience.vision_contract import validate_review,load_archive,selected_shots


def archive_fixture(path):
    items=[{"shot_id":i,"image":f"shots/shot_{i:04d}.jpg",
            "start_seconds":i-1.,"end_seconds":float(i),
            "frame_time_seconds":i-.5} for i in range(1,4)]
    with zipfile.ZipFile(path,"w") as zipf:
        zipf.writestr("manifest.json",json.dumps({"shots":items,"source_video_name":"test.mp4"}))
        for item in items:
            zipf.writestr(item["image"],b"\xff\xd8fake-jpeg\xff\xd9")
    return items


def response(batch):
    return {"message":{"content":json.dumps({"shots":[
        {"shot_id":s["shot_id"],"visible_description":"A distant figure.",
         "visible_objects":["person"],"visible_action":"unknown",
         "uncertainties":["No movement known"]} for s in batch]})}}


class LocalBackendTests(unittest.TestCase):
    def test_schema_and_images_go_to_local_ollama_only(self):
        shots=[{"shot_id":1,"frame_time_seconds":.5}]
        payload=request_payload(shots,[b"\xff\xd8abcd"],"qwen3-vl:8b")
        self.assertEqual(payload["format"]["required"],["shots"])
        self.assertEqual(len(payload["messages"][1]["images"]),1)
        self.assertEqual(payload["model"],"qwen3-vl:8b")
        self.assertEqual(payload["stream"],False)
        self.assertEqual(OLLAMA_API,"http://127.0.0.1:11434/api/chat")

    def test_review_checkpoint_and_resume(self):
        with tempfile.TemporaryDirectory() as td:
            archive=Path(td)/"shots.zip"
            items=archive_fixture(archive)
            output=Path(td)/"qwen_review.json"
            calls=[]
            def sender(payload):
                data=json.loads(payload["messages"][1]["content"].split(": ",1)[1])
                group=[{"shot_id":s["shot_id"]} for s in data]
                calls.extend(s["shot_id"] for s in group)
                return response(group)
            report=review_archive(archive,output,model="qwen3-vl:8b",
                                  max_shots=2,requester=sender)
            self.assertEqual(calls,[1,2])
            self.assertEqual(report["backend"],"ollama")
            self.assertEqual(report["status"],"complete")
            validate_review(report,items[:2])
            again=review_archive(archive,output,model="qwen3-vl:8b",
                                  max_shots=2,resume=True,requester=sender)
            self.assertEqual(again["shots"],report["shots"])
            self.assertEqual(calls,[1,2])
            with self.assertRaises(ValueError):
                review_archive(archive,output,model="gemma3:12b",
                               max_shots=2,resume=True,requester=sender)

    def test_no_network_for_dry_run(self):
        with tempfile.TemporaryDirectory() as td:
            archive=Path(td)/"shots.zip"
            archive_fixture(archive)
            with patch("true_experience.ollama_vision_review.urllib.request.urlopen",
                       side_effect=AssertionError("No network allowed")):
                plan=review_archive(archive,Path(td)/"review.json",dry_run=True)
                self.assertEqual(plan["shots"],3)
                self.assertEqual(plan["batch_size"],1)
                self.assertEqual(plan["requests"],3)

    def test_wrong_shot_id_fails_and_does_not_save(self):
        with tempfile.TemporaryDirectory() as td:
            archive=Path(td)/"shots.zip"
            archive_fixture(archive)
            out=Path(td)/"wrong.json"
            with self.assertRaisesRegex(ValueError,"Expected shot IDs"):
                review_archive(archive,out,max_shots=1,
                               requester=lambda _:response([{"shot_id":99}]))
            self.assertFalse(out.exists())

    def test_invalid_payload_errors(self):
        with self.assertRaisesRegex(ValueError,"non-JPEG"):
            request_payload([{"shot_id":1,"frame_time_seconds":.5}],[b"not-image"],"qwen3-vl:8b")
        with self.assertRaisesRegex(ValueError,"parseable"):
            extract_response({"message":{"content":"not json"}})
        with self.assertRaises(ValueError):
            review_archive(Path("no_file.zip"),Path("x.json"),dry_run=True)


class CombinedCLITests(unittest.TestCase):
    def test_default_is_local_and_dry_run(self):
        with tempfile.TemporaryDirectory() as td:
            archive=Path(td)/"shots.zip"
            archive_fixture(archive)
            with patch("true_experience.vision_review.local_review",return_value={"dry_run":True,"shots":3}) as local:
                self.assertEqual(unified_cli([str(archive),"--dry-run"]),0)
                self.assertEqual(local.call_args.kwargs["model"],"qwen3-vl:8b")

    def test_openai_always_requires_consent(self):
        with tempfile.TemporaryDirectory() as td:
            archive=Path(td)/"shots.zip"
            archive_fixture(archive)
            with patch("true_experience.vision_review.api_review") as api:
                self.assertEqual(unified_cli([str(archive),"--backend","openai"]),0)
                # Mocked backend is not a realistic consent check, but verifies
                # the CLI forwards explicit False and does not invent consent.
                self.assertFalse(api.call_args.kwargs["confirm_upload"])
            with patch("true_experience.vision_review.local_review") as local:
                self.assertEqual(unified_cli([str(archive),"--confirm-upload"]),1)
                local.assert_not_called()


if __name__=="__main__":
    unittest.main()
