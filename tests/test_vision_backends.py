"""Provider-neutral shot schema and explicit OpenAI API tests."""
import json
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

from true_experience.vision_contract import (
    BATCH_SCHEMA, load_archive, source_digest, selected_shots,
    validate_batch, validate_review,
)
from true_experience.openai_vision_review import (
    request_payload,extract_response,review_archive,
)


def fixture_zip(path):
    shots=[
        {"shot_id":1,"start_seconds":0,"end_seconds":2,
         "frame_time_seconds":1,"image":"shots/shot_0001.jpg"},
        {"shot_id":2,"start_seconds":2,"end_seconds":4,
         "frame_time_seconds":3,"image":"shots/shot_0002.jpg"},
        {"shot_id":3,"start_seconds":4,"end_seconds":6,
         "frame_time_seconds":5,"image":"shots/shot_0003.jpg"}]
    # Minimal magic-byte-valid JPEG for network mock, without image model.
    with zipfile.ZipFile(path,"w") as z:
        z.writestr("manifest.json",json.dumps(
            {"schema_version":"1.0","source_shot_count":3,
             "video_duration_seconds":6,"source_video_name":"test.mp4",
             "shots":shots}))
        for s in shots:
            z.writestr(s["image"],b"\xff\xd8stub-jpeg\xff\xd9")
    return shots


def observations(shots):
    return {"shots":[{"shot_id":s["shot_id"],
                      "visible_description":f"Unverified frame {s['shot_id']}",
                      "visible_objects":["object"],"visible_action":"unknown",
                      "uncertainties":["Cannot infer motion from one frame"]}
                     for s in shots]}


def api_response(shots):
    return {"status":"completed","output":[{"type":"message",
            "content":[{"type":"output_text",
                        "text":json.dumps(observations(shots))}]}],
            "usage":{"input_tokens":200,"output_tokens":80}}


class ContractTests(unittest.TestCase):
    def test_manifest_and_validation(self):
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/"shots.zip"
            shots=fixture_zip(path)
            manifest=load_archive(path)
            self.assertEqual(selected_shots(manifest,2),shots[:2])
            self.assertEqual(len(source_digest(shots)),64)
            self.assertEqual(validate_batch(observations(shots),shots),
                             observations(shots)["shots"])
            bad=observations(shots)
            bad["shots"][0]["shot_id"]=22
            with self.assertRaises(ValueError):
                validate_batch(bad,shots)

    def test_bad_zip_paths_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/"malicious.zip"
            with zipfile.ZipFile(path,"w") as z:
                z.writestr("manifest.json",json.dumps({"shots":[{
                    "shot_id":1,"image":"shots/../secret.jpg",
                    "start_seconds":0,"end_seconds":1,
                    "frame_time_seconds":.5}]}))
                z.writestr("shots/../secret.jpg",b"x")
            with self.assertRaises(ValueError):
                load_archive(path)



class OpenAIReviewTests(unittest.TestCase):
    def test_payload_uses_valid_responses_api_images_and_strict_schema(self):
        shots=fixture_shots()
        payload=request_payload(shots[:2],[b"\xff\xd8a",b"\xff\xd8b"],
                                "gpt-5.4-mini","high")
        self.assertEqual(payload["store"],False)
        self.assertEqual(payload["text"]["format"]["type"],"json_schema")
        self.assertEqual(payload["text"]["format"]["schema"],BATCH_SCHEMA)
        images=payload["input"][1]["content"][1:]
        self.assertEqual(len(images),2)
        self.assertTrue(all(i["image_url"].startswith("data:image/jpeg;base64,") for i in images))
        self.assertNotIn("OPENAI_API_KEY",json.dumps(payload))

    def test_no_api_without_explicit_permission(self):
        with tempfile.TemporaryDirectory() as temp:
            archive=Path(temp)/"a.zip"
            fixture_zip(archive)
            with patch("urllib.request.urlopen",side_effect=AssertionError("network called")):
                result=review_archive(archive,Path(temp)/"review.json",dry_run=True)
                self.assertEqual(result["shots"],3)
                self.assertEqual(result["requests"],1)
                with self.assertRaisesRegex(ValueError,"confirm-upload"):
                    review_archive(archive,Path(temp)/"review.json")

    def test_responds_and_resumes_only_remaining_batches(self):
        with tempfile.TemporaryDirectory() as temp:
            archive=Path(temp)/"shots.zip"
            fixture_zip(archive)
            output=Path(temp)/"review.json"
            calls=[]
            def requester(payload):
                ids=json.loads(payload["input"][1]["content"][0]["text"].split(": ",1)[1])
                batch=[{"shot_id":i["shot_id"]} for i in ids]
                calls.append([s["shot_id"] for s in batch])
                return api_response(batch)
            report=review_archive(archive,output,max_shots=3,batch_size=2,
                                  confirm_upload=True,requester=requester)
            self.assertEqual(calls,[[1,2],[3]])
            self.assertEqual(len(report["shots"]),3)
            self.assertEqual(report["status"],"complete")
            self.assertEqual(len(report["usage_batches"]),2)
            again=review_archive(archive,output,max_shots=3,batch_size=2,
                                 confirm_upload=True,resume=True,requester=requester)
            self.assertEqual(len(again["shots"]),3)
            self.assertEqual(calls,[[1,2],[3]])
            with self.assertRaises(FileExistsError):
                review_archive(archive,output,confirm_upload=True,requester=requester)

    def test_hallucinated_shot_id_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            archive=Path(temp)/"shots.zip"
            fixture_zip(archive)
            def requester(payload):
                return api_response([{"shot_id":999}])
            with self.assertRaises(ValueError):
                review_archive(archive,Path(temp)/"bad.json",max_shots=1,
                               confirm_upload=True,requester=requester)

    def test_refusal_and_incomplete_response_not_mistaken_for_review(self):
        with self.assertRaisesRegex(ValueError,"incomplete"):
            extract_response({"status":"incomplete","output":[]})
        with self.assertRaisesRegex(ValueError,"refused"):
            extract_response({"status":"completed",
                              "output":[{"content":[{"type":"refusal","refusal":"No"}]}]})


def fixture_shots():
    return [{"shot_id":i,"start_seconds":i-1,"end_seconds":i,
             "frame_time_seconds":i-.5,"image":f"shots/shot_{i:04d}.jpg"}
            for i in range(1,4)]


if __name__=="__main__":
    unittest.main()
