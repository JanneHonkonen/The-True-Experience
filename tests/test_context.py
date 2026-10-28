import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from true_experience.context import (
    build_context, choose_frames, make_windows, validate_hypothesis, local_ollama)
from true_experience.pipeline import create_shots, build_timeline
from true_experience.local_trailer import main


def analysis():
    shots = create_shots([6, 12], 24)
    visual = [
        {"frame_time_seconds": 3., "shot_id": 1, "description": "truck",
         "source": "vision_language_model", "model": "blip"},
        {"frame_time_seconds": 9., "shot_id": 2, "description": "a man",
         "source": "vision_language_model", "model": "blip"},
        {"frame_time_seconds": 15., "shot_id": 3, "description": "suit",
         "source": "vision_language_model", "model": "blip"},
        {"frame_time_seconds": 21., "shot_id": 3, "description": "helmet",
         "source": "vision_language_model", "model": "blip"},
    ]
    speech = [
        {"start_seconds": 4, "end_seconds": 6, "text": "We need to move.",
         "source": "faster_whisper", "model": "tiny"},
        {"start_seconds": 14, "end_seconds": 16, "text": "That is our chance.",
         "source": "faster_whisper", "model": "tiny"},
    ]
    return {"metadata": {"duration_seconds": 24, "has_audio": True,
                         "perception": {"caption_model": "blip", "speech_model": "tiny",
                                        "visual_sampling": {"unsampled_shot_count": 0}}},
            "shots": shots, "timeline": build_timeline(24, 6, shots, [], True, []),
            "visual_observations": visual, "speech_segments": speech}


def response(current,previous):
    return {"summary": "Unverified context hypothesis",
            "events": [{"claim": "A person may be visible", "kind": "observation",
                        "evidence_ids": [current]}] if current else [],
            "continuity": "Possible change; no identity established",
            "links_previous_windows": previous,
            "open_questions": ["Is this the same person?"]}


class FakeImage:
    def save(self, destination, format, quality):
        destination.write(b"not-a-real-jpeg-but-encoding-exercised")


class WindowTests(unittest.TestCase):
    def test_window_alignment(self):
        windows=make_windows(analysis(), 12)
        self.assertEqual(len(windows),2)
        self.assertEqual([x["id"] for x in windows[0]["frames"]],["v0001","v0002"])
        self.assertEqual([x["id"] for x in windows[1]["frames"]],["v0003","v0004"])
        self.assertEqual([x["id"] for x in windows[0]["speech"]],["a0001"])
        self.assertEqual([x["id"] for x in windows[1]["speech"]],["a0002"])
        self.assertEqual(windows[0]["window_id"],"w0001")

    def test_validated_chain_and_no_film_title_prior(self):
        inputs=[]
        checkpoints=[]
        def backend(messages, model, timeout):
            data=json.loads(messages[1]["content"])
            inputs.append((messages, data))
            self.assertNotIn("Iron Man",messages[0]["content"])
            self.assertEqual(len(messages[1]["images"]),2)
            refs=[x["id"] for x in data["current_evidence"]]
            prior=[x["window_id"] for x in data["unverified_previous_memory"]]
            return response(refs[0],prior)
        result=build_context(analysis(),Path("clip.mp4"),
                             model="test-vision-model",window_seconds=12,max_images=2,
                             backend=backend,frame_loader=lambda p,t:FakeImage(),
                             on_window=lambda r:checkpoints.append(len(r["windows"])))
        self.assertEqual(len(result["windows"]),2)
        self.assertEqual(result["status"],"complete")
        self.assertEqual(checkpoints,[1,2])
        self.assertEqual(result["windows"][1]["links_previous_windows"],["w0001"])
        self.assertEqual(inputs[1][1]["unverified_previous_memory"][0]["summary_hypothesis"],
                         "Unverified context hypothesis")
        self.assertTrue(all(x["human_groundedness_review"]=="not_reviewed"
                            for window in result["windows"] for x in window["events"]))
        self.assertNotIn("We need to move",json.dumps(result))
        self.assertEqual(result["windows"][0]["validation"],"schema_and_citations_only")

    def test_spoken_words_are_not_promoted_to_fact(self):
        def backend(messages, model, timeout):
            data=json.loads(messages[1]["content"])
            self.assertEqual(data["current_evidence"][-1]["asr_hypothesis"],"We need to move.")
            return response("a0001",[])
        first={**analysis()}
        first["metadata"]={**first["metadata"],"duration_seconds":12}
        first["shots"]=create_shots([6],12)
        first["speech_segments"]=[first["speech_segments"][0]]
        first["visual_observations"]=[x for x in first["visual_observations"] if x["frame_time_seconds"]<12]
        result=build_context(first,Path("clip.mp4"),backend=backend,
                             frame_loader=lambda p,t:FakeImage())
        self.assertEqual(result["windows"][0]["events"][0]["evidence_ids"],["a0001"])

    def test_reject_fabricated_evidence_and_prior(self):
        good=response("v0001",[])
        good["events"][0]["evidence_ids"]=["v9999"]
        with self.assertRaisesRegex(ValueError,"cite only existing"):
            validate_hypothesis(good,{"v0001"},set())
        bad=response("v0001",["w9999"])
        with self.assertRaisesRegex(ValueError,"previous windows"):
            validate_hypothesis(bad,{"v0001"},set())

    def test_invalid_model_schema(self):
        with self.assertRaises(ValueError):
            validate_hypothesis({"summary":"Fine","events":[],
                                 "continuity":"","links_previous_windows":[],
                                 "open_questions":"unknown"},set(),set())
        with self.assertRaises(ValueError):
            validate_hypothesis({"summary":"Fine","events":[{"claim":"?", "kind":"fact",
                                  "evidence_ids":["v0001"]}],
                                 "continuity":"","links_previous_windows":[],
                                 "open_questions":[]},{"v0001"},set())

    def test_frame_limit_budget(self):
        frames=list(range(12))
        self.assertEqual(choose_frames(frames,3),[0,6,11])
        self.assertEqual(choose_frames(frames,1),[6])
        with self.assertRaises(ValueError):
            choose_frames(frames,9)
        with self.assertRaises(ValueError):
            make_windows(analysis(),0)

    def test_offline_backend_never_uses_a_remote_endpoint(self):
        self.assertEqual(local_ollama.__defaults__, (300,))


    def test_empty_window_abstains_without_model_call(self):
        raw=analysis()
        raw["visual_observations"]=[]
        raw["speech_segments"]=[]
        def forbidden(*args):
            raise AssertionError("Model should not invent missing evidence")
        result=build_context(raw,Path("clip.mp4"),backend=forbidden,
                             frame_loader=lambda p,t:FakeImage())
        self.assertEqual(len(result["windows"]),2)
        self.assertTrue(all(not w["events"] for w in result["windows"]))
        self.assertTrue(all(w["summary_hypothesis"].startswith("No sampled")
                            for w in result["windows"]))

    def test_bad_model_citations_retried_once(self):
        attempts=[]
        def backend(messages,model,timeout):
            attempts.append(len(messages))
            data=json.loads(messages[1]["content"])
            if len(attempts)==1:
                return response("v9999",[])
            ids=[e["id"] for e in data["current_evidence"]]
            return response(ids[0],[])
        raw=analysis()
        raw["metadata"]={**raw["metadata"],"duration_seconds":12}
        raw["shots"]=create_shots([6],12)
        raw["visual_observations"]=raw["visual_observations"][:2]
        raw["speech_segments"]=raw["speech_segments"][:1]
        result=build_context(raw,Path("clip.mp4"),backend=backend,
                             frame_loader=lambda p,t:FakeImage())
        self.assertEqual(attempts,[2,4])
        self.assertEqual(len(result["windows"][0]["events"]),1)


class LocalTrailerContextTests(unittest.TestCase):
    def test_optional_context_file_and_public_report_remains_separate(self):
        a=analysis()
        with tempfile.TemporaryDirectory() as tmp:
            video=Path(tmp)/"test.mp4"
            video.write_bytes(b"mock")
            public=Path(tmp)/"review.json"
            private=Path(tmp)/"context-private.json"
            def produce_context(report, source, **kwargs):
                result={"windows":[{"summary_hypothesis":"Unverified person",
                                    "evidence_ids":["v0001"],
                                    "window_id":"w0001"}], "status":"complete"}
                kwargs["on_window"](result)
                return result
            with patch("true_experience.local_trailer.analyze",return_value=a), \
                 patch("true_experience.local_trailer.build_context",side_effect=produce_context):
                status=main([str(video),"-o",str(public),"--context-model","gemma3:4b",
                             "--context-output",str(private)])
            self.assertEqual(status,0)
            self.assertTrue(public.is_file())
            self.assertTrue(private.is_file())
            self.assertNotIn("summary_hypothesis",public.read_text())
            self.assertIn("summary_hypothesis",private.read_text())
            self.assertNotIn("We need to move.",private.read_text())

    def test_context_disabled_does_not_call_model(self):
        with tempfile.TemporaryDirectory() as tmp:
            video=Path(tmp)/"test.mp4"
            video.write_bytes(b"mock")
            with patch("true_experience.local_trailer.analyze",return_value=analysis()), \
                 patch("true_experience.local_trailer.build_context") as context:
                self.assertEqual(main([str(video),"-o",str(Path(tmp)/"public.json")]),0)
                context.assert_not_called()
