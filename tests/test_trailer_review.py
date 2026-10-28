import unittest
from true_experience.trailer_review import make_review
from true_experience.pipeline import create_shots, build_timeline


def fixture():
    shots=create_shots([5],10)
    timeline=build_timeline(10,5,shots,[],True,[])
    return {"metadata":{
        "duration_seconds":10,"has_audio":True,"frame_rate":25,
        "perception":{"speech_model":"tiny","caption_model":"example",
                      "visual_sampling":{"unsampled_shot_count":0}}},
        "shots":shots,"timeline":timeline,
        "speech_segments":[{"start_seconds":1,"end_seconds":2.5,
                            "text":"Film script should not leak",
                            "model":"tiny","source":"faster_whisper"}],
        "visual_observations":[{"frame_time_seconds":2.5,"shot_id":1,
                               "description":"a shiny robot","model":"example",
                               "source":"vision_language_model"},
                               {"frame_time_seconds":7.5,"shot_id":2,
                               "description":"a person in a room","model":"example",
                               "source":"vision_language_model"}]}


class ReportTests(unittest.TestCase):
    def test_counts_source_links_and_sanitization(self):
        report=make_review(fixture(),"8ugaeA-nMTc")
        self.assertEqual(report["structure"]["shots_detected"],2)
        self.assertEqual(report["speech"]["word_count"],5)
        self.assertIsNone(report["speech"]["word_error_rate"])
        self.assertEqual(report["visual"]["total_captions"],2)
        self.assertTrue(report["visual"]["caption_hypotheses"][0]["view_source_at_time"].endswith("&t=2s"))
        import json
        result=json.dumps(report)
        self.assertNotIn("Film script should not leak",result)
        self.assertNotIn("speech_segments",result)
        self.assertNotIn("frame_pixels",result)

    def test_invalid_id(self):
        for value in ("https://youtube.com/x","bad/id","x"*100,""):
            with self.assertRaises(ValueError):
                make_review(fixture(),value)

    def test_invalid_structure_fails(self):
        broken=fixture()
        broken["timeline"]=[]
        with self.assertRaisesRegex(ValueError,"Invalid analysis"):
            make_review(broken,"8ugaeA-nMTc")

    def test_capped_caption_count_is_disclosed(self):
        report=make_review(fixture(),"8ugaeA-nMTc",max_captions=1)
        self.assertEqual(report["visual"]["total_captions"],2)
        self.assertEqual(report["visual"]["caption_examples_in_report"],1)
