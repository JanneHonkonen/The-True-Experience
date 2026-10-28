import unittest
from true_experience.evaluation import validate_report, word_error_rate
from true_experience.pipeline import build_timeline, create_shots


class ReportEvaluationTests(unittest.TestCase):
    def report(self):
        shots = create_shots([2], 4)
        timeline = build_timeline(4, 2, shots, [], False, [])
        return {"metadata": {"duration_seconds": 4}, "shots": shots, "timeline": timeline,
                "speech_segments": [], "visual_observations": []}

    def test_valid_baseline(self):
        result = validate_report(self.report())
        self.assertTrue(result["valid"], result["errors"])

    def test_bad_shot_gap_and_caption_time(self):
        report = self.report()
        report["shots"][1]["start_seconds"] = 2.3
        report["visual_observations"] = [
            {"shot_id": 1, "frame_time_seconds": 3, "source": "vision_language_model",
             "model": "example", "description": "wrong"}]
        result = validate_report(report)
        self.assertFalse(result["valid"])
        self.assertTrue(any("Gap" in x for x in result["errors"]))
        self.assertTrue(any("outside referenced shot" in x for x in result["errors"]))

    def test_out_of_interval_caption(self):
        report = self.report()
        report["timeline"][0]["visual_observations"] = [{"frame_time_seconds": 3}]
        self.assertFalse(validate_report(report)["valid"])

    def test_wer_equal_after_normalization(self):
        self.assertEqual(word_error_rate("Hei, maailma!", "hei maailma")["wer"], 0)

    def test_wer_substitution(self):
        result = word_error_rate("A blue door", "a red door")
        self.assertAlmostEqual(result["wer"], 1/3, places=5)
        self.assertEqual(result["edit_distance"], 1)

    def test_empty_reference_is_an_error(self):
        with self.assertRaises(ValueError):
            word_error_rate("", "hello")
