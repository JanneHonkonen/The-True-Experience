import unittest

from true_experience.fixture_benchmark import assess_synthetic_fixture


class SyntheticBenchmarkTests(unittest.TestCase):
    def test_known_real_model_output_flags_border_hallucination(self):
        report={"visual_observations":[
            {"frame_time_seconds":5.0,"description":"a red background with a white border"},
            {"frame_time_seconds":15.0,"description":"a blue background with a white border"},
            {"frame_time_seconds":25.0,"description":"a green background with a white border"},
        ]}
        result=assess_synthetic_fixture(report)
        self.assertEqual(result["correct_color_count"],3)
        self.assertEqual(result["unsupported_border_claim_count"],3)
        self.assertEqual(result["wrong_color_word_count"],3) # white added 3 times

    def test_missing_caption_is_not_scored_as_false_hallucination(self):
        report={"visual_observations":[{"frame_time_seconds":5.0,"description":"red"}]}
        result=assess_synthetic_fixture(report)
        self.assertEqual(result["evaluated_caption_count"],1)
        self.assertEqual(result["correct_color_count"],1)
        self.assertEqual(result["unsupported_border_claim_count"],0)
        self.assertIsNone(result["details"][1]["color_mentioned"])

    def test_bad_text_is_rejected(self):
        with self.assertRaises(ValueError):
            assess_synthetic_fixture({"visual_observations":[
                {"frame_time_seconds":5.0,"description":123}]})
