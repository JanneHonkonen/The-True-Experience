import unittest
from true_experience.vision_review_validate import compare_reviews

class ComparisonTests(unittest.TestCase):
    def test_disagreement_is_not_accuracy(self):
        a={"shots":[{"shot_id":1,"visible_description":"a red car"},
                    {"shot_id":2,"visible_description":"a blue person"}]}
        b={"shots":[{"shot_id":1,"visible_description":"A red car "},
                    {"shot_id":2,"visible_description":"a standing person"},
                    {"shot_id":3,"visible_description":"street"}]}
        result=compare_reviews(a,b)
        self.assertEqual(result["overlapping_shots"],2)
        self.assertEqual(result["different_wording_count"],1)
        self.assertIsNone(result["accuracy"])
        self.assertEqual(result["different_wording_examples"][0]["shot_id"],2)
