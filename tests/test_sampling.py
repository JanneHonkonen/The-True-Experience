import unittest
from true_experience.sampling import plan_frame_samples, describe_coverage


def shots(*durations):
    t = 0
    result = []
    for i, duration in enumerate(durations,1):
        result.append({"shot_id": i,"start_seconds":t,"end_seconds":t+duration})
        t += duration
    return result


class SamplePlannerTests(unittest.TestCase):
    def test_original_synthetic_fixture_unchanged(self):
        plan = plan_frame_samples(shots(10,10,10), 3, 10)
        self.assertEqual([x['frame_time_seconds'] for x in plan],[5,15,25])

    def test_long_continuous_shot_gets_multiple_samples(self):
        plan = plan_frame_samples(shots(35), 20, 10)
        self.assertEqual([round(x['frame_time_seconds'],3) for x in plan],
                         [4.375, 13.125, 21.875, 30.625])
        coverage = describe_coverage(shots(35), plan)
        self.assertAlmostEqual(coverage['largest_gap_between_sample_points_seconds'],8.75)

    def test_strict_budget_and_balanced_allocation(self):
        plan = plan_frame_samples(shots(25,5,20), 4, 5)
        self.assertEqual(len(plan),4)
        self.assertEqual({x['shot_id'] for x in plan},{1,2,3})
        self.assertEqual(len([x for x in plan if x['shot_id']==1]),2)
        self.assertTrue(all(x['frame_time_seconds'] < 50 for x in plan))

    def test_many_shots_covered_but_gap_disclosed(self):
        timeline=shots(*([2]*10))
        plan = plan_frame_samples(timeline,3,10)
        self.assertEqual([p['shot_id'] for p in plan],[1,5,10])
        stats=describe_coverage(timeline,plan)
        self.assertEqual(stats['unsampled_shot_count'],7)
        self.assertEqual(stats['unsampled_shot_duration_seconds'],14)

    def test_single_budget_picks_middle_shot(self):
        plan=plan_frame_samples(shots(*([1]*9)),1,10)
        self.assertEqual(len(plan),1)
        self.assertEqual(plan[0]['shot_id'],5)

    def test_tiny_and_invalid_intervals(self):
        self.assertEqual(len(plan_frame_samples(shots(.001),1,5)),1)
        for bad in (0,-1,float('nan'),float('inf')):
            with self.assertRaises(ValueError):
                plan_frame_samples(shots(10),1,bad)
        with self.assertRaises(ValueError):
            plan_frame_samples(shots(1),True,5)
        with self.assertRaises(ValueError):
            plan_frame_samples(shots(1),0,5)
        with self.assertRaises(ValueError):
            plan_frame_samples([{"shot_id":1,"start_seconds":0,"end_seconds":0}],10,5)

    def test_no_shots(self):
        self.assertEqual(plan_frame_samples([],3,10),[])
        self.assertEqual(describe_coverage([],[])['unsampled_shot_count'],0)


if __name__ == '__main__':
    unittest.main()
