"""Character names from dialogue and scene readings; one-line profiles."""
import unittest

from true_experience.naming import add_profiles, clean_name, group_names, name_characters, profile
from true_experience.story import validate_scene


class NamingTests(unittest.TestCase):
    def test_clean_name(self):
        self.assertEqual(clean_name("Mr. Stark"), "Stark")
        self.assertEqual(clean_name("Dr. jane foster"), "Jane Foster")
        self.assertEqual(clean_name("Sorry"), "")
        self.assertEqual(clean_name("Mr."), "")

    def test_forms_of_one_name_merge_under_the_fullest(self):
        groups = group_names([{"name": n} for n in ("Stark", "Tony Stark", "Tony", "Rhodey")])
        self.assertEqual(sorted(c for c, _ in groups), ["Rhodey", "Tony Stark"])

    def test_names_from_transcript_and_scenes(self):
        chars = [{"id": "C1"}, {"id": "C2"}, {"id": "C3"}]
        speech = [{"start_seconds": 1, "end_seconds": 3, "text": "I'm Tony Stark, and I build things.",
                   "character": "C1"},
                  {"start_seconds": 4, "end_seconds": 5, "text": "I'm sorry.", "character": "C2"}]
        scenes = [{"scene_id": "SC1", "start_seconds": 0, "reading": {"name_mentions": [
                      {"name": "Mr. Stark", "refers_to": "C1", "cue": "addressed"},
                      {"name": "Christine", "refers_to": "C2", "cue": "mentioned_present"},
                      {"name": "Obadiah", "refers_to": "absent", "cue": "mentioned_absent"}]}},
                  {"scene_id": "SC2", "start_seconds": 9, "reading": {"name_mentions": [
                      {"name": "Christine Everhart", "refers_to": "C2", "cue": "self_introduction"},
                      {"name": "Stark", "refers_to": "C3", "cue": "mentioned_present"}]}}]
        unseen, meta = name_characters(chars, speech, scenes)
        self.assertEqual(chars[0]["name"]["name"], "Tony Stark")
        self.assertIn("Mr. Stark", chars[0]["name"]["also_called"])
        self.assertEqual(chars[1]["name"]["name"], "Christine Everhart")
        self.assertIsNone(chars[2]["name"])          # one weak vote, and the name is taken
        self.assertEqual([u["name"] for u in unseen], ["Obadiah"])
        self.assertEqual(meta["named"], 2)
        self.assertEqual(meta["sources"], ["scene_reading", "transcript"])

    def test_single_passing_mention_is_not_enough(self):
        chars = [{"id": "C1"}]
        scenes = [{"scene_id": "SC1", "reading": {"name_mentions": [
            {"name": "Bob", "refers_to": "C1", "cue": "mentioned_present"}]}}]
        name_characters(chars, [], scenes)
        self.assertIsNone(chars[0]["name"])

    def test_scene_names_must_be_heard(self):
        out = {"summary": "s", "location": "l", "time_of_day": "day", "scene_type": "dialogue",
               "story_beat": "setup", "emotional_turn": "t", "character_actions": [], "confidence": 0.5,
               "name_mentions": [{"name": "Mr. Stark", "refers_to": "C1", "cue": "addressed"},
                                 {"name": "Pepper", "refers_to": "C2", "cue": "mentioned_present"},
                                 {"name": "Stark Expo", "refers_to": "C9", "cue": "on_screen_text"}]}
        r = validate_scene(out, {"C1", "C2"}, "Mr. Stark, a question.")
        self.assertEqual(r["name_mentions"], [
            {"name": "Mr. Stark", "refers_to": "C1", "cue": "addressed"},
            {"name": "Stark Expo", "refers_to": "unknown", "cue": "on_screen_text"}])

    def test_profile_sentence(self):
        c = {"id": "C1", "name": {"name": "Tony Stark", "also_called": ["Mr. Stark"]},
             "role": {"narrative_role": "protagonist", "archetypes": ["hero", "business_person"],
                      "appearance": "Dark suit and tinted glasses.",
                      "physical": {"apparent_gender": "man", "apparent_age_range": "40s",
                                   "build": "slim, fit", "hair": "short dark hair",
                                   "facial_hair": "Van Dyke beard", "distinctive_features": []}}}
        self.assertEqual(profile(c), 'Man named Tony Stark ("Mr. Stark"); appears to be in his 40s; '
                                     'slim, fit; short dark hair; Van Dyke beard; '
                                     'dark suit and tinted glasses; protagonist (hero, business person).')
        masked = {"id": "C6", "role": {"narrative_role": "unclear", "archetypes": ["unclear"],
                                       "physical": {"apparent_gender": "unclear", "apparent_age_range": "unclear",
                                                    "build": "unclear", "hair": "unclear",
                                                    "facial_hair": "unclear",
                                                    "distinctive_features": ["full metal armour"]}}}
        self.assertEqual(profile(masked), "Person; full metal armour.")
        bare = [{"id": "C9"}, {"id": "C8", "background": True}]
        add_profiles(bare)
        self.assertIsNone(bare[0]["profile"])
        self.assertNotIn("profile", bare[1])


if __name__ == "__main__":
    unittest.main()
