import tempfile
import unittest
from pathlib import Path
from true_experience.pipeline import build_timeline,create_shots,format_time
from true_experience.srt import load_srt

class TimelineTests(unittest.TestCase):
    def test_shots_and_partial_interval(self):
        shots=create_shots([5,2,2,-1,11],10.5)
        self.assertEqual([(s["start_seconds"],s["end_seconds"]) for s in shots],[(0,2),(2,5),(5,10.5)])
        timeline=build_timeline(10.5,5,shots,[(0,2.5),(9.5,10.5)],True,[])
        self.assertEqual(len(timeline),3)
        self.assertEqual(timeline[0]["shot_ids"],[1,2])
        self.assertEqual(timeline[0]["cut_times_seconds"],[2])
        self.assertEqual(timeline[-1]["end"],"00:00:10.500")
        self.assertEqual(timeline[-1]["audio"]["state"],"silent")
        self.assertEqual(timeline[0]["audio"]["state"],"mixed")
        self.assertIsNone(timeline[0]["emotion"])

    def test_overlap_and_missing_audio(self):
        cues=[{"start_seconds":4.8,"end_seconds":5.3,"text":"Hello","source":"subtitle"}]
        timeline=build_timeline(10,5,create_shots([],10),[],False,cues)
        self.assertEqual([len(t["dialogue"]) for t in timeline],[1,1])
        self.assertEqual(timeline[0]["audio"]["state"],"no_audio")

    def test_time_rollover(self):
        self.assertEqual(format_time(3600.9995),"01:00:01.000")

    def test_bad_interval(self):
        with self.assertRaises(ValueError):
            build_timeline(10,0,[],[],False,[])

class SubtitleTests(unittest.TestCase):
    def test_parse(self):
        content="2\n00:00:05,000 --> 00:00:06,500\nGoodbye\n\n1\n00:00:01,200 --> 00:00:02,300\nHello\nworld\n"
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/"test.srt"
            path.write_text(content)
            cues=load_srt(path)
        self.assertEqual([c["text"] for c in cues],["Hello\nworld","Goodbye"])
        self.assertEqual(cues[0]["start_seconds"],1.2)

    def test_invalid_order(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/"bad.srt"
            path.write_text("1\n00:00:02,000 --> 00:00:01,000\nOops")
            with self.assertRaises(ValueError):
                load_srt(path)
