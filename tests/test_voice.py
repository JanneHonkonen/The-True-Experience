"""Voice stage: prosody, speaker grouping, emotion mapping, streaming and fusion."""
import shutil
import subprocess
import tempfile
import unittest
import wave
from pathlib import Path

try:
    import numpy as np
except ImportError:  # pragma: no cover
    np = None

RATE = 16000


def voice(f0, seconds, *, vibrato=0.0, amp=0.3, seed=0):
    """Harmonic 'voice': fundamental with harmonics and optional pitch movement."""
    t = np.arange(int(seconds * RATE)) / RATE
    freq = f0 * (1 + vibrato * np.sin(2 * np.pi * 1.5 * t))
    phase = 2 * np.pi * np.cumsum(freq) / RATE
    rng = np.random.default_rng(seed)
    sig = sum(np.sin(k * phase) / k for k in range(1, 8)) + 0.01 * rng.normal(size=len(t))
    return (amp * sig / np.max(np.abs(sig))).astype(np.float32)


@unittest.skipIf(np is None, "NumPy not installed")
class ProsodyTests(unittest.TestCase):
    def test_pitch_level(self):
        from true_experience.voice import prosody
        low = prosody(voice(110, 2), "a b c d")
        high = prosody(voice(220, 2), "a b c d")
        self.assertAlmostEqual(low["pitch_median_hz"], 110, delta=6)
        self.assertAlmostEqual(high["pitch_median_hz"], 220, delta=10)
        self.assertEqual(low["words_per_second"], 2.0)

    def test_animated_voice_has_more_pitch_variability(self):
        from true_experience.voice import prosody
        flat = prosody(voice(150, 3), "x")
        lively = prosody(voice(150, 3, vibrato=0.25), "x")
        self.assertGreater(lively["pitch_variability_semitones"],
                           flat["pitch_variability_semitones"] + 1.5)

    def test_silence_is_unvoiced(self):
        from true_experience.voice import prosody
        quiet = prosody(np.zeros(RATE, np.float32), "")
        self.assertIsNone(quiet["pitch_median_hz"])
        self.assertEqual(quiet["voiced_share"], 0.0)


@unittest.skipIf(np is None, "NumPy not installed")
class SpeakerTests(unittest.TestCase):
    def test_two_voices_form_two_groups(self):
        from true_experience.voice import voice_fingerprint, cluster_speakers
        clips = [(voice(105, 1.5, seed=i), 105) for i in range(3)] + \
                [(voice(230, 1.5, seed=10 + i), 230) for i in range(3)]
        prints = [voice_fingerprint(c, pitch_hz=p) for c, p in clips]
        labels = cluster_speakers(prints)
        self.assertEqual(labels[:3], ["S1"] * 3)
        self.assertEqual(labels[3:], ["S2"] * 3)

    def test_single_and_empty(self):
        from true_experience.voice import cluster_speakers
        self.assertEqual(cluster_speakers([]), [])
        self.assertEqual(cluster_speakers([np.ones(25)]), ["S1"])


class EmotionTests(unittest.TestCase):
    def test_reading_from_scores(self):
        from true_experience.voice import emotion_reading
        r = emotion_reading([{"label": "ang", "score": 0.8}, {"label": "neu", "score": 0.2}])
        self.assertEqual(r["label"], "ang")
        self.assertLess(r["valence"], -0.4)
        self.assertGreater(r["arousal"], 0.5)
        self.assertIsNone(emotion_reading([{"label": "weird", "score": 1.0}]))


@unittest.skipIf(np is None, "NumPy not installed")
class AnalyzeVoiceTests(unittest.TestCase):
    def test_fake_backends_and_closing(self):
        from true_experience.voice import analyze_voice
        clips = [voice(110, 1.5), voice(230, 1.5, seed=3), voice(112, 1.5, seed=5)]
        speech = [{"start_seconds": i * 2.0, "end_seconds": i * 2.0 + 1.5, "text": "one two three"}
                  for i in range(3)]
        calls, closed = [], []

        happy = [0.9, 0.2, 0.2]

        def emotion(samples, rate):
            h = happy[len(calls)]
            calls.append(len(samples))
            return [{"label": "hap", "score": h}, {"label": "neu", "score": 1 - h}]
        emotion.close = lambda: closed.append(True)
        meta = analyze_voice("x.mp4", speech, emotion_backend=emotion, speaker_model=None,
                             audio_loader=lambda: iter(clips))
        self.assertEqual(len(calls), 3)
        self.assertEqual(closed, [True])
        self.assertEqual(speech[0]["voice"]["emotion"]["label"], "hap")
        self.assertEqual(speech[0]["speaker"], speech[2]["speaker"])
        self.assertNotEqual(speech[0]["speaker"], speech[1]["speaker"])
        self.assertEqual(meta["speakers"], 2)
        self.assertEqual(speech[1]["voice"]["emotion"]["label"], "neu")  # below the film's average
        self.assertEqual(speech[1]["voice"]["emotion"]["model_label"], "neu")

    def test_speaker_embeddings_group_voices(self):
        from true_experience.voice import analyze_voice
        rng = np.random.default_rng(0)
        a, b = rng.normal(size=192), rng.normal(size=192)
        who = ["a", "b", "a", "a", "b"]
        lengths = [2.0, 2.0, 0.6, 1.5, 3.0]
        clips = [voice(150, n) for n in lengths]
        speech, t = [], 0.0
        for n in lengths:
            speech.append({"start_seconds": t, "end_seconds": t + n, "text": "x"})
            t += n + 0.5
        order = iter(who)
        closed = []

        def speaker(samples, rate):
            base = a if next(order) == "a" else b
            v = base + 0.2 * rng.normal(size=192)
            return v / np.linalg.norm(v)
        speaker.close = lambda: closed.append(True)
        meta = analyze_voice("x", speech, emotion_model=None, speaker_backend=speaker,
                             audio_loader=lambda: iter(clips))
        labels = [s.get("speaker") for s in speech]
        self.assertEqual(labels, ["S1", "S2", "S1", "S1", "S2"])  # short line joins its voice
        self.assertEqual(meta["speakers"], 2)
        self.assertEqual(meta["speaker_method"], "wavlm_xvector_average_linkage")
        self.assertEqual(speech[0]["speaker_source"], "speaker_embedding_cluster")
        self.assertEqual(closed, [True])

    def test_relative_emotion_removes_a_constant_lean(self):
        from true_experience.voice import relative_emotions
        raw = [[{"label": "ang", "score": 0.75}, {"label": "sad", "score": 0.15}, {"label": "neu", "score": 0.1}],
               [{"label": "ang", "score": 0.75}, {"label": "sad", "score": 0.15}, {"label": "neu", "score": 0.1}],
               [{"label": "ang", "score": 0.5}, {"label": "sad", "score": 0.4}, {"label": "neu", "score": 0.1}],
               None]
        out = relative_emotions(raw)
        self.assertEqual([o["model_label"] for o in out[:3]], ["ang"] * 3)
        self.assertEqual(out[2]["label"], "sad")
        self.assertLess(out[2]["valence"], 0)
        self.assertLess(out[2]["arousal"], 0)
        self.assertEqual(out[0]["label"], "neu")
        self.assertIsNone(out[3])

    def test_no_speech_is_a_no_op(self):
        from true_experience.voice import analyze_voice
        self.assertEqual(analyze_voice("x", [], emotion_model=None)["segments"], 0)

    @unittest.skipIf(not shutil.which("ffmpeg"), "FFmpeg missing")
    def test_stream_segments_from_real_file(self):
        from true_experience.voice import stream_segments
        with tempfile.TemporaryDirectory() as td:
            wav, mp4 = Path(td) / "a.wav", Path(td) / "a.mp4"
            audio = np.concatenate([voice(110, 2), np.zeros(RATE, np.float32), voice(220, 2)])
            with wave.open(str(wav), "wb") as out:
                out.setnchannels(1); out.setsampwidth(2); out.setframerate(RATE)
                out.writeframes((audio * 32767).astype("<i2").tobytes())
            subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "lavfi",
                            "-i", "color=c=black:s=64x64:r=5:d=5", "-i", str(wav),
                            "-c:v", "mpeg4", "-c:a", "aac", "-shortest", str(mp4)], check=True)
            segs = [{"start_seconds": 0.2, "end_seconds": 1.8},
                    {"start_seconds": 1.5, "end_seconds": 2.5},
                    {"start_seconds": 3.2, "end_seconds": 4.8}]
            parts = list(stream_segments(mp4, segs))
            self.assertEqual([len(p) for p in parts], [25600, 16000, 25600])
            from true_experience.voice import prosody
            self.assertAlmostEqual(prosody(parts[0], "")["pitch_median_hz"], 110, delta=8)
            self.assertAlmostEqual(prosody(parts[2], "")["pitch_median_hz"], 220, delta=12)


class FusionVoiceTests(unittest.TestCase):
    def test_voice_reading_drives_fused_valence(self):
        from true_experience import fusion
        window = {"start_seconds": 0, "end_seconds": 5, "audio": {"state": "not_detected_silent"},
                  "speech": [{"start_seconds": 0.5, "end_seconds": 4.5, "speaker": "S1",
                              "voice": {"pitch_variability_semitones": 6.0, "words_per_second": 4.0,
                                        "loudness_dbfs": -12.0,
                                        "emotion": {"label": "ang", "score": 0.9,
                                                    "valence": -0.6, "arousal": 0.8}}}],
                  "feeling_signals": {"pacing": {"cuts_per_minute": 10}, "look": None, "audio": None}}
        fusion.fuse_timeline([window])
        feeling = window["feeling"]
        self.assertEqual(feeling["voice"]["speakers"], ["S1"])
        self.assertEqual(feeling["voice"]["emotions"], ["ang"])
        self.assertGreater(feeling["voice"]["arousal"], 0.6)
        self.assertLess(feeling["fused"]["valence"], -0.3)
        self.assertIn(feeling["fused"]["label"], ("tension", "unease"))


if __name__ == "__main__":
    unittest.main()
