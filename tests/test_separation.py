"""Separation stage: chunking, stem writing and caching (fake model), music reading."""
import io
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


class FakeProc:
    def __init__(self, data):
        self.stdout = io.BytesIO(data)


@unittest.skipIf(np is None, "NumPy not installed")
class ChunkTests(unittest.TestCase):
    def test_chunks_cover_every_sample_once_with_context(self):
        from true_experience import separation
        total = int(2.5 * 44100 * 3)  # 7.5 s at 3 s chunks
        ramp = np.arange(total, dtype=np.float32)
        stereo = np.stack([ramp, -ramp], axis=1).astype("<f4").tobytes()
        bodies = []
        for context, start, length in separation.stream_chunks(FakeProc(stereo), chunk=3.0, pad=0.5):
            body = context[0, start:start + length]
            bodies.append(body)
            if start:  # context before the body is the previous samples
                self.assertEqual(context[0, start - 1], body[0] - 1)
            if start + length < context.shape[1]:
                self.assertEqual(context[0, start + length], body[-1] + 1)
        joined = np.concatenate(bodies)
        self.assertTrue(np.array_equal(joined, ramp))
        self.assertEqual(len(bodies), 3)


@unittest.skipIf(np is None or not shutil.which("ffmpeg"), "NumPy or FFmpeg missing")
class SeparateAudioTests(unittest.TestCase):
    def test_fake_model_writes_stems_and_reuses_them(self):
        from true_experience import separation
        rate = 44100
        seconds = 65
        t = np.arange(seconds * rate) / rate
        left, right = 0.3 * np.sin(2 * np.pi * 220 * t), 0.3 * np.sin(2 * np.pi * 440 * t)
        calls = []

        def fake(stereo):
            calls.append(stereo.shape[1])
            return stereo[0], stereo[1]
        fake.close = lambda: calls.append("closed")
        with tempfile.TemporaryDirectory() as td:
            src = Path(td) / "clip.wav"
            with wave.open(str(src), "wb") as out:
                out.setnchannels(2); out.setsampwidth(2); out.setframerate(rate)
                out.writeframes((np.stack([left, right], 1) * 32767).astype("<i2").tobytes())
            result = separation.separate_audio(src, cache_root=Path(td) / "cache", backend=fake)
            self.assertEqual(calls[-1], "closed")
            self.assertEqual(len(calls), 3)  # two chunks, then close
            self.assertAlmostEqual(result["settings"]["audio_seconds"], seconds, delta=0.1)
            for path, rate_out, freq in ((result["vocals_path"], 16000, 220),
                                         (result["music_path"], 22050, 440)):
                with wave.open(str(path)) as w:
                    self.assertEqual(w.getframerate(), rate_out)
                    self.assertAlmostEqual(w.getnframes() / rate_out, seconds, delta=0.1)
                    data = np.frombuffer(w.readframes(w.getnframes()), "<i2").astype(float)
                # Across the chunk boundary at 60 s the tone is unbroken.
                part = data[int(59.5 * rate_out):int(60.5 * rate_out)]
                peak = np.argmax(np.abs(np.fft.rfft(part))) * rate_out / len(part)
                self.assertAlmostEqual(peak, freq, delta=2)
                self.assertGreater(np.sqrt(np.mean(part ** 2)), 0.25 * 32767 * 0.7)
            again = separation.separate_audio(src, cache_root=Path(td) / "cache", backend=fake)
            self.assertTrue(again["settings"]["reused"])
            self.assertEqual(len(calls), 3)


class MusicReadingTests(unittest.TestCase):
    def test_separated_music_counts_more_and_silence_is_no_music(self):
        from true_experience.fusion import read_music
        audio = {"mode": "minor", "mode_confidence": 0.6, "tempo_bpm": 130, "tempo_confidence": 0.5,
                 "onset_rate_per_second": 3.0, "rms_dbfs": -20.0}
        mix, stem = read_music(audio), read_music(audio, separated=True)
        self.assertGreater(stem["confidence"], mix["confidence"])
        self.assertEqual(stem["source"], "music_stem")
        self.assertIsNone(read_music({**audio, "rms_dbfs": -60.0}, separated=True))

    def test_fusion_prefers_music_stem(self):
        from true_experience import fusion
        window = {"start_seconds": 0, "end_seconds": 5, "audio": {"state": "not_detected_silent"},
                  "feeling_signals": {"pacing": {"cuts_per_minute": 10}, "look": None,
                                      "audio": {"mode": "major", "mode_confidence": 0.9, "rms_dbfs": -20.0},
                                      "music": {"mode": "minor", "mode_confidence": 0.9, "rms_dbfs": -25.0}}}
        fusion.fuse_timeline([window])
        self.assertEqual(window["feeling"]["music"]["mode"], "minor")
        self.assertEqual(window["feeling"]["music"]["source"], "music_stem")


if __name__ == "__main__":
    unittest.main()
