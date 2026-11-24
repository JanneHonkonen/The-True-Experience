"""CPU feeling signals on synthetic audio, frames and a generated video."""
import json
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

try:
    import numpy as np
except ImportError:  # pragma: no cover - baseline CI without NumPy
    np = None

RATE = 22050


def tone(freqs, seconds, rate=RATE, amp=0.2):
    t = np.arange(int(seconds * rate)) / rate
    return (amp * sum(np.sin(2 * np.pi * f * t) for f in freqs) / len(freqs)).astype(np.float32)


def clicks(bpm, seconds, rate=RATE):
    out = np.zeros(int(seconds * rate), dtype=np.float32)
    step = 60.0 / bpm
    for k in range(int(seconds / step) + 1):
        i = int(k * step * rate)
        out[i:i + 200] = np.hanning(200)[:len(out[i:i + 200])] * 0.8
    return out


@unittest.skipIf(np is None, "NumPy not installed")
class AudioFeatureTests(unittest.TestCase):
    def test_key_detection_major_and_minor_chords(self):
        from true_experience import signals
        # C major triad with octave doubling; A minor triad.
        c_major = tone([261.63, 329.63, 392.00, 523.25], 4)
        a_minor = tone([220.00, 261.63, 329.63, 440.00], 4)
        for samples, expected in ((c_major, "C major"), (a_minor, "A minor")):
            mag, freqs = signals._spectrogram(samples, RATE)
            key, mode, conf = signals.estimate_key(signals.chroma(mag, freqs))
            self.assertEqual(key, expected)
            self.assertEqual(mode, expected.split()[1])
            self.assertGreater(conf, 0)

    def test_noise_has_no_clear_key(self):
        from true_experience import signals
        self.assertEqual(signals.estimate_key(np.ones(12))[1], "unclear")
        self.assertEqual(signals.estimate_key(np.zeros(12))[1], "unclear")

    def test_tempo_of_click_track(self):
        from true_experience import signals
        for bpm in (90, 120):
            mag, _ = signals._spectrogram(clicks(bpm, 8), RATE)
            tempo, conf = signals.estimate_tempo(signals.onset_envelope(mag), RATE)
            self.assertIsNotNone(tempo)
            self.assertAlmostEqual(tempo, bpm, delta=bpm * 0.04)
            self.assertGreater(conf, 0.1)

    def test_steady_tone_has_no_onsets_tempo_or_key(self):
        from true_experience import signals
        noisy = tone([330], 5) + np.random.default_rng(0).normal(0, 0.001, RATE * 5).astype(np.float32)
        result = signals.summarize_audio(noisy, RATE, np.zeros(0, np.float32), 0)
        self.assertEqual(result["onset_rate_per_second"], 0.0)
        self.assertIsNone(result["tempo_bpm"])
        self.assertEqual(result["mode"], "unclear")

    def test_silence_has_no_tempo(self):
        from true_experience import signals
        mag, _ = signals._spectrogram(np.zeros(RATE * 4, dtype=np.float32), RATE)
        self.assertEqual(signals.estimate_tempo(signals.onset_envelope(mag), RATE), (None, 0.0))

    def test_startle_after_quiet_but_not_in_steady_loudness(self):
        from true_experience import signals
        quiet = tone([200], 2, amp=0.001)
        bang = tone([200], 0.5, amp=0.9)
        result = signals.summarize_audio(np.concatenate([quiet, bang]), RATE,
                                         np.zeros(0, dtype=np.float32), 10.0)
        self.assertTrue(result["startle"])
        self.assertAlmostEqual(result["startle_times_seconds"][0], 12.0, delta=0.1)
        steady = signals.summarize_audio(tone([200], 3, amp=0.5), RATE,
                                         np.zeros(0, dtype=np.float32), 0.0)
        self.assertFalse(steady["startle"])

    def test_startle_uses_previous_window_as_context(self):
        from true_experience import signals
        history = tone([200], 1, amp=0.9)
        loud = tone([200], 2, amp=0.9)
        result = signals.summarize_audio(loud, RATE, history, 5.0)
        self.assertFalse(result["startle"])

    def test_loudness_and_brightness(self):
        from true_experience import signals
        low = signals.summarize_audio(tone([150], 2), RATE, np.zeros(0, np.float32), 0)
        high = signals.summarize_audio(tone([3000], 2), RATE, np.zeros(0, np.float32), 0)
        self.assertLess(low["spectral_centroid_hz"], high["spectral_centroid_hz"])
        self.assertAlmostEqual(low["rms_dbfs"], 20 * np.log10(0.2 / np.sqrt(2)), delta=0.5)
        self.assertEqual(low["source"], "full_mix")


@unittest.skipIf(np is None, "NumPy not installed")
class LookAndPacingTests(unittest.TestCase):
    def test_look_values(self):
        from true_experience import signals
        red = np.zeros((8, 8, 3), np.uint8); red[..., 0] = 255
        blue = np.zeros((8, 8, 3), np.uint8); blue[..., 2] = 255
        gray = np.full((8, 8, 3), 40, np.uint8)
        r, b, g = (signals.frame_look(x) for x in (red, blue, gray))
        self.assertGreater(r["warmth"], 0.9)
        self.assertLess(b["warmth"], -0.9)
        self.assertAlmostEqual(g["saturation"], 0.0)
        self.assertGreater(r["saturation"], 0.9)
        self.assertLess(g["brightness"], 0.2)

    def test_motion_ignores_pairs_across_cuts(self):
        from true_experience import signals
        black = np.zeros((4, 4, 3), np.uint8)
        white = np.full((4, 4, 3), 255, np.uint8)
        moving = signals.summarize_look([black, white], [0.0, 0.5], [])
        cut = signals.summarize_look([black, white], [0.0, 0.5], [0.25])
        self.assertAlmostEqual(moving["motion"], 1.0)
        self.assertIsNone(cut["motion"])
        self.assertIsNone(signals.summarize_look([], [], []))

    def test_pacing(self):
        from true_experience import signals
        from true_experience.pipeline import create_shots
        shots = create_shots([1, 2, 3, 7], 10)
        pacing = signals.summarize_pacing(0, 5, shots)
        self.assertEqual(pacing["cuts_in_window"], 3)
        self.assertEqual(pacing["cuts_per_minute"], 36.0)


@unittest.skipIf(np is None or not shutil.which("ffmpeg"), "NumPy or FFmpeg missing")
class VideoIntegrationTests(unittest.TestCase):
    def test_signals_on_generated_video(self):
        from true_experience.pipeline import analyze
        with tempfile.TemporaryDirectory() as td:
            video = Path(td) / "clip.mp4"
            wav = Path(td) / "audio.wav"
            # 0-5 s: bright warm frame + quiet A minor chord.
            # 5-10 s: dark frame, near-silence, loud bang at 7 s.
            rate = 44100
            audio = np.zeros(rate * 10, dtype=np.float32)
            audio[:rate * 5] = tone([220.00, 261.63, 329.63, 440.00], 5, rate, amp=0.15)
            audio[rate * 7:int(rate * 7.6)] = tone([180], 0.6, rate, amp=0.9)
            import wave
            with wave.open(str(wav), "wb") as out:
                out.setnchannels(1); out.setsampwidth(2); out.setframerate(rate)
                out.writeframes((audio * 32767).astype("<i2").tobytes())
            subprocess.run([
                "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
                "-f", "lavfi", "-i", "color=c=0xFFB060:s=160x90:r=10:d=5",
                "-f", "lavfi", "-i", "color=c=0x101018:s=160x90:r=10:d=5",
                "-i", str(wav),
                "-filter_complex", "[0:v][1:v]concat=n=2:v=1:a=0[v]",
                "-map", "[v]", "-map", "2:a", "-c:v", "mpeg4", "-c:a", "aac", "-t", "10", str(video)],
                check=True)
            report = analyze(video, interval=5.0, feeling_signals=True)
            json.dumps(report, allow_nan=False)
            first, second = (w["feeling_signals"] for w in report["timeline"])
            self.assertIn("cpu_feeling_signals", report["metadata"]["methods"])
            self.assertGreater(first["look"]["brightness"], second["look"]["brightness"])
            self.assertGreater(first["look"]["warmth"], 0.3)
            self.assertEqual(first["audio"]["mode"], "minor")
            self.assertTrue(second["audio"]["startle"])
            self.assertAlmostEqual(second["audio"]["startle_times_seconds"][0], 7.0, delta=0.2)
            self.assertFalse(first["audio"]["startle"])
            self.assertGreaterEqual(first["look"]["frame_count"], 9)


if __name__ == "__main__":
    unittest.main()
