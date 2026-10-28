"""Catch the faster-whisper/PyAV 19 metadata_errors API mismatch without model weights.

This runs for real on Windows CI. Standard FFmpeg CI may skip when optional
Whisper dependencies are absent.
"""
import math
import os
import struct
import tempfile
import unittest
import wave
from pathlib import Path


class WhisperAudioDecodeTests(unittest.TestCase):
    def test_actual_wav_decode_using_pinned_pyav(self):
        try:
            import av
            from faster_whisper.audio import decode_audio
        except ImportError as exc:
            if os.environ.get("TTE_REQUIRE_ASR_DECODE") == "1":
                self.fail(f"Expected Whisper/PyAV for compatibility smoke: {exc}")
            self.skipTest("Optional Whisper/PyAV dependencies not installed")

        # Regression for faster-whisper 1.2.1 + PyAV 19:
        # av.open() no longer accepts metadata_errors in PyAV 19.
        major = int(av.__version__.split(".")[0])
        self.assertLess(major, 19, f"Expected PyAV <19; installed {av.__version__}")
        sample_rate = 16000
        count = 6400
        pcm = b"".join(struct.pack("<h", round(10000 * math.sin(
            2 * math.pi * 440 * n / sample_rate))) for n in range(count))
        with tempfile.TemporaryDirectory() as td:
            source = Path(td) / "synthetic.wav"
            with wave.open(str(source), "wb") as output:
                output.setnchannels(1)
                output.setsampwidth(2)
                output.setframerate(sample_rate)
                output.writeframes(pcm)
            # Calls exactly the decoder that failed for the user's MP4.
            decoded = decode_audio(str(source), sampling_rate=sample_rate)
        self.assertGreaterEqual(len(decoded), count - 2)
        self.assertLessEqual(len(decoded), count + 2)
        self.assertGreater(max(abs(float(x)) for x in decoded), .05)
