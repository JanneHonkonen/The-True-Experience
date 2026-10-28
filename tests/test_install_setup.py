"""Guard the simple requirements/venv installation path against documentation drift."""
import os
from pathlib import Path
import subprocess
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]


class SetupDocsTests(unittest.TestCase):
    def test_requirements_is_not_an_editable_specifier(self):
        data = (ROOT / "requirements.txt").read_text(encoding="utf-8")
        packages = [line.strip() for line in data.splitlines()
                    if line.strip() and not line.lstrip().startswith("#")]
        self.assertEqual(packages, [
            "faster-whisper>=1.1,<2",
            "av>=11,<19",
            "transformers>=4.45,<5",
            "torch>=2.2",
            "pillow>=10",
        ])
        self.assertFalse(any(x.startswith(("-e ", "--editable ")) for x in packages))

    def test_requirements_matches_pyproject_all_extra(self):
        try:
            import tomllib
        except ImportError:
            self.skipTest("Requires Python 3.11+ to parse pyproject with stdlib tomllib")
        meta = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
        expected = sorted(meta["project"]["optional-dependencies"]["all"])
        installed = sorted(line.strip() for line in
                           (ROOT / "requirements.txt").read_text(encoding="utf-8").splitlines()
                           if line.strip() and not line.startswith("#"))
        self.assertEqual(expected, installed)

    def test_docs_use_requirements_and_venv(self):
        for doc in ("README.md", "QUICKSTART.md", "docs/IRON_MAN_TRAILER_EVALUATION.md"):
            contents = (ROOT / doc).read_text(encoding="utf-8")
            self.assertIn("python -m pip install -r requirements.txt", contents, doc)
            self.assertIn("venv", contents, doc)

    def test_cli_help_works_from_checkout_without_editable_install(self):
        result = subprocess.run([sys.executable, "-m", "true_experience.local_trailer", "--help"],
                                cwd=ROOT, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--max-caption-frames", result.stdout)
