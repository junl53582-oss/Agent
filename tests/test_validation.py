import tempfile
import unittest
from pathlib import Path

from stockpilot.config import Settings
from stockpilot.validation import run_validation_v2


class ValidationTests(unittest.TestCase):
    def test_existing_report_requires_explicit_force(self):
        with tempfile.TemporaryDirectory() as tmp:
            artifact_dir = Path(tmp)
            report_dir = artifact_dir / "validation_v2"
            report_dir.mkdir()
            (report_dir / "report.json").write_text("{}", encoding="utf-8")
            with self.assertRaisesRegex(FileExistsError, "--force"):
                run_validation_v2(
                    "missing-market.csv",
                    "missing-membership.csv",
                    "2024-01-01",
                    "2025-01-01",
                    settings=Settings(artifact_dir=artifact_dir),
                )


if __name__ == "__main__":
    unittest.main()
