import subprocess
import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class CliTests(unittest.TestCase):
    def test_audit_wrapper_runs_from_repository_root(self):
        completed = subprocess.run(
            [sys.executable, "scripts/audit_pipeline.py"],
            cwd=PROJECT_ROOT,
            check=False,
            capture_output=True,
            text=True,
        )

        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn('"usable_labelled_rows"', completed.stdout)


if __name__ == "__main__":
    unittest.main()
