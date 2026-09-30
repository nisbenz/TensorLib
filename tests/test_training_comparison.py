"""Run with python3 tests/test_training_comparison.py [bench_tensorlib]."""

import csv
import importlib.util
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("comparison", ROOT / "scripts/compare_training.py")
comparison = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(comparison)
EXECUTABLE = sys.argv.pop(1) if len(sys.argv) > 1 else None


class TrainingComparisonTests(unittest.TestCase):
    def test_paired_ratios_and_invalid_sequences(self):
        key = ("nn", "model", "shape", "layout", "fixed", "1")
        before = {key: {"median_seconds": "1", "iterations": "1", "checksum": "2"}}
        after = {key: {"median_seconds": ".8", "iterations": "1", "checksum": "2"}}
        row = comparison.summarize([(before, after)] * 3)[0]
        self.assertAlmostEqual(row["change_percent"], -20)
        for bad in ({}, {key: dict(after[key], iterations="2")},
                    {key: dict(after[key], checksum="9")},
                    {key: dict(after[key], cpu="different CPU")}):
            with self.assertRaises(ValueError):
                comparison.summarize([(before, bad)] * 3)
        changed = {key: dict(after[key], executable_sha256="different binary")}
        with self.assertRaises(ValueError):
            comparison.summarize([(before, after), (before, changed), (before, after)])

    @unittest.skipUnless(EXECUTABLE, "benchmark executable was not supplied")
    def test_training_thread_inventory_and_diagnostic_boundary(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "results.csv"
            for diagnostics in (False, True):
                command = [EXECUTABLE, "--suite", "training", "--smoke",
                           "--threads", "1,2", "--csv", str(path)]
                if not diagnostics:
                    command.append("--no-diagnostics")
                subprocess.run(command, check=True, stdout=subprocess.DEVNULL)
                with path.open(newline="") as source:
                    rows = list(csv.DictReader(source))
                training = [r for r in rows if r["case"].endswith("_train_step")]
                keys = [(r["case"], r["requested_threads"]) for r in training]
                self.assertEqual(len(keys), 6)
                self.assertEqual(len(set(keys)), 6)
                self.assertEqual({r["requested_threads"] for r in training}, {"1", "2"})
                self.assertEqual(any(r["suite"] == "nn_phase" for r in rows), diagnostics)
                if not diagnostics:
                    self.assertEqual(len(comparison.read_rows(path)), 6)


if __name__ == "__main__":
    unittest.main()
