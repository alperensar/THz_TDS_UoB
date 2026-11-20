"""Startup, spreadsheet export and spawned-worker checks for supported platforms."""

from pathlib import Path
from types import SimpleNamespace
from tempfile import TemporaryDirectory
import multiprocessing
import subprocess
import sys
import unittest

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import tds_extraction_core as core
from config import load_and_validate_config


class InitialisationTests(unittest.TestCase):
    def test_documented_cli_entry_points_start(self):
        for name in ("tds_preprocessing_CLI.py", "tds_extraction_CLI.py"):
            with self.subTest(script=name):
                result = subprocess.run(
                    [sys.executable, str(ROOT / "src" / name), "--help"],
                    cwd=ROOT, capture_output=True, text=True, encoding="utf-8",
                    errors="replace", timeout=60,
                )
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                self.assertIn("--root-dir", result.stdout)

    def test_missing_dataset_path_is_reported_before_gui(self):
        with TemporaryDirectory() as directory:
            missing = Path(directory) / "missing measurements"
            for name in ("tds_preprocessing_CLI.py", "tds_extraction_CLI.py"):
                with self.subTest(script=name):
                    result = subprocess.run(
                        [sys.executable, str(ROOT / "src" / name),
                         "--config", str(ROOT / "config.yaml"),
                         "--root-dir", str(missing)],
                        cwd=ROOT, capture_output=True, text=True, encoding="utf-8",
                        errors="replace", timeout=60,
                    )
                    self.assertNotEqual(result.returncode, 0)
                    self.assertIn("does not exist", result.stderr)

    def test_default_configuration_uses_included_example(self):
        config = load_and_validate_config(ROOT / "config.yaml")
        self.assertTrue((ROOT / config.paths.root_dir).is_dir())
        self.assertTrue((ROOT / config.paths.root_dir / config.paths.thickness_file).is_file())

    def test_excel_export_round_trip(self):
        frame = pd.DataFrame({"frequency_THz": [.3, .4], "epsilon": [3.7, 3.7]})
        with TemporaryDirectory() as directory:
            output = Path(directory) / "material results.xlsx"
            frame.to_excel(output, index=False)
            pd.testing.assert_frame_equal(pd.read_excel(output), frame)

    def test_extraction_runs_in_a_spawned_process(self):
        frequency = np.array([.5e12])
        pair = SimpleNamespace(
            freqs_interest=frequency, fft_freqs=frequency, sigma_um=0.,
            fp_term_count=1, x_min_index=0, x_max_index=0,
        )
        pair.measured_H = core.calculate_model_H(
            pair, np.array([2.]), np.array([.01]), .002, roughness=False, FP=False,
        )
        pair.unwrapped_phase = np.unwrap(np.angle(pair.measured_H))
        options = dict(
            algo="Optimise", whole_freq=False, roughness=False, fp=False,
            bounds=dict(n_bound=.15, k_bound=3),
            seed_n=np.array([2.]), seed_k=np.array([.01]),
        )
        with multiprocessing.get_context("spawn").Pool(1) as pool:
            _, result = pool.apply_async(
                core.run_param_extraction, ((pair, .002, options),),
            ).get(timeout=60)
        self.assertIsNotNone(result)
        np.testing.assert_allclose(result[0], [2.], rtol=0, atol=1e-5)
        np.testing.assert_allclose(result[1], [.01], rtol=0, atol=1e-5)


if __name__ == "__main__":
    unittest.main()
