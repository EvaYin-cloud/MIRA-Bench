import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from src.main import load_cli_config


ROOT = Path(__file__).resolve().parents[1]


class ConfigCliTests(unittest.TestCase):
    def run_cli(self, directory, *arguments, module="src.main"):
        env = dict(os.environ)
        env["PYTHONPATH"] = str(ROOT)
        env["PYTHONDONTWRITEBYTECODE"] = "1"
        return subprocess.run(
            [sys.executable, "-B", "-m", module, *arguments],
            cwd=directory, env=env, capture_output=True, text=True,
        )

    def test_print_config_creates_no_outputs_or_model_imports(self):
        with tempfile.TemporaryDirectory() as directory:
            result = self.run_cli(
                directory, "--model", "deepseek-vl2-tiny", "--mode", "rgbdm",
                "--max-samples", "20", "--print-config",
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            cfg = json.loads(result.stdout)
            self.assertEqual(cfg["input"]["max_samples"], 20)
            self.assertEqual(cfg["model"]["provider"], "deepseek_vl2")
            self.assertEqual(list(Path(directory).iterdir()), [])
        probe = subprocess.run(
            [sys.executable, "-B", "-c",
             "import sys; from src.main import load_cli_config; "
             "load_cli_config(['--model', 'deepseek-vl2-tiny']); "
             "assert 'src.core.benchmark_runner' not in sys.modules; "
             "assert 'torch' not in sys.modules"],
            cwd=ROOT, capture_output=True, text=True,
        )
        self.assertEqual(probe.returncode, 0, probe.stderr)

    def test_health_check_uses_shared_selectors_and_keeps_its_options(self):
        with tempfile.TemporaryDirectory() as directory:
            result = self.run_cli(
                directory, "--model", "gemini-2.5-pro", "--mode", "rgbd",
                "--sample_index", "3", "--output", "health.json", "--print-config",
                module="src.tools.check_gemini_response",
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            _, expected = load_cli_config(["--model", "gemini-2.5-pro", "--mode", "rgbd"])
            self.assertEqual(json.loads(result.stdout), expected)
            self.assertEqual(list(Path(directory).iterdir()), [])

    def test_model_selection_is_required_before_creating_outputs(self):
        with tempfile.TemporaryDirectory() as directory:
            result = self.run_cli(directory, "--print-config")
            self.assertEqual(result.returncode, 2)
            self.assertIn("required: --model", result.stderr)
            self.assertEqual(list(Path(directory).iterdir()), [])

    def test_unknown_model_fails_before_creating_outputs(self):
        with tempfile.TemporaryDirectory() as directory:
            result = self.run_cli(directory, "--model", "missing-model", "--print-config")
            self.assertEqual(result.returncode, 2)
            self.assertEqual(list(Path(directory).iterdir()), [])


if __name__ == "__main__":
    unittest.main()
