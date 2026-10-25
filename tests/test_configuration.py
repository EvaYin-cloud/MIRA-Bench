from pathlib import Path
import unittest

import yaml

from src.core.configuration import load_config


class ConfigurationTests(unittest.TestCase):
    def test_four_modes_derive_paths_and_flags(self):
        for mode, rgb, depth in (
            ("rgb", "Dataset/images", None),
            ("rgbd", "Dataset/images", "Dataset"),
            ("rgbm", "Dataset/mirror_marked/rgb", None),
            ("rgbdm", "Dataset/mirror_marked/rgb", "Dataset/mirror_marked/depth"),
        ):
            with self.subTest(mode=mode):
                cfg = load_config(model="gpt-5.4", mode=mode)
                self.assertEqual(cfg["input"]["use_depth"], "d" in mode)
                self.assertEqual(cfg["input"]["use_mirror_marked"], "m" in mode)
                self.assertEqual(cfg["paths"]["rgb_dir"], rgb)
                self.assertEqual(cfg["paths"].get("depth_dir"), depth)
                if depth is None:
                    self.assertNotIn("depth_dir", cfg["paths"])
                self.assertEqual(cfg["paths"]["qa_json"], "Dataset/annotations/questions.json")
                self.assertEqual(cfg["experiment_name"], f"gpt-5.4_{mode}")
                self.assertEqual(cfg["paths"]["output_dir"], f"Outputs/gpt-5.4_{mode}")

    def test_default_mode_and_qwen_model(self):
        cfg = load_config(model="qwen3")
        self.assertEqual(cfg["model"]["model_name"], "Qwen/Qwen3.5-9B")
        self.assertEqual(cfg["experiment_name"], "qwen3_rgbdm")
        self.assertTrue(cfg["model"]["use_json_schema"])
        self.assertNotIn("disable_response_format", cfg["model"])
        self.assertNotIn("resume_from_checkpoint", cfg["runtime"])

    def test_gemini_runtime_and_prompt(self):
        for name in ("flash", "pro"):
            with self.subTest(model=name):
                cfg = load_config(model=f"gemini-2.5-{name}")
                self.assertEqual(cfg["model"]["max_tokens"], 4096)
                self.assertTrue(cfg["model"]["raise_on_api_error"])
                self.assertEqual(cfg["prompt"]["explanation_max_words"], 20)
                self.assertEqual(cfg["runtime"]["retry_times"], 5)
                self.assertEqual(cfg["runtime"]["sleep_between_retries_sec"], 10)
                self.assertTrue(cfg["runtime"]["resume_success_only"])
                if name == "pro":
                    self.assertEqual(cfg["runtime"]["max_new_samples_per_run"], 10)
                else:
                    self.assertNotIn("max_new_samples_per_run", cfg["runtime"])

    def test_local_and_gpt_special_values(self):
        for size in ("tiny", "small"):
            cfg = load_config(model=f"deepseek-vl2-{size}")
            self.assertEqual(cfg["model"]["model_path"], f"models/deepseek-vl2-{size}")
            self.assertEqual(cfg["model"]["chunk_size"], -1)
            self.assertEqual(cfg["model"]["max_tokens"], 512)
            self.assertEqual(cfg["model"]["dtype"], "bfloat16")
        mini = load_config(model="minicpm-v-2_6")
        self.assertEqual(mini["model"]["attn_implementation"], "sdpa")
        self.assertFalse(mini["model"]["sampling"])
        gpt = load_config(model="gpt-5.4")
        self.assertEqual(gpt["model"]["max_completion_tokens"], 2048)
        self.assertTrue(gpt["model"]["use_max_completion_tokens"])

    def test_model_ports_and_internvl_mode_override(self):
        for model, port in (("llava-ov-7b", 8004), ("qwen2.5-vl-7b", 8003),
                            ("qwen3-vl-8b", 8002), ("qwen3", 8000)):
            self.assertEqual(load_config(model=model)["model"]["api_base"],
                             f"http://localhost:{port}/v1")
        for mode in ("rgb", "rgbd", "rgbm", "rgbdm"):
            cfg = load_config(model="internvl3", mode=mode)
            port = 8001 if mode == "rgbdm" else 8000
            self.assertEqual(cfg["model"]["api_base"], f"http://localhost:{port}/v1")
            self.assertNotIn("mode_overrides", cfg)

    def test_mirror_hint_matches_input_mode(self):
        cfg = load_config(model="gpt-5.4")
        self.assertEqual(
            cfg["prompt"]["mirror_hint"],
            "The semi-transparent green overlay on the RGB image indicates reflective surface regions,\n"
            "which may include mirrors, glass, lake surfaces, or convex mirrors.\n"
            "The semi-transparent white overlay on the depth map marks the same reflective regions.\n",
        )
        self.assertEqual(load_config(model="gpt-5.4", mode="rgbd")["prompt"]["mirror_hint"], "")

    def test_explicit_overrides_are_last_and_do_not_mutate_presets(self):
        cfg = load_config(
            model="qwen3", mode="rgbd", data_root=Path("/data/bench"),
            max_samples=12, qa_json="selected.json", experiment_name="qwen3_custom",
            output_dir="/tmp/custom-output", overrides=[
                "input.max_samples=4", "input.allowed_question_types=[yes_no, count]",
                "model.api_base=http://localhost:9000/v1", "model.temperature=0.2",
                "runtime.resume_from_checkpoint=false", "runtime.max_new_samples_per_run=5",
            ],
        )
        self.assertEqual(cfg["input"]["max_samples"], 12)
        self.assertEqual(cfg["input"]["allowed_question_types"], ["yes_no", "count"])
        self.assertEqual(cfg["paths"], {
            "data_root": "/data/bench", "rgb_dir": "/data/bench/images",
            "depth_dir": "/data/bench", "qa_json": "selected.json",
            "output_dir": "/tmp/custom-output",
        })
        self.assertEqual(cfg["model"]["api_base"], "http://localhost:9000/v1")
        self.assertFalse(cfg["runtime"]["resume_from_checkpoint"])
        self.assertEqual(cfg["runtime"]["max_new_samples_per_run"], 5)
        original = load_config(model="qwen3")
        self.assertIsNone(original["input"]["max_samples"])
        self.assertNotIn("resume_from_checkpoint", original["runtime"])

    def test_name_override_gets_separate_default_output(self):
        cfg = load_config(model="qwen3", experiment_name="qwen3_custom")
        self.assertEqual(cfg["paths"]["output_dir"], "Outputs/qwen3_custom")

    def test_repeated_numeric_overrides_keep_the_same_type_contract(self):
        cfg = load_config(model="qwen3", overrides=[
            "model.temperature=0", "model.temperature=0.1",
            "runtime.sleep_between_retries_sec=0.5",
        ])
        self.assertEqual(cfg["model"]["temperature"], 0.1)
        self.assertEqual(cfg["runtime"]["sleep_between_retries_sec"], 0.5)

    def test_bad_selectors_and_explicit_values(self):
        for kwargs in (
            {}, {"model": "missing"}, {"model": "qwen3", "mode": "depth"},
            {"model": "qwen3", "mode": ""}, {"model": "qwen3", "max_samples": 0},
            {"model": "qwen3", "max_samples": True}, {"model": "qwen3", "data_root": ""},
            {"model": "qwen3", "experiment_name": "../bad"},
        ):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                load_config(**kwargs)

    def test_bad_override_paths_types_and_protected_switches(self):
        for override in (
            "input.use_depth=false", "input.use_mirror_marked=false", "paths.rgb_dir=x",
            "model.missing=1", "model=1", "model.max_tokens", "runtime.retry_times=true",
            "model.max_tokens=0", "model.temperature='warm'", "input.max_samples=-1",
            "input.allowed_template_ids=abc", "input.allowed_template_ids=[3]",
            "runtime.resume_from_checkpoint='false'", "prompt.ask_for_brief_explanation=1",
            "model.temperature=.nan", "runtime.sleep_between_retries_sec=.inf",
        ):
            with self.subTest(override=override), self.assertRaises(ValueError):
                load_config(model="qwen3", overrides=[override])
        with self.assertRaises(ValueError):
            load_config(model="qwen3", overrides="input.max_samples=10")

    def test_safe_yaml_overrides_cannot_construct_python_objects(self):
        with self.assertRaises(yaml.YAMLError):
            load_config(model="qwen3", overrides=[
                "prompt.system_prompt=!!python/object/apply:os.system ['echo unexpected']"
            ])


if __name__ == "__main__":
    unittest.main()
