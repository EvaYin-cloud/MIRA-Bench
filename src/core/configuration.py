from __future__ import annotations

from copy import deepcopy
import math
from pathlib import Path
from typing import Any

import yaml


CONFIG_DIR = Path(__file__).resolve().parents[2] / "configs"
MODES = ("rgb", "rgbd", "rgbm", "rgbdm")
_OPTIONAL_RUNTIME = {
    "checkpoint_every_n": int,
    "write_checkpoint_reports": bool,
    "resume_from_checkpoint": bool,
    "resume_success_only": bool,
    "max_new_samples_per_run": int,
}
_NULLABLE_INTS = {"input.max_samples", "runtime.max_new_samples_per_run"}
_FILTERS = {
    "allowed_question_types", "allowed_referring_styles", "allowed_template_ids",
    "include_categories", "exclude_categories",
}
_POSITIVE = {
    "input.max_samples", "runtime.max_new_samples_per_run", "model.max_tokens",
    "model.max_completion_tokens", "model.repair_max_tokens", "model.timeout_sec", "prompt.explanation_max_words",
}
_NONNEGATIVE = {
    "runtime.checkpoint_every_n", "runtime.retry_times",
    "runtime.sleep_between_retries_sec", "model.temperature",
}
_NUMERIC = {
    "model.temperature", "model.top_p", "model.repetition_penalty", "model.timeout_sec",
    "runtime.sleep_between_retries_sec",
}


def _read_mapping(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a YAML mapping in {path}")
    return value


def _merge(base: dict, patch: dict) -> dict:
    merged = deepcopy(base)
    for key, value in patch.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _merge(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged


def _path(value: Any, name: str) -> str:
    if not isinstance(value, (str, Path)) or not str(value).strip():
        raise ValueError(f"{name} must be a nonempty path")
    return str(value)


def _data_paths(cfg: dict, data_root: str, mode_paths: dict) -> None:
    root = Path(data_root)
    cfg["paths"]["data_root"] = str(root)
    cfg["paths"]["qa_json"] = str(root / "annotations" / "questions.json")
    for key in ("rgb_dir", "depth_dir"):
        if key in mode_paths:
            relative = Path(_path(mode_paths[key], f"mode paths.{key}"))
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError(f"Mode paths.{key} must be relative to data_root")
            cfg["paths"][key] = str(root / relative)
        else:
            cfg["paths"].pop(key, None)


def _check_value(key: str, value: Any, expected: Any) -> None:
    if key in _NULLABLE_INTS:
        valid = value is None or type(value) is int
    elif key.startswith("input.") and key.split(".")[1] in _FILTERS:
        valid = value is None or (
            isinstance(value, list) and all(isinstance(item, str) for item in value)
        )
    elif key in _NUMERIC:
        valid = type(value) in (int, float)
    elif type(expected) is bool or expected is bool:
        valid = type(value) is bool
    elif type(expected) is int or expected is int:
        valid = type(value) is int
    elif type(expected) is float:
        valid = type(value) in (int, float)
    elif isinstance(expected, list):
        valid = isinstance(value, list) and all(isinstance(item, str) for item in value)
    elif isinstance(expected, str):
        valid = isinstance(value, str)
    else:
        valid = type(value) is type(expected)
    if not valid:
        raise ValueError(f"Invalid type for {key}: {type(value).__name__}")
    if type(value) is float and not math.isfinite(value):
        raise ValueError(f"{key} must be finite")
    if value is not None:
        if key in _POSITIVE and value <= 0:
            raise ValueError(f"{key} must be positive")
        if key in _NONNEGATIVE and value < 0:
            raise ValueError(f"{key} must be nonnegative")


def _apply_overrides(cfg: dict, overrides: list[str] | None) -> None:
    if overrides is None:
        return
    if not isinstance(overrides, (list, tuple)):
        raise ValueError("overrides must be a list of key.path=value strings")
    original = deepcopy(cfg)
    for override in overrides:
        if not isinstance(override, str) or "=" not in override:
            raise ValueError(f"Expected key.path=value, got {override!r}")
        dotted, raw = override.split("=", 1)
        parts = dotted.split(".")
        if len(parts) != 2 or not all(parts):
            raise ValueError(f"Invalid override path: {dotted}")
        section, key = parts
        if section not in {"model", "prompt", "runtime", "input"}:
            raise ValueError(f"Use the named CLI arguments to override {dotted}")
        if section == "input" and key in {"use_depth", "use_mirror_marked"}:
            raise ValueError(f"Use --mode instead of overriding {dotted}")
        optional = section == "runtime" and key in _OPTIONAL_RUNTIME
        if section not in cfg or (key not in cfg[section] and not optional):
            raise ValueError(f"Unknown override path: {dotted}")
        expected = original[section].get(key, _OPTIONAL_RUNTIME.get(key))
        if isinstance(expected, dict):
            raise ValueError(f"Override a leaf value, not {dotted}")
        value = yaml.safe_load(raw)
        _check_value(dotted, value, expected)
        cfg[section][key] = value


def load_config(
    *, model=None, mode=None, data_root=None, max_samples=None,
    qa_json=None, output_dir=None, experiment_name=None, overrides=None, config_dir=None,
) -> dict:
    """Combine the base, model, and input mode settings, then apply CLI overrides."""
    directory = Path(config_dir) if config_dir is not None else CONFIG_DIR
    models = _read_mapping(directory / "models.yaml")
    modes = _read_mapping(directory / "modes.yaml")
    if not isinstance(model, str) or model not in models:
        raise ValueError(f"Unknown model {model!r}; choose from: {', '.join(models)}")
    mode = "rgbdm" if mode is None else mode
    if not isinstance(mode, str) or mode not in MODES or mode not in modes:
        raise ValueError(f"Unknown mode {mode!r}; choose from: {', '.join(MODES)}")
    preset = deepcopy(models[model])
    mode_overrides = preset.pop("mode_overrides", {})
    cfg = _merge(_read_mapping(directory / "base.yaml"), preset)
    cfg = _merge(cfg, modes[mode])
    cfg = _merge(cfg, mode_overrides.get(mode, {}))
    mode_paths = modes[mode]["paths"]
    expected_flags = ("d" in mode, "m" in mode)
    actual_flags = (cfg["input"].get("use_depth"), cfg["input"].get("use_mirror_marked"))
    if actual_flags != expected_flags or any(type(flag) is not bool for flag in actual_flags):
        raise ValueError(f"Input switches do not match mode {mode}")
    if expected_flags[0] != ("depth_dir" in mode_paths):
        raise ValueError(f"Depth directory does not match mode {mode}")
    _data_paths(cfg, _path(cfg["paths"]["data_root"], "data_root"), mode_paths)
    cfg["experiment_name"] = f"{model}_{mode}"
    cfg["paths"]["output_dir"] = str(Path("Outputs") / cfg["experiment_name"])

    _apply_overrides(cfg, overrides)
    if data_root is not None:
        for key in ("use_depth", "use_mirror_marked"):
            _check_value(f"input.{key}", cfg["input"].get(key, False), False)
        _data_paths(cfg, _path(data_root, "data_root"), mode_paths)
    if max_samples is not None:
        _check_value("input.max_samples", max_samples, None)
        cfg["input"]["max_samples"] = max_samples
    if qa_json is not None:
        cfg["paths"]["qa_json"] = _path(qa_json, "qa_json")
    if experiment_name is not None:
        if not isinstance(experiment_name, str) or not experiment_name.strip():
            raise ValueError("experiment_name must be a nonempty string")
        if experiment_name in {".", ".."} or "/" in experiment_name or "\\" in experiment_name:
            raise ValueError("experiment_name must be a name, not a path")
        cfg["experiment_name"] = experiment_name
        cfg["paths"]["output_dir"] = str(Path(cfg["paths"]["output_dir"]).parent / experiment_name)
    if output_dir is not None:
        cfg["paths"]["output_dir"] = _path(output_dir, "output_dir")
    return cfg
