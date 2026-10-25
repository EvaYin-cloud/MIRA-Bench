from __future__ import annotations

import argparse
import json
import traceback
from pathlib import Path
from typing import Any

from ..main import load_cli_config


def _response_to_dict(resp: Any) -> dict[str, Any]:
    return {
        "model_name": resp.model_name,
        "finish_reason": resp.finish_reason,
        "latency_sec": resp.latency_sec,
        "usage": resp.usage,
        "raw_text": resp.raw_text,
        "parsed_json": resp.parsed_json,
        "visible_reasoning": resp.visible_reasoning,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample_index", type=int, default=0)
    parser.add_argument("--output", type=str, default=None)
    args, cfg = load_cli_config(parser=parser)
    if args.print_config:
        print(json.dumps(cfg, ensure_ascii=False, indent=2))
        return

    from ..core.benchmark_runner import BenchmarkRunner
    from ..core.prompt_builder import build_user_prompt
    from ..core.utils import ensure_dir, save_json, set_seed

    set_seed(int(cfg.get("seed", 42)))
    cfg = dict(cfg)
    cfg["runtime"] = dict(cfg.get("runtime", {}))
    cfg["runtime"]["save_raw_response"] = False

    runner = BenchmarkRunner(cfg)
    if len(runner.dataset) == 0:
        raise ValueError("Dataset is empty after filtering.")
    if args.sample_index < 0 or args.sample_index >= len(runner.dataset):
        raise IndexError(
            f"sample_index {args.sample_index} is out of range for "
            f"{len(runner.dataset)} samples."
        )

    sample = runner.dataset.samples[args.sample_index]
    rgb_path, depth_path, mask_path = runner._select_images(sample)
    user_prompt = build_user_prompt(
        sample,
        cfg["prompt"],
        cfg.get("prompt_version", "v1"),
        has_depth=(depth_path is not None),
        has_mask=(mask_path is not None),
    )

    result: dict[str, Any] = {
        "ok": False,
        "model_preset": args.model,
        "input_mode": args.mode or "rgbdm",
        "experiment_name": cfg.get("experiment_name"),
        "model": cfg.get("model", {}),
        "sample_index": args.sample_index,
        "question_id": sample.question_id,
        "question_type": sample.question_type,
        "question": sample.question,
        "choices": sample.choices,
        "gold_answer": sample.answer,
        "rgb_path": rgb_path,
        "depth_path": depth_path,
        "mask_path": mask_path,
    }

    try:
        resp = runner.model.infer(
            system_prompt=runner.system_prompt,
            user_prompt=user_prompt,
            rgb_path=rgb_path,
            depth_path=depth_path,
            mask_path=mask_path,
            question_type=sample.question_type,
        )
        result.update(_response_to_dict(resp))
        result["ok"] = (
            bool(resp.raw_text)
            and resp.parsed_json is not None
            and "MAX_TOKENS" not in str(resp.finish_reason)
            and not str(resp.finish_reason).lower().startswith("error")
        )
    except Exception as e:
        result["error"] = f"{type(e).__name__}: {e}"
        result["traceback"] = traceback.format_exc()

    output_path = (
        Path(args.output)
        if args.output
        else ensure_dir("logs") / f"{cfg.get('experiment_name', 'gemini')}_health_check.json"
    )
    save_json(result, output_path)

    print(f"Saved health check to: {output_path}")
    print(f"ok: {result['ok']}")
    if "finish_reason" in result:
        print(f"finish_reason: {result['finish_reason']}")
    if "error" in result:
        print(f"error: {result['error']}")
    raw_text = str(result.get("raw_text", ""))
    if raw_text:
        print("raw_text preview:")
        print(raw_text[:1000])


if __name__ == "__main__":
    main()
