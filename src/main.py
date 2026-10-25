from __future__ import annotations

import argparse
import json
import yaml

from .core.configuration import load_config


def load_cli_config(
    argv: list[str] | None = None, *, parser: argparse.ArgumentParser | None = None,
) -> tuple[argparse.Namespace, dict]:
    if parser is None:
        parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Model preset from configs/models.yaml.")
    parser.add_argument("--mode", choices=("rgb", "rgbd", "rgbm", "rgbdm"),
                        help="Input mode; defaults to rgbdm for model presets.")
    parser.add_argument("--data-root", help="Root of the prepared dataset.")
    parser.add_argument("--qa-json", help="Override the question annotation file.")
    parser.add_argument("--max-samples", type=int, help="Evaluate a positive number of samples.")
    parser.add_argument("--output-dir", help="Override the experiment output directory.")
    parser.add_argument("--experiment-name", help="Override the experiment name.")
    parser.add_argument("--set", dest="overrides", action="append", default=[],
                        metavar="KEY=VALUE", help="Override a supported dotted config key.")
    parser.add_argument("--print-config", action="store_true",
                        help="Print the resolved config without loading a model or writing results.")
    args = parser.parse_args(argv)
    try:
        cfg = load_config(
            model=args.model, mode=args.mode,
            data_root=args.data_root, qa_json=args.qa_json,
            max_samples=args.max_samples, output_dir=args.output_dir,
            experiment_name=args.experiment_name, overrides=args.overrides,
        )
    except (ValueError, TypeError, OSError, yaml.YAMLError) as error:
        parser.error(str(error))
    return args, cfg


def main() -> None:
    args, cfg = load_cli_config()
    if args.print_config:
        print(json.dumps(cfg, ensure_ascii=False, indent=2))
        return

    from .core.benchmark_runner import BenchmarkRunner
    from .core.utils import set_seed, save_json, ensure_dir

    set_seed(int(cfg.get("seed", 42)))
    output_dir = ensure_dir(cfg["paths"]["output_dir"])
    save_json(cfg, output_dir / f"{cfg['experiment_name']}_config_used.json")

    runner = BenchmarkRunner(cfg)
    df = runner.run()
    print(f"Done. Evaluated {len(df)} samples.")


if __name__ == "__main__":
    main()
