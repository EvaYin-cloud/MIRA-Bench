from __future__ import annotations

import time
import traceback
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd
from tqdm import tqdm

from .dataset import BenchmarkDataset, QASample
from ..models import build_model_adapter
from .parsing import extract_prediction, validate_prediction_against_choices
from .prompt_builder import build_user_prompt
from .reporting import write_reports
from .scoring import score_prediction, compute_random_baseline
from .utils import ensure_dir, save_json


class BenchmarkRunner:
    def __init__(self, cfg: dict[str, Any]) -> None:
        self.cfg = cfg
        self.output_dir = ensure_dir(cfg["paths"]["output_dir"])
        self.prompt_version = cfg.get("prompt_version", "v1")

        self.dataset = BenchmarkDataset(
            qa_json=cfg["paths"]["qa_json"],
            rgb_dir=cfg["paths"]["rgb_dir"],
            depth_dir=cfg["paths"].get("depth_dir"),
            data_root=cfg["paths"].get("data_root"),
            use_depth=bool(cfg["input"].get("use_depth", False)),
            use_mirror_marked=bool(cfg["input"].get("use_mirror_marked", False)),
            image_suffixes=cfg["input"].get("image_suffixes", [".jpg", ".jpeg", ".png"]),
            max_samples=cfg["input"].get("max_samples"),
            allowed_question_types=cfg["input"].get("allowed_question_types"),
            allowed_referring_styles=cfg["input"].get("allowed_referring_styles"),
            allowed_template_ids=cfg["input"].get("allowed_template_ids"),
            include_categories=cfg["input"].get("include_categories"),
            exclude_categories=cfg["input"].get("exclude_categories"),
        )

        self.use_depth = bool(cfg["input"].get("use_depth", False))
        self.use_mirror_marked = bool(cfg["input"].get("use_mirror_marked", False))

        model_cfg = dict(cfg["model"])
        model_cfg["ask_for_brief_explanation"] = bool(
            cfg.get("prompt", {}).get("ask_for_brief_explanation", True)
        )
        model_cfg["explanation_max_words"] = int(
            cfg.get("prompt", {}).get("explanation_max_words", 8)
        )
        self.model = build_model_adapter(model_cfg)
        self.system_prompt = cfg["prompt"].get("system_prompt", "")
        self.save_raw_response = bool(cfg["runtime"].get("save_raw_response", True))
        self.experiment_name = cfg["experiment_name"]
        self.retry_times = int(cfg["runtime"].get("retry_times", 2))
        self.sleep_between_retries_sec = float(
            cfg["runtime"].get("sleep_between_retries_sec", 1)
        )
        self.checkpoint_every_n = int(cfg["runtime"].get("checkpoint_every_n", 0) or 0)
        self.write_checkpoint_reports = bool(
            cfg["runtime"].get("write_checkpoint_reports", True)
        )
        self.resume_from_checkpoint = bool(
            cfg["runtime"].get("resume_from_checkpoint", False)
        )
        self.resume_success_only = bool(
            cfg["runtime"].get("resume_success_only", True)
        )
        self.max_new_samples_per_run = cfg["runtime"].get("max_new_samples_per_run")
        if self.max_new_samples_per_run is not None:
            self.max_new_samples_per_run = int(self.max_new_samples_per_run)
        self.per_sample_path = self.output_dir / f"{self.experiment_name}_per_sample.csv"
        self.checkpoint_status_path = (
            self.output_dir / f"{self.experiment_name}_checkpoint_status.json"
        )

    # ------------------------------------------------------------------
    #  Image selection: referring_style x use_mirror_marked x use_depth
    # ------------------------------------------------------------------

    def _select_images(
        self, sample: QASample
    ) -> tuple[str, str | None, str | None]:
        """
        Select the images sent to the model according to referring_style and
        the current experiment settings.

        Returns: (rgb_path, depth_path, mask_path)

        Selection logic:
        - bbox_text / bbox_text_desc:
            rgb = original (rgb_dir controls rgb/ vs rgb_mirror_marked/)
            depth = original depth (controlled by depth_dir)
            mask = None

        - visual_bbox / visual_bbox_desc:
            if use_mirror_marked:
                rgb = rgbm_bbox_path,  depth = depthm_bbox_path
            else:
                rgb = rgb_bbox_path,   depth = depth_bbox_path
            mask = None

        - visual_mask:
            rgb = original
            depth = original depth
            mask = mask_path
        """
        style = sample.referring_style

        if style in ("bbox_text", "bbox_text_desc"):
            rgb = sample.rgb_original_path
            depth = sample.depth_original_path if self.use_depth else None
            return rgb, depth, None

        if style in ("visual_bbox", "visual_bbox_desc"):
            if self.use_mirror_marked:
                rgb = sample.rgbm_bbox_path or sample.rgb_bbox_path
                depth = (sample.depthm_bbox_path or sample.depth_bbox_path) if self.use_depth else None
            else:
                rgb = sample.rgb_bbox_path
                depth = sample.depth_bbox_path if self.use_depth else None
            # Fallback to the original image if a cached image is missing.
            if rgb is None:
                rgb = sample.rgb_original_path
            if self.use_depth and depth is None:
                depth = sample.depth_original_path
            return rgb, depth, None

        if style == "visual_mask":
            rgb = sample.rgb_original_path
            depth = sample.depth_original_path if self.use_depth else None
            return rgb, depth, sample.mask_path

        # Unknown style: fall back to the original image.
        rgb = sample.rgb_original_path
        depth = sample.depth_original_path if self.use_depth else None
        return rgb, depth, None

    # ------------------------------------------------------------------
    #  Helpers
    # ------------------------------------------------------------------

    def _extract_usage_fields(self, usage: dict[str, Any] | None) -> dict[str, float]:
        usage = usage or {}
        return {
            "usage_prompt_tokens": float(usage.get("prompt_tokens", 0) or 0),
            "usage_completion_tokens": float(usage.get("completion_tokens", 0) or 0),
            "usage_total_tokens": float(usage.get("total_tokens", 0) or 0),
        }

    def _call_model_with_retry(
        self,
        system_prompt: str,
        user_prompt: str,
        rgb_path: str,
        depth_path: str | None,
        mask_path: str | None,
        question_type: str,
    ):
        last_exc: Exception | None = None
        for attempt in range(self.retry_times + 1):
            try:
                return self.model.infer(
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    rgb_path=rgb_path,
                    depth_path=depth_path,
                    mask_path=mask_path,
                    question_type=question_type,
                )
            except Exception as e:
                last_exc = e
                if attempt < self.retry_times:
                    time.sleep(self.sleep_between_retries_sec)
                else:
                    raise last_exc
        raise last_exc

    def _should_repair_format(
        self,
        parse_success: bool,
        invalid_answer_format: bool,
        invalid_option: bool,
        raw_text: str,
    ) -> bool:
        if not raw_text or not raw_text.strip():
            return False
        return not parse_success or invalid_answer_format or invalid_option

    def _load_checkpoint_rows(self) -> tuple[list[dict[str, Any]], Counter[str]]:
        if not self.resume_from_checkpoint or not self.per_sample_path.exists():
            return [], Counter()

        df = pd.read_csv(self.per_sample_path)
        if self.resume_success_only and "api_success" in df.columns:
            df = df[df["api_success"].astype(str).isin({"1", "1.0", "True", "true"})]

        if "question_id" not in df.columns:
            return [], Counter()

        completed_counts = Counter(df["question_id"].astype(str))
        return df.to_dict("records"), completed_counts

    def _write_checkpoint(self, rows: list[dict[str, Any]], reason: str) -> None:
        if not rows:
            return

        df = pd.DataFrame(rows)
        if self.write_checkpoint_reports:
            write_reports(df, self.output_dir, self.experiment_name)
        else:
            df.to_csv(self.per_sample_path, index=False)

        save_json(
            {
                "experiment_name": self.experiment_name,
                "rows_saved": len(rows),
                "reason": reason,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "per_sample_csv": str(self.per_sample_path),
            },
            self.checkpoint_status_path,
        )

    # ------------------------------------------------------------------
    #  Main loop
    # ------------------------------------------------------------------

    def run(self) -> pd.DataFrame:
        rows, completed_counts = self._load_checkpoint_rows()
        raw_dir = ensure_dir(self.output_dir / f"{self.experiment_name}_raw")
        seen_counts: Counter[str] = Counter()
        samples = []
        for sample in self.dataset:
            question_id = str(sample.question_id)
            seen_counts[question_id] += 1
            if seen_counts[question_id] <= completed_counts.get(question_id, 0):
                continue
            samples.append(sample)
        if self.max_new_samples_per_run is not None:
            samples = samples[: self.max_new_samples_per_run]
        rows_since_checkpoint = 0

        try:
            iterator = tqdm(samples, total=len(samples), desc="Benchmark")
            for sample in iterator:
                if completed_counts:
                    iterator.set_postfix(saved=len(rows), refresh=False)
                # Select images before building the prompt because the prompt
                # must know whether depth or mask inputs are present.
                rgb_path, depth_path, mask_path = self._select_images(sample)

                user_prompt = build_user_prompt(
                    sample, self.cfg["prompt"], self.prompt_version,
                    has_depth=(depth_path is not None),
                    has_mask=(mask_path is not None),
                )

                random_baseline = compute_random_baseline(
                    sample.choices, sample.answer, sample.question_type
                )

                base_row = {
                    "experiment_name": self.experiment_name,
                    "prompt_version": self.prompt_version,
                    "model_name": self.cfg["model"]["model_name"],
                    "question_id": sample.question_id,
                    "original_question_id": sample.original_question_id,
                    "image_name": sample.image_name,
                    "image_stem": sample.image_stem,
                    "rgb_path_used": rgb_path,
                    "depth_path_used": depth_path,
                    "mask_path_used": mask_path,
                    "use_mirror_marked": self.use_mirror_marked,
                    "use_depth": self.use_depth,
                    "category": sample.category,
                    "subcategory": sample.subcategory,
                    "template_id": sample.template_id,
                    "question_type": sample.question_type,
                    "referring_style": sample.referring_style,
                    "requires_mirror_reasoning": sample.requires_mirror_reasoning,
                    "requires_depth": sample.requires_depth,
                    "question": sample.question,
                    "choices": str(sample.choices),
                    "gold_answer": str(sample.answer),
                    "target_objects": str(sample.target_objects),
                    "random_baseline": random_baseline,
                }

                try:
                    model_resp = self._call_model_with_retry(
                        system_prompt=self.system_prompt,
                        user_prompt=user_prompt,
                        rgb_path=rgb_path,
                        depth_path=depth_path,
                        mask_path=mask_path,
                        question_type=sample.question_type,
                    )

                    pred, parse_success = extract_prediction(
                        model_resp.parsed_json,
                        model_resp.raw_text,
                        sample.question_type,
                    )

                    invalid_answer_format, invalid_option, invalid_values = (
                        validate_prediction_against_choices(
                            pred, sample.choices, sample.question_type
                        )
                    )

                    format_repaired = 0
                    original_raw_text = model_resp.raw_text

                    if self._should_repair_format(
                        parse_success, invalid_answer_format, invalid_option,
                        model_resp.raw_text,
                    ):
                        try:
                            repaired_resp = self.model.repair_format(
                                bad_raw_text=model_resp.raw_text,
                                question_type=sample.question_type,
                                choices=sample.choices,
                            )
                            repaired_pred, repaired_parse_success = extract_prediction(
                                repaired_resp.parsed_json,
                                repaired_resp.raw_text,
                                sample.question_type,
                            )
                            (
                                repaired_invalid_fmt,
                                repaired_invalid_opt,
                                repaired_invalid_vals,
                            ) = validate_prediction_against_choices(
                                repaired_pred, sample.choices, sample.question_type
                            )
                            if (
                                repaired_parse_success
                                and not repaired_invalid_fmt
                                and not repaired_invalid_opt
                            ):
                                model_resp = repaired_resp
                                pred = repaired_pred
                                parse_success = repaired_parse_success
                                invalid_answer_format = repaired_invalid_fmt
                                invalid_option = repaired_invalid_opt
                                invalid_values = repaired_invalid_vals
                                format_repaired = 1
                        except Exception:
                            pass

                    score = score_prediction(pred, sample.answer, sample.question_type)

                    if self.save_raw_response and model_resp.raw_response is not None:
                        save_json(model_resp.raw_response, raw_dir / f"{sample.question_id}.json")

                    row = {
                        **base_row,
                        **self._extract_usage_fields(model_resp.usage),
                        "api_success": 1,
                        "parse_success": int(parse_success),
                        "invalid_answer_format": int(invalid_answer_format),
                        "invalid_option": int(invalid_option),
                        "invalid_values": str(invalid_values),
                        "format_repaired": int(format_repaired),
                        "pred_answer": str(score.pred_norm),
                        "gold_answer_normalized": str(score.gold_norm),
                        "exact_correct": score.exact_correct,
                        "partial_score": score.partial_score,
                        "latency_sec": model_resp.latency_sec,
                        "finish_reason": model_resp.finish_reason,
                        "visible_reasoning": model_resp.visible_reasoning,
                        "raw_text": model_resp.raw_text,
                        "raw_text_before_repair": (
                            original_raw_text if format_repaired else ""
                        ),
                        "error": "",
                        "traceback": "",
                    }
                    rows.append(row)

                except Exception as e:
                    rows.append(
                        {
                            **base_row,
                            "usage_prompt_tokens": 0.0,
                            "usage_completion_tokens": 0.0,
                            "usage_total_tokens": 0.0,
                            "api_success": 0,
                            "parse_success": 0,
                            "invalid_answer_format": 0,
                            "invalid_option": 0,
                            "invalid_values": "[]",
                            "format_repaired": 0,
                            "pred_answer": "",
                            "gold_answer_normalized": "",
                            "exact_correct": 0,
                            "partial_score": 0.0,
                            "latency_sec": 0.0,
                            "finish_reason": "error",
                            "visible_reasoning": "",
                            "raw_text": "",
                            "raw_text_before_repair": "",
                            "error": f"{type(e).__name__}: {e}",
                            "traceback": traceback.format_exc(),
                        }
                    )

                rows_since_checkpoint += 1
                if (
                    self.checkpoint_every_n > 0
                    and rows_since_checkpoint >= self.checkpoint_every_n
                ):
                    self._write_checkpoint(rows, "periodic")
                    rows_since_checkpoint = 0

        except KeyboardInterrupt:
            self._write_checkpoint(rows, "keyboard_interrupt")
            raise

        if rows_since_checkpoint > 0:
            self._write_checkpoint(rows, "final_partial")

        df = pd.DataFrame(rows)
        write_reports(df, self.output_dir, self.experiment_name)
        return df
