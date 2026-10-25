from __future__ import annotations

from pathlib import Path

import pandas as pd

from .scoring import (
    build_group_metrics,
    build_overall_metrics,
    build_vc_joint_metrics,
    build_vc_joint_summary,
)
from .utils import ensure_dir, save_json


def _get_excel_engine() -> str | None:
    try:
        import openpyxl  # noqa: F401
        return "openpyxl"
    except Exception:
        return None


def _write_excel_sheets(
    sheets: list[tuple[str, pd.DataFrame]],
    output_dir: Path,
    experiment_name: str,
) -> None:
    summary_path = output_dir / f"{experiment_name}_summary.xlsx"
    engine = _get_excel_engine()
    writer_kwargs = {"engine": engine} if engine else {}

    try:
        with pd.ExcelWriter(summary_path, **writer_kwargs) as writer:
            for sheet_name, sheet_df in sheets:
                sheet_df.to_excel(writer, index=False, sheet_name=sheet_name[:31])
    except Exception as e:
        fallback_dir = ensure_dir(output_dir / f"{experiment_name}_summary_csv")
        for sheet_name, sheet_df in sheets:
            sheet_df.to_csv(fallback_dir / f"{sheet_name[:31]}.csv", index=False)

        save_json(
            {
                "summary_xlsx": str(summary_path),
                "fallback_dir": str(fallback_dir),
                "error": f"{type(e).__name__}: {e}",
            },
            output_dir / f"{experiment_name}_summary_excel_error.json",
        )


def write_reports(
    df: pd.DataFrame, output_dir: str | Path, experiment_name: str
) -> None:
    output_dir = ensure_dir(output_dir)

    # 1. Per-sample result CSV.
    df.to_csv(output_dir / f"{experiment_name}_per_sample.csv", index=False)

    # 2. Overall metrics JSON.
    overall = build_overall_metrics(df)
    save_json(overall, output_dir / f"{experiment_name}_overall.json")

    # 3. VC joint scoring.
    vc_joint_summary = build_vc_joint_summary(df)
    if vc_joint_summary:
        save_json(vc_joint_summary, output_dir / f"{experiment_name}_vc_joint.json")
        overall.update(vc_joint_summary)
        save_json(overall, output_dir / f"{experiment_name}_overall.json")

    # 4. Grouping dimensions.
    group_cols = [
        "category",
        "subcategory",
        "template_id",
        "question_type",
        "referring_style",
        "requires_mirror_reasoning",
        "requires_depth",
        "model_name",
    ]

    sheets: list[tuple[str, pd.DataFrame]] = [
        ("overall", pd.DataFrame([overall])),
    ]

    for col in group_cols:
        if col not in df.columns:
            continue
        gdf = build_group_metrics(df, col)
        sheets.append((str(col), gdf))

    # Keep question-type metrics with random baseline as an explicit sheet.
    if "question_type" in df.columns:
        qt_df = build_group_metrics(df, "question_type")
        if not qt_df.empty:
            sheets.append(("qtype_with_random", qt_df))

    # VC joint-scoring details.
    vc_joint_df = build_vc_joint_metrics(df)
    if not vc_joint_df.empty:
        sheets.append(("vc_joint_detail", vc_joint_df))

    # Error and failure-analysis sheets.
    if not df.empty:
        if "exact_correct" in df.columns and (df["exact_correct"] == 0).any():
            sheets.append(("errors", df[df["exact_correct"] == 0]))
        if "parse_success" in df.columns and (df["parse_success"] == 0).any():
            sheets.append(("parse_failures", df[df["parse_success"] == 0]))
        if "invalid_option" in df.columns and (df["invalid_option"] == 1).any():
            sheets.append(("invalid_options", df[df["invalid_option"] == 1]))
        if (
            "invalid_answer_format" in df.columns
            and (df["invalid_answer_format"] == 1).any()
        ):
            sheets.append(
                ("invalid_format", df[df["invalid_answer_format"] == 1])
            )

    _write_excel_sheets(sheets, output_dir, experiment_name)
