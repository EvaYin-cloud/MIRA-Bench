"""
Verify whether depth data is passed correctly through the benchmark pipeline.

Usage:
    python verify_depth.py \
        --rgb_csv  /path/to/rgb_only_per_sample.csv \
        --rgbd_csv /path/to/rgbd_per_sample.csv

Or check only one RGBD experiment:
    python verify_depth.py --rgbd_csv /path/to/rgbd_per_sample.csv
"""

import argparse
import pandas as pd
import numpy as np
from pathlib import Path


def check_depth_paths(df: pd.DataFrame, label: str) -> None:
    """Check whether the depth_path_used column has values."""
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")
    print(f"  Total questions: {len(df)}")

    if "depth_path_used" not in df.columns:
        print("  X Missing depth_path_used column. benchmark_runner.py may be outdated.")
        return

    has_depth = df["depth_path_used"].notna() & (df["depth_path_used"] != "") & (df["depth_path_used"] != "None")
    print(f"  Rows with depth paths: {has_depth.sum()} / {len(df)}")
    print(f"  Rows without depth paths: {(~has_depth).sum()} / {len(df)}")

    if has_depth.sum() > 0:
        # Check whether files exist.
        sample_paths = df.loc[has_depth, "depth_path_used"].head(5)
        print(f"\n  First 5 depth paths:")
        for p in sample_paths:
            exists = Path(str(p)).exists() if pd.notna(p) else False
            status = "OK exists" if exists else "X missing"
            print(f"    {status}: {p}")

    if "use_depth" in df.columns:
        print(f"\n  use_depth value distribution:")
        print(f"    {df['use_depth'].value_counts().to_dict()}")


def compare_results(rgb_df: pd.DataFrame, rgbd_df: pd.DataFrame) -> None:
    """Compare RGB and RGBD result differences."""
    print(f"\n{'='*60}")
    print(f"  RGB vs RGBD result comparison")
    print(f"{'='*60}")

    # Merge by question_id.
    merged = rgb_df.merge(
        rgbd_df,
        on="question_id",
        suffixes=("_rgb", "_rgbd"),
        how="inner",
    )
    print(f"  Matched questions: {len(merged)}")

    if len(merged) == 0:
        print("  X No matching question_id values; cannot compare.")
        return

    # 1. Ratio of exactly identical pred_answer values.
    same_pred = (merged["pred_answer_rgb"] == merged["pred_answer_rgbd"]).mean()
    print(f"\n  Identical predicted answers: {same_pred:.1%}")

    # 2. exact_correct comparison.
    rgb_acc = merged["exact_correct_rgb"].mean()
    rgbd_acc = merged["exact_correct_rgbd"].mean()
    print(f"  RGB  exact_acc: {rgb_acc:.4f}")
    print(f"  RGBD exact_acc: {rgbd_acc:.4f}")
    print(f"  Difference: {rgbd_acc - rgb_acc:+.4f}")

    # 3. Analyze by question_type.
    if "question_type_rgb" in merged.columns:
        print(f"\n  Comparison by question_type:")
        for qt in sorted(merged["question_type_rgb"].unique()):
            sub = merged[merged["question_type_rgb"] == qt]
            rgb_a = sub["exact_correct_rgb"].mean()
            rgbd_a = sub["exact_correct_rgbd"].mean()
            same = (sub["pred_answer_rgb"] == sub["pred_answer_rgbd"]).mean()
            print(f"    {qt:30s}  RGB={rgb_a:.3f}  RGBD={rgbd_a:.3f}  same_pred={same:.1%}")

    # 4. Find questions where RGB and RGBD predictions differ.
    diff_mask = merged["pred_answer_rgb"] != merged["pred_answer_rgbd"]
    diff = merged[diff_mask]
    print(f"\n  Questions with different answers: {len(diff)} / {len(merged)}")
    if len(diff) > 0 and len(diff) <= 20:
        print(f"\n  Different-answer details:")
        for _, row in diff.iterrows():
            print(f"    {row['question_id']}")
            print(f"      RGB:  pred={row['pred_answer_rgb']}  gold={row.get('gold_answer_rgb', '?')}  correct={row['exact_correct_rgb']}")
            print(f"      RGBD: pred={row['pred_answer_rgbd']}  gold={row.get('gold_answer_rgbd', '?')}  correct={row['exact_correct_rgbd']}")
    elif len(diff) > 20:
        print(f"    Too many rows; showing the first 10.")
        for _, row in diff.head(10).iterrows():
            print(f"    {row['question_id']}")
            print(f"      RGB:  pred={row['pred_answer_rgb']}  correct={row['exact_correct_rgb']}")
            print(f"      RGBD: pred={row['pred_answer_rgbd']}  correct={row['exact_correct_rgbd']}")

    # 5. Ratio of exactly identical raw_text values, the strongest similarity signal.
    if "raw_text_rgb" in merged.columns and "raw_text_rgbd" in merged.columns:
        same_raw = (merged["raw_text_rgb"] == merged["raw_text_rgbd"]).mean()
        print(f"\n  Identical raw_text values: {same_raw:.1%}")
        if same_raw > 0.9:
            print("  WARNING: More than 90% of raw outputs are identical; depth may not be used effectively.")
            print("     Possible causes:")
            print("     1. The depth image was not actually sent to the model; check depth_path_used.")
            print("     2. The prompt did not tell the model that the second image is a depth map; check prompt_builder.")
            print("     3. The model itself may be insensitive to depth, which is common for some 7B models.")


def check_parse_failures(df: pd.DataFrame, label: str) -> None:
    """Print details for parse failures."""
    failures = df[df["parse_success"] == 0]
    if len(failures) == 0:
        print(f"\n  {label}: no parse failures")
        return

    print(f"\n  {label}: {len(failures)} parse failures")
    for _, row in failures.iterrows():
        print(f"    question_id: {row['question_id']}")
        print(f"    question_type: {row.get('question_type', '?')}")
        print(f"    raw_text: {str(row.get('raw_text', ''))[:200]}")
        print(f"    error: {row.get('error', '')}")
        print()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rgb_csv", type=str, default=None, help="per_sample.csv from the RGB-only experiment")
    parser.add_argument("--rgbd_csv", type=str, required=True, help="per_sample.csv from the RGBD experiment")
    args = parser.parse_args()

    rgbd_df = pd.read_csv(args.rgbd_csv)
    check_depth_paths(rgbd_df, "RGBD experiment")
    check_parse_failures(rgbd_df, "RGBD")

    if args.rgb_csv:
        rgb_df = pd.read_csv(args.rgb_csv)
        check_depth_paths(rgb_df, "RGB experiment")
        check_parse_failures(rgb_df, "RGB")
        compare_results(rgb_df, rgbd_df)
    else:
        print("\n  Tip: add --rgb_csv to compare RGB and RGBD result differences.")


if __name__ == "__main__":
    main()
