from __future__ import annotations

import re
from dataclasses import dataclass
from math import comb
from typing import Any, Dict, Optional

import pandas as pd

from .parsing import normalize_answer, _extract_choice_letter


# Display order for subcategories.
SUBCATEGORY_ORDER = [
    "visibility_classification",
    "mirror_correspondence_r2m",
    "mirror_correspondence_m2r",
    "mirror_correspondence",
    "same_object_verification",
    "spatial_relation_reasoning",
    "mirror_to_real_spatial_inference",
    "relative_distance_ordering",
    "counting_based_spatial_reasoning",
    "reachability",
    "navigation_movement_planning",
    "safety_feasibility_awareness",
    "next_step_action_selection_real_target",
    "collision_risk_awareness",
]

SUBCATEGORY_RANK = {name: idx for idx, name in enumerate(SUBCATEGORY_ORDER)}


@dataclass
class ScoreResult:
    exact_correct: int
    partial_score: float
    pred_norm: Any
    gold_norm: Any


# ========================
#  Random baseline
# ========================

def compute_random_baseline(
    choices: Any, answer: Any, question_type: str
) -> float:
    """
    Compute the theoretical random baseline for a single question.

    - yes_no:                  1/len(choices)  (1/2 or 1/3)
    - single_choice:           1/len(choices)
    - single_choice_plus_text: 1/len(choices)
    - multi_choice:            1/C(n,k), k=len(gold)
    - spatial_3d_axes:         1/len(choices)
    - count:                   0.0 (open-ended)
    """
    if question_type == "count":
        return 0.0

    if not isinstance(choices, list) or len(choices) == 0:
        # Some yes_no samples have no choices field; default to 1/2.
        if question_type == "yes_no":
            return 0.5
        return 0.0

    n = len(choices)

    if question_type == "multi_choice":
        # Assume a random answer selects k items, where k = len(gold_answer).
        gold_norm = normalize_answer(answer, "multi_choice")
        k = len(gold_norm) if isinstance(gold_norm, list) and gold_norm else 1
        if k > n:
            k = n
        c_n_k = comb(n, k)
        return 1.0 / c_n_k if c_n_k > 0 else 0.0

    # single_choice, yes_no, spatial_3d_axes, single_choice_plus_text
    return 1.0 / n


# ========================
#  Spatial axis scoring helpers
# ========================

def _normalize_axis_value(axis: str, value: Any) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip().lower()
    if not text:
        return None
    text = text.replace("_", " ").replace("-", " ")
    text = re.sub(r"\s+", " ", text)

    if axis == "horizontal":
        if re.search(r"\bleft\b", text):
            return "left"
        if re.search(r"\bright\b", text):
            return "right"
        if re.search(r"\b(center|centre|middle|same horizontal|same x)\b", text):
            return "center"
        return None
    if axis == "vertical":
        if re.search(r"\b(upper|up|above|top)\b", text):
            return "upper"
        if re.search(r"\b(lower|down|below|bottom)\b", text):
            return "lower"
        if re.search(r"\b(center|centre|middle|same vertical|same y)\b", text):
            return "center"
        return None
    if axis == "depth":
        if re.search(r"\b(in front|front|closer|nearer)\b", text):
            return "front"
        if re.search(r"\b(behind|back|farther|further)\b", text):
            return "behind"
        if re.search(r"\b(same depth|same z|same distance|equal depth)\b", text):
            return "center"
        return None
    return None


def _extract_spatial_axes(answer: Any) -> Dict[str, Optional[str]]:
    axes = {"horizontal": None, "vertical": None, "depth": None}
    if answer is None:
        return axes
    if isinstance(answer, dict):
        for axis in axes.keys():
            axes[axis] = _normalize_axis_value(axis, answer.get(axis))
        return axes
    text = str(answer).strip().lower()
    if not text:
        return axes
    text = text.replace("_", " ").replace("-", " ")
    text = re.sub(r"\s+", " ", text)
    axes["horizontal"] = _normalize_axis_value("horizontal", text)
    axes["vertical"] = _normalize_axis_value("vertical", text)
    axes["depth"] = _normalize_axis_value("depth", text)
    return axes


def _score_spatial_3d_axes_partial(pred_norm: Any, gold_norm: Any) -> float:
    pred_axes = _extract_spatial_axes(pred_norm)
    gold_axes = _extract_spatial_axes(gold_norm)
    axes_to_score = [a for a in ["horizontal", "vertical", "depth"] if gold_axes[a] is not None]
    if len(axes_to_score) == 0:
        return float(int(pred_norm == gold_norm))
    matched = sum(1 for a in axes_to_score if pred_axes[a] == gold_axes[a])
    return float(matched) / float(len(axes_to_score))


# ========================
#  Score prediction
# ========================

def score_prediction(pred: Any, gold: Any, question_type: str) -> ScoreResult:
    if gold is None:
        return ScoreResult(exact_correct=0, partial_score=0.0, pred_norm=pred, gold_norm=gold)

    pred_norm = normalize_answer(pred, question_type)
    gold_norm = normalize_answer(gold, question_type)

    if question_type == "multi_choice":
        pred_set = set(pred_norm)
        gold_set = set(gold_norm)
        exact_correct = int(pred_set == gold_set)
        union = pred_set | gold_set
        partial = 1.0 if len(union) == 0 else len(pred_set & gold_set) / len(union)
        return ScoreResult(
            exact_correct=exact_correct, partial_score=float(partial),
            pred_norm=sorted(pred_set), gold_norm=sorted(gold_set),
        )

    if question_type == "spatial_3d_axes":
        exact_correct = int(pred_norm == gold_norm)
        partial = _score_spatial_3d_axes_partial(pred_norm, gold_norm)
        return ScoreResult(
            exact_correct=exact_correct, partial_score=float(partial),
            pred_norm=pred_norm, gold_norm=gold_norm,
        )

    # yes_no, single_choice, single_choice_plus_text, and count use exact match.
    exact_correct = int(pred_norm == gold_norm)
    return ScoreResult(
        exact_correct=exact_correct, partial_score=float(exact_correct),
        pred_norm=pred_norm, gold_norm=gold_norm,
    )


# ========================
#  VC joint scoring
# ========================

def _compute_vc_group_key(question_id: str) -> str:
    return re.sub(r"vc_00[123]", "vc_XXX", question_id)


def build_vc_joint_metrics(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "template_id" not in df.columns:
        return pd.DataFrame()

    vc_df = df[df["template_id"].isin({"vc_001", "vc_002", "vc_003"})].copy()
    if vc_df.empty:
        return pd.DataFrame()

    vc_df["vc_group_key"] = vc_df["question_id"].apply(_compute_vc_group_key)

    group_results = (
        vc_df.groupby("vc_group_key")
        .agg(
            n_questions=("question_id", "count"),
            n_correct=("exact_correct", "sum"),
            question_ids=("question_id", lambda x: list(x)),
        )
        .reset_index()
    )

    group_results["joint_correct"] = (
        (group_results["n_questions"] == 3) & (group_results["n_correct"] == 3)
    ).astype(int)

    return group_results


def build_vc_joint_summary(df: pd.DataFrame) -> dict[str, float]:
    joint_df = build_vc_joint_metrics(df)
    if joint_df.empty:
        return {}

    complete_groups = joint_df[joint_df["n_questions"] == 3]
    if complete_groups.empty:
        return {}

    vc_df = df[df["template_id"].isin({"vc_001", "vc_002", "vc_003"})]

    return {
        "vc_joint_n_groups": int(len(complete_groups)),
        "vc_joint_acc": float(complete_groups["joint_correct"].mean()),
        "vc_individual_acc": float(vc_df["exact_correct"].mean()),
    }


# ========================
#  VC joint collapsing for exact_acc / partial_acc
# ========================

def _collapse_vc_to_joint(df: pd.DataFrame) -> pd.DataFrame:
    """
    Collapse each VC triplet (vc_001/002/003) into one row so exact_acc and
    partial_acc use joint scoring. Non-VC questions are kept unchanged.
    """
    if df.empty or "template_id" not in df.columns:
        return df

    vc_mask = df["template_id"].isin({"vc_001", "vc_002", "vc_003"})
    non_vc_df = df[~vc_mask].copy()
    vc_df = df[vc_mask].copy()

    if vc_df.empty:
        return df

    vc_df["_vc_group_key"] = vc_df["question_id"].apply(_compute_vc_group_key)

    joint_rows = []
    for group_key, group in vc_df.groupby("_vc_group_key"):
        first = group.iloc[0]
        n = len(group)
        n_correct = int(group["exact_correct"].sum())
        joint_correct = 1 if (n == 3 and n_correct == 3) else 0
        partial = float(n_correct) / 3.0

        row = first.to_dict()
        row["question_id"] = group_key
        row["template_id"] = "vc_joint"
        row["exact_correct"] = joint_correct
        row["partial_score"] = partial
        row["latency_sec"] = float(group["latency_sec"].mean())
        row["usage_prompt_tokens"] = float(group["usage_prompt_tokens"].mean())
        row["usage_completion_tokens"] = float(group["usage_completion_tokens"].mean())
        row["usage_total_tokens"] = float(group["usage_total_tokens"].mean())
        row["api_success"] = int(group["api_success"].min())
        row["parse_success"] = int(group["parse_success"].min())
        row["invalid_answer_format"] = int(group["invalid_answer_format"].max())
        row["invalid_option"] = int(group["invalid_option"].max())
        # random_baseline: VC questions are yes_no. The joint baseline is the
        # product of the three per-question baselines.
        if "random_baseline" in group.columns:
            row["random_baseline"] = float(group["random_baseline"].prod())
        joint_rows.append(row)

    joint_df = pd.DataFrame(joint_rows)
    return pd.concat([non_vc_df, joint_df], ignore_index=True)


def _get_scorable_df(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    return _collapse_vc_to_joint(df)


# ========================
#  Grouping and summaries
# ========================

def _sort_group_metrics(
    out: pd.DataFrame, df_eval: pd.DataFrame, group_col: str
) -> pd.DataFrame:
    if out.empty:
        return out

    if group_col == "subcategory":
        out = out.copy()
        out["_sort_rank"] = out["subcategory"].map(
            lambda x: SUBCATEGORY_RANK.get(str(x), 10**9)
        )
        out["_sort_name"] = out["subcategory"].astype(str)
        out = out.sort_values(["_sort_rank", "_sort_name"], ascending=[True, True]).drop(
            columns=["_sort_rank", "_sort_name"]
        )
        return out.reset_index(drop=True)

    if group_col == "template_id":
        out = out.copy()
        template_to_subcat = (
            df_eval.groupby(["template_id", "subcategory"], dropna=False)
            .size().reset_index(name="cnt")
            .sort_values(["template_id", "cnt", "subcategory"], ascending=[True, False, True])
            .drop_duplicates(subset=["template_id"], keep="first")
            .set_index("template_id")["subcategory"].to_dict()
        )
        out["_subcategory"] = out["template_id"].map(template_to_subcat)
        out["_sort_rank"] = out["_subcategory"].map(lambda x: SUBCATEGORY_RANK.get(str(x), 10**9))
        out["_sort_name"] = out["template_id"].astype(str)
        out = out.sort_values(["_sort_rank", "_sort_name"], ascending=[True, True]).drop(
            columns=["_subcategory", "_sort_rank", "_sort_name"]
        )
        return out.reset_index(drop=True)

    return out.sort_values(["n", group_col], ascending=[False, True]).reset_index(drop=True)


def build_group_metrics(df: pd.DataFrame, group_col: str) -> pd.DataFrame:
    df_eval = _get_scorable_df(df)
    if df_eval.empty:
        return pd.DataFrame()

    agg_dict = {
        "n": ("question_id", "count"),
        "exact_acc": ("exact_correct", "mean"),
        "partial_acc": ("partial_score", "mean"),
        "api_success_rate": ("api_success", "mean"),
        "parse_success_rate": ("parse_success", "mean"),
        "invalid_answer_format_rate": ("invalid_answer_format", "mean"),
        "invalid_option_rate": ("invalid_option", "mean"),
        "avg_latency_sec": ("latency_sec", "mean"),
        "avg_prompt_tokens": ("usage_prompt_tokens", "mean"),
        "avg_completion_tokens": ("usage_completion_tokens", "mean"),
        "avg_total_tokens": ("usage_total_tokens", "mean"),
    }

    # random_baseline may be absent in older result files.
    if "random_baseline" in df_eval.columns:
        agg_dict["random_baseline"] = ("random_baseline", "mean")

    out = df_eval.groupby(group_col, dropna=False).agg(**agg_dict).reset_index()

    # Compute norm_exact_acc.
    if "random_baseline" in out.columns:
        denom = (1 - out["random_baseline"]).clip(lower=0.001)
        out["norm_exact_acc"] = (out["exact_acc"] - out["random_baseline"]) / denom

    out = _sort_group_metrics(out, df_eval, group_col)
    return out


def build_overall_metrics(df: pd.DataFrame) -> dict[str, float]:
    df_eval = _get_scorable_df(df)

    if df_eval.empty:
        return {
            "n": 0, "exact_acc": 0.0, "partial_acc": 0.0,
            "api_success_rate": 0.0, "parse_success_rate": 0.0,
            "invalid_answer_format_rate": 0.0, "invalid_option_rate": 0.0,
            "avg_latency_sec": 0.0, "avg_prompt_tokens": 0.0,
            "avg_completion_tokens": 0.0, "avg_total_tokens": 0.0,
        }

    result = {
        "n": int(len(df_eval)),
        "exact_acc": float(df_eval["exact_correct"].mean()),
        "partial_acc": float(df_eval["partial_score"].mean()),
        "api_success_rate": float(df_eval["api_success"].mean()),
        "parse_success_rate": float(df_eval["parse_success"].mean()),
        "invalid_answer_format_rate": float(df_eval["invalid_answer_format"].mean()),
        "invalid_option_rate": float(df_eval["invalid_option"].mean()),
        "avg_latency_sec": float(df_eval["latency_sec"].mean()),
        "avg_prompt_tokens": float(df_eval["usage_prompt_tokens"].mean()),
        "avg_completion_tokens": float(df_eval["usage_completion_tokens"].mean()),
        "avg_total_tokens": float(df_eval["usage_total_tokens"].mean()),
    }

    if "random_baseline" in df_eval.columns:
        rb = float(df_eval["random_baseline"].mean())
        result["random_baseline"] = rb
        denom = max(1 - rb, 0.001)
        result["norm_exact_acc"] = (result["exact_acc"] - rb) / denom

    return result
