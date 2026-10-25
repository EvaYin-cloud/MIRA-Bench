from __future__ import annotations

import ast
import json
import re
from typing import Any


YES_SYNONYMS = {"yes", "true", "correct"}
NO_SYNONYMS = {"no", "false", "incorrect"}
NOT_SURE_SYNONYMS = {
    "not sure",
    "unknown",
    "uncertain",
    "cannot determine",
    "can't determine",
}


def _normalize_label_like_string(s: str) -> str:
    """
    Normalize label-like strings, especially Mirror_00-style labels from the
    choice board.

    Examples:
      "Mirror_00" -> "00"
      "mirror_01" -> "01"
      "Mirror-02" -> "02"
      " 15 " -> "15"
    """
    s = str(s).strip()

    m = re.fullmatch(r"(?i)mirror[_\-\s]*([0-9]+)", s)
    if m:
        return m.group(1)

    return s


def normalize_atomic_answer(x: Any) -> str:
    if x is None:
        return ""
    return _normalize_label_like_string(str(x).strip())


def _extract_choice_letter(text: Any) -> str:
    """
    Extract the option letter from a single_choice_plus_text answer.
    Examples:
      "C. The mirror reveals..." -> "C"
      "C" -> "C"
      "c. something" -> "C"
      "D" -> "D"
    """
    if text is None:
        return ""
    s = str(text).strip()
    if not s:
        return ""

    # Match answers that start with "X." or "X)".
    m = re.match(r"^([A-Za-z])\s*[.\)]\s*", s)
    if m:
        return m.group(1).upper()

    # A single letter is already the answer.
    if len(s) == 1 and s.isalpha():
        return s.upper()

    return s


def normalize_yes_no(x: Any) -> str:
    s = normalize_atomic_answer(x).lower()
    if s in YES_SYNONYMS:
        return "yes"
    if s in NO_SYNONYMS:
        return "no"
    if s in NOT_SURE_SYNONYMS:
        return "not sure"
    return normalize_atomic_answer(x)


def _looks_like_analysis_blob(text: str) -> bool:
    """
    Detect long analysis or malformed-schema text that should not be treated as
    a real answer.
    """
    if not text:
        return False

    t = text.strip()

    # Long paragraphs are usually not answers.
    if len(t) > 200:
        return True

    lowered = t.lower()

    suspicious_markers = [
        '"analysis"',
        '"task"',
        '"reasoning"',
        '"explanation"',
        "analysis:",
        "task:",
        "reasoning:",
        "step 1",
        "step 2",
        "locate mirror region",
        "scan mirror region",
        "compare:",
        "wait, let me",
    ]

    return any(marker in lowered for marker in suspicious_markers)


def _try_parse_json_obj(raw_text: str) -> dict[str, Any] | None:
    """
    Try to parse raw_text as a JSON object.
    """
    if not raw_text:
        return None

    text = raw_text.strip()
    if not (text.startswith("{") and text.endswith("}")):
        return None

    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except Exception:
        return None

    return None


def _extract_final_answer_from_raw_text(raw_text: str) -> Any | None:
    """
    Conservatively recover final_answer from raw_text.
    - Use raw_text itself when it is JSON and contains final_answer.
    - Otherwise, look for a "final_answer": ... fragment.
    Return None if neither strategy works.
    """
    if not raw_text:
        return None

    # 1) Try the whole text as JSON first.
    obj = _try_parse_json_obj(raw_text)
    if isinstance(obj, dict) and "final_answer" in obj:
        return obj["final_answer"]

    # 2) Then try extracting a "final_answer": ... fragment with regex.
    m = re.search(r'"final_answer"\s*:\s*(\[.*?\]|".*?"|[^,\}\n]+)', raw_text, re.DOTALL)
    if not m:
        return None

    candidate = m.group(1).strip()

    # Try literal_eval first.
    try:
        return ast.literal_eval(candidate)
    except Exception:
        pass

    # Then try json.loads.
    try:
        return json.loads(candidate)
    except Exception:
        pass

    # Final fallback: return the candidate as a string.
    return candidate


def _coerce_to_list(answer: Any) -> list[Any]:
    """
    Coerce possible model-output forms into a list, mainly for multi_choice.
    """
    if answer is None:
        return []

    if isinstance(answer, list):
        return answer

    if isinstance(answer, str):
        stripped = answer.strip()

        # Treat obvious analysis or malformed-schema blobs as empty.
        if _looks_like_analysis_blob(stripped):
            return []

        # Parse strings like "['1', '2']" or '["1","2"]'.
        try:
            obj = ast.literal_eval(stripped)
            if isinstance(obj, list):
                return obj
        except Exception:
            pass

        # Split on commas only for short text to avoid slicing long analysis
        # paragraphs into junk answers.
        if "," in stripped and len(stripped) <= 80:
            return [v.strip() for v in stripped.split(",") if v.strip()]

        # Treat short text as a single answer.
        if len(stripped) <= 40:
            return [stripped]

        # Long text is not treated as an answer.
        return []

    # Wrap all other types as a single-item list.
    return [answer]


# =========================
# spatial_3d_axes normalization
# =========================

def _normalize_spatial_text(s: str) -> str:
    s = str(s).strip().lower()
    if not s:
        return ""

    # Normalize separators.
    s = s.replace("_", " ")
    s = s.replace("-", " ")
    s = re.sub(r"\s+", " ", s).strip()

    # Remove trailing "of".
    s = re.sub(r"\s+of$", "", s)

    return s


def _extract_spatial_horizontal(text: str) -> str | None:
    if re.search(r"\bleft\b", text):
        return "left"
    if re.search(r"\bright\b", text):
        return "right"
    if re.search(r"\b(aligned|align|center|centre|same horizontal|same x)\b", text):
        return "aligned"
    return None


def _extract_spatial_vertical(text: str) -> str | None:
    # Align with the annotation scheme: upper / lower.
    if re.search(r"\b(upper|above|top|up)\b", text):
        return "upper"
    if re.search(r"\b(lower|below|bottom|down|under|beneath)\b", text):
        return "lower"
    if re.search(r"\b(aligned|align|center|centre|same vertical|same y)\b", text):
        return "aligned"
    return None


def _extract_spatial_depth(text: str) -> str | None:
    # Align with the annotation scheme: front / back.
    if re.search(r"\b(front|in front|closer|nearer)\b", text):
        return "front"
    if re.search(r"\b(back|behind|farther|further)\b", text):
        return "back"
    if re.search(r"\b(same depth|same_depth|equal depth|same z|aligned depth)\b", text):
        return "same_depth"
    return None


def _canonicalize_spatial_3d_axes_answer(answer: Any) -> str:
    """
    Canonicalize spatial_3d_axes answer variants into forms such as:
      upper-front-left of
      lower-back-left of
      aligned-same_depth-right of

    Rules:
    - above -> upper
    - below -> lower
    - behind -> back
    - front -> front
    - aligned -> aligned
    - same depth -> same_depth
    - Output order is fixed as: vertical-depth-horizontal of
    """
    if answer is None:
        return ""

    text = _normalize_spatial_text(str(answer))
    if not text:
        return ""

    vertical = _extract_spatial_vertical(text)
    depth = _extract_spatial_depth(text)
    horizontal = _extract_spatial_horizontal(text)

    # If no axis can be extracted, fall back to normal atomic normalization.
    if vertical is None and depth is None and horizontal is None:
        return normalize_atomic_answer(answer)

    # Fill missing axes with aligned / same_depth to match the annotation
    # standard.
    if vertical is None:
        vertical = "aligned"
    if depth is None:
        depth = "same_depth"
    if horizontal is None:
        horizontal = "aligned"

    return f"{vertical}-{depth}-{horizontal} of"


def normalize_answer(answer: Any, question_type: str) -> Any:
    """
    Normalize model outputs and gold answers.

    Rules:
    - multi_choice:
        * coerce to list
        * deduplicate
        * remove empty strings
        * sort
        * normalize Mirror_00 -> 00
    - yes_no:
        * normalize synonyms
    - spatial_3d_axes:
        * map above/below/front/behind onto upper/lower/front/back
        * fix output order as vertical-depth-horizontal of
    - single_choice:
        * standard string normalization
    - count:
        * numeric-string normalization
    - single_choice_plus_text:
        * extract the option letter, e.g. "C. xxx" -> "C"
    """
    if question_type == "count":
        if isinstance(answer, list):
            answer = answer[0] if len(answer) == 1 else ""
        return normalize_atomic_answer(answer)

    if question_type == "single_choice_plus_text":
        if isinstance(answer, list):
            answer = answer[0] if len(answer) == 1 else ""
        return _extract_choice_letter(answer)

    if question_type == "multi_choice":
        vals = _coerce_to_list(answer)

        norm = sorted(
            {
                normalize_atomic_answer(v)
                for v in vals
                if normalize_atomic_answer(v) != ""
            }
        )
        return norm

    if isinstance(answer, list):
        if len(answer) == 1:
            answer = answer[0]
        else:
            return [normalize_atomic_answer(v) for v in answer]

    if question_type == "yes_no":
        return normalize_yes_no(answer)

    if question_type == "spatial_3d_axes":
        if isinstance(answer, str) and _looks_like_analysis_blob(answer):
            return ""
        return _canonicalize_spatial_3d_axes_answer(answer)

    # For non-multi_choice answers, do not accept long analysis text as an
    # answer.
    if isinstance(answer, str) and _looks_like_analysis_blob(answer):
        return ""

    return normalize_atomic_answer(answer)


def extract_prediction(
    parsed_json: dict[str, Any] | None,
    raw_text: str,
    question_type: str,
) -> tuple[Any, bool]:
    """
    Returns:
    - normalized prediction
    - parse_success

    Logic:
    1. Prefer parsed_json["final_answer"].
    2. If parsed_json has no final_answer, conservatively recover from raw_text.
    3. If recovery fails, return an empty answer instead of treating all
       raw_text as the answer.
    """
    if parsed_json and "final_answer" in parsed_json:
        return normalize_answer(parsed_json["final_answer"], question_type), True

    rescued = _extract_final_answer_from_raw_text(raw_text)
    if rescued is not None:
        return normalize_answer(rescued, question_type), False

    # Do not treat the whole raw_text as the prediction.
    if question_type == "multi_choice":
        return [], False
    return "", False


def validate_prediction_against_choices(
    pred: Any,
    choices: Any,
    question_type: str,
) -> tuple[bool, bool, list[str]]:
    """
    Returns:
    - invalid_answer_format
    - invalid_option
    - invalid_values
    """
    invalid_answer_format = False
    invalid_option = False
    invalid_values: list[str] = []

    if question_type == "multi_choice":
        if not isinstance(pred, list):
            invalid_answer_format = True
            return invalid_answer_format, invalid_option, invalid_values

        if not isinstance(choices, list):
            invalid_answer_format = True
            return invalid_answer_format, invalid_option, invalid_values

        allowed = {str(c) for c in choices}
        bad = [str(v) for v in pred if str(v) not in allowed]

        if bad:
            invalid_option = True
            invalid_values.extend(bad)

        return invalid_answer_format, invalid_option, invalid_values

    if question_type in {"single_choice", "yes_no", "spatial_3d_axes"}:
        if isinstance(pred, list):
            invalid_answer_format = True
            return invalid_answer_format, invalid_option, invalid_values

        if isinstance(choices, list) and choices:
            if question_type == "spatial_3d_axes":
                allowed = {normalize_answer(c, "spatial_3d_axes") for c in choices}
                pred_norm = normalize_answer(pred, "spatial_3d_axes")
                if str(pred_norm) not in allowed:
                    invalid_option = True
                    invalid_values.append(str(pred_norm))
            else:
                allowed = {str(c) for c in choices}
                if str(pred) not in allowed:
                    invalid_option = True
                    invalid_values.append(str(pred))

        return invalid_answer_format, invalid_option, invalid_values

    if question_type == "single_choice_plus_text":
        # pred is already the extracted letter, e.g. "C"; choices need the same
        # extraction.
        if isinstance(pred, list):
            invalid_answer_format = True
            return invalid_answer_format, invalid_option, invalid_values

        if isinstance(choices, list) and choices:
            allowed = {_extract_choice_letter(c) for c in choices}
            if str(pred) not in allowed:
                invalid_option = True
                invalid_values.append(str(pred))

        return invalid_answer_format, invalid_option, invalid_values

    if question_type == "count":
        # Count answers are numeric strings; choices are empty, so there is no
        # option validation.
        if isinstance(pred, list):
            invalid_answer_format = True
        return invalid_answer_format, invalid_option, invalid_values

    return invalid_answer_format, invalid_option, invalid_values
