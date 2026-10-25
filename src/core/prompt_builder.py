from __future__ import annotations

import json
from typing import Any

from .dataset import QASample


def format_choices(choices: Any) -> str:
    if isinstance(choices, (list, dict)):
        return json.dumps(choices, ensure_ascii=False)
    return str(choices)


def _build_global_output_rules(
    ask_for_brief_explanation: bool,
    explanation_max_words: int,
    question_type: str,
) -> list[str]:
    lines: list[str] = []

    lines.append("Return exactly one valid JSON object.")
    lines.append("Do not output markdown.")
    lines.append("Do not output any text before or after the JSON object.")
    lines.append(
        'The JSON object must contain exactly these three keys in this order: '
        'final_answer, brief_explanation, confidence.'
    )
    lines.append("Do not output any additional keys.")
    lines.append("Do not output step-by-step reasoning.")
    lines.append("Do not output analysis.")
    lines.append("confidence must be a JSON number between 0 and 1.")

    if question_type in {"single_choice", "yes_no", "spatial_3d_axes", "count", "single_choice_plus_text"}:
        lines.append("final_answer must be a JSON string.")
    elif question_type == "multi_choice":
        lines.append("final_answer must be a JSON array of strings.")

    if ask_for_brief_explanation:
        lines.append(
            f"brief_explanation stay within about {explanation_max_words} words."
        )
        lines.append(
            "brief_explanation must explain how the visual evidence supports the final_answer."
        )
        lines.append(
            "If you are uncertain, still provide the best final_answer first, "
            "then give only one short evidence sentence."
        )
        lines.append(
            "Do not discuss alternative interpretations, re-checks, or "
            "self-corrections in brief_explanation."
        )
        lines.append(
            "Do not restate the full question or list multiple candidate answers."
        )
    else:
        lines.append('brief_explanation must be "".')

    return lines


def _build_question_type_rules(sample: QASample) -> list[str]:
    qt = sample.question_type
    lines: list[str] = []

    if qt == "yes_no":
        lines.append("Question-type-specific rules for yes_no:")
        lines.append('- final_answer must be one JSON string, not a list.')
        lines.append('- final_answer must be exactly one of: "yes", "no", "not sure".')
        lines.append(
            '- Valid example: {"final_answer": "yes", "brief_explanation": "", "confidence": 0.9}'
        )
        lines.append(
            '- Invalid example: {"final_answer": ["yes"], "brief_explanation": "", "confidence": 0.9}'
        )
        return lines

    if qt == "single_choice":
        lines.append("Question-type-specific rules for single_choice:")
        lines.append('- final_answer must be one JSON string, not a list.')
        lines.append("- final_answer must be exactly one option from the provided choices.")
        lines.append(
            '- Valid example: {"final_answer": "17", "brief_explanation": "", "confidence": 0.9}'
        )
        lines.append(
            '- Invalid example: {"final_answer": ["17"], "brief_explanation": "", "confidence": 0.9}'
        )
        return lines

    if qt == "multi_choice":
        lines.append("Question-type-specific rules for multi_choice:")
        lines.append('- final_answer must be a JSON list of strings.')
        lines.append("- Every item in final_answer must come from the provided choices.")
        lines.append("- Do not repeat items in final_answer.")
        lines.append(
            '- Valid example: {"final_answer": ["1", "3"], "brief_explanation": "", "confidence": 0.9}'
        )
        lines.append(
            '- Invalid example: {"final_answer": "1,3", "brief_explanation": "", "confidence": 0.9}'
        )
        return lines

    if qt == "spatial_3d_axes":
        lines.append("Question-type-specific rules for spatial_3d_axes:")
        lines.append('- final_answer must be one JSON string, not a list.')
        lines.append("- Return only the final relation text.")
        lines.append("- Do not return separate horizontal/vertical/depth fields.")
        lines.append(
            '- Valid example: {"final_answer": "upper-left of", "brief_explanation": "", "confidence": 0.9}'
        )
        lines.append(
            '- Invalid example: {"final_answer": {"horizontal": "left"}, '
            '"brief_explanation": "", "confidence": 0.9}'
        )
        return lines

    if qt == "count":
        lines.append("Question-type-specific rules for count:")
        lines.append('- final_answer must be a JSON string containing a number (e.g., "3").')
        lines.append("- Count only the objects that match the query criteria.")
        lines.append(
            '- Valid example: {"final_answer": "2", "brief_explanation": "", "confidence": 0.9}'
        )
        lines.append(
            '- Invalid example: {"final_answer": 2, "brief_explanation": "", "confidence": 0.9}'
        )
        return lines

    if qt == "single_choice_plus_text":
        lines.append("Question-type-specific rules for single_choice_plus_text:")
        lines.append('- final_answer must be a JSON string containing only the option letter.')
        lines.append("- Do not include the option text, only the letter.")
        lines.append("- Select the best option from the provided choices.")
        lines.append(
            '- Valid example: {"final_answer": "C", "brief_explanation": "", "confidence": 0.9}'
        )
        lines.append(
            '- Invalid example: {"final_answer": "C. The mirror reveals...", '
            '"brief_explanation": "", "confidence": 0.9}'
        )
        return lines

    # Fallback.
    lines.append("Question-type-specific rules:")
    lines.append(
        "- Follow the provided choices and return final_answer in the most direct valid JSON form."
    )
    return lines


def _build_self_check_rules(sample: QASample) -> list[str]:
    qt = sample.question_type
    lines: list[str] = []
    lines.append("Before replying, check the following:")

    if qt == "multi_choice":
        lines.append("- Is final_answer a JSON list?")
        lines.append("- Are all items unique?")
        lines.append("- Are all items contained in the provided choices?")
    elif qt in {"yes_no", "single_choice", "spatial_3d_axes", "count", "single_choice_plus_text"}:
        lines.append("- Is final_answer a JSON string, not a list?")
        if qt == "yes_no":
            lines.append('- Is final_answer exactly "yes", "no", or "not sure"?')
        elif qt == "single_choice":
            lines.append("- Is final_answer valid under the provided choices?")
        elif qt == "single_choice_plus_text":
            lines.append("- Is final_answer only the option letter (e.g., 'C'), not the full text?")
        elif qt == "count":
            lines.append("- Is final_answer a string containing a number?")
    else:
        lines.append("- Is final_answer in the correct JSON type for this question?")

    lines.append("- Is the JSON object complete and valid?")
    lines.append("- Is there any text outside the JSON object? If yes, remove it.")
    return lines


def _build_compact_schema_block() -> list[str]:
    lines: list[str] = []
    lines.append("Required JSON schema:")
    lines.append("{")
    lines.append('  "final_answer": "string or list of strings",')
    lines.append('  "brief_explanation": "string",')
    lines.append('  "confidence": 0.0')
    lines.append("}")
    return lines


def _get_task_background(prompt_cfg: dict[str, Any]) -> str:
    return prompt_cfg.get(
        "task_background",
        (
            "Task background:\n"
            "- Some objects are seen directly in the real world.\n"
            "- Some objects are seen only through a mirror reflection.\n"
            "- A mirror instance and a real-world instance may refer to the same physical object.\n"
            "- All spatial descriptions use the ego-centered coordinate system of the input image.\n"
            '- "Left" and "right" refer to the image/observer\'s left and right.\n'
            '- "Front" and "behind" refer to depth relative to the observer in the current image view.\n'
            "- Unless explicitly stated otherwise, answer from the viewpoint of the current image.\n"
            "- Use only the provided visual evidence."
        ),
    ).strip()


def _get_referring_style_hint(sample: QASample, prompt_cfg: dict[str, Any]) -> str:
    """Build an image-input hint from the referring_style."""
    style = sample.referring_style

    if style == "visual_bbox":
        return (
            "The image contains one or more red bounding boxes highlighting the queried object(s). "
            "Use the bounding box to identify which object is being asked about."
        )

    if style in {"visual_bbox_desc", "visual_bbox_text_desc"}:
        return (
            "The image contains one or more red bounding boxes highlighting the queried object(s). "
            "The question text also includes a description of the object. "
            "Use both the bounding box and the description to identify the queried object."
        )

    if style == "visual_mask":
        return (
            "Use the provided mask to identify which object is being asked about. "
            "The white region in the mask corresponds to the queried object in the main image."
        )

    if style == "visual_mask_desc":
        return (
            "An additional binary mask image is provided alongside the main image. "
            "The white region in the mask indicates the queried object. "
            "The question text also includes a description of the object."
        )

    # bbox_text, bbox_text_desc, and original do not need extra visual hints.
    return ""


def _get_subcategory_rule(sample: QASample, prompt_cfg: dict[str, Any]) -> str:
    """Return any extra rule defined at the subcategory level."""
    subcat = str(sample.subcategory).strip()

    if subcat == "visibility_classification":
        return prompt_cfg.get(
            "visibility_classification_rule",
            (
                "For visibility classification questions:\n"
                "- Classify the queried object based on where it is visible in the current image.\n"
                '- "Only in the mirror" means visible in the mirror region but not directly '
                'visible in the real-world region.\n'
                '- "Only in the real world" means directly visible in the real-world region but '
                'not visible in the mirror region.\n'
                '- "In both" means directly visible in the real-world region and also visible '
                'in the mirror region.\n'
                "- These three categories are mutually exclusive.\n"
                "- Do not classify based on whether the object physically exists in the room.\n"
                "- Do not classify based on whether the object could theoretically have a reflection."
            ),
        ).strip()

    return ""


def _build_image_input_description(
    has_depth: bool, has_mask: bool, prompt_cfg: dict[str, Any]
) -> str:
    """
    Build an exact input description for the image combination actually sent.
    Image order: RGB -> mask if present -> depth if present.

    Four combinations:
    - RGB only         -> no extra description
    - RGB + depth      -> "2 images: 1st RGB, 2nd depth map"
    - RGB + mask       -> "2 images: 1st RGB, 2nd binary mask"
    - RGB + mask + depth -> "3 images: 1st RGB, 2nd mask, 3rd depth map"
    """
    # Allow config-level override for advanced use.
    custom = prompt_cfg.get("image_input_description", "").strip()
    if custom:
        return custom

    depth_desc = (
        "a depth map using jet colormap "
        "(blue = near/close to camera, red = far from camera). "
        "Use the depth map to reason about relative distances and spatial positions of objects"
    )
    mask_desc = (
        "a binary mask where the white region indicates the queried object"
    )

    if has_mask and has_depth:
        return (
            "You are provided with three images in order: "
            f"(1) an RGB image, (2) {mask_desc}, and (3) {depth_desc}."
        )

    if has_depth:
        return (
            "You are provided with two images: "
            f"the first is an RGB image, and the second is {depth_desc}."
        )

    if has_mask:
        return (
            "You are provided with two images: "
            f"the first is an RGB image, and the second is {mask_desc}."
        )

    return ""


def build_user_prompt(
    sample: QASample,
    prompt_cfg: dict[str, Any],
    prompt_version: str,
    has_depth: bool = False,
    has_mask: bool = False,
) -> str:
    lines: list[str] = []

    # 1. Task instruction.
    task_instruction = prompt_cfg.get("task_instruction", "").strip()
    if task_instruction:
        lines.append(task_instruction)
        lines.append("")

    # 2. Task background.
    task_background = _get_task_background(prompt_cfg)
    if task_background:
        lines.append(task_background)
        lines.append("")

    # 3. Dynamic image-input description. Tell the model exactly which images
    #    were sent. Image order: RGB -> mask if present -> depth if present.
    image_desc = _build_image_input_description(has_depth, has_mask, prompt_cfg)
    if image_desc:
        lines.append(image_desc)
        lines.append("")

    # 4. Mirror-region hint.
    mirror_hint = prompt_cfg.get("mirror_hint", "").strip()
    if mirror_hint:
        lines.append(mirror_hint)
        lines.append("")

    # 5. Referring-style visual hint.
    ref_hint = _get_referring_style_hint(sample, prompt_cfg)
    if ref_hint:
        lines.append(ref_hint)
        lines.append("")

    # 5. Subcategory-specific rule.
    subcat_rule = _get_subcategory_rule(sample, prompt_cfg)
    if subcat_rule:
        lines.append(subcat_rule)
        lines.append("")

    # 6. Question body.
    lines.append("Question Type: {}".format(sample.question_type))
    lines.append("Question: {}".format(sample.question))

    if sample.choices:
        lines.append("Choices: {}".format(format_choices(sample.choices)))

    lines.append("")

    # 7. JSON schema.
    schema_instruction = prompt_cfg.get("answer_schema_instruction", "").strip()
    if schema_instruction:
        lines.append(schema_instruction)
    else:
        lines.extend(_build_compact_schema_block())
    lines.append("")

    # 8. Global output rules.
    ask_for_brief_explanation = bool(prompt_cfg.get("ask_for_brief_explanation", True))
    explanation_max_words = int(prompt_cfg.get("explanation_max_words", 8))

    lines.append("Global output rules:")
    lines.extend(
        _build_global_output_rules(
            ask_for_brief_explanation=ask_for_brief_explanation,
            explanation_max_words=explanation_max_words,
            question_type=sample.question_type,
        )
    )
    lines.append("")

    # 9. Question-type rules.
    lines.extend(_build_question_type_rules(sample))
    lines.append("")

    # 10. Self-check rules.
    lines.extend(_build_self_check_rules(sample))
    lines.append("")

    lines.append("Now answer the question using exactly one valid JSON object only.")

    return "\n".join(lines).strip()
