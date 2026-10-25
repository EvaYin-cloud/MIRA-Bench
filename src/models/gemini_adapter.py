from __future__ import annotations

import base64
import json
import os
import time
from pathlib import Path
from typing import Any

from .base import BaseVLMAdapter, ModelResponse
from ..core.utils import extract_json_object


def _load_image_part(image_path: str) -> dict[str, Any]:
    """Build an image part in Gemini inline_data format."""
    p = Path(image_path)
    mime = "image/png" if p.suffix.lower() == ".png" else "image/jpeg"
    with open(p, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")
    return {"inline_data": {"mime_type": mime, "data": b64}}


class GeminiAdapter(BaseVLMAdapter):
    """Google Gemini API adapter using the google-genai SDK."""

    def __init__(self, cfg: dict[str, Any]) -> None:
        try:
            from google import genai
            from google.genai import types
        except ImportError:
            raise ImportError(
                "google-genai package is required for Gemini adapter. "
                "Install with: pip install google-genai"
            )

        api_key = os.environ.get(cfg["api_key_env"])
        if not api_key:
            raise ValueError(f"Missing env var: {cfg['api_key_env']}")

        self.client = genai.Client(api_key=api_key)
        self.model_name = cfg["model_name"]
        self.temperature = cfg.get("temperature", 0.0)
        self.max_tokens = cfg.get("max_tokens", 2048)
        self.timeout_sec = cfg.get("timeout_sec", 120)
        self.raise_on_api_error = bool(cfg.get("raise_on_api_error", True))
        self.ask_for_brief_explanation = bool(cfg.get("ask_for_brief_explanation", True))
        self.explanation_max_words = int(cfg.get("explanation_max_words", 8))

    def _build_contents(
        self,
        user_prompt: str,
        rgb_path: str,
        depth_path: str | None = None,
        mask_path: str | None = None,
    ) -> list[Any]:
        """Build Gemini contents from images and text."""
        from google.genai import types

        parts = []

        # Image order: rgb -> mask -> depth.
        parts.append(types.Part.from_bytes(
            data=Path(rgb_path).read_bytes(),
            mime_type="image/png" if rgb_path.lower().endswith(".png") else "image/jpeg",
        ))

        if mask_path:
            parts.append(types.Part.from_bytes(
                data=Path(mask_path).read_bytes(),
                mime_type="image/png" if mask_path.lower().endswith(".png") else "image/jpeg",
            ))

        if depth_path:
            parts.append(types.Part.from_bytes(
                data=Path(depth_path).read_bytes(),
                mime_type="image/png" if depth_path.lower().endswith(".png") else "image/jpeg",
            ))

        parts.append(types.Part.from_text(text=user_prompt))
        return parts

    def infer(
        self,
        system_prompt: str,
        user_prompt: str,
        rgb_path: str,
        depth_path: str | None = None,
        mask_path: str | None = None,
        question_type: str | None = None,
    ) -> ModelResponse:
        from google.genai import types

        contents = self._build_contents(
            user_prompt=user_prompt,
            rgb_path=rgb_path,
            depth_path=depth_path,
            mask_path=mask_path,
        )

        # Build response_mime_type.
        response_mime = "application/json" if (question_type and question_type != "free_text") else None

        config = types.GenerateContentConfig(
            system_instruction=system_prompt if system_prompt else None,
            temperature=self.temperature,
            max_output_tokens=self.max_tokens,
            response_mime_type=response_mime,
        )

        start = time.time()
        try:
            resp = self.client.models.generate_content(
                model=self.model_name,
                contents=contents,
                config=config,
            )
        except Exception as e:
            if self.raise_on_api_error:
                raise
            # Handle safety-filter and similar exceptions.
            return ModelResponse(
                model_name=self.model_name,
                raw_text="",
                parsed_json=None,
                visible_reasoning=None,
                usage=None,
                latency_sec=time.time() - start,
                finish_reason=f"error: {type(e).__name__}: {e}",
                raw_response=None,
            )
        latency = time.time() - start

        # Extract text.
        raw_text = ""
        finish_reason = "unknown"
        try:
            raw_text = resp.text or ""
        except Exception:
            # resp.text can raise when the response is blocked by safety filters.
            if resp.candidates:
                for part in resp.candidates[0].content.parts:
                    if hasattr(part, "text") and part.text:
                        raw_text += part.text
                finish_reason = str(resp.candidates[0].finish_reason) if resp.candidates[0].finish_reason else "unknown"
            else:
                finish_reason = "blocked"

        if resp.candidates and finish_reason == "unknown":
            finish_reason = str(resp.candidates[0].finish_reason) if resp.candidates[0].finish_reason else "stop"

        # Extract usage.
        usage_dict = None
        if hasattr(resp, "usage_metadata") and resp.usage_metadata:
            um = resp.usage_metadata
            usage_dict = {
                "prompt_tokens": getattr(um, "prompt_token_count", 0) or 0,
                "completion_tokens": getattr(um, "candidates_token_count", 0) or 0,
                "total_tokens": getattr(um, "total_token_count", 0) or 0,
            }

        parsed = extract_json_object(raw_text) if raw_text else None
        visible_reasoning = parsed.get("brief_explanation") if parsed else None

        return ModelResponse(
            model_name=self.model_name,
            raw_text=raw_text,
            parsed_json=parsed,
            visible_reasoning=visible_reasoning,
            usage=usage_dict,
            latency_sec=latency,
            finish_reason=finish_reason,
            raw_response=None,  # Gemini response objects are not directly serializable.
        )

    def repair_format(
        self, bad_raw_text: str, question_type: str, choices: Any,
    ) -> ModelResponse:
        from google.genai import types

        if question_type == "multi_choice":
            final_answer_rule = (
                "final_answer must be a JSON array of strings, and every item "
                "must come from the provided choices."
            )
        elif question_type == "count":
            final_answer_rule = (
                'final_answer must be a JSON string containing a number (e.g., "3").'
            )
        elif question_type == "single_choice_plus_text":
            final_answer_rule = (
                'final_answer must be a JSON string containing only the option letter '
                '(e.g., "C"). Do not include the option text.'
            )
        else:
            final_answer_rule = (
                "final_answer must be a JSON string, and it must come from the "
                "provided choices when choices are provided."
            )

        repair_prompt = (
            "Rewrite the following assistant output into exactly one valid JSON object.\n"
            "Do not add new reasoning.\n"
            "Return exactly these keys only: final_answer, brief_explanation, confidence.\n"
            f"{final_answer_rule}\n"
            'brief_explanation must be a JSON string. Use "" if needed.\n'
            "confidence must be a JSON number between 0 and 1.\n"
        )
        if choices:
            repair_prompt += f"Choices: {json.dumps(choices, ensure_ascii=False)}\n"
        repair_prompt += f"Bad assistant output:\n{bad_raw_text}"

        config = types.GenerateContentConfig(
            system_instruction="You only repair output format. You do not add new reasoning.",
            temperature=0.0,
            max_output_tokens=256,
            response_mime_type="application/json",
        )

        start = time.time()
        try:
            resp = self.client.models.generate_content(
                model=self.model_name,
                contents=[repair_prompt],
                config=config,
            )
            raw_text = resp.text or ""
        except Exception as e:
            if self.raise_on_api_error:
                raise
            raw_text = ""
        latency = time.time() - start

        parsed = extract_json_object(raw_text) if raw_text else None
        visible_reasoning = parsed.get("brief_explanation") if parsed else None

        return ModelResponse(
            model_name=self.model_name, raw_text=raw_text, parsed_json=parsed,
            visible_reasoning=visible_reasoning, usage=None,
            latency_sec=latency, finish_reason="repair",
            raw_response=None,
        )
