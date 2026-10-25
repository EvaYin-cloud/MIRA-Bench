from __future__ import annotations

import base64
import json
import time
from pathlib import Path
from typing import Any

from openai import OpenAI

from .base import BaseVLMAdapter, ModelResponse
from ..core.utils import extract_json_object


def _to_data_url(image_path: str) -> str:
    p = Path(image_path)
    mime = "image/png" if p.suffix.lower() == ".png" else "image/jpeg"
    with open(p, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")
    return f"data:{mime};base64,{b64}"


class QwenOpenAICompatibleAdapter(BaseVLMAdapter):
    def __init__(self, cfg: dict[str, Any]) -> None:
        import os

        api_key = os.environ.get(cfg["api_key_env"], "EMPTY")
        self.client = OpenAI(base_url=cfg["api_base"], api_key=api_key)
        self.model_name = cfg["model_name"]
        self.temperature = cfg.get("temperature", 0.0)
        self.max_tokens = cfg.get("max_tokens", 1024)
        self.repair_max_tokens = cfg.get("repair_max_tokens", 256)
        self.top_p = cfg.get("top_p")
        self.repetition_penalty = cfg.get("repetition_penalty")
        self.timeout_sec = cfg.get("timeout_sec", 120)

        # Whether to enable strict schema constraints.
        self.use_json_schema = bool(cfg.get("use_json_schema", True))
        self.disable_response_format = bool(cfg.get("disable_response_format", False))
        self.ask_for_brief_explanation = bool(cfg.get("ask_for_brief_explanation", True))
        self.explanation_max_words = int(cfg.get("explanation_max_words", 8))

    def _generation_options(self) -> dict[str, Any]:
        options: dict[str, Any] = {}
        if self.top_p is not None:
            options["top_p"] = self.top_p
        if self.repetition_penalty is not None:
            options["extra_body"] = {"repetition_penalty": self.repetition_penalty}
        return options

    # ------------------------------------------------------------------
    #  response_format construction
    # ------------------------------------------------------------------

    def _is_free_text(self, question_type: str | None) -> bool:
        return str(question_type).strip().lower() == "free_text"

    def _build_response_format(self, question_type: str | None = None) -> dict[str, Any] | None:
        """
        Non-free_text -> use strict JSON schema / json_object.
        free_text     -> do not use response_format; return natural language.
        """
        if self.disable_response_format:
            return None

        if self._is_free_text(question_type):
            return None

        # final_answer schema depends on question_type.
        if question_type == "multi_choice":
            final_answer_schema: dict[str, Any] = {
                "type": "array",
                "items": {"type": "string"},
            }
        else:
            # yes_no, single_choice, spatial_3d_axes, and count are strings.
            final_answer_schema = {"type": "string"}

        if not self.use_json_schema:
            return {"type": "json_object"}

        # brief_explanation schema
        if self.ask_for_brief_explanation:
            brief_explanation_max_chars = max(20, self.explanation_max_words * 8)
            brief_explanation_schema: dict[str, Any] = {
                "type": "string",
                "maxLength": brief_explanation_max_chars,
            }
        else:
            brief_explanation_schema = {
                "type": "string",
                "enum": [""],
            }

        return {
            "type": "json_schema",
            "json_schema": {
                "name": "benchmark_answer",
                "schema": {
                    "type": "object",
                    "properties": {
                        "final_answer": final_answer_schema,
                        "brief_explanation": brief_explanation_schema,
                        "confidence": {
                            "type": "number",
                            "minimum": 0.0,
                            "maximum": 1.0,
                        },
                    },
                    "required": ["final_answer", "brief_explanation", "confidence"],
                    "additionalProperties": False,
                },
            },
        }

    # ------------------------------------------------------------------
    #  Message construction with mask_path support
    # ------------------------------------------------------------------

    def _build_messages(
        self,
        system_prompt: str,
        user_prompt: str,
        rgb_path: str,
        depth_path: str | None = None,
        mask_path: str | None = None,
    ) -> list[dict[str, Any]]:
        """
        Build the messages list.
        Image order: rgb -> mask if present -> depth if present.
        Text is placed before images.
        """
        content: list[dict[str, Any]] = [
            {"type": "text", "text": user_prompt},
            {"type": "image_url", "image_url": {"url": _to_data_url(rgb_path)}},
        ]

        if mask_path:
            content.append(
                {"type": "image_url", "image_url": {"url": _to_data_url(mask_path)}}
            )

        if depth_path:
            content.append(
                {"type": "image_url", "image_url": {"url": _to_data_url(depth_path)}}
            )

        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": content},
        ]

    # ------------------------------------------------------------------
    #  Inference with mask_path and reasoning_content capture
    # ------------------------------------------------------------------

    def infer(
        self,
        system_prompt: str,
        user_prompt: str,
        rgb_path: str,
        depth_path: str | None = None,
        mask_path: str | None = None,
        question_type: str | None = None,
    ) -> ModelResponse:
        messages = self._build_messages(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            rgb_path=rgb_path,
            depth_path=depth_path,
            mask_path=mask_path,
        )

        req_kwargs: dict[str, Any] = {
            "model": self.model_name,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "messages": messages,
            "timeout": self.timeout_sec,
            **self._generation_options(),
        }

        response_format = self._build_response_format(question_type)
        if response_format is not None:
            req_kwargs["response_format"] = response_format

        start = time.time()
        resp = self.client.chat.completions.create(**req_kwargs)
        latency = time.time() - start

        choice = resp.choices[0]
        raw_text = choice.message.content or ""

        # Capture Qwen3 thinking / reasoning_content when available.
        visible_reasoning = None
        if hasattr(choice.message, "reasoning_content") and choice.message.reasoning_content:
            visible_reasoning = choice.message.reasoning_content

        # free_text: treat the full text as final_answer.
        if self._is_free_text(question_type):
            parsed = {
                "final_answer": raw_text.strip(),
            }
            if not visible_reasoning:
                visible_reasoning = raw_text.strip()
        else:
            parsed = extract_json_object(raw_text)
            if not visible_reasoning:
                visible_reasoning = parsed.get("brief_explanation") if parsed else None

        return ModelResponse(
            model_name=self.model_name,
            raw_text=raw_text,
            parsed_json=parsed,
            visible_reasoning=visible_reasoning,
            usage=resp.usage.model_dump() if getattr(resp, "usage", None) else None,
            latency_sec=latency,
            finish_reason=choice.finish_reason,
            raw_response=resp.model_dump(),
        )

    # ------------------------------------------------------------------
    #  Format repair with count-type handling
    # ------------------------------------------------------------------

    def repair_format(
        self,
        bad_raw_text: str,
        question_type: str,
        choices: Any,
    ) -> ModelResponse:
        """
        Second call: do not inspect images or re-run reasoning.
        Only rewrite the existing bad output into valid JSON.
        free_text bypasses repair_format and keeps the text.
        """
        if self._is_free_text(question_type):
            cleaned_text = str(bad_raw_text).strip()
            return ModelResponse(
                model_name=self.model_name,
                raw_text=cleaned_text,
                parsed_json={"final_answer": cleaned_text},
                visible_reasoning=cleaned_text,
                usage=None,
                latency_sec=0.0,
                finish_reason="free_text_passthrough",
                raw_response=None,
            )

        if question_type == "multi_choice":
            final_answer_rule = (
                "final_answer must be a JSON array of strings, and every item "
                "must come from the provided choices."
            )
        elif question_type == "count":
            final_answer_rule = (
                'final_answer must be a JSON string containing a number (e.g., "3").'
            )
        else:
            final_answer_rule = (
                "final_answer must be a JSON string, and it must come from the "
                "provided choices when choices are provided."
            )

        repair_prompt = (
            "Rewrite the following assistant output into exactly one valid JSON object.\n"
            "Do not add new reasoning.\n"
            "Do not explain.\n"
            "Do not output markdown.\n"
            "Return exactly these keys only: final_answer, brief_explanation, confidence.\n"
            f"{final_answer_rule}\n"
            'brief_explanation must be a JSON string. Use "" if needed.\n'
            "confidence must be a JSON number between 0 and 1.\n"
        )

        # Count questions have no choices, so omit the choices line.
        if choices:
            repair_prompt += f"Choices: {json.dumps(choices, ensure_ascii=False)}\n"

        repair_prompt += (
            "Bad assistant output:\n"
            f"{bad_raw_text}"
        )

        req_kwargs: dict[str, Any] = {
            "model": self.model_name,
            "temperature": 0.0,
            "max_tokens": self.repair_max_tokens,
            "messages": [
                {
                    "role": "system",
                    "content": "You only repair output format. You do not add new reasoning.",
                },
                {
                    "role": "user",
                    "content": repair_prompt,
                },
            ],
            "timeout": self.timeout_sec,
            **self._generation_options(),
        }

        response_format = self._build_response_format(question_type)
        if response_format is not None:
            req_kwargs["response_format"] = response_format

        start = time.time()
        resp = self.client.chat.completions.create(**req_kwargs)
        latency = time.time() - start

        choice = resp.choices[0]
        raw_text = choice.message.content or ""
        parsed = extract_json_object(raw_text)
        visible_reasoning = parsed.get("brief_explanation") if parsed else None

        return ModelResponse(
            model_name=self.model_name,
            raw_text=raw_text,
            parsed_json=parsed,
            visible_reasoning=visible_reasoning,
            usage=resp.usage.model_dump() if getattr(resp, "usage", None) else None,
            latency_sec=latency,
            finish_reason=choice.finish_reason,
            raw_response=resp.model_dump(),
        )
