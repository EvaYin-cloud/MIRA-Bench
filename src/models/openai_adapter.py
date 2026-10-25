from __future__ import annotations

import base64
import json
import os
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


class OpenAIChatVisionAdapter(BaseVLMAdapter):
    """Adapter for the official OpenAI API, including GPT-4o / GPT-5."""

    def __init__(self, cfg: dict[str, Any]) -> None:
        api_key = os.environ.get(cfg["api_key_env"])
        if not api_key:
            raise ValueError(f"Missing env var: {cfg['api_key_env']}")
        self.client = OpenAI(api_key=api_key)
        self.model_name = cfg["model_name"]
        self.temperature = cfg.get("temperature", 0.0)
        self.max_completion_tokens = cfg.get(
            "max_completion_tokens", cfg.get("max_tokens", 2048)
        )
        self.use_max_completion_tokens = bool(
            cfg.get(
                "use_max_completion_tokens",
                self.model_name.lower().startswith("gpt-5"),
            )
        )
        self.timeout_sec = cfg.get("timeout_sec", 120)
        self.ask_for_brief_explanation = bool(cfg.get("ask_for_brief_explanation", True))
        self.explanation_max_words = int(cfg.get("explanation_max_words", 8))

    def _add_token_limit(
        self, req_kwargs: dict[str, Any], token_limit: int
    ) -> None:
        if self.use_max_completion_tokens:
            req_kwargs["max_completion_tokens"] = token_limit
        else:
            req_kwargs["max_tokens"] = token_limit

    def _build_messages(
        self,
        system_prompt: str,
        user_prompt: str,
        rgb_path: str,
        depth_path: str | None = None,
        mask_path: str | None = None,
    ) -> list[dict[str, Any]]:
        content: list[dict[str, Any]] = [
            {"type": "text", "text": user_prompt},
            {"type": "image_url", "image_url": {"url": _to_data_url(rgb_path)}},
        ]
        if mask_path:
            content.append({"type": "image_url", "image_url": {"url": _to_data_url(mask_path)}})
        if depth_path:
            content.append({"type": "image_url", "image_url": {"url": _to_data_url(depth_path)}})
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": content},
        ]

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
            "messages": messages,
            "timeout": self.timeout_sec,
        }
        self._add_token_limit(req_kwargs, int(self.max_completion_tokens))

        # The official OpenAI API only needs json_object mode here.
        if question_type and question_type != "free_text":
            req_kwargs["response_format"] = {"type": "json_object"}

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

    def repair_format(
        self, bad_raw_text: str, question_type: str, choices: Any,
    ) -> ModelResponse:
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
            "Do not explain.\n"
            "Do not output markdown.\n"
            "Return exactly these keys only: final_answer, brief_explanation, confidence.\n"
            f"{final_answer_rule}\n"
            'brief_explanation must be a JSON string. Use "" if needed.\n'
            "confidence must be a JSON number between 0 and 1.\n"
        )
        if choices:
            repair_prompt += f"Choices: {json.dumps(choices, ensure_ascii=False)}\n"
        repair_prompt += f"Bad assistant output:\n{bad_raw_text}"

        req_kwargs: dict[str, Any] = {
            "model": self.model_name,
            "temperature": 0.0,
            "messages": [
                {"role": "system", "content": "You only repair output format. You do not add new reasoning."},
                {"role": "user", "content": repair_prompt},
            ],
            "timeout": self.timeout_sec,
            "response_format": {"type": "json_object"},
        }
        self._add_token_limit(req_kwargs, 256)

        start = time.time()
        resp = self.client.chat.completions.create(**req_kwargs)
        latency = time.time() - start

        choice = resp.choices[0]
        raw_text = choice.message.content or ""
        parsed = extract_json_object(raw_text)
        visible_reasoning = parsed.get("brief_explanation") if parsed else None

        return ModelResponse(
            model_name=self.model_name, raw_text=raw_text, parsed_json=parsed,
            visible_reasoning=visible_reasoning,
            usage=resp.usage.model_dump() if getattr(resp, "usage", None) else None,
            latency_sec=latency, finish_reason=choice.finish_reason,
            raw_response=resp.model_dump(),
        )
