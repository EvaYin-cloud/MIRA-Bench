from __future__ import annotations

import json
import time
from typing import Any

from PIL import Image

from .base import BaseVLMAdapter, ModelResponse
from ..core.utils import extract_json_object


def _load_rgb_image(path: str) -> Image.Image:
    return Image.open(path).convert("RGB")


class MiniCPMVAdapter(BaseVLMAdapter):
    """Direct Transformers adapter for MiniCPM-V models."""

    def __init__(self, cfg: dict[str, Any]) -> None:
        try:
            import torch
            from transformers import AutoModel, AutoTokenizer
        except ImportError as e:
            raise ImportError(
                "MiniCPM-V dependencies are missing. Run "
                "src/Setup_minicpm_v_2_6.sh in the target environment first."
            ) from e

        self.torch = torch
        self.model_name = cfg["model_name"]
        self.model_path = cfg.get("model_path", self.model_name)
        self.dtype_name = str(cfg.get("dtype", "bfloat16"))
        self.dtype = getattr(torch, self.dtype_name)
        self.device = cfg.get("device", "cuda")
        self.max_tokens = int(cfg.get("max_tokens", 512))
        self.temperature = float(cfg.get("temperature", 0.0))
        self.sampling = bool(cfg.get("sampling", self.temperature > 0.0))
        self.attn_implementation = cfg.get("attn_implementation", "sdpa")

        self.model = AutoModel.from_pretrained(
            self.model_path,
            trust_remote_code=True,
            attn_implementation=self.attn_implementation,
            torch_dtype=self.dtype,
        )
        self.model = self.model.eval().to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            trust_remote_code=True,
        )

    def _chat(
        self,
        system_prompt: str,
        user_prompt: str,
        image_paths: list[str],
        finish_reason: str,
    ) -> ModelResponse:
        images = [_load_rgb_image(path) for path in image_paths]

        prompt = user_prompt
        if system_prompt:
            prompt = f"{system_prompt}\n\n{user_prompt}"

        content: Any
        if images:
            content = images + [prompt]
        else:
            content = prompt

        msgs = [{"role": "user", "content": content}]

        params: dict[str, Any] = {
            "sampling": self.sampling,
            "max_new_tokens": self.max_tokens,
        }
        if self.sampling:
            params["temperature"] = self.temperature

        start = time.time()
        with self.torch.no_grad():
            raw_text = self.model.chat(
                image=None,
                msgs=msgs,
                tokenizer=self.tokenizer,
                **params,
            )
        latency = time.time() - start

        if not isinstance(raw_text, str):
            raw_text = str(raw_text)
        raw_text = raw_text.strip()
        parsed = extract_json_object(raw_text)
        visible_reasoning = parsed.get("brief_explanation") if parsed else None

        return ModelResponse(
            model_name=self.model_name,
            raw_text=raw_text,
            parsed_json=parsed,
            visible_reasoning=visible_reasoning,
            usage=None,
            latency_sec=latency,
            finish_reason=finish_reason,
            raw_response=None,
        )

    def infer(
        self,
        system_prompt: str,
        user_prompt: str,
        rgb_path: str,
        depth_path: str | None = None,
        mask_path: str | None = None,
        question_type: str | None = None,
    ) -> ModelResponse:
        image_paths = [rgb_path]
        if mask_path:
            image_paths.append(mask_path)
        if depth_path:
            image_paths.append(depth_path)
        return self._chat(system_prompt, user_prompt, image_paths, "stop")

    def repair_format(
        self,
        bad_raw_text: str,
        question_type: str,
        choices: Any,
    ) -> ModelResponse:
        if question_type == "multi_choice":
            final_answer_rule = (
                "final_answer must be a JSON array of strings, and every item "
                "must come from the provided choices."
            )
        elif question_type == "count":
            final_answer_rule = (
                'final_answer must be a JSON string containing a number, e.g. "3".'
            )
        elif question_type == "single_choice_plus_text":
            final_answer_rule = (
                'final_answer must be a JSON string containing only the option letter, '
                'e.g. "C". Do not include the option text.'
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

        return self._chat(
            system_prompt="You only repair output format. You do not add new reasoning.",
            user_prompt=repair_prompt,
            image_paths=[],
            finish_reason="repair",
        )
