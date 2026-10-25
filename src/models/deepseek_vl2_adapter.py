from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from PIL import Image

from .base import BaseVLMAdapter, ModelResponse
from ..core.utils import extract_json_object


def _load_rgb_image(path: str) -> Image.Image:
    return Image.open(path).convert("RGB")


class DeepSeekVL2Adapter(BaseVLMAdapter):
    """Direct Transformers adapter for DeepSeek-VL2 models."""

    def __init__(self, cfg: dict[str, Any]) -> None:
        try:
            import torch
            from transformers import AutoModelForCausalLM
        except ImportError as e:
            raise ImportError(
                "DeepSeek-VL2 dependencies are missing. Run "
                "one of the DeepSeek-VL2 setup scripts in the target environment first."
            ) from e

        try:
            from deepseek_vl.models import DeepseekVLV2Processor
        except ImportError:
            from deepseek_vl2.models import DeepseekVLV2Processor

        self.torch = torch
        self.model_name = cfg["model_name"]
        self.model_path = cfg.get("model_path", self.model_name)
        self.dtype_name = str(cfg.get("dtype", "bfloat16"))
        self.dtype = getattr(torch, self.dtype_name)
        self.device = cfg.get("device", "cuda")
        self.max_tokens = int(cfg.get("max_tokens", 512))
        self.temperature = float(cfg.get("temperature", 0.0))
        self.top_p = float(cfg.get("top_p", 0.9))
        self.repetition_penalty = float(cfg.get("repetition_penalty", 1.1))
        self.chunk_size = int(cfg.get("chunk_size", -1))

        self.processor = DeepseekVLV2Processor.from_pretrained(self.model_path)
        self.tokenizer = self.processor.tokenizer
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            trust_remote_code=True,
            torch_dtype=self.dtype,
        )
        self.model = self.model.to(self.device).eval()

    def _build_conversation(
        self,
        user_prompt: str,
        image_paths: list[str],
    ) -> list[dict[str, Any]]:
        if len(image_paths) == 0:
            content = user_prompt
        elif len(image_paths) == 1:
            content = f"<image>\n{user_prompt}"
        else:
            placeholders = "<image>\n" * len(image_paths)
            content = f"{placeholders}{user_prompt}"

        return [
            {
                "role": "<|User|>",
                "content": content,
                "images": image_paths,
            },
            {"role": "<|Assistant|>", "content": ""},
        ]

    def _generate(
        self,
        system_prompt: str,
        user_prompt: str,
        image_paths: list[str],
        finish_reason: str,
    ) -> ModelResponse:
        conversation = self._build_conversation(user_prompt, image_paths)
        pil_images = [_load_rgb_image(p) for p in image_paths]

        start = time.time()
        with self.torch.no_grad():
            prepare_inputs = self.processor(
                conversations=conversation,
                images=pil_images,
                force_batchify=True,
                system_prompt=system_prompt or "",
            ).to(self.model.device, dtype=self.dtype)

            if self.chunk_size == -1:
                inputs_embeds = self.model.prepare_inputs_embeds(**prepare_inputs)
                past_key_values = None
            else:
                inputs_embeds, past_key_values = self.model.incremental_prefilling(
                    input_ids=prepare_inputs.input_ids,
                    images=prepare_inputs.images,
                    images_seq_mask=prepare_inputs.images_seq_mask,
                    images_spatial_crop=prepare_inputs.images_spatial_crop,
                    attention_mask=prepare_inputs.attention_mask,
                    chunk_size=self.chunk_size,
                )

            do_sample = self.temperature > 0.0
            outputs = self.model.generate(
                inputs_embeds=inputs_embeds,
                input_ids=prepare_inputs.input_ids,
                images=prepare_inputs.images,
                images_seq_mask=prepare_inputs.images_seq_mask,
                images_spatial_crop=prepare_inputs.images_spatial_crop,
                attention_mask=prepare_inputs.attention_mask,
                past_key_values=past_key_values,
                pad_token_id=self.tokenizer.eos_token_id,
                bos_token_id=self.tokenizer.bos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                max_new_tokens=self.max_tokens,
                do_sample=do_sample,
                temperature=self.temperature if do_sample else None,
                top_p=self.top_p if do_sample else None,
                repetition_penalty=self.repetition_penalty,
                use_cache=True,
            )

        latency = time.time() - start
        prompt_len = len(prepare_inputs.input_ids[0])
        raw_text = self.tokenizer.decode(
            outputs[0][prompt_len:].detach().cpu().tolist(),
            skip_special_tokens=True,
        ).strip()
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
        return self._generate(system_prompt, user_prompt, image_paths, "stop")

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

        return self._generate(
            system_prompt="You only repair output format. You do not add new reasoning.",
            user_prompt=repair_prompt,
            image_paths=[],
            finish_reason="repair",
        )
