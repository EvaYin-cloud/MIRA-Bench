from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any


@dataclass
class ModelResponse:
    model_name: str
    raw_text: str
    parsed_json: dict[str, Any] | None
    visible_reasoning: str | None
    usage: dict[str, Any] | None
    latency_sec: float
    finish_reason: str | None
    raw_response: dict[str, Any] | None


class BaseVLMAdapter(ABC):
    """Abstract base class for all VLM adapters."""

    @abstractmethod
    def infer(
        self,
        system_prompt: str,
        user_prompt: str,
        rgb_path: str,
        depth_path: str | None = None,
        mask_path: str | None = None,
        question_type: str | None = None,
    ) -> ModelResponse:
        """
        Run a single inference call.
        Image order: rgb -> mask if present -> depth if present.
        """
        raise NotImplementedError

    @abstractmethod
    def repair_format(
        self,
        bad_raw_text: str,
        question_type: str,
        choices: Any,
    ) -> ModelResponse:
        """Try to repair a model output with an invalid format."""
        raise NotImplementedError
