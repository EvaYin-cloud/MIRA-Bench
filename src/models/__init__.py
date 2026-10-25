from __future__ import annotations

from typing import Any

from .base import BaseVLMAdapter, ModelResponse


def build_model_adapter(cfg: dict[str, Any]) -> BaseVLMAdapter:
    """Build the model adapter selected by the provider field in config."""
    provider = cfg.get("provider", "qwen_openai_compatible")

    if provider == "qwen_openai_compatible":
        from .qwen_adapter import QwenOpenAICompatibleAdapter
        return QwenOpenAICompatibleAdapter(cfg)

    if provider == "openai":
        from .openai_adapter import OpenAIChatVisionAdapter
        return OpenAIChatVisionAdapter(cfg)

    if provider == "google":
        from .gemini_adapter import GeminiAdapter
        return GeminiAdapter(cfg)

    if provider == "deepseek_vl2":
        from .deepseek_vl2_adapter import DeepSeekVL2Adapter
        return DeepSeekVL2Adapter(cfg)

    if provider == "minicpm_v":
        from .minicpm_v_adapter import MiniCPMVAdapter
        return MiniCPMVAdapter(cfg)

    raise ValueError(f"Unknown model provider: {provider}")


__all__ = ["build_model_adapter", "BaseVLMAdapter", "ModelResponse"]
