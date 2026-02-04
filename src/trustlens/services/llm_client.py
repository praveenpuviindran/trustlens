"""LLM client abstractions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from trustlens.config.settings import settings


class LLMClient(Protocol):
    """Minimal interface for LLM generation."""

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        """Generate a response given system + user prompts."""
        raise NotImplementedError


@dataclass(frozen=True)
class StubLLMClient:
    """Deterministic stub for tests."""

    response_text: str = "STUB_RESPONSE"

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        return self.response_text


@dataclass(frozen=True)
class OpenAIChatClient:
    """
    Placeholder for OpenAI chat client (disabled by default).
    """

    api_key: str
    model_name: str

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        raise RuntimeError("OpenAIChatClient is not enabled in this environment.")


def get_llm_client() -> LLMClient:
    """Factory for LLM clients based on settings."""
    provider = settings.llm_provider.lower()
    if provider == "stub":
        return StubLLMClient()
    if provider == "openai":
        if not settings.openai_api_key:
            raise RuntimeError("OPENAI_API_KEY is required for OpenAI provider.")
        model_name = settings.llm_model_name or "gpt-4o-mini"
        return OpenAIChatClient(api_key=settings.openai_api_key, model_name=model_name)
    raise ValueError(f"Unknown LLM provider: {settings.llm_provider}")
