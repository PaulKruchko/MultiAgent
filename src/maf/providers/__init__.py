"""Provider adapters and the per-run provider set.

Owner: providers.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from maf.config import ModelRole, Settings
from maf.providers.base import (
    Attachment,
    Citation,
    CompletionRequest,
    CompletionResult,
    Message,
    Provider,
    ProviderError,
    ProviderRefusal,
    StructuredOutputError,
)
from maf.providers.claude_code import SENSITIVE_READ_PATHS, ClaudeCodeProvider
from maf.providers.claude_provider import ClaudeProvider
from maf.providers.gemini_provider import GeminiProvider
from maf.providers.openai_provider import OpenAIProvider


@dataclass(frozen=True)
class Providers:
    """The four model slots available to stages for one run."""

    chatgpt: Provider
    gemini: Provider
    claude: Provider
    claude_code: Provider

    def for_role(self, role: ModelRole) -> Provider:
        return getattr(self, role)


def build_providers(settings: Settings, workspace: Path) -> Providers:
    """Construct real adapters (lazy SDK clients, no network at construction).
    ``claude_code`` is bound to ``workspace`` with ``settings.claude_code_tools`` and may not read the vault
    or ``SENSITIVE_READ_PATHS`` (stages copy what it needs into the prompt); the configured
    Python's ``bin`` directory goes first on its ``PATH`` so ``python`` resolves to the project venv."""
    timeout = settings.provider_timeout_s
    python_bin = settings.python_executable.parent
    return Providers(
        chatgpt=OpenAIProvider(timeout_s=timeout),
        gemini=GeminiProvider(timeout_s=timeout),
        claude=ClaudeProvider(timeout_s=timeout),
        claude_code=ClaudeCodeProvider(
            workspace,
            executable=settings.claude_executable,
            allowed_tools=settings.claude_code_tools,
            timeout_s=settings.claude_code_timeout_s,
            path_prepend=(python_bin,) if python_bin.is_dir() else (),
            deny_read=(*SENSITIVE_READ_PATHS, str(settings.vault_path.expanduser().resolve())),
            turn_context_tokens=settings.claude_code_turn_context_tokens,
            turn_output_tokens=settings.claude_code_turn_output_tokens,
        ),
    )


__all__ = [
    "Attachment",
    "Citation",
    "ClaudeCodeProvider",
    "ClaudeProvider",
    "CompletionRequest",
    "CompletionResult",
    "GeminiProvider",
    "Message",
    "OpenAIProvider",
    "Provider",
    "ProviderError",
    "ProviderRefusal",
    "Providers",
    "StructuredOutputError",
    "build_providers",
]
