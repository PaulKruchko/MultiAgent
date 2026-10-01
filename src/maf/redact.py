"""Masking API keys in text that leaves maf: status output, MCP tool results, prompts to web-enabled models.

Owner: orchestration.

``redact(text, secrets)`` replaces the exact ``secrets`` (longest first) and anything shaped like an OpenAI, Anthropic
or Google API key; ``redact(..., patterns=LOG_PATTERNS)`` also masks ``Bearer`` tokens (log lines, not reports, where
"Bearer authentication" is ordinary prose). ``known_secrets(environ)`` collects the key values of ``KEY_VARIABLES``
from an environment, so a key that has no recognizable shape is masked too.
"""

from __future__ import annotations

import os
import re
from collections.abc import Iterable, Mapping

KEY_VARIABLES = ("OPENAI_API_KEY", "OPENAI_ADMIN_KEY", "GEMINI_API_KEY", "GOOGLE_API_KEY", "ANTHROPIC_API_KEY",
                 "CONTROL_PLANE_API_KEY")
"""Environment variables that hold API keys maf or tunnel-client use."""

MIN_SECRET_CHARS = 8
"""Shorter values are not treated as secrets (they would mask ordinary words)."""

KEY_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\bsk-[A-Za-z0-9_-]{12,}"),  # OpenAI (sk-, sk-proj-) and Anthropic (sk-ant-)
    re.compile(r"\bAIza[0-9A-Za-z_-]{20,}"),  # Google
)
LOG_PATTERNS: tuple[re.Pattern[str], ...] = (
    *KEY_PATTERNS,
    re.compile(r"(?i)(\bbearer\s+)[A-Za-z0-9._~+/=-]{8,}"),
)
MASK = "[redacted]"


def known_secrets(environ: Mapping[str, str] | None = None) -> set[str]:
    """The values of ``KEY_VARIABLES`` in ``environ`` (default ``os.environ``) that are long enough to be keys."""
    environ = os.environ if environ is None else environ
    return {value for key in KEY_VARIABLES if len(value := environ.get(key, "").strip()) >= MIN_SECRET_CHARS}


def redact(text: str, secrets: Iterable[str] = (), *, patterns: Iterable[re.Pattern[str]] = KEY_PATTERNS) -> str:
    """Mask the exact ``secrets`` and every match of ``patterns`` (a pattern's first group, if any, is kept)."""
    for value in sorted((s for s in secrets if s), key=len, reverse=True):
        text = text.replace(value, MASK)
    for pattern in patterns:
        text = pattern.sub(lambda m: (m.group(1) if m.groups() else "") + MASK, text)
    return text
