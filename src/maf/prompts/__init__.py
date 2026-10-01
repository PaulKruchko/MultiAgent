"""Markdown prompt templates. Placeholders are ``{{name}}``; ``$`` and single braces are left alone
(LaTeX and JSON appear in prompts).

Owner: stages (templates); the loader is frozen.
"""

from __future__ import annotations

import re
from functools import cache
from importlib import resources

_PLACEHOLDER = re.compile(r"\{\{\s*([a-z_][a-z0-9_]*)\s*\}\}")


@cache
def load_prompt(name: str) -> str:
    """Raw template text for ``name`` (e.g. ``"roles/chatgpt"``, ``"critique"``), without ``.md``."""
    return resources.files(__package__).joinpath(f"{name}.md").read_text(encoding="utf-8")


def placeholders(name: str) -> set[str]:
    return set(_PLACEHOLDER.findall(load_prompt(name)))


def render_prompt(name: str, **values: str) -> str:
    """Substitute every placeholder. Missing values raise ``KeyError``; extra values raise ``TypeError``."""
    template = load_prompt(name)
    wanted = set(_PLACEHOLDER.findall(template))
    missing = wanted - values.keys()
    if missing:
        raise KeyError(f"prompt {name!r} missing values: {sorted(missing)}")
    extra = values.keys() - wanted
    if extra:
        raise TypeError(f"prompt {name!r} got unexpected values: {sorted(extra)}")
    return _PLACEHOLDER.sub(lambda m: values[m.group(1)], template)
