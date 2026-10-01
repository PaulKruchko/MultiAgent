"""Claude Code sandbox rules shared by settings validation and the provider.

Owner: providers.

A leaf module (it imports nothing from maf): ``maf.config`` checks ``claude_code_tmp_base`` and ``claude_code_tools``
with these rules when settings load, before a run spends anything, and ``maf.providers.claude_code`` applies the same
rules again before each call. Config cannot import the providers package itself: the providers import config for
their prices.
"""

from __future__ import annotations

import os
import secrets
from collections.abc import Sequence
from pathlib import Path

DEFAULT_TMP_BASE = Path("/tmp")
TMPDIR_PREFIX = "maf-"
TMPDIR_RANDOM_BYTES = 6
"""``TMPDIR`` is ``<tmp_base>/maf-<12 random hex>``: 21 bytes under ``/tmp``."""
CLI_CHILD_TMPDIR_MAX_BYTES = 44
"""Claude Code 2.1.284 (``HXn`` in the binary) exports ``TMPDIR=<TMPDIR>/claude-<uid>`` to sandboxed commands, sized
to at most 44 bytes so their own sockets keep 63 of the 108 ``sun_path`` bytes. A longer one is still used (without
``CLAUDE_CODE_TMPDIR`` the CLI's fallback is the same directory), so ``check_tmpdir`` enforces the budget. It binds
before the runtime's own sockets do: they add at most 35 bytes to ``TMPDIR`` (``claude-socks-<16 hex>.sock``;
``srt-obs-*/s<8 hex>.sock`` and ``srt-mux-<pid>-<n>.sock`` are shorter), and ``cc-socks``/daemon sockets move to
``/tmp`` on their own."""

PATH_SCOPED_TOOLS = frozenset({"Read", "Edit", "Write", "MultiEdit", "NotebookEdit"})
"""File tools whose allow rules must carry a path scope: a bare rule matches every path on the machine."""


def scratch_tmpdir(base: Path = DEFAULT_TMP_BASE) -> Path:
    """A new random ``<base>/maf-<12 hex>`` path (not created). Not derived from the workspace: the ``--settings``
    argv shows the name to every local user, and a predictable one could be created first by someone else."""
    return base / f"{TMPDIR_PREFIX}{secrets.token_hex(TMPDIR_RANDOM_BYTES)}"


def max_tmpdir_bytes(uid: int | None = None) -> int:
    """Longest ``TMPDIR`` (in bytes) whose ``<TMPDIR>/claude-<uid>`` fits ``CLI_CHILD_TMPDIR_MAX_BYTES``: 32 for a
    4-digit uid. ``uid`` defaults to ``os.getuid()``."""
    return CLI_CHILD_TMPDIR_MAX_BYTES - len(f"/claude-{os.getuid() if uid is None else uid}")


def tmpdir_bytes(base: Path) -> int:
    """Length in bytes (as ``sun_path`` counts it) of every ``scratch_tmpdir(base)``: the random part has a fixed
    length."""
    return len(os.fsencode(base / f"{TMPDIR_PREFIX}{'0' * 2 * TMPDIR_RANDOM_BYTES}"))


def check_tmp_base(base: Path, uid: int | None = None) -> None:
    """``ValueError`` when ``base`` (``Settings.claude_code_tmp_base``) is relative, or so long that the ``TMPDIR``
    under it exceeds ``max_tmpdir_bytes(uid)``: the rule ``check_tmpdir`` applies before each call, checked when
    settings load instead."""
    if not base.is_absolute():
        raise ValueError(f"claude_code_tmp_base must be an absolute path, got {str(base)!r}")
    length, limit = tmpdir_bytes(base), max_tmpdir_bytes(uid)
    if length > limit:
        uid_text = os.getuid() if uid is None else uid
        raise ValueError(
            f"claude_code_tmp_base {base} is too long: Claude Code's TMPDIR under it ({base}/{TMPDIR_PREFIX}<12 hex>) "
            f"would be {length} bytes, at most {limit} fit (sandboxed commands get TMPDIR=<TMPDIR>/claude-{uid_text}, "
            f"which must stay within {CLI_CHILD_TMPDIR_MAX_BYTES} bytes for their Unix sockets); use a directory of "
            f"at most {limit - (length - len(os.fsencode(base)))} bytes, such as {DEFAULT_TMP_BASE}"
        )


def check_scoped_tools(tools: Sequence[str]) -> None:
    """``ValueError`` for a bare ``Read``/``Edit``/``Write``-style allow rule (it would grant every path)."""
    bare = [t for t in tools if t.strip() in PATH_SCOPED_TOOLS]
    if bare:
        raise ValueError(f"file tool rules need a path scope such as 'Edit(./**)': {', '.join(bare)}")


__all__ = [
    "CLI_CHILD_TMPDIR_MAX_BYTES",
    "DEFAULT_TMP_BASE",
    "PATH_SCOPED_TOOLS",
    "TMPDIR_PREFIX",
    "TMPDIR_RANDOM_BYTES",
    "check_scoped_tools",
    "check_tmp_base",
    "max_tmpdir_bytes",
    "scratch_tmpdir",
    "tmpdir_bytes",
]
