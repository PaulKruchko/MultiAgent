"""Stage backends. ``default_backends()`` is the registry the pipeline uses unless overridden.

Owner: stages.
"""

from __future__ import annotations

from maf.stages.base import NoteOut, StageBackend, StageContext, StageOutput, generate_handoff
from maf.types import StageName


def default_backends() -> dict[StageName, StageBackend]:
    """One backend instance per stage name, in ``STAGE_ORDER``."""
    from maf.stages.crosscheck import CrosscheckBackend
    from maf.stages.execution import ExecutionBackend
    from maf.stages.final import FinalBackend
    from maf.stages.ingestion import IngestionBackend
    from maf.stages.strategy import StrategyBackend

    return {
        "ingestion": IngestionBackend(),
        "strategy": StrategyBackend(),
        "execution": ExecutionBackend(),
        "crosscheck": CrosscheckBackend(),
        "final": FinalBackend(),
    }


__all__ = ["NoteOut", "StageBackend", "StageContext", "StageOutput", "default_backends", "generate_handoff"]
