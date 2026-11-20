"""Every stage, in the order a release runs them."""

from __future__ import annotations

from . import advertising, benchmarking, building, documents, film, hub, measuring, packaging, publishing, versions
from .stages import Stage


def build() -> tuple[Stage, ...]:
    """The stages in order; each stage's `needs` come before it."""
    before = (*hub.STAGES, *documents.STAGES[:1], *versions.STAGES, *benchmarking.STAGES, *measuring.STAGES, *documents.STAGES[1:], *film.STAGES, *building.STAGES, *packaging.STAGES)
    ordered = (*before, *publishing.stages(before), *advertising.STAGES)
    names = [stage.name for stage in ordered]
    assert len(set(names)) == len(names), f"a stage is named twice: {names}"
    for position, stage in enumerate(ordered):
        assert all(need in names[:position] for need in stage.needs), f"{stage.name} needs a stage that does not come before it"
    return ordered
