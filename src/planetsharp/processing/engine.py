from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from planetsharp.core.models import Session


@dataclass
class RenderResult:
    stage1: dict[str, list[list[float]]]
    stage2: list[list[float]]
    final: list[list[float]]


class WorkflowEngine:
    """Deterministic placeholder engine for v1 scaffolding."""

    def render(self, session: Session) -> RenderResult:
        stage1 = {"L": [[0.0]], "R": [[0.0]], "G": [[0.0]], "B": [[0.0]]}
        stage2 = [[self._workflow_signal(session.stage2_workflow.blocks)]]
        mixed = self._mix(stage1, stage2, session.mixer.stage2_mix_amount / 100.0)
        return RenderResult(stage1=stage1, stage2=stage2, final=mixed)

    def _workflow_signal(self, blocks: list[Any]) -> float:
        signal = 0.0
        for index, block in enumerate(blocks):
            if block.enabled:
                signal += (index + 1) * 0.01
        return signal

    def _mix(self, stage1: dict[str, list[list[float]]], stage2: list[list[float]], m: float) -> list[list[float]]:
        base = sum(stage1[c][0][0] for c in ("R", "G", "B")) / 3.0
        return [[(1 - m) * base + m * stage2[0][0]]]
