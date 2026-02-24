from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any
import uuid

from planetsharp.processing.blocks import complete_block_params


class Role(str, Enum):
    L = "L"
    R = "R"
    G = "G"
    B = "B"
    FILTER = "FILTER"


@dataclass
class InputImage:
    path: str
    role: Role = Role.FILTER
    filter_name: str = ""
    assume_linear: bool = False


@dataclass
class FilterWeights:
    r: float = 0.0
    g: float = 0.0
    b: float = 0.0
    l: float = 0.0

    def validate(self) -> None:
        for value in (self.r, self.g, self.b, self.l):
            if not (0.0 <= value <= 100.0):
                raise ValueError("Filter weight values must be between 0 and 100")


@dataclass
class FilterMapping:
    normalize_weights: bool = True
    mappings: dict[str, FilterWeights] = field(default_factory=dict)


@dataclass
class ROI:
    x: int = 0
    y: int = 0
    width: int = 0
    height: int = 0
    enabled: bool = False


@dataclass
class PerformanceSettings:
    threads: int = 1
    memory_percent: int = 50

    def clamp(self, logical_cores: int) -> None:
        self.threads = max(1, min(self.threads, logical_cores))
        self.memory_percent = max(1, min(self.memory_percent, 95))


@dataclass
class ViewerState:
    zoom: float = 1.0
    pan_x: float = 0.0
    pan_y: float = 0.0
    show_l: bool = True
    show_r: bool = True
    show_g: bool = True
    show_b: bool = True
    show_lrgb: bool = True
    stage_display: str = "Final Mixed"


@dataclass
class MixerSettings:
    stage2_mix_amount: float = 100.0
    mix_rgb: float = 100.0
    mix_l: float = 100.0


@dataclass
class BlockInstance:
    type: str
    enabled: bool = True
    params: dict[str, Any] = field(default_factory=dict)
    channel: str | None = None
    id: str = field(default_factory=lambda: str(uuid.uuid4()))


@dataclass
class Workflow:
    blocks: list[BlockInstance] = field(default_factory=list)

    def reset(self) -> None:
        self.blocks.clear()


@dataclass
class Session:
    inputs: list[InputImage] = field(default_factory=list)
    filter_mapping: FilterMapping = field(default_factory=FilterMapping)
    stage1_workflows: dict[str, Workflow] = field(
        default_factory=lambda: {"L": Workflow(), "R": Workflow(), "G": Workflow(), "B": Workflow(), "FILTER": Workflow()}
    )
    stage2_workflow: Workflow = field(default_factory=Workflow)
    mixer: MixerSettings = field(default_factory=MixerSettings)
    roi: ROI = field(default_factory=ROI)
    performance: PerformanceSettings = field(default_factory=PerformanceSettings)
    viewer_state: ViewerState = field(default_factory=ViewerState)
    stage1_blocks: list[BlockInstance] = field(default_factory=list)
    stage2_blocks: list[BlockInstance] = field(default_factory=list)
    selected_stage: int = 1

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @staticmethod
    def _fw_from_dict(data: dict[str, Any]) -> FilterWeights:
        return FilterWeights(**data)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Session":
        mappings = {
            key: cls._fw_from_dict(value)
            for key, value in data.get("filter_mapping", {}).get("mappings", {}).items()
        }
        filter_mapping = FilterMapping(
            normalize_weights=data.get("filter_mapping", {}).get("normalize_weights", True),
            mappings=mappings,
        )

        stage1 = {}
        for key in ("L", "R", "G", "B", "FILTER"):
            blocks = [BlockInstance(**b) for b in data.get("stage1_workflows", {}).get(key, {}).get("blocks", [])]
            stage1[key] = Workflow(blocks=blocks)

        stage2 = Workflow(blocks=[BlockInstance(**b) for b in data.get("stage2_workflow", {}).get("blocks", [])])

        stage1_blocks = [BlockInstance(**b) for b in data.get("stage1_blocks", [])]
        stage2_blocks = [BlockInstance(**b) for b in data.get("stage2_blocks", [])]
        if stage1_blocks and not any(w.blocks for w in stage1.values()):
            for block in stage1_blocks:
                key = block.channel if block.channel in stage1 else "L"
                stage1[key].blocks.append(block)
        if not stage1_blocks and not stage2_blocks:
            stage2_blocks = [BlockInstance(**b) for b in data.get("stage2_workflow", {}).get("blocks", [])]

        for workflow in stage1.values():
            for block in workflow.blocks:
                block.params = complete_block_params(block.type, block.params)
        for block in stage2.blocks:
            block.params = complete_block_params(block.type, block.params)
        for block in stage1_blocks:
            block.params = complete_block_params(block.type, block.params)
        for block in stage2_blocks:
            block.params = complete_block_params(block.type, block.params)

        session = cls(
            inputs=[InputImage(path=i["path"], role=Role(i.get("role", "FILTER")), filter_name=i.get("filter_name", ""), assume_linear=i.get("assume_linear", False)) for i in data.get("inputs", [])],
            filter_mapping=filter_mapping,
            stage1_workflows=stage1,
            stage2_workflow=stage2,
            mixer=MixerSettings(**data.get("mixer", {})),
            roi=ROI(**data.get("roi", {})),
            performance=PerformanceSettings(**data.get("performance", {})),
            viewer_state=ViewerState(**data.get("viewer_state", {})),
            stage1_blocks=stage1_blocks,
            stage2_blocks=stage2_blocks,
            selected_stage=data.get("selected_stage", 1),
        )
        if not session.stage2_workflow.blocks and session.stage2_blocks:
            session.stage2_workflow.blocks = list(session.stage2_blocks)
        return session

    def resolve_paths(self, base: Path) -> None:
        for image in self.inputs:
            p = Path(image.path)
            if not p.is_absolute():
                image.path = str((base / p).resolve())
