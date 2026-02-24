from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from planetsharp.core.models import BlockInstance, Session

TEMPLATE_SUFFIX = ".planetsharp-template.json"
SCHEMA_VERSION = 1


class TemplateStore:
    @staticmethod
    def save(path: str, session: Session) -> None:
        stage1 = [TemplateStore._serialize_block(b) for ch in ("L", "R", "G", "B", "FILTER") for b in session.stage1_workflows[ch].blocks]
        if not stage1 and session.stage1_blocks:
            stage1 = [TemplateStore._serialize_block(b) for b in session.stage1_blocks]
        payload = {
            "schema_version": SCHEMA_VERSION,
            "stage1": stage1,
            "stage2": [TemplateStore._serialize_block(b) for b in session.stage2_blocks],
        }
        Path(path).write_text(json.dumps(payload, indent=2), encoding="utf-8")

    @staticmethod
    def load(path: str) -> dict[str, list[BlockInstance]]:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if payload.get("schema_version") != SCHEMA_VERSION:
            raise ValueError(f"Unsupported template schema_version: {payload.get('schema_version')}")
        return {
            "stage1": [TemplateStore._block_from_dict(b) for b in payload.get("stage1", [])],
            "stage2": [TemplateStore._block_from_dict(b) for b in payload.get("stage2", [])],
        }

    @staticmethod
    def _serialize_block(block: BlockInstance) -> dict[str, Any]:
        return {
            "id": block.id,
            "type": block.type,
            "enabled": block.enabled,
            "params": block.params,
            "channel": block.channel,
            "metadata": {},
        }

    @staticmethod
    def _block_from_dict(data: dict[str, Any]) -> BlockInstance:
        kwargs = {
            "type": data["type"],
            "enabled": data.get("enabled", True),
            "params": data.get("params", {}),
            "channel": data.get("channel"),
        }
        if data.get("id"):
            kwargs["id"] = data["id"]
        return BlockInstance(**kwargs)
