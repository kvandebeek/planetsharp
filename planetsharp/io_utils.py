from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import tifffile
from PIL import Image

PIPELINE_VERSION = 1
SUPPORTED_DTYPES = {np.uint8, np.uint16, np.uint32, np.float32, np.int32}
IMAGECODECS_REQUIRED_MESSAGE = "requires the 'imagecodecs' package"


def _as_rgb(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        image = np.stack([image, image, image], axis=-1)
    if image.ndim == 3 and image.shape[2] == 4:
        image = image[..., :3]
    if image.ndim != 3 or image.shape[2] < 3:
        raise ValueError("Unsupported image channel layout.")
    return image[..., :3]


def load_image(path: str) -> tuple[np.ndarray, np.ndarray]:
    p = Path(path)
    suffix = p.suffix.lower()
    if suffix not in {".png", ".tif", ".tiff"}:
        raise ValueError("Only PNG/TIF files are allowed.")

    if suffix == ".png":
        arr = np.array(Image.open(path))
    else:
        arr = _read_tiff(path)

    if arr.dtype.type not in SUPPORTED_DTYPES:
        raise ValueError("Only 8-bit, 16-bit, or 32-bit images are supported.")

    arr = _as_rgb(arr)

    if np.issubdtype(arr.dtype, np.integer):
        info = np.iinfo(arr.dtype)
        normalized = arr.astype(np.float32) / float(info.max)
    else:
        normalized = arr.astype(np.float32)
        normalized = np.clip(normalized, 0.0, 1.0)
    return arr, normalized


def _read_tiff(path: str) -> np.ndarray:
    try:
        return tifffile.imread(path)
    except ValueError as exc:
        if IMAGECODECS_REQUIRED_MESSAGE not in str(exc):
            raise
        return np.array(Image.open(path))


def save_image_16bit(path: str, rendered_float: np.ndarray) -> None:
    clipped = np.clip(rendered_float, 0.0, 1.0)
    out = (clipped * 65535.0).round().astype(np.uint16)
    suffix = Path(path).suffix.lower()
    if suffix == ".png":
        Image.fromarray(out, mode="RGB").save(path)
    elif suffix in {".tif", ".tiff"}:
        tifffile.imwrite(path, out)
    else:
        raise ValueError("Output file must be PNG or TIF.")


def save_pipeline(path: str, blocks: list[dict[str, Any]]) -> None:
    payload = {"version": PIPELINE_VERSION, "blocks": blocks}
    Path(path).write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_pipeline(path: str) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("version") != PIPELINE_VERSION:
        raise ValueError(
            f"Unsupported pipeline version {payload.get('version')}; expected {PIPELINE_VERSION}."
        )
    if "blocks" not in payload or not isinstance(payload["blocks"], list):
        raise ValueError("Invalid pipeline JSON: missing 'blocks' list.")
    return payload
