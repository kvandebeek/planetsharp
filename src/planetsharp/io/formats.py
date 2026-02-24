from __future__ import annotations

from pathlib import Path

SUPPORTED_FORMATS = {".png", ".bmp", ".tif", ".tiff", ".jpg", ".jpeg", ".xisf", ".fits"}


def detect_format(path: str) -> str:
    ext = Path(path).suffix.lower()
    if ext not in SUPPORTED_FORMATS:
        raise ValueError(f"Unsupported format: {ext}")
    return ext


def read_image(path: str) -> dict:
    ext = detect_format(path)
    return {"path": path, "format": ext, "data": [[0.0]], "bit_depth": 32, "channels": "RGB"}


def write_image(path: str, image: dict, bit_depth: int = 16) -> None:
    detect_format(path)
    if bit_depth not in (8, 16, 32):
        raise ValueError("Bit depth must be 8, 16, or 32")
    Path(path).write_text(f"PlanetSharp placeholder image {bit_depth}bit\n", encoding="utf-8")
