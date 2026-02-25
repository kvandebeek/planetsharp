from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np


@dataclass
class ParameterSpec:
    key: str
    label: str
    min_value: float
    max_value: float
    step: float
    default: float


@dataclass
class BlockDefinition:
    block_type: str
    label: str
    parameters: list[ParameterSpec]
    apply_fn: Callable[[np.ndarray, dict[str, float]], np.ndarray]


@dataclass
class PipelineBlock:
    block_type: str
    label: str
    enabled: bool = True
    parameters: dict[str, float] = field(default_factory=dict)

    def reset_defaults(self, definition: BlockDefinition) -> None:
        self.parameters = {spec.key: spec.default for spec in definition.parameters}


# --- Processing functions ---
def _clip01(image: np.ndarray) -> np.ndarray:
    return np.clip(image, 0.0, 1.0)


def apply_brightness(image: np.ndarray, params: dict[str, float]) -> np.ndarray:
    return _clip01(image + params["brightness"])


def apply_contrast(image: np.ndarray, params: dict[str, float]) -> np.ndarray:
    factor = params["contrast"]
    return _clip01((image - 0.5) * factor + 0.5)


def apply_saturation(image: np.ndarray, params: dict[str, float]) -> np.ndarray:
    saturation = params["saturation"]
    if image.ndim != 3 or image.shape[2] < 3:
        return image
    luminance = (
        0.2126 * image[..., 0:1] + 0.7152 * image[..., 1:2] + 0.0722 * image[..., 2:3]
    )
    return _clip01(luminance + (image - luminance) * saturation)


def _gaussian_kernel_1d(size: int, strength: float) -> np.ndarray:
    size = max(1, int(size))
    if size % 2 == 0:
        size += 1
    radius = size // 2
    sigma = max(0.01, strength * max(1.0, radius))
    x = np.arange(-radius, radius + 1, dtype=np.float32)
    kernel = np.exp(-(x * x) / (2.0 * sigma * sigma))
    kernel /= np.sum(kernel)
    return kernel


def _convolve_axis(image: np.ndarray, kernel: np.ndarray, axis: int) -> np.ndarray:
    radius = len(kernel) // 2
    pad_width = [(0, 0)] * image.ndim
    pad_width[axis] = (radius, radius)
    padded = np.pad(image, pad_width, mode="reflect")

    result = np.zeros_like(image)
    for idx, weight in enumerate(kernel):
        shift = idx
        slicer = [slice(None)] * image.ndim
        slicer[axis] = slice(shift, shift + image.shape[axis])
        result += padded[tuple(slicer)] * weight
    return result


def apply_gaussian_blur(image: np.ndarray, params: dict[str, float]) -> np.ndarray:
    size = int(round(params["size"]))
    strength = params["strength"]
    if size <= 1 or strength <= 0.0:
        return image
    kernel = _gaussian_kernel_1d(size, strength)
    blurred = _convolve_axis(image, kernel, axis=1)
    blurred = _convolve_axis(blurred, kernel, axis=0)
    return _clip01(blurred)


def block_definitions() -> dict[str, BlockDefinition]:
    definitions = {
        "brightness": BlockDefinition(
            block_type="brightness",
            label="Brightness",
            parameters=[ParameterSpec("brightness", "Brightness", -1.0, 1.0, 0.01, 0.0)],
            apply_fn=apply_brightness,
        ),
        "contrast": BlockDefinition(
            block_type="contrast",
            label="Contrast",
            parameters=[ParameterSpec("contrast", "Contrast", 0.0, 3.0, 0.01, 1.0)],
            apply_fn=apply_contrast,
        ),
        "saturation": BlockDefinition(
            block_type="saturation",
            label="Saturation",
            parameters=[ParameterSpec("saturation", "Saturation", 0.0, 3.0, 0.01, 1.0)],
            apply_fn=apply_saturation,
        ),
        "gaussian_blur": BlockDefinition(
            block_type="gaussian_blur",
            label="Gaussian blur",
            parameters=[
                ParameterSpec("size", "Blur size (px)", 1.0, 51.0, 1.0, 3.0),
                ParameterSpec("strength", "Blur strength", 0.0, 1.0, 0.01, 0.3),
            ],
            apply_fn=apply_gaussian_blur,
        ),
    }
    return definitions


def instantiate_block(block_type: str, definitions: dict[str, BlockDefinition]) -> PipelineBlock:
    definition = definitions[block_type]
    return PipelineBlock(
        block_type=block_type,
        label=definition.label,
        enabled=True,
        parameters={spec.key: spec.default for spec in definition.parameters},
    )


def serialize_block(block: PipelineBlock) -> dict[str, Any]:
    return {
        "type": block.block_type,
        "label": block.label,
        "enabled": block.enabled,
        "parameters": block.parameters,
    }


def deserialize_block(data: dict[str, Any], definitions: dict[str, BlockDefinition]) -> PipelineBlock:
    block_type = data["type"]
    block = instantiate_block(block_type, definitions)
    block.enabled = bool(data.get("enabled", True))
    incoming = data.get("parameters", {})
    for spec in definitions[block_type].parameters:
        if spec.key in incoming:
            block.parameters[spec.key] = float(incoming[spec.key])
    return block
