from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Literal

import numpy as np


@dataclass
class ParameterSpec:
    key: str
    label: str
    min_value: float
    max_value: float
    step: float
    default: float | str
    input_type: Literal["slider", "choice"] = "slider"
    choices: list[str] = field(default_factory=list)


@dataclass
class BlockDefinition:
    block_type: str
    label: str
    parameters: list[ParameterSpec]
    apply_fn: Callable[[np.ndarray, dict[str, float | str]], np.ndarray]


@dataclass
class PipelineBlock:
    block_type: str
    label: str
    enabled: bool = True
    parameters: dict[str, float | str] = field(default_factory=dict)

    def reset_defaults(self, definition: BlockDefinition) -> None:
        self.parameters = {spec.key: spec.default for spec in definition.parameters}


# --- Processing functions ---
def _clip01(image: np.ndarray) -> np.ndarray:
    return np.clip(image, 0.0, 1.0)


def apply_brightness(image: np.ndarray, params: dict[str, float | str]) -> np.ndarray:
    return _clip01(image + float(params["brightness"]))


def apply_contrast(image: np.ndarray, params: dict[str, float | str]) -> np.ndarray:
    factor = float(params["contrast"])
    return _clip01((image - 0.5) * factor + 0.5)


def apply_saturation(image: np.ndarray, params: dict[str, float | str]) -> np.ndarray:
    saturation = float(params["saturation"])
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


def _srgb_to_linear(image: np.ndarray) -> np.ndarray:
    return np.where(image <= 0.04045, image / 12.92, ((image + 0.055) / 1.055) ** 2.4)


def _linear_to_srgb(image: np.ndarray) -> np.ndarray:
    return np.where(image <= 0.0031308, 12.92 * image, 1.055 * (image ** (1.0 / 2.4)) - 0.055)


def _rgb_to_lab(image: np.ndarray) -> np.ndarray:
    linear = _srgb_to_linear(_clip01(image)).astype(np.float32)
    matrix = np.array(
        [
            [0.4124564, 0.3575761, 0.1804375],
            [0.2126729, 0.7151522, 0.0721750],
            [0.0193339, 0.1191920, 0.9503041],
        ],
        dtype=np.float32,
    )
    xyz = linear @ matrix.T
    white = np.array([0.95047, 1.0, 1.08883], dtype=np.float32)
    xyz_scaled = xyz / white

    delta = 6.0 / 29.0
    def f(t: np.ndarray) -> np.ndarray:
        return np.where(t > delta**3, np.cbrt(t), (t / (3.0 * delta**2)) + (4.0 / 29.0))

    fx, fy, fz = (f(xyz_scaled[..., i]) for i in range(3))
    l = (116.0 * fy) - 16.0
    a = 500.0 * (fx - fy)
    b = 200.0 * (fy - fz)
    return np.stack([l, a, b], axis=-1).astype(np.float32)


def _lab_to_rgb(lab: np.ndarray) -> np.ndarray:
    l = lab[..., 0]
    a = lab[..., 1]
    b = lab[..., 2]
    fy = (l + 16.0) / 116.0
    fx = fy + (a / 500.0)
    fz = fy - (b / 200.0)

    delta = 6.0 / 29.0
    def finv(t: np.ndarray) -> np.ndarray:
        return np.where(t > delta, t**3, 3.0 * delta**2 * (t - 4.0 / 29.0))

    x = finv(fx) * 0.95047
    y = finv(fy) * 1.0
    z = finv(fz) * 1.08883
    xyz = np.stack([x, y, z], axis=-1)
    inv_matrix = np.array(
        [
            [3.2404542, -1.5371385, -0.4985314],
            [-0.9692660, 1.8760108, 0.0415560],
            [0.0556434, -0.2040259, 1.0572252],
        ],
        dtype=np.float32,
    )
    linear = xyz @ inv_matrix.T
    srgb = _linear_to_srgb(np.clip(linear, 0.0, 1.0))
    return _clip01(srgb.astype(np.float32))


def apply_gaussian_blur(image: np.ndarray, params: dict[str, float | str]) -> np.ndarray:
    size = float(params["size"])
    strength = float(params["strength"])
    channel_mode = str(params.get("channel_mode", "All channels"))
    lab_balance = float(params.get("lab_balance", 0.0))
    lab_balance = float(np.clip(lab_balance, 0.0, 1.0))
    if size <= 0.0 or strength <= 0.0:
        return image

    kernel_size = max(1, int(round(size)))
    kernel = _gaussian_kernel_1d(kernel_size, strength)

    def blur_channels(data: np.ndarray) -> np.ndarray:
        blurred_data = _convolve_axis(data, kernel, axis=1)
        return _convolve_axis(blurred_data, kernel, axis=0)

    if channel_mode == "Luminance only (LAB L)":
        lab = _rgb_to_lab(image)
        l_blur = lab.copy()
        l_blur[..., 0:1] = blur_channels(l_blur[..., 0:1])
        if lab_balance <= 0.0:
            return _lab_to_rgb(l_blur)

        ab_blur = lab.copy()
        ab_blur[..., 1:3] = blur_channels(ab_blur[..., 1:3])
        blended = (1.0 - lab_balance) * l_blur + lab_balance * ab_blur
        return _lab_to_rgb(blended)
    if channel_mode == "A/B only (LAB chroma)":
        lab = _rgb_to_lab(image)
        ab_blur = lab.copy()
        ab_blur[..., 1:3] = blur_channels(ab_blur[..., 1:3])
        if lab_balance <= 0.0:
            return _lab_to_rgb(ab_blur)

        l_blur = lab.copy()
        l_blur[..., 0:1] = blur_channels(l_blur[..., 0:1])
        blended = (1.0 - lab_balance) * ab_blur + lab_balance * l_blur
        return _lab_to_rgb(blended)

    blurred = blur_channels(image)
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
                ParameterSpec("size", "Blur size (px)", 0.0, 8.0, 0.05, 0.0),
                ParameterSpec("strength", "Blur strength", 0.0, 1.0, 0.05, 0.0),
                ParameterSpec(
                    "channel_mode",
                    "Blur channels",
                    0.0,
                    0.0,
                    1.0,
                    "All channels",
                    input_type="choice",
                    choices=["All channels", "Luminance only (LAB L)", "A/B only (LAB chroma)"],
                ),
                ParameterSpec("lab_balance", "L/AB balance", 0.0, 1.0, 0.01, 0.0),
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
        if spec.key not in incoming:
            continue
        if spec.input_type == "choice":
            value = str(incoming[spec.key])
            block.parameters[spec.key] = value if value in spec.choices else spec.default
        else:
            block.parameters[spec.key] = float(incoming[spec.key])
    return block
