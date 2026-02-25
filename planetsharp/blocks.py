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


def apply_brightness_contrast(image: np.ndarray, params: dict[str, float | str]) -> np.ndarray:
    brightness = float(params.get("brightness", 0.0))
    contrast = float(params.get("contrast", 1.0))
    shifted = image + brightness
    return _clip01((shifted - 0.5) * contrast + 0.5)


def apply_hue_shift(image: np.ndarray, params: dict[str, float | str]) -> np.ndarray:
    if image.ndim != 3 or image.shape[2] < 3:
        return image

    hue_degrees = float(params.get("hue_shift", 0.0))
    hue_delta = (hue_degrees / 360.0) % 1.0

    rgb = _clip01(image.astype(np.float32))
    r = rgb[..., 0]
    g = rgb[..., 1]
    b = rgb[..., 2]

    cmax = np.maximum(np.maximum(r, g), b)
    cmin = np.minimum(np.minimum(r, g), b)
    delta = cmax - cmin

    h = np.zeros_like(cmax)
    nonzero = delta > 1e-8
    r_mask = (cmax == r) & nonzero
    g_mask = (cmax == g) & nonzero
    b_mask = (cmax == b) & nonzero
    h[r_mask] = ((g[r_mask] - b[r_mask]) / delta[r_mask]) % 6.0
    h[g_mask] = ((b[g_mask] - r[g_mask]) / delta[g_mask]) + 2.0
    h[b_mask] = ((r[b_mask] - g[b_mask]) / delta[b_mask]) + 4.0
    h = (h / 6.0 + hue_delta) % 1.0

    s = np.zeros_like(cmax)
    nonzero_value = cmax > 1e-8
    s[nonzero_value] = delta[nonzero_value] / cmax[nonzero_value]
    v = cmax

    i = np.floor(h * 6.0).astype(np.int32)
    f = h * 6.0 - i
    p = v * (1.0 - s)
    q = v * (1.0 - f * s)
    t = v * (1.0 - (1.0 - f) * s)

    i_mod = i % 6
    out_r = np.select([i_mod == k for k in range(6)], [v, q, p, p, t, v], default=v)
    out_g = np.select([i_mod == k for k in range(6)], [t, v, v, q, p, p], default=v)
    out_b = np.select([i_mod == k for k in range(6)], [p, p, t, v, v, q], default=v)
    return _clip01(np.stack([out_r, out_g, out_b], axis=-1).astype(np.float32))


def apply_channel_balance(image: np.ndarray, params: dict[str, float | str]) -> np.ndarray:
    if image.ndim != 3 or image.shape[2] < 3:
        return image
    red = float(params.get("red_balance", 1.0))
    green = float(params.get("green_balance", 1.0))
    blue = float(params.get("blue_balance", 1.0))
    gains = np.array([red, green, blue], dtype=np.float32)
    return _clip01(image * gains)


def apply_saturation(image: np.ndarray, params: dict[str, float | str]) -> np.ndarray:
    if image.ndim != 3 or image.shape[2] < 3:
        return image

    # Backward compatibility: if an old pipeline provides only "saturation",
    # use the same value for every tonal range.
    legacy_saturation = float(params.get("saturation", 1.0))
    saturation_shadows = float(params.get("saturation_shadows", legacy_saturation))
    saturation_midtones = float(params.get("saturation_midtones", legacy_saturation))
    saturation_highlights = float(params.get("saturation_highlights", legacy_saturation))

    luminance = (
        0.2126 * image[..., 0:1] + 0.7152 * image[..., 1:2] + 0.0722 * image[..., 2:3]
    )

    # Build smooth per-tonal-range weights from luminance in [0, 1].
    shadow_weight = np.clip((0.5 - luminance) * 2.0, 0.0, 1.0)
    highlight_weight = np.clip((luminance - 0.5) * 2.0, 0.0, 1.0)
    midtone_weight = np.clip(1.0 - np.abs(luminance - 0.5) * 2.0, 0.0, 1.0)

    saturation_map = (
        shadow_weight * saturation_shadows
        + midtone_weight * saturation_midtones
        + highlight_weight * saturation_highlights
    )
    return _clip01(luminance + (image - luminance) * saturation_map)


def apply_midtone_transfer(image: np.ndarray, params: dict[str, float | str]) -> np.ndarray:
    transfer = float(np.clip(float(params["midtone_transfer"]), 0.0, 1.0))
    centered = (transfer - 0.5) * 2.0  # -1.0 to 1.0, neutral at 0.5
    gamma = 1.0 - (centered * 0.15)  # gentle range: 0.85 to 1.15
    return _clip01(np.power(_clip01(image), gamma))


def apply_levels_midtone_transfer(image: np.ndarray, params: dict[str, float | str]) -> np.ndarray:
    return apply_levels(apply_midtone_transfer(image, params), params)


def _smoothstep(edge0: float, edge1: float, x: np.ndarray) -> np.ndarray:
    if edge1 <= edge0:
        return np.where(x >= edge1, 1.0, 0.0).astype(np.float32)
    t = np.clip((x - edge0) / (edge1 - edge0), 0.0, 1.0)
    return (t * t * (3.0 - 2.0 * t)).astype(np.float32)


def normalize_levels_boundaries(params: dict[str, float | str]) -> dict[str, float]:
    # Keep three canonical boundaries so tonal ranges always "stick" together
    # with no gaps:
    # shadows/high-mid boundary  => shadow_upper == low_mid_lower
    # low-mid/high-mid boundary => low_mid_upper == high_mid_lower
    # high-mid/highlights       => high_mid_upper == highlights_lower
    shadow_upper = float(params.get("shadow_upper", params.get("low_mid_lower", 0.25)))
    low_mid_upper = float(params.get("low_mid_upper", params.get("high_mid_lower", 0.50)))
    high_mid_upper = float(params.get("high_mid_upper", params.get("highlights_lower", 0.78)))

    shadow_upper = float(np.clip(shadow_upper, 0.01, 0.98))
    low_mid_upper = float(np.clip(low_mid_upper, shadow_upper + 0.01, 0.99))
    high_mid_upper = float(np.clip(high_mid_upper, low_mid_upper + 0.01, 1.0))

    low_mid_lower = shadow_upper
    high_mid_lower = low_mid_upper
    highlights_lower = high_mid_upper

    return {
        "shadow_upper": shadow_upper,
        "low_mid_lower": low_mid_lower,
        "low_mid_upper": low_mid_upper,
        "high_mid_lower": high_mid_lower,
        "high_mid_upper": high_mid_upper,
        "highlights_lower": highlights_lower,
    }


def apply_levels(image: np.ndarray, params: dict[str, float | str]) -> np.ndarray:
    if image.ndim != 3 or image.shape[2] < 3:
        return image

    boundaries = normalize_levels_boundaries(params)
    shadows = float(params.get("shadows", 0.0))
    low_mid = float(params.get("low_mid", 0.0))
    high_mid = float(params.get("high_mid", 0.0))
    highlights = float(params.get("highlights", 0.0))

    luminance = (
        0.2126 * image[..., 0:1] + 0.7152 * image[..., 1:2] + 0.0722 * image[..., 2:3]
    ).astype(np.float32)

    softness = 0.03
    shadow_weight = 1.0 - _smoothstep(
        boundaries["shadow_upper"] - softness,
        boundaries["shadow_upper"] + softness,
        luminance,
    )
    low_mid_weight = _smoothstep(
        boundaries["low_mid_lower"] - softness,
        boundaries["low_mid_lower"] + softness,
        luminance,
    ) * (1.0 - _smoothstep(
        boundaries["low_mid_upper"] - softness,
        boundaries["low_mid_upper"] + softness,
        luminance,
    ))
    high_mid_weight = _smoothstep(
        boundaries["high_mid_lower"] - softness,
        boundaries["high_mid_lower"] + softness,
        luminance,
    ) * (1.0 - _smoothstep(
        boundaries["high_mid_upper"] - softness,
        boundaries["high_mid_upper"] + softness,
        luminance,
    ))
    highlight_weight = _smoothstep(
        boundaries["highlights_lower"] - softness,
        boundaries["highlights_lower"] + softness,
        luminance,
    )

    total_weight = shadow_weight + low_mid_weight + high_mid_weight + highlight_weight
    total_weight = np.clip(total_weight, 1e-6, None)
    shadow_weight /= total_weight
    low_mid_weight /= total_weight
    high_mid_weight /= total_weight
    highlight_weight /= total_weight

    gain = (
        shadow_weight * (1.0 + shadows)
        + low_mid_weight * (1.0 + low_mid)
        + high_mid_weight * (1.0 + high_mid)
        + highlight_weight * (1.0 + highlights)
    )
    return _clip01(image * gain)


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


def _gaussian_blur_all_channels(image: np.ndarray, size: float, strength: float) -> np.ndarray:
    if size <= 0.0 or strength <= 0.0:
        return image
    kernel_size = max(1, int(round(size)))
    kernel = _gaussian_kernel_1d(kernel_size, strength)
    blurred_data = _convolve_axis(image, kernel, axis=1)
    return _convolve_axis(blurred_data, kernel, axis=0)


def apply_unsharp_mask(image: np.ndarray, params: dict[str, float | str]) -> np.ndarray:
    radius = float(params.get("radius", 0.0))
    amount = float(params.get("amount", 0.0))
    threshold = float(params.get("threshold", 0.0))
    mode = str(params.get("mode", "Luminance only"))

    if radius <= 0.0 or amount <= 0.0:
        return image

    if mode == "Luminance only" and image.ndim == 3 and image.shape[2] >= 3:
        lab = _rgb_to_lab(image)
        l = lab[..., 0:1]
        l_blur = _gaussian_blur_all_channels(l, radius, 0.8)
        detail = l - l_blur
        if threshold > 0.0:
            mask = (np.abs(detail) >= threshold).astype(np.float32)
            detail *= mask
        lab[..., 0:1] = l + detail * amount
        return _lab_to_rgb(lab)

    blur = _gaussian_blur_all_channels(image, radius, 0.8)
    detail = image - blur
    if threshold > 0.0:
        mask = (np.abs(detail) >= threshold).astype(np.float32)
        detail *= mask
    return _clip01(image + detail * amount)


def apply_high_pass_detail(image: np.ndarray, params: dict[str, float | str]) -> np.ndarray:
    radius = float(params.get("radius", 0.0))
    amount = float(params.get("amount", 0.0))
    softness = float(np.clip(float(params.get("softness", 0.0)), 0.0, 1.0))
    if radius <= 0.0 or amount <= 0.0:
        return image

    base = _gaussian_blur_all_channels(image, radius, 0.8)
    detail = image - base
    if softness > 0.0:
        limiter = 1.0 - softness * np.clip(np.abs(detail) * 4.0, 0.0, 1.0)
        detail *= limiter
    return _clip01(image + detail * amount)


def apply_rl_deconvolution(image: np.ndarray, params: dict[str, float | str]) -> np.ndarray:
    radius = float(params.get("radius", 0.0))
    iterations = int(round(float(params.get("iterations", 0.0))))
    damping = float(np.clip(float(params.get("damping", 0.0)), 0.0, 1.0))

    if radius <= 0.0 or iterations <= 0:
        return image

    work = _clip01(image.astype(np.float32))
    kernel_size = max(3, int(round(radius * 2.0 + 1.0)))
    kernel = _gaussian_kernel_1d(kernel_size, 0.9)

    if work.ndim == 3 and work.shape[2] >= 3:
        lab = _rgb_to_lab(work)
        estimate = np.clip(lab[..., 0:1] / 100.0, 1e-4, 1.0)
        observed = estimate.copy()
        for _ in range(iterations):
            conv = _convolve_axis(_convolve_axis(estimate, kernel, axis=1), kernel, axis=0)
            relative_blur = observed / np.clip(conv, 1e-5, None)
            estimate *= _convolve_axis(_convolve_axis(relative_blur, kernel, axis=1), kernel, axis=0)
            if damping > 0.0:
                estimate = (1.0 - damping) * estimate + damping * observed
            estimate = np.clip(estimate, 0.0, 1.0)
        lab[..., 0:1] = estimate * 100.0
        return _lab_to_rgb(lab)

    estimate = np.clip(work, 1e-4, 1.0)
    observed = estimate.copy()
    for _ in range(iterations):
        conv = _convolve_axis(_convolve_axis(estimate, kernel, axis=1), kernel, axis=0)
        relative_blur = observed / np.clip(conv, 1e-5, None)
        estimate *= _convolve_axis(_convolve_axis(relative_blur, kernel, axis=1), kernel, axis=0)
        if damping > 0.0:
            estimate = (1.0 - damping) * estimate + damping * observed
        estimate = np.clip(estimate, 0.0, 1.0)
    return _clip01(estimate)


def _atrous_blur(image: np.ndarray, step: int, mode: str) -> np.ndarray:
    weights = np.array([1, 4, 6, 4, 1], dtype=np.float32)
    if mode == "bspline":
        weights = np.array([1, 3, 8, 3, 1], dtype=np.float32)
    kernel = weights / np.sum(weights)
    if step <= 1:
        return _convolve_axis(_convolve_axis(image, kernel, axis=1), kernel, axis=0)

    radius = 2 * step
    sparse = np.zeros(radius * 2 + 1, dtype=np.float32)
    sparse[0] = kernel[0]
    sparse[step] = kernel[1]
    sparse[2 * step] = kernel[2]
    sparse[3 * step] = kernel[3]
    sparse[4 * step] = kernel[4]
    return _convolve_axis(_convolve_axis(image, sparse, axis=1), sparse, axis=0)


def apply_wavelet_sharpening(image: np.ndarray, params: dict[str, float | str]) -> np.ndarray:
    mode = str(params.get("wavelet_mode", "linear"))
    if mode not in {"linear", "bspline"}:
        mode = "linear"

    layer_amounts = [
        float(params.get("layer1_amount", 0.0)),
        float(params.get("layer2_amount", 0.0)),
        float(params.get("layer3_amount", 0.0)),
    ]
    layer_noise = [
        float(np.clip(float(params.get("layer1_noise_reduction", 0.0)), 0.0, 1.0)),
        float(np.clip(float(params.get("layer2_noise_reduction", 0.0)), 0.0, 1.0)),
        float(np.clip(float(params.get("layer3_noise_reduction", 0.0)), 0.0, 1.0)),
    ]
    recovery = float(params.get("recovery_amount", 0.0))

    if all(abs(amount) < 1e-6 for amount in layer_amounts) and recovery <= 0.0:
        return image

    base = _clip01(image.astype(np.float32))
    current = base
    detail_layers: list[np.ndarray] = []
    for level in range(3):
        smooth = _atrous_blur(current, step=2**level, mode=mode)
        detail_layers.append(current - smooth)
        current = smooth

    sharpened = base.copy()
    for amount, noise_reduction, detail in zip(layer_amounts, layer_noise, detail_layers):
        reduced = detail * (1.0 - noise_reduction)
        sharpened += reduced * amount

    if recovery > 0.0:
        recovery_blur = _atrous_blur(base, step=4, mode=mode)
        sharpened = (1.0 - recovery) * sharpened + recovery * recovery_blur

    return _clip01(sharpened)


def _make_psf(psf_size: int, seeing_index: float, psf_strength: float) -> np.ndarray:
    psf_size = max(3, int(psf_size))
    if psf_size % 2 == 0:
        psf_size += 1
    radius = psf_size // 2
    y, x = np.mgrid[-radius : radius + 1, -radius : radius + 1]
    sigma = max(0.3, seeing_index)
    gaussian = np.exp(-((x * x + y * y) / (2.0 * sigma * sigma))).astype(np.float32)
    halo_sigma = max(0.5, sigma * 1.8)
    halo = np.exp(-((x * x + y * y) / (2.0 * halo_sigma * halo_sigma))).astype(np.float32)
    blend = float(np.clip(psf_strength, 0.0, 1.0))
    psf = (1.0 - blend) * gaussian + blend * halo
    psf /= np.sum(psf)
    return psf.astype(np.float32)


def _fft_deconvolve(channel: np.ndarray, psf: np.ndarray, strength: float) -> np.ndarray:
    h, w = channel.shape
    psf_pad = np.zeros((h, w), dtype=np.float32)
    ph, pw = psf.shape
    psf_pad[:ph, :pw] = psf
    psf_pad = np.roll(psf_pad, -ph // 2, axis=0)
    psf_pad = np.roll(psf_pad, -pw // 2, axis=1)

    otf = np.fft.fft2(psf_pad)
    channel_f = np.fft.fft2(channel)
    k = max(1e-5, (1.0 - np.clip(strength, 0.0, 1.0)) * 0.03)
    deconv = np.fft.ifft2(channel_f * np.conj(otf) / (np.abs(otf) ** 2 + k)).real
    return deconv.astype(np.float32)


def apply_psd_deconvolution(image: np.ndarray, params: dict[str, float | str]) -> np.ndarray:
    psf_size = int(round(float(params.get("psf_size", 3.0))))
    seeing_index = float(params.get("seeing_index", 1.2))
    psf_strength = float(params.get("psf_strength", 0.3))
    blend = float(np.clip(float(params.get("blend", 1.0)), 0.0, 1.0))

    if blend <= 0.0:
        return image

    psf = _make_psf(psf_size, seeing_index, psf_strength)
    work = _clip01(image.astype(np.float32))

    if work.ndim == 3 and work.shape[2] >= 3:
        r_amt = float(np.clip(float(params.get("red_amount", 1.0)), 0.0, 1.0))
        g_amt = float(np.clip(float(params.get("green_amount", 1.0)), 0.0, 1.0))
        b_amt = float(np.clip(float(params.get("blue_amount", 1.0)), 0.0, 1.0))
        l_amt = float(np.clip(float(params.get("luminance_amount", 1.0)), 0.0, 1.0))

        channels = [work[..., 0], work[..., 1], work[..., 2]]
        per_channel = [r_amt, g_amt, b_amt]
        processed = []
        for chan, amount in zip(channels, per_channel):
            deconv = _fft_deconvolve(chan, psf, amount)
            processed.append((1.0 - amount) * chan + amount * deconv)
        rgb = np.stack(processed, axis=-1)

        lab = _rgb_to_lab(_clip01(rgb))
        l_norm = np.clip(lab[..., 0:1] / 100.0, 0.0, 1.0)
        l_deconv = _fft_deconvolve(l_norm[..., 0], psf, l_amt)[..., np.newaxis]
        lab[..., 0:1] = np.clip((1.0 - l_amt) * l_norm + l_amt * l_deconv, 0.0, 1.0) * 100.0
        wet = _lab_to_rgb(lab)
    else:
        amount = float(np.clip(float(params.get("luminance_amount", 1.0)), 0.0, 1.0))
        deconv = _fft_deconvolve(work, psf, amount)
        wet = (1.0 - amount) * work + amount * deconv

    return _clip01((1.0 - blend) * work + blend * wet)


def apply_chroma_denoise(image: np.ndarray, params: dict[str, float | str]) -> np.ndarray:
    radius = float(params.get("radius", 0.0))
    strength = float(params.get("strength", 0.0))
    if radius <= 0.0 or strength <= 0.0:
        return image
    if image.ndim != 3 or image.shape[2] < 3:
        return image

    lab = _rgb_to_lab(image)
    blurred_ab = lab[..., 1:3].copy()
    blurred_ab = _gaussian_blur_all_channels(blurred_ab, radius, 0.8)
    blend = float(np.clip(strength, 0.0, 1.0))
    lab[..., 1:3] = (1.0 - blend) * lab[..., 1:3] + blend * blurred_ab
    return _lab_to_rgb(lab)


def block_definitions() -> dict[str, BlockDefinition]:
    definitions = {
        "brightness_contrast": BlockDefinition(
            block_type="brightness_contrast",
            label="Brightness/Contrast",
            parameters=[
                ParameterSpec("brightness", "Brightness", -1.0, 1.0, 0.01, 0.0),
                ParameterSpec("contrast", "Contrast", 0.0, 3.0, 0.01, 1.0),
            ],
            apply_fn=apply_brightness_contrast,
        ),
        "saturation": BlockDefinition(
            block_type="saturation",
            label="Saturation",
            parameters=[
                ParameterSpec(
                    "saturation_shadows",
                    "Saturation of shadows",
                    0.0,
                    3.0,
                    0.01,
                    1.0,
                ),
                ParameterSpec(
                    "saturation_midtones",
                    "Saturation of midtones",
                    0.0,
                    3.0,
                    0.01,
                    1.0,
                ),
                ParameterSpec(
                    "saturation_highlights",
                    "Saturation of highlights",
                    0.0,
                    3.0,
                    0.01,
                    1.0,
                ),
            ],
            apply_fn=apply_saturation,
        ),
        "hue_shift": BlockDefinition(
            block_type="hue_shift",
            label="Hue shift",
            parameters=[ParameterSpec("hue_shift", "Hue shift (°)", -180.0, 180.0, 1.0, 0.0)],
            apply_fn=apply_hue_shift,
        ),
        "channel_balance": BlockDefinition(
            block_type="channel_balance",
            label="Channel balance",
            parameters=[
                ParameterSpec("red_balance", "Red gain", 0.5, 1.5, 0.01, 1.0),
                ParameterSpec("green_balance", "Green gain", 0.5, 1.5, 0.01, 1.0),
                ParameterSpec("blue_balance", "Blue gain", 0.5, 1.5, 0.01, 1.0),
            ],
            apply_fn=apply_channel_balance,
        ),
        "levels_midtone_transfer": BlockDefinition(
            block_type="levels_midtone_transfer",
            label="Levels / Midtone Transfer",
            parameters=[
                ParameterSpec(
                    "midtone_transfer",
                    "Midtone transfer",
                    0.0,
                    1.0,
                    0.01,
                    0.5,
                ),
                ParameterSpec("shadows", "Shadows tone", -0.3, 0.3, 0.01, 0.0),
                ParameterSpec("low_mid", "Low-mid tone", -0.3, 0.3, 0.01, 0.0),
                ParameterSpec("high_mid", "High-mid tone", -0.3, 0.3, 0.01, 0.0),
                ParameterSpec("highlights", "Highlights tone", -0.3, 0.3, 0.01, 0.0),
                ParameterSpec("shadow_upper", "Shadows / Low-mid boundary", 0.01, 0.98, 0.01, 0.25),
                ParameterSpec("low_mid_upper", "Low-mid / High-mid boundary", 0.02, 0.99, 0.01, 0.50),
                ParameterSpec("high_mid_upper", "High-mid / Highlights boundary", 0.03, 1.00, 0.01, 0.78),
            ],
            apply_fn=apply_levels_midtone_transfer,
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
        "unsharp_mask": BlockDefinition(
            block_type="unsharp_mask",
            label="Unsharp mask",
            parameters=[
                ParameterSpec("radius", "Radius (px)", 0.0, 8.0, 0.05, 0.0),
                ParameterSpec("amount", "Amount", 0.0, 3.0, 0.01, 0.0),
                ParameterSpec("threshold", "Threshold", 0.0, 0.2, 0.005, 0.0),
                ParameterSpec(
                    "mode",
                    "Sharpen mode",
                    0.0,
                    0.0,
                    1.0,
                    "Luminance only",
                    input_type="choice",
                    choices=["Luminance only", "All channels"],
                ),
            ],
            apply_fn=apply_unsharp_mask,
        ),
        "high_pass_detail": BlockDefinition(
            block_type="high_pass_detail",
            label="High-pass detail",
            parameters=[
                ParameterSpec("radius", "Radius (px)", 0.0, 10.0, 0.05, 0.0),
                ParameterSpec("amount", "Amount", 0.0, 3.0, 0.01, 0.0),
                ParameterSpec("softness", "Halo softness", 0.0, 1.0, 0.01, 0.4),
            ],
            apply_fn=apply_high_pass_detail,
        ),
        "wavelet_sharpening": BlockDefinition(
            block_type="wavelet_sharpening",
            label="Wavelet Sharpening",
            parameters=[
                ParameterSpec(
                    "wavelet_mode",
                    "Mode",
                    0.0,
                    0.0,
                    1.0,
                    "linear",
                    input_type="choice",
                    choices=["linear", "bspline"],
                ),
                ParameterSpec("layer1_amount", "Layer 1 amount", -2.0, 4.0, 0.01, 0.0),
                ParameterSpec("layer1_noise_reduction", "Layer 1 noise reduction", 0.0, 1.0, 0.01, 0.0),
                ParameterSpec("layer2_amount", "Layer 2 amount", -2.0, 4.0, 0.01, 0.0),
                ParameterSpec("layer2_noise_reduction", "Layer 2 noise reduction", 0.0, 1.0, 0.01, 0.0),
                ParameterSpec("layer3_amount", "Layer 3 amount", -2.0, 4.0, 0.01, 0.0),
                ParameterSpec("layer3_noise_reduction", "Layer 3 noise reduction", 0.0, 1.0, 0.01, 0.0),
                ParameterSpec("recovery_amount", "Recovery layer amount", 0.0, 1.0, 0.01, 0.0),
            ],
            apply_fn=apply_wavelet_sharpening,
        ),
        "psd_deconvolution": BlockDefinition(
            block_type="psd_deconvolution",
            label="PSD Deconvolution",
            parameters=[
                ParameterSpec("psf_size", "PSF Size", 3.0, 25.0, 2.0, 7.0),
                ParameterSpec("seeing_index", "Seeing Index", 0.3, 6.0, 0.05, 1.2),
                ParameterSpec("psf_strength", "PSF Strength", 0.0, 1.0, 0.01, 0.3),
                ParameterSpec("blend", "Dry/Wet Blend", 0.0, 1.0, 0.01, 1.0),
                ParameterSpec("red_amount", "R channel", 0.0, 1.0, 0.01, 1.0),
                ParameterSpec("green_amount", "G channel", 0.0, 1.0, 0.01, 1.0),
                ParameterSpec("blue_amount", "B channel", 0.0, 1.0, 0.01, 1.0),
                ParameterSpec("luminance_amount", "Luminance", 0.0, 1.0, 0.01, 1.0),
            ],
            apply_fn=apply_psd_deconvolution,
        ),
        "chroma_denoise": BlockDefinition(
            block_type="chroma_denoise",
            label="Chroma denoise",
            parameters=[
                ParameterSpec("radius", "Radius (px)", 0.0, 8.0, 0.05, 0.0),
                ParameterSpec("strength", "Strength", 0.0, 1.0, 0.01, 0.0),
            ],
            apply_fn=apply_chroma_denoise,
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
    incoming = data.get("parameters", {})

    if block_type in {"brightness", "contrast"}:
        block_type = "brightness_contrast"
    elif block_type in {"midtone_transfer", "levels"}:
        block_type = "levels_midtone_transfer"
    elif block_type == "rl_deconvolution":
        block_type = "psd_deconvolution"

    block = instantiate_block(block_type, definitions)
    block.enabled = bool(data.get("enabled", True))
    if (
        block_type == "saturation"
        and "saturation" in incoming
        and "saturation_shadows" not in incoming
        and "saturation_midtones" not in incoming
        and "saturation_highlights" not in incoming
    ):
        legacy_value = float(incoming["saturation"])
        incoming = {
            **incoming,
            "saturation_shadows": legacy_value,
            "saturation_midtones": legacy_value,
            "saturation_highlights": legacy_value,
        }

    if block_type == "levels_midtone_transfer":
        # Backward compatibility with older pipelines that had 6 boundaries.
        incoming = {
            **incoming,
            "shadow_upper": incoming.get("shadow_upper", incoming.get("low_mid_lower", 0.25)),
            "low_mid_upper": incoming.get("low_mid_upper", incoming.get("high_mid_lower", 0.50)),
            "high_mid_upper": incoming.get("high_mid_upper", incoming.get("highlights_lower", 0.78)),
        }


    if data["type"] in {"brightness", "contrast"}:
        incoming = {
            **incoming,
            "brightness": float(incoming.get("brightness", 0.0)),
            "contrast": float(incoming.get("contrast", 1.0)),
        }

    if data["type"] == "rl_deconvolution":
        incoming = {
            **incoming,
            "psf_size": max(3.0, float(incoming.get("radius", 0.0)) * 2.0 + 1.0),
            "seeing_index": 1.2,
            "psf_strength": 0.3,
            "blend": 1.0,
        }

    for spec in definitions[block_type].parameters:
        if spec.key not in incoming:
            continue
        if spec.input_type == "choice":
            value = str(incoming[spec.key])
            block.parameters[spec.key] = value if value in spec.choices else spec.default
        else:
            block.parameters[spec.key] = float(incoming[spec.key])
    return block
