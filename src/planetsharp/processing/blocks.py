from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any


@dataclass
class BlockDefinition:
    block_type: str
    display_name: str
    defaults: dict[str, Any]


BLOCK_DEFINITIONS: dict[str, BlockDefinition] = {
    "DECON": BlockDefinition("DECON", "Deconvolution", {"algorithm": "RL", "psf_model": "Gaussian", "radius": 2.0, "sigma": 1.2, "beta": 2.0, "anisotropy": 0.0, "angle": 0.0, "iterations": 15, "damping": 0.05, "noise_floor": 0.01, "edge_protection": 0.25, "highlights_protect": 0.5, "black_clamp": 0.0, "edge_mask_amount": 0.0}),
    "AWAVE": BlockDefinition("AWAVE", "À trous wavelet sharpening", {"layers": [{"sharpen": 0.2, "denoise": 0.1, "threshold": 0.0} for _ in range(6)], "strength": 0.5, "linked_layers": False, "residual_boost": 0.0}),
    "RWAVE": BlockDefinition("RWAVE", "Wavelet sharpening", {"layers": [{"sharpen": 0.2, "denoise": 0.1, "bias": 0.0, "threshold": 0.0} for _ in range(6)], "dyadic_scaling": True, "gamma": 1.0, "linked_sliders": False, "luminance_only": False}),
    "UMASK": BlockDefinition("UMASK", "Unsharp mask", {"radius": 1.5, "amount": 0.6, "threshold": 0.01, "high_pass_mode": False}),
    "COBAL": BlockDefinition("COBAL", "Color balance", {"shadows": {"cr": 0.0, "mg": 0.0, "yb": 0.0}, "midtones": {"cr": 0.0, "mg": 0.0, "yb": 0.0}, "highlights": {"cr": 0.0, "mg": 0.0, "yb": 0.0}, "preserve_luminosity": True, "tonal_split": {"shadow_end": 0.33, "highlight_start": 0.66}}),
    "WHBAL": BlockDefinition("WHBAL", "White balance", {"temperature": 6500, "tint": 0.0, "illuminant": "D65", "gray_point": None}),
    "ALIGN": BlockDefinition("ALIGN", "Channel alignment", {"mode": "auto", "dx": {"r": 0.0, "g": 0.0, "b": 0.0}, "dy": {"r": 0.0, "g": 0.0, "b": 0.0}, "search_radius": 20, "reference_channel": "G", "rotation_correction": False}),
    "SELCO": BlockDefinition("SELCO", "Selective color", {"ranges": {name: {"c": 0.0, "m": 0.0, "y": 0.0, "k": 0.0} for name in ["Reds", "Yellows", "Greens", "Cyans", "Blues", "Magentas", "Whites", "Neutrals", "Blacks"]}, "softness": 0.2}),
    "DERIN": BlockDefinition("DERIN", "Deringing", {"edge_sensitivity": 0.5, "detection_radius": 2, "correction_strength": 0.5, "blend": 0.7, "detail_protection": 0.4, "bright_mode": True, "dark_mode": True}),
    "GBLUR": BlockDefinition("GBLUR", "Gaussian blur", {"sigma": 1.0, "radius": 3, "apply_l": True, "apply_r": True, "apply_g": True, "apply_b": True}),
    "BILAT": BlockDefinition("BILAT", "Bilateral filter", {"spatial_sigma": 2.0, "range_sigma": 0.1, "iterations": 1, "edge_preservation": 0.7}),
    "NOISE": BlockDefinition("NOISE", "Noise reduction", {"method": "nl_means", "strength": 0.5, "detail_preservation": 0.5, "luma_chroma_balance": 0.5, "grain_retention": 0.0}),
    "LINST": BlockDefinition("LINST", "Linear stretch", {"black_point": 0.0, "white_point": 1.0, "midtone_gamma": 1.0, "auto_stretch": False, "target_median": 0.25}),
    "LEVEL": BlockDefinition("LEVEL", "Levels", {"input_black": 0.0, "input_mid": 0.5, "input_white": 1.0, "output_black": 0.0, "output_white": 1.0, "per_channel": False, "channel_link": True}),
    "CURVE": BlockDefinition("CURVE", "Curves", {"composite_points": [(0.0, 0.0), (1.0, 1.0)], "per_channel_points": {"R": [], "G": [], "B": [], "L": []}, "interpolation": "spline"}),
    "CONTR": BlockDefinition("CONTR", "Contrast", {"global_contrast": 1.0, "midtone_contrast": 0.0, "highlight_protection": True}),
    "SATUR": BlockDefinition("SATUR", "Saturation", {"global_saturation": 1.0, "vibrance": 0.0, "luma_protection": True}),
}


def complete_block_params(block_type: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
    """Return params with all required defaults for a block type filled in."""
    if block_type not in BLOCK_DEFINITIONS:
        return copy.deepcopy(params or {})
    return _merge_defaults(BLOCK_DEFINITIONS[block_type].defaults, params or {})


def _merge_defaults(defaults: Any, params: Any) -> Any:
    if isinstance(defaults, dict):
        incoming = params if isinstance(params, dict) else {}
        merged: dict[str, Any] = {}
        for key, default_value in defaults.items():
            merged[key] = _merge_defaults(default_value, incoming.get(key))
        for key, value in incoming.items():
            if key not in merged:
                merged[key] = copy.deepcopy(value)
        return merged
    if isinstance(defaults, list):
        incoming = params if isinstance(params, list) else []
        merged_list: list[Any] = []
        for idx, default_item in enumerate(defaults):
            merged_list.append(_merge_defaults(default_item, incoming[idx] if idx < len(incoming) else None))
        if len(incoming) > len(defaults):
            merged_list.extend(copy.deepcopy(item) for item in incoming[len(defaults) :])
        return merged_list
    return copy.deepcopy(defaults if params is None else params)
