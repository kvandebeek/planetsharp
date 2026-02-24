from __future__ import annotations

from .models import FilterWeights


DEFAULT_FILTER_PRESETS: dict[str, dict[str, FilterWeights]] = {
    "CH4→B": {"CH4": FilterWeights(b=100)},
    "CH4→R": {"CH4": FilterWeights(r=100)},
    "IR→L": {"IR": FilterWeights(l=100)},
    "IR→RGB": {"IR": FilterWeights(r=33, g=33, b=33)},
    "UV→B": {"UV": FilterWeights(b=100)},
    "LP/IR→L": {"LP/IR": FilterWeights(l=100)},
    "LP/IR→R": {"LP/IR": FilterWeights(r=100)},
    "Ha→R": {"Ha": FilterWeights(r=100)},
    "OIII→G": {"OIII": FilterWeights(g=100)},
    "OIII→B": {"OIII": FilterWeights(b=100)},
    "SII→R": {"SII": FilterWeights(r=100)},
    "HOO": {"Ha": FilterWeights(r=100), "OIII": FilterWeights(g=50, b=50)},
    "SHO": {"SII": FilterWeights(r=100), "Ha": FilterWeights(g=100), "OIII": FilterWeights(b=100)},
}
