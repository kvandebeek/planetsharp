#!/usr/bin/env python3
from __future__ import annotations

import argparse
import cProfile
import pstats
import time

import numpy as np

from planetsharp.blocks import block_definitions, instantiate_block


def build_pipeline(definitions):
    chain = []
    order = [
        "levels_midtone_transfer",
        "saturation",
        "gaussian_blur",
        "unsharp_mask",
        "high_pass_detail",
        "wavelet_sharpening",
        "psd_deconvolution",
        "chroma_denoise",
    ]
    for t in order:
        block = instantiate_block(t, definitions)
        if t == "gaussian_blur":
            block.parameters.update({"size": 4.0, "strength": 0.9, "channel_mode": "All channels"})
        elif t == "unsharp_mask":
            block.parameters.update({"radius": 3.0, "amount": 1.5, "threshold": 0.01})
        elif t == "high_pass_detail":
            block.parameters.update({"radius": 2.0, "amount": 1.2, "softness": 0.4})
        elif t == "wavelet_sharpening":
            block.parameters.update(
                {
                    "layer1_amount": 1.2,
                    "layer2_amount": 0.8,
                    "layer3_amount": 0.4,
                    "layer1_noise_reduction": 0.1,
                    "layer2_noise_reduction": 0.2,
                    "layer3_noise_reduction": 0.3,
                }
            )
        elif t == "psd_deconvolution":
            block.parameters.update({"psf_size": 9.0, "seeing_index": 1.5, "psf_strength": 0.5, "blend": 1.0})
        elif t == "chroma_denoise":
            block.parameters.update({"radius": 3.0, "strength": 0.7})
        chain.append(block)
    return chain


def run_once(image, chain, definitions):
    out = image
    per_block = []
    for block in chain:
        fn = definitions[block.block_type].apply_fn
        start = time.perf_counter()
        out = fn(out, block.parameters)
        per_block.append((block.block_type, (time.perf_counter() - start) * 1000.0))
    return out, per_block


def main() -> None:
    parser = argparse.ArgumentParser(description="Profile PlanetSharp pipeline blocks.")
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--runs", type=int, default=10)
    args = parser.parse_args()

    definitions = block_definitions()
    chain = build_pipeline(definitions)
    image = np.random.rand(args.height, args.width, 3).astype(np.float32)

    for _ in range(args.warmup):
        run_once(image, chain, definitions)

    aggregate = {b.block_type: 0.0 for b in chain}
    start_total = time.perf_counter()

    profiler = cProfile.Profile()
    profiler.enable()
    for _ in range(args.runs):
        _, per_block = run_once(image, chain, definitions)
        for name, ms in per_block:
            aggregate[name] += ms
    profiler.disable()
    total_ms = (time.perf_counter() - start_total) * 1000.0

    print("=== Per-block average (ms) ===")
    for name, total in sorted(aggregate.items(), key=lambda kv: kv[1], reverse=True):
        print(f"{name:24s} {total / args.runs:8.3f}")

    print(f"\nTotal average frame time: {total_ms / args.runs:.3f} ms")
    print(f"Approx FPS: {1000.0 / max(1e-6, total_ms / args.runs):.2f}")

    print("\n=== cProfile hotspots ===")
    pstats.Stats(profiler).sort_stats("cumulative").print_stats(20)


if __name__ == "__main__":
    main()
