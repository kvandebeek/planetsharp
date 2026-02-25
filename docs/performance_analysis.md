# PlanetSharp Performance Analysis (Level 10 Deep Dive)

## 1) Performance model for this app

PlanetSharp is a **single-process, CPU-bound, memory-bandwidth-heavy** real-time pipeline:

1. UI event (`QSlider.valueChanged`) triggers `apply_pipeline()`.
2. `QTimer` schedules `_apply_pipeline_now()` for near-immediate execution.
3. Pipeline iterates each enabled block, transforming full-frame `float32` RGB arrays.
4. Result is clipped, converted for display, and histogram is recomputed.

At 1920×1080, each frame is ~6.2 million float channels. Most blocks make multiple full-frame passes, so many operations are memory throughput limited.

---

## 2) High-impact bottlenecks and why they matter

### A) Convolution core (`_convolve_axis`)
- Used by Gaussian blur, unsharp mask, high-pass detail, RL deconvolution, wavelet, chroma denoise.
- Previous approach built `sliding_window_view` then `tensordot`; this can create very large intermediate views and high allocator pressure.
- This is often the #1 hotspot when blur/deconvolution blocks are active.

### B) Repeated expensive block composition
- Several blocks convert RGB↔LAB and run multiple full-image passes.
- Wavelet and deconvolution can chain many convolution and FFT operations per frame.

### C) Per-frame non-pipeline overhead
- Original clip statistics were recomputed every frame even when source image is unchanged.
- Histogram painting did minor avoidable repeat work (max computed twice, extra copies).

### D) Real-time interaction behavior
- During drag, UI can trigger many frames quickly; if each frame is heavy, responsiveness degrades.

---

## 3) Instrumentation added (so you can measure every block)

A new rolling profiler was added:

- `planetsharp/perf.py` introduces `PerfMonitor`.
- Enable with `PLANETSHARP_PROFILE=1`.
- Reporting cadence controlled by `PLANETSHARP_PROFILE_EVERY` (default 20 frames).
- Logs avg/max timings for:
  - whole frame
  - per pipeline block (`block.<name>`)
  - display/histogram stages
  - clipping/stat stages

This makes bottlenecks observable in live runs, not just synthetic benchmarks.

---

## 4) Implemented optimizations

### 4.1 Optimized separable convolution implementation

`_convolve_axis` now uses reflect-padding + weighted shifted accumulation, avoiding a huge sliding-window tensor and reducing peak memory pressure.

Expected effect:
- Lower memory overhead
- Better cache behavior
- Lower latency jitter on larger images/kernels

### 4.2 Gaussian kernel cache

`_gaussian_kernel_1d` now caches kernels by `(odd_size, quantized_strength)`.

Expected effect:
- Avoid repeated kernel builds when users scrub sliders near the same values.

### 4.3 Avoid repeated original clip-stat recomputation

`original_clip_counts` is now computed once when an image opens and reused during frame updates.

Expected effect:
- Removes one full-image scan per frame.

### 4.4 Minor histogram paint-path cleanup

`HistogramWidget._paint_hist_section` now avoids duplicated `np.max` and unnecessary histogram copy.

Expected effect:
- Small but consistent GUI paint overhead reduction.

### 4.5 RGB/LAB matrix constant hoisting

Static matrices/white point are now module-level constants instead of being allocated repeatedly in conversion functions.

Expected effect:
- Small reduction in per-frame allocation churn.

---

## 5) How to measure in your environment

### Live app profiling

```bash
PLANETSHARP_PROFILE=1 PLANETSHARP_PROFILE_EVERY=10 python3 main.py
```

Look for log lines from `planetsharp.perf` and sort by highest avg ms.

### Synthetic repeatable benchmark

```bash
python3 scripts/profile_pipeline.py --width 1920 --height 1080 --runs 15
```

This prints per-block averages + cProfile hotspots.

---

## 6) Practical optimization roadmap (next steps)

### Priority 1 (high confidence, high impact)
1. **Incremental pipeline recompute**:
   - Cache outputs after each block.
   - Recompute only from the first changed block onward.
2. **Histogram throttling**:
   - Recompute histogram at e.g. 10–20 Hz during drag, final full update on release.
3. **Adaptive preview resolution**:
   - While dragging sliders, process downscaled proxy; compute full-res after short idle.

### Priority 2 (compute acceleration)
4. **FFT-vs-spatial switch**:
   - Use spatial convolution for tiny kernels, FFT for large kernels.
5. **Optional multithreading/chunking**:
   - Run expensive blocks in a worker thread with cancellation/versioning.

### Priority 3 (low-level/native acceleration)
6. **Native module for hot kernels**:
   - Implement convolution and color conversion in Rust/C++ (PyO3 or pybind11).
   - Keep Python orchestration, move math kernels native.

When this becomes necessary:
- Trigger condition: profiling shows >60–70% in convolution/color conversion and target FPS cannot be reached with algorithmic changes alone.

---

## 7) Suggested "level-10" production architecture target

- Real-time preview pipeline (proxy resolution, aggressive throttling, cancellation).
- Full-quality commit pipeline (full-res, exact output) on idle or save.
- Per-block perf telemetry persisted for session-level analysis.
- Native accelerated math backend behind a stable Python API.
- Optional GPU path (future) if dependency footprint is acceptable.

This gives both interactivity and output quality without forcing all operations to run full-cost on every slider tick.
