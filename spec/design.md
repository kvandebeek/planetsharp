# PlanetSharp v1 Product Requirements and Delivery Backlog

## 1) Product scope

### Goal
Build a **desktop Python application** for planetary and lunar/surface astrophotography enhancement with:
- A visual, reorderable workflow editor.
- Two-stage processing (per-channel then combined).
- Real-time preview with responsive UI.

### Non-goals (explicitly out of scope for v1)
- Deep-sky calibration/stacking (bias/dark/flat), drizzle, frame registration/derotation, WinJUPOS-like operations.
- Video capture / SER ingest pipeline.
- AI upscaling, generative edits, content-aware synthesis.

---

## 2) Supported files, bit depths, and image data model

### Import (must)
- Formats: `PNG`, `BMP`, `TIF/TIFF`, `JPG/JPEG`, `XISF`, `FITS`.
- Bit depths: 8 / 16 / 32-bit.
- Channels: grayscale or RGB color.
- Multi-file import per session.
- Per-file import interpretation: **Assume sRGB** (default for nonlinear formats like JPEG) or **Assume linear**.

### Export (must)
- Formats: `PNG`, `BMP`, `TIF/TIFF`, `JPG/JPEG`, `XISF`, `FITS`.
- Bit depths: 8 / 16 / 32-bit.
- Preserve metadata where feasible (especially FITS header and XISF properties).

### Internal processing representation (must)
- Canonical working format: **float32 linear** per channel.
- Working range is not clamped by default.
- No silent bit-depth truncation.

### Acceptance criteria
- All supported formats load and display without crash.
- 16-bit/32-bit sources remain high precision internally.
- Export honors selected format and bit depth.
- Saved output visually matches preview within expected codec limitations (e.g., JPEG).

---

## 3) Session/project model

A session stores:
- Input image list.
- Per-input role assignment: `L | R | G | B | FILTER`.
- `filter_name` (preset or custom text).
- Filter channel mapping and weight settings.
- Stage 1 workflows (per channel) and Stage 2 combined workflow.
- Mixer settings.
- ROI settings.
- Performance settings (threads, memory limit).
- UI state (zoom/pan, view mode, channel visibility).

### Persistence
- Save/load session as JSON (example extension: `.planetflow.json`).
- Store absolute and/or relative paths robustly.
- Reloading a session should reproduce output within tolerance.

---

## 4) Filter mapping defaults (editable)

### Requirement
Users can:
- Assign files to standard channels (`L/R/G/B`) or `FILTER`.
- Apply a preset mapping for FILTER inputs.
- Edit channel contribution weights manually.
- Save custom mappings as reusable presets.

### Default presets (starting points)
- `CH4→B`: B=100%
- `CH4→R`: R=100%
- `IR→L`: L=100%
- `IR→RGB`: R=33%, G=33%, B=33%
- `UV→B`: B=100%
- `LP/IR→L`: L=100%
- `LP/IR→R`: R=100%
- `Ha→R`: R=100%
- `OIII→G`: G=100%
- `OIII→B`: B=100%
- `SII→R`: R=100%
- `HOO`: Ha→R, OIII→G+B (50/50)
- `SHO`: SII→R, Ha→G, OIII→B

### Weight handling
- Weight range: 0–100 per target channel.
- Provide **Normalize Weights** toggle:
  - On: normalize per-filter contributions to sum=1 before combine.
  - Off: absolute contributions, then scale/clamp at combine step.

### Acceptance criteria
- Preset selection updates mapping immediately and triggers re-render.
- Manual edits trigger immediate re-render.
- Custom presets can be created and reused.

---

## 5) Two-stage workflow architecture

### Stage 1 (must): per-channel workflows
- Independent ordered pipelines for `L`, `R`, `G`, `B`.
- FILTER contributions are routed into channels before or during Stage 1 combine (implementation-defined, documented).
- Each block instance has:
  - `id` (unique)
  - `type`
  - `enabled`
  - `params` (typed, serializable)

### Stage 2 (must): combined workflow
- Single ordered pipeline applied to combined Stage 1 output (RGB or LRGB).

### Mixer between stages (must)
- Minimum: `Stage2 Mix Amount` (0–100).
- Recommended: `Mix_RGB` and `Mix_L` split controls.
- Formula (documented):
  - `final = (1 - m) * stage1 + m * stage2` (per channel).

### Acceptance criteria
- Reorder, enable/disable, and param edits update preview (debounced).
- Duplicate block instances are supported with independent params.
- In-flight renders are cancellable; latest edit wins.

---

## 6) Workflow editor and UI requirements

### Required panels
- Building Blocks Library.
- Stage 1 editor (tabs/columns for L/R/G/B).
- Stage 2 editor (combined LRGB/RGB).
- Block Inspector (parameters).
- Always-visible Viewer.
- ROI + Performance panel.
- Progress/status area.

### Drag-and-drop behavior (must)
- Library → pipeline insertion at drop point.
- Intra-pipeline drag reorder.
- Remove via context menu and/or trash target.
- Multiple instances of same block type allowed.

### Parameter editing behavior
- Selecting block opens parameter inspector.
- Parameter changes trigger debounced render.
- Provide `Reset Params` (per block) and `Reset Workflow` (per channel/stage).

### Acceptance criteria
- UI remains responsive while rendering.
- Viewer stays interactive (zoom/pan) during long operations.

---

## 7) Viewer requirements

### Navigation
- Zoom in/out.
- Mouse/touchpad pan.
- Navigator/minimap with viewport rectangle.

### Channel visibility toggles
- `L`, `R`, `G`, `B`, `LRGB`.
- Rules:
  - If `LRGB` enabled: show full composite.
  - Single channel only: grayscale display.
  - Multiple RGB without LRGB: show selected composite subset.

### Optional stage display selector
- `Original`
- `After Stage 1`
- `After Stage 2`
- `Final Mixed`

### Acceptance criteria
- Toggle changes update view immediately.
- Stage switching leverages cached intermediates when possible.

---

## 8) ROI, performance, and progress

### ROI (must)
- Create/move/resize/clear ROI in viewer.
- Modes:
  - ROI preview (default when ROI exists).
  - Full-frame processing (toggle override).

### Performance controls (must)
- Threads control: clamp `[1..logical_cores]`.
- Memory limit control: percentage budget used to bound caches and intermediate buffers.
- Allow tiled processing fallback for large images.

### Progress and cancellation (must)
- Render runs off UI thread.
- Progress area shows stage + current block.
- Parameter edits cancel active render and restart with latest state.

### Acceptance criteria
- ROI meaningfully reduces preview latency for expensive blocks.
- No crashes on edge/tiny ROI.
- App remains responsive under heavy operations.

---

## 9) Error handling and validation

- Unsupported type: clear non-blocking error message; no crash.
- Missing channel inputs: allow processing, treat missing channels as **zero-filled** (documented behavior).
- Dimension mismatch between inputs: for v1, provide **resample to reference image** prompt on import.
- Invalid parameter values: reject with inline validation and preserve last valid value.

---

## 10) Building blocks and pro-grade parameter sets (v1)

## Global block contract
Every block must provide:
- `enabled` toggle.
- Deterministic processing (same input + params => same output within tolerance).
- Typed serializable params.
- Numeric fields with safe ranges + slider + direct numeric entry.
- Reset-to-default control.

## Required blocks and minimum parameter depth

### DECON — Deconvolution
- Algorithm: RL (required), optional Wiener/Regularized RL.
- PSF model: Gaussian, Moffat, custom kernel.
- PSF controls: radius, sigma, beta, anisotropy, angle.
- Iterations, damping, noise floor, edge protection.
- Ringing controls: highlights protect, black clamp.
- Optional edge mask amount.

### AWAVE — À trous wavelet sharpening
- 6+ layers.
- Per layer: sharpen, denoise, threshold.
- Global: strength, linked-layers toggle.
- Optional residual boost.

### RWAVE — RegiStax-style wavelets
- 6+ layers.
- Per layer: sharpen, denoise, bias, threshold.
- Global: dyadic scaling, gamma, linked sliders.
- Optional luminance-only mode.

### UMASK — Unsharp mask
- Radius, amount, threshold.
- Optional high-pass mode.

### COBAL — Color balance
- Shadows/midtones/highlights wheels/sliders:
  - Cyan↔Red, Magenta↔Green, Yellow↔Blue.
- Preserve luminosity toggle.
- Tonal range split controls.

### WHBAL — White balance
- Temperature, tint.
- Optional gray-point picker.
- Illuminant presets.

### ALIGN — RGB align
- Auto (phase correlation) or manual mode.
- Manual subpixel `dx/dy` per channel.
- Auto settings: search radius, reference channel.
- Optional rotational correction.

### SELCO — Selective color
- Ranges: Reds, Yellows, Greens, Cyans, Blues, Magentas, Whites, Neutrals, Blacks.
- Per-range CMYK (or equivalent RGB formulation).
- Mask softness/overlap.

### DERIN — Deringing
- Edge sensitivity, detection radius.
- Correction strength, blend, detail protection.
- Optional bright/dark ring modes.

### GBLUR — Gaussian blur
- Sigma.
- Radius/kernel size.
- Per-channel apply toggles in LRGB context.

### BILAT — Bilateral filter
- Spatial sigma, range sigma, iterations.
- Edge preservation strength.

### NOISE — Denoise
- Include at least one method (recommended two): NL-means, wavelet denoise, guided filter.
- Strength, detail preservation, luma/chroma balance.
- Optional grain retention.

### LINST — Linear stretch
- Black point, white point, midtone gamma.
- Optional auto-stretch with target median.

### LEVEL — Levels
- Input black/mid/white.
- Output black/white.
- Per-channel mode + channel-link toggle.

### CURVE — Curves
- Interactive control points.
- Composite curve + optional per-channel curves (R/G/B/L).
- Interpolation mode (spline/linear).

### CONTR — Contrast
- Global contrast.
- Optional local/midtone contrast (clarity).
- Highlight protection toggle.

### SATUR — Saturation
- Global saturation.
- Vibrance.
- Optional luma protection.

### Acceptance criteria (blocks)
- Every block has more than a single amount slider.
- Default values are sane for planetary workflows.
- Param changes visibly affect output and persist via session save/load.

---

## 11) Implementation blueprint (deliverable A)

### Module breakdown
- `io/`
  - Format readers/writers (PNG/BMP/TIFF/JPEG/FITS/XISF).
  - Metadata adapters and import color-space interpretation.
- `core/`
  - Data models, typed schemas, validation.
  - Workflow graph execution, block registry.
- `processing/`
  - Block implementations, channel routing, mixer, color transforms.
  - Caching, tiling, thread pool integration.
- `render/`
  - Render scheduler, cancellation tokens, latest-wins orchestration.
- `ui/`
  - Workflow editor, inspector, viewer, ROI tools, progress and settings panels.
- `persistence/`
  - Session JSON serialize/deserialize with path resolution.
- `tests/`
  - Unit, integration, deterministic image regression checks.

### Typed data model
- `Session`
- `InputImage`
- `FilterMapping`
- `Workflow`
- `BlockInstance`
- `ROI`
- `PerformanceSettings`
- `ViewerState`
- `MixerSettings`

---

## 12) Acceptance test suite (deliverable B)

Minimum high-level tests:
1. Load at least one sample for each supported format.
2. Verify internal precision path for 16/32-bit inputs.
3. Apply each block with controlled param change and verify output delta (hash/SSIM/PSNR threshold).
4. Reorder blocks and assert output changes.
5. Disable block and assert output reverts to prior graph state.
6. ROI render matches full-frame render inside ROI bounds within tolerance.
7. Save and reload session and assert reproducible output.
8. Verify cancellation: rapid parameter changes produce final image for latest state only.
9. Validate filter preset application and custom preset persistence.
10. Verify memory cap behavior under large image workloads.

---

# Epics → User Stories → Acceptance Criteria

## Epic E1 — Image I/O and precision-safe data model

### US1: Import supported formats and bit depths
**Story:** As a user, I can import PNG/BMP/TIFF/JPEG/XISF/FITS in 8/16/32-bit so I can process my data without conversion loss.

**Acceptance criteria:**
- Each format imports successfully with correct dimensions/channels.
- Effective source bit depth is detected and preserved internally via float32 conversion.
- Import interpretation toggle (sRGB vs linear) is available per file.

### US2: Export supported formats and bit depths
**Story:** As a user, I can export to all supported formats in chosen bit depth.

**Acceptance criteria:**
- Exports open correctly in external software.
- Selected bit depth is honored.
- Metadata pass-through occurs where supported.

### US3: Session persistence
**Story:** As a user, I can save and reload complete sessions.

**Acceptance criteria:**
- Input list, roles, filter mappings, workflows, params, mixer, ROI, perf settings, and viewer state restore.
- Reopened session reproduces output within numeric tolerance.

## Epic E2 — Channel assignment and filter mapping

### US4: Assign L/R/G/B/FILTER roles
**Acceptance criteria:**
- Role assignment UI exists per input.
- Missing channels are zero-filled and documented.

### US5: Editable FILTER mappings
**Acceptance criteria:**
- Weights support 0–100 values with validation.
- Multiple output targets per filter are supported.
- Normalize weights option is available.

### US6: Presets and custom presets
**Acceptance criteria:**
- Built-in presets (CH4/IR/UV/Ha/OIII/SII/HOO/SHO) apply immediately.
- Users can save, rename, and reuse custom presets.

## Epic E3 — Two-stage workflow engine

### US7: Stage 1 per-channel workflows
**Acceptance criteria:**
- Independent block lists for L/R/G/B.
- Stage 1 combine produces RGB/LRGB according to available channels.

### US8: Stage 2 combined workflow
**Acceptance criteria:**
- Stage 2 processes Stage 1 output.
- Any compatible block type may be inserted.

### US9: Mixer controls
**Acceptance criteria:**
- At least one stage mix slider exists.
- Mix behavior matches documented equation.

### US10: Latest-wins rendering
**Acceptance criteria:**
- In-progress render cancels on new edits.
- Final preview reflects most recent settings.

## Epic E4 — Workflow editor UX

### US11: Add blocks by drag/drop
**Acceptance criteria:**
- Library-to-pipeline insertion works at target index.
- Multiple instances per type supported.

### US12: Reorder blocks
**Acceptance criteria:**
- Drag reorder updates pipeline order and output.
- Order persists in saved session.

### US13: Remove blocks
**Acceptance criteria:**
- Remove action available and discoverable.
- Output updates after removal.

### US14: Enable/disable blocks
**Acceptance criteria:**
- Toggle bypass is per instance.
- Disabled blocks are true no-ops in execution.

### US15: Parameter inspector controls
**Acceptance criteria:**
- Inspector shows full typed params.
- Edits are debounced and persisted.
- Reset params/workflow controls are available.

## Epic E5 — Real-time viewer and navigation

### US16: Zoom/pan viewer
**Acceptance criteria:**
- Smooth zoom/pan interactions.
- Viewer remains responsive during rendering.

### US17: Navigator/minimap
**Acceptance criteria:**
- Minimap displays viewport rectangle.
- Minimap interactions reposition main viewport.

### US18: Channel/stage display modes
**Acceptance criteria:**
- L/R/G/B/LRGB toggles work per specification.
- Optional stage dropdown switches display without unnecessary recomputation.

## Epic E6 — ROI acceleration

### US19: ROI authoring
**Acceptance criteria:**
- ROI create/move/resize/clear tools function correctly.
- Bounds are clamped safely within image.

### US20: ROI/full-frame mode switch
**Acceptance criteria:**
- Mode toggle is clear and immediate.
- ROI mode is measurably faster on heavy blocks.

## Epic E7 — Performance, memory, progress

### US21: Thread control
**Acceptance criteria:**
- Threads clamp to valid range.
- Changes apply without app restart.

### US22: Memory budget control
**Acceptance criteria:**
- Cache/intermediate allocations honor memory budget.
- Graceful tiled fallback for large workloads.

### US23: Progress and status
**Acceptance criteria:**
- UI shows current stage and active block.
- Progress updates while rendering and clears on completion/cancel.

## Epic E8 — High-quality block implementations

### US24: Pro-grade parameter depth for all required blocks
**Acceptance criteria:**
- All required block types implemented.
- Parameter sets meet or exceed section 10.
- Input validation prevents invalid values.

### US25: Deterministic reproducibility
**Acceptance criteria:**
- Same input + params yields stable output within float tolerance.
- Reloading session reproduces output consistently.
