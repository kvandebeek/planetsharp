1) Product scope
Goal

A desktop Python application for planetary and surface astrophotography image enhancement, with a visual, reorderable workflow and real-time preview.

Non-goals (explicitly out of scope for v1)

Deep-sky stacking/calibration (bias/dark/flat), drizzle, registration/derotation, WinJUPOS-like operations.

Video capture / SER handling (unless explicitly added later).

AI upscaling / generative edits.

2) Supported files and image data requirements
Import formats (must)

Read: PNG, BMP, TIF, TIFF, JPG/JPEG, XISF, FITS

Bit depth: 8-bit, 16-bit, 32-bit

Channels: grayscale or color

Multiple files per session.

Export formats (must)

Write: PNG, BMP, TIF/TIFF, JPG/JPEG, XISF, FITS

Bit depth: 8-bit, 16-bit, 32-bit

Internal processing representation (requirement)

Convert all sources into a canonical internal format:

float32 linear working space (per channel), range not clamped by default.

Preserve metadata where possible (FITS header, XISF properties) and pass-through on save when feasible.

Acceptance criteria

Import and display each supported format without crashing.

16-bit and 32-bit sources must not be truncated to 8-bit internally.

Exported file matches selected bit depth and format; preview visually matches saved result (within expected format limitations like JPEG compression).

3) Project/session model
Session

A “session” contains:

A list of inputs (images) each tagged with:

role: L | R | G | B | FILTER

filter_name (string, e.g., "CH4 889", "IR685", "UV", "Ha 656", "OIII 500", "Custom")

channel_mapping (how this input contributes to output; details below)

A two-stage workflow (stage-1 per-channel workflows, stage-2 combined workflow)

Mixer settings

ROI settings

Performance settings (threads, memory limit)

UI state (zoom/pan, visible channels)

Persistence (recommended for Codex)

Save/load session to a JSON file (e.g., .planetflow.json) referencing file paths + relative paths.

4) Filter mapping defaults (must be editable)
Requirement

User can:

Assign imported files to standard roles (L/R/G/B) or “special filters”.

For special filters: select from presets or define custom.

Map each special-filter image to one or more RGB channels with percent weights.

Defaults are prefilled, but always user-editable.

Presets and default mappings (v1)

Use these defaults (editable). Where conventions vary, provide more than one preset option.

A) Planetary common practice (recommended defaults)

Methane (CH4 889 nm): often used as a cloud-top/high-altitude indicator; map to Blue by default (false-color emphasis). NASA/Jupiter publications also show composites using CH4 in Red in some cases, so include an alternate preset.

Preset “CH4→B”: B=100%

Preset “CH4→R”: R=100%

IR-pass (e.g., IR685, IR742, IR807, IR850, IR1000): typically used as luminance/detail; default contribution to L (or to RGB equally if user wants). Product descriptions commonly position IR685 as luminance/sharpness aid for planetary.

Preset “IR→L”: L=100%

Preset “IR→RGB”: R=33%, G=33%, B=33%

UV (Venus/upper haze work; “UV”, ~350 nm): emphasize in Blue (or Blue+Green). (Provide as a preset, user editable.)

Preset “UV→B”: B=100%

“Mercury filter” is not a single wavelength standard; in amateur planetary contexts it often refers to long-pass/IR-leaning filters used for Mercury imaging/contrast (example discussions mention 610LP). Treat as “Long-pass/IR” and default to L or R.

Preset “LP/IR→L”: L=100%

Preset “LP/IR→R”: R=100%

B) Narrowband (include because users asked: Ha/OIII)

There is no single “correct” mapping; even mainstream tutorials note the channel assignment is largely a convention and can be arbitrary; still provide common presets.

H-alpha (Ha 656 nm): commonly blended into Red when combining with RGB.

Preset “Ha→R”: R=100%

OIII (500.7 nm):

Preset “OIII→G” or “OIII→B” (offer both)

SII (672 nm):

Preset “SII→R” (common in SHO/Hubble variants)

Also include combined palettes (useful as quick options):

HOO: Ha→R, OIII→G+B (split 50/50)

SHO (Hubble): SII→R, Ha→G, OIII→B
(Explain in UI as “preset starting points”.)

Acceptance criteria

User can change weights; weights clamp 0–100; sum does not need to be 100 (app normalizes or uses absolute mixing—pick one and document it).

Switching presets updates weights instantly and triggers re-render.

5) Two-stage workflow architecture (core feature)
Stage 1: per-channel workflows (must)

Independent pipeline per input role:

L, R, G, B, plus any FILTER inputs that are mapped into RGB/L via weights.

Each channel has a list of building blocks (ordered).

Each block has:

id (unique instance ID)

type (one of the listed block types)

enabled (bool)

params (typed; varies per block)

Output of Stage 1 is a combined LRGB (or RGB if no L).

Stage 2: combined workflow (must)

Pipeline applied to the combined output from Stage 1.

Same list behavior (reorder, add multiple instances, remove).

Mixer between stages (must)

A “mixer” blends:

stage1_output vs stage2_output contribution, with per-channel mixing controls.

Define mixer requirements precisely:

Minimum: one slider Mix Stage2 Amount (0–100%).

Recommended: per-channel sliders:

Mix_L, Mix_R, Mix_G, Mix_B, and/or Mix_RGB + Mix_L.

Document the math (example):

final = (1-m)*stage1 + m*stage2, per channel.

Acceptance criteria

Reordering any block updates preview (with debounced refresh).

Disabling a block updates preview without removing it.

Adding duplicate block instances works and maintains independent parameters.

6) Workflow editor UI requirements
Windows/panels (must)

Workflow Editor

Left: “Building Blocks Library”

Middle: Stage 1 (tabs or columns: L/R/G/B/Filters)

Right: Stage 2 (LRGB)

Always-visible Image Viewer

Real-time render output

Zoom/pan + navigator

Channel visibility toggles

Performance/ROI panel

ROI selection tools

Threads and memory controls

Progress indicator area

Drag & drop behavior (must)

Drag from library → insert into a pipeline at drop position.

Drag within a pipeline → reorder.

Drag block to trash/remove area (or context menu Remove).

Blocks can be inserted multiple times.

Block selection & parameters (must)

Clicking a block shows its parameter UI (right-side inspector).

Parameters changes re-render in real time (debounced).

Provide “Reset params” per block and “Reset workflow” per channel.

7) Image viewer requirements
Navigation (must)

Zoom in/out

Pan via mouse drag

“Navigator/minimap” similar to Photoshop (overview rectangle). (Implementation detail is up to Codex, but behavior must match.)

Channel visibility toggles (must)

Checkboxes:

LUMINANCE, R, G, B, LRGB
Behavior requirement:

If LRGB checked: show full composite.

If a single channel checked: show that channel as grayscale.

If multiple RGB checked without LRGB: show only those channels combined.

Viewer must update immediately.

Display pipeline selection (recommended)

Optional dropdown:

View “Original”, “After Stage 1”, “After Stage 2”, “Final Mixed”

8) Building blocks (functional requirements)
Required block types (must implement)

deconvolution (DECON)

atrous wavelet sharpening (AWAVE)

registax wavelet sharpening (RWAVE)

unsharp masking (UMASK)

color balance (COBAL)

white balance (WHBAL)

RGB align (ALIGN)

selective color (SELCO)

deringing (DERIN)

gaussian blurring (GBLUR)

bilateral filter (BILAT)

denoise (NOISE)

linear stretch (LINST)

levels (LEVEL)

curves (CURVE)

contrast (CONTR)

saturation (SATUR)

Block interface contract (must)

Every block implements:

input: image (single channel or RGB/LRGB depending on location)

output: same dimensionality as input

params: typed, serializable

enabled: bypass when false

Must be deterministic (same input+params → same output)

Parameter requirements (v1 minimum)

Define at least:

DECON: method (Richardson-Lucy), iterations, PSF radius/sigma, damping/regularization.

AWAVE/RWAVE: layer sliders (at least 5), denoise per layer, overall strength.

UMASK: radius, amount, threshold.

ALIGN: subpixel shift per channel (auto mode + manual override).

DERIN: strength, edge width.

NOISE: strength, detail preservation.

LEVEL/CURVE: black/mid/white points; curve control points (limited set).
(Other blocks similarly; Codex should propose exact param sets.)

Acceptance criteria

Each block visibly affects output when parameters changed.

Disabling block returns to previous state without lag beyond the re-render.

9) ROI (Region of Interest)
Requirement (must)

User can define ROI rectangle on the viewer.

Processing can run in:

ROI preview mode (fast, only ROI processed)

Full-frame mode (entire image)

Behavior

ROI preview mode is default when ROI is active.

A toggle: Process full image overrides ROI.

ROI changes trigger re-render.

Acceptance criteria

ROI substantially reduces processing time for heavy blocks (decon/wavelets).

No crashes when ROI is near edges or tiny.

10) Performance and progress
Threads/CPU control (must)

UI: numeric value with +/- buttons.

Range:

min 1

max = detected logical cores (or user settable, but clamp safely)

Memory limit (must)

UI: percentage with +/- buttons.

App must respect it by:

limiting cache sizes

avoiding unbounded intermediate buffers

optionally falling back to tiled processing for large images

Progress indicator (must)

When applying changes (i.e., re-rendering):

show progress bar/spinner

show current stage and block name

Real-time preview updates should not freeze the UI:

Render runs off the UI thread

Cancel/restart rendering when parameters change again (latest-wins)

Acceptance criteria

UI remains responsive during heavy processing.

Changing a parameter mid-render cancels the previous render and starts a new one.

11) Error handling and validation (must)

Unsupported file type: show clear message and do nothing.

Missing channel inputs (e.g., only R and G loaded):

allow operation; treat missing channels as zeros or duplicates (document chosen strategy).

Mismatched dimensions between channels:

either block import with a clear error OR offer “resample to match” (pick one for v1).

Nonlinear/gamma sources (JPEG):

treat as sRGB by default; allow “assume linear” toggle in import settings.

12) Concrete deliverables for Codex (what to build)
Deliverable A: Requirements-to-implementation plan

Codex outputs:

A module breakdown:

IO layer (FITS/XISF/TIFF/PNG/JPG)

Processing engine (workflow execution, caching, threading, cancellation)

UI layer (workflow editor + viewer)

Session persistence

A typed data model for:

Session, InputImage, FilterMapping, Workflow, BlockInstance, ROI, PerformanceSettings

Deliverable B: Acceptance tests (high-level)

Minimum test list:

Load each format (at least one sample per type).

Apply each block with a known parameter change and verify output differs (hash/metrics).

Reorder blocks changes output.

Disable block restores expected output.

ROI processing matches full processing within ROI area.

Save and reload session reproduces same output.

If you want this converted into a Codex-ready “task list” (epics → user stories → acceptance criteria per story), I can rewrite it in that format next.

also make sure that each building block has really good customizable parameters as found in common editing software

please create the above again with updates
also create epics, user stories and acceptance criteria for each
Updated requirements (v1) — with “pro-grade” building block parameters
1) Product scope

Goal: Desktop Python application for planetary and surface image enhancement with a visual, reorderable, two-stage workflow and real-time preview.

Non-goals (v1): deep-sky calibration/stacking, derotation/registration of video stacks, SER capture, AI/generative edits.

2) Supported file types, bit depths, and image modes
Import (must)

Formats: PNG, BMP, TIF/TIFF, JPG/JPEG, XISF, FITS

Bit depth: 8 / 16 / 32-bit

Image type: grayscale or color (RGB)

Multi-file import per session.

Export (must)

Formats: PNG, BMP, TIF/TIFF, JPG/JPEG, XISF, FITS

Bit depth: 8 / 16 / 32-bit

Internal representation (must)

Convert all inputs to float32 linear working space for processing.

Preserve metadata when possible (FITS headers, XISF properties) and carry through on export where feasible.

Allow a per-file import option: Assume sRGB (gamma) vs Assume linear.

Acceptance criteria

No silent down-conversion to 8-bit.

Export matches chosen format + bit depth.

Session reload reproduces the same rendered result.

3) Inputs: LRGB + special filters with editable mapping
Channels/roles (must)

User can assign each imported file a role: L, R, G, B, or FILTER.

FILTER inputs have:

filter_name (preset list + custom string)

mapping to output channels with adjustable weights (R/G/B and optionally L)

Filter presets & default mappings (must, editable)

Provide starting presets (user editable). Because conventions vary, include multiple presets where common.

CH4 889 (methane): presets CH4→B and CH4→R

IR-pass (e.g., 685/742/807/850): presets IR→L and IR→RGB (equal)

UV: preset UV→B

Ha: preset Ha→R

OIII: presets OIII→G and OIII→B

SII: preset SII→R

HOO palette: Ha→R, OIII→G+B (50/50)

SHO palette: SII→R, Ha→G, OIII→B

Weight handling (must)

Weights are 0–100% each.

Define behavior explicitly (choose one):

Normalize weights per filter so sum=1, or

Absolute mixing then clamp/scale at combine step.

Provide an option: Normalize weights (on/off).

Acceptance criteria

Switching a preset updates weights immediately and triggers re-render.

Custom filters can be named, saved as a preset, and reused.

4) Two-stage workflow (core)
Stage 1 (must): per-channel workflows

Separate pipelines for L, R, G, B (and FILTER contributions routed into these).

Each channel pipeline is an ordered list of building block instances.

Output: combined RGB or LRGB (if L exists).

Stage 2 (must): combined workflow

One ordered pipeline applied to the combined output from Stage 1.

Mixer (must)

Blends Stage 1 output with Stage 2 output.

Minimum: one slider Stage2 mix (0–100%).

Recommended: separate Mix_L and Mix_RGB.

Acceptance criteria

Drag reorder / enable-disable / parameter change updates preview (debounced).

Latest change cancels in-progress render (“latest-wins”).

5) UI requirements
Windows/panels (must)

Workflow Editor

Building Blocks Library

Stage 1 pipelines (L/R/G/B tabs or columns)

Stage 2 pipeline (LRGB)

Block Inspector (parameters)

Always-visible Viewer

Real-time render

Zoom/pan and minimap “navigator”

Channel toggles: L, R, G, B, LRGB

ROI/Performance panel

ROI controls

Threads + memory controls

Progress indicator

Drag & drop (must)

Library → pipeline inserts at drop position.

Reorder within pipeline via drag.

Remove via context menu or trash area.

Same block type can be added multiple times.

Acceptance criteria

Viewer never blocks (UI stays responsive).

Undo/redo is recommended; if implemented, must cover: add/remove/reorder/enable/param changes.

6) ROI (Region of Interest)

ROI rectangle selectable in viewer.

Modes:

ROI preview (process only ROI)

Full frame

ROI mode default-on when ROI exists, but toggleable.

Acceptance criteria

ROI changes re-render and significantly reduce render time for heavy blocks.

7) Performance controls & progress

Threads: +/- control, clamp to [1 .. logical_cores].

Memory limit: +/- percentage, used to bound caches and optionally trigger tiled processing.

Progress indicator shows current stage + block name.

Rendering must run off UI thread and be cancellable.

Acceptance criteria

Changing params mid-render cancels and restarts quickly.

No crashes on large images within memory limit.

8) Building blocks: “common editing software” parameter depth

Global requirements for every block

enabled toggle (bypass)

strength (where relevant)

Precise control with numeric input + slider where appropriate

Optional: preview quality (fast/accurate) for expensive blocks

Parameter serialization for session save/load

Below are minimum pro-grade parameter sets expected in v1.

DECON — Deconvolution

Algorithm: Richardson-Lucy | Wiener (optional) | Regularized RL (optional)

PSF model: Gaussian | Moffat | Custom kernel import

PSF controls: radius, sigma, beta (Moffat), anisotropy, angle

Iterations: 1..N

Damping/regularization: damping, noise floor, edge protection

Clamp/restore: prevent ringing, highlights protect, black clamp

Masking (optional v1): edge mask amount

AWAVE — À trous wavelet sharpening

Layers: at least 6 layers

Per-layer: sharpen amount, denoise, threshold

Global: overall strength, linked layers on/off

Optional: residual layer boost

RWAVE — “RegiStax-like” wavelets

Layers: at least 6

Per-layer: sharpen, denoise, bias, threshold

Global: dyadic scaling, gamma, linked sliders

Optional: use luminance only toggle when applied to RGB/LRGB

UMASK — Unsharp mask

radius

amount

threshold

Optional: high-pass mode toggle

COBAL — Color balance

Separate shadows/midtones/highlights adjustments:

Cyan–Red, Magenta–Green, Yellow–Blue

Preserve luminosity toggle

Range sliders defining shadow/highlight split

WHBAL — White balance

Temperature, tint

Gray-point picker (optional)

Target illuminant presets (Daylight, Tungsten, etc.) as convenience

ALIGN — RGB align

Mode: Auto (phase correlation) | Manual

Manual per-channel: dx, dy subpixel shifts

Auto parameters: search radius, reference channel

Optional: rotation correction (v1 optional)

SELCO — Selective color

Select color ranges: Reds/Yellows/Greens/Cyans/Blues/Magentas/Whites/Neutrals/Blacks

For each range: C/M/Y/K adjustments (or simplified RGB)

Mask softness / overlap control

DERIN — Deringing

Ring detection: edge sensitivity, radius

Correction: strength, blend, protect details

Optional: bright ring vs dark ring toggles

GBLUR — Gaussian blur

sigma

radius / kernel size

Apply per-channel toggle (when in LRGB stage)

BILAT — Bilateral filter

spatial sigma

range sigma

iterations

Edge preservation strength

NOISE — Denoise

Method: NL-means | wavelet denoise | guided filter (pick at least one for v1; ideally 2)

Key params: strength, detail preservation, luma/chroma balance

Optional: grain control

LINST — Linear stretch

Black point, white point

Midtones (gamma)

Optional: auto stretch with configurable target median

LEVEL — Levels

Input: black/mid/white

Output: black/white

Per-channel option + “link channels” toggle

CURVE — Curves

Curve editor (control points)

Separate curves: RGB composite + optionally R, G, B, L

Optional: spline vs linear interpolation

CONTR — Contrast

contrast (linear)

clarity/midtone contrast (local contrast) optional

preserve highlights toggle

SATUR — Saturation

saturation (global)

vibrance (protect low-sat regions)

Optional: luma protect

Acceptance criteria (blocks)

Every block exposes parameters beyond a single “amount” control.

Parameters have sane defaults and safe ranges.

Parameter edits trigger re-render and persist in session save/load.

Epics → User Stories → Acceptance Criteria
Epic E1 — Image I/O and data model
US1: Import supported formats and bit depths

As a user, I can import PNG/BMP/TIFF/JPG/XISF/FITS in 8/16/32-bit so I can work with my planetary stacks.

AC

Import succeeds for each format type.

Imported image reports correct dimensions, channel count, and effective bit depth.

Internal storage uses float32 without truncation artifacts.

US2: Export supported formats and bit depths

As a user, I can export to PNG/BMP/TIFF/JPG/XISF/FITS in 8/16/32-bit.

AC

Exported file opens in an external viewer for that format.

Chosen bit depth is honored.

Metadata is preserved where format supports it (at least FITS header pass-through when possible).

US3: Session save/load

As a user, I can save and reload a session to continue later.

AC

Reopened session restores inputs, mappings, workflows, params, ROI, and performance settings.

Rendered output matches prior session result (within numeric tolerance).

Epic E2 — Channel assignment and filter mapping
US4: Assign L/R/G/B roles

As a user, I can map files to L/R/G/B so I can build LRGB.

AC

UI allows role assignment per input.

Missing channels are handled per defined strategy (documented).

US5: Assign special filters with editable RGB/L contribution

As a user, I can map special filter images to RGB (and optionally L) with weights.

AC

Weights are editable (0–100) with numeric entry.

Multiple target channels per filter input are supported.

Re-render occurs immediately after changes.

US6: Apply mapping presets

As a user, I can select preset mappings (CH4, IR-pass, Ha, OIII, HOO, SHO) as a starting point.

AC

Selecting a preset updates mapping values instantly.

User can modify and save as a custom preset.

Epic E3 — Workflow engine (two-stage)
US7: Stage 1 per-channel workflows

As a user, I can build independent workflows per L/R/G/B before combining.

AC

Each channel has its own ordered block list.

Output is combined into RGB or LRGB (depending on available channels).

US8: Stage 2 combined workflow

As a user, I can apply additional processing to the combined image.

AC

Stage 2 operates on Stage 1 combined output.

Stage 2 can include any building block valid for color/LRGB context.

US9: Mixer between Stage 1 and Stage 2 outputs

As a user, I can blend Stage 1 and Stage 2 results.

AC

Mixer control updates output immediately.

Mixing behavior is documented and consistent.

US10: Cancellable rendering (“latest-wins”)

As a user, my newest change applies quickly without waiting for older renders.

AC

In-progress render cancels on new changes.

UI remains responsive throughout.

Epic E4 — Workflow editor UX
US11: Drag blocks from library into pipelines

As a user, I can add blocks by dragging them into the workflow.

AC

Drag-drop inserts at intended position.

Multiple instances of the same block type are allowed.

US12: Reorder blocks by drag & drop

As a user, I can reorder blocks to change results.

AC

Reordering triggers re-render.

Order changes persist in session save.

US13: Remove blocks

As a user, I can remove blocks from the workflow.

AC

Remove action is available (context menu or trash drop zone).

Removal triggers re-render and persists.

US14: Enable/disable blocks

As a user, I can bypass blocks without deleting them.

AC

Toggle bypass triggers re-render.

Disabled blocks do not alter the output.

US15: Inspect and edit parameters

As a user, I can edit block parameters with fine control.

AC

Inspector shows parameters for selected block instance.

Changes re-render (debounced) and persist.

Epic E5 — Viewer (real-time + navigation)
US16: Always-visible viewer with zoom/pan

As a user, I can zoom/pan to inspect details.

AC

Smooth zoom and pan.

Viewer does not freeze during render.

US17: Navigator/minimap

As a user, I can use a minimap to navigate like Photoshop.

AC

Minimap shows viewport rectangle.

Clicking/dragging in minimap moves viewport.

US18: Channel visibility toggles

As a user, I can view L/R/G/B/LRGB outputs.

AC

Toggles work as specified (single-channel grayscale, multi-channel composite, full LRGB).

Switching view does not recompute the pipeline unnecessarily (uses cached stages when possible).

Epic E6 — ROI processing
US19: Define ROI

As a user, I can select an ROI rectangle to speed previews.

AC

ROI can be created, moved, resized, cleared.

ROI boundaries are respected.

US20: ROI vs full-frame mode

As a user, I can switch between ROI preview and full-frame.

AC

Mode toggle exists and is honored.

ROI preview is faster for expensive blocks.

Epic E7 — Performance controls & progress
US21: Threads control

As a user, I can set CPU threads used.

AC

Threads clamp to valid range.

Performance changes take effect without restart.

US22: Memory limit control

As a user, I can cap memory usage.

AC

Cache sizes are bounded.

Large images do not crash the app within configured limits (graceful fallback).

US23: Progress indicator

As a user, I can see progress during processing.

AC

Progress shows current stage and block name.

Progress updates while rendering; disappears/settles when done.

Epic E8 — Building blocks implementation quality
US24: Implement each required block with pro-grade parameters

As a user, each block provides controls comparable to common editors.

AC

All listed blocks exist and are usable.

Each block includes the parameter depth defined above (or better).

Parameter ranges are validated; invalid values are rejected with a clear error.

US25: Deterministic output

As a user, I get consistent results given the same inputs and settings.

AC

Same input + params produces same output (within float tolerance).

Session reload yields same result.