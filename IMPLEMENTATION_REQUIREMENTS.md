# PlanetSharp "All Ideas" Implementation Requirements

This checklist captures the requirements requested for a fuller planetary-imaging sharpener workflow.

## R1 — Organize block library into clear sections
- [x] Group block library by processing intent: Tone & Dynamic Range, Color, Sharpening & Detail, Blurring & Denoise.
- Sign-off: ✅ Implemented.

## R2 — Add core color control blocks missing for imaging workflows
- [x] Add Hue Shift block.
- [x] Add Channel Balance (R/G/B gain) block.
- Sign-off: ✅ Implemented.

## R3 — Add dedicated sharpening/detail blocks
- [x] Add Unsharp Mask block with radius/amount/threshold and channel mode.
- [x] Add High-pass Detail block with radius/amount/softness.
- [x] Add Richardson–Lucy deconvolution block with radius/iterations/damping.
- Sign-off: ✅ Implemented.

## R4 — Add chroma noise reduction support
- [x] Add Chroma Denoise block for A/B channel smoothing in LAB space.
- Sign-off: ✅ Implemented.

## R5 — Keep integration consistent with existing pipeline architecture
- [x] Register all new blocks in `block_definitions` so they are serializable and pipeline-ready.
- [x] Expose all new blocks in the block library UI.
- Sign-off: ✅ Implemented.

## R6 — Validate implementation sanity
- [x] Run static sanity check with Python bytecode compilation.
- Sign-off: ✅ Implemented.
