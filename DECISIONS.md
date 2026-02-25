# DECISIONS

## Implemented choices for ambiguous areas

1. **Fullscreen-sized behavior (REQ4)**
   - Used `QScreen.availableGeometry()` and applied it to the main window. This fills the usable desktop area while preserving taskbar/dock visibility.

2. **Close-only window controls (REQ6)**
   - Used Qt window flags: `WindowTitleHint + WindowCloseButtonHint + CustomizeWindowHint` to keep close and suppress minimize/maximize.

3. **16/32-bit validation for PNG/TIF (REQ10)**
   - Accepted dtypes: `uint16`, `uint32`, `int32`, `float32`.
   - Rejected all others with explicit message.

4. **Internal format**
   - Converted opened image to normalized `float32` RGB in range `[0,1]` for pipeline processing.

5. **Gaussian blur implementation**
   - Implemented a pure-numpy separable Gaussian blur (no OpenCV dependency), with odd kernel size adjustment.

6. **Pipeline reorder (REQ41)**
   - Implemented drag-and-drop reorder in the pipeline list (`InternalMove`).

7. **Histogram behavior (REQ48/REQ49)**
   - Histogram is computed from the rendered image.
   - Added mode toggle: luminance/per-channel and scale toggle: linear/log.

8. **Pipeline JSON stability (REQ51)**
   - Saved format includes `{ "version": 1, "blocks": [...] }`.
   - Loader enforces version match and structural validation.

## Phase self-checks

- **Phase 1 (shell/layout):** Implemented dark-only shell, close-only controls, 3-row layout skeleton.
- **Phase 2 (file actions/viewer/hist placeholder):** Implemented open/save image and save/load pipeline wiring; viewer black background; histogram widget visible.
- **Phase 3 (pipeline core):** Implemented block library, add-multiple instances, per-instance params, realtime rendering in order.
- **Phase 4 (management/viewer):** Implemented drag-drop reorder, remove, enable/disable, reset, zoom/pan/reset, pixel readout.
- **Phase 5 (hist toggles/persistence final):** Implemented luminance/per-channel and linear/log toggles; versioned JSON finalized.
