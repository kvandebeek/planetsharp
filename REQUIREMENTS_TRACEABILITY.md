# Requirements Checklist + Traceability Matrix

Status legend: ✅ Implemented

## Phase checkpoints

- Phase 1 acceptance: ✅
- Phase 2 acceptance: ✅
- Phase 3 acceptance: ✅
- Phase 4 acceptance: ✅
- Phase 5 acceptance: ✅

## REQ1..REQ51

| REQ | Status | Traceability (files + key functions) |
|---|---|---|
| REQ1 | ✅ | `main.py` entrypoint + `planetsharp/app.py::run` |
| REQ2 | ✅ | `main.py` runs `run()` with `python3 main.py` |
| REQ3 | ✅ | `planetsharp/app.py::DARK_STYLE` applied globally |
| REQ4 | ✅ | `planetsharp/app.py::MainWindow.__init__` uses `availableGeometry()` |
| REQ5 | ✅ | `planetsharp/app.py::setWindowFlags(...WindowCloseButtonHint...)` |
| REQ6 | ✅ | `planetsharp/app.py::setWindowFlags` omits min/max hints |
| REQ7 | ✅ | `planetsharp/app.py::MainWindow.__init__` 3 major rows |
| REQ8 | ✅ | `planetsharp/app.py` top row layout first |
| REQ9 | ✅ | `planetsharp/app.py` Open/Save/Save pipeline buttons |
| REQ10 | ✅ | `planetsharp/io_utils.py::load_image` ext + dtype validation |
| REQ11 | ✅ | `planetsharp/io_utils.py::save_image_16bit` always writes `uint16` |
| REQ12 | ✅ | `planetsharp/app.py::on_save_pipeline`, `io_utils.py::save_pipeline` |
| REQ13 | ✅ | `planetsharp/app.py::on_open -> apply_pipeline -> viewer.set_image` |
| REQ14 | ✅ | `planetsharp/app.py` viewer row given high stretch factors |
| REQ15 | ✅ | `planetsharp/widgets.py::ImageViewer.setBackgroundBrush(black)` |
| REQ16 | ✅ | `planetsharp/widgets.py::ImageViewer.set_image` draws pixmap over black scene |
| REQ17 | ✅ | `planetsharp/widgets.py::HistogramWidget`; `app.py` integration |
| REQ18 | ✅ | `planetsharp/app.py` bottom row pipeline section |
| REQ19 | ✅ | `planetsharp/app.py` bottom row has 3 columns |
| REQ20 | ✅ | `planetsharp/app.py` library_frame |
| REQ21 | ✅ | `planetsharp/app.py` pipeline_frame + list |
| REQ22 | ✅ | `planetsharp/app.py` adjust_frame + dynamic sliders |
| REQ23 | ✅ | `planetsharp/blocks.py::block_definitions` brightness |
| REQ24 | ✅ | `planetsharp/blocks.py::block_definitions` contrast |
| REQ25 | ✅ | `planetsharp/blocks.py::block_definitions` saturation |
| REQ26 | ✅ | `planetsharp/blocks.py::block_definitions` gaussian_blur |
| REQ27 | ✅ | `planetsharp/app.py` brightness row with `->` button |
| REQ28 | ✅ | `planetsharp/app.py` contrast row with `->` button |
| REQ29 | ✅ | `planetsharp/app.py` saturation row with `->` button |
| REQ30 | ✅ | `planetsharp/app.py` gaussian blur row with `->` button |
| REQ31 | ✅ | `planetsharp/app.py::add_block` appends unlimited instances |
| REQ32 | ✅ | `planetsharp/blocks.py::PipelineBlock.parameters` per-instance |
| REQ33 | ✅ | `planetsharp/blocks.py` brightness parameter + `app.py::rebuild_adjustment_panel` |
| REQ34 | ✅ | `planetsharp/blocks.py` contrast parameter + dynamic slider UI |
| REQ35 | ✅ | `planetsharp/blocks.py` saturation parameter + dynamic slider UI |
| REQ36 | ✅ | `planetsharp/blocks.py` gaussian `size` parameter + slider |
| REQ37 | ✅ | `planetsharp/blocks.py` gaussian `strength` 0..1 step 0.01 |
| REQ38 | ✅ | `planetsharp/app.py::apply_pipeline` called on any control change |
| REQ39 | ✅ | `planetsharp/blocks.py::BlockDefinition` registry architecture |
| REQ40 | ✅ | `planetsharp/blocks.py::parameters` list supports many params |
| REQ41 | ✅ | `planetsharp/app.py::pipeline_list InternalMove + sync_pipeline_from_ui` |
| REQ42 | ✅ | `planetsharp/app.py::remove_selected` |
| REQ43 | ✅ | `planetsharp/app.py::toggle_selected_enabled` + checkbox |
| REQ44 | ✅ | `planetsharp/app.py::reset_selected` |
| REQ45 | ✅ | `planetsharp/widgets.py::zoom_in/zoom_out/reset_zoom` + UI buttons |
| REQ46 | ✅ | `planetsharp/widgets.py::mousePress/Move` panning behavior |
| REQ47 | ✅ | `planetsharp/widgets.py::pixelHovered`, `app.py::pixel_label` |
| REQ48 | ✅ | `planetsharp/widgets.py::HistogramWidget._compute_hist` + mode combo |
| REQ49 | ✅ | `planetsharp/widgets.py::paintEvent` log/linear behavior + scale combo |
| REQ50 | ✅ | `planetsharp/app.py::on_load_pipeline`, `io_utils.py::load_pipeline` |
| REQ51 | ✅ | `planetsharp/io_utils.py::PIPELINE_VERSION`, JSON `version` enforcement |

