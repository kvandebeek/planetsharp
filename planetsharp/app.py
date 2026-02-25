from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import psutil
from PySide6.QtCore import QTimer, Qt
from PySide6.QtGui import QGuiApplication
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QProgressBar,
    QSlider,
    QSpacerItem,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from .blocks import (
    _make_psf,
    block_definitions,
    deserialize_block,
    instantiate_block,
    normalize_levels_boundaries,
    serialize_block,
)
from .io_utils import (
    load_default_image_folder,
    load_image,
    load_pipeline,
    save_default_image_folder,
    save_image_16bit,
    save_pipeline,
)
from .widgets import HistogramWidget, ImageViewer

DARK_STYLE = """
QWidget { background-color: #121212; color: #E0E0E0; font-size: 12px; }
QPushButton { background-color: #2A2A2A; border: 1px solid #3A3A3A; border-radius: 4px; padding: 6px; }
QPushButton:hover { background-color: #333333; }
QListWidget { background-color: #0F0F0F; border: 1px solid #333333; }
QFrame { border: 1px solid #2E2E2E; }
QSlider::groove:horizontal { height: 6px; background: #2A2A2A; }
QSlider::handle:horizontal { width: 12px; background: #8AB4F8; margin: -4px 0; border-radius: 6px; }
"""


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("PlanetSharp")
        self.setWindowFlags(
            Qt.WindowType.Window
            | Qt.WindowType.CustomizeWindowHint
            | Qt.WindowType.WindowTitleHint
            | Qt.WindowType.WindowCloseButtonHint
        )
        self.setStyleSheet(DARK_STYLE)

        self.definitions = block_definitions()
        self.original_native: np.ndarray | None = None
        self.original_float: np.ndarray | None = None
        self.rendered_float: np.ndarray | None = None
        self.original_clip_counts: tuple[int, int] = (0, 0)
        self.pipeline = []
        self.default_image_folder = load_default_image_folder()
        self._pipeline_apply_timer = QTimer(self)
        self._pipeline_apply_timer.setSingleShot(True)
        self._pipeline_apply_timer.timeout.connect(self._apply_pipeline_now)

        central = QWidget()
        root = QVBoxLayout(central)
        self.setCentralWidget(central)

        # Row 1: File actions
        top_row = QHBoxLayout()
        self.open_btn = QPushButton("Open")
        self.save_btn = QPushButton("Save")
        self.save_pipeline_btn = QPushButton("Save pipeline")
        self.load_pipeline_btn = QPushButton("Load pipeline")
        for b in [self.open_btn, self.save_btn, self.save_pipeline_btn, self.load_pipeline_btn]:
            top_row.addWidget(b)
        top_row.addStretch(1)

        stats_layout = QHBoxLayout()
        stats_layout.setSpacing(12)
        self.cpu_label = QLabel("CPU 0.0%")
        self.cpu_bar = QProgressBar()
        self.cpu_bar.setRange(0, 100)
        self.cpu_bar.setTextVisible(False)
        self.cpu_bar.setFixedWidth(140)

        self.mem_label = QLabel("MEM 0.00 / 0.00 GB")
        self.mem_bar = QProgressBar()
        self.mem_bar.setRange(0, 100)
        self.mem_bar.setTextVisible(False)
        self.mem_bar.setFixedWidth(140)

        stats_layout.addWidget(self.cpu_label)
        stats_layout.addWidget(self.cpu_bar)
        stats_layout.addWidget(self.mem_label)
        stats_layout.addWidget(self.mem_bar)
        top_row.addLayout(stats_layout)
        root.addLayout(top_row)

        # Row 2: Viewer + histogram
        viewer_row = QHBoxLayout()
        viewer_frame = QFrame()
        viewer_layout = QVBoxLayout(viewer_frame)
        self.viewer = ImageViewer()
        zoom_row = QHBoxLayout()
        self.zoom_in_btn = QPushButton("Zoom +")
        self.zoom_out_btn = QPushButton("Zoom -")
        self.zoom_reset_btn = QPushButton("Reset zoom")
        self.pixel_label = QLabel("x=-, y=-, value=-")
        self.clip_label = QLabel("clipped low=0, high=0")
        for b in [self.zoom_in_btn, self.zoom_out_btn, self.zoom_reset_btn]:
            zoom_row.addWidget(b)
        zoom_row.addWidget(self.pixel_label)
        zoom_row.addWidget(self.clip_label)
        zoom_row.addStretch(1)
        viewer_layout.addWidget(self.viewer)
        viewer_layout.addLayout(zoom_row)

        hist_frame = QFrame()
        hist_layout = QVBoxLayout(hist_frame)
        hist_controls = QHBoxLayout()
        self.hist_mode = QComboBox()
        self.hist_mode.addItems(["luminance", "per-channel"])
        hist_controls.addWidget(QLabel("Mode"))
        hist_controls.addWidget(self.hist_mode)
        hist_controls.addStretch(1)
        self.histogram = HistogramWidget()
        hist_layout.addLayout(hist_controls)
        hist_layout.addWidget(self.histogram)

        viewer_row.addWidget(viewer_frame, 3)
        viewer_row.addWidget(hist_frame, 1)
        root.addLayout(viewer_row, 3)

        # Row 3: Pipeline (3 columns)
        bottom = QHBoxLayout()

        library_frame = QFrame()
        library_layout = QVBoxLayout(library_frame)
        library_layout.addWidget(QLabel("Block library"))

        self.library_sections = {
            "Tone & Dynamic Range": ["brightness_contrast", "levels_midtone_transfer"],
            "Color": ["saturation", "hue_shift", "channel_balance"],
            "Sharpening & Detail": ["psd_deconvolution", "wavelet_sharpening", "unsharp_mask", "high_pass_detail"],
            "Blurring & Denoise": ["gaussian_blur", "chroma_denoise"],
        }
        for section, keys in self.library_sections.items():
            section_label = QLabel(section)
            section_label.setStyleSheet("font-weight: 600; color: #8AB4F8;")
            library_layout.addWidget(section_label)
            for key in keys:
                row = QHBoxLayout()
                row.addWidget(QLabel(self.definitions[key].label))
                btn = QPushButton("->")
                btn.clicked.connect(lambda _=False, k=key: self.add_block(k))
                row.addWidget(btn)
                library_layout.addLayout(row)
        library_layout.addStretch(1)

        pipeline_frame = QFrame()
        pipeline_layout = QVBoxLayout(pipeline_frame)
        pipeline_layout.addWidget(QLabel("Pipeline overview"))
        self.pipeline_list = QListWidget()
        self.pipeline_list.setDragDropMode(QListWidget.DragDropMode.InternalMove)
        self.pipeline_list.setDefaultDropAction(Qt.DropAction.MoveAction)
        self.pipeline_list.model().rowsMoved.connect(self.sync_pipeline_from_ui)
        pipeline_layout.addWidget(self.pipeline_list)
        action_row = QHBoxLayout()
        self.remove_btn = QPushButton("Remove")
        self.reset_btn = QPushButton("Reset")
        self.enable_checkbox = QCheckBox("Enabled")
        action_row.addWidget(self.remove_btn)
        action_row.addWidget(self.reset_btn)
        action_row.addWidget(self.enable_checkbox)
        pipeline_layout.addLayout(action_row)

        adjust_frame = QFrame()
        self.adjust_layout = QVBoxLayout(adjust_frame)
        self.adjust_layout.addWidget(QLabel("Adjustment panel"))
        self.adjust_host = QWidget()
        self.adjust_form = QGridLayout(self.adjust_host)
        self.adjust_layout.addWidget(self.adjust_host)
        self.adjust_layout.addStretch(1)

        bottom.addWidget(library_frame, 1)
        bottom.addWidget(pipeline_frame, 1)
        bottom.addWidget(adjust_frame, 2)
        root.addLayout(bottom, 2)

        self.open_btn.clicked.connect(self.on_open)
        self.save_btn.clicked.connect(self.on_save)
        self.save_pipeline_btn.clicked.connect(self.on_save_pipeline)
        self.load_pipeline_btn.clicked.connect(self.on_load_pipeline)
        self.zoom_in_btn.clicked.connect(self.viewer.zoom_in)
        self.zoom_out_btn.clicked.connect(self.viewer.zoom_out)
        self.zoom_reset_btn.clicked.connect(self.viewer.reset_zoom)
        self.viewer.pixelHovered.connect(self.pixel_label.setText)
        self.pipeline_list.currentRowChanged.connect(self.on_selected_block_changed)
        self.remove_btn.clicked.connect(self.remove_selected)
        self.reset_btn.clicked.connect(self.reset_selected)
        self.enable_checkbox.toggled.connect(self.toggle_selected_enabled)
        self.hist_mode.currentTextChanged.connect(self.on_hist_mode_changed)

        screen = QGuiApplication.primaryScreen()
        if screen:
            self.setGeometry(screen.availableGeometry())

        self._init_resource_timer()

    def _init_resource_timer(self) -> None:
        self.resource_timer = self.startTimer(1000)
        psutil.cpu_percent(interval=None)
        self._refresh_resource_usage()

    def timerEvent(self, event) -> None:  # type: ignore[override]
        if event.timerId() == self.resource_timer:
            self._refresh_resource_usage()
            return
        super().timerEvent(event)

    def _refresh_resource_usage(self) -> None:
        cpu_percent = psutil.cpu_percent(interval=None)
        mem = psutil.virtual_memory()
        mem_total_gb = mem.total / (1024**3)
        mem_used_gb = mem.used / (1024**3)
        mem_percent = mem.percent

        self.cpu_label.setText(f"CPU {cpu_percent:.1f}%")
        self.cpu_bar.setValue(int(round(cpu_percent)))
        self._set_usage_color(self.cpu_bar, cpu_percent)

        self.mem_label.setText(f"MEM {mem_used_gb:.2f} / {mem_total_gb:.2f} GB")
        self.mem_bar.setValue(int(round(mem_percent)))
        self._set_usage_color(self.mem_bar, mem_percent)

    def _set_usage_color(self, bar: QProgressBar, value: float) -> None:
        if value < 60:
            color = "#2E7D32"  # green
        elif value < 85:
            color = "#F9A825"  # orange
        else:
            color = "#C62828"  # red
        bar.setStyleSheet(
            "QProgressBar {"
            " border: 1px solid #3A3A3A;"
            " border-radius: 3px;"
            " background-color: #1D1D1D;"
            "}"
            "QProgressBar::chunk {"
            f" background-color: {color};"
            "}"
        )

    def on_hist_mode_changed(self, value: str) -> None:
        self.histogram.set_mode("luminance" if value == "luminance" else "per-channel")

    def on_open(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Open image",
            self.default_image_folder,
            "Images (*.png *.tif *.tiff)",
        )
        if not path:
            return
        try:
            self.original_native, self.original_float = load_image(path)
            self._remember_image_folder(path)
            self.apply_pipeline()
            self.viewer.reset_zoom()
        except Exception as exc:
            QMessageBox.critical(self, "Open failed", str(exc))

    def on_save(self) -> None:
        if self.rendered_float is None:
            return
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Save image",
            self.default_image_folder,
            "Images (*.png *.tif *.tiff)",
        )
        if not path:
            return
        try:
            save_image_16bit(path, self.rendered_float)
            self._remember_image_folder(path)
        except Exception as exc:
            QMessageBox.critical(self, "Save failed", str(exc))

    def _remember_image_folder(self, file_path: str) -> None:
        folder = str(Path(file_path).parent)
        self.default_image_folder = folder
        save_default_image_folder(folder)

    def on_save_pipeline(self) -> None:
        path, _ = QFileDialog.getSaveFileName(self, "Save pipeline", "pipeline.json", "JSON (*.json)")
        if not path:
            return
        save_pipeline(path, [serialize_block(block) for block in self.pipeline])

    def on_load_pipeline(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Load pipeline", "", "JSON (*.json)")
        if not path:
            return
        try:
            payload = load_pipeline(path)
            self.pipeline = [deserialize_block(b, self.definitions) for b in payload["blocks"]]
            self.refresh_pipeline_list()
            self.apply_pipeline()
        except Exception as exc:
            QMessageBox.critical(self, "Load pipeline failed", str(exc))

    def add_block(self, block_type: str) -> None:
        self.pipeline.append(instantiate_block(block_type, self.definitions))
        self.refresh_pipeline_list()
        self.pipeline_list.setCurrentRow(len(self.pipeline) - 1)
        self.apply_pipeline()

    def refresh_pipeline_list(self) -> None:
        self.pipeline_list.clear()
        for block in self.pipeline:
            item = QListWidgetItem(f"{block.label} ({'on' if block.enabled else 'off'})")
            self.pipeline_list.addItem(item)

    def sync_pipeline_from_ui(self, *_args) -> None:
        old = list(self.pipeline)
        reordered = []
        used = [False] * len(old)
        for i in range(self.pipeline_list.count()):
            text = self.pipeline_list.item(i).text()
            for j, block in enumerate(old):
                label = f"{block.label} ({'on' if block.enabled else 'off'})"
                if not used[j] and text == label:
                    reordered.append(block)
                    used[j] = True
                    break
        if len(reordered) == len(old):
            self.pipeline = reordered
            self.apply_pipeline()

    def on_selected_block_changed(self, row: int) -> None:
        self.rebuild_adjustment_panel(row)
        self.update_histogram_levels_overlay()

    def _sanitize_levels_block_parameters(self, block) -> None:
        if block.block_type != "levels_midtone_transfer":
            return
        boundaries = normalize_levels_boundaries(block.parameters)
        for key, value in boundaries.items():
            block.parameters[key] = value

    def update_histogram_levels_overlay(self) -> None:
        row = self.pipeline_list.currentRow()
        if row < 0 or row >= len(self.pipeline):
            self.histogram.set_levels_overlay(None)
            return
        block = self.pipeline[row]
        if block.block_type != "levels_midtone_transfer":
            self.histogram.set_levels_overlay(None)
            return
        self._sanitize_levels_block_parameters(block)
        boundaries = {
            "shadow_upper": float(block.parameters["shadow_upper"]),
            "low_mid_lower": float(block.parameters["low_mid_lower"]),
            "low_mid_upper": float(block.parameters["low_mid_upper"]),
            "high_mid_lower": float(block.parameters["high_mid_lower"]),
            "high_mid_upper": float(block.parameters["high_mid_upper"]),
            "highlights_lower": float(block.parameters["highlights_lower"]),
        }
        adjustments = {
            "shadows": float(block.parameters["shadows"]),
            "low_mid": float(block.parameters["low_mid"]),
            "high_mid": float(block.parameters["high_mid"]),
            "highlights": float(block.parameters["highlights"]),
        }
        self.histogram.set_levels_overlay({"boundaries": boundaries, "adjustments": adjustments})

    def rebuild_adjustment_panel(self, row: int) -> None:
        while self.adjust_form.count():
            child = self.adjust_form.takeAt(0)
            if child.widget():
                child.widget().deleteLater()
        if row < 0 or row >= len(self.pipeline):
            self.enable_checkbox.setChecked(False)
            return

        block = self.pipeline[row]
        self._sanitize_levels_block_parameters(block)
        self.enable_checkbox.blockSignals(True)
        self.enable_checkbox.setChecked(block.enabled)
        self.enable_checkbox.blockSignals(False)

        definition = self.definitions[block.block_type]
        visible_specs = []
        for spec in definition.parameters:
            if spec.key == "lab_balance":
                mode = str(block.parameters.get("channel_mode", "All channels"))
                if mode == "All channels":
                    continue
            visible_specs.append(spec)

        row_index = 0
        for spec in visible_specs:
            if spec.key == "shadow_upper":
                self.adjust_form.addItem(
                    QSpacerItem(0, 14, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Fixed),
                    row_index,
                    0,
                    1,
                    3,
                )
                row_index += 1

            label = QLabel(spec.label)
            self.adjust_form.addWidget(label, row_index, 0)

            if spec.input_type == "choice":
                dropdown = QComboBox()
                dropdown.addItems(spec.choices)
                current = str(block.parameters[spec.key])
                if current not in spec.choices:
                    current = str(spec.default)
                    block.parameters[spec.key] = current
                dropdown.setCurrentText(current)

                def make_choice_handler(b=block, s=spec, selected_row=row):
                    def handler(value: str) -> None:
                        b.parameters[s.key] = value
                        if b.block_type == "gaussian_blur" and s.key == "channel_mode":
                            self.rebuild_adjustment_panel(selected_row)
                        self.apply_pipeline()
                    return handler

                dropdown.currentTextChanged.connect(make_choice_handler())
                self.adjust_form.addWidget(dropdown, row_index, 1)
                self.adjust_form.addWidget(QLabel(current), row_index, 2)
                row_index += 1
                continue

            slider = QSlider(Qt.Orientation.Horizontal)
            slider.setMinimum(0)
            steps = int(round((spec.max_value - spec.min_value) / spec.step))
            slider.setMaximum(steps)
            current = float(block.parameters[spec.key])
            slider.setValue(int(round((current - spec.min_value) / spec.step)))
            value_label = QLabel(f"{current:.2f}")

            def make_handler(b=block, s=spec, vl=value_label):
                def handler(value: int) -> None:
                    b.parameters[s.key] = s.min_value + value * s.step
                    if b.block_type == "levels_midtone_transfer":
                        self._sanitize_levels_block_parameters(b)
                    vl.setText(f"{float(b.parameters[s.key]):.2f}")
                    self.apply_pipeline()
                    self.update_histogram_levels_overlay()
                return handler

            slider.valueChanged.connect(make_handler())
            self.adjust_form.addWidget(slider, row_index, 1)
            self.adjust_form.addWidget(value_label, row_index, 2)
            row_index += 1

        if block.block_type == "psd_deconvolution":
            psf_size = int(round(float(block.parameters.get("psf_size", 7.0))))
            seeing_index = float(block.parameters.get("seeing_index", 1.2))
            psf_strength = float(block.parameters.get("psf_strength", 0.3))
            psf = _make_psf(psf_size, seeing_index, psf_strength)
            psf_label = QLabel("PSF kernel")
            psf_label.setStyleSheet("font-weight: 600; color: #8AB4F8;")
            self.adjust_form.addWidget(psf_label, row_index, 0, 1, 3)
            row_index += 1
            psf_text = "\n".join(" ".join(f"{v:0.3f}" for v in r) for r in psf)
            psf_value = QLabel(psf_text)
            psf_value.setStyleSheet("font-family: monospace;")
            psf_value.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
            self.adjust_form.addWidget(psf_value, row_index, 0, 1, 3)
            row_index += 1

    def remove_selected(self) -> None:
        row = self.pipeline_list.currentRow()
        if row < 0:
            return
        self.pipeline.pop(row)
        self.refresh_pipeline_list()
        if self.pipeline:
            self.pipeline_list.setCurrentRow(min(row, len(self.pipeline) - 1))
        self.apply_pipeline()

    def reset_selected(self) -> None:
        row = self.pipeline_list.currentRow()
        if row < 0:
            return
        block = self.pipeline[row]
        for spec in self.definitions[block.block_type].parameters:
            block.parameters[spec.key] = spec.default
        self.rebuild_adjustment_panel(row)
        self.apply_pipeline()
        self.update_histogram_levels_overlay()

    def toggle_selected_enabled(self, enabled: bool) -> None:
        row = self.pipeline_list.currentRow()
        if row < 0:
            return
        self.pipeline[row].enabled = enabled
        self.refresh_pipeline_list()
        self.pipeline_list.setCurrentRow(row)
        self.apply_pipeline()
        self.update_histogram_levels_overlay()

    def apply_pipeline(self) -> None:
        if self._pipeline_apply_timer.isActive():
            self._pipeline_apply_timer.stop()
        self._pipeline_apply_timer.start(0)

    def _apply_pipeline_now(self) -> None:
        if self.original_float is None:
            self.rendered_float = None
            self.viewer.set_image(None)
            self.histogram.set_image(None)
            self.original_clip_counts = (0, 0)
            self.clip_label.setText("clip O(low=0, high=0) → N(low=0, high=0) Δ(low=+0, high=+0)")
            return

        self.original_clip_counts = self._compute_clipped_pixel_counts(self.original_float)
        out = self.original_float.copy()
        for block in self.pipeline:
            if not block.enabled:
                continue
            definition = self.definitions[block.block_type]
            out = definition.apply_fn(out, block.parameters)
        low_clipped, high_clipped = self._compute_clipped_pixel_counts(out)
        original_low, original_high = self.original_clip_counts
        delta_low = low_clipped - original_low
        delta_high = high_clipped - original_high
        self.clip_label.setText(
            "clip "
            f"O(low={original_low}, high={original_high}) "
            f"→ N(low={low_clipped}, high={high_clipped}) "
            f"Δ(low={delta_low:+d}, high={delta_high:+d})"
        )
        self.rendered_float = np.clip(out, 0.0, 1.0).astype(np.float32)
        self.viewer.set_image(self.rendered_float)
        self.histogram.set_image(self.rendered_float)
        self.update_histogram_levels_overlay()

    @staticmethod
    def _compute_clipped_pixel_counts(image: np.ndarray) -> tuple[int, int]:
        low_clipped = int(np.count_nonzero(np.any(image < 0.0, axis=2)))
        high_clipped = int(np.count_nonzero(np.any(image > 1.0, axis=2)))
        return low_clipped, high_clipped


def run() -> None:
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
