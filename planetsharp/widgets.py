from __future__ import annotations

from typing import Callable

import numpy as np
from PySide6.QtCore import QPoint, Qt, Signal
from PySide6.QtGui import QColor, QImage, QMouseEvent, QPainter, QPen, QPixmap
from PySide6.QtWidgets import QGraphicsPixmapItem, QGraphicsScene, QGraphicsView, QWidget


class ImageViewer(QGraphicsView):
    pixelHovered = Signal(str)

    def __init__(self) -> None:
        super().__init__()
        self._scene = QGraphicsScene(self)
        self.setScene(self._scene)
        self._pixmap_item = QGraphicsPixmapItem()
        self._scene.addItem(self._pixmap_item)
        self._current_float: np.ndarray | None = None
        self._display_buffer: np.ndarray | None = None
        self._zoom = 1.0
        self._panning = False
        self._pan_start = QPoint()

        self.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorViewCenter)
        self.setBackgroundBrush(Qt.GlobalColor.black)
        self.setMouseTracking(True)

    def set_image(self, image_float: np.ndarray | None) -> None:
        self._current_float = image_float
        if image_float is None:
            self._display_buffer = None
            self._pixmap_item.setPixmap(QPixmap())
            return

        self._display_buffer = np.clip(image_float * 255.0, 0, 255).astype(np.uint8, copy=False)
        h, w, _ = self._display_buffer.shape
        qimg = QImage(self._display_buffer.data, w, h, 3 * w, QImage.Format.Format_RGB888)
        self._pixmap_item.setPixmap(QPixmap.fromImage(qimg))
        self._scene.setSceneRect(0, 0, w, h)
        if self._zoom == 1.0:
            self.fitInView(self._scene.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)

    def zoom_in(self) -> None:
        self._apply_zoom(1.25)

    def zoom_out(self) -> None:
        self._apply_zoom(0.8)

    def reset_zoom(self) -> None:
        self.resetTransform()
        self._zoom = 1.0
        if not self._scene.sceneRect().isNull():
            self.fitInView(self._scene.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)

    def _apply_zoom(self, factor: float) -> None:
        self._zoom *= factor
        self.scale(factor, factor)

    def mousePressEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            self._panning = True
            self._pan_start = event.pos()
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        if self._panning:
            delta = event.pos() - self._pan_start
            self._pan_start = event.pos()
            self.horizontalScrollBar().setValue(self.horizontalScrollBar().value() - delta.x())
            self.verticalScrollBar().setValue(self.verticalScrollBar().value() - delta.y())

        if self._current_float is not None:
            scene_pos = self.mapToScene(event.pos())
            x = int(scene_pos.x())
            y = int(scene_pos.y())
            h, w, _ = self._current_float.shape
            if 0 <= x < w and 0 <= y < h:
                pixel = self._current_float[y, x]
                self.pixelHovered.emit(
                    f"x={x}, y={y}, value=({pixel[0]:.4f}, {pixel[1]:.4f}, {pixel[2]:.4f})"
                )
            else:
                self.pixelHovered.emit("x=-, y=-, value=-")
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            self._panning = False
            self.setCursor(Qt.CursorShape.ArrowCursor)
        super().mouseReleaseEvent(event)


class HistogramWidget(QWidget):
    def __init__(self) -> None:
        super().__init__()
        self._image: np.ndarray | None = None
        self._mode = "luminance"
        self._levels_overlay: dict[str, dict[str, float]] | None = None
        self._hist_cache: dict[str, list[np.ndarray]] = {}
        self.setMinimumHeight(180)

    def set_image(self, image_float: np.ndarray | None) -> None:
        self._image = image_float
        self._hist_cache.clear()
        self.update()

    def set_mode(self, mode: str) -> None:
        if mode == self._mode:
            return
        self._mode = mode
        self.update()

    def set_levels_overlay(self, payload: dict[str, dict[str, float]] | None) -> None:
        self._levels_overlay = payload
        self.update()

    def _compute_hist(self) -> list[np.ndarray]:
        if self._image is None:
            return []

        cached = self._hist_cache.get(self._mode)
        if cached is not None:
            return cached

        if self._mode == "luminance":
            lum = 0.2126 * self._image[..., 0] + 0.7152 * self._image[..., 1] + 0.0722 * self._image[..., 2]
            hist, _ = np.histogram(np.clip(lum, 0.0, 1.0), bins=256, range=(0.0, 1.0))
            result = [hist.astype(np.float32)]
            self._hist_cache[self._mode] = result
            return result

        chans = []
        for c in range(3):
            hist, _ = np.histogram(np.clip(self._image[..., c], 0.0, 1.0), bins=256, range=(0.0, 1.0))
            chans.append(hist.astype(np.float32))
        self._hist_cache[self._mode] = chans
        return chans

    def paintEvent(self, _event) -> None:
        painter = QPainter(self)
        painter.fillRect(self.rect(), Qt.GlobalColor.black)
        hists = self._compute_hist()
        if not hists:
            painter.setPen(Qt.GlobalColor.gray)
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "Histogram")
            return

        colors = [Qt.GlobalColor.white] if self._mode == "luminance" else [Qt.GlobalColor.red, Qt.GlobalColor.green, Qt.GlobalColor.blue]
        w = max(1, self.width())
        h = max(1, self.height())
        section_h = max(1, h // 2)

        self._paint_hist_section(painter, hists, colors, "linear", 0, w, section_h)
        self._paint_hist_section(painter, hists, colors, "log", section_h, w, h - section_h)

        if self._levels_overlay is not None:
            self._paint_levels_overlay(painter, w, section_h, top=0)
            self._paint_levels_overlay(painter, w, h - section_h, top=section_h)

        painter.setPen(Qt.GlobalColor.lightGray)
        painter.drawText(6, 14, "Linear")
        painter.drawText(6, section_h + 14, "Log")

    def _paint_hist_section(
        self,
        painter: QPainter,
        hists: list[np.ndarray],
        colors: list[Qt.GlobalColor],
        scale: str,
        top: int,
        width: int,
        height: int,
    ) -> None:
        if height <= 1:
            return
        painter.setClipRect(0, top, width, height)
        for hist, color in zip(hists, colors):
            values = np.log1p(hist) if scale == "log" else hist.copy()
            maxv = np.max(values) if np.max(values) > 0 else 1.0
            norm = values / maxv
            painter.setPen(QPen(color, 1))
            for x in range(256):
                px = int((x / 255.0) * (width - 1))
                bar_h = int(norm[x] * (height - 1))
                y_base = top + height - 1
                painter.drawLine(px, y_base, px, y_base - bar_h)
        painter.setClipping(False)

    def _paint_levels_overlay(self, painter: QPainter, width: int, height: int, top: int = 0) -> None:
        if self._levels_overlay is None:
            return
        boundaries = self._levels_overlay.get("boundaries", {})
        adjustments = self._levels_overlay.get("adjustments", {})
        segments = [
            ("Shadows", 0.0, float(boundaries.get("shadow_upper", 0.25)), float(adjustments.get("shadows", 0.0)), Qt.GlobalColor.darkBlue),
            ("Low-mid", float(boundaries.get("low_mid_lower", 0.30)), float(boundaries.get("low_mid_upper", 0.50)), float(adjustments.get("low_mid", 0.0)), Qt.GlobalColor.darkCyan),
            ("High-mid", float(boundaries.get("high_mid_lower", 0.60)), float(boundaries.get("high_mid_upper", 0.78)), float(adjustments.get("high_mid", 0.0)), Qt.GlobalColor.darkYellow),
            ("Highlights", float(boundaries.get("highlights_lower", 0.86)), 1.0, float(adjustments.get("highlights", 0.0)), Qt.GlobalColor.darkRed),
        ]

        for _name, left, right, amount, color in segments:
            x0 = int(np.clip(left, 0.0, 1.0) * (width - 1))
            x1 = int(np.clip(right, 0.0, 1.0) * (width - 1))
            if x1 <= x0:
                continue
            alpha = int(35 + min(110, abs(amount) * 360))
            tonal_overlay = QColor(255, 255, 255, alpha) if amount >= 0 else QColor(0, 0, 0, alpha)
            painter.fillRect(x0, top, x1 - x0, height, tonal_overlay)
            painter.setOpacity(0.30)
            painter.fillRect(x0, top, x1 - x0, height, color)
            painter.setOpacity(1.0)

        boundary_color = QPen(Qt.GlobalColor.lightGray, 1, Qt.PenStyle.DashLine)
        painter.setPen(boundary_color)
        boundary_keys = [
            "shadow_upper",
            "low_mid_upper",
            "high_mid_upper",
        ]
        for key in boundary_keys:
            x = int(np.clip(float(boundaries.get(key, 0.0)), 0.0, 1.0) * (width - 1))
            painter.drawLine(x, top, x, top + height - 1)
