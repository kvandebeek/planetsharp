from __future__ import annotations

import json
import time
import tkinter as tk
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from tkinter import messagebox, simpledialog, ttk
from typing import Any

from planetsharp.core.models import BlockInstance, Session
from planetsharp.processing.blocks import BLOCK_DEFINITIONS

LAYOUT_STATE = Path.home() / ".planetsharp.layout.json"


@dataclass
class LibraryItem:
    code: str
    name: str
    description: str
    category: str


BLOCK_LIBRARY: list[LibraryItem] = [
    LibraryItem("DECON", "Deconvolution", "Sharpen using PSF-aware iterative recovery.", "Sharpening"),
    LibraryItem("AWAVE", "A trous Wavelets", "Layered wavelet sharpening/denoise controls.", "Sharpening"),
    LibraryItem("RWAVE", "RegiStax Wavelets", "RegiStax-style detail enhancement.", "Sharpening"),
    LibraryItem("UMASK", "Unsharp Mask", "Classic local contrast/sharpening.", "Sharpening"),
    LibraryItem("GBLUR", "Gaussian Blur", "Gaussian smoothing for noise reduction.", "Noise / smoothing"),
    LibraryItem("BILAT", "Bilateral Filter", "Edge-preserving smoothing.", "Noise / smoothing"),
    LibraryItem("NOISE", "Denoise", "General luminance/chroma noise cleanup.", "Noise / smoothing"),
    LibraryItem("WHBAL", "White Balance", "Temperature/tint balancing.", "Color"),
    LibraryItem("SATUR", "Saturation", "Global saturation and vibrance.", "Color"),
    LibraryItem("CURVE", "Curves", "Tone remapping with curve points.", "Color"),
    LibraryItem("CONTR", "Contrast", "Global and midtone contrast.", "Color"),
    LibraryItem("ALIGN", "Channel Align", "Align channels/geometry correction.", "Geometry"),
    LibraryItem("DERIN", "Deringing", "Suppress ringing artifacts.", "Geometry"),
    LibraryItem("SELCO", "Selective Color", "Mask-like targeted color edits.", "Masking / selection"),
]


class CollapsibleSection(ttk.Frame):
    def __init__(self, parent: tk.Widget, title: str):
        super().__init__(parent)
        self._visible = tk.BooleanVar(value=True)
        self._button = ttk.Checkbutton(self, text=title, variable=self._visible, command=self._toggle, style="Toolbutton")
        self._button.pack(fill="x")
        self.body = ttk.Frame(self)
        self.body.pack(fill="x", padx=4, pady=(0, 4))

    def _toggle(self) -> None:
        if self._visible.get():
            self.body.pack(fill="x", padx=4, pady=(0, 4))
        else:
            self.body.pack_forget()


class PlanetSharpApp:
    def __init__(self, session: Session | None = None):
        self.session = session or Session()
        self.root = tk.Tk()
        self.root.title("PlanetSharp v1")
        self.root.geometry("1400x900")
        self.status = tk.StringVar(value="Ready")
        self.library_filter = tk.StringVar()
        self.active_image = tk.StringVar(value="R")
        self.viewer_mode = tk.StringVar(value="Before/After")
        self.live_preview = tk.BooleanVar(value=True)
        self.lock_roi = tk.BooleanVar(value=False)
        self.drag_library_code: str | None = None
        self.selected_block_id: str | None = None
        self.param_clipboard: dict[str, Any] = {}
        self.block_favorites: set[str] = set()
        self.param_widgets: dict[str, tuple[tk.Variable, ttk.Scale]] = {}
        self._last_log_id = 0
        self._build_layout()
        self._bind_shortcuts()
        self._restore_ui_state()
        self._populate_library()
        self._seed_images_and_pipeline()
        self._refresh_pipeline()

    def _build_layout(self) -> None:
        self._build_toolbar()
        self.main_panes = ttk.Panedwindow(self.root, orient="horizontal")
        self.main_panes.pack(fill="both", expand=True)

        self.left_panel = ttk.Frame(self.main_panes)
        self.center_panel = ttk.Frame(self.main_panes)
        self.right_panel = ttk.Frame(self.main_panes)

        self.main_panes.add(self.left_panel, weight=2)
        self.main_panes.add(self.center_panel, weight=6)
        self.main_panes.add(self.right_panel, weight=3)

        self._build_left_toolbox()
        self._build_center_workspace()
        self._build_inspector()

    def _build_toolbar(self) -> None:
        bar = ttk.Frame(self.root)
        bar.pack(fill="x", padx=6, pady=4)
        for label, cmd in [
            ("Open", self._not_implemented),
            ("Save", self._save_ui_state),
            ("Export", self._not_implemented),
            ("Undo", self._not_implemented),
            ("Redo", self._not_implemented),
            ("Run pipeline", self._simulate_render),
        ]:
            ttk.Button(bar, text=label, command=cmd).pack(side="left", padx=2)
        ttk.Checkbutton(bar, text="Live Preview", variable=self.live_preview).pack(side="left", padx=10)
        ttk.Label(bar, textvariable=self.status).pack(side="right")

    def _build_left_toolbox(self) -> None:
        nb = ttk.Notebook(self.left_panel)
        nb.pack(fill="both", expand=True)
        library = ttk.Frame(nb)
        presets = ttk.Frame(nb)
        history = ttk.Frame(nb)
        nb.add(library, text="Library")
        nb.add(presets, text="Presets")
        nb.add(history, text="History")

        ttk.Entry(library, textvariable=self.library_filter).pack(fill="x", padx=4, pady=4)
        self.library_filter.trace_add("write", lambda *_: self._populate_library())
        self.library_tree = ttk.Treeview(library, show="tree", selectmode="browse")
        self.library_tree.pack(fill="both", expand=True, padx=4, pady=4)
        self.library_tree.bind("<<TreeviewSelect>>", self._select_library_item)
        self.library_tree.bind("<Double-1>", lambda *_: self._add_selected_library_block())
        self.library_tree.bind("<ButtonPress-1>", self._start_library_drag)

        ttk.Label(presets, text="Saved pipeline presets").pack(anchor="w", padx=4, pady=(4, 0))
        self.preset_list = tk.Listbox(presets, height=8)
        self.preset_list.pack(fill="both", expand=True, padx=4, pady=4)
        ttk.Button(presets, text="Save current as preset", command=self._save_pipeline_preset).pack(fill="x", padx=4)
        ttk.Button(presets, text="Load selected preset", command=self._load_pipeline_preset).pack(fill="x", padx=4, pady=(4, 8))

        ttk.Label(history, text="Undo/redo stack + snapshots").pack(anchor="w", padx=4, pady=4)
        self.history_list = tk.Listbox(history)
        self.history_list.pack(fill="both", expand=True, padx=4, pady=4)
        ttk.Button(history, text="Add snapshot", command=self._snapshot).pack(fill="x", padx=4, pady=(0, 8))

    def _build_center_workspace(self) -> None:
        center = ttk.Panedwindow(self.center_panel, orient="vertical")
        center.pack(fill="both", expand=True)

        viewer = ttk.LabelFrame(center, text="Viewer")
        pipeline = ttk.LabelFrame(center, text="Pipeline Canvas")
        filmstrip = ttk.LabelFrame(center, text="Filmstrip")
        center.add(viewer, weight=6)
        center.add(pipeline, weight=3)
        center.add(filmstrip, weight=1)

        controls = ttk.Frame(viewer)
        controls.pack(fill="x", padx=4, pady=4)
        ttk.OptionMenu(controls, self.viewer_mode, self.viewer_mode.get(), "Before/After", "Split", "Blink (A/B)").pack(side="left")
        for zoom in ["Fit", "100%", "200%", "1:1"]:
            ttk.Button(controls, text=zoom, command=lambda z=zoom: self._set_zoom(z)).pack(side="left", padx=2)
        ttk.Checkbutton(controls, text="Lock ROI", variable=self.lock_roi).pack(side="left", padx=8)
        ttk.Button(controls, text="Toggle overlays", command=self._not_implemented).pack(side="left", padx=2)

        self.viewer_canvas = tk.Canvas(viewer, background="#0d0d10", height=360)
        self.viewer_canvas.pack(fill="both", expand=True, padx=4)
        self.viewer_canvas.create_text(420, 180, fill="white", text="Before / After / Split / Blink preview")
        self.viewer_canvas.create_rectangle(230, 120, 610, 260, outline="#6bc1ff", width=2)

        status = ttk.Frame(viewer)
        status.pack(fill="x", padx=4, pady=4)
        self.viewer_status = tk.StringVar(value="Zoom 100% | 16-bit | RGB | x:0 y:0 | pixel:0.000")
        ttk.Label(status, textvariable=self.viewer_status).pack(anchor="w")

        self.pipeline_tree = ttk.Treeview(pipeline, columns=("enabled", "name"), show="headings", selectmode="browse")
        self.pipeline_tree.heading("enabled", text="On")
        self.pipeline_tree.heading("name", text="Block")
        self.pipeline_tree.column("enabled", width=40, anchor="center")
        self.pipeline_tree.column("name", width=300, anchor="w")
        self.pipeline_tree.pack(fill="both", expand=True, padx=4, pady=4)
        self.pipeline_tree.bind("<<TreeviewSelect>>", self._on_pipeline_select)

        row = ttk.Frame(pipeline)
        row.pack(fill="x", padx=4, pady=(0, 4))
        for label, cmd in [
            ("▲", self._move_up),
            ("▼", self._move_down),
            ("Enable/Disable", self._toggle_enabled),
            ("Duplicate", self._duplicate_selected),
            ("Delete", self._delete_selected),
            ("Apply", self._simulate_render),
            ("Drop from Library", self._add_selected_library_block),
        ]:
            ttk.Button(row, text=label, command=cmd).pack(side="left", padx=2)

        apply_opts = ttk.Frame(pipeline)
        apply_opts.pack(fill="x", padx=4, pady=(0, 4))
        self.apply_mode = tk.StringVar(value="Live preview")
        for mode in ["Live preview", "Manual apply", "Apply after 250ms idle"]:
            ttk.Radiobutton(apply_opts, text=mode, variable=self.apply_mode, value=mode).pack(side="left", padx=4)

        self.filmstrip_list = tk.Listbox(filmstrip, exportselection=False, height=3)
        self.filmstrip_list.pack(fill="both", expand=True, padx=4, pady=4)
        self.filmstrip_list.bind("<<ListboxSelect>>", self._set_active_image_from_filmstrip)
        ttk.Button(filmstrip, text="Sync pipeline to selected images", command=self._sync_selected_images).pack(side="left", padx=4, pady=(0, 4))
        ttk.Button(filmstrip, text="Sync selected block only", command=self._sync_selected_block).pack(side="left", padx=4, pady=(0, 4))

    def _build_inspector(self) -> None:
        tools = ttk.Frame(self.right_panel)
        tools.pack(fill="x", padx=4, pady=(4, 0))
        ttk.Button(tools, text="Hide", command=lambda: self.main_panes.forget(self.right_panel)).pack(side="left")
        ttk.Button(tools, text="Show", command=self._show_right_panel).pack(side="left", padx=4)

        self.inspector_scroll = ttk.Frame(self.right_panel)
        self.inspector_scroll.pack(fill="both", expand=True, padx=4, pady=4)

        self.block_section = CollapsibleSection(self.inspector_scroll, "Block parameters")
        self.block_section.pack(fill="x")
        self.block_search = tk.StringVar()
        ttk.Entry(self.block_section.body, textvariable=self.block_search).pack(fill="x", pady=(0, 4))
        self.block_search.trace_add("write", lambda *_: self._render_param_editor())
        ttk.Frame(self.block_section.body).pack(fill="x")
        toolbar = ttk.Frame(self.block_section.body)
        toolbar.pack(fill="x", pady=(0, 4))
        ttk.Button(toolbar, text="Defaults", command=self._reset_selected_params).pack(side="left")
        ttk.Button(toolbar, text="Recommended", command=self._recommended_params).pack(side="left", padx=2)
        ttk.Button(toolbar, text="Copy", command=self._copy_params).pack(side="left", padx=2)
        ttk.Button(toolbar, text="Paste", command=self._paste_params).pack(side="left", padx=2)
        self.param_area = ttk.Frame(self.block_section.body)
        self.param_area.pack(fill="x")

        hist = CollapsibleSection(self.inspector_scroll, "Histogram / Waveform")
        hist.pack(fill="x")
        ttk.Label(hist.body, text="Histogram source: ROI if enabled").pack(anchor="w")

        roi = CollapsibleSection(self.inspector_scroll, "ROI tools")
        roi.pack(fill="x")
        self.roi_text = tk.StringVar(value="Rect: x=230 y=120 w=380 h=140")
        ttk.Label(roi.body, textvariable=self.roi_text).pack(anchor="w")
        ttk.Checkbutton(roi.body, text="Use ROI for histogram", variable=tk.BooleanVar(value=True)).pack(anchor="w")

        perf = CollapsibleSection(self.inspector_scroll, "Performance + queue")
        perf.pack(fill="x")
        self.progress = tk.IntVar(value=0)
        ttk.Progressbar(perf.body, mode="determinate", maximum=100, variable=self.progress).pack(fill="x")
        self.perf_text = tk.StringVar(value="Idle | CPU 0% | GPU N/A | Cache: warm")
        ttk.Label(perf.body, textvariable=self.perf_text).pack(anchor="w")

        log = CollapsibleSection(self.inspector_scroll, "Log / status")
        log.pack(fill="both", expand=True)
        self.log_box = tk.Text(log.body, height=10)
        self.log_box.pack(fill="both", expand=True)
        actions = ttk.Frame(log.body)
        actions.pack(fill="x")
        ttk.Button(actions, text="Copy log", command=self._copy_log).pack(side="left")
        ttk.Button(actions, text="Clear log", command=lambda: self.log_box.delete("1.0", "end")).pack(side="left", padx=4)

    def _bind_shortcuts(self) -> None:
        self.root.bind("<Control-z>", lambda *_: self._log("Undo requested"))
        self.root.bind("<Control-y>", lambda *_: self._log("Redo requested"))
        self.root.bind("<space>", lambda *_: self.status.set("Pan mode (hold space)"))
        self.root.bind("<Control-MouseWheel>", lambda e: self._set_zoom("200%" if e.delta > 0 else "100%"))
        self.root.bind("b", lambda *_: self._toggle_before_after())
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    def _seed_images_and_pipeline(self) -> None:
        channels = ["L", "R", "G", "B", "CH4"]
        for channel in channels:
            self.filmstrip_list.insert("end", f"{channel}  • mapped")
        if not self.session.stage2_workflow.blocks:
            for code in ("DECON", "SATUR", "CURVE"):
                self.session.stage2_workflow.blocks.append(BlockInstance(type=code, params=dict(BLOCK_DEFINITIONS[code].defaults)))

    def _pipeline(self) -> list[BlockInstance]:
        return self.session.stage2_workflow.blocks

    def _populate_library(self) -> None:
        self.library_tree.delete(*self.library_tree.get_children())
        needle = self.library_filter.get().strip().lower()
        by_cat: dict[str, list[LibraryItem]] = defaultdict(list)
        for item in BLOCK_LIBRARY:
            if needle and needle not in f"{item.name} {item.code} {item.description}".lower():
                continue
            by_cat[item.category].append(item)
        for category in ("Sharpening", "Noise / smoothing", "Color", "Geometry", "Masking / selection"):
            parent = self.library_tree.insert("", "end", text=category, open=True)
            for item in by_cat.get(category, []):
                star = "⭐" if item.code in self.block_favorites else "☆"
                self.library_tree.insert(parent, "end", iid=f"lib:{item.code}", text=f"{star} {item.name} ({item.code})", values=())

    def _select_library_item(self, _event: Any) -> None:
        sel = self.library_tree.selection()
        if not sel:
            return
        item_id = sel[0]
        if item_id.startswith("lib:"):
            code = item_id.split(":", 1)[1]
            desc = next((x.description for x in BLOCK_LIBRARY if x.code == code), "")
            self.status.set(desc)

    def _start_library_drag(self, event: tk.Event) -> None:
        row = self.library_tree.identify_row(event.y)
        if row.startswith("lib:"):
            self.drag_library_code = row.split(":", 1)[1]

    def _add_selected_library_block(self) -> None:
        sel = self.library_tree.selection()
        code = None
        if sel and sel[0].startswith("lib:"):
            code = sel[0].split(":", 1)[1]
        elif self.drag_library_code:
            code = self.drag_library_code
        if not code:
            return
        self._pipeline().append(BlockInstance(type=code, params=dict(BLOCK_DEFINITIONS[code].defaults)))
        self._refresh_pipeline()
        self._log(f"Added block {code} by drag/drop")

    def _refresh_pipeline(self) -> None:
        self.pipeline_tree.delete(*self.pipeline_tree.get_children())
        for block in self._pipeline():
            on = "☑" if block.enabled else "☐"
            self.pipeline_tree.insert("", "end", iid=block.id, values=(on, block.type))
        if self._pipeline() and self.selected_block_id not in {b.id for b in self._pipeline()}:
            self.selected_block_id = self._pipeline()[0].id
        if self.selected_block_id:
            self.pipeline_tree.selection_set(self.selected_block_id)
            self._on_pipeline_select(None)

    def _on_pipeline_select(self, _event: Any) -> None:
        sel = self.pipeline_tree.selection()
        if not sel:
            return
        self.selected_block_id = sel[0]
        self._render_param_editor()

    def _selected_block(self) -> BlockInstance | None:
        return next((b for b in self._pipeline() if b.id == self.selected_block_id), None)

    def _render_param_editor(self) -> None:
        for child in self.param_area.winfo_children():
            child.destroy()
        self.param_widgets.clear()
        block = self._selected_block()
        if not block:
            return
        q = self.block_search.get().strip().lower()
        for name, value in block.params.items():
            if q and q not in name.lower():
                continue
            if isinstance(value, (int, float)):
                row = ttk.Frame(self.param_area)
                row.pack(fill="x", pady=2)
                ttk.Label(row, text=name, width=18).pack(side="left")
                var = tk.DoubleVar(value=float(value))
                scale = ttk.Scale(row, from_=0.0, to=max(2.0, float(value) * 2 + 1), variable=var, command=lambda _v, n=name, v=var: self._set_param(n, v.get()))
                scale.pack(side="left", fill="x", expand=True, padx=4)
                entry = ttk.Entry(row, width=8)
                entry.pack(side="left")
                entry.insert(0, f"{value:.3f}" if isinstance(value, float) else str(value))
                entry.bind("<Return>", lambda e, n=name, ent=entry: self._entry_set_param(n, ent.get()))
                ttk.Button(row, text="↺", width=2, command=lambda n=name: self._reset_param(n)).pack(side="left", padx=2)
                self.param_widgets[name] = (var, scale)
            elif isinstance(value, bool):
                var = tk.BooleanVar(value=value)
                ttk.Checkbutton(self.param_area, text=name, variable=var, command=lambda n=name, v=var: self._set_param(n, v.get())).pack(anchor="w")

    def _set_param(self, name: str, value: Any) -> None:
        block = self._selected_block()
        if not block:
            return
        block.params[name] = round(float(value), 4) if isinstance(value, (float, int)) else value
        if self.apply_mode.get() == "Live preview" and self.live_preview.get():
            self._simulate_render()

    def _entry_set_param(self, name: str, raw: str) -> None:
        try:
            value = float(raw)
        except ValueError:
            return
        self._set_param(name, value)
        self._render_param_editor()

    def _reset_param(self, name: str) -> None:
        block = self._selected_block()
        if not block:
            return
        block.params[name] = BLOCK_DEFINITIONS[block.type].defaults.get(name, block.params[name])
        self._render_param_editor()

    def _reset_selected_params(self) -> None:
        block = self._selected_block()
        if block:
            block.params = dict(BLOCK_DEFINITIONS[block.type].defaults)
            self._render_param_editor()

    def _recommended_params(self) -> None:
        block = self._selected_block()
        if not block:
            return
        for k, v in block.params.items():
            if isinstance(v, (int, float)):
                block.params[k] = round(v * 1.1, 4)
        self._render_param_editor()

    def _copy_params(self) -> None:
        block = self._selected_block()
        if block:
            self.param_clipboard = {"type": block.type, "params": dict(block.params)}
            self._log(f"Copied parameters from {block.type}")

    def _paste_params(self) -> None:
        block = self._selected_block()
        if not block or self.param_clipboard.get("type") != block.type:
            self._log("Paste failed: select same block type")
            return
        block.params = dict(self.param_clipboard["params"])
        self._render_param_editor()
        self._simulate_render()

    def _toggle_enabled(self) -> None:
        block = self._selected_block()
        if block:
            block.enabled = not block.enabled
            self._refresh_pipeline()
            self._simulate_render()

    def _move_up(self) -> None:
        block = self._selected_block()
        if not block:
            return
        pipe = self._pipeline()
        i = pipe.index(block)
        if i > 0:
            pipe[i - 1], pipe[i] = pipe[i], pipe[i - 1]
            self._refresh_pipeline()
            self._simulate_render()

    def _move_down(self) -> None:
        block = self._selected_block()
        if not block:
            return
        pipe = self._pipeline()
        i = pipe.index(block)
        if i < len(pipe) - 1:
            pipe[i + 1], pipe[i] = pipe[i], pipe[i + 1]
            self._refresh_pipeline()
            self._simulate_render()

    def _duplicate_selected(self) -> None:
        block = self._selected_block()
        if block:
            self._pipeline().append(BlockInstance(type=block.type, params=dict(block.params), enabled=block.enabled))
            self._refresh_pipeline()

    def _delete_selected(self) -> None:
        block = self._selected_block()
        if block:
            self._pipeline().remove(block)
            self.selected_block_id = None
            self._refresh_pipeline()

    def _save_pipeline_preset(self) -> None:
        name = simpledialog.askstring("Save preset", "Preset name:", parent=self.root)
        if not name:
            return
        notes = simpledialog.askstring("Save preset", "Notes (optional):", parent=self.root) or ""
        preset = {
            "name": name,
            "notes": notes,
            "blocks": [{"type": b.type, "enabled": b.enabled, "params": b.params} for b in self._pipeline()],
        }
        self.preset_list.insert("end", json.dumps(preset))

    def _load_pipeline_preset(self) -> None:
        sel = self.preset_list.curselection()
        if not sel:
            return
        preset = json.loads(self.preset_list.get(sel[0]))
        self.session.stage2_workflow.blocks = [
            BlockInstance(type=b["type"], enabled=b["enabled"], params=b["params"]) for b in preset["blocks"]
        ]
        self._refresh_pipeline()

    def _snapshot(self) -> None:
        self.history_list.insert("end", f"Snapshot {self.history_list.size() + 1}: {len(self._pipeline())} blocks")

    def _simulate_render(self) -> None:
        self.progress.set(0)
        self.perf_text.set("Running pipeline | CPU 52% | GPU N/A | Cache: preview")
        self.root.update_idletasks()
        for pct in (15, 35, 65, 100):
            self.progress.set(pct)
            self.root.update_idletasks()
        self.perf_text.set("Idle | CPU 0% | GPU N/A | Cache: warm")
        self._log(f"Render complete: {[b.type for b in self._pipeline() if b.enabled]}")

    def _set_zoom(self, zoom: str) -> None:
        self.viewer_status.set(f"Zoom {zoom} | 16-bit | RGB | x:128 y:72 | pixel:0.553")

    def _toggle_before_after(self) -> None:
        self.viewer_mode.set("Split" if self.viewer_mode.get() == "Before/After" else "Before/After")

    def _set_active_image_from_filmstrip(self, _event: Any) -> None:
        sel = self.filmstrip_list.curselection()
        if sel:
            img = self.filmstrip_list.get(sel[0]).split()[0]
            self.active_image.set(img)
            self.status.set(f"Active image: {img}")

    def _sync_selected_images(self) -> None:
        sel = self.filmstrip_list.curselection()
        names = [self.filmstrip_list.get(i).split()[0] for i in sel]
        self._log(f"Sync edits applied to: {', '.join(names) if names else 'none'}")

    def _sync_selected_block(self) -> None:
        block = self._selected_block()
        if block:
            self._log(f"Synced block {block.type} to selected images")

    def _copy_log(self) -> None:
        text = self.log_box.get("1.0", "end").strip()
        self.root.clipboard_clear()
        self.root.clipboard_append(text)

    def _log(self, message: str) -> None:
        self._last_log_id += 1
        stamp = time.strftime("%H:%M:%S")
        self.log_box.insert("end", f"[{stamp}] #{self._last_log_id:03d} {message}\n")
        self.log_box.see("end")

    def _show_right_panel(self) -> None:
        if self.right_panel not in self.main_panes.panes():
            self.main_panes.add(self.right_panel, weight=3)

    def _save_ui_state(self) -> None:
        state = {
            "viewer_mode": self.viewer_mode.get(),
            "live_preview": self.live_preview.get(),
            "lock_roi": self.lock_roi.get(),
            "geometry": self.root.geometry(),
        }
        LAYOUT_STATE.write_text(json.dumps(state, indent=2), encoding="utf-8")
        self.status.set("Layout saved")

    def _restore_ui_state(self) -> None:
        if not LAYOUT_STATE.exists():
            return
        try:
            state = json.loads(LAYOUT_STATE.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return
        self.viewer_mode.set(state.get("viewer_mode", self.viewer_mode.get()))
        self.live_preview.set(state.get("live_preview", self.live_preview.get()))
        self.lock_roi.set(state.get("lock_roi", self.lock_roi.get()))
        if geometry := state.get("geometry"):
            self.root.geometry(geometry)

    def _not_implemented(self) -> None:
        messagebox.showinfo("PlanetSharp", "This action is a placeholder in v1 prototype.")

    def _on_close(self) -> None:
        self._save_ui_state()
        self.root.destroy()

    def run(self) -> None:
        self.root.mainloop()


def main() -> None:
    PlanetSharpApp().run()


if __name__ == "__main__":
    main()
