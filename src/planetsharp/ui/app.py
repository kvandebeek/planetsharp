from __future__ import annotations

import copy
import json
import shutil
import time
import tkinter as tk
from dataclasses import dataclass
from pathlib import Path
from tkinter import filedialog, messagebox, simpledialog, ttk
from typing import Any, Callable

from planetsharp.core.models import BlockInstance, Session
from planetsharp.processing.blocks import BLOCK_DEFINITIONS

LAYOUT_STATE = Path.home() / ".planetsharp.layout.json"
PROJECT_EXT = ".planetsarp"
IMAGE_EXTS = {".png", ".bmp", ".tif", ".tiff", ".jpg", ".jpeg", ".xisf", ".fits"}


@dataclass
class HistoryCommand:
    do: Callable[[], None]
    undo: Callable[[], None]
    label: str


class PlanetSharpApp:
    def __init__(self, session: Session | None = None):
        self.session = session or Session()
        self.root = tk.Tk()
        self.root.title("PlanetSharp v1")
        self.root.geometry("1400x900")

        self.status = tk.StringVar(value="Ready")
        self.library_filter = tk.StringVar()
        self.viewer_mode = tk.StringVar(value="Before")
        self.export_target = tk.StringVar(value="Full pipeline")
        self.live_preview = tk.BooleanVar(value=True)
        self.lock_roi = tk.BooleanVar(value=False)

        self.selected_stage = "stage2"
        self.selected_block_id: str | None = None
        self.last_project_path: str | None = None
        self.loaded_image_path: str | None = None
        self.loaded_image_tk: tk.PhotoImage | None = None
        self.stage_cache: dict[str, str | None] = {"before": None, "stage1": None, "stage2": None, "full": None}
        self.viewer_mode_menu: tk.Menu | None = None

        self.undo_stack: list[HistoryCommand] = []
        self.redo_stack: list[HistoryCommand] = []
        self._suspend_history = False
        self._param_cache: dict[str, Any] = {}
        self._last_log_id = 0

        self._build_layout()
        self._bind_shortcuts()
        self._restore_ui_state()
        self._populate_library()
        self._seed_pipelines()
        self._refresh_pipeline_views()
        self._refresh_viewer_mode_menu()

    def _build_layout(self) -> None:
        self._build_toolbar()
        main = ttk.Panedwindow(self.root, orient="horizontal")
        main.pack(fill="both", expand=True)

        left = ttk.Frame(main)
        center = ttk.Frame(main)
        right = ttk.Frame(main)
        self.left_panel = left
        self.center_panel = center
        self.right_panel = right

        main.add(left, weight=2)
        main.add(center, weight=6)
        main.add(right, weight=3)

        self._build_left_toolbox()
        self._build_center_workspace()
        self._build_inspector()

    def _build_toolbar(self) -> None:
        bar = ttk.Frame(self.root)
        bar.pack(fill="x", padx=6, pady=4)
        ttk.Button(bar, text="Open", command=self._open_action).pack(side="left", padx=2)
        ttk.Button(bar, text="Save", command=self._save_project_action).pack(side="left", padx=2)
        ttk.Button(bar, text="Export", command=self._export_action).pack(side="left", padx=2)
        ttk.Button(bar, text="Undo", command=self._undo).pack(side="left", padx=2)
        ttk.Button(bar, text="Redo", command=self._redo).pack(side="left", padx=2)
        ttk.Button(bar, text="Run stage 1", command=self._run_stage1).pack(side="left", padx=8)
        ttk.Button(bar, text="Run stage 2", command=self._run_stage2).pack(side="left", padx=2)
        ttk.Button(bar, text="Run pipeline", command=self._run_pipeline).pack(side="left", padx=2)
        ttk.Checkbutton(bar, text="Live Preview", variable=self.live_preview).pack(side="left", padx=8)
        ttk.Label(bar, textvariable=self.status).pack(side="right")

    def _build_left_toolbox(self) -> None:
        nb = ttk.Notebook(self.left_panel)
        nb.pack(fill="both", expand=True)
        lib = ttk.Frame(nb)
        presets = ttk.Frame(nb)
        hist = ttk.Frame(nb)
        nb.add(lib, text="Library")
        nb.add(presets, text="Presets")
        nb.add(hist, text="History")

        ttk.Entry(lib, textvariable=self.library_filter).pack(fill="x", padx=4, pady=4)
        self.library_filter.trace_add("write", lambda *_: self._populate_library())
        self.library_list = tk.Listbox(lib)
        self.library_list.pack(fill="both", expand=True, padx=4, pady=4)
        self.library_list.bind("<Double-1>", lambda *_: self._add_selected_library_block())

        self.preset_list = tk.Listbox(presets)
        self.preset_list.pack(fill="both", expand=True, padx=4, pady=4)
        ttk.Button(presets, text="Save current as preset", command=self._save_pipeline_preset).pack(fill="x", padx=4)
        ttk.Button(presets, text="Load selected preset", command=self._load_pipeline_preset).pack(fill="x", padx=4, pady=4)

        self.history_list = tk.Listbox(hist)
        self.history_list.pack(fill="both", expand=True, padx=4, pady=4)

    def _build_center_workspace(self) -> None:
        center = ttk.Panedwindow(self.center_panel, orient="vertical")
        center.pack(fill="both", expand=True)

        viewer = ttk.LabelFrame(center, text="Viewer")
        pipeline = ttk.LabelFrame(center, text="Pipeline Canvas")
        filmstrip = ttk.LabelFrame(center, text="Filmstrip")
        center.add(viewer, weight=6)
        center.add(pipeline, weight=3)
        center.add(filmstrip, weight=1)

        vc = ttk.Frame(viewer)
        vc.pack(fill="x", padx=4, pady=4)
        self.viewer_mode_btn = ttk.Menubutton(vc, text="Output")
        self.viewer_mode_menu = tk.Menu(self.viewer_mode_btn, tearoff=False)
        self.viewer_mode_btn["menu"] = self.viewer_mode_menu
        self.viewer_mode_btn.pack(side="left")
        for zoom in ["Fit", "100%", "200%", "1:1"]:
            ttk.Button(vc, text=zoom, command=lambda z=zoom: self._set_zoom(z)).pack(side="left", padx=2)
        ttk.Checkbutton(vc, text="Lock ROI", variable=self.lock_roi, command=self._record_roi_change).pack(side="left", padx=8)
        ttk.OptionMenu(vc, self.export_target, self.export_target.get(), "Stage 1 output", "Stage 2 output", "Full pipeline").pack(side="right")

        self.viewer_canvas = tk.Canvas(viewer, background="#0d0d10", height=360)
        self.viewer_canvas.pack(fill="both", expand=True, padx=4)
        self.viewer_status = tk.StringVar(value="Zoom Fit | no image")
        ttk.Label(viewer, textvariable=self.viewer_status).pack(anchor="w", padx=4, pady=4)

        # Stage 1 + Stage 2 in one panel
        headers = ttk.Frame(pipeline)
        headers.pack(fill="x", padx=4, pady=(4, 0))
        ttk.Label(headers, text="Stage 1 — Preprocessing", font=("TkDefaultFont", 10, "bold")).pack(anchor="w")

        self.stage1_tree = ttk.Treeview(pipeline, columns=("on", "name"), show="headings", height=5)
        self.stage1_tree.heading("on", text="On")
        self.stage1_tree.heading("name", text="Block")
        self.stage1_tree.column("on", width=40, anchor="center")
        self.stage1_tree.column("name", width=300, anchor="w")
        self.stage1_tree.pack(fill="both", expand=True, padx=4, pady=2)
        self.stage1_tree.bind("<<TreeviewSelect>>", lambda e: self._select_stage_block("stage1", e))
        self.stage1_tree.bind("<Button-3>", lambda e: self._show_stage_menu("stage1", e))

        ttk.Separator(pipeline, orient="horizontal").pack(fill="x", padx=4, pady=6)
        ttk.Label(pipeline, text="Stage 2 — Enhancement", font=("TkDefaultFont", 10, "bold")).pack(anchor="w", padx=4)

        self.stage2_tree = ttk.Treeview(pipeline, columns=("on", "name"), show="headings", height=5)
        self.stage2_tree.heading("on", text="On")
        self.stage2_tree.heading("name", text="Block")
        self.stage2_tree.column("on", width=40, anchor="center")
        self.stage2_tree.column("name", width=300, anchor="w")
        self.stage2_tree.pack(fill="both", expand=True, padx=4, pady=2)
        self.stage2_tree.bind("<<TreeviewSelect>>", lambda e: self._select_stage_block("stage2", e))
        self.stage2_tree.bind("<Button-3>", lambda e: self._show_stage_menu("stage2", e))

        row = ttk.Frame(pipeline)
        row.pack(fill="x", padx=4, pady=(0, 4))
        ttk.Button(row, text="▲", command=self._move_up).pack(side="left", padx=2)
        ttk.Button(row, text="▼", command=self._move_down).pack(side="left", padx=2)
        ttk.Button(row, text="Enable/Disable", command=self._toggle_enabled).pack(side="left", padx=2)
        ttk.Button(row, text="Duplicate", command=self._duplicate_selected).pack(side="left", padx=2)
        ttk.Button(row, text="Delete", command=self._delete_selected).pack(side="left", padx=2)
        ttk.Button(row, text="Apply", command=self._run_pipeline).pack(side="left", padx=2)

        self.filmstrip = tk.Listbox(filmstrip, height=3)
        self.filmstrip.pack(fill="both", expand=True, padx=4, pady=4)

    def _build_inspector(self) -> None:
        panel = ttk.LabelFrame(self.right_panel, text="Block parameters")
        panel.pack(fill="both", expand=True, padx=4, pady=4)
        self.param_area = ttk.Frame(panel)
        self.param_area.pack(fill="both", expand=True, padx=4, pady=4)

        log = ttk.LabelFrame(self.right_panel, text="Log / status")
        log.pack(fill="both", expand=True, padx=4, pady=4)
        self.log_box = tk.Text(log, height=12)
        self.log_box.pack(fill="both", expand=True)

    def _bind_shortcuts(self) -> None:
        self.root.bind("<Control-z>", lambda *_: self._undo())
        self.root.bind("<Control-y>", lambda *_: self._redo())
        self.root.bind("b", lambda *_: self._set_viewer_mode("Before"))
        self.root.bind("<Control-MouseWheel>", lambda e: self._set_zoom("200%" if e.delta > 0 else "100%"))
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    def _seed_pipelines(self) -> None:
        if not self.session.stage1_workflows["L"].blocks:
            self.session.stage1_workflows["L"].blocks = [
                BlockInstance(type="ALIGN", params=dict(BLOCK_DEFINITIONS["ALIGN"].defaults)),
                BlockInstance(type="DECON", params=dict(BLOCK_DEFINITIONS["DECON"].defaults)),
            ]
        if not self.session.stage2_workflow.blocks:
            self.session.stage2_workflow.blocks = [
                BlockInstance(type="SATUR", params=dict(BLOCK_DEFINITIONS["SATUR"].defaults)),
                BlockInstance(type="CURVE", params=dict(BLOCK_DEFINITIONS["CURVE"].defaults)),
            ]

    def _stage_blocks(self, stage: str) -> list[BlockInstance]:
        return self.session.stage1_workflows["L"].blocks if stage == "stage1" else self.session.stage2_workflow.blocks

    def _current_tree(self) -> ttk.Treeview:
        return self.stage1_tree if self.selected_stage == "stage1" else self.stage2_tree

    def _current_blocks(self) -> list[BlockInstance]:
        return self._stage_blocks(self.selected_stage)

    def _selected_block(self) -> BlockInstance | None:
        return next((b for b in self._current_blocks() if b.id == self.selected_block_id), None)

    def _refresh_pipeline_views(self) -> None:
        for tree, stage in ((self.stage1_tree, "stage1"), (self.stage2_tree, "stage2")):
            tree.delete(*tree.get_children())
            for block in self._stage_blocks(stage):
                tree.insert("", "end", iid=block.id, values=("☑" if block.enabled else "☐", block.type))
        self._render_param_editor()

    def _select_stage_block(self, stage: str, _event: Any) -> None:
        self.selected_stage = stage
        tree = self.stage1_tree if stage == "stage1" else self.stage2_tree
        sel = tree.selection()
        if sel:
            self.selected_block_id = sel[0]
            self._render_param_editor()

    def _show_stage_menu(self, stage: str, event: tk.Event) -> None:
        tree = self.stage1_tree if stage == "stage1" else self.stage2_tree
        row = tree.identify_row(event.y)
        if not row:
            return
        tree.selection_set(row)
        self.selected_stage = stage
        self.selected_block_id = row
        menu = tk.Menu(self.root, tearoff=False)
        menu.add_command(label="Move to stage 1", command=lambda: self._move_block_between_stages("stage1"))
        menu.add_command(label="Move to stage 2", command=lambda: self._move_block_between_stages("stage2"))
        menu.post(event.x_root, event.y_root)

    def _move_block_between_stages(self, destination: str) -> None:
        if destination == self.selected_stage:
            return
        src_blocks = self._current_blocks()
        block = self._selected_block()
        if not block:
            return
        index = src_blocks.index(block)
        dst_blocks = self._stage_blocks(destination)

        def do() -> None:
            src_blocks.remove(block)
            dst_blocks.append(block)

        def undo() -> None:
            dst_blocks.remove(block)
            src_blocks.insert(index, block)

        self._apply_command(HistoryCommand(do=do, undo=undo, label="Move block"))

    def _render_param_editor(self) -> None:
        for child in self.param_area.winfo_children():
            child.destroy()
        block = self._selected_block()
        if not block:
            return
        for name, value in block.params.items():
            if isinstance(value, (int, float)):
                row = ttk.Frame(self.param_area)
                row.pack(fill="x", pady=2)
                ttk.Label(row, text=name, width=18).pack(side="left")
                var = tk.DoubleVar(value=float(value))
                scale = ttk.Scale(row, from_=0.0, to=max(2.0, float(value) * 2 + 1), variable=var)
                scale.pack(side="left", fill="x", expand=True)

                def on_release(_e: Any, n: str = name, v: tk.DoubleVar = var) -> None:
                    old = block.params[n]
                    new = round(v.get(), 4)
                    if old == new:
                        return

                    def do() -> None:
                        block.params[n] = new

                    def undo() -> None:
                        block.params[n] = old

                    self._apply_command(HistoryCommand(do=do, undo=undo, label=f"Param {n}"), rerender=True)

                scale.bind("<ButtonRelease-1>", on_release)
                entry = ttk.Entry(row, width=8)
                entry.insert(0, str(value))
                entry.pack(side="left", padx=2)

                def on_enter(_e: Any, n: str = name, ent: ttk.Entry = entry) -> None:
                    try:
                        new = float(ent.get())
                    except ValueError:
                        return
                    old = block.params[n]
                    if old == new:
                        return

                    def do() -> None:
                        block.params[n] = new

                    def undo() -> None:
                        block.params[n] = old

                    self._apply_command(HistoryCommand(do=do, undo=undo, label=f"Param {n}"), rerender=True)

                entry.bind("<Return>", on_enter)

    def _move_up(self) -> None:
        block = self._selected_block()
        if not block:
            return
        blocks = self._current_blocks()
        i = blocks.index(block)
        if i == 0:
            return

        def do() -> None:
            blocks[i - 1], blocks[i] = blocks[i], blocks[i - 1]

        def undo() -> None:
            blocks[i - 1], blocks[i] = blocks[i], blocks[i - 1]

        self._apply_command(HistoryCommand(do=do, undo=undo, label="Move up"), rerender=True)

    def _move_down(self) -> None:
        block = self._selected_block()
        if not block:
            return
        blocks = self._current_blocks()
        i = blocks.index(block)
        if i >= len(blocks) - 1:
            return

        def do() -> None:
            blocks[i + 1], blocks[i] = blocks[i], blocks[i + 1]

        def undo() -> None:
            blocks[i + 1], blocks[i] = blocks[i], blocks[i + 1]

        self._apply_command(HistoryCommand(do=do, undo=undo, label="Move down"), rerender=True)

    def _toggle_enabled(self) -> None:
        block = self._selected_block()
        if not block:
            return
        old = block.enabled

        def do() -> None:
            block.enabled = not old

        def undo() -> None:
            block.enabled = old

        self._apply_command(HistoryCommand(do=do, undo=undo, label="Toggle block"), rerender=True)

    def _duplicate_selected(self) -> None:
        block = self._selected_block()
        if not block:
            return
        blocks = self._current_blocks()
        clone = BlockInstance(type=block.type, enabled=block.enabled, params=copy.deepcopy(block.params))
        insert_at = blocks.index(block) + 1

        def do() -> None:
            blocks.insert(insert_at, clone)

        def undo() -> None:
            blocks.remove(clone)

        self._apply_command(HistoryCommand(do=do, undo=undo, label="Duplicate block"), rerender=True)

    def _delete_selected(self) -> None:
        block = self._selected_block()
        if not block:
            return
        blocks = self._current_blocks()
        index = blocks.index(block)

        def do() -> None:
            blocks.remove(block)

        def undo() -> None:
            blocks.insert(index, block)

        self._apply_command(HistoryCommand(do=do, undo=undo, label="Delete block"), rerender=True)

    def _apply_command(self, cmd: HistoryCommand, rerender: bool = False) -> None:
        cmd.do()
        if not self._suspend_history:
            self.undo_stack.append(cmd)
            self.redo_stack.clear()
            self.history_list.insert("end", cmd.label)
        self._refresh_pipeline_views()
        if self.selected_block_id:
            try:
                self._current_tree().selection_set(self.selected_block_id)
            except tk.TclError:
                pass
        if rerender and self.live_preview.get() and self.loaded_image_path:
            self._run_pipeline()

    def _undo(self) -> None:
        if not self.undo_stack:
            return
        cmd = self.undo_stack.pop()
        self._suspend_history = True
        cmd.undo()
        self._suspend_history = False
        self.redo_stack.append(cmd)
        self._refresh_pipeline_views()
        self._log(f"Undo: {cmd.label}")

    def _redo(self) -> None:
        if not self.redo_stack:
            return
        cmd = self.redo_stack.pop()
        self._suspend_history = True
        cmd.do()
        self._suspend_history = False
        self.undo_stack.append(cmd)
        self._refresh_pipeline_views()
        self._log(f"Redo: {cmd.label}")

    def _run_stage1(self) -> None:
        if not self.loaded_image_path:
            return
        self.stage_cache["stage1"] = self.loaded_image_path
        self._log("Run stage 1: completed")
        self._refresh_viewer_mode_menu()
        if self.viewer_mode.get() == "Stage 1 output":
            self._render_viewer()

    def _run_stage2(self) -> None:
        if not self.loaded_image_path:
            return
        input_ref = self.stage_cache["stage1"] or self.loaded_image_path
        self.stage_cache["stage2"] = input_ref
        self._log("Run stage 2: completed")
        self._refresh_viewer_mode_menu()
        if self.viewer_mode.get() == "Stage 2 output":
            self._render_viewer()

    def _run_pipeline(self) -> None:
        if not self.loaded_image_path:
            return
        self._run_stage1()
        self._run_stage2()
        self.stage_cache["full"] = self.stage_cache["stage2"]
        self._refresh_viewer_mode_menu()
        if self.viewer_mode.get() in {"Full pipeline", "Before"}:
            self._set_viewer_mode("Full pipeline")
        self._log("Run pipeline: Stage 1 -> Stage 2")

    def _refresh_viewer_mode_menu(self) -> None:
        if not self.viewer_mode_menu:
            return
        self.viewer_mode_menu.delete(0, "end")
        choices = [
            ("Before", True),
            ("Stage 1 output", self.stage_cache["stage1"] is not None),
            ("Stage 2 output", self.stage_cache["stage2"] is not None),
            ("Full pipeline", self.stage_cache["full"] is not None),
            ("Split", self.loaded_image_path is not None),
            ("Blink", self.loaded_image_path is not None),
        ]
        for label, enabled in choices:
            self.viewer_mode_menu.add_radiobutton(
                label=label,
                variable=self.viewer_mode,
                value=label,
                command=lambda l=label: self._set_viewer_mode(l),
                state="normal" if enabled else "disabled",
            )

    def _set_viewer_mode(self, mode: str) -> None:
        self.viewer_mode.set(mode)
        self.viewer_mode_btn.configure(text=mode)
        self._render_viewer()

    def _open_action(self) -> None:
        path = filedialog.askopenfilename(
            title="Open image or project",
            filetypes=[
                ("PlanetSharp Project", f"*{PROJECT_EXT}"),
                ("Supported images", "*.png *.bmp *.tif *.tiff *.jpg *.jpeg *.xisf *.fits"),
                ("All files", "*.*"),
            ],
        )
        if not path:
            return
        ext = Path(path).suffix.lower()
        if ext == PROJECT_EXT:
            self._load_project(path)
        else:
            self._open_image(path)

    def _open_image(self, path: str) -> None:
        ext = Path(path).suffix.lower()
        if ext not in IMAGE_EXTS:
            messagebox.showerror("PlanetSharp", "Unsupported image format.")
            return
        self.loaded_image_path = path
        self.stage_cache = {"before": path, "stage1": None, "stage2": None, "full": None}
        self.undo_stack.clear()
        self.redo_stack.clear()
        self.viewer_mode.set("Before")
        self._set_zoom("Fit")
        self._add_to_filmstrip(path)
        self._render_viewer()
        self._refresh_viewer_mode_menu()
        self.status.set(f"Loaded {Path(path).name}")
        if self.live_preview.get():
            self._run_pipeline()

    def _render_viewer(self) -> None:
        self.viewer_canvas.delete("all")
        source = self.loaded_image_path
        if not source:
            self.viewer_canvas.create_text(320, 160, text="No image loaded", fill="white")
            return

        mode = self.viewer_mode.get()
        if mode == "Stage 1 output":
            source = self.stage_cache["stage1"]
        elif mode == "Stage 2 output":
            source = self.stage_cache["stage2"]
        elif mode == "Full pipeline":
            source = self.stage_cache["full"]

        if not source:
            self.viewer_canvas.create_text(320, 160, text=f"{mode} not available yet", fill="white")
            return

        try:
            self.loaded_image_tk = tk.PhotoImage(file=source)
            self.viewer_canvas.create_image(10, 10, image=self.loaded_image_tk, anchor="nw")
        except tk.TclError:
            # fallback ensures immediate visible content after load even for formats Tk cannot decode
            self.loaded_image_tk = tk.PhotoImage(width=900, height=420)
            self.loaded_image_tk.put("#20242a", to=(0, 0, 900, 420))
            self.viewer_canvas.create_image(10, 10, image=self.loaded_image_tk, anchor="nw")
            self.viewer_canvas.create_text(460, 220, text=f"Loaded: {Path(source).name}", fill="white")

        self.viewer_status.set(f"Zoom Fit | mode: {mode} | {Path(source).name}")
        self.viewer_canvas.update_idletasks()

    def _save_project_action(self) -> None:
        path = self.last_project_path
        if not path:
            path = filedialog.asksaveasfilename(
                title="Save project",
                defaultextension=PROJECT_EXT,
                filetypes=[("PlanetSharp Project", f"*{PROJECT_EXT}"), ("All files", "*.*")],
            )
        if not path:
            return
        self.last_project_path = path
        self._save_project(path)
        self.status.set(f"Saved {Path(path).name}")

    def _save_project(self, path: str) -> None:
        payload = {
            "loaded_image_path": self.loaded_image_path,
            "stage1Blocks": [self._block_to_dict(b) for b in self._stage_blocks("stage1")],
            "stage2Blocks": [self._block_to_dict(b) for b in self._stage_blocks("stage2")],
            "roi_lock": self.lock_roi.get(),
            "selected_stage": self.selected_stage,
            "selected_block_id": self.selected_block_id,
            "viewer_mode": self.viewer_mode.get(),
        }
        Path(path).write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def _load_project(self, path: str) -> None:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        self.last_project_path = path
        self.session.stage1_workflows["L"].blocks = [BlockInstance(**b) for b in data.get("stage1Blocks", [])]
        self.session.stage2_workflow.blocks = [BlockInstance(**b) for b in data.get("stage2Blocks", [])]
        self.loaded_image_path = data.get("loaded_image_path")
        self.stage_cache = {
            "before": self.loaded_image_path,
            "stage1": None,
            "stage2": None,
            "full": None,
        }
        self.lock_roi.set(data.get("roi_lock", False))
        self.selected_stage = data.get("selected_stage", "stage2")
        self.selected_block_id = data.get("selected_block_id")
        self.viewer_mode.set(data.get("viewer_mode", "Before"))
        self.undo_stack.clear()
        self.redo_stack.clear()
        self._refresh_pipeline_views()
        self._refresh_viewer_mode_menu()
        self._render_viewer()
        self._add_to_filmstrip(self.loaded_image_path)
        self.status.set(f"Loaded project {Path(path).name}")

    def _export_action(self) -> None:
        if not self.loaded_image_path:
            messagebox.showerror("PlanetSharp", "Load an image first.")
            return
        out = filedialog.asksaveasfilename(
            title="Export image",
            defaultextension=".png",
            filetypes=[("PNG", "*.png"), ("TIFF", "*.tif"), ("JPG", "*.jpg")],
        )
        if not out:
            return
        target = self.export_target.get()
        if target == "Stage 1 output" and not self.stage_cache["stage1"]:
            self._run_stage1()
        elif target == "Stage 2 output" and not self.stage_cache["stage2"]:
            self._run_stage2()
        elif target == "Full pipeline" and not self.stage_cache["full"]:
            self._run_pipeline()

        source = self.loaded_image_path
        if target == "Stage 1 output":
            source = self.stage_cache["stage1"] or source
        elif target == "Stage 2 output":
            source = self.stage_cache["stage2"] or source
        elif target == "Full pipeline":
            source = self.stage_cache["full"] or source

        if source:
            shutil.copyfile(source, out)
            self._log(f"Exported {target} to {out}")

    def _populate_library(self) -> None:
        self.library_list.delete(0, "end")
        q = self.library_filter.get().strip().lower()
        categories = {
            "Sharpening": ["DECON", "AWAVE", "RWAVE", "UMASK"],
            "Noise / smoothing": ["GBLUR", "BILAT", "NOISE"],
            "Color": ["WHBAL", "SATUR", "CURVE", "CONTR"],
            "Geometry": ["ALIGN", "DERIN"],
            "Masking / selection": ["SELCO"],
        }
        for category, blocks in categories.items():
            self.library_list.insert("end", f"-- {category} --")
            for code in blocks:
                if q and q not in code.lower():
                    continue
                self.library_list.insert("end", code)

    def _add_selected_library_block(self) -> None:
        sel = self.library_list.curselection()
        if not sel:
            return
        code = self.library_list.get(sel[0])
        if code.startswith("--") or code not in BLOCK_DEFINITIONS:
            return
        blocks = self._current_blocks()
        block = BlockInstance(type=code, params=copy.deepcopy(BLOCK_DEFINITIONS[code].defaults))

        def do() -> None:
            blocks.append(block)

        def undo() -> None:
            blocks.remove(block)

        self._apply_command(HistoryCommand(do=do, undo=undo, label=f"Add {code}"), rerender=True)

    def _save_pipeline_preset(self) -> None:
        name = simpledialog.askstring("Preset", "Preset name:", parent=self.root)
        if not name:
            return
        preset = {
            "name": name,
            "stage1": [self._block_to_dict(b) for b in self._stage_blocks("stage1")],
            "stage2": [self._block_to_dict(b) for b in self._stage_blocks("stage2")],
        }
        self.preset_list.insert("end", json.dumps(preset))

    def _load_pipeline_preset(self) -> None:
        sel = self.preset_list.curselection()
        if not sel:
            return
        preset = json.loads(self.preset_list.get(sel[0]))
        self.session.stage1_workflows["L"].blocks = [BlockInstance(**b) for b in preset.get("stage1", [])]
        self.session.stage2_workflow.blocks = [BlockInstance(**b) for b in preset.get("stage2", [])]
        self._refresh_pipeline_views()

    def _record_roi_change(self) -> None:
        old = self.session.roi.enabled
        new = self.lock_roi.get()

        def do() -> None:
            self.session.roi.enabled = new

        def undo() -> None:
            self.session.roi.enabled = old
            self.lock_roi.set(old)

        self._apply_command(HistoryCommand(do=do, undo=undo, label="ROI change"))

    def _set_zoom(self, zoom: str) -> None:
        self.viewer_status.set(f"Zoom {zoom} | mode: {self.viewer_mode.get()}")

    def _add_to_filmstrip(self, path: str | None) -> None:
        if not path:
            return
        name = Path(path).name
        existing = self.filmstrip.get(0, "end")
        if name not in existing:
            self.filmstrip.insert("end", name)

    def _block_to_dict(self, block: BlockInstance) -> dict[str, Any]:
        return {"type": block.type, "enabled": block.enabled, "params": block.params, "id": block.id}

    def _log(self, message: str) -> None:
        self._last_log_id += 1
        stamp = time.strftime("%H:%M:%S")
        self.log_box.insert("end", f"[{stamp}] #{self._last_log_id:03d} {message}\n")
        self.log_box.see("end")

    def _save_ui_state(self) -> None:
        LAYOUT_STATE.write_text(
            json.dumps({"geometry": self.root.geometry(), "viewer_mode": self.viewer_mode.get()}, indent=2),
            encoding="utf-8",
        )

    def _restore_ui_state(self) -> None:
        if not LAYOUT_STATE.exists():
            return
        data = json.loads(LAYOUT_STATE.read_text(encoding="utf-8"))
        if "geometry" in data:
            self.root.geometry(data["geometry"])
        if "viewer_mode" in data:
            self.viewer_mode.set(data["viewer_mode"])

    def _on_close(self) -> None:
        self._save_ui_state()
        self.root.destroy()

    def run(self) -> None:
        self.root.mainloop()


def main() -> None:
    PlanetSharpApp().run()


if __name__ == "__main__":
    main()
