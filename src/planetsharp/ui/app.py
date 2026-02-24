from __future__ import annotations

import copy
import json
import time
import tkinter as tk
from dataclasses import dataclass
from pathlib import Path
from tkinter import filedialog, messagebox, simpledialog, ttk
from typing import Any, Callable

from planetsharp.core.models import BlockInstance, InputImage, Role, Session
from planetsharp.io.formats import read_image, write_image
from planetsharp.persistence.session_store import SessionStore
from planetsharp.processing.blocks import BLOCK_DEFINITIONS

LAYOUT_STATE = Path.home() / ".planetsharp.layout.json"
PROJECT_SUFFIX = ".planetsharp"


@dataclass
class Command:
    do: Callable[[], None]
    undo: Callable[[], None]
    description: str


class History:
    def __init__(self) -> None:
        self._undo: list[Command] = []
        self._redo: list[Command] = []

    def execute(self, command: Command) -> None:
        command.do()
        self._undo.append(command)
        self._redo.clear()

    def undo(self) -> bool:
        if not self._undo:
            return False
        command = self._undo.pop()
        command.undo()
        self._redo.append(command)
        return True

    def redo(self) -> bool:
        if not self._redo:
            return False
        command = self._redo.pop()
        command.do()
        self._undo.append(command)
        return True

    def clear(self) -> None:
        self._undo.clear()
        self._redo.clear()


class PlanetSharpApp:
    def __init__(self, session: Session | None = None):
        self.session = session or Session()
        self.root = tk.Tk()
        self.root.title("PlanetSharp v1")
        self.root.geometry("1400x900")

        self.status = tk.StringVar(value="Ready")
        self.viewer_mode = tk.StringVar(value="Before")
        self.live_preview = tk.BooleanVar(value=True)
        self.selected_stage = tk.IntVar(value=1)

        self.loaded_image: dict[str, Any] | None = None
        self.loaded_image_path: str | None = None
        self.current_project_path: str | None = None
        self.stage_cache: dict[str, dict[str, Any] | None] = {"stage1": None, "stage2": None, "full": None}

        self.history = History()
        self.selected_block_id: str | None = None

        self._build_layout()
        self._bind_shortcuts()
        self._restore_ui_state()
        self._seed_pipeline()
        self._refresh_pipeline_views()
        self._refresh_viewer_mode_options()
        self._refresh_viewer()

    def _seed_pipeline(self) -> None:
        if not self.session.stage2_blocks:
            for code in ("DECON", "SATUR", "CURVE"):
                self.session.stage2_blocks.append(BlockInstance(type=code, params=dict(BLOCK_DEFINITIONS[code].defaults)))

    def _build_layout(self) -> None:
        self._build_toolbar()
        self.main = ttk.Panedwindow(self.root, orient="horizontal")
        self.main.pack(fill="both", expand=True)

        left = ttk.Frame(self.main)
        center = ttk.Frame(self.main)
        right = ttk.Frame(self.main)
        self.main.add(left, weight=2)
        self.main.add(center, weight=6)
        self.main.add(right, weight=3)

        self._build_library(left)
        self._build_center(center)
        self._build_inspector(right)

    def _build_toolbar(self) -> None:
        bar = ttk.Frame(self.root)
        bar.pack(fill="x", padx=6, pady=4)
        for label, cmd in [
            ("Open", self._open_image),
            ("Save", self._save_project),
            ("Export", self._export_image),
            ("Undo", self._undo),
            ("Redo", self._redo),
            ("Run stage 1", self._run_stage1),
            ("Run stage 2", self._run_stage2),
            ("Run pipeline", self._run_pipeline),
        ]:
            ttk.Button(bar, text=label, command=cmd).pack(side="left", padx=2)
        ttk.Checkbutton(bar, text="Live Preview", variable=self.live_preview).pack(side="left", padx=8)
        ttk.Label(bar, textvariable=self.status).pack(side="right")

    def _build_library(self, parent: ttk.Frame) -> None:
        ttk.Label(parent, text="Building Blocks Library").pack(anchor="w", padx=4, pady=4)
        self.library = tk.Listbox(parent, exportselection=False)
        self.library.pack(fill="both", expand=True, padx=4, pady=4)
        for code in sorted(BLOCK_DEFINITIONS):
            self.library.insert("end", code)
        ttk.Button(parent, text="Add to selected stage", command=self._add_library_block).pack(fill="x", padx=4, pady=4)

    def _build_center(self, parent: ttk.Frame) -> None:
        viewer = ttk.LabelFrame(parent, text="Viewer")
        viewer.pack(fill="both", expand=True, padx=4, pady=4)
        row = ttk.Frame(viewer)
        row.pack(fill="x", padx=4, pady=4)
        ttk.Label(row, text="Mode").pack(side="left")
        self.viewer_combo = ttk.Combobox(row, textvariable=self.viewer_mode, state="readonly", width=20)
        self.viewer_combo.pack(side="left", padx=4)
        self.viewer_combo.bind("<<ComboboxSelected>>", lambda _e: self._refresh_viewer())
        ttk.Button(row, text="Fit", command=self._fit_viewer).pack(side="left", padx=4)

        self.viewer_canvas = tk.Canvas(viewer, background="#111", height=380)
        self.viewer_canvas.pack(fill="both", expand=True, padx=4, pady=4)

        pipeline = ttk.LabelFrame(parent, text="Pipeline Canvas")
        pipeline.pack(fill="both", expand=True, padx=4, pady=4)

        s1_header = ttk.Label(pipeline, text="Stage 1 — Preprocessing", font=("TkDefaultFont", 10, "bold"))
        s1_header.pack(fill="x", padx=4, pady=(4, 2))
        self.stage1_tree = self._make_stage_tree(pipeline)

        divider = ttk.Separator(pipeline, orient="horizontal")
        divider.pack(fill="x", padx=4, pady=8)

        s2_header = ttk.Label(pipeline, text="Stage 2 — Enhancement", font=("TkDefaultFont", 10, "bold"))
        s2_header.pack(fill="x", padx=4, pady=(2, 2))
        self.stage2_tree = self._make_stage_tree(pipeline)

        for tree, stage in ((self.stage1_tree, 1), (self.stage2_tree, 2)):
            tree.bind("<<TreeviewSelect>>", lambda _e, s=stage: self._on_select(s))
            tree.bind("<Button-3>", lambda e, s=stage: self._show_context_menu(e, s))

        controls = ttk.Frame(pipeline)
        controls.pack(fill="x", padx=4, pady=4)
        for label, cmd in [
            ("▲", self._move_up),
            ("▼", self._move_down),
            ("Enable/Disable", self._toggle_enabled),
            ("Duplicate", self._duplicate_selected),
            ("Delete", self._delete_selected),
            ("Apply", self._run_pipeline),
        ]:
            ttk.Button(controls, text=label, command=cmd).pack(side="left", padx=2)

    def _make_stage_tree(self, parent: ttk.Widget) -> ttk.Treeview:
        tree = ttk.Treeview(parent, columns=("enabled", "name"), show="headings", selectmode="browse", height=6)
        tree.heading("enabled", text="On")
        tree.heading("name", text="Block")
        tree.column("enabled", width=45, anchor="center")
        tree.column("name", width=300, anchor="w")
        tree.pack(fill="x", padx=4)
        return tree

    def _build_inspector(self, parent: ttk.Frame) -> None:
        inspector = ttk.LabelFrame(parent, text="Block parameters")
        inspector.pack(fill="both", expand=True, padx=4, pady=4)
        self.param_area = ttk.Frame(inspector)
        self.param_area.pack(fill="both", expand=True, padx=4, pady=4)

    def _bind_shortcuts(self) -> None:
        self.root.bind("<Control-z>", lambda *_: self._undo())
        self.root.bind("<Control-y>", lambda *_: self._redo())

    def _active_stage_blocks(self) -> list[BlockInstance]:
        return self.session.stage1_blocks if self.selected_stage.get() == 1 else self.session.stage2_blocks

    def _refresh_pipeline_views(self) -> None:
        for tree, blocks in ((self.stage1_tree, self.session.stage1_blocks), (self.stage2_tree, self.session.stage2_blocks)):
            tree.delete(*tree.get_children())
            for block in blocks:
                tree.insert("", "end", iid=block.id, values=("☑" if block.enabled else "☐", block.type))

        available_ids = {b.id for b in (self.session.stage1_blocks + self.session.stage2_blocks)}
        if self.selected_block_id not in available_ids:
            self.selected_block_id = None
        if self.selected_block_id:
            for tree in (self.stage1_tree, self.stage2_tree):
                if self.selected_block_id in tree.get_children():
                    tree.selection_set(self.selected_block_id)

        self._render_param_editor()

    def _on_select(self, stage: int) -> None:
        tree = self.stage1_tree if stage == 1 else self.stage2_tree
        sel = tree.selection()
        if not sel:
            return
        self.selected_stage.set(stage)
        self.session.selected_stage = stage
        self.selected_block_id = sel[0]
        other = self.stage2_tree if stage == 1 else self.stage1_tree
        other.selection_remove(other.selection())
        self._render_param_editor()

    def _selected_block(self) -> BlockInstance | None:
        for block in self.session.stage1_blocks + self.session.stage2_blocks:
            if block.id == self.selected_block_id:
                return block
        return None

    def _render_param_editor(self) -> None:
        for child in self.param_area.winfo_children():
            child.destroy()
        block = self._selected_block()
        if not block:
            ttk.Label(self.param_area, text="Select a block.").pack(anchor="w")
            return
        ttk.Label(self.param_area, text=f"{block.type}", font=("TkDefaultFont", 10, "bold")).pack(anchor="w", pady=(0, 4))

        for key, value in block.params.items():
            if isinstance(value, bool):
                var = tk.BooleanVar(value=value)
                ttk.Checkbutton(self.param_area, text=key, variable=var, command=lambda k=key, v=var: self._set_param(k, v.get())).pack(anchor="w")
            elif isinstance(value, (int, float)):
                row = ttk.Frame(self.param_area)
                row.pack(fill="x", pady=2)
                ttk.Label(row, text=key, width=18).pack(side="left")
                var = tk.DoubleVar(value=float(value))
                scale = ttk.Scale(row, from_=0, to=max(2.0, float(value) * 2 + 1), variable=var, command=lambda _v, k=key, v=var: self._set_param(k, v.get()))
                scale.pack(side="left", fill="x", expand=True, padx=4)

    def _set_param(self, key: str, value: Any) -> None:
        block = self._selected_block()
        if not block:
            return
        old = block.params.get(key)
        new = round(float(value), 4) if isinstance(value, (float, int)) else value
        if old == new:
            return

        def do() -> None:
            block.params[key] = new
            self._invalidate_cache()
            self._render_param_editor()
            self._auto_preview()

        def undo() -> None:
            block.params[key] = old
            self._invalidate_cache()
            self._render_param_editor()
            self._auto_preview()

        self.history.execute(Command(do=do, undo=undo, description=f"Param {key}"))

    def _add_library_block(self) -> None:
        sel = self.library.curselection()
        if not sel:
            return
        code = self.library.get(sel[0])
        blocks = self._active_stage_blocks()
        block = BlockInstance(type=code, params=dict(BLOCK_DEFINITIONS[code].defaults))

        def do() -> None:
            blocks.append(block)
            self.selected_block_id = block.id
            self._invalidate_cache()
            self._refresh_pipeline_views()

        def undo() -> None:
            blocks.remove(block)
            self.selected_block_id = None
            self._invalidate_cache()
            self._refresh_pipeline_views()

        self.history.execute(Command(do=do, undo=undo, description="Add block"))

    def _toggle_enabled(self) -> None:
        block = self._selected_block()
        if not block:
            return
        old = block.enabled

        def do() -> None:
            block.enabled = not old
            self._invalidate_cache()
            self._refresh_pipeline_views()
            self._auto_preview()

        def undo() -> None:
            block.enabled = old
            self._invalidate_cache()
            self._refresh_pipeline_views()
            self._auto_preview()

        self.history.execute(Command(do=do, undo=undo, description="Toggle"))

    def _move_up(self) -> None:
        self._move_selected(-1)

    def _move_down(self) -> None:
        self._move_selected(1)

    def _move_selected(self, delta: int) -> None:
        blocks = self._active_stage_blocks()
        block = self._selected_block()
        if not block or block not in blocks:
            return
        src = blocks.index(block)
        dst = src + delta
        if dst < 0 or dst >= len(blocks):
            return

        def do() -> None:
            blocks[src], blocks[dst] = blocks[dst], blocks[src]
            self._invalidate_cache()
            self._refresh_pipeline_views()
            self._auto_preview()

        def undo() -> None:
            blocks[src], blocks[dst] = blocks[dst], blocks[src]
            self._invalidate_cache()
            self._refresh_pipeline_views()
            self._auto_preview()

        self.history.execute(Command(do=do, undo=undo, description="Reorder"))

    def _duplicate_selected(self) -> None:
        block = self._selected_block()
        if not block:
            return
        blocks = self._active_stage_blocks()
        clone = BlockInstance(type=block.type, enabled=block.enabled, params=copy.deepcopy(block.params))

        def do() -> None:
            blocks.append(clone)
            self.selected_block_id = clone.id
            self._invalidate_cache()
            self._refresh_pipeline_views()

        def undo() -> None:
            blocks.remove(clone)
            self.selected_block_id = block.id
            self._invalidate_cache()
            self._refresh_pipeline_views()

        self.history.execute(Command(do=do, undo=undo, description="Duplicate"))

    def _delete_selected(self) -> None:
        block = self._selected_block()
        if not block:
            return
        blocks = self.session.stage1_blocks if block in self.session.stage1_blocks else self.session.stage2_blocks
        idx = blocks.index(block)

        def do() -> None:
            blocks.remove(block)
            self.selected_block_id = None
            self._invalidate_cache()
            self._refresh_pipeline_views()

        def undo() -> None:
            blocks.insert(idx, block)
            self.selected_block_id = block.id
            self._invalidate_cache()
            self._refresh_pipeline_views()

        self.history.execute(Command(do=do, undo=undo, description="Delete"))

    def _show_context_menu(self, event: tk.Event, stage: int) -> None:
        tree = self.stage1_tree if stage == 1 else self.stage2_tree
        row = tree.identify_row(event.y)
        if not row:
            return
        tree.selection_set(row)
        self._on_select(stage)
        menu = tk.Menu(self.root, tearoff=0)
        menu.add_command(label="Move to stage 1", command=lambda: self._move_block_to_stage(1))
        menu.add_command(label="Move to stage 2", command=lambda: self._move_block_to_stage(2))
        menu.tk_popup(event.x_root, event.y_root)

    def _move_block_to_stage(self, target_stage: int) -> None:
        block = self._selected_block()
        if not block:
            return
        src_blocks = self.session.stage1_blocks if block in self.session.stage1_blocks else self.session.stage2_blocks
        dst_blocks = self.session.stage1_blocks if target_stage == 1 else self.session.stage2_blocks
        if src_blocks is dst_blocks:
            return
        src_idx = src_blocks.index(block)

        def do() -> None:
            src_blocks.remove(block)
            dst_blocks.append(block)
            self.selected_stage.set(target_stage)
            self.selected_block_id = block.id
            self._invalidate_cache()
            self._refresh_pipeline_views()

        def undo() -> None:
            dst_blocks.remove(block)
            src_blocks.insert(src_idx, block)
            self.selected_stage.set(1 if src_blocks is self.session.stage1_blocks else 2)
            self.selected_block_id = block.id
            self._invalidate_cache()
            self._refresh_pipeline_views()

        self.history.execute(Command(do=do, undo=undo, description="Move stage"))

    def _run_blocks(self, blocks: list[BlockInstance], input_image: dict[str, Any], stage_name: str) -> dict[str, Any]:
        signal = sum((idx + 1) * 0.01 for idx, b in enumerate(blocks) if b.enabled)
        output = dict(input_image)
        output["signal"] = round(signal, 4)
        output["stage"] = stage_name
        output["blocks"] = [b.type for b in blocks if b.enabled]
        return output

    def _run_stage1(self) -> None:
        if not self.loaded_image:
            return
        self.stage_cache["stage1"] = self._run_blocks(self.session.stage1_blocks, self.loaded_image, "stage1")
        self.stage_cache["full"] = None
        self._refresh_viewer_mode_options()
        self.status.set("Stage 1 complete")
        self._refresh_viewer()

    def _run_stage2(self) -> None:
        if not self.loaded_image:
            return
        stage2_input = self.stage_cache["stage1"] or self.loaded_image
        self.stage_cache["stage2"] = self._run_blocks(self.session.stage2_blocks, stage2_input, "stage2")
        self.stage_cache["full"] = self.stage_cache["stage2"]
        self._refresh_viewer_mode_options()
        self.status.set("Stage 2 complete")
        self._refresh_viewer()

    def _run_pipeline(self) -> None:
        if not self.loaded_image:
            return
        self._run_stage1()
        self._run_stage2()
        self.status.set("Pipeline complete")

    def _source_for_mode(self, mode: str) -> dict[str, Any] | None:
        if mode == "Before":
            return self.loaded_image
        if mode == "Stage 1 output":
            return self.stage_cache["stage1"]
        if mode == "Stage 2 output":
            return self.stage_cache["stage2"]
        if mode == "Full pipeline":
            return self.stage_cache["full"]
        return self.loaded_image

    def _fit_viewer(self) -> None:
        self.session.viewer_state.zoom = 1.0
        self.status.set("Viewer fit")
        self._refresh_viewer()

    def _refresh_viewer(self) -> None:
        self.viewer_canvas.delete("all")
        source = self._source_for_mode(self.viewer_mode.get())
        if not source:
            self.viewer_canvas.create_text(420, 180, fill="white", text="No image loaded")
            return
        label = Path(source.get("path", "")).name or "image"
        mode = self.viewer_mode.get()
        self.viewer_canvas.create_rectangle(80, 50, 760, 320, outline="#6bc1ff", width=2)
        self.viewer_canvas.create_text(420, 140, fill="white", text=f"{label}")
        self.viewer_canvas.create_text(420, 180, fill="#90ee90", text=f"Mode: {mode}")
        self.viewer_canvas.create_text(420, 220, fill="#cccccc", text=f"Signal: {source.get('signal', 0.0)}")
        self.viewer_canvas.update_idletasks()

    def _refresh_viewer_mode_options(self) -> None:
        modes = ["Before"]
        if self.stage_cache["stage1"] is not None:
            modes.append("Stage 1 output")
        if self.stage_cache["stage2"] is not None:
            modes.append("Stage 2 output")
        if self.stage_cache["full"] is not None:
            modes.append("Full pipeline")
        modes.extend(["Split", "Blink"])
        self.viewer_combo["values"] = modes
        if self.viewer_mode.get() not in modes:
            self.viewer_mode.set("Before")

    def _invalidate_cache(self) -> None:
        self.stage_cache = {"stage1": None, "stage2": None, "full": None}
        self._refresh_viewer_mode_options()

    def _auto_preview(self) -> None:
        if self.live_preview.get() and self.loaded_image:
            self._run_pipeline()

    def _open_image(self) -> None:
        path = filedialog.askopenfilename(
            title="Open image",
            filetypes=[("Image", "*.png *.bmp *.tif *.tiff *.jpg *.jpeg *.xisf *.fits")],
        )
        if not path:
            return
        try:
            image = read_image(path)
        except ValueError as exc:
            messagebox.showerror("PlanetSharp", str(exc))
            return

        self.loaded_image = image
        self.loaded_image_path = path
        self._invalidate_cache()
        self.viewer_mode.set("Before")
        self._fit_viewer()
        self.status.set(f"Loaded {Path(path).name}")

        if self.live_preview.get():
            self._run_pipeline()
        else:
            self._refresh_viewer()

        self.history.clear()

    def _save_project(self) -> None:
        path = self.current_project_path
        if not path:
            path = filedialog.asksaveasfilename(
                title="Save project",
                defaultextension=PROJECT_SUFFIX,
                filetypes=[("PlanetSharp Project", f"*{PROJECT_SUFFIX}")],
            )
        if not path:
            return
        self.current_project_path = path

        self.session.stage1_blocks = self.session.stage1_blocks
        self.session.stage2_blocks = self.session.stage2_blocks
        self.session.stage2_workflow.blocks = list(self.session.stage2_blocks)
        self.session.viewer_state.stage_display = self.viewer_mode.get()
        self.session.viewer_state.zoom = 1.0
        if self.loaded_image_path:
            self.session.inputs = [InputImage(path=self.loaded_image_path, role=Role.FILTER)]

        SessionStore.save(path, self.session)
        self.status.set(f"Saved project: {Path(path).name}")

    def _export_image(self) -> None:
        if not self.loaded_image:
            return
        target = simpledialog.askstring("Export", "Target (stage1/stage2/full):", initialvalue="full", parent=self.root)
        if not target:
            return
        target = target.lower().strip()

        if target == "stage1" and self.stage_cache["stage1"] is None:
            self._run_stage1()
        elif target in {"stage2", "full"} and self.stage_cache["stage2"] is None:
            self._run_pipeline()

        source = {
            "stage1": self.stage_cache["stage1"],
            "stage2": self.stage_cache["stage2"],
            "full": self.stage_cache["full"],
        }.get(target)
        if source is None:
            messagebox.showerror("PlanetSharp", "Requested output unavailable")
            return

        path = filedialog.asksaveasfilename(
            title="Export image",
            defaultextension=".png",
            filetypes=[("PNG", "*.png"), ("TIFF", "*.tiff"), ("JPG", "*.jpg")],
        )
        if not path:
            return
        write_image(path, source, bit_depth=16)
        self.status.set(f"Exported {Path(path).name}")

    def _undo(self) -> None:
        if self.history.undo():
            self.status.set("Undo")
            self._refresh_pipeline_views()
            self._refresh_viewer()

    def _redo(self) -> None:
        if self.history.redo():
            self.status.set("Redo")
            self._refresh_pipeline_views()
            self._refresh_viewer()

    def _save_ui_state(self) -> None:
        state = {"viewer_mode": self.viewer_mode.get(), "geometry": self.root.geometry()}
        LAYOUT_STATE.write_text(json.dumps(state, indent=2), encoding="utf-8")

    def _restore_ui_state(self) -> None:
        if not LAYOUT_STATE.exists():
            return
        try:
            state = json.loads(LAYOUT_STATE.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return
        self.viewer_mode.set(state.get("viewer_mode", "Before"))
        if geometry := state.get("geometry"):
            self.root.geometry(geometry)

    def _on_close(self) -> None:
        self._save_ui_state()
        self.root.destroy()

    def run(self) -> None:
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        self.root.mainloop()


def main() -> None:
    PlanetSharpApp().run()


if __name__ == "__main__":
    main()
