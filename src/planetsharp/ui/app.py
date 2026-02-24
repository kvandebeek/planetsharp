from __future__ import annotations

import copy
import hashlib
import json
import math
import threading
import tkinter as tk
from dataclasses import dataclass
from pathlib import Path
from tkinter import TclError, filedialog, messagebox, ttk
from typing import Any, Callable

from planetsharp.core.models import BlockInstance, Session
from planetsharp.io.formats import read_image, write_image
from planetsharp.persistence.template_store import TEMPLATE_SUFFIX, TemplateStore
from planetsharp.processing.blocks import BLOCK_DEFINITIONS

LAYOUT_STATE = Path.home() / ".planetsharp.layout.json"


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
        self.selected_stage = tk.IntVar(value=1)

        self.loaded_image: dict[str, Any] | None = None
        self.loaded_image_path: str | None = None
        self.processed_image: dict[str, Any] | None = None
        self.selected_block_id: str | None = None
        self.history = History()

        self._last_pipeline_sigs: list[str] = []
        self._block_cache: dict[str, dict[str, Any]] = {}
        self._debounce_id: str | None = None
        self._generation = 0
        self._processing_thread: threading.Thread | None = None
        self._drag_from_library: str | None = None
        self._drag_block_id: str | None = None
        self._drag_source_stage: int | None = None
        self._drag_insert_row: str | None = None
        self._drag_insert_after = False
        self._viewer_source_image: tk.PhotoImage | None = None
        self._viewer_display_image: tk.PhotoImage | None = None

        self._build_layout()
        self._bind_shortcuts()
        self._restore_ui_state()
        self._seed_pipeline()
        self._refresh_pipeline_views()
        self._refresh_viewer()

    def _seed_pipeline(self) -> None:
        if not self.session.stage2_blocks:
            for code in ("DECON", "SATUR", "CURVE"):
                self.session.stage2_blocks.append(BlockInstance(type=code, params=dict(BLOCK_DEFINITIONS[code].defaults)))

    def _build_layout(self) -> None:
        bar = ttk.Frame(self.root)
        bar.pack(fill="x", padx=6, pady=4)
        for label, cmd in [
            ("Open", self._open_any),
            ("Save", self._save_template),
            ("Export", self._export_image),
            ("Undo", self._undo),
            ("Redo", self._redo),
        ]:
            ttk.Button(bar, text=label, command=cmd).pack(side="left", padx=2)
        ttk.Label(bar, textvariable=self.status).pack(side="right")

        self.main = ttk.Panedwindow(self.root, orient="vertical")
        self.main.pack(fill="both", expand=True)

        viewer_row = ttk.Frame(self.main)
        workflow_row = ttk.Frame(self.main)
        self.main.add(viewer_row, weight=6)
        self.main.add(workflow_row, weight=3)

        viewer = ttk.LabelFrame(viewer_row, text="Viewer")
        viewer.pack(fill="both", expand=True, padx=4, pady=4)
        row = ttk.Frame(viewer)
        row.pack(fill="x", padx=4, pady=4)
        ttk.Label(row, text="Mode").pack(side="left")
        self.viewer_combo = ttk.Combobox(row, textvariable=self.viewer_mode, state="readonly", values=["Before", "After"], width=20)
        self.viewer_combo.pack(side="left", padx=4)
        self.viewer_combo.bind("<<ComboboxSelected>>", lambda _e: self._refresh_viewer())
        ttk.Button(row, text="Fit", command=self._refresh_viewer).pack(side="left", padx=4)
        self.viewer_canvas = tk.Canvas(viewer, background="#111")
        self.viewer_canvas.pack(fill="both", expand=True, padx=4, pady=4)
        self.viewer_canvas.bind("<Configure>", lambda _e: self._refresh_viewer())

        workflow = ttk.Panedwindow(workflow_row, orient="horizontal")
        workflow.pack(fill="both", expand=True)
        left = ttk.LabelFrame(workflow, text="Building Blocks Library")
        center = ttk.LabelFrame(workflow, text="Pipeline Canvas")
        right = ttk.LabelFrame(workflow, text="Block Parameters")
        workflow.add(left, weight=2)
        workflow.add(center, weight=4)
        workflow.add(right, weight=2)

        self.library = tk.Listbox(left, exportselection=False)
        self.library.pack(fill="both", expand=True, padx=4, pady=4)
        for code in sorted(BLOCK_DEFINITIONS):
            name = BLOCK_DEFINITIONS[code].display_name
            self.library.insert("end", f"{name} ({code})")
        self.library.bind("<ButtonPress-1>", self._start_library_drag)

        ttk.Label(center, text="Stage 1 — Preprocessing", font=("TkDefaultFont", 10, "bold")).pack(fill="x", padx=4, pady=(4, 2))
        self.stage1_frame = tk.Frame(center, highlightthickness=2, highlightbackground="#555")
        self.stage1_frame.pack(fill="both", expand=True, padx=4)
        self.stage1_tree = self._make_stage_tree(self.stage1_frame)

        ttk.Separator(center, orient="horizontal").pack(fill="x", padx=4, pady=8)
        ttk.Label(center, text="Stage 2 — Enhancement", font=("TkDefaultFont", 10, "bold")).pack(fill="x", padx=4, pady=(2, 2))
        self.stage2_frame = tk.Frame(center, highlightthickness=2, highlightbackground="#555")
        self.stage2_frame.pack(fill="both", expand=True, padx=4, pady=(0, 4))
        self.stage2_tree = self._make_stage_tree(self.stage2_frame)
        self.stage1_tree.tag_configure("insert_marker", background="#264f78")
        self.stage2_tree.tag_configure("insert_marker", background="#264f78")

        for tree, stage in ((self.stage1_tree, 1), (self.stage2_tree, 2)):
            tree.bind("<<TreeviewSelect>>", lambda _e, s=stage: self._on_select(s))
            tree.bind("<ButtonPress-1>", lambda e, s=stage: self._start_tree_drag(e, s))
            tree.bind("<B1-Motion>", self._drag_motion)
            tree.bind("<ButtonRelease-1>", self._drop_on_tree)

        controls = ttk.Frame(center)
        controls.pack(fill="x", padx=4, pady=4)
        for label, cmd in [("▲", self._move_up), ("▼", self._move_down), ("Enable/Disable", self._toggle_enabled), ("Duplicate", self._duplicate_selected), ("Delete", self._delete_selected)]:
            ttk.Button(controls, text=label, command=cmd).pack(side="left", padx=2)

        self.param_area = ttk.Frame(right)
        self.param_area.pack(fill="both", expand=True, padx=4, pady=4)

    def _make_stage_tree(self, parent: tk.Widget) -> ttk.Treeview:
        tree = ttk.Treeview(parent, columns=("enabled", "name"), show="headings", selectmode="browse", height=6)
        tree.heading("enabled", text="Enabled")
        tree.heading("name", text="Block")
        tree.column("enabled", width=80, anchor="center")
        tree.column("name", width=320, anchor="w")
        tree.pack(fill="both", expand=True, padx=4, pady=2)
        return tree

    def _bind_shortcuts(self) -> None:
        self.root.bind("<Control-z>", lambda *_: self._undo())
        self.root.bind("<Control-y>", lambda *_: self._redo())

    def _pipeline_blocks(self) -> list[BlockInstance]:
        return self.session.stage1_blocks + self.session.stage2_blocks

    def _refresh_pipeline_views(self) -> None:
        for tree, blocks in ((self.stage1_tree, self.session.stage1_blocks), (self.stage2_tree, self.session.stage2_blocks)):
            tree.delete(*tree.get_children())
            for block in blocks:
                name = BLOCK_DEFINITIONS[block.type].display_name
                tree.insert("", "end", iid=block.id, values=("☑" if block.enabled else "☐", f"{name} ({block.type})"))
        self._render_param_editor()

    def _on_select(self, stage: int) -> None:
        tree = self.stage1_tree if stage == 1 else self.stage2_tree
        sel = tree.selection()
        if not sel:
            return
        self.selected_stage.set(stage)
        self.selected_block_id = sel[0]
        self._render_param_editor()

    def _selected_block(self) -> BlockInstance | None:
        for block in self._pipeline_blocks():
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
        ttk.Label(self.param_area, text=f"{BLOCK_DEFINITIONS[block.type].display_name} ({block.type})", font=("TkDefaultFont", 10, "bold")).pack(anchor="w", pady=(0, 4))
        for key, value in block.params.items():
            if isinstance(value, (int, float)):
                row = ttk.Frame(self.param_area)
                row.pack(fill="x", pady=2)
                ttk.Label(row, text=key, width=18).pack(side="left")
                var = tk.DoubleVar(value=float(value))
                scale = ttk.Scale(row, from_=0, to=max(2.0, float(value) * 2 + 1), variable=var, command=lambda _v, k=key, v=var: self._set_param(k, v.get()))
                scale.pack(side="left", fill="x", expand=True, padx=4)

    def _active_stage_blocks(self) -> list[BlockInstance] | None:
        stage = self.selected_stage.get()
        if stage == 1:
            return self.session.stage1_blocks
        if stage == 2:
            return self.session.stage2_blocks
        return None

    def _set_param(self, key: str, value: Any) -> None:
        block = self._selected_block()
        if not block:
            return
        old = block.params.get(key)
        new = round(float(value), 4)
        if old == new:
            return

        def do() -> None:
            block.params[key] = new
            self._schedule_processing("parameter")

        def undo() -> None:
            block.params[key] = old
            self._schedule_processing("parameter")

        self.history.execute(Command(do=do, undo=undo, description="Param"))

    def _code_from_library_selection(self) -> str | None:
        sel = self.library.curselection()
        if not sel:
            return None
        text = self.library.get(sel[0])
        return text.rsplit("(", 1)[-1].rstrip(")")

    def _add_library_block(self) -> None:
        code = self._code_from_library_selection()
        if not code:
            return
        blocks = self._active_stage_blocks()
        if blocks is None:
            messagebox.showinfo("PlanetSharp", "Select Stage 1 or Stage 2 before adding a block.")
            return
        block = BlockInstance(type=code, params=dict(BLOCK_DEFINITIONS[code].defaults))

        def do() -> None:
            blocks.append(block)
            self.selected_block_id = block.id
            self._refresh_pipeline_views()
            self._schedule_processing("add")

        def undo() -> None:
            blocks.remove(block)
            self.selected_block_id = None
            self._refresh_pipeline_views()
            self._schedule_processing("remove")

        self.history.execute(Command(do=do, undo=undo, description="Add"))

    def _toggle_enabled(self) -> None:
        block = self._selected_block()
        if not block:
            return
        old = block.enabled

        def do() -> None:
            block.enabled = not old
            self._refresh_pipeline_views()
            self._schedule_processing("toggle")

        def undo() -> None:
            block.enabled = old
            self._refresh_pipeline_views()
            self._schedule_processing("toggle")

        self.history.execute(Command(do=do, undo=undo, description="Toggle"))

    def _move_up(self) -> None:
        self._move_selected(-1)

    def _move_down(self) -> None:
        self._move_selected(1)

    def _move_selected(self, delta: int) -> None:
        block = self._selected_block()
        blocks = self._active_stage_blocks()
        if not block or blocks is None or block not in blocks:
            return
        src = blocks.index(block)
        dst = src + delta
        if dst < 0 or dst >= len(blocks):
            return

        def do() -> None:
            blocks[src], blocks[dst] = blocks[dst], blocks[src]
            self._refresh_pipeline_views()
            self._schedule_processing("reorder")

        self.history.execute(Command(do=do, undo=do, description="Reorder"))

    def _duplicate_selected(self) -> None:
        block = self._selected_block()
        blocks = self._active_stage_blocks()
        if not block or blocks is None:
            return
        clone = BlockInstance(type=block.type, enabled=block.enabled, params=copy.deepcopy(block.params))

        def do() -> None:
            blocks.append(clone)
            self.selected_block_id = clone.id
            self._refresh_pipeline_views()
            self._schedule_processing("duplicate")

        def undo() -> None:
            blocks.remove(clone)
            self._refresh_pipeline_views()
            self._schedule_processing("duplicate")

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
            self._refresh_pipeline_views()
            self._schedule_processing("delete")

        def undo() -> None:
            blocks.insert(idx, block)
            self.selected_block_id = block.id
            self._refresh_pipeline_views()
            self._schedule_processing("delete")

        self.history.execute(Command(do=do, undo=undo, description="Delete"))

    def _start_library_drag(self, _event: tk.Event) -> None:
        self._drag_from_library = self._code_from_library_selection()
        if self._drag_from_library:
            self.root.configure(cursor="hand2")

    def _start_tree_drag(self, event: tk.Event, stage: int) -> None:
        tree = self.stage1_tree if stage == 1 else self.stage2_tree
        row = tree.identify_row(event.y)
        self._drag_block_id = row or None
        self._drag_source_stage = stage if row else None
        if row:
            self.root.configure(cursor="fleur")

    def _tree_under_pointer(self, x_root: int, y_root: int) -> tuple[ttk.Treeview | None, int | None]:
        widget = self.root.winfo_containing(x_root, y_root)
        for tree, stage in ((self.stage1_tree, 1), (self.stage2_tree, 2)):
            if widget is tree or (widget and str(widget).startswith(str(tree))):
                return tree, stage
        return None, None

    def _clear_drag_visuals(self) -> None:
        for frame in (self.stage1_frame, self.stage2_frame):
            frame.configure(highlightbackground="#555")
        for tree in (self.stage1_tree, self.stage2_tree):
            for iid in tree.get_children():
                tree.item(iid, tags=())

    def _drag_motion(self, event: tk.Event) -> None:
        if not (self._drag_from_library or self._drag_block_id):
            return
        tree, stage = self._tree_under_pointer(event.x_root, event.y_root)
        self._drag_insert_row = None
        self._drag_insert_after = False
        self._clear_drag_visuals()
        if not tree or not stage:
            return
        (self.stage1_frame if stage == 1 else self.stage2_frame).configure(highlightbackground="#4a90e2")
        y = event.y_root - tree.winfo_rooty()
        row = tree.identify_row(y)
        if not row:
            return
        self._drag_insert_row = row
        bbox = tree.bbox(row)
        if bbox:
            self._drag_insert_after = y > (bbox[1] + bbox[3] / 2)
        tree.item(row, tags=("insert_marker",))

    def _drop_on_tree(self, event: tk.Event) -> None:
        tree, stage = self._tree_under_pointer(event.x_root, event.y_root)
        self.root.configure(cursor="")
        if not tree or not stage:
            self._drag_from_library = None
            self._drag_block_id = None
            self._drag_source_stage = None
            self._clear_drag_visuals()
            return
        target_blocks = self.session.stage1_blocks if stage == 1 else self.session.stage2_blocks
        target_row = self._drag_insert_row
        insert_index = len(target_blocks)
        if target_row:
            insert_index = target_blocks.index(next(b for b in target_blocks if b.id == target_row)) + (1 if self._drag_insert_after else 0)
        if self._drag_from_library:
            block = BlockInstance(type=self._drag_from_library, params=dict(BLOCK_DEFINITIONS[self._drag_from_library].defaults))
            target_blocks.insert(insert_index, block)
            self._drag_from_library = None
            self.selected_stage.set(stage)
            self.selected_block_id = block.id
            self._refresh_pipeline_views()
            self._schedule_processing("drag-insert")
            self._clear_drag_visuals()
            return
        if not self._drag_block_id or not self._drag_source_stage:
            self._clear_drag_visuals()
            return
        src_blocks = self.session.stage1_blocks if self._drag_source_stage == 1 else self.session.stage2_blocks
        block = next((candidate for candidate in src_blocks if candidate.id == self._drag_block_id), None)
        if block is None:
            self._drag_block_id = None
            self._drag_source_stage = None
            self._clear_drag_visuals()
            return

        src_index = src_blocks.index(block)
        src_blocks.remove(block)
        if src_blocks is target_blocks and insert_index > src_index:
            insert_index -= 1
        target_blocks.insert(min(insert_index, len(target_blocks)), block)

        self.selected_stage.set(stage)
        self.selected_block_id = block.id
        self._drag_block_id = None
        self._drag_source_stage = None
        self._clear_drag_visuals()
        self._refresh_pipeline_views()
        self._schedule_processing("drag-move")

    def _block_signature(self, block: BlockInstance) -> str:
        payload = {"type": block.type, "enabled": block.enabled, "params": block.params}
        return json.dumps(payload, sort_keys=True, default=str)

    def _schedule_processing(self, reason: str) -> None:
        if self._debounce_id:
            self.root.after_cancel(self._debounce_id)
        self._debounce_id = self.root.after(100, lambda: self._start_processing(reason))

    def _start_processing(self, reason: str) -> None:
        if not self.loaded_image:
            self.processed_image = None
            self._refresh_viewer()
            return
        self._generation += 1
        generation = self._generation
        signatures = [self._block_signature(b) for b in self._pipeline_blocks()]
        start = 0
        for idx, (old, new) in enumerate(zip(self._last_pipeline_sigs, signatures)):
            if old != new:
                start = idx
                break
            start = idx + 1
        start = min(start, len(signatures))
        self.status.set(f"Processing… ({reason})")
        print(f"[pipeline] recompute_start={start}")

        def worker() -> None:
            upstream = self.loaded_image
            upstream_key = hashlib.sha1((self.loaded_image.get("path", "") + str(self.loaded_image.get("format", ""))).encode()).hexdigest()
            for idx, block in enumerate(self._pipeline_blocks()):
                sig = signatures[idx]
                key = hashlib.sha1(f"{upstream_key}|{sig}".encode()).hexdigest()
                if idx < start and key in self._block_cache:
                    upstream = self._block_cache[key]
                    upstream_key = key
                    print(f"[pipeline] cache-hit idx={idx}")
                    continue
                print(f"[pipeline] cache-miss idx={idx}")
                if generation != self._generation:
                    return
                out = dict(upstream)
                out["signal"] = round(float(out.get("signal", 0.0)) + ((idx + 1) * 0.01 if block.enabled else 0.0), 4)
                out["last_block"] = block.type
                self._block_cache[key] = out
                upstream = out
                upstream_key = key
            if generation != self._generation:
                return
            self._last_pipeline_sigs = signatures
            self.root.after(0, lambda: self._finish_processing(upstream))

        self._processing_thread = threading.Thread(target=worker, daemon=True)
        self._processing_thread.start()

    def _finish_processing(self, output: dict[str, Any]) -> None:
        self.processed_image = output
        self.status.set("Ready")
        self._refresh_viewer()

    def _refresh_viewer(self) -> None:
        self.viewer_canvas.delete("all")
        canvas_w = max(self.viewer_canvas.winfo_width(), 1)
        canvas_h = max(self.viewer_canvas.winfo_height(), 1)
        if not self.loaded_image:
            self.viewer_canvas.create_text(canvas_w // 2, canvas_h // 2, fill="white", text="No image loaded")
            return
        source = self.loaded_image if self.viewer_mode.get() == "Before" else self.processed_image
        if source is None:
            self.viewer_canvas.create_text(canvas_w // 2, canvas_h // 2, fill="white", text="Processing…")
            return
        if not self._viewer_source_image:
            self.viewer_canvas.create_text(canvas_w // 2, canvas_h // 2, fill="white", text="Image unavailable")
            return
        self._viewer_display_image = self._fit_photo(self._viewer_source_image, canvas_w - 16, canvas_h - 16)
        self.viewer_canvas.create_image(canvas_w // 2, canvas_h // 2, image=self._viewer_display_image, anchor="center")
        mode = self.viewer_mode.get()
        signal = source.get("signal", 0.0)
        self.viewer_canvas.create_text(8, 8, anchor="nw", fill="#90ee90", text=f"{mode} • signal {signal}")

    def _open_any(self) -> None:
        path = filedialog.askopenfilename(
            title="Open image or template",
            filetypes=[("Image/Template", f"*.png *.bmp *.tif *.tiff *.jpg *.jpeg *.xisf *.fits *{TEMPLATE_SUFFIX}"), ("Template", f"*{TEMPLATE_SUFFIX}")],
        )
        if not path:
            return
        if path.endswith(TEMPLATE_SUFFIX):
            self._open_template(path)
        else:
            self._open_image(path)

    def _open_image(self, path: str) -> None:
        previous_meta = self.loaded_image
        previous_preview = self._viewer_source_image
        try:
            loaded = read_image(path)
            preview = tk.PhotoImage(file=path)
        except ValueError as exc:
            messagebox.showerror("PlanetSharp", str(exc))
            return
        except TclError as exc:
            self.loaded_image = previous_meta
            self._viewer_source_image = previous_preview
            messagebox.showerror("PlanetSharp", f"Unable to decode image: {exc}")
            return
        self.loaded_image = loaded
        self._viewer_source_image = preview
        self.loaded_image_path = path
        self.processed_image = None
        self.viewer_mode.set("Before")
        self.status.set(f"Loaded {Path(path).name}")
        self._refresh_viewer()
        self._schedule_processing("image-load")

    def _fit_photo(self, image: tk.PhotoImage, max_w: int, max_h: int) -> tk.PhotoImage:
        max_w = max(max_w, 1)
        max_h = max(max_h, 1)
        if image.width() <= max_w and image.height() <= max_h:
            return image
        factor = max(1, math.ceil(max(image.width() / max_w, image.height() / max_h)))
        return image.subsample(factor, factor)

    def _open_template(self, path: str) -> None:
        try:
            loaded = TemplateStore.load(path)
        except Exception as exc:
            messagebox.showerror("PlanetSharp", f"Template load failed: {exc}")
            return
        self.session.stage1_blocks = loaded["stage1"]
        self.session.stage2_blocks = loaded["stage2"]
        self.selected_block_id = None
        self._refresh_pipeline_views()
        self.status.set(f"Loaded template {Path(path).name}")
        self._schedule_processing("template-load")

    def _save_template(self) -> None:
        path = filedialog.asksaveasfilename(title="Save template", defaultextension=TEMPLATE_SUFFIX, filetypes=[("PlanetSharp Template", f"*{TEMPLATE_SUFFIX}")])
        if not path:
            return
        TemplateStore.save(path, self.session)
        self.status.set(f"Saved template {Path(path).name}")

    def _export_image(self) -> None:
        source = self.loaded_image if self.viewer_mode.get() == "Before" else self.processed_image
        if source is None:
            messagebox.showerror("PlanetSharp", "No image loaded")
            return
        path = filedialog.asksaveasfilename(title="Export image", defaultextension=".png", filetypes=[("PNG", "*.png"), ("TIFF", "*.tiff"), ("JPG", "*.jpg")])
        if not path:
            return
        write_image(path, source, bit_depth=16)
        self.status.set(f"Exported {Path(path).name} ({self.viewer_mode.get()})")

    def _undo(self) -> None:
        if self.history.undo():
            self._refresh_pipeline_views()
            self._schedule_processing("undo")

    def _redo(self) -> None:
        if self.history.redo():
            self._refresh_pipeline_views()
            self._schedule_processing("redo")

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
