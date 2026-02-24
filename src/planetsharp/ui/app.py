from __future__ import annotations

import copy
import hashlib
import json
import math
import os
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
STAGE1_CHANNELS = ("L", "R", "G", "B")
BLOCK_COLORS = {
    "DECON": "#a264ff",
    "AWAVE": "#a264ff",
    "RWAVE": "#a264ff",
    "UMASK": "#a264ff",
    "DERIN": "#a264ff",
    "COBAL": "#3f8efc",
    "WHBAL": "#3f8efc",
    "SELCO": "#3f8efc",
    "SATUR": "#3f8efc",
    "ALIGN": "#1abc9c",
    "GBLUR": "#f39c12",
    "BILAT": "#f39c12",
    "NOISE": "#f39c12",
    "LINST": "#27ae60",
    "LEVEL": "#27ae60",
    "CURVE": "#27ae60",
    "CONTR": "#27ae60",
}


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


class PlanetSharpApp:
    def __init__(self, session: Session | None = None):
        self.session = session or Session()
        self.root = tk.Tk()
        self.root.title("PlanetSharp v1")
        self.root.geometry("1400x900")

        self.status = tk.StringVar(value="Ready")
        self.viewer_mode = tk.StringVar(value="Before")
        self.selected_stage = tk.IntVar(value=1)
        self.selected_channel = tk.StringVar(value="L")

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
        self._viewer_source_image: tk.PhotoImage | None = None
        self._viewer_display_image: tk.PhotoImage | None = None
        self._cpu_history: list[float] = [0.0] * 20

        self._build_layout()
        self._bind_shortcuts()
        self._restore_ui_state()
        self._seed_pipeline()
        self._refresh_pipeline_views()
        self._refresh_viewer()
        self._tick_cpu_usage()

    def _seed_pipeline(self) -> None:
        if not self.session.stage2_blocks:
            for code in ("DECON", "SATUR", "CURVE"):
                self.session.stage2_blocks.append(BlockInstance(type=code, params=dict(BLOCK_DEFINITIONS[code].defaults)))

    def _build_layout(self) -> None:
        self.style = ttk.Style()
        self.style.configure("Treeview", rowheight=28)
        self.style.map("Treeview", background=[("selected", "#1e1e1e")], foreground=[("selected", "#ffffff")])

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
        ttk.Label(bar, text="CPU (20s)").pack(side="left", padx=(12, 4))
        self.cpu_canvas = tk.Canvas(bar, width=180, height=40, background="#111", highlightthickness=1, highlightbackground="#444")
        self.cpu_canvas.pack(side="left")
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
        left = ttk.LabelFrame(workflow, text="Processes")
        center = ttk.LabelFrame(workflow, text="Pipeline Canvas")
        right = ttk.LabelFrame(workflow, text="Process Parameters")
        workflow.add(left, weight=2)
        workflow.add(center, weight=6)
        workflow.add(right, weight=2)

        self.library = tk.Listbox(left, exportselection=False)
        self.library.pack(fill="both", expand=True, padx=4, pady=4)
        for code in sorted(BLOCK_DEFINITIONS):
            name = BLOCK_DEFINITIONS[code].display_name
            self.library.insert("end", f"{name} ({code})")
        self.library.bind("<ButtonPress-1>", self._start_library_drag)

        ttk.Label(center, text="Stage 1 — Preprocessing (L/R/G/B)", font=("TkDefaultFont", 10, "bold")).pack(fill="x", padx=4, pady=(4, 2))
        stage1_grid = ttk.Frame(center)
        stage1_grid.pack(fill="both", expand=True, padx=4)
        self.stage1_trees: dict[str, ttk.Treeview] = {}
        for idx, ch in enumerate(STAGE1_CHANNELS):
            pane = tk.Frame(stage1_grid, highlightthickness=2, highlightbackground="#555")
            pane.grid(row=0, column=idx, sticky="nsew", padx=2)
            stage1_grid.grid_columnconfigure(idx, weight=1, minsize=120)
            ttk.Label(pane, text=ch, font=("TkDefaultFont", 9, "bold")).pack(fill="x")
            tree = self._make_stage_tree(pane)
            self.stage1_trees[ch] = tree
            tree.bind("<<TreeviewSelect>>", lambda _e, c=ch: self._on_select_stage1(c))
            tree.bind("<B1-Motion>", self._drag_motion)
            tree.bind("<ButtonRelease-1>", self._drop_on_tree)

        ttk.Separator(center, orient="horizontal").pack(fill="x", padx=4, pady=8)
        ttk.Label(center, text="Stage 2 — Enhancement", font=("TkDefaultFont", 10, "bold")).pack(fill="x", padx=4, pady=(2, 2))
        self.stage2_frame = tk.Frame(center, highlightthickness=2, highlightbackground="#555")
        self.stage2_frame.pack(fill="both", expand=True, padx=4, pady=(0, 4))
        self.stage2_tree = self._make_stage_tree(self.stage2_frame)
        self.stage2_tree.bind("<<TreeviewSelect>>", lambda _e: self._on_select(2))
        self.stage2_tree.bind("<B1-Motion>", self._drag_motion)
        self.stage2_tree.bind("<ButtonRelease-1>", self._drop_on_tree)

        controls = ttk.Frame(center)
        controls.pack(fill="x", padx=4, pady=4)
        for label, cmd in [("▲", self._move_up), ("▼", self._move_down), ("Enable/Disable", self._toggle_enabled), ("Duplicate", self._duplicate_selected), ("Delete", self._delete_selected)]:
            ttk.Button(controls, text=label, command=cmd).pack(side="left", padx=2)

        self.param_area = ttk.Frame(right)
        self.param_area.pack(fill="both", expand=True, padx=4, pady=4)

    def _make_stage_tree(self, parent: tk.Widget) -> ttk.Treeview:
        tree = ttk.Treeview(parent, columns=("name",), show="headings", selectmode="browse", height=6)
        tree.heading("name", text="Block")
        tree.column("name", width=120, anchor="w")
        tree.pack(fill="both", expand=True, padx=2, pady=2)
        return tree

    def _bind_shortcuts(self) -> None:
        self.root.bind("<Control-z>", lambda *_: self._undo())
        self.root.bind("<Control-y>", lambda *_: self._redo())

    def _stage1_blocks(self, channel: str) -> list[BlockInstance]:
        return self.session.stage1_workflows[channel].blocks

    def _pipeline_blocks(self) -> list[BlockInstance]:
        blocks: list[BlockInstance] = []
        for ch in STAGE1_CHANNELS:
            for b in self._stage1_blocks(ch):
                b.channel = ch
                blocks.append(b)
        blocks.extend(self.session.stage2_blocks)
        return blocks

    def _refresh_pipeline_views(self) -> None:
        for ch, tree in self.stage1_trees.items():
            tree.delete(*tree.get_children())
            for block in self._stage1_blocks(ch):
                name = BLOCK_DEFINITIONS[block.type].display_name
                tree.insert("", "end", iid=block.id, values=(f"{name} ({block.type})",), tags=(block.type,))
        self.stage2_tree.delete(*self.stage2_tree.get_children())
        for block in self.session.stage2_blocks:
            name = BLOCK_DEFINITIONS[block.type].display_name
            self.stage2_tree.insert("", "end", iid=block.id, values=(f"{name} ({block.type})",), tags=(block.type,))
        for tree in [*self.stage1_trees.values(), self.stage2_tree]:
            for code, color in BLOCK_COLORS.items():
                tree.tag_configure(code, background=color)
        self._update_selection_outline()
        self._render_param_editor()

    def _update_selection_outline(self) -> None:
        all_frames = [self.stage2_frame, *[t.master for t in self.stage1_trees.values()]]
        for frame in all_frames:
            frame.configure(highlightbackground="#555")
        block = self._selected_block()
        if not block:
            return
        target = self.stage2_frame if self.selected_stage.get() == 2 else self.stage1_trees[self.selected_channel.get()].master
        target.configure(highlightbackground=self._lighten_color(BLOCK_COLORS.get(block.type, "#777"), 0.35))

    def _lighten_color(self, color: str, factor: float) -> str:
        color = color.lstrip("#")
        rgb = [int(color[i : i + 2], 16) for i in (0, 2, 4)]
        out = [min(255, int(c + (255 - c) * factor)) for c in rgb]
        return f"#{out[0]:02x}{out[1]:02x}{out[2]:02x}"

    def _on_select_stage1(self, channel: str) -> None:
        sel = self.stage1_trees[channel].selection()
        if not sel:
            return
        self.selected_stage.set(1)
        self.selected_channel.set(channel)
        self.selected_block_id = sel[0]
        self._render_param_editor()
        self._update_selection_outline()

    def _on_select(self, stage: int) -> None:
        if stage == 2:
            sel = self.stage2_tree.selection()
            if not sel:
                return
            self.selected_stage.set(2)
            self.selected_block_id = sel[0]
            self._render_param_editor()
            self._update_selection_outline()

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
        location = f"Stage 1 [{block.channel}]" if block.channel else "Stage 2"
        ttk.Label(self.param_area, text=f"{BLOCK_DEFINITIONS[block.type].display_name} ({block.type})", font=("TkDefaultFont", 10, "bold")).pack(anchor="w", pady=(0, 4))
        ttk.Label(self.param_area, text=f"Location: {location}").pack(anchor="w", pady=(0, 6))
        for path, value in self._iter_param_leaves(block.params):
            self._render_param_control(path, value)

    def _iter_param_leaves(self, params: Any, path: tuple[Any, ...] = ()) -> list[tuple[tuple[Any, ...], Any]]:
        items: list[tuple[tuple[Any, ...], Any]] = []
        if isinstance(params, dict):
            for key, value in params.items():
                items.extend(self._iter_param_leaves(value, (*path, key)))
        elif isinstance(params, (list, tuple)):
            for idx, value in enumerate(params):
                items.extend(self._iter_param_leaves(value, (*path, idx)))
        elif isinstance(params, (int, float, bool, str)):
            items.append((path, params))
        return items

    def _render_param_control(self, path: tuple[Any, ...], value: Any) -> None:
        row = ttk.Frame(self.param_area)
        row.pack(fill="x", pady=2)
        label = " / ".join(str(part) for part in path)
        ttk.Label(row, text=label, width=24).pack(side="left")
        if isinstance(value, bool):
            var = tk.BooleanVar(value=value)
            ttk.Checkbutton(row, variable=var, command=lambda p=path, v=var: self._set_param_path(p, v.get())).pack(side="left")
            return
        if isinstance(value, (int, float)):
            var = tk.DoubleVar(value=float(value))
            scale = ttk.Scale(row, from_=0, to=max(2.0, float(value) * 2 + 1), variable=var, command=lambda _v, p=path, v=var: self._set_param_path(p, v.get()))
            scale.pack(side="left", fill="x", expand=True, padx=4)
            return
        var = tk.StringVar(value=str(value))
        entry = ttk.Entry(row, textvariable=var)
        entry.pack(side="left", fill="x", expand=True, padx=4)
        entry.bind("<FocusOut>", lambda _e, p=path, v=var: self._set_param_path(p, v.get()))
        entry.bind("<Return>", lambda _e, p=path, v=var: self._set_param_path(p, v.get()))

    def _active_stage_blocks(self) -> list[BlockInstance] | None:
        if self.selected_stage.get() == 1:
            return self._stage1_blocks(self.selected_channel.get())
        if self.selected_stage.get() == 2:
            return self.session.stage2_blocks
        return None

    def _set_param(self, key: str, value: Any) -> None:
        self._set_param_path((key,), value)

    def _set_param_path(self, path: tuple[Any, ...], value: Any) -> None:
        block = self._selected_block()
        if not block:
            return
        old = self._get_path_value(block.params, path)
        if old is None:
            return
        if isinstance(old, bool):
            new = bool(value)
        elif isinstance(old, int) and not isinstance(old, bool):
            new = int(round(float(value)))
        elif isinstance(old, float):
            new = round(float(value), 4)
        else:
            new = str(value)
        if old == new:
            return

        def do() -> None:
            self._set_path_value(block.params, path, new)
            self._schedule_processing("parameter")

        def undo() -> None:
            self._set_path_value(block.params, path, old)
            self._schedule_processing("parameter")

        self.history.execute(Command(do=do, undo=undo, description="Param"))

    def _get_path_value(self, root: Any, path: tuple[Any, ...]) -> Any:
        current = root
        for part in path:
            current = current[part]
        return current

    def _set_path_value(self, root: Any, path: tuple[Any, ...], value: Any) -> None:
        current = root
        for part in path[:-1]:
            current = current[part]
        leaf = path[-1]
        if isinstance(current, tuple):
            mutable = list(current)
            mutable[leaf] = value
            value = tuple(mutable)
            self._set_path_value(root, path[:-1], value)
            return
        current[leaf] = value

    def _code_from_library_selection(self) -> str | None:
        sel = self.library.curselection()
        if not sel:
            return None
        return self.library.get(sel[0]).rsplit("(", 1)[-1].rstrip(")")

    def _toggle_enabled(self) -> None:
        block = self._selected_block()
        if not block:
            return
        old = block.enabled

        def do() -> None:
            block.enabled = not old
            self._refresh_pipeline_views()
            self._schedule_processing("toggle")

        self.history.execute(Command(do=do, undo=do, description="Toggle"))

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
        clone = BlockInstance(type=block.type, enabled=block.enabled, params=copy.deepcopy(block.params), channel=block.channel)

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
        blocks = self._active_stage_blocks()
        if blocks is None or block not in blocks:
            return
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

    def _tree_under_pointer(self, x_root: int, y_root: int) -> tuple[ttk.Treeview | None, int | None, str | None]:
        widget = self.root.winfo_containing(x_root, y_root)
        for ch, tree in self.stage1_trees.items():
            if widget is tree or (widget and str(widget).startswith(str(tree))):
                return tree, 1, ch
        tree = self.stage2_tree
        if widget is tree or (widget and str(widget).startswith(str(tree))):
            return tree, 2, None
        return None, None, None

    def _drag_motion(self, event: tk.Event) -> None:
        if not self._drag_from_library:
            return
        tree, stage, _ch = self._tree_under_pointer(event.x_root, event.y_root)
        for frame in [self.stage2_frame, *[t.master for t in self.stage1_trees.values()]]:
            frame.configure(highlightbackground="#555")
        if tree and stage:
            tree.master.configure(highlightbackground="#4a90e2")

    def _drop_on_tree(self, event: tk.Event) -> None:
        tree, stage, channel = self._tree_under_pointer(event.x_root, event.y_root)
        self.root.configure(cursor="")
        for frame in [self.stage2_frame, *[t.master for t in self.stage1_trees.values()]]:
            frame.configure(highlightbackground="#555")
        if not self._drag_from_library:
            return
        if not tree or not stage:
            self._drag_from_library = None
            return
        block = BlockInstance(type=self._drag_from_library, params=dict(BLOCK_DEFINITIONS[self._drag_from_library].defaults), channel=channel)
        if stage == 1 and channel:
            self._stage1_blocks(channel).append(block)
            self.selected_channel.set(channel)
        else:
            self.session.stage2_blocks.append(block)
        self.selected_stage.set(stage)
        self.selected_block_id = block.id
        self._drag_from_library = None
        self._refresh_pipeline_views()
        self._schedule_processing("drag-insert")

    def _block_signature(self, block: BlockInstance) -> str:
        payload = {"type": block.type, "enabled": block.enabled, "params": block.params, "channel": block.channel}
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
        self.status.set(f"Processing… ({reason})")

        def worker() -> None:
            upstream = dict(self.loaded_image)
            upstream["channels_signal"] = {"L": 0.0, "R": 0.0, "G": 0.0, "B": 0.0}
            upstream_key = hashlib.sha1((self.loaded_image.get("path", "") + str(self.loaded_image.get("format", ""))).encode()).hexdigest()
            for idx, block in enumerate(self._pipeline_blocks()):
                sig = signatures[idx]
                key = hashlib.sha1(f"{upstream_key}|{sig}".encode()).hexdigest()
                out = dict(upstream)
                out["channels_signal"] = dict(upstream["channels_signal"])
                if block.enabled:
                    bump = round((idx + 1) * 0.01 + self._param_signal(block.params), 4)
                    channel = block.channel or "L"
                    out["channels_signal"][channel] = round(out["channels_signal"].get(channel, 0.0) + bump, 4)
                out["signal"] = round(sum(out["channels_signal"].values()), 4)
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

    def _param_signal(self, value: Any) -> float:
        if isinstance(value, bool):
            return 0.005 if value else 0.0
        if isinstance(value, (int, float)):
            return abs(float(value)) * 0.001
        if isinstance(value, str):
            return (sum(ord(ch) for ch in value) % 10) * 0.0005
        if isinstance(value, dict):
            return sum(self._param_signal(v) for v in value.values())
        if isinstance(value, (list, tuple)):
            return sum(self._param_signal(v) for v in value)
        return 0.0

    def _finish_processing(self, output: dict[str, Any]) -> None:
        self.processed_image = output
        self.status.set("Ready")
        self._refresh_viewer()

    def _refresh_viewer(self) -> None:
        self.viewer_canvas.delete("all")
        w = max(self.viewer_canvas.winfo_width(), 1)
        h = max(self.viewer_canvas.winfo_height(), 1)
        if not self.loaded_image:
            self.viewer_canvas.create_text(w // 2, h // 2, fill="white", text="No image loaded")
            return
        source = self.loaded_image if self.viewer_mode.get() == "Before" else self.processed_image
        if source is None:
            self.viewer_canvas.create_text(w // 2, h // 2, fill="white", text="Processing…")
            return
        if not self._viewer_source_image:
            self.viewer_canvas.create_text(w // 2, h // 2, fill="white", text="Image unavailable")
            return
        self._viewer_display_image = self._fit_photo(self._viewer_source_image, w - 16, h - 16)
        self.viewer_canvas.create_image(w // 2, h // 2, image=self._viewer_display_image, anchor="center")
        sig = source.get("channels_signal", {})
        overlay = f"signal {source.get('signal', 0.0)} | L:{sig.get('L',0)} R:{sig.get('R',0)} G:{sig.get('G',0)} B:{sig.get('B',0)}"
        self.viewer_canvas.create_text(8, 8, anchor="nw", fill="#90ee90", text=overlay)

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
        if image.width() <= max_w and image.height() <= max_h:
            return image
        factor = max(1, math.ceil(max(image.width() / max(max_w, 1), image.height() / max(max_h, 1))))
        return image.subsample(factor, factor)

    def _open_template(self, path: str) -> None:
        try:
            loaded = TemplateStore.load(path)
        except Exception as exc:
            messagebox.showerror("PlanetSharp", f"Template load failed: {exc}")
            return
        for ch in STAGE1_CHANNELS:
            self.session.stage1_workflows[ch].blocks.clear()
        for block in loaded["stage1"]:
            ch = block.channel if block.channel in STAGE1_CHANNELS else "L"
            self.session.stage1_workflows[ch].blocks.append(block)
        self.session.stage2_blocks = loaded["stage2"]
        self.selected_block_id = None
        self._refresh_pipeline_views()
        self.status.set(f"Loaded template {Path(path).name}")
        self._schedule_processing("template-load")

    def _save_template(self) -> None:
        path = filedialog.asksaveasfilename(title="Save template", defaultextension=TEMPLATE_SUFFIX, filetypes=[("PlanetSharp Template", f"*{TEMPLATE_SUFFIX}")])
        if path:
            TemplateStore.save(path, self.session)

    def _export_image(self) -> None:
        source = self.loaded_image if self.viewer_mode.get() == "Before" else self.processed_image
        if source is None:
            messagebox.showerror("PlanetSharp", "No image loaded")
            return
        path = filedialog.asksaveasfilename(title="Export image", defaultextension=".png", filetypes=[("PNG", "*.png"), ("BMP", "*.bmp"), ("TIFF", "*.tif *.tiff"), ("JPEG", "*.jpg *.jpeg"), ("XISF", "*.xisf"), ("FITS", "*.fits")])
        if path:
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

    def _tick_cpu_usage(self) -> None:
        cpu = self._cpu_percent()
        self._cpu_history = (self._cpu_history + [cpu])[-20:]
        self.cpu_canvas.delete("all")
        w = int(self.cpu_canvas["width"])
        h = int(self.cpu_canvas["height"])
        points = []
        for i, val in enumerate(self._cpu_history):
            x = int(i * (w - 1) / 19)
            y = int(h - (val / 100.0) * (h - 2) - 1)
            points.extend([x, y])
        if len(points) >= 4:
            self.cpu_canvas.create_line(*points, fill="#4ade80", width=2)
        self.cpu_canvas.create_text(4, 4, anchor="nw", fill="#ddd", text=f"{cpu:.1f}%")
        self.root.after(1000, self._tick_cpu_usage)

    def _cpu_percent(self) -> float:
        try:
            load = os.getloadavg()[0]
            cores = max(1, os.cpu_count() or 1)
            return max(0.0, min(100.0, (load / cores) * 100.0))
        except (AttributeError, OSError):
            return 0.0

    def _save_ui_state(self) -> None:
        LAYOUT_STATE.write_text(json.dumps({"viewer_mode": self.viewer_mode.get(), "geometry": self.root.geometry()}, indent=2), encoding="utf-8")

    def _restore_ui_state(self) -> None:
        if not LAYOUT_STATE.exists():
            return
        try:
            state = json.loads(LAYOUT_STATE.read_text(encoding="utf-8"))
            self.viewer_mode.set(state.get("viewer_mode", "Before"))
            if geometry := state.get("geometry"):
                self.root.geometry(geometry)
        except json.JSONDecodeError:
            return

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
