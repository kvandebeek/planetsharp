from __future__ import annotations

import tkinter as tk
from tkinter import ttk

from planetsharp.core.models import Session
from planetsharp.processing.blocks import BLOCK_DEFINITIONS


class PlanetSharpApp:
    def __init__(self, session: Session | None = None):
        self.session = session or Session()
        self.root = tk.Tk()
        self.root.title("PlanetSharp v1")
        self._build_layout()

    def _panel(self, parent: tk.Widget, title: str) -> ttk.LabelFrame:
        frame = ttk.LabelFrame(parent, text=title)
        frame.pack(fill="both", expand=True, padx=4, pady=4)
        return frame

    def _build_layout(self) -> None:
        self.root.geometry("1200x800")
        container = ttk.Panedwindow(self.root, orient="horizontal")
        container.pack(fill="both", expand=True)

        left = ttk.Frame(container)
        center = ttk.Frame(container)
        right = ttk.Frame(container)
        container.add(left, weight=2)
        container.add(center, weight=5)
        container.add(right, weight=3)

        library = self._panel(left, "Building Blocks Library")
        for name in BLOCK_DEFINITIONS:
            ttk.Label(library, text=name).pack(anchor="w")

        self._panel(center, "Viewer")
        self._panel(center, "Stage 1 editor (L/R/G/B)")
        self._panel(center, "Stage 2 editor")

        self._panel(right, "Block Inspector")
        self._panel(right, "ROI + Performance")
        self._panel(right, "Progress / Status")

    def run(self) -> None:
        self.root.mainloop()


def main() -> None:
    PlanetSharpApp().run()


if __name__ == "__main__":
    main()
