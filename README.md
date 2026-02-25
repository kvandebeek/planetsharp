# PlanetSharp

A PySide6-based planetary image editor with a block pipeline and realtime rendering.

## How to run

1. Install dependencies:
   ```bash
   pip install PySide6 numpy pillow tifffile psutil
   ```
2. Start the app from project root:
   ```bash
   python3 main.py
   ```

## Performance profiling

- Enable live per-frame/per-block timing logs:
  ```bash
  PLANETSHARP_PROFILE=1 PLANETSHARP_PROFILE_EVERY=10 python3 main.py
  ```
- Run synthetic benchmark to identify hotspot blocks:
  ```bash
  python3 scripts/profile_pipeline.py --width 1920 --height 1080 --runs 15
  ```
- Deep analysis report: `docs/performance_analysis.md`.
