from __future__ import annotations

import logging
import os
import time
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass


@dataclass
class _Sample:
    total_ms: float = 0.0
    count: int = 0
    max_ms: float = 0.0


class PerfMonitor:
    """Low-overhead rolling profiler for real-time pipeline timing."""

    def __init__(self, enabled: bool = False, report_every: int = 30) -> None:
        self.enabled = enabled
        self.report_every = max(1, report_every)
        self._samples: dict[str, _Sample] = defaultdict(_Sample)
        self._frames = 0
        self._logger = logging.getLogger("planetsharp.perf")
        if self.enabled and not logging.getLogger().handlers:
            logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    @classmethod
    def from_env(cls) -> "PerfMonitor":
        enabled = os.getenv("PLANETSHARP_PROFILE", "0") in {"1", "true", "TRUE", "yes", "YES"}
        report_every = int(os.getenv("PLANETSHARP_PROFILE_EVERY", "20"))
        return cls(enabled=enabled, report_every=report_every)

    @contextmanager
    def track(self, name: str):
        if not self.enabled:
            yield
            return
        start = time.perf_counter()
        try:
            yield
        finally:
            dt_ms = (time.perf_counter() - start) * 1000.0
            sample = self._samples[name]
            sample.total_ms += dt_ms
            sample.count += 1
            sample.max_ms = max(sample.max_ms, dt_ms)

    def end_frame(self) -> None:
        if not self.enabled:
            return
        self._frames += 1
        if self._frames % self.report_every != 0:
            return

        rows: list[tuple[str, float, float, int]] = []
        for name, sample in self._samples.items():
            if sample.count <= 0:
                continue
            avg = sample.total_ms / sample.count
            rows.append((name, avg, sample.max_ms, sample.count))

        rows.sort(key=lambda item: item[1], reverse=True)
        if rows:
            summary = " | ".join(
                f"{name}: avg={avg:.2f}ms max={max_ms:.2f}ms n={n}" for name, avg, max_ms, n in rows[:12]
            )
            self._logger.info("Pipeline perf frame=%d :: %s", self._frames, summary)

        self._samples.clear()
