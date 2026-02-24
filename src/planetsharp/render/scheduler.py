from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
from threading import Event, Lock
from typing import Callable

from planetsharp.core.models import Session
from planetsharp.processing.engine import RenderResult, WorkflowEngine


class RenderScheduler:
    def __init__(self, engine: WorkflowEngine, max_workers: int = 1):
        self._engine = engine
        self._executor = ThreadPoolExecutor(max_workers=max_workers)
        self._cancel_event = Event()
        self._lock = Lock()
        self._ticket = 0

    def render_latest(self, session: Session, on_complete: Callable[[RenderResult], None]) -> Future:
        with self._lock:
            self._ticket += 1
            ticket = self._ticket
            self._cancel_event.set()
            self._cancel_event = Event()

        def task() -> None:
            if self._cancel_event.is_set():
                return
            result = self._engine.render(session)
            with self._lock:
                if ticket != self._ticket:
                    return
            on_complete(result)

        return self._executor.submit(task)

    def shutdown(self) -> None:
        self._executor.shutdown(wait=False)
