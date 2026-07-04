"""QThread workers — every long-running operation runs off the main thread.

`FunctionWorker` wraps any blocking callable that accepts `progress` and
`cancel_check` keyword arguments (the pipeline, separation, downloads all
follow that convention). Progress arrives on the GUI thread via signals.
"""

from __future__ import annotations

import traceback
from typing import Any, Callable

from PySide6.QtCore import QThread, Signal

from lyricsync.utils.logs import get_logger

log = get_logger("workers")


class FunctionWorker(QThread):
    """Run `fn(*args, **kwargs)` on a QThread with progress + cancellation.

    If `fn` accepts them, `progress=(msg, frac)->None` and
    `cancel_check=()->bool` kwargs are injected automatically.
    """

    progressed = Signal(str, float)      # message, fraction 0..1
    succeeded = Signal(object)           # fn return value
    failed = Signal(str)                 # traceback-ish message

    def __init__(self, fn: Callable[..., Any], *args: Any,
                 inject_progress: bool = True, **kwargs: Any):
        super().__init__()
        self._fn = fn
        self._args = args
        self._kwargs = kwargs
        self._inject = inject_progress
        self._cancelled = False

    def cancel(self) -> None:
        self._cancelled = True

    @property
    def cancelled(self) -> bool:
        return self._cancelled

    def run(self) -> None:  # executes on the worker thread
        kwargs = dict(self._kwargs)
        if self._inject:
            kwargs.setdefault("progress", self._emit_progress)
            kwargs.setdefault("cancel_check", lambda: self._cancelled)
        try:
            result = self._fn(*self._args, **kwargs)
        except Exception as e:
            if self._cancelled:
                self.failed.emit("Cancelled.")
            else:
                log.exception("worker failed: %s", self._fn)
                self.failed.emit(f"{type(e).__name__}: {e}\n\n"
                                 + traceback.format_exc(limit=6))
            return
        self.succeeded.emit(result)

    def _emit_progress(self, message: str, fraction: float) -> None:
        self.progressed.emit(message, fraction)


class DownloadWorker(FunctionWorker):
    """FunctionWorker variant whose progress is (bytes_done, bytes_total)."""

    bytes_progressed = Signal(int, int)

    def _emit_progress(self, done: int, total: int) -> None:  # type: ignore[override]
        self.bytes_progressed.emit(done, total)
