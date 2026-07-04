"""Custom QPainter waveform widget.

Renders downsampled (min,max) peak pairs — never raw samples — with:
  * playhead synced to playback position,
  * click-to-seek,
  * per-line region shading for the selected line,
  * draggable start/end markers for the selected line or word,
  * zoom (Ctrl+wheel) and horizontal pan (wheel / drag on empty space).
"""

from __future__ import annotations

from enum import Enum

import numpy as np
from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QColor, QPainter, QPen, QWheelEvent
from PySide6.QtWidgets import QWidget

_BG = QColor("#14161a")
_WAVE = QColor("#3d7bd9")
_PLAYHEAD = QColor("#ff5252")
_REGION = QColor(61, 123, 217, 60)
_MARKER_START = QColor("#4caf50")
_MARKER_END = QColor("#ff9800")
_WORD_REGION = QColor(255, 235, 59, 50)


class DragTarget(Enum):
    NONE = 0
    REGION_START = 1
    REGION_END = 2
    WORD_START = 3
    WORD_END = 4


_MARKER_GRAB_PX = 6


class WaveformWidget(QWidget):
    seek_requested = Signal(float)                 # seconds
    region_changed = Signal(float, float)          # line start, end (during drag)
    region_committed = Signal(float, float)        # on mouse release
    word_changed = Signal(float, float)
    word_committed = Signal(float, float)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(140)
        self.setMouseTracking(True)
        self._peaks: np.ndarray = np.zeros((0, 2), dtype=np.float32)
        self._duration = 0.0
        self._playhead = 0.0
        # visible window [t0, t1]
        self._t0 = 0.0
        self._t1 = 1.0
        # selected line region + optional selected word within it
        self._region: tuple[float, float] | None = None
        self._word: tuple[float, float] | None = None
        self._drag = DragTarget.NONE

    # --- data ----------------------------------------------------------

    def set_peaks(self, peaks: np.ndarray, duration: float) -> None:
        self._peaks = peaks
        self._duration = max(duration, 0.001)
        self._t0, self._t1 = 0.0, self._duration
        self.update()

    def clear(self) -> None:
        self._peaks = np.zeros((0, 2), dtype=np.float32)
        self._duration = 0.0
        self._region = None
        self._word = None
        self.update()

    def set_playhead(self, seconds: float) -> None:
        self._playhead = seconds
        self.update()

    def set_region(self, start: float | None, end: float | None) -> None:
        self._region = (start, end) if start is not None and end is not None else None
        self.update()

    def set_word_region(self, start: float | None, end: float | None) -> None:
        self._word = (start, end) if start is not None and end is not None else None
        self.update()

    def focus_region(self, start: float, end: float, padding: float = 2.0) -> None:
        """Zoom the view to a line (with context padding)."""
        t0 = max(0.0, start - padding)
        t1 = min(self._duration, end + padding)
        if t1 - t0 > 0.1:
            self._t0, self._t1 = t0, t1
            self.update()

    def zoom_full(self) -> None:
        self._t0, self._t1 = 0.0, self._duration
        self.update()

    # --- coordinate mapping ---------------------------------------------

    def _x_to_time(self, x: float) -> float:
        frac = min(1.0, max(0.0, x / max(1, self.width())))
        return self._t0 + frac * (self._t1 - self._t0)

    def _time_to_x(self, t: float) -> float:
        span = max(0.001, self._t1 - self._t0)
        return (t - self._t0) / span * self.width()

    # --- painting ---------------------------------------------------------

    def paintEvent(self, event) -> None:
        p = QPainter(self)
        p.fillRect(self.rect(), _BG)
        w, h = self.width(), self.height()
        mid = h / 2

        if self._peaks.shape[0] and self._duration > 0:
            n = self._peaks.shape[0]
            i0 = int(self._t0 / self._duration * n)
            i1 = max(i0 + 1, int(self._t1 / self._duration * n))
            visible = self._peaks[i0:i1]
            p.setPen(QPen(_WAVE, 1))
            m = visible.shape[0]
            for px in range(w):
                j = int(px / w * m)
                lo, hi = visible[min(j, m - 1)]
                y1 = mid - float(hi) * (mid - 4)
                y2 = mid - float(lo) * (mid - 4)
                p.drawLine(QPointF(px, y1), QPointF(px, y2))

        # selected line region + markers
        if self._region:
            x1, x2 = self._time_to_x(self._region[0]), self._time_to_x(self._region[1])
            p.fillRect(QRectF(x1, 0, max(1.0, x2 - x1), h), _REGION)
            p.setPen(QPen(_MARKER_START, 2))
            p.drawLine(QPointF(x1, 0), QPointF(x1, h))
            p.setPen(QPen(_MARKER_END, 2))
            p.drawLine(QPointF(x2, 0), QPointF(x2, h))

        # selected word sub-region
        if self._word:
            x1, x2 = self._time_to_x(self._word[0]), self._time_to_x(self._word[1])
            p.fillRect(QRectF(x1, h * 0.25, max(1.0, x2 - x1), h * 0.5), _WORD_REGION)
            p.setPen(QPen(_MARKER_START, 1, Qt.PenStyle.DashLine))
            p.drawLine(QPointF(x1, h * 0.25), QPointF(x1, h * 0.75))
            p.setPen(QPen(_MARKER_END, 1, Qt.PenStyle.DashLine))
            p.drawLine(QPointF(x2, h * 0.25), QPointF(x2, h * 0.75))

        # playhead
        if self._duration:
            x = self._time_to_x(self._playhead)
            p.setPen(QPen(_PLAYHEAD, 1))
            p.drawLine(QPointF(x, 0), QPointF(x, h))
        p.end()

    # --- interaction -------------------------------------------------------

    def _hit_test(self, x: float) -> DragTarget:
        """Markers win over seeking when the cursor is close enough."""
        if self._word:
            if abs(x - self._time_to_x(self._word[0])) <= _MARKER_GRAB_PX:
                return DragTarget.WORD_START
            if abs(x - self._time_to_x(self._word[1])) <= _MARKER_GRAB_PX:
                return DragTarget.WORD_END
        if self._region:
            if abs(x - self._time_to_x(self._region[0])) <= _MARKER_GRAB_PX:
                return DragTarget.REGION_START
            if abs(x - self._time_to_x(self._region[1])) <= _MARKER_GRAB_PX:
                return DragTarget.REGION_END
        return DragTarget.NONE

    def mousePressEvent(self, event) -> None:
        if event.button() != Qt.MouseButton.LeftButton or not self._duration:
            return
        x = event.position().x()
        self._drag = self._hit_test(x)
        if self._drag is DragTarget.NONE:
            self.seek_requested.emit(self._x_to_time(x))

    def mouseMoveEvent(self, event) -> None:
        x = event.position().x()
        if self._drag is DragTarget.NONE:
            hover = self._hit_test(x)
            self.setCursor(Qt.CursorShape.SizeHorCursor if hover is not DragTarget.NONE
                           else Qt.CursorShape.ArrowCursor)
            return
        t = self._x_to_time(x)
        if self._drag is DragTarget.REGION_START and self._region:
            self._region = (min(t, self._region[1] - 0.01), self._region[1])
            self.region_changed.emit(*self._region)
        elif self._drag is DragTarget.REGION_END and self._region:
            self._region = (self._region[0], max(t, self._region[0] + 0.01))
            self.region_changed.emit(*self._region)
        elif self._drag is DragTarget.WORD_START and self._word:
            self._word = (min(t, self._word[1] - 0.01), self._word[1])
            self.word_changed.emit(*self._word)
        elif self._drag is DragTarget.WORD_END and self._word:
            self._word = (self._word[0], max(t, self._word[0] + 0.01))
            self.word_changed.emit(*self._word)
        self.update()

    def mouseReleaseEvent(self, event) -> None:
        if self._drag in (DragTarget.REGION_START, DragTarget.REGION_END) and self._region:
            self.region_committed.emit(*self._region)
        elif self._drag in (DragTarget.WORD_START, DragTarget.WORD_END) and self._word:
            self.word_committed.emit(*self._word)
        self._drag = DragTarget.NONE

    def wheelEvent(self, event: QWheelEvent) -> None:
        if not self._duration:
            return
        delta = event.angleDelta().y()
        span = self._t1 - self._t0
        if event.modifiers() & Qt.KeyboardModifier.ControlModifier:
            # zoom around cursor
            anchor = self._x_to_time(event.position().x())
            factor = 0.8 if delta > 0 else 1.25
            new_span = min(self._duration, max(0.25, span * factor))
            frac = (anchor - self._t0) / span
            self._t0 = max(0.0, anchor - frac * new_span)
            self._t1 = min(self._duration, self._t0 + new_span)
        else:
            # pan
            shift = span * (-0.1 if delta > 0 else 0.1)
            self._t0 = max(0.0, min(self._t0 + shift, self._duration - span))
            self._t1 = self._t0 + span
        self.update()
