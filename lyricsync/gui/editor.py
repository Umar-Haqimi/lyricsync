"""Manual correction UI.

Left: line list. Right: word table for the selected line with inline text
editing and low-confidence highlighting. Top: waveform with playback sync,
click-to-seek and draggable start/end markers for the selected line/word.
"""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt, QUrl, Slot
from PySide6.QtGui import QBrush, QColor
from PySide6.QtMultimedia import QAudioOutput, QMediaPlayer
from PySide6.QtWidgets import (
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from lyricsync.config import AppConfig
from lyricsync.core.audio import compute_peaks, prepare_working_wav, probe_duration
from lyricsync.core.model import LyricDocument
from lyricsync.core.syllables import make_splitter, split_word_timed
from lyricsync.exporters import EXPORTERS
from lyricsync.exporters.embed import embed_lyrics
from lyricsync.gui.waveform import WaveformWidget
from lyricsync.gui.workers import FunctionWorker
from lyricsync.utils.logs import get_logger

log = get_logger("editor")

_CONF_OK = QBrush(QColor("#e8eaed"))
_CONF_WARN = QBrush(QColor("#ffb300"))   # amber
_CONF_BAD = QBrush(QColor("#ff5252"))    # red


class EditorPage(QWidget):
    def __init__(self, config: AppConfig, parent=None):
        super().__init__(parent)
        self.config = config
        self.doc: LyricDocument | None = None
        self._playback_wav: Path | None = None
        self._peaks_worker: FunctionWorker | None = None
        self._suppress_word_edits = False

        self.player = QMediaPlayer(self)
        self.audio_out = QAudioOutput(self)
        self.player.setAudioOutput(self.audio_out)
        self.player.positionChanged.connect(self._on_position)

        self._build_ui()
        self.set_document(None)

    # --- UI scaffolding ---------------------------------------------------

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)

        self.waveform = WaveformWidget()
        self.waveform.seek_requested.connect(self._seek)
        self.waveform.region_changed.connect(self._on_region_drag)
        self.waveform.region_committed.connect(self._on_region_commit)
        self.waveform.word_changed.connect(self._on_word_drag)
        self.waveform.word_committed.connect(self._on_word_commit)
        layout.addWidget(self.waveform)

        controls = QHBoxLayout()
        self.play_btn = QPushButton("▶ Play")
        self.play_btn.clicked.connect(self._toggle_play)
        self.play_line_btn = QPushButton("▶ Line")
        self.play_line_btn.setToolTip("Play just the selected line")
        self.play_line_btn.clicked.connect(self._play_selected_line)
        self.zoom_btn = QPushButton("Zoom line")
        self.zoom_btn.clicked.connect(self._zoom_selected_line)
        self.zoom_out_btn = QPushButton("Full view")
        self.zoom_out_btn.clicked.connect(self.waveform.zoom_full)
        self.time_label = QLabel("0:00.00")
        for w in (self.play_btn, self.play_line_btn, self.zoom_btn, self.zoom_out_btn):
            controls.addWidget(w)
        controls.addStretch(1)
        controls.addWidget(self.time_label)
        layout.addLayout(controls)

        split = QSplitter(Qt.Orientation.Horizontal)

        self.line_list = QListWidget()
        self.line_list.currentRowChanged.connect(self._on_line_selected)
        split.addWidget(self.line_list)

        right = QWidget()
        right_layout = QVBoxLayout(right)
        hint = QLabel(
            "Double-click a word to edit its text. Drag the green/orange markers "
            "on the waveform to fix line timing (dashed = selected word). "
            "Amber/red = low-confidence words worth double-checking."
        )
        hint.setWordWrap(True)
        right_layout.addWidget(hint)

        self.word_table = QTableWidget(0, 4)
        self.word_table.setHorizontalHeaderLabels(["Word", "Start", "End", "Confidence"])
        self.word_table.itemChanged.connect(self._on_word_edited)
        self.word_table.currentCellChanged.connect(self._on_word_selected)
        right_layout.addWidget(self.word_table)
        split.addWidget(right)
        split.setStretchFactor(0, 1)
        split.setStretchFactor(1, 2)
        layout.addWidget(split, stretch=1)

        actions = QHBoxLayout()
        self.export_btns: dict[str, QPushButton] = {}
        for fmt in EXPORTERS:
            btn = QPushButton(f"Export .{fmt}")
            btn.clicked.connect(lambda _=False, f=fmt: self._export(f))
            actions.addWidget(btn)
            self.export_btns[fmt] = btn
        self.embed_btn = QPushButton("Embed into audio tags")
        self.embed_btn.clicked.connect(self._embed)
        actions.addWidget(self.embed_btn)
        self.save_json_btn = QPushButton("Save project JSON")
        self.save_json_btn.clicked.connect(self._save_json)
        actions.addWidget(self.save_json_btn)
        actions.addStretch(1)
        layout.addLayout(actions)

    # --- document loading -----------------------------------------------

    def set_document(self, doc: LyricDocument | None) -> None:
        self.doc = doc
        enabled = doc is not None
        for w in (self.play_btn, self.play_line_btn, self.zoom_btn, self.zoom_out_btn,
                  self.embed_btn, self.save_json_btn, *self.export_btns.values()):
            w.setEnabled(enabled)
        self.line_list.clear()
        self.word_table.setRowCount(0)
        self.waveform.clear()
        if not doc:
            return

        for line in doc.lines:
            flagged = sum(1 for w in line.words
                          if w.confidence < self.config.confidence_warn_threshold)
            suffix = f"   ⚠{flagged}" if flagged else ""
            item = QListWidgetItem(f"{_fmt_time(line.start)}  {line.text}{suffix}")
            if flagged:
                item.setForeground(_CONF_WARN)
            self.line_list.addItem(item)

        self._load_audio(Path(doc.audio_source))
        if doc.lines:
            self.line_list.setCurrentRow(0)

    def _load_audio(self, source: Path) -> None:
        """Prepare a WAV working copy for playback + peaks off the UI thread."""
        if not source.exists():
            log.warning("audio source missing: %s", source)
            return

        def job(progress=None, cancel_check=None):
            wav = prepare_working_wav(source)
            return wav, compute_peaks(wav), probe_duration(wav)

        self._peaks_worker = FunctionWorker(job)
        self._peaks_worker.succeeded.connect(self._on_audio_ready)
        self._peaks_worker.failed.connect(
            lambda msg: log.error("audio load failed: %s", msg))
        self._peaks_worker.start()

    @Slot(object)
    def _on_audio_ready(self, result) -> None:
        wav, peaks, duration = result
        self._playback_wav = wav
        self.waveform.set_peaks(peaks, duration)
        self.player.setSource(QUrl.fromLocalFile(str(wav)))

    # --- selection & tables ------------------------------------------------

    def _current_line(self):
        if self.doc and 0 <= self.line_list.currentRow() < len(self.doc.lines):
            return self.doc.lines[self.line_list.currentRow()]
        return None

    def _current_word(self):
        line = self._current_line()
        row = self.word_table.currentRow()
        if line and 0 <= row < len(line.words):
            return line.words[row]
        return None

    @Slot(int)
    def _on_line_selected(self, row: int) -> None:
        line = self._current_line()
        if not line:
            self.waveform.set_region(None, None)
            return
        self.waveform.set_region(line.start, line.end)
        self.waveform.set_word_region(None, None)
        self._populate_word_table(line)

    def _populate_word_table(self, line) -> None:
        self._suppress_word_edits = True
        self.word_table.setRowCount(len(line.words))
        warn = self.config.confidence_warn_threshold
        bad = self.config.confidence_bad_threshold
        for i, word in enumerate(line.words):
            text_item = QTableWidgetItem(word.text)
            brush = _CONF_BAD if word.confidence < bad else (
                _CONF_WARN if word.confidence < warn else _CONF_OK)
            text_item.setForeground(brush)
            self.word_table.setItem(i, 0, text_item)
            for col, value in ((1, _fmt_time(word.start)), (2, _fmt_time(word.end)),
                               (3, f"{word.confidence:.2f}")):
                item = QTableWidgetItem(value)
                item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
                item.setForeground(brush)
                self.word_table.setItem(i, col, item)
        self.word_table.resizeColumnsToContents()
        self._suppress_word_edits = False

    @Slot(int, int, int, int)
    def _on_word_selected(self, row: int, col: int, prow: int, pcol: int) -> None:
        word = self._current_word()
        if word:
            self.waveform.set_word_region(word.start, word.end)

    @Slot(QTableWidgetItem)
    def _on_word_edited(self, item: QTableWidgetItem) -> None:
        if self._suppress_word_edits or item.column() != 0:
            return
        line = self._current_line()
        if not line or not (0 <= item.row() < len(line.words)):
            return
        word = line.words[item.row()]
        new_text = item.text().strip()
        if new_text and new_text != word.text:
            word.text = new_text
            # Re-split syllables so eLRC/ASS stay consistent with the edit.
            splitter = make_splitter(self.config.syllable_backend,
                                     self.config.syllable_lang)
            word.syllables = split_word_timed(word, splitter)
            self._refresh_line_item(item.row())

    def _refresh_line_item(self, _word_row: int) -> None:
        row = self.line_list.currentRow()
        line = self._current_line()
        if line and row >= 0:
            self.line_list.item(row).setText(
                f"{_fmt_time(line.start)}  {line.text}")

    # --- marker drags ------------------------------------------------------

    @Slot(float, float)
    def _on_region_drag(self, start: float, end: float) -> None:
        pass  # live visual only; model updates on commit

    @Slot(float, float)
    def _on_region_commit(self, start: float, end: float) -> None:
        line = self._current_line()
        if not line:
            return
        # Rescale word timings into the new line bounds.
        old_span = max(0.001, line.end - line.start)
        scale = (end - start) / old_span
        for word in line.words:
            word.start = start + (word.start - line.start) * scale
            word.end = start + (word.end - line.start) * scale
            for syl in word.syllables:
                syl.start = start + (syl.start - line.start) * scale
                syl.end = start + (syl.end - line.start) * scale
        line.start, line.end = start, end
        self._refresh_line_item(0)
        self._populate_word_table(line)

    @Slot(float, float)
    def _on_word_drag(self, start: float, end: float) -> None:
        pass

    @Slot(float, float)
    def _on_word_commit(self, start: float, end: float) -> None:
        word = self._current_word()
        line = self._current_line()
        if not word or not line:
            return
        word.start, word.end = start, end
        splitter = make_splitter(self.config.syllable_backend, self.config.syllable_lang)
        word.syllables = split_word_timed(word, splitter)
        line.recompute_bounds()
        self.waveform.set_region(line.start, line.end)
        self._populate_word_table(line)

    # --- playback ------------------------------------------------------------

    def _toggle_play(self) -> None:
        if self.player.playbackState() == QMediaPlayer.PlaybackState.PlayingState:
            self.player.pause()
            self.play_btn.setText("▶ Play")
        else:
            self.player.play()
            self.play_btn.setText("⏸ Pause")

    def _play_selected_line(self) -> None:
        line = self._current_line()
        if line:
            self._seek(line.start)
            self.player.play()
            self.play_btn.setText("⏸ Pause")

    def _zoom_selected_line(self) -> None:
        line = self._current_line()
        if line:
            self.waveform.focus_region(line.start, line.end)

    def _seek(self, seconds: float) -> None:
        self.player.setPosition(int(seconds * 1000))

    @Slot(int)
    def _on_position(self, ms: int) -> None:
        t = ms / 1000.0
        self.waveform.set_playhead(t)
        self.time_label.setText(_fmt_time(t))

    # --- export / save --------------------------------------------------------

    def _export(self, fmt: str) -> None:
        if not self.doc:
            return
        exporter, ext = EXPORTERS[fmt]
        default = str(Path(self.doc.audio_source).with_suffix(ext))
        path, _ = QFileDialog.getSaveFileName(self, f"Export {fmt.upper()}",
                                              default, f"*{ext}")
        if not path:
            return
        Path(path).write_text(exporter(self.doc), encoding="utf-8")
        log.info("exported %s -> %s", fmt, path)

    def _embed(self) -> None:
        if not self.doc:
            return
        result = embed_lyrics(self.doc)
        box = QMessageBox(self)
        box.setWindowTitle("Embed lyrics")
        box.setIcon(QMessageBox.Icon.Information if result.ok
                    else QMessageBox.Icon.Warning)
        box.setText(result.message)
        box.exec()

    def _save_json(self) -> None:
        if not self.doc:
            return
        default = str(Path(self.doc.audio_source).with_suffix(".lyricsync.json"))
        path, _ = QFileDialog.getSaveFileName(self, "Save project JSON",
                                              default, "*.json")
        if path:
            self.doc.save_json(path)


def _fmt_time(t: float) -> str:
    m = int(max(0, t) // 60)
    return f"{m}:{max(0.0, t) - m * 60:05.2f}"
