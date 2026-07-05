"""Main window: Transcribe / Editor / Queue / Models / Settings tabs.

The Transcribe tab drives one-off runs: file picker (+ drag & drop),
separation tier radio buttons, Whisper model-size slider, device/compute
selectors, VAD control, optional official-lyrics alignment, and a progress
bar fed by the pipeline worker.
"""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt, Slot
from PySide6.QtWidgets import (
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QRadioButton,
    QSlider,
    QSpinBox,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from lyricsync import __version__
from lyricsync.config import AppConfig
from lyricsync.core.alignment import make_engine
from lyricsync.core.audio import SUPPORTED_EXTENSIONS, read_tags
from lyricsync.core.model import LyricDocument
from lyricsync.core.pipeline import PipelineOptions, run_pipeline
from lyricsync.core.separation import CUSTOM_TIER, SKIP_TIER, TIERS
from lyricsync.core.syllables import make_splitter, split_document
from lyricsync.core.transcription import COMPUTE_TYPES, WHISPER_SIZES
from lyricsync.gui.editor import EditorPage
from lyricsync.gui.models_page import ModelsPage
from lyricsync.gui.queue_page import QueuePage
from lyricsync.gui.settings_page import SettingsPage
from lyricsync.gui.workers import FunctionWorker
from lyricsync.utils.logs import get_logger

log = get_logger("main_window")


class MainWindow(QMainWindow):
    def __init__(self, config: AppConfig):
        super().__init__()
        self.config = config
        self.setWindowTitle(f"LyricSync {__version__}")
        self.resize(1200, 800)
        self.setAcceptDrops(True)

        self._pipeline_worker: FunctionWorker | None = None
        self._source: Path | None = None

        self.tabs = QTabWidget()
        self.setCentralWidget(self.tabs)

        self.transcribe_page = self._build_transcribe_page()
        self.editor_page = EditorPage(config)
        self.queue_page = QueuePage(config)
        self.queue_page.job_finished.connect(self._on_queue_job_finished)
        self.models_page = ModelsPage(config)
        self.settings_page = SettingsPage(config)

        self.tabs.addTab(self.transcribe_page, "Transcribe")
        self.tabs.addTab(self.editor_page, "Editor")
        self.tabs.addTab(self.queue_page, "Queue")
        self.tabs.addTab(self.models_page, "Models")
        self.tabs.addTab(self.settings_page, "Settings")

    # --- Transcribe tab -----------------------------------------------------

    def _build_transcribe_page(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)

        # File picker
        file_row = QHBoxLayout()
        self.file_edit = QLineEdit()
        self.file_edit.setPlaceholderText(
            "Drop an audio file anywhere in this window, or browse…")
        pick_btn = QPushButton("Browse…")
        pick_btn.clicked.connect(self._pick_file)
        open_json_btn = QPushButton("Open project JSON…")
        open_json_btn.clicked.connect(self._open_project)
        file_row.addWidget(self.file_edit, stretch=1)
        file_row.addWidget(pick_btn)
        file_row.addWidget(open_json_btn)
        layout.addLayout(file_row)

        # Separation tiers
        sep_box = QGroupBox("Vocal separation")
        sep_layout = QVBoxLayout(sep_box)
        self.tier_group = QButtonGroup(self)
        skip_rb = QRadioButton("Skip separation (already-clean vocals)")
        self.tier_group.addButton(skip_rb, SKIP_TIER)
        sep_layout.addWidget(skip_rb)
        for tier in TIERS.values():
            rb = QRadioButton(f"Tier {tier.tier} — {tier.label}")
            rb.setToolTip(tier.description)
            self.tier_group.addButton(rb, tier.tier)
            sep_layout.addWidget(rb)
        custom_row = QHBoxLayout()
        custom_rb = QRadioButton("Custom model:")
        self.tier_group.addButton(custom_rb, CUSTOM_TIER)
        self.custom_model_edit = QLineEdit(self.config.custom_model_filename)
        self.custom_model_edit.setPlaceholderText(
            "model filename in the model directory (see Models tab)")
        custom_row.addWidget(custom_rb)
        custom_row.addWidget(self.custom_model_edit, stretch=1)
        sep_layout.addLayout(custom_row)
        btn = self.tier_group.button(self.config.separation_tier)
        (btn or self.tier_group.button(2)).setChecked(True)
        self.normalize_check = QCheckBox(
            "Loudness-normalize first (EBU R128) — helps quiet/badly mastered files")
        self.normalize_check.setChecked(self.config.normalize_loudness)
        sep_layout.addWidget(self.normalize_check)
        layout.addWidget(sep_box)

        # Transcription settings
        tr_box = QGroupBox("Transcription (faster-whisper)")
        form = QFormLayout(tr_box)
        slider_row = QHBoxLayout()
        self.size_slider = QSlider(Qt.Orientation.Horizontal)
        self.size_slider.setRange(0, len(WHISPER_SIZES) - 1)
        self.size_slider.setPageStep(1)
        self.size_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.size_label = QLabel()
        self.size_slider.valueChanged.connect(
            lambda v: self.size_label.setText(WHISPER_SIZES[v]))
        try:
            self.size_slider.setValue(WHISPER_SIZES.index(self.config.whisper_model))
        except ValueError:
            self.size_slider.setValue(2)
        self.size_label.setText(WHISPER_SIZES[self.size_slider.value()])
        slider_row.addWidget(self.size_slider, stretch=1)
        slider_row.addWidget(self.size_label)
        form.addRow("Model size", slider_row)

        self.device_combo = QComboBox()
        self.device_combo.addItems(["auto", "cpu", "cuda"])
        self.device_combo.setCurrentText(self.config.device)
        form.addRow("Device", self.device_combo)
        self.compute_combo = QComboBox()
        self.compute_combo.addItems(COMPUTE_TYPES)
        self.compute_combo.setCurrentText(self.config.compute_type)
        form.addRow("Compute type", self.compute_combo)
        self.lang_edit = QLineEdit(self.config.language)
        self.lang_edit.setPlaceholderText("auto-detect")
        form.addRow("Language", self.lang_edit)
        self.vad_spin = QSpinBox()
        self.vad_spin.setRange(50, 5000)
        self.vad_spin.setSingleStep(50)
        self.vad_spin.setSuffix(" ms")
        self.vad_spin.setValue(self.config.vad_min_silence_ms)
        self.vad_spin.setToolTip(
            "VAD minimum silence duration — higher values merge phrases, "
            "lower values split more aggressively; prevents phantom lines "
            "in instrumental sections")
        form.addRow("VAD min silence", self.vad_spin)
        layout.addWidget(tr_box)

        # Optional alignment
        align_box = QGroupBox(
            "Official lyrics (optional) — align known lyrics to the audio timing "
            "instead of trusting Whisper's wording")
        align_layout = QVBoxLayout(align_box)
        self.lyrics_text = QPlainTextEdit()
        self.lyrics_text.setPlaceholderText(
            "Paste official lyrics here (one line per lyric line), or fetch from LRCLIB…")
        self.lyrics_text.setMaximumHeight(120)
        align_layout.addWidget(self.lyrics_text)
        fetch_row = QHBoxLayout()
        self.artist_edit = QLineEdit()
        self.artist_edit.setPlaceholderText("artist")
        self.title_edit = QLineEdit()
        self.title_edit.setPlaceholderText("title")
        fetch_btn = QPushButton("Fetch from LRCLIB")
        fetch_btn.clicked.connect(self._fetch_lyrics)
        fetch_row.addWidget(self.artist_edit)
        fetch_row.addWidget(self.title_edit)
        fetch_row.addWidget(fetch_btn)
        align_layout.addLayout(fetch_row)
        layout.addWidget(align_box)

        # Run + queue + progress
        run_row = QHBoxLayout()
        self.run_btn = QPushButton("Transcribe")
        self.run_btn.setDefault(True)
        self.run_btn.clicked.connect(self._run)
        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.setEnabled(False)
        self.cancel_btn.clicked.connect(self._cancel)
        queue_btn = QPushButton("Add to queue instead")
        queue_btn.clicked.connect(self._enqueue)
        run_row.addWidget(self.run_btn)
        run_row.addWidget(self.cancel_btn)
        run_row.addWidget(queue_btn)
        run_row.addStretch(1)
        layout.addLayout(run_row)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 1000)
        self.progress_label = QLabel("Idle.")
        layout.addWidget(self.progress_bar)
        layout.addWidget(self.progress_label)
        layout.addStretch(1)
        return page

    # --- drag & drop -----------------------------------------------------------

    def dragEnterEvent(self, event) -> None:
        if any(Path(u.toLocalFile()).suffix.lower() in SUPPORTED_EXTENSIONS
               for u in event.mimeData().urls()):
            event.acceptProposedAction()

    def dropEvent(self, event) -> None:
        paths = [Path(u.toLocalFile()) for u in event.mimeData().urls()
                 if Path(u.toLocalFile()).suffix.lower() in SUPPORTED_EXTENSIONS]
        if not paths:
            return
        self.file_edit.setText(str(paths[0]))
        self._autofill_tags(paths[0])
        for extra in paths[1:]:
            self.queue_page.add_job(extra)
        if len(paths) > 1:
            self.tabs.setCurrentWidget(self.queue_page)

    # --- actions -------------------------------------------------------------------

    def _pick_file(self) -> None:
        patterns = " ".join(f"*{e}" for e in sorted(SUPPORTED_EXTENSIONS))
        path, _ = QFileDialog.getOpenFileName(self, "Choose audio file", "",
                                              f"Audio files ({patterns})")
        if path:
            self.file_edit.setText(path)
            self._autofill_tags(Path(path))

    def _autofill_tags(self, source: Path) -> None:
        title, artist = read_tags(source)
        if title and not self.title_edit.text().strip():
            self.title_edit.setText(title)
        if artist and not self.artist_edit.text().strip():
            self.artist_edit.setText(artist)

    def _open_project(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Open project JSON", "",
                                              "LyricSync JSON (*.json)")
        if not path:
            return
        try:
            doc = LyricDocument.load_json(path)
        except (OSError, ValueError, KeyError) as e:
            QMessageBox.critical(self, "Open project", f"Could not load:\n{e}")
            return
        self.editor_page.set_document(doc)
        self.tabs.setCurrentWidget(self.editor_page)

    def _collect_options(self) -> PipelineOptions:
        cfg = self.config
        cfg.separation_tier = self.tier_group.checkedId()
        cfg.custom_model_filename = self.custom_model_edit.text().strip()
        cfg.normalize_loudness = self.normalize_check.isChecked()
        cfg.whisper_model = WHISPER_SIZES[self.size_slider.value()]
        cfg.device = self.device_combo.currentText()
        cfg.compute_type = self.compute_combo.currentText()
        cfg.language = self.lang_edit.text().strip()
        cfg.vad_min_silence_ms = self.vad_spin.value()
        cfg.save()
        return PipelineOptions.from_config(cfg)

    def _run(self) -> None:
        source = Path(self.file_edit.text().strip())
        if not source.is_file():
            QMessageBox.warning(self, "Transcribe", "Choose an audio file first.")
            return
        self._source = source
        options = self._collect_options()

        self.run_btn.setEnabled(False)
        self.cancel_btn.setEnabled(True)
        worker = FunctionWorker(run_pipeline, source, options)
        worker.progressed.connect(self._on_progress)
        worker.succeeded.connect(self._on_pipeline_done)
        worker.failed.connect(self._on_pipeline_failed)
        # succeeded/failed fire from inside run(), just before it returns — the
        # worker thread hasn't necessarily joined yet. Only drop the last Python
        # reference once QThread's own finished() confirms the thread has
        # actually stopped, or Qt logs "QThread: Destroyed while thread is
        # still running".
        worker.finished.connect(self._on_pipeline_worker_finished)
        self._pipeline_worker = worker
        worker.start()

    def _cancel(self) -> None:
        if self._pipeline_worker:
            self._pipeline_worker.cancel()
            self.progress_label.setText("Cancelling…")

    def _enqueue(self) -> None:
        source = Path(self.file_edit.text().strip())
        if not source.is_file():
            QMessageBox.warning(self, "Queue", "Choose an audio file first.")
            return
        self._collect_options()  # persist current settings; queue reads config
        self.queue_page.add_job(source)
        self.tabs.setCurrentWidget(self.queue_page)

    def _fetch_lyrics(self) -> None:
        from lyricsync.core.alignment import fetch_lrclib_lyrics

        artist = self.artist_edit.text().strip()
        title = self.title_edit.text().strip()
        if not artist or not title:
            QMessageBox.information(self, "LRCLIB", "Enter artist and title first.")
            return
        text = fetch_lrclib_lyrics(artist, title)
        if text:
            self.lyrics_text.setPlainText(text)
        else:
            QMessageBox.information(self, "LRCLIB",
                                    "No lyrics found (or network unavailable).")

    # --- worker callbacks --------------------------------------------------------

    @Slot(str, float)
    def _on_progress(self, message: str, fraction: float) -> None:
        self.progress_bar.setValue(int(fraction * 1000))
        self.progress_label.setText(message)

    @Slot(object)
    def _on_pipeline_done(self, doc: LyricDocument) -> None:
        self._reset_run_ui()

        official = self.lyrics_text.toPlainText().strip()
        if official:
            try:
                doc = make_engine().align(doc, official)
                split_document(doc, make_splitter(self.config.syllable_backend,
                                                  self.config.syllable_lang))
            except ValueError as e:
                QMessageBox.warning(self, "Alignment",
                                    f"Alignment failed ({e}); showing raw transcription.")

        if self._source:
            autosave = self._source.with_suffix(self._source.suffix + ".lyricsync.json")
            doc.save_json(autosave)
            self.progress_label.setText(f"Done — autosaved {autosave.name}")

        self.editor_page.set_document(doc)
        self.tabs.setCurrentWidget(self.editor_page)

    @Slot(str)
    def _on_pipeline_failed(self, message: str) -> None:
        self._reset_run_ui()
        self.progress_label.setText("Failed.")
        if message != "Cancelled.":
            QMessageBox.critical(self, "Pipeline failed", message)
        else:
            self.progress_label.setText("Cancelled.")

    def _reset_run_ui(self) -> None:
        self.run_btn.setEnabled(True)
        self.cancel_btn.setEnabled(False)

    def _on_pipeline_worker_finished(self) -> None:
        self._pipeline_worker = None

    @Slot(object)
    def _on_queue_job_finished(self, job) -> None:
        # Open the most recent finished job if the editor is empty.
        if job.result and self.editor_page.doc is None:
            self.editor_page.set_document(job.result)
