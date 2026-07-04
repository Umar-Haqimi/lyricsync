"""Settings page: storage location, editor thresholds, syllable backend,
batch parallelism. Transcription/separation settings live on the main
Transcribe tab where they're used."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from lyricsync.config import AppConfig


class SettingsPage(QWidget):
    def __init__(self, config: AppConfig, parent=None):
        super().__init__(parent)
        self.config = config
        layout = QVBoxLayout(self)

        storage = QGroupBox("Storage")
        form = QFormLayout(storage)
        self.model_dir_edit = QLineEdit(config.model_dir)
        browse = QPushButton("Browse…")
        browse.clicked.connect(self._browse_model_dir)
        row = QHBoxLayout()
        row.addWidget(self.model_dir_edit)
        row.addWidget(browse)
        form.addRow("Model directory", row)
        note = QLabel(
            "Models are persistent user data — keep this outside any project/"
            "build directory so packaging or redeployment can never wipe it.")
        note.setWordWrap(True)
        form.addRow(note)
        layout.addWidget(storage)

        editor = QGroupBox("Editor")
        eform = QFormLayout(editor)
        self.warn_spin = QDoubleSpinBox()
        self.warn_spin.setRange(0.0, 1.0)
        self.warn_spin.setSingleStep(0.05)
        self.warn_spin.setValue(config.confidence_warn_threshold)
        self.bad_spin = QDoubleSpinBox()
        self.bad_spin.setRange(0.0, 1.0)
        self.bad_spin.setSingleStep(0.05)
        self.bad_spin.setValue(config.confidence_bad_threshold)
        eform.addRow("Flag words below confidence (amber)", self.warn_spin)
        eform.addRow("Strong flag below confidence (red)", self.bad_spin)
        layout.addWidget(editor)

        syl = QGroupBox("Syllables (eLRC / karaoke)")
        sform = QFormLayout(syl)
        self.backend_combo = QComboBox()
        self.backend_combo.addItems(["hybrid", "pyphen"])
        self.backend_combo.setCurrentText(config.syllable_backend)
        self.backend_combo.setToolTip(
            "hybrid = pyphen + rule-based fallback + overrides (recommended); "
            "pyphen = plain hyphenation")
        self.lang_edit = QLineEdit(config.syllable_lang)
        sform.addRow("Splitter backend", self.backend_combo)
        sform.addRow("Language", self.lang_edit)
        layout.addWidget(syl)

        batch = QGroupBox("Batch queue")
        bform = QFormLayout(batch)
        self.parallel_spin = QSpinBox()
        self.parallel_spin.setRange(1, 4)
        self.parallel_spin.setValue(config.max_parallel_jobs)
        self.parallel_spin.setToolTip(
            "Keep at 1 unless you have VRAM/CPU to spare — a single job "
            "already saturates most machines.")
        bform.addRow("Max parallel jobs", self.parallel_spin)
        layout.addWidget(batch)

        save_btn = QPushButton("Save settings")
        save_btn.clicked.connect(self._save)
        layout.addWidget(save_btn)
        layout.addStretch(1)

    def _browse_model_dir(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "Choose model directory",
                                                self.model_dir_edit.text())
        if path:
            self.model_dir_edit.setText(path)

    def _save(self) -> None:
        cfg = self.config
        model_dir = self.model_dir_edit.text().strip()
        if model_dir:
            Path(model_dir).mkdir(parents=True, exist_ok=True)
            cfg.model_dir = model_dir
        cfg.confidence_warn_threshold = self.warn_spin.value()
        cfg.confidence_bad_threshold = self.bad_spin.value()
        cfg.syllable_backend = self.backend_combo.currentText()
        cfg.syllable_lang = self.lang_edit.text().strip() or "en"
        cfg.max_parallel_jobs = self.parallel_spin.value()
        cfg.save()
