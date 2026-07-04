"""Model manager page: manifest table, disk usage, custom model install
with a real progress bar (httpx streaming → Qt signal), manual delete.

No automatic eviction — surprise deletions are worse than manual cleanup.
"""

from __future__ import annotations

import time
from pathlib import Path

from PySide6.QtCore import Slot
from PySide6.QtWidgets import (
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from lyricsync.config import AppConfig
from lyricsync.models.catalog import ensure_builtin_tier_records, install_custom_model
from lyricsync.models.manifest import ModelManifest
from lyricsync.gui.workers import DownloadWorker
from lyricsync.utils.logs import get_logger

log = get_logger("models_page")


class ModelsPage(QWidget):
    def __init__(self, config: AppConfig, parent=None):
        super().__init__(parent)
        self.config = config
        self.manifest = ModelManifest(config.model_dir)
        ensure_builtin_tier_records(self.manifest)
        self._download_worker: DownloadWorker | None = None
        self._build_ui()
        self.refresh()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)

        self.dir_label = QLabel()
        self.dir_label.setWordWrap(True)
        layout.addWidget(self.dir_label)

        self.table = QTableWidget(0, 6)
        self.table.setHorizontalHeaderLabels(
            ["Model", "Tier", "Architecture", "Source", "On disk", "Last used"])
        self.table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        layout.addWidget(self.table)

        row = QHBoxLayout()
        refresh_btn = QPushButton("Refresh")
        refresh_btn.clicked.connect(self.refresh)
        delete_btn = QPushButton("Delete selected model files")
        delete_btn.clicked.connect(self._delete_selected)
        row.addWidget(refresh_btn)
        row.addWidget(delete_btn)
        row.addStretch(1)
        layout.addLayout(row)

        box = QGroupBox("Install custom separation model")
        form = QFormLayout(box)
        self.name_edit = QLineEdit()
        self.name_edit.setPlaceholderText("Display name (optional)")
        self.url_edit = QLineEdit()
        self.url_edit.setPlaceholderText("https://…/model.ckpt or .onnx")
        self.config_url_edit = QLineEdit()
        self.config_url_edit.setPlaceholderText(
            "Sidecar YAML config URL (Roformer/MDXC models often need one)")
        self.hf_edit = QLineEdit()
        self.hf_edit.setPlaceholderText("Hugging Face repo, e.g. user/repo")
        self.hf_file_edit = QLineEdit()
        self.hf_file_edit.setPlaceholderText("filename inside the repo")
        form.addRow("Name", self.name_edit)
        form.addRow("Direct URL", self.url_edit)
        form.addRow("Config URL", self.config_url_edit)
        form.addRow("HF repo", self.hf_edit)
        form.addRow("HF filename", self.hf_file_edit)

        btns = QHBoxLayout()
        self.install_btn = QPushButton("Download && install")
        self.install_btn.clicked.connect(self._install_from_form)
        local_btn = QPushButton("Install from local file…")
        local_btn.clicked.connect(self._install_local)
        self.cancel_dl_btn = QPushButton("Cancel download")
        self.cancel_dl_btn.setEnabled(False)
        self.cancel_dl_btn.clicked.connect(self._cancel_download)
        btns.addWidget(self.install_btn)
        btns.addWidget(local_btn)
        btns.addWidget(self.cancel_dl_btn)
        form.addRow(btns)

        self.dl_bar = QProgressBar()
        self.dl_bar.setRange(0, 100)
        self.dl_bar.setValue(0)
        form.addRow(self.dl_bar)
        layout.addWidget(box)

    # --- table ---------------------------------------------------------------

    def refresh(self) -> None:
        self.manifest = ModelManifest(self.config.model_dir)
        ensure_builtin_tier_records(self.manifest)
        usage = self.manifest.disk_usage()
        records = list(self.manifest.records.values())
        untracked = self.manifest.untracked_files()

        self.table.setRowCount(len(records) + len(untracked))
        for i, rec in enumerate(records):
            on_disk = usage.get(rec.model_id, 0)
            last = time.strftime("%Y-%m-%d", time.localtime(rec.last_used)) \
                if rec.last_used else "never"
            cells = [rec.display_name, str(rec.tier), rec.architecture,
                     rec.source, _fmt_size(on_disk) if on_disk else "not downloaded",
                     last]
            for col, text in enumerate(cells):
                self.table.setItem(i, col, QTableWidgetItem(text))
        for j, f in enumerate(untracked):
            i = len(records) + j
            cells = [f.name, "-", "-", "auto-downloaded",
                     _fmt_size(f.stat().st_size), "-"]
            for col, text in enumerate(cells):
                self.table.setItem(i, col, QTableWidgetItem(text))
        self.table.resizeColumnsToContents()

        total = sum(usage.values()) + sum(f.stat().st_size for f in untracked)
        self.dir_label.setText(
            f"Model directory: {self.config.model_dir}   "
            f"(total on disk: {_fmt_size(total)}). Built-in tier models are "
            f"auto-downloaded on first use by audio-separator.")

    def _delete_selected(self) -> None:
        rows = sorted({i.row() for i in self.table.selectedIndexes()})
        if not rows:
            return
        records = list(self.manifest.records.values())
        names = []
        for r in rows:
            if r < len(records):
                names.append(records[r].display_name)
            else:
                names.append(self.manifest.untracked_files()[r - len(records)].name)
        confirm = QMessageBox.question(
            self, "Delete model files",
            "Delete the files for:\n  " + "\n  ".join(names) +
            "\n\nThe manifest entry for built-in tiers is kept; weights "
            "re-download on next use.")
        if confirm != QMessageBox.StandardButton.Yes:
            return
        untracked = self.manifest.untracked_files()
        for r in rows:
            if r < len(records):
                rec = records[r]
                for name in rec.filenames:
                    (Path(self.config.model_dir) / name).unlink(missing_ok=True)
                if rec.tier == 4:
                    self.manifest.remove(rec.model_id)
            else:
                untracked[r - len(records)].unlink(missing_ok=True)
        self.refresh()

    # --- custom install ---------------------------------------------------------

    def _install_from_form(self) -> None:
        url = self.url_edit.text().strip()
        hf_repo = self.hf_edit.text().strip()
        hf_file = self.hf_file_edit.text().strip()
        if not url and not (hf_repo and hf_file):
            QMessageBox.warning(self, "Install model",
                                "Provide a direct URL or a Hugging Face repo + filename.")
            return

        self.install_btn.setEnabled(False)
        self.cancel_dl_btn.setEnabled(True)
        worker = DownloadWorker(
            install_custom_model, self.manifest,
            display_name=self.name_edit.text().strip(),
            url=url, hf_repo=hf_repo, hf_filename=hf_file,
            config_url=self.config_url_edit.text().strip(),
        )
        worker.bytes_progressed.connect(self._on_dl_progress)
        worker.succeeded.connect(self._on_install_done)
        worker.failed.connect(self._on_install_failed)
        # Only drop the last reference once the thread has actually stopped
        # (finished fires after succeeded/failed, once run() has returned) —
        # otherwise Qt logs "QThread: Destroyed while thread is still running".
        worker.finished.connect(self._on_download_worker_finished)
        self._download_worker = worker
        worker.start()

    def _install_local(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Select model file", "",
            "Model files (*.onnx *.ckpt *.pth *.yaml);;All files (*)")
        if not path:
            return
        try:
            rec = install_custom_model(
                self.manifest,
                display_name=self.name_edit.text().strip(),
                local_path=path)
        except (OSError, ValueError) as e:
            QMessageBox.critical(self, "Install model", str(e))
            return
        QMessageBox.information(self, "Install model",
                                f"Installed {rec.display_name}.")
        self.refresh()

    def _cancel_download(self) -> None:
        if self._download_worker:
            self._download_worker.cancel()

    @Slot(int, int)
    def _on_dl_progress(self, done: int, total: int) -> None:
        if total > 0:
            self.dl_bar.setValue(int(done / total * 100))
            self.dl_bar.setFormat(f"{_fmt_size(done)} / {_fmt_size(total)}")

    @Slot(object)
    def _on_install_done(self, rec) -> None:
        self._reset_download_ui()
        QMessageBox.information(self, "Install model",
                                f"Installed {rec.display_name} ({rec.architecture}).")
        self.refresh()

    @Slot(str)
    def _on_install_failed(self, msg: str) -> None:
        self._reset_download_ui()
        QMessageBox.warning(self, "Install model",
                            f"Download failed or was cancelled:\n{msg.splitlines()[0]}\n\n"
                            "Partial downloads resume automatically on retry.")

    def _reset_download_ui(self) -> None:
        self.install_btn.setEnabled(True)
        self.cancel_dl_btn.setEnabled(False)
        self.dl_bar.setValue(0)
        self.dl_bar.setFormat("%p%")

    def _on_download_worker_finished(self) -> None:
        self._download_worker = None


def _fmt_size(n: int | float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.0f} {unit}" if unit == "B" else f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} TB"
