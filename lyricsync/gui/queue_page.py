"""Batch queue: multiple files processed sequentially on worker threads.

A simple in-app job list (no external broker) with per-job status
(queued / running / done / failed / cancelled), progress and cancellation.
Jobs run one at a time by default — separation + transcription already
saturate CPU/VRAM, so parallelism is bounded by config.max_parallel_jobs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

from PySide6.QtCore import Qt, Signal, Slot
from PySide6.QtWidgets import (
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from lyricsync.config import AppConfig
from lyricsync.core.audio import SUPPORTED_EXTENSIONS
from lyricsync.core.model import LyricDocument
from lyricsync.core.pipeline import PipelineOptions, run_pipeline
from lyricsync.gui.workers import FunctionWorker
from lyricsync.utils.logs import get_logger

log = get_logger("queue")


class JobStatus(Enum):
    QUEUED = "queued"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class Job:
    source: Path
    options: PipelineOptions
    status: JobStatus = JobStatus.QUEUED
    message: str = ""
    progress: float = 0.0
    result: LyricDocument | None = None
    worker: FunctionWorker | None = field(default=None, repr=False)


class QueuePage(QWidget):
    job_finished = Signal(object)  # Job — main window opens it in the editor

    def __init__(self, config: AppConfig, parent=None):
        super().__init__(parent)
        self.config = config
        self.jobs: list[Job] = []
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        top = QHBoxLayout()
        add_btn = QPushButton("Add files…")
        add_btn.clicked.connect(self._add_files)
        self.start_btn = QPushButton("Start queue")
        self.start_btn.clicked.connect(self._pump)
        self.cancel_btn = QPushButton("Cancel selected")
        self.cancel_btn.clicked.connect(self._cancel_selected)
        clear_btn = QPushButton("Clear finished")
        clear_btn.clicked.connect(self._clear_finished)
        for w in (add_btn, self.start_btn, self.cancel_btn, clear_btn):
            top.addWidget(w)
        top.addStretch(1)
        layout.addLayout(top)

        self.table = QTableWidget(0, 4)
        self.table.setHorizontalHeaderLabels(["File", "Status", "Progress", "Detail"])
        self.table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        layout.addWidget(self.table)

        self.hint = QLabel("Queued files are processed with the current settings "
                           "from the Transcribe tab at the moment they were added.")
        self.hint.setWordWrap(True)
        layout.addWidget(self.hint)

    # --- queue operations ---------------------------------------------------

    def _add_files(self) -> None:
        patterns = " ".join(f"*{e}" for e in sorted(SUPPORTED_EXTENSIONS))
        paths, _ = QFileDialog.getOpenFileNames(
            self, "Add audio files", "", f"Audio files ({patterns})")
        for p in paths:
            self.add_job(Path(p))

    def add_job(self, source: Path) -> None:
        job = Job(source=source, options=PipelineOptions.from_config(self.config))
        self.jobs.append(job)
        row = self.table.rowCount()
        self.table.insertRow(row)
        self.table.setItem(row, 0, QTableWidgetItem(source.name))
        self.table.setItem(row, 1, QTableWidgetItem(job.status.value))
        bar = QProgressBar()
        bar.setRange(0, 100)
        self.table.setCellWidget(row, 2, bar)
        self.table.setItem(row, 3, QTableWidgetItem(""))
        self.table.resizeColumnsToContents()

    def _running_count(self) -> int:
        return sum(1 for j in self.jobs if j.status is JobStatus.RUNNING)

    def _pump(self) -> None:
        """Start queued jobs up to the parallelism bound."""
        limit = max(1, self.config.max_parallel_jobs)
        for job in self.jobs:
            if self._running_count() >= limit:
                break
            if job.status is JobStatus.QUEUED:
                self._start(job)

    def _start(self, job: Job) -> None:
        job.status = JobStatus.RUNNING
        self._set_status(self._row_of(job), job)

        worker = FunctionWorker(run_pipeline, job.source, job.options)
        job.worker = worker
        # Row is looked up at callback time — "Clear finished" can reshuffle
        # rows while a job is still running.
        worker.progressed.connect(lambda msg, frac, j=job: self._on_progress(self._row_of(j), j, msg, frac))
        worker.succeeded.connect(lambda doc, j=job: self._on_done(self._row_of(j), j, doc))
        worker.failed.connect(lambda msg, j=job: self._on_failed(self._row_of(j), j, msg))
        worker.start()
        log.info("queue: started %s", job.source.name)

    def _row_of(self, job: Job) -> int:
        try:
            return self.jobs.index(job)
        except ValueError:
            return -1

    def _cancel_selected(self) -> None:
        for index in {i.row() for i in self.table.selectedIndexes()}:
            if index < len(self.jobs):
                job = self.jobs[index]
                if job.status is JobStatus.RUNNING and job.worker:
                    job.worker.cancel()
                    job.message = "Cancelling…"
                elif job.status is JobStatus.QUEUED:
                    job.status = JobStatus.CANCELLED
                self._set_status(index, job)

    def _clear_finished(self) -> None:
        keep = [j for j in self.jobs
                if j.status in (JobStatus.QUEUED, JobStatus.RUNNING)]
        self.jobs = keep
        self.table.setRowCount(0)
        for job in keep:
            row = self.table.rowCount()
            self.table.insertRow(row)
            self.table.setItem(row, 0, QTableWidgetItem(job.source.name))
            self.table.setItem(row, 1, QTableWidgetItem(job.status.value))
            bar = QProgressBar()
            bar.setRange(0, 100)
            bar.setValue(int(job.progress * 100))
            self.table.setCellWidget(row, 2, bar)
            self.table.setItem(row, 3, QTableWidgetItem(job.message))

    # --- worker callbacks (delivered on GUI thread via signals) -------------

    def _on_progress(self, row: int, job: Job, msg: str, frac: float) -> None:
        job.progress = frac
        job.message = msg
        bar = self.table.cellWidget(row, 2)
        if isinstance(bar, QProgressBar):
            bar.setValue(int(frac * 100))
        item = self.table.item(row, 3)
        if item:
            item.setText(msg)

    def _on_done(self, row: int, job: Job, doc: LyricDocument) -> None:
        job.status = JobStatus.DONE
        job.result = doc
        job.message = f"{len(doc.lines)} lines"
        # Autosave the project JSON next to the source.
        out = job.source.with_suffix(job.source.suffix + ".lyricsync.json")
        doc.save_json(out)
        self._set_status(row, job)
        self.job_finished.emit(job)
        self._pump()

    def _on_failed(self, row: int, job: Job, msg: str) -> None:
        cancelled = job.worker.cancelled if job.worker else False
        job.status = JobStatus.CANCELLED if cancelled else JobStatus.FAILED
        job.message = msg.splitlines()[0] if msg else "failed"
        log.error("queue: %s failed: %s", job.source.name, msg)
        self._set_status(row, job)
        self._pump()

    def _set_status(self, row: int, job: Job) -> None:
        item = self.table.item(row, 1)
        if item:
            item.setText(job.status.value)
        detail = self.table.item(row, 3)
        if detail:
            detail.setText(job.message)
