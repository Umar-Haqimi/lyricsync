"""GUI entry point: `lyricsync` (or `python -m lyricsync.app`)."""

from __future__ import annotations

import sys

from lyricsync.config import AppConfig
from lyricsync.utils.logs import get_logger, setup_logging


def main() -> int:
    setup_logging()
    log = get_logger("app")

    from PySide6.QtWidgets import QApplication, QMessageBox

    from lyricsync.core.audio import FFmpegNotFoundError, ensure_ffmpeg
    from lyricsync.gui.main_window import MainWindow

    app = QApplication(sys.argv)
    app.setApplicationName("LyricSync")

    try:
        ensure_ffmpeg()
    except FFmpegNotFoundError as e:
        QMessageBox.critical(None, "ffmpeg missing", str(e))
        return 1

    config = AppConfig.load()
    window = MainWindow(config)
    window.show()
    log.info("LyricSync started")
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
