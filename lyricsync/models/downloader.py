"""Streaming, resumable model downloads via httpx.

Downloads go to `<name>.partial` and are renamed into place on completion.
On reconnect, an HTTP Range request resumes from the partial file's size
(when the server supports it; otherwise we start over). Progress is
reported through a plain callback — the GUI wires it to a Qt signal.
"""

from __future__ import annotations

import hashlib
import shutil
from pathlib import Path
from typing import Callable

import httpx

from lyricsync.utils.logs import get_logger

log = get_logger("downloader")

# (bytes_done, bytes_total_or_0) -> None
DownloadProgressFn = Callable[[int, int], None]


class DownloadCancelled(Exception):
    pass


def hf_url(repo: str, filename: str, revision: str = "main") -> str:
    return f"https://huggingface.co/{repo}/resolve/{revision}/{filename}"


def download_file(
    url: str,
    dest: str | Path,
    progress: DownloadProgressFn | None = None,
    cancel_check: Callable[[], bool] | None = None,
    expected_sha256: str = "",
    timeout: float = 30.0,
) -> Path:
    """Download `url` to `dest`, resuming a previous .partial if present."""
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    partial = dest.with_name(dest.name + ".partial")

    resume_from = partial.stat().st_size if partial.exists() else 0
    headers = {"User-Agent": "LyricSync/0.1"}
    if resume_from:
        headers["Range"] = f"bytes={resume_from}-"
        log.info("resuming %s from byte %d", dest.name, resume_from)

    with httpx.stream("GET", url, headers=headers, timeout=timeout,
                      follow_redirects=True) as r:
        if resume_from and r.status_code != 206:
            # Server ignored the Range request — restart from scratch.
            log.info("server did not honor Range; restarting %s", dest.name)
            resume_from = 0
            partial.unlink(missing_ok=True)
            r.raise_for_status()
        elif r.status_code >= 400:
            r.raise_for_status()

        total = int(r.headers.get("content-length", 0)) + resume_from
        mode = "ab" if resume_from else "wb"
        done = resume_from
        with open(partial, mode) as f:
            for chunk in r.iter_bytes(chunk_size=256 * 1024):
                if cancel_check and cancel_check():
                    log.info("download cancelled: %s (%d bytes kept for resume)",
                             dest.name, done)
                    raise DownloadCancelled(dest.name)
                f.write(chunk)
                done += len(chunk)
                if progress:
                    progress(done, total)

    if expected_sha256:
        actual = sha256_of(partial)
        if actual != expected_sha256.lower():
            partial.unlink(missing_ok=True)
            raise ValueError(
                f"sha256 mismatch for {dest.name}: expected {expected_sha256}, got {actual}"
            )

    partial.replace(dest)
    log.info("downloaded %s (%d bytes)", dest.name, dest.stat().st_size)
    return dest


def sha256_of(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def copy_local_model(src: str | Path, model_dir: str | Path) -> Path:
    """Install a model from a local file path by copying it into the model dir."""
    src = Path(src)
    dest = Path(model_dir) / src.name
    if src.resolve() != dest.resolve():
        shutil.copy2(src, dest)
    return dest
