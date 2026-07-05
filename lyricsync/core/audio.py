"""Audio input & preprocessing via the external ffmpeg binary.

All input formats (MP3/WAV/FLAC/M4A/OGG/…) are converted to a standard
16-bit 44.1 kHz stereo WAV working file before separation/transcription.
Also provides downsampled peak extraction for waveform display.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import numpy as np

from lyricsync.utils.logs import get_logger
from lyricsync.utils.paths import work_dir

log = get_logger("audio")

SUPPORTED_EXTENSIONS = {".mp3", ".wav", ".flac", ".m4a", ".ogg", ".opus", ".aac", ".wma", ".aiff"}


def read_tags(path: str | Path) -> tuple[str, str]:
    """Best-effort (title, artist) read from the file's embedded tags.

    Returns empty strings for whatever mutagen can't find (missing tags,
    unsupported format, corrupt file) rather than raising — this is only
    ever used to pre-fill a form field.
    """
    import mutagen

    try:
        tags = mutagen.File(str(path), easy=True)
    except Exception:
        return "", ""
    if not tags:
        return "", ""
    title = (tags.get("title") or [""])[0]
    artist = (tags.get("artist") or [""])[0]
    return title.strip(), artist.strip()


class FFmpegNotFoundError(RuntimeError):
    pass


def ensure_ffmpeg() -> str:
    path = shutil.which("ffmpeg")
    if not path:
        raise FFmpegNotFoundError(
            "ffmpeg was not found on PATH. Install it (e.g. 'sudo pacman -S ffmpeg', "
            "'brew install ffmpeg', or from ffmpeg.org) and restart the app."
        )
    return path


def probe_duration(path: str | Path) -> float:
    """Duration in seconds via ffprobe (falls back to 0.0 on failure)."""
    ffprobe = shutil.which("ffprobe")
    if not ffprobe:
        return 0.0
    try:
        out = subprocess.run(
            [ffprobe, "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", str(path)],
            capture_output=True, text=True, timeout=30, check=True,
        )
        return float(out.stdout.strip())
    except (subprocess.SubprocessError, ValueError):
        return 0.0


def prepare_working_wav(
    source: str | Path,
    normalize: bool = False,
    out_dir: Path | None = None,
) -> Path:
    """Convert `source` to a 16-bit 44.1 kHz stereo WAV working copy.

    normalize=True adds a single-pass EBU R128 loudness normalization
    (ffmpeg loudnorm) which improves separation/transcription consistency
    on quiet or badly mastered files.
    """
    ffmpeg = ensure_ffmpeg()
    source = Path(source)
    out_dir = out_dir or work_dir()
    suffix = "_norm" if normalize else ""
    dest = out_dir / f"{source.stem}{suffix}_work.wav"

    cmd = [ffmpeg, "-y", "-i", str(source)]
    if normalize:
        cmd += ["-af", "loudnorm=I=-16:TP=-1.5:LRA=11"]
    cmd += ["-ar", "44100", "-ac", "2", "-c:a", "pcm_s16le", str(dest)]

    log.info("ffmpeg: converting %s -> %s (normalize=%s)", source.name, dest.name, normalize)
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(
            f"ffmpeg failed converting {source.name}:\n{proc.stderr[-2000:]}"
        )
    return dest


def compute_peaks(wav_path: str | Path, target_bins: int = 4000) -> np.ndarray:
    """Downsampled (min, max) peak pairs for waveform display.

    Returns an array of shape (bins, 2) with values in [-1, 1]. Never
    renders every sample — a few thousand bins is plenty for a widget.
    """
    import soundfile as sf

    with sf.SoundFile(str(wav_path)) as f:
        frames = len(f)
        if frames == 0:
            return np.zeros((0, 2), dtype=np.float32)
        hop = max(1, frames // target_bins)
        peaks = np.zeros((min(target_bins, (frames + hop - 1) // hop), 2), dtype=np.float32)
        for i in range(peaks.shape[0]):
            f.seek(i * hop)
            block = f.read(hop, dtype="float32", always_2d=True)
            if block.size == 0:
                break
            mono = block.mean(axis=1)
            peaks[i, 0] = mono.min()
            peaks[i, 1] = mono.max()
    return peaks
