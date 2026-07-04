"""Transcription via faster-whisper with word-level timestamps.

Produces the canonical LyricDocument directly; per-word probabilities are
surfaced as `confidence` for the correction UI's low-confidence flagging.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

from lyricsync.core.model import Line, LyricDocument, Word
from lyricsync.utils.logs import get_logger

log = get_logger("transcription")

ProgressFn = Callable[[str, float], None]

WHISPER_SIZES = ["tiny", "base", "small", "medium", "large-v3", "turbo"]
COMPUTE_TYPES = ["auto", "int8", "int8_float16", "float16", "float32"]


def resolve_device(device: str = "auto") -> tuple[str, str]:
    """Resolve (device, default_compute_type) honoring auto-detection."""
    if device == "auto":
        try:
            import torch

            device = "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            device = "cpu"
    return device, ("float16" if device == "cuda" else "int8")


def transcribe(
    audio_path: str | Path,
    model_size: str = "small",
    device: str = "auto",
    compute_type: str = "auto",
    language: str | None = None,
    vad_min_silence_ms: int = 500,
    progress: ProgressFn | None = None,
    cancel_check: Callable[[], bool] | None = None,
) -> LyricDocument:
    """Blocking transcription — run on a worker thread.

    VAD filtering (Silero, built into faster-whisper) suppresses phantom
    lines during instrumental sections; `vad_min_silence_ms` is exposed in
    the UI. `cancel_check` is polled between segments.
    """
    from faster_whisper import WhisperModel

    audio_path = Path(audio_path)
    resolved_device, default_ct = resolve_device(device)
    ct = default_ct if compute_type in ("", "auto") else compute_type

    if progress:
        progress(f"Loading Whisper {model_size} ({resolved_device}, {ct})…", 0.02)
    log.info("loading whisper %s device=%s compute_type=%s", model_size, resolved_device, ct)

    def load(device: str, compute_type: str) -> "WhisperModel":
        return WhisperModel(model_size, device=device, compute_type=compute_type)

    def start_transcribe(model: "WhisperModel"):
        # Eagerly triggers the first CUDA kernel launch (language detection),
        # so a missing runtime library (e.g. libcublas.so.12) surfaces here —
        # not just at construction time.
        return model.transcribe(
            str(audio_path),
            language=language or None,
            word_timestamps=True,
            vad_filter=True,
            vad_parameters={"min_silence_duration_ms": vad_min_silence_ms},
        )

    try:
        model = load(resolved_device, ct)
        segments, info = start_transcribe(model)
    except (RuntimeError, ValueError) as e:
        if resolved_device == "cuda":
            # Missing cuDNN/cuBLAS, driver mismatch, out of VRAM, etc. —
            # degrade to CPU rather than fail the whole job.
            log.warning("CUDA transcription failed (%s); falling back to CPU int8", e)
            if progress:
                progress("CUDA unavailable — falling back to CPU…", 0.02)
            model = load("cpu", "int8")
            segments, info = start_transcribe(model)
        else:
            raise

    doc = LyricDocument(
        audio_source=str(audio_path),
        language=info.language or "",
        metadata={
            "whisper_model": model_size,
            "device": resolved_device,
            "compute_type": ct,
            "language_probability": round(getattr(info, "language_probability", 0.0) or 0.0, 3),
            "vad_min_silence_ms": vad_min_silence_ms,
        },
    )

    total = getattr(info, "duration", 0.0) or 0.0
    for segment in segments:  # generator — transcription happens lazily here
        if cancel_check and cancel_check():
            log.info("transcription cancelled at %.1fs", segment.start)
            break
        words = [
            Word(
                text=w.word.strip(),
                start=float(w.start),
                end=float(w.end),
                confidence=float(w.probability),
            )
            for w in (segment.words or [])
            if w.word.strip()
        ]
        if not words:
            continue
        line = Line(words=words)
        line.recompute_bounds()
        doc.lines.append(line)
        if progress and total:
            frac = 0.05 + 0.95 * min(1.0, segment.end / total)
            progress(f"Transcribing… {segment.end:0.0f}s / {total:0.0f}s", frac)

    doc.sort_lines()
    log.info("transcription done: %d lines, language=%s", len(doc.lines), doc.language)
    return doc
