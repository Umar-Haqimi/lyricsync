"""Headless end-to-end pipeline: preprocess → separate → transcribe → model.

Pure orchestration with progress/cancel callbacks — usable from the CLI
and from GUI worker threads identically. Stage weights give a single
smooth 0..1 progress value across the whole run.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from lyricsync.config import AppConfig
from lyricsync.core import separation
from lyricsync.core.audio import prepare_working_wav
from lyricsync.core.model import LyricDocument
from lyricsync.core.syllables import make_splitter, split_document
from lyricsync.core.transcription import transcribe
from lyricsync.utils.logs import get_logger
from lyricsync.utils.paths import work_dir

log = get_logger("pipeline")

ProgressFn = Callable[[str, float], None]


@dataclass
class PipelineOptions:
    separation_tier: int = 2                # 0 = skip
    custom_model_filename: str = ""
    normalize_loudness: bool = False
    whisper_model: str = "small"
    device: str = "auto"
    compute_type: str = "auto"
    language: str = ""
    vad_min_silence_ms: int = 500
    split_syllables: bool = True
    syllable_backend: str = "hybrid"
    syllable_lang: str = "en"
    model_dir: str = ""
    keep_stems: bool = False
    extra: dict = field(default_factory=dict)

    @classmethod
    def from_config(cls, cfg: AppConfig) -> "PipelineOptions":
        return cls(
            separation_tier=cfg.separation_tier,
            custom_model_filename=cfg.custom_model_filename,
            normalize_loudness=cfg.normalize_loudness,
            whisper_model=cfg.whisper_model,
            device=cfg.device,
            compute_type=cfg.compute_type,
            language=cfg.language,
            vad_min_silence_ms=cfg.vad_min_silence_ms,
            syllable_backend=cfg.syllable_backend,
            syllable_lang=cfg.syllable_lang,
            model_dir=cfg.model_dir,
        )


# Rough wall-clock weights per stage for a combined progress fraction.
_W_PREP, _W_SEP, _W_TRANS, _W_SYL = 0.05, 0.45, 0.48, 0.02


def run_pipeline(
    source: str | Path,
    options: PipelineOptions,
    progress: ProgressFn | None = None,
    cancel_check: Callable[[], bool] | None = None,
) -> LyricDocument:
    """Run the full pipeline on one file and return the populated document."""
    source = Path(source)

    def sub_progress(offset: float, weight: float):
        def fn(msg: str, frac: float) -> None:
            if progress:
                progress(msg, offset + weight * max(0.0, min(1.0, frac)))
        return fn

    def cancelled() -> bool:
        return bool(cancel_check and cancel_check())

    # 1. Preprocess ----------------------------------------------------
    sub_progress(0.0, _W_PREP)("Converting audio…", 0.2)
    wav = prepare_working_wav(source, normalize=options.normalize_loudness)
    if cancelled():
        raise PipelineCancelled()

    # 2. Separation (optional) -----------------------------------------
    transcribe_input = wav
    tier = options.separation_tier
    if tier != separation.SKIP_TIER:
        model_filename = separation.resolve_model_filename(tier, options.custom_model_filename)
        stem_dir = work_dir() / "stems"
        vocals = separation.separate_vocals(
            wav, model_filename,
            model_dir=options.model_dir, output_dir=stem_dir,
            progress=sub_progress(_W_PREP, _W_SEP),
        )
        transcribe_input = vocals
    else:
        sub_progress(_W_PREP, _W_SEP)("Skipping separation.", 1.0)
    if cancelled():
        raise PipelineCancelled()

    # 3. Transcription --------------------------------------------------
    doc = transcribe(
        transcribe_input,
        model_size=options.whisper_model,
        device=options.device,
        compute_type=options.compute_type,
        language=options.language or None,
        vad_min_silence_ms=options.vad_min_silence_ms,
        progress=sub_progress(_W_PREP + _W_SEP, _W_TRANS),
        cancel_check=cancel_check,
    )
    if cancelled():
        raise PipelineCancelled()

    # The document points at the ORIGINAL file (that's what exports/embedding
    # reference); working/stem paths are provenance in metadata.
    doc.metadata["transcribed_from"] = str(transcribe_input)
    doc.metadata["separation_tier"] = tier
    doc.metadata["normalized"] = options.normalize_loudness
    doc.audio_source = str(source)

    # 4. Syllables -------------------------------------------------------
    if options.split_syllables:
        sub_progress(_W_PREP + _W_SEP + _W_TRANS, _W_SYL)("Splitting syllables…", 0.5)
        split_document(doc, make_splitter(options.syllable_backend, options.syllable_lang))

    if not options.keep_stems and transcribe_input != wav:
        # Keep the working wav (editor plays it); stems are disposable.
        try:
            Path(transcribe_input).unlink(missing_ok=True)
        except OSError:
            log.warning("could not remove stem %s", transcribe_input)

    if progress:
        progress("Done.", 1.0)
    return doc


class PipelineCancelled(Exception):
    """Raised when cancel_check reports True between stages."""
