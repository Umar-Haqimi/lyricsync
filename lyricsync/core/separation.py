"""Vocal separation via audio-separator, organized into quality tiers.

Tier models are referenced by their filename in audio-separator's hosted
catalog, so first use auto-downloads weights (and any sidecar YAML config
for MDXC/Roformer models) into the configured model directory. Custom
models installed by the model manager are loaded the same way.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from lyricsync.utils.logs import get_logger

log = get_logger("separation")

ProgressFn = Callable[[str, float], None]  # (message, fraction 0..1)


@dataclass(frozen=True)
class SeparationTier:
    tier: int
    label: str
    model_filename: str
    description: str


TIERS: dict[int, SeparationTier] = {
    1: SeparationTier(
        1, "Potato (MDX-Net Inst HQ 3)",
        "UVR-MDX-NET-Inst_HQ_3.onnx",
        "CPU-friendly and fast; good enough for clean mixes.",
    ),
    2: SeparationTier(
        2, "Balanced (htdemucs_ft)",
        "htdemucs_ft.yaml",
        "Good default; moderate resource use.",
    ),
    3: SeparationTier(
        3, "Max Quality (MelBand Roformer INSTV8 by Gabox)",
        "mel_band_roformer_instrumental_instv8_gabox.ckpt",
        "GPU-heavy; best separation quality.",
    ),
}

SKIP_TIER = 0
CUSTOM_TIER = 4


def resolve_model_filename(tier: int, custom_filename: str = "") -> str:
    if tier == CUSTOM_TIER:
        if not custom_filename:
            raise ValueError("Custom separation tier selected but no model filename set.")
        return custom_filename
    if tier in TIERS:
        return TIERS[tier].model_filename
    raise ValueError(f"Unknown separation tier: {tier}")


def separate_vocals(
    wav_path: str | Path,
    model_filename: str,
    model_dir: str | Path,
    output_dir: str | Path,
    progress: ProgressFn | None = None,
) -> Path:
    """Run separation and return the path of the vocals stem WAV.

    Blocking — callers run this on a worker thread (see gui.workers).
    """
    from audio_separator.separator import Separator

    wav_path = Path(wav_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if progress:
        progress(f"Loading separation model {model_filename}…", 0.05)

    separator = Separator(
        model_file_dir=str(model_dir),
        output_dir=str(output_dir),
        output_format="wav",
    )
    # Downloads weights + sidecar config on first use for catalog models.
    separator.load_model(model_filename=model_filename)

    if progress:
        progress("Separating vocals…", 0.20)

    output_files = separator.separate(str(wav_path))
    log.info("separation produced: %s", output_files)

    vocals = _pick_vocals_stem([Path(output_dir) / f for f in output_files])
    if vocals is None:
        raise RuntimeError(
            f"Separation finished but no vocals stem was found among: {output_files}"
        )
    if progress:
        progress("Separation complete.", 1.0)
    return vocals


def _pick_vocals_stem(files: list[Path]) -> Path | None:
    """Pick the vocals stem from separator output.

    Instrumental models emit '(Vocals)' and '(Instrumental)'; demucs emits
    per-stem names. Prefer an explicit vocals file, never the instrumental.
    """
    for f in files:
        if "vocal" in f.name.lower():
            return f
    non_instrumental = [f for f in files if "instrument" not in f.name.lower()]
    return non_instrumental[0] if non_instrumental else None
