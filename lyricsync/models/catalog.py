"""Bridge to audio-separator's hosted model catalog + custom model install.

For catalog models we rely on audio-separator's own auto-download-on-first-
use (it also fetches sidecar YAML configs that MDXC/Roformer checkpoints
need). Custom models are downloaded/copied into the same model_file_dir so
they're picked up like any locally cached model.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

from lyricsync.core.separation import TIERS
from lyricsync.models.downloader import (
    DownloadProgressFn,
    copy_local_model,
    download_file,
    hf_url,
    sha256_of,
)
from lyricsync.models.manifest import ModelManifest, ModelRecord
from lyricsync.utils.logs import get_logger

log = get_logger("catalog")

_ARCH_BY_EXT = {".onnx": "mdx", ".ckpt": "mdxc", ".yaml": "demucs", ".pth": "vr_arch"}


def guess_architecture(filename: str) -> str:
    return _ARCH_BY_EXT.get(Path(filename).suffix.lower(), "unknown")


def list_catalog_models(model_dir: str | Path) -> dict[str, dict]:
    """audio-separator's supported-model listing (filename → info).

    Imported lazily and wrapped: the listing may hit the network to refresh
    its JSON index, so callers should treat this as slow / fallible.
    """
    from audio_separator.separator import Separator

    sep = Separator(model_file_dir=str(model_dir), info_only=True)
    return sep.get_simplified_model_list()


def ensure_builtin_tier_records(manifest: ModelManifest) -> None:
    """Seed the manifest with the three built-in tiers (idempotent)."""
    for tier in TIERS.values():
        if tier.model_filename in manifest.records:
            continue
        manifest.upsert(ModelRecord(
            model_id=tier.model_filename,
            display_name=tier.label,
            tier=tier.tier,
            architecture=guess_architecture(tier.model_filename),
            filenames=[tier.model_filename],
            source="catalog",
        ))


def install_custom_model(
    manifest: ModelManifest,
    *,
    display_name: str = "",
    url: str = "",
    hf_repo: str = "",
    hf_filename: str = "",
    local_path: str = "",
    config_url: str = "",
    progress: DownloadProgressFn | None = None,
    cancel_check: Callable[[], bool] | None = None,
) -> ModelRecord:
    """Install a custom model from a URL, a HF repo+filename, or a local path.

    MDXC/Roformer checkpoints frequently need a sidecar YAML config —
    pass `config_url` to fetch it alongside the weights with the same name
    scheme audio-separator expects (same stem, .yaml extension).
    """
    model_dir = manifest.model_dir
    filenames: list[str] = []
    source = "url"
    source_url = url

    if local_path:
        primary = copy_local_model(local_path, model_dir)
        source, source_url = "local", str(local_path)
    elif hf_repo and hf_filename:
        source, source_url = "huggingface", f"{hf_repo}/{hf_filename}"
        primary = download_file(hf_url(hf_repo, hf_filename),
                                model_dir / Path(hf_filename).name,
                                progress=progress, cancel_check=cancel_check)
    elif url:
        primary = download_file(url, model_dir / _filename_from_url(url),
                                progress=progress, cancel_check=cancel_check)
    else:
        raise ValueError("Provide a URL, a Hugging Face repo+filename, or a local path.")

    filenames.append(primary.name)

    if config_url:
        config_dest = model_dir / _filename_from_url(config_url)
        download_file(config_url, config_dest, cancel_check=cancel_check)
        filenames.append(config_dest.name)
    elif primary.suffix.lower() == ".ckpt":
        # Auto-detect a sidecar YAML next to the source when installing
        # Roformer/MDXC checkpoints from a local path.
        if local_path:
            sidecar = Path(local_path).with_suffix(".yaml")
            if sidecar.exists():
                copied = copy_local_model(sidecar, model_dir)
                filenames.append(copied.name)
        else:
            log.warning(
                "%s is an MDXC/Roformer checkpoint; it may need a sidecar YAML "
                "config. Provide its URL if loading fails.", primary.name,
            )

    rec = ModelRecord(
        model_id=primary.name,
        display_name=display_name or primary.stem,
        tier=4,
        architecture=guess_architecture(primary.name),
        filenames=filenames,
        source=source,
        source_url=source_url,
        sha256=sha256_of(primary),
        size_bytes=primary.stat().st_size,
    )
    manifest.upsert(rec)
    log.info("installed custom model %s (%s)", rec.model_id, rec.architecture)
    return rec


def _filename_from_url(url: str) -> str:
    name = url.split("?")[0].rstrip("/").split("/")[-1]
    if not name:
        raise ValueError(f"Cannot derive a filename from URL: {url}")
    return name
