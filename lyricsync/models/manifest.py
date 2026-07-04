"""File-based model registry — a JSON manifest, no database.

Tracks every separation model LyricSync knows about: the three built-in
tiers (delegated to audio-separator's own catalog + auto-download) and any
custom models the user has installed from a URL / Hugging Face / local
file. Lives next to the model weights in the persistent user data dir.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

from lyricsync.utils.logs import get_logger

log = get_logger("manifest")

ARCHITECTURES = ("mdx", "vr_arch", "demucs", "mdxc", "unknown")


@dataclass
class ModelRecord:
    model_id: str                 # stable key, usually the primary filename
    display_name: str
    tier: int                     # 1..3 built-in, 4 = custom
    architecture: str = "unknown" # mdx | vr_arch | demucs | mdxc
    filenames: list[str] = field(default_factory=list)  # weights + sidecar YAML
    source: str = "catalog"       # catalog | url | huggingface | local
    source_url: str = ""
    sha256: str = ""              # of the primary file, where known
    size_bytes: int = 0
    last_used: float = 0.0        # unix timestamp, 0 = never

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "ModelRecord":
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in d.items() if k in known})


class ModelManifest:
    """JSON-backed collection of ModelRecords stored inside the model dir."""

    def __init__(self, model_dir: str | Path):
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.path = self.model_dir / "manifest.json"
        self.records: dict[str, ModelRecord] = {}
        self._load()

    def _load(self) -> None:
        if not self.path.exists():
            return
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
            for d in data.get("models", []):
                rec = ModelRecord.from_dict(d)
                self.records[rec.model_id] = rec
        except (OSError, json.JSONDecodeError, TypeError) as e:
            log.warning("could not read manifest (%s); starting fresh", e)

    def save(self) -> None:
        payload = {"version": 1, "models": [r.to_dict() for r in self.records.values()]}
        tmp = self.path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        tmp.replace(self.path)

    def upsert(self, rec: ModelRecord) -> None:
        self.records[rec.model_id] = rec
        self.save()

    def remove(self, model_id: str, delete_files: bool = False) -> None:
        rec = self.records.pop(model_id, None)
        if rec and delete_files:
            for name in rec.filenames:
                f = self.model_dir / name
                if f.exists():
                    f.unlink()
                    log.info("deleted model file %s", f)
        self.save()

    def touch(self, model_id: str) -> None:
        rec = self.records.get(model_id)
        if rec:
            rec.last_used = time.time()
            self.save()

    def disk_usage(self) -> dict[str, int]:
        """model_id → bytes actually on disk right now."""
        usage: dict[str, int] = {}
        for rec in self.records.values():
            usage[rec.model_id] = sum(
                (self.model_dir / n).stat().st_size
                for n in rec.filenames
                if (self.model_dir / n).exists()
            )
        return usage

    def untracked_files(self) -> list[Path]:
        """Files in the model dir no record claims (e.g. auto-downloaded
        catalog weights) — shown in the storage screen so nothing is
        invisible, but never auto-deleted."""
        claimed = {n for r in self.records.values() for n in r.filenames}
        out = []
        for f in sorted(self.model_dir.iterdir()):
            if f.is_file() and f.name not in claimed and f.name != "manifest.json" \
                    and not f.name.endswith(".partial"):
                out.append(f)
        return out
