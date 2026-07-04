"""Model manifest + downloader unit tests (no network)."""

from lyricsync.models.downloader import sha256_of
from lyricsync.models.manifest import ModelManifest, ModelRecord


def test_manifest_round_trip(tmp_path):
    m = ModelManifest(tmp_path)
    m.upsert(ModelRecord(model_id="x.onnx", display_name="X", tier=4,
                         architecture="mdx", filenames=["x.onnx"],
                         source="url", size_bytes=123))
    reloaded = ModelManifest(tmp_path)
    rec = reloaded.records["x.onnx"]
    assert rec.display_name == "X"
    assert rec.tier == 4
    assert rec.filenames == ["x.onnx"]


def test_disk_usage_and_untracked(tmp_path):
    (tmp_path / "x.onnx").write_bytes(b"a" * 100)
    (tmp_path / "stray.ckpt").write_bytes(b"b" * 50)
    (tmp_path / "part.onnx.partial").write_bytes(b"c")
    m = ModelManifest(tmp_path)
    m.upsert(ModelRecord(model_id="x.onnx", display_name="X", tier=4,
                         filenames=["x.onnx"]))
    assert m.disk_usage()["x.onnx"] == 100
    untracked = [f.name for f in m.untracked_files()]
    assert untracked == ["stray.ckpt"]  # manifest.json + .partial excluded


def test_remove_with_files(tmp_path):
    (tmp_path / "x.onnx").write_bytes(b"a")
    m = ModelManifest(tmp_path)
    m.upsert(ModelRecord(model_id="x.onnx", display_name="X", tier=4,
                         filenames=["x.onnx"]))
    m.remove("x.onnx", delete_files=True)
    assert not (tmp_path / "x.onnx").exists()
    assert "x.onnx" not in ModelManifest(tmp_path).records


def test_sha256(tmp_path):
    f = tmp_path / "f.bin"
    f.write_bytes(b"hello")
    assert sha256_of(f) == (
        "2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824")
