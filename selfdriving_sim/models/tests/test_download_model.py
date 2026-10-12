"""Tests for models/download_model.py. The network is replaced by fake downloaders."""

import hashlib
from pathlib import Path

import pytest
from huggingface_hub.errors import EntryNotFoundError, LocalEntryNotFoundError

import download_model
from dave2_config import CHECKPOINT_FILENAME, CHECKPOINTS, HF_REPO_ID, HF_REVISION, Checkpoint, local_checkpoint_path

PAYLOAD = b"stand-in checkpoint bytes"


def checkpoint_for(payload: bytes) -> Checkpoint:
    return Checkpoint(
        folder="demo_run",
        sha256=hashlib.sha256(payload).hexdigest(),
        size_bytes=len(payload),
        train_sequences=1,
        val_sequences=1,
    )


def fake_hub(files: dict[str, bytes], attempts: list[str]):
    """Return a downloader that serves ``files`` by repository path and records each request."""

    def download(*, repo_id, filename, revision, local_dir):
        assert repo_id == HF_REPO_ID
        assert revision == HF_REVISION
        attempts.append(filename)
        if filename not in files:
            raise EntryNotFoundError(f"{filename} is not in the repository")
        target = Path(local_dir) / filename
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(files[filename])
        return str(target)

    return download


def test_genroads_pins_match_the_hugging_face_metadata():
    genroads = CHECKPOINTS["genroads"]
    assert genroads.folder == "genroads_20251215-174930"
    assert genroads.size_bytes == 33_328_880
    assert genroads.sha256 == "fbee0740ce53b0ad7be35b708c0242fc98143699f8f7024af0d21d28719a32bf"


def test_existing_verified_file_is_used_without_any_request(tmp_path):
    checkpoint = checkpoint_for(PAYLOAD)
    target = local_checkpoint_path(checkpoint, tmp_path)
    target.parent.mkdir(parents=True)
    target.write_bytes(PAYLOAD)

    def no_network(**_):
        raise AssertionError("a verified local file must not trigger a download")

    assert download_model.ensure_checkpoint(checkpoint, tmp_path, download=no_network) == target


def test_wrong_local_file_is_rejected_and_kept(tmp_path):
    checkpoint = checkpoint_for(PAYLOAD)
    target = local_checkpoint_path(checkpoint, tmp_path)
    target.parent.mkdir(parents=True)
    target.write_bytes(b"x" * len(PAYLOAD))  # same size, different digest

    with pytest.raises(download_model.IntegrityError, match="SHA-256"):
        download_model.ensure_checkpoint(checkpoint, tmp_path, download=fake_hub({}, []))
    assert target.exists(), "a file the user placed must not be deleted"


def test_wrong_size_is_reported_before_hashing(tmp_path):
    checkpoint = checkpoint_for(PAYLOAD)
    target = local_checkpoint_path(checkpoint, tmp_path)
    target.parent.mkdir(parents=True)
    target.write_bytes(PAYLOAD + b"!")

    with pytest.raises(download_model.IntegrityError, match="bytes"):
        download_model.ensure_checkpoint(checkpoint, tmp_path, download=fake_hub({}, []))


def test_candidates_are_tried_in_order_until_one_exists(tmp_path):
    checkpoint = checkpoint_for(PAYLOAD)
    attempts: list[str] = []
    files = {"best_model.h5": PAYLOAD}  # only the root-level fallback exists

    path = download_model.ensure_checkpoint(
        checkpoint, tmp_path, download=fake_hub(files, attempts)
    )

    assert attempts == ["demo_run/best_model.h5", "best_model.h5"]
    assert path == tmp_path / "best_model.h5"


def test_missing_everywhere_gives_a_clear_error(tmp_path):
    checkpoint = checkpoint_for(PAYLOAD)
    with pytest.raises(download_model.DownloadError, match="demo_run/best_model.h5"):
        download_model.ensure_checkpoint(checkpoint, tmp_path, download=fake_hub({}, []))


def test_offline_cache_miss_points_to_manual_placement(tmp_path):
    checkpoint = checkpoint_for(PAYLOAD)

    def offline(**_):
        raise LocalEntryNotFoundError("offline mode is on")

    with pytest.raises(download_model.DownloadError, match="Place the checkpoint at"):
        download_model.ensure_checkpoint(checkpoint, tmp_path, download=offline)


def test_network_failure_points_to_manual_placement(tmp_path):
    checkpoint = checkpoint_for(PAYLOAD)

    def unreachable(**_):
        raise OSError("name resolution failed")

    with pytest.raises(download_model.DownloadError, match="Could not reach huggingface.co"):
        download_model.ensure_checkpoint(checkpoint, tmp_path, download=unreachable)


def test_downloaded_file_that_fails_verification_is_removed(tmp_path):
    checkpoint = checkpoint_for(PAYLOAD)
    files = {f"demo_run/{CHECKPOINT_FILENAME}": b"y" * len(PAYLOAD)}

    with pytest.raises(download_model.IntegrityError):
        download_model.ensure_checkpoint(checkpoint, tmp_path, download=fake_hub(files, []))
    assert not local_checkpoint_path(checkpoint, tmp_path).exists()


def test_main_returns_nonzero_and_prints_the_error(monkeypatch, capsys):
    def fail(checkpoint, *args, **kwargs):
        raise download_model.DownloadError("boom")

    monkeypatch.setattr(download_model, "ensure_checkpoint", fail)
    assert download_model.main([]) == 1
    assert "ERROR: boom" in capsys.readouterr().err
