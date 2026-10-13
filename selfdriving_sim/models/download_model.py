"""Download the DAVE-2-GRU checkpoint from Hugging Face.

Run from selfdriving_sim/::

    python models/download_model.py              # genroads checkpoint (default)
    python models/download_model.py --checkpoint jungle

The file is fetched at the revision pinned in ``dave2_config.HF_REVISION`` and
must match the recorded SHA-256 and size. If ``models/data/<folder>/best_model.h5``
already exists, for example because the file was copied in by hand, it is
verified and used without any network request.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from collections.abc import Callable
from pathlib import Path

from huggingface_hub import hf_hub_download
from huggingface_hub.errors import EntryNotFoundError, HfHubHTTPError, LocalEntryNotFoundError

from dave2_config import (
    CHECKPOINT_FILENAME,
    CHECKPOINTS,
    DATA_DIR,
    DEFAULT_CHECKPOINT,
    HF_MODEL_PAGE,
    HF_REPO_ID,
    HF_REVISION,
    Checkpoint,
    local_checkpoint_path,
)

# File names tried in order. The first one is the checkpoint path in the pinned
# revision. The bare root-level names are kept as fallbacks for a hand-made
# copy; the pinned revision does not contain them.
CANDIDATE_TEMPLATES = (
    "{folder}/" + CHECKPOINT_FILENAME,
    "best_model.h5",
    "model.h5",
    "dave2_gru.h5",
)

Downloader = Callable[..., str]


class DownloadError(RuntimeError):
    """A failure with a message that tells the user what to do next."""


class IntegrityError(DownloadError):
    """A file exists but does not match the pinned digest or size."""


def sha256_of(path: Path) -> str:
    """Return the SHA-256 hex digest of a file, read in 1 MiB blocks."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_checkpoint(path: Path, checkpoint: Checkpoint) -> None:
    """Raise IntegrityError unless the file matches the pinned size and SHA-256."""
    size = path.stat().st_size
    if size != checkpoint.size_bytes:
        raise IntegrityError(
            f"{path} has {size} bytes; expected {checkpoint.size_bytes}. "
            "Remove the file and run this script again, or copy the correct checkpoint."
        )
    actual = sha256_of(path)
    if actual != checkpoint.sha256:
        raise IntegrityError(
            f"{path} has SHA-256 {actual}; expected {checkpoint.sha256}. "
            "Remove the file and run this script again, or copy the correct checkpoint."
        )


def candidate_filenames(checkpoint: Checkpoint) -> list[str]:
    """Return the repository paths to try, in order."""
    return [template.format(folder=checkpoint.folder) for template in CANDIDATE_TEMPLATES]


def _manual_hint(target: Path) -> str:
    return (
        f"Place the checkpoint at {target} and run this script again. "
        f"It will verify the file instead of downloading it. Model page: {HF_MODEL_PAGE}"
    )


def ensure_checkpoint(
    checkpoint: Checkpoint,
    data_dir: Path = DATA_DIR,
    download: Downloader = hf_hub_download,
) -> Path:
    """Return a verified local path for the checkpoint, downloading it if needed."""
    target = local_checkpoint_path(checkpoint, data_dir)
    if target.exists():
        verify_checkpoint(target, checkpoint)
        return target

    missing: list[str] = []
    for filename in candidate_filenames(checkpoint):
        try:
            downloaded = Path(
                download(
                    repo_id=HF_REPO_ID,
                    filename=filename,
                    revision=HF_REVISION,
                    local_dir=str(data_dir),
                )
            )
        # LocalEntryNotFoundError is a subclass of EntryNotFoundError, so it must be
        # handled first; otherwise offline mode would look like a missing file.
        except LocalEntryNotFoundError as exc:
            raise DownloadError(
                f"The file is not cached and offline mode is on. {_manual_hint(target)}"
            ) from exc
        except EntryNotFoundError:
            missing.append(filename)
            continue
        except HfHubHTTPError as exc:
            raise DownloadError(
                f"Hugging Face refused {filename} at revision {HF_REVISION[:7]}: {exc}"
            ) from exc
        except OSError as exc:
            # DNS, TLS or connection failures, for example when huggingface.co is
            # not reachable from this machine.
            raise DownloadError(
                f"Could not reach huggingface.co ({type(exc).__name__}: {exc}). {_manual_hint(target)}"
            ) from exc

        try:
            verify_checkpoint(downloaded, checkpoint)
        except IntegrityError:
            downloaded.unlink(missing_ok=True)
            raise
        return downloaded

    raise DownloadError(
        f"None of the candidate files exist at revision {HF_REVISION[:7]}: {', '.join(missing)}."
    )


def main(argv: list[str] | None = None) -> int:
    """Command-line entry point. Returns the process exit code."""
    parser = argparse.ArgumentParser(description="Download and verify a DAVE-2-GRU checkpoint.")
    parser.add_argument(
        "--checkpoint",
        choices=sorted(CHECKPOINTS),
        default=DEFAULT_CHECKPOINT,
        help="Checkpoint to fetch (default: %(default)s).",
    )
    args = parser.parse_args(argv)

    try:
        path = ensure_checkpoint(CHECKPOINTS[args.checkpoint])
    except DownloadError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    print(f"OK: {path} ({path.stat().st_size} bytes, size and SHA-256 verified)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
