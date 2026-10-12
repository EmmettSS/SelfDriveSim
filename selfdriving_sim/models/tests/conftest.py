"""Shared fixtures for the model scripts.

The scripts import each other by plain module name (``from dave2_config import``),
so the models/ folder is put on sys.path here, the same way it is when a script is
run from the command line.
"""

import os
import sys
from pathlib import Path

import pytest

MODELS_DIR = Path(__file__).resolve().parents[1]
if str(MODELS_DIR) not in sys.path:
    sys.path.insert(0, str(MODELS_DIR))

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

from models.tests.helpers import break_stored_lambda, save_tiny_checkpoint  # noqa: E402


@pytest.fixture(scope="session")
def tiny_checkpoint_path(tmp_path_factory) -> Path:
    """An untrained checkpoint saved as HDF5, the format of the real file."""
    return save_tiny_checkpoint(tmp_path_factory.mktemp("checkpoint") / "best_model.h5")


@pytest.fixture(scope="session")
def unreadable_lambda_checkpoint_path(tmp_path_factory, tiny_checkpoint_path) -> Path:
    """The same checkpoint as one saved by another Python version: its lambda cannot be read."""
    return break_stored_lambda(
        tiny_checkpoint_path, tmp_path_factory.mktemp("unreadable") / "best_model.h5"
    )
