"""Tests for models/checkpoint.py: reading a checkpoint whichever Python wrote it."""

import numpy as np
import pytest
import tf_keras as keras

import checkpoint
from dave2_config import CHECKPOINT_NORMALIZATION, PIXEL_MAX
from models.tests.helpers import (
    TINY_INPUT_SHAPE,
    build_checkpoint_without_normalization,
    build_tiny_checkpoint,
    erase_stored_lambda_code,
)


def random_pixels(seed: int = 0, samples: int = 2) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return (rng.random((samples, *TINY_INPUT_SHAPE)) * PIXEL_MAX).astype(np.float32)


def test_a_readable_checkpoint_is_loaded_directly(tiny_checkpoint_path):
    loaded = checkpoint.load_base_model(tiny_checkpoint_path)

    assert loaded.rebuilt is False
    assert "directly" in loaded.note
    assert tuple(int(size) for size in loaded.model.input_shape[1:]) == TINY_INPUT_SHAPE
    assert loaded.model.count_params() == build_tiny_checkpoint().count_params()


def test_a_checkpoint_from_another_python_version_is_rebuilt_and_computes_the_same(
    tiny_checkpoint_path, unreadable_lambda_checkpoint_path
):
    # The failure this path exists for: Keras cannot unmarshal the stored lambda.
    with pytest.raises(ValueError, match="marshal"):
        keras.models.load_model(unreadable_lambda_checkpoint_path, compile=False)

    loaded = checkpoint.load_base_model(unreadable_lambda_checkpoint_path)
    assert loaded.rebuilt is True
    assert "marshal" in loaded.note

    readable = checkpoint.load_base_model(tiny_checkpoint_path).model
    batch = random_pixels(3)
    np.testing.assert_allclose(
        loaded.model.predict(batch, verbose=0),
        readable.predict(batch, verbose=0),
        atol=1e-6,
    )


def test_the_normalization_is_measured_instead_of_assumed(tiny_checkpoint_path):
    model = checkpoint.load_base_model(tiny_checkpoint_path).model
    assert checkpoint.normalization_scale(model) == pytest.approx(CHECKPOINT_NORMALIZATION, rel=1e-6)


def test_a_different_normalization_is_measured():
    model = build_tiny_checkpoint(normalization=1 / 127.5)
    assert checkpoint.normalization_scale(model) == pytest.approx(1 / 127.5, rel=1e-6)


def test_a_normalization_that_is_not_a_plain_scaling_is_rejected():
    with pytest.raises(checkpoint.UnsupportedCheckpoint, match="not a plain scaling"):
        checkpoint.normalization_scale(build_tiny_checkpoint(offset=-0.5))


def test_a_backbone_without_normalization_is_rejected():
    with pytest.raises(checkpoint.UnsupportedCheckpoint, match="normalization Lambda"):
        checkpoint.normalization_scale(build_checkpoint_without_normalization())


def test_the_frame_backbone_and_its_normalization_are_found(tiny_checkpoint_path):
    model = checkpoint.load_base_model(tiny_checkpoint_path).model

    assert checkpoint.frame_backbone(model).name == "dave2_backbone"
    assert checkpoint.normalization_layer(model).name == "norm"


def test_an_unreadable_normalization_is_not_guessed(tiny_checkpoint_path, tmp_path):
    erased = erase_stored_lambda_code(tiny_checkpoint_path, tmp_path / "erased.h5")

    with pytest.raises(checkpoint.UnsupportedCheckpoint, match="normalization-scale"):
        checkpoint.load_base_model(erased)


def test_a_missing_file_is_reported(tmp_path):
    with pytest.raises(checkpoint.UnsupportedCheckpoint, match="does not exist"):
        checkpoint.load_base_model(tmp_path / "nope.h5")


def test_a_weights_only_file_has_no_topology(tmp_path):
    path = tmp_path / "weights.h5"
    build_tiny_checkpoint().save_weights(path)

    with pytest.raises(checkpoint.UnsupportedCheckpoint, match="model_config"):
        checkpoint.read_model_config(path)


def test_describe_reports_the_facts_the_scripts_print(tiny_checkpoint_path):
    facts = checkpoint.describe(checkpoint.load_base_model(tiny_checkpoint_path).model)

    assert facts["name"] == "tiny_dave2_gru_T3"
    assert facts["input_shape"] == TINY_INPUT_SHAPE
    assert facts["output_shape"] == (2,)
    assert facts["parameters"] > 0
    assert "td_backbone (TimeDistributed)" in facts["layers"]
    assert "gru (GRU)" in facts["layers"]
