"""Tests for models/architecture.py: rebuilding a topology from the file's own JSON."""

import pytest

import architecture
import checkpoint
from dave2_config import CHECKPOINT_NORMALIZATION, PIXEL_MAX
from models.tests.helpers import TINY_INPUT_SHAPE, build_tiny_checkpoint, normalization_spec


@pytest.fixture
def tiny_config(tiny_checkpoint_path):
    """A fresh copy per test: some tests change the config to provoke a rejection."""
    return checkpoint.read_model_config(tiny_checkpoint_path)


def test_the_normalization_constant_survives_in_the_stored_code(tiny_config):
    spec = normalization_spec(tiny_config)

    assert architecture.lambda_mentions_constant(spec, PIXEL_MAX)
    assert not architecture.lambda_mentions_constant(spec, 127.5)


def test_a_layer_type_outside_the_published_architecture_is_rejected(tiny_config):
    wrapper = next(
        layer for layer in architecture.layer_specs(tiny_config) if layer["class_name"] == "TimeDistributed"
    )
    nested = architecture.nested_model_spec(wrapper)
    nested["config"]["layers"].append({"class_name": "BatchNormalization", "config": {"name": "bn"}})

    with pytest.raises(architecture.UnsupportedTopology, match="BatchNormalization"):
        architecture.build_model(tiny_config, CHECKPOINT_NORMALIZATION)


def test_a_wrapper_around_a_plain_layer_is_rejected(tiny_config):
    wrapper = next(
        layer for layer in architecture.layer_specs(tiny_config) if layer["class_name"] == "TimeDistributed"
    )
    wrapper["config"]["layer"] = {"class_name": "Conv2D", "config": {"name": "conv2d_1"}}

    with pytest.raises(architecture.UnsupportedTopology, match="does not wrap a model"):
        architecture.build_model(tiny_config, CHECKPOINT_NORMALIZATION)


def test_more_than_one_output_is_rejected(tiny_config):
    tiny_config["config"]["output_layers"].append(["gru", 0, 0])

    with pytest.raises(architecture.UnsupportedTopology, match="exactly one output"):
        architecture.build_model(tiny_config, CHECKPOINT_NORMALIZATION)


def test_the_rebuild_keeps_the_names_and_sizes_load_weights_matches_on(tiny_config):
    rebuilt = architecture.build_model(tiny_config, CHECKPOINT_NORMALIZATION)
    original = build_tiny_checkpoint()

    assert rebuilt.name == original.name
    assert [layer.name for layer in rebuilt.layers] == [layer.name for layer in original.layers]
    assert tuple(int(size) for size in rebuilt.input_shape[1:]) == TINY_INPUT_SHAPE
    assert rebuilt.count_params() == original.count_params()
    # The nested backbone keeps its own layer names, including the normalization.
    assert [layer.name for layer in checkpoint.frame_backbone(rebuilt).layers] == [
        layer.name for layer in checkpoint.frame_backbone(original).layers
    ]
