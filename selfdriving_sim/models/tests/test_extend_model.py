"""Tests for models/extend_model.py on a small stand-in with the checkpoint's layout.

The stand-in has the two features that make the checkpoint hard to run in a
browser: a normalization Lambda and a nested model inside TimeDistributed. Both
must be gone from the extended model, and the network must still compute the same
steering and throttle from the same raw pixels.
"""

import numpy as np
import pytest
import tf_keras as keras

import checkpoint
import extend_model
from dave2_config import BRAKE, PIXEL_MAX, STEERING, THROTTLE, TURN_SIGNAL
from models.tests.helpers import (
    TINY_INPUT_SHAPE,
    build_checkpoint_without_normalization,
    build_tiny_checkpoint,
)

# Layer types tfjs-layers can deserialize. A Lambda or a nested model is not among
# them, which is why the graph is flattened before it is converted.
BROWSER_LAYERS = {"InputLayer", "TimeDistributed", "Conv2D", "Dropout", "Flatten", "GRU", "Dense"}


@pytest.fixture(scope="module")
def tiny_base():
    return build_tiny_checkpoint()


@pytest.fixture(scope="module")
def extension(tiny_base):
    return extend_model.extend_checkpoint(tiny_base, TINY_INPUT_SHAPE)


@pytest.fixture(scope="module")
def tiny_extended(extension):
    return extension[0]


@pytest.fixture(scope="module")
def report(extension):
    return extension[1]


def random_pixels(seed: int, samples: int = 2) -> np.ndarray:
    return extend_model.random_pixels(TINY_INPUT_SHAPE, samples, seed)


def test_extended_model_has_four_heads_in_contract_order(tiny_extended):
    assert tiny_extended.output_names == [STEERING, THROTTLE, BRAKE, TURN_SIGNAL]
    assert [output.shape[-1] for output in tiny_extended.outputs] == [1, 1, 1, 3]


def test_the_heads_reproduce_the_trained_outputs(tiny_base, tiny_extended):
    errors = extend_model.check_wiring(tiny_base, tiny_extended, TINY_INPUT_SHAPE)
    assert max(errors.values()) < extend_model.WIRING_TOLERANCE


def test_steering_and_throttle_keep_the_trained_linear_head(tiny_extended):
    # The trained head is linear and its values are already the protocol's ranges,
    # so the copies stay linear and the phone clamps (see toCommand in inference.js).
    for name in (STEERING, THROTTLE):
        assert tiny_extended.get_layer(name).get_config()["activation"] == "linear"
    assert tiny_extended.get_layer(BRAKE).get_config()["activation"] == "sigmoid"
    assert tiny_extended.get_layer(TURN_SIGNAL).get_config()["activation"] == "softmax"


def test_neutral_heads_do_not_depend_on_the_input(tiny_extended):
    _, _, brake_a, turn_a = tiny_extended.predict(random_pixels(1), verbose=0)
    _, _, brake_b, turn_b = tiny_extended.predict(random_pixels(2), verbose=0)

    np.testing.assert_allclose(brake_a, brake_b, atol=1e-7)
    np.testing.assert_allclose(turn_a, turn_b, atol=1e-7)
    assert np.all(brake_a < 1e-6)  # brake stays released
    assert np.all(turn_a[:, 0] > 0.999)  # class 0 (signal off) wins


def test_compile_uses_the_losses_from_the_brief(tiny_extended):
    assert tiny_extended.loss == {
        STEERING: "mse",
        THROTTLE: "mse",
        BRAKE: "mse",
        TURN_SIGNAL: "categorical_crossentropy",
    }


def test_the_flattened_network_computes_what_the_checkpoint_computes(tiny_base):
    scale = checkpoint.normalization_scale(tiny_base)
    flat = extend_model.restructure(tiny_base, scale)

    drift = extend_model.check_equivalence(tiny_base, flat, TINY_INPUT_SHAPE)

    assert drift < extend_model.EQUIVALENCE_TOLERANCE
    assert flat.output_names == tiny_base.output_names


def test_the_report_measures_the_normalization_and_the_drift(report):
    assert report["normalization_scale"] == pytest.approx(1 / PIXEL_MAX, rel=1e-6)
    assert report["equivalence_drift"] < extend_model.EQUIVALENCE_TOLERANCE
    assert max(report["wiring_errors"].values()) < extend_model.WIRING_TOLERANCE
    assert report["gru"]["units"] == 8


def test_the_browser_graph_has_no_lambda_and_no_nested_model(tiny_extended):
    for layer in tiny_extended.layers:
        assert type(layer).__name__ in BROWSER_LAYERS, f"{layer.name} cannot run in the browser"
        inner = getattr(layer, "layer", None)
        if inner is not None:
            assert not isinstance(inner, keras.Model), f"{layer.name} wraps a nested model"
            assert type(inner).__name__ in BROWSER_LAYERS


def test_the_normalization_is_folded_into_the_first_convolution(tiny_base, tiny_extended):
    scale = checkpoint.normalization_scale(tiny_base)
    original = checkpoint.frame_backbone(tiny_base).get_layer("conv2d_1").get_weights()[0]
    folded = tiny_extended.get_layer("td_conv2d_1").layer.get_weights()[0]

    np.testing.assert_allclose(folded, original * scale, rtol=1e-6)


def test_the_extended_model_keeps_the_checkpoint_pixel_range(tiny_base, tiny_extended):
    assert tuple(int(size) for size in tiny_extended.input_shape[1:]) == TINY_INPUT_SHAPE
    assert tiny_extended.inputs[0].name.startswith("frames")

    batch = random_pixels(5)
    trained = tiny_base.predict(batch, verbose=0)
    steering, throttle, _, _ = tiny_extended.predict(batch, verbose=0)

    np.testing.assert_allclose(steering, trained[:, :1], atol=1e-6)
    np.testing.assert_allclose(throttle, trained[:, 1:2], atol=1e-6)


def test_wrong_input_shape_is_rejected(tiny_base):
    with pytest.raises(ValueError, match="Expected per-sample input"):
        extend_model.validate_checkpoint(tiny_base, expected_input_shape=(3, 120, 160, 3))


def test_squashed_output_layer_is_rejected():
    with pytest.raises(ValueError, match="must be linear"):
        extend_model.validate_checkpoint(build_tiny_checkpoint(output_activation="tanh"), TINY_INPUT_SHAPE)


def test_unsupported_gru_activation_is_rejected():
    with pytest.raises(ValueError, match="does not implement these settings"):
        extend_model.describe_gru(build_tiny_checkpoint(gru_activation="relu"))


def test_a_backbone_without_normalization_is_rejected():
    with pytest.raises(checkpoint.UnsupportedCheckpoint, match="normalization Lambda"):
        extend_model.validate_checkpoint(build_checkpoint_without_normalization(), TINY_INPUT_SHAPE)


def test_gru_settings_are_reported(tiny_base):
    settings = extend_model.describe_gru(tiny_base)

    assert settings["units"] == 8
    assert settings["reset_after"] is True  # Keras 2 default, which the browser layer implements
    assert settings["activation"] == "tanh"
    assert settings["recurrent_activation"] == "sigmoid"
    assert settings["use_bias"] is True


def test_extended_model_survives_an_hdf5_round_trip(tmp_path, tiny_extended):
    path = tmp_path / "dave2_gru_extended.h5"
    tiny_extended.save(path)
    reloaded = keras.models.load_model(path, compile=False)

    batch = random_pixels(3)
    for before, after in zip(tiny_extended.predict(batch, verbose=0), reloaded.predict(batch, verbose=0)):
        np.testing.assert_allclose(before, after, atol=1e-6)
    assert reloaded.output_names == tiny_extended.output_names


def test_main_reports_a_missing_checkpoint(tmp_path, capsys):
    exit_code = extend_model.main(["--source", str(tmp_path / "missing.h5")])

    assert exit_code == 1
    assert "download_model.py" in capsys.readouterr().err


def test_main_refuses_a_checkpoint_that_breaks_the_input_contract(tmp_path, tiny_checkpoint_path, capsys):
    """main() holds the contract from dave2_config, so the tiny stand-in is too small."""
    output = tmp_path / "extended.h5"

    exit_code = extend_model.main(["--source", str(tiny_checkpoint_path), "--output", str(output)])

    assert exit_code == 1
    assert "Expected per-sample input" in capsys.readouterr().err
    assert not output.exists()
