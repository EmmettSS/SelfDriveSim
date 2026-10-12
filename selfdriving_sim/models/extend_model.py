"""Turn the DAVE-2-GRU checkpoint into the four-head model the phone runs (phase 4).

The checkpoint predicts ``[steering, throttle]`` from three consecutive frames of
raw 0..255 pixels. This script produces ``dave2_gru_extended.h5`` with four heads
on the last hidden Dense layer::

    steering     Dense(1, linear)   trained steering weights copied in
    throttle     Dense(1, linear)   trained throttle weights copied in
    brake        Dense(1, sigmoid)  constant neutral output (no training signal)
    turn_signal  Dense(3, softmax)  constant neutral output, class 0 = off

Two deliberate changes make the network run in the browser, without changing a
single computed value:

* The graph is flattened. ``tfjs-layers`` has no ``Lambda`` layer and cannot
  deserialize a nested model inside ``TimeDistributed``, and the checkpoint uses
  both: ``TimeDistributed(backbone)`` with ``Lambda(t / 255)`` first in the
  backbone. Here every backbone layer gets its own ``TimeDistributed`` wrapper,
  and the scaling is folded into the first convolution's kernel, because
  ``W·(s·x) + b == (W·s)·x + b``. The extended model therefore still takes the
  raw 0..255 pixels the checkpoint was trained on, and the phone does not scale.
* ``steering`` and ``throttle`` keep the trained head's linear activation. The
  phase-4 brief suggested ``tanh`` and ``sigmoid``, but the trained values are
  already the protocol's ranges (steering about [-1, 1], throttle about [0, 1]),
  so squashing them again would shrink full lock to ``tanh(1) == 0.76`` and turn a
  trained ``throttle == 0`` into ``sigmoid(0) == 0.5``. The phone clamps instead
  (see ``toCommand`` in ``client/static/client/js/inference.js``).

Both steps are checked numerically before the model is saved: the restructured
network must reproduce the checkpoint's outputs, and the heads must reproduce the
trained steering and throttle while brake and turn signal stay neutral.

Run from selfdriving_sim/::

    python models/extend_model.py                 # genroads checkpoint (default)
    python models/extend_model.py --checkpoint jungle
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

# The checkpoint is a Keras 2 HDF5 file (its GRU config includes 'time_major',
# which Keras 3 rejects). The TensorFlow.js converter also loads HDF5 files with
# tf_keras, so the extension uses tf_keras too.
import tf_keras as keras

import checkpoint
from checkpoint import UnsupportedCheckpoint, frame_backbone, normalization_layer, normalization_scale
from dave2_config import (
    BRAKE,
    CHECKPOINTS,
    CHECKPOINT_NORMALIZATION,
    DATA_DIR,
    DEFAULT_CHECKPOINT,
    EXTENDED_MODEL_PATH,
    INPUT_SHAPE,
    LEARNING_RATE,
    NEUTRAL_BRAKE_LOGIT,
    PIXEL_MAX,
    STEERING,
    THROTTLE,
    TURN_SIGNAL,
    TURN_SIGNAL_CLASSES,
    TURN_SIGNAL_OFF_LOGIT,
    local_checkpoint_path,
)

# Absolute tolerances for the two numerical checks. Restructuring only reorders
# float32 operations, so the drift stays around 1e-8.
EQUIVALENCE_TOLERANCE = 1e-5
WIRING_TOLERANCE = 1e-5

# Name of the extended model's input, and of the trunk it feeds.
FRAMES_INPUT_NAME = "frames"
TRUNK_NAME = "dave2_gru_trunk"
EXTENDED_NAME = "dave2_gru_extended"

# GRU settings that client/static/client/js/reset-after-gru.js implements. The
# reset_after flag is not listed because the browser layer supports both values.
BROWSER_GRU_SETTINGS = {
    "activation": "tanh",
    "recurrent_activation": "sigmoid",
    "use_bias": True,
    "stateful": False,
    "go_backwards": False,
    "return_state": False,
}


def random_pixels(
    input_shape: tuple[int, ...] = INPUT_SHAPE, samples: int = 4, seed: int = 0
) -> np.ndarray:
    """Return a reproducible batch in the checkpoint's own input range, 0..255."""
    rng = np.random.default_rng(seed)
    return (rng.random((samples, *input_shape)) * PIXEL_MAX).astype(np.float32)


def describe_gru(model: keras.Model) -> dict:
    """Return the settings of the checkpoint's GRU layer and check the browser can run them.

    Raises ValueError when the layer is missing, repeated, or uses a setting that
    the browser layer does not implement.
    """
    grus = [layer for layer in model.layers if isinstance(layer, keras.layers.GRU)]
    if len(grus) != 1:
        raise ValueError(f"Expected exactly one GRU layer, found {len(grus)}.")
    config = grus[0].get_config()
    mismatched = {
        key: config.get(key) for key, expected in BROWSER_GRU_SETTINGS.items() if config.get(key) != expected
    }
    if mismatched:
        raise ValueError(f"The browser GRU layer does not implement these settings: {mismatched}")
    described = ("units", "reset_after", "return_sequences", *BROWSER_GRU_SETTINGS)
    return {key: config.get(key) for key in described}


def validate_checkpoint(model: keras.Model, expected_input_shape: tuple[int, ...] = INPUT_SHAPE) -> None:
    """Reject a file whose layout the restructuring or the head transfer would mis-wire."""
    if len(model.inputs) != 1 or len(model.outputs) != 1:
        raise ValueError(
            f"Expected one input and one output, found {len(model.inputs)} input(s) "
            f"and {len(model.outputs)} output(s)."
        )
    input_shape = tuple(int(size) for size in model.inputs[0].shape[1:])
    if input_shape != tuple(expected_input_shape):
        raise ValueError(f"Expected per-sample input {tuple(expected_input_shape)}, found {input_shape}.")
    output_shape = tuple(int(size) for size in model.outputs[0].shape[1:])
    if output_shape != (2,):
        raise ValueError(f"Expected a 2-value output [steering, throttle], found {output_shape}.")

    output_layer = model.layers[-1]
    if not isinstance(output_layer, keras.layers.Dense) or output_layer.units != 2:
        raise ValueError("The final layer must be the Dense(2) output layer of the checkpoint.")
    # The copied weights only reproduce the trained outputs if the head is not squashed.
    activation = output_layer.get_config()["activation"]
    if activation != "linear":
        raise ValueError(f"The output layer must be linear, found activation {activation!r}.")

    describe_gru(model)

    # The backbone must be a chain that scales the pixels and then convolves them,
    # which is what the folding step needs.
    backbone = frame_backbone(model)
    norm = normalization_layer(model)
    after_input = [layer for layer in backbone.layers if not isinstance(layer, keras.layers.InputLayer)]
    if after_input[0] is not norm:
        raise ValueError(
            f"The normalization layer must be the first layer of the frame backbone, "
            f"found {after_input[0].name!r} before {norm.name!r}."
        )
    if not isinstance(after_input[1], keras.layers.Conv2D):
        raise ValueError(
            f"The layer after the normalization must be a Conv2D to fold the scaling into, "
            f"found {type(after_input[1]).__name__}."
        )


def build_trunk(base: keras.Model, scale: float) -> keras.Model:
    """Rebuild ``frames -> features`` as a flat graph TensorFlow.js can load.

    Every backbone layer is wrapped in its own ``TimeDistributed``, the
    normalization Lambda is dropped, and its scaling is folded into the first
    convolution's kernel. The GRU and the hidden Dense layers are cloned with
    their weights. The checkpoint's ``Dense(2)`` output is not part of the trunk.
    """
    backbone = frame_backbone(base)
    norm = normalization_layer(base)

    frames = keras.layers.Input(shape=tuple(int(size) for size in base.input_shape[1:]), name=FRAMES_INPUT_NAME)
    tensor = frames
    clones: list[tuple[keras.layers.Layer, list[np.ndarray]]] = []
    folded = False

    for layer in backbone.layers:
        if isinstance(layer, keras.layers.InputLayer) or layer is norm:
            continue
        clone = layer.__class__.from_config(layer.get_config())
        weights = layer.get_weights()
        if not folded and isinstance(clone, keras.layers.Conv2D):
            weights = [np.asarray(weights[0], dtype=np.float32) * np.float32(scale), *weights[1:]]
            folded = True
        clones.append((clone, weights))
        tensor = keras.layers.TimeDistributed(clone, name=f"td_{layer.name}")(tensor)
    if not folded:
        raise ValueError("The frame backbone has no Conv2D to fold the normalization into.")

    # Everything between the backbone and the output head: the GRU and the hidden
    # Dense layers. The trained Dense(2) head is left out, because it is replaced.
    for layer in base.layers:
        if isinstance(layer, (keras.layers.InputLayer, keras.layers.TimeDistributed)):
            continue
        if layer is base.layers[-1]:
            continue
        clone = layer.__class__.from_config(layer.get_config())
        clones.append((clone, layer.get_weights()))
        tensor = clone(tensor)

    trunk = keras.Model(inputs=frames, outputs=tensor, name=TRUNK_NAME)
    for clone, weights in clones:
        clone.set_weights(weights)
    return trunk


def restructure(base: keras.Model, scale: float) -> keras.Model:
    """Return the flattened network with the checkpoint's own Dense(2) head.

    It exists for the equivalence check: it must compute exactly what the
    checkpoint computes, which proves the restructuring before any head is added.
    """
    trunk = build_trunk(base, scale)
    output_layer = base.layers[-1]
    head = output_layer.__class__.from_config(output_layer.get_config())
    outputs = head(trunk.output)
    model = keras.Model(inputs=trunk.input, outputs=outputs, name=f"{base.name}_flat")
    head.set_weights(output_layer.get_weights())
    return model


def build_extended_model(base: keras.Model, scale: float) -> keras.Model:
    """Return the flattened network with the four control heads attached."""
    trunk = build_trunk(base, scale)
    # The trunk ends where the checkpoint's layers[-2] ends: the last hidden Dense.
    feature = trunk.output
    trained_kernel, trained_bias = base.layers[-1].get_weights()  # (width, 2) and (2,)
    if feature.shape[-1] != trained_kernel.shape[0]:
        raise ValueError(
            f"Feature width {feature.shape[-1]} does not match the output layer's input "
            f"width {trained_kernel.shape[0]}."
        )

    # Linear, like the trained head: the network's own regression values are kept.
    steering = keras.layers.Dense(1, name=STEERING)(feature)
    throttle = keras.layers.Dense(1, name=THROTTLE)(feature)
    brake = keras.layers.Dense(1, activation="sigmoid", name=BRAKE)(feature)
    turn_signal = keras.layers.Dense(TURN_SIGNAL_CLASSES, activation="softmax", name=TURN_SIGNAL)(feature)

    extended = keras.Model(
        inputs=trunk.input,
        outputs=[steering, throttle, brake, turn_signal],
        name=EXTENDED_NAME,
    )
    _copy_head_weights(extended, trained_kernel, trained_bias)
    return extended


def _copy_head_weights(extended: keras.Model, kernel: np.ndarray, bias: np.ndarray) -> None:
    """Copy trained weights into steering and throttle; set brake and turn signal to neutral."""
    width = kernel.shape[0]
    float32 = np.float32

    extended.get_layer(STEERING).set_weights(
        [kernel[:, 0:1].astype(float32), bias[0:1].astype(float32)]
    )
    extended.get_layer(THROTTLE).set_weights(
        [kernel[:, 1:2].astype(float32), bias[1:2].astype(float32)]
    )
    # Zero weights make the output independent of the features: the same value every frame.
    extended.get_layer(BRAKE).set_weights(
        [np.zeros((width, 1), float32), np.full((1,), NEUTRAL_BRAKE_LOGIT, float32)]
    )
    turn_bias = np.zeros((TURN_SIGNAL_CLASSES,), float32)
    turn_bias[0] = TURN_SIGNAL_OFF_LOGIT
    extended.get_layer(TURN_SIGNAL).set_weights(
        [np.zeros((width, TURN_SIGNAL_CLASSES), float32), turn_bias]
    )


def check_equivalence(
    base: keras.Model,
    restructured: keras.Model,
    input_shape: tuple[int, ...] = INPUT_SHAPE,
    samples: int = 4,
    seed: int = 0,
) -> float:
    """Return the largest output difference between the checkpoint and its flat copy."""
    batch = random_pixels(input_shape, samples, seed)
    expected = base.predict(batch, verbose=0)
    actual = restructured.predict(batch, verbose=0)
    drift = float(np.max(np.abs(expected - actual)))
    if drift > EQUIVALENCE_TOLERANCE:
        raise RuntimeError(
            f"The flattened network differs from the checkpoint by {drift} "
            f"(tolerance {EQUIVALENCE_TOLERANCE}). Refusing to save it."
        )
    return drift


def check_wiring(
    base: keras.Model,
    extended: keras.Model,
    input_shape: tuple[int, ...] = INPUT_SHAPE,
    samples: int = 4,
    seed: int = 0,
) -> dict[str, float]:
    """Check the four heads against the checkpoint. Raises when a head is mis-wired."""
    batch = random_pixels(input_shape, samples, seed)
    trained = base.predict(batch, verbose=0)
    steering, throttle, brake, turn_signal = extended.predict(batch, verbose=0)

    errors = {
        STEERING: float(np.max(np.abs(steering - trained[:, :1]))),
        THROTTLE: float(np.max(np.abs(throttle - trained[:, 1:2]))),
        BRAKE: float(np.max(np.abs(brake))),
        TURN_SIGNAL: float(np.max(np.abs(turn_signal[:, 0] - 1.0))),
    }
    failed = {name: error for name, error in errors.items() if error > WIRING_TOLERANCE}
    if failed:
        raise RuntimeError(f"The heads do not reproduce the checkpoint: {failed}")
    return errors


def compile_model(model: keras.Model) -> None:
    """Compile with the losses from the phase-4 brief. Training is not run in this phase."""
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss={
            STEERING: "mse",
            THROTTLE: "mse",
            BRAKE: "mse",
            TURN_SIGNAL: "categorical_crossentropy",
        },
    )


def extend_checkpoint(
    base: keras.Model, input_shape: tuple[int, ...] = INPUT_SHAPE
) -> tuple[keras.Model, dict]:
    """Return the compiled four-head model for a checkpoint, plus what was measured.

    Validates the checkpoint, flattens it, checks that the flat network still
    computes the checkpoint's outputs, attaches the heads, checks them and compiles.
    """
    validate_checkpoint(base, input_shape)
    scale = normalization_scale(base)
    drift = check_equivalence(base, restructure(base, scale), input_shape)
    extended = build_extended_model(base, scale)
    errors = check_wiring(base, extended, input_shape)
    compile_model(extended)
    report = {
        "normalization_scale": scale,
        "equivalence_drift": drift,
        "wiring_errors": errors,
        "gru": describe_gru(base),
    }
    return extended, report


def load_checkpoint(path: Path, pixel_scale: float = CHECKPOINT_NORMALIZATION) -> checkpoint.LoadedCheckpoint:
    """Load a checkpoint file, rebuilding its topology if Keras cannot read the lambda."""
    return checkpoint.load_base_model(path, pixel_scale)


def main(argv: list[str] | None = None) -> int:
    """Command-line entry point. Returns the process exit code."""
    parser = argparse.ArgumentParser(description="Attach four heads to the DAVE-2-GRU checkpoint.")
    parser.add_argument(
        "--checkpoint",
        choices=sorted(CHECKPOINTS),
        default=DEFAULT_CHECKPOINT,
        help="Checkpoint to extend (default: %(default)s).",
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=None,
        help="Checkpoint .h5 file. Defaults to the file that download_model.py writes.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=EXTENDED_MODEL_PATH,
        help="Where to save the extended model (default: %(default)s).",
    )
    parser.add_argument(
        "--normalization-scale",
        type=float,
        default=CHECKPOINT_NORMALIZATION,
        help=(
            "Pixel scaling to use when the checkpoint's lambda cannot be read. "
            "Ignored when Keras loads the file, because the scaling is then measured "
            "(default: %(default)s, the 1/255 of the training code)."
        ),
    )
    args = parser.parse_args(argv)

    source = args.source or local_checkpoint_path(CHECKPOINTS[args.checkpoint], DATA_DIR)
    if not source.exists():
        print(f"ERROR: {source} does not exist. Run models/download_model.py first.", file=sys.stderr)
        return 1

    try:
        loaded = load_checkpoint(source, args.normalization_scale)
        base = loaded.model
        print("=== Checkpoint summary (check this before continuing) ===")
        base.summary()
        print(f"Loaded: {loaded.note}")
        print(f"Topology: {checkpoint.describe(base)}")
        print(f"GRU settings: {describe_gru(base)}")

        extended, report = extend_checkpoint(base)
    except (UnsupportedCheckpoint, ValueError, RuntimeError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1

    print("=== Extended model summary ===")
    extended.summary()
    print(f"Normalization folded into conv1: {report['normalization_scale']:g} (input stays 0..{PIXEL_MAX:g})")
    print(f"Flat network vs checkpoint, max output difference: {report['equivalence_drift']:.3e}")
    print(f"Head wiring, max difference per head: {report['wiring_errors']}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    extended.save(args.output)
    print(f"OK: saved {args.output} ({args.output.stat().st_size:,} bytes)")
    print("Next: python models/convert.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
