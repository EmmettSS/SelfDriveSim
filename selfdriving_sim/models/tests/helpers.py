"""Small stand-ins for the DAVE-2-GRU checkpoint, so the tests run in seconds.

The layout is the published one (see ``models/architecture.py``): a normalization
lambda and convolutions inside a per-frame model, that model wrapped in a single
``TimeDistributed``, then a GRU, a hidden Dense layer and a linear ``Dense(2)``
head. Only the sizes differ from the real checkpoint.
"""

from __future__ import annotations

import base64
import json
from pathlib import Path

import h5py
import tf_keras as keras

from dave2_config import CHECKPOINT_NORMALIZATION

TINY_FRAME_SHAPE = (12, 16, 3)
TINY_INPUT_SHAPE = (3, *TINY_FRAME_SHAPE)


def build_tiny_backbone(
    frame_shape: tuple[int, int, int] = TINY_FRAME_SHAPE,
    normalization: float | None = None,
    offset: float = 0.0,
) -> keras.Model:
    """Return the per-frame model of the checkpoint at test size.

    The default normalization is written with a literal, exactly like the published
    training code, so a saved stand-in stores the same kind of lambda as the real
    checkpoint: a constant inside the code object instead of a captured value.
    """
    image = keras.layers.Input(shape=frame_shape, name="img_in")
    if normalization is None and offset == 0.0:
        pixels = keras.layers.Lambda(lambda value: value / 255.0, name="norm")(image)
    else:
        # Plain floats, so what Keras stores in the file stays JSON serializable.
        scale = float(CHECKPOINT_NORMALIZATION if normalization is None else normalization)
        shift = float(offset)
        pixels = keras.layers.Lambda(lambda value: value * scale + shift, name="norm")(image)
    x = keras.layers.Conv2D(4, 3, strides=2, activation="relu", name="conv2d_1")(pixels)
    x = keras.layers.Dropout(0.1, name="dropout")(x)
    x = keras.layers.Conv2D(8, 3, activation="relu", name="conv2d_2")(x)
    x = keras.layers.Flatten(name="flattened")(x)
    return keras.Model(image, x, name="dave2_backbone")


def build_tiny_checkpoint(
    output_activation: str | None = None,
    gru_activation: str = "tanh",
    normalization: float | None = None,
    offset: float = 0.0,
) -> keras.Model:
    """Return an untrained model shaped like the DAVE-2-GRU checkpoint."""
    frames = keras.layers.Input(shape=TINY_INPUT_SHAPE, name="seq_img_in")
    backbone = build_tiny_backbone(normalization=normalization, offset=offset)
    x = keras.layers.TimeDistributed(backbone, name="td_backbone")(frames)
    x = keras.layers.GRU(8, activation=gru_activation, name="gru")(x)
    x = keras.layers.Dense(6, activation="relu", name="fc_6")(x)
    outputs = keras.layers.Dense(2, activation=output_activation, name="outputs")(x)
    return keras.Model(frames, outputs, name="tiny_dave2_gru_T3")


def build_checkpoint_without_normalization() -> keras.Model:
    """A checkpoint-shaped model whose backbone does not scale its pixels."""
    image = keras.layers.Input(shape=TINY_FRAME_SHAPE, name="img_in")
    x = keras.layers.Conv2D(4, 3, strides=2, activation="relu", name="conv2d_1")(image)
    x = keras.layers.Flatten(name="flattened")(x)
    backbone = keras.Model(image, x, name="dave2_backbone")

    frames = keras.layers.Input(shape=TINY_INPUT_SHAPE, name="seq_img_in")
    x = keras.layers.TimeDistributed(backbone, name="td_backbone")(frames)
    x = keras.layers.GRU(8, name="gru")(x)
    outputs = keras.layers.Dense(2, name="outputs")(x)
    return keras.Model(frames, outputs, name="tiny_dave2_gru_T3")


def save_tiny_checkpoint(path: Path, **kwargs) -> Path:
    """Save a tiny checkpoint as HDF5, the format of the real file."""
    build_tiny_checkpoint(**kwargs).save(Path(path))
    return Path(path)


# The published backbone at the real frame size: five convolutions with dropout
# between them, then a flatten. Sizes and activations are the training code's.
REFERENCE_CONVOLUTIONS = (
    (24, (5, 5), (2, 2)),
    (32, (5, 5), (2, 2)),
    (64, (5, 5), (2, 2)),
    (64, (3, 3), (1, 1)),
    (64, (3, 3), (1, 1)),
)
REFERENCE_GRU_UNITS = 128
REFERENCE_HIDDEN_UNITS = (100, 50)


def build_reference_checkpoint(frame_shape: tuple[int, int, int] = (120, 160, 3)) -> keras.Model:
    """Return an untrained model with the published architecture at the real size.

    The parity test needs the frame size the phone sends, because the browser code
    validates the model against that contract before it runs it.
    """
    image = keras.layers.Input(shape=frame_shape, name="img_in")
    x = keras.layers.Lambda(lambda value: value / 255.0, name="norm")(image)
    for index, (filters, kernel, strides) in enumerate(REFERENCE_CONVOLUTIONS, start=1):
        x = keras.layers.Conv2D(
            filters, kernel, strides=strides, activation="relu", name=f"conv2d_{index}"
        )(x)
        x = keras.layers.Dropout(0.1, name=f"dropout_{index}")(x)
    x = keras.layers.Flatten(name="flattened")(x)
    backbone = keras.Model(image, x, name="dave2_backbone")

    frames = keras.layers.Input(shape=(3, *frame_shape), name="seq_img_in")
    x = keras.layers.TimeDistributed(backbone, name="td_backbone")(frames)
    x = keras.layers.GRU(REFERENCE_GRU_UNITS, dropout=0.1, name="gru")(x)
    for units in REFERENCE_HIDDEN_UNITS:
        x = keras.layers.Dense(units, activation="relu", name=f"fc_{units}")(x)
    outputs = keras.layers.Dense(2, activation="linear", name="outputs")(x)
    return keras.Model(frames, outputs, name="dave2_gru_T3")


def normalization_spec(model_config: dict) -> dict | None:
    """Return the Lambda layer of a model config, looking inside nested models."""
    for layer in model_config["config"]["layers"]:
        nested = layer.get("config", {}).get("layer")
        if isinstance(nested, dict) and "layers" in nested.get("config", {}):
            found = normalization_spec(nested)
            if found is not None:
                return found
        if layer["class_name"] == "Lambda":
            return layer
    return None


def break_stored_lambda(source: Path, destination: Path) -> Path:
    """Copy a checkpoint whose stored normalization lambda no longer unmarshals.

    Keras stores a lambda as marshalled bytecode, which only reads back in the
    Python version that wrote it. Replacing the first byte of that code object
    reproduces the failure while leaving the constants and all weights intact,
    which is what a checkpoint written by another Python version looks like here.
    """
    destination = Path(destination)
    destination.write_bytes(Path(source).read_bytes())
    with h5py.File(destination, "r+") as handle:
        config = json.loads(handle.attrs["model_config"])
        spec = normalization_spec(config)
        code = base64.b64decode(spec["config"]["function"][0])
        spec["config"]["function"][0] = base64.b64encode(b"\xff" + code[1:]).decode("ascii")
        handle.attrs["model_config"] = json.dumps(config)
    return destination


def erase_stored_lambda_code(source: Path, destination: Path) -> Path:
    """Copy a checkpoint whose stored lambda holds no readable constant at all."""
    destination = Path(destination)
    destination.write_bytes(Path(source).read_bytes())
    with h5py.File(destination, "r+") as handle:
        config = json.loads(handle.attrs["model_config"])
        spec = normalization_spec(config)
        spec["config"]["function"][0] = base64.b64encode(b"\x00" * 16).decode("ascii")
        handle.attrs["model_config"] = json.dumps(config)
    return destination
