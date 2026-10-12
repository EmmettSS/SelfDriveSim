"""Rebuild a DAVE-2-GRU checkpoint from the topology JSON stored inside its HDF5 file.

Why this module exists: the checkpoint normalizes its input with
``Lambda(lambda t: t / 255.0, name="norm")``. Keras stores a lambda as marshalled
Python bytecode, and bytecode only unmarshals in the Python version that wrote it.
A checkpoint saved under Python 3.10 (TensorFlow 2.9, the training environment)
therefore fails to load under a newer Python with ``ValueError: bad marshal data``.

The topology next to that bytecode is plain JSON, so it can be read without
touching the code object. This module rebuilds the same layers from that JSON,
with a normalization lambda written here, and ``Model.load_weights`` then fills
them from the untouched weights in the file. Layer names and their order come
from the file, which is what ``load_weights`` matches on.

The topology mirrors the published training code at
https://github.com/MaximIgenbergs/thesis-robust-e2e-vision-ads
(``scripts/udacity/models/dave2_gru/model.py``): a per-frame CNN backbone wrapped
in one ``TimeDistributed``, a GRU, two hidden Dense layers and a linear
``Dense(2)`` head.
"""

from __future__ import annotations

import base64
import binascii
import json
import struct

# The checkpoint is a Keras 2 HDF5 file (its GRU config includes 'time_major',
# which Keras 3 rejects), so the rebuild uses tf_keras too.
import tf_keras as keras

# Layer types this module can rebuild. Anything else means the checkpoint is not
# the published architecture, and guessing would silently produce a wrong model.
REBUILDABLE_LAYERS = ("Conv2D", "Dense", "Dropout", "Flatten", "GRU")


class UnsupportedTopology(ValueError):
    """The file describes a topology this module does not rebuild."""


def layer_specs(model_config: dict) -> list[dict]:
    """Return the layer specifications of a Keras model config."""
    specs = model_config.get("config", {}).get("layers")
    if not isinstance(specs, list) or not specs:
        raise UnsupportedTopology("The model config has no layer list.")
    return specs


def single_endpoint(model_config: dict, key: str) -> str:
    """Return the name of the config's only input (or output) layer."""
    endpoints = model_config.get("config", {}).get(key) or []
    if len(endpoints) != 1:
        raise UnsupportedTopology(
            f"Expected exactly one {key.removesuffix('_layers')} layer, found {len(endpoints)}."
        )
    return endpoints[0][0]


def nested_model_spec(spec: dict) -> dict | None:
    """Return the model config a TimeDistributed wrapper holds, if it holds one."""
    inner = spec.get("config", {}).get("layer")
    if isinstance(inner, dict) and isinstance(inner.get("config", {}).get("layers"), list):
        return inner
    return None


def find_normalization(specs: list[dict]) -> dict | None:
    """Return the Lambda layer that scales the pixels, or None when there is none."""
    return next((spec for spec in specs if spec["class_name"] == "Lambda"), None)


def lambda_mentions_constant(spec: dict, value: float) -> bool:
    """Check a stored Lambda for a numeric constant, in its code or its closure.

    The bytecode cannot be executed here — that is the reason for this module —
    but a constant survives inside it as a little-endian float or as decimal text,
    depending on how the Python version marshalled floats. A scaling can also be
    captured as a closure value, which Keras stores as plain JSON next to the code.
    Finding either form of ``value`` or of its reciprocal confirms the pixel range;
    finding neither means the file does not describe the published normalization.
    """
    function = spec.get("config", {}).get("function")
    if not isinstance(function, (list, tuple)) or not function:
        return False

    haystack = b""
    blob = function[0]
    if isinstance(blob, str):
        try:
            haystack += base64.b64decode(blob.encode("ascii", "ignore"), validate=False)
        except (binascii.Error, ValueError):
            pass
    try:
        haystack += json.dumps(function[1:]).encode("ascii")
    except (TypeError, ValueError):
        pass

    number = float(value)
    candidates = {number, 1.0 / number} if number else {number}
    for candidate in candidates:
        for pattern in (
            struct.pack("<d", candidate),
            struct.pack("<f", candidate),
            repr(candidate).encode("ascii"),
            f"{candidate:g}".encode("ascii"),
        ):
            if pattern in haystack:
                return True
    return False


def build_model(model_config: dict, normalization_scale: float) -> keras.Model:
    """Rebuild one model (nested or top level) from its Keras config JSON."""
    specs = layer_specs(model_config)
    input_name = single_endpoint(model_config, "input_layers")
    output_name = single_endpoint(model_config, "output_layers")

    tensors = {}  # layer name -> symbolic tensor, to find the input and output ends
    tensor = None
    for spec in specs:
        layer_type = spec["class_name"]
        config = dict(spec["config"])
        name = config["name"]

        if layer_type == "InputLayer":
            shape = tuple(config["batch_input_shape"][1:])
            tensor = keras.layers.Input(shape=shape, name=name)
        elif layer_type == "Lambda":
            # The checkpoint's own lambda cannot be unmarshalled; this one computes
            # the same scaling and keeps the layer name, so weights still match.
            scale = float(normalization_scale)
            tensor = keras.layers.Lambda(lambda pixels: pixels * scale, name=name)(tensor)
        elif layer_type == "TimeDistributed":
            nested = nested_model_spec(spec)
            if nested is None:
                raise UnsupportedTopology(
                    f"TimeDistributed layer {name!r} does not wrap a model this module can rebuild."
                )
            tensor = keras.layers.TimeDistributed(
                build_model(nested, normalization_scale), name=name
            )(tensor)
        elif layer_type in REBUILDABLE_LAYERS:
            tensor = getattr(keras.layers, layer_type).from_config(config)(tensor)
        else:
            raise UnsupportedTopology(
                f"Layer {name!r} has type {layer_type!r}, which this module does not rebuild. "
                f"Supported: {', '.join(REBUILDABLE_LAYERS)}, InputLayer, Lambda, TimeDistributed."
            )
        tensors[name] = tensor

    for required in (input_name, output_name):
        if required not in tensors:
            raise UnsupportedTopology(f"The config names {required!r}, which is not one of its layers.")
    return keras.Model(
        inputs=tensors[input_name],
        outputs=tensors[output_name],
        name=model_config.get("config", {}).get("name"),
    )
