"""Read a DAVE-2-GRU checkpoint into memory, whichever Python version saved it.

``tf_keras.models.load_model`` is tried first because it reads the file's own
topology. When the file's normalization lambda was marshalled by another Python
version, loading fails with ``bad marshal data``; the topology is then rebuilt
from the JSON in the same file (see ``architecture.py``) and the weights are read
into it. Both paths return the same model, so the rest of the pipeline does not
care which one was used.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
import tf_keras as keras

import architecture
from dave2_config import CHECKPOINT_NORMALIZATION, PIXEL_MAX

# Tolerance for treating the normalization layer as a plain multiplication.
SCALING_TOLERANCE = 1e-6


class UnsupportedCheckpoint(ValueError):
    """The file is not a DAVE-2-GRU checkpoint this pipeline can extend."""


@dataclass(frozen=True)
class LoadedCheckpoint:
    """A checkpoint in memory, plus how it got there."""

    model: keras.Model
    rebuilt: bool
    note: str


def read_model_config(path: Path) -> dict:
    """Return the Keras topology JSON stored in an HDF5 model file."""
    with h5py.File(path, "r") as handle:
        raw = handle.attrs.get("model_config")
    if raw is None:
        raise UnsupportedCheckpoint(f"{path} has no model_config attribute; it is not a Keras HDF5 model.")
    return json.loads(raw.decode("utf-8") if isinstance(raw, bytes) else raw)


def frame_backbone(model: keras.Model) -> keras.Model:
    """Return the per-frame model the checkpoint wraps in its TimeDistributed layer."""
    wrappers = [layer for layer in model.layers if isinstance(layer, keras.layers.TimeDistributed)]
    if len(wrappers) != 1:
        raise UnsupportedCheckpoint(
            f"Expected one TimeDistributed frame backbone, found {len(wrappers)}."
        )
    backbone = wrappers[0].layer
    if not isinstance(backbone, keras.Model):
        raise UnsupportedCheckpoint(
            f"TimeDistributed wraps {type(backbone).__name__}, not the per-frame model of the checkpoint."
        )
    return backbone


def normalization_layer(model: keras.Model) -> keras.layers.Layer:
    """Return the backbone layer that scales the pixels before the first convolution."""
    backbone = frame_backbone(model)
    found = [layer for layer in backbone.layers if isinstance(layer, keras.layers.Lambda)]
    if len(found) != 1:
        raise UnsupportedCheckpoint(
            f"Expected one normalization Lambda in the frame backbone, found {len(found)}. "
            "The pixel range the checkpoint was trained on cannot be determined, so it is "
            "not safe to convert this file."
        )
    return found[0]


def normalization_scale(model: keras.Model) -> float:
    """Measure the backbone's pixel scaling by feeding it known constant frames.

    The published training code divides by 255. The value is measured instead of
    assumed, so a checkpoint that normalizes differently is rejected here rather
    than driving with the wrong pixel range.
    """
    backbone = frame_backbone(model)
    norm = normalization_layer(model)
    probe = keras.Model(backbone.input, norm.output)
    shape = (1, *[int(size) for size in backbone.input_shape[1:]])

    full = np.full(shape, PIXEL_MAX, dtype=np.float32)
    half = np.full(shape, PIXEL_MAX / 2, dtype=np.float32)
    zero = np.zeros(shape, dtype=np.float32)
    out_full, out_half, out_zero = (probe.predict(frame, verbose=0) for frame in (full, half, zero))

    scale = float(np.mean(out_full) / PIXEL_MAX)
    if scale <= 0:
        raise UnsupportedCheckpoint(f"The normalization layer scales pixels by {scale}.")
    residual = float(np.max(np.abs(out_full - full * scale)))
    offset = float(np.max(np.abs(out_zero)))
    ratio = float(np.mean(out_full) / np.mean(out_half)) if np.mean(out_half) else float("nan")
    if residual > SCALING_TOLERANCE or offset > SCALING_TOLERANCE or abs(ratio - 2.0) > 1e-4:
        raise UnsupportedCheckpoint(
            "The normalization layer is not a plain scaling of the pixels "
            f"(scale {scale}, residual {residual}, offset {offset}, full/half ratio {ratio}). "
            "extend_model.py folds a plain scaling into the first convolution and cannot fold this."
        )
    return scale


def _is_marshal_failure(error: Exception) -> bool:
    return "marshal" in str(error).lower()


def _rebuild(path: Path, pixel_scale: float, cause: Exception) -> keras.Model:
    """Rebuild the topology from the file's JSON and read the weights into it."""
    model_config = read_model_config(path)
    outer = architecture.layer_specs(model_config)
    wrapper = next((spec for spec in outer if spec["class_name"] == "TimeDistributed"), None)
    nested = architecture.nested_model_spec(wrapper) if wrapper else None
    norm = architecture.find_normalization(architecture.layer_specs(nested)) if nested else None
    if norm is None:
        raise UnsupportedCheckpoint(
            f"{path} cannot be loaded ({cause}) and its topology has no normalization Lambda "
            "to rebuild."
        ) from cause
    if not architecture.lambda_mentions_constant(norm, PIXEL_MAX):
        raise UnsupportedCheckpoint(
            f"{path} cannot be loaded ({cause}), and the stored code of its normalization "
            f"layer does not contain the constant {PIXEL_MAX:g}. Refusing to guess the pixel "
            "range; pass --normalization-scale with the value from the training code."
        ) from cause

    model = architecture.build_model(model_config, pixel_scale)
    model.load_weights(str(path))
    return model


def load_base_model(path: Path, pixel_scale: float = CHECKPOINT_NORMALIZATION) -> LoadedCheckpoint:
    """Return the checkpoint's two-output model, without its optimizer state."""
    if not Path(path).exists():
        raise UnsupportedCheckpoint(f"{path} does not exist. Run models/download_model.py first.")
    try:
        model = keras.models.load_model(path, compile=False)
    except Exception as error:  # noqa: BLE001 - any Keras failure may hide marshalled bytecode
        if not _is_marshal_failure(error):
            raise
        model = _rebuild(Path(path), pixel_scale, error)
        return LoadedCheckpoint(
            model=model,
            rebuilt=True,
            note=(
                "Keras could not read the stored normalization lambda "
                f"({error}), so the topology was rebuilt from the file's JSON "
                f"with a {pixel_scale:g} scaling and the weights were read from the file."
            ),
        )
    return LoadedCheckpoint(model=model, rebuilt=False, note="Keras read the checkpoint directly.")


def describe(model: keras.Model) -> dict:
    """Return the facts about a checkpoint that the scripts print before converting."""
    return {
        "name": model.name,
        "input_shape": tuple(int(size) for size in model.input_shape[1:]),
        "output_shape": tuple(int(size) for size in model.outputs[0].shape[1:]),
        "parameters": int(model.count_params()),
        "layers": [f"{layer.name} ({type(layer).__name__})" for layer in model.layers],
    }
