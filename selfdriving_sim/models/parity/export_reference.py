"""Export a Keras reference so the browser model can be compared against it.

The converted TensorFlow.js model must compute what the extended Keras model
computes; the difference should only be the float16 rounding of the weights. This
script writes one reproducible batch of frames and the four heads' Keras outputs
into ``models/data/parity/``, and ``compare_tfjs.mjs`` runs the same batch through
the converted model.

Run from selfdriving_sim/ after extend_model.py and convert.py::

    python models/parity/export_reference.py
    node models/parity/compare_tfjs.mjs

Both folders are inside ``models/data/``, which Git ignores.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import tf_keras as keras  # noqa: E402  (path set above)

from dave2_config import EXTENDED_MODEL_PATH, INPUT_SHAPE, PARITY_DIR, PIXEL_MAX  # noqa: E402

INPUT_BINARY = "reference_input.bin"
REFERENCE_JSON = "reference.json"


def export(source: Path, output_dir: Path, samples: int, seed: int) -> dict:
    """Write the reference batch and return what was written."""
    if not source.exists():
        raise FileNotFoundError(f"{source} does not exist. Run models/extend_model.py first.")
    model = keras.models.load_model(source, compile=False)

    rng = np.random.default_rng(seed)
    frames = (rng.random((samples, *INPUT_SHAPE)) * PIXEL_MAX).astype(np.float32)
    heads = model.predict(frames, verbose=0)
    names = list(model.output_names)

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / INPUT_BINARY).write_bytes(frames.tobytes(order="C"))
    reference = {
        "source": str(source),
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "input_shape": list(frames.shape),
        "input_dtype": "float32",
        "pixel_range": [0.0, PIXEL_MAX],
        "names": names,
        "outputs": [np.asarray(head).tolist() for head in heads],
    }
    (output_dir / REFERENCE_JSON).write_text(json.dumps(reference))
    return reference


def main(argv: list[str] | None = None) -> int:
    """Command-line entry point. Returns the process exit code."""
    parser = argparse.ArgumentParser(description="Export Keras outputs for the TensorFlow.js parity check.")
    parser.add_argument("--source", type=Path, default=EXTENDED_MODEL_PATH, help="Extended .h5 model.")
    parser.add_argument("--output-dir", type=Path, default=PARITY_DIR, help="Where to write the reference.")
    parser.add_argument("--samples", type=int, default=4, help="Frames windows in the batch (default: %(default)s).")
    parser.add_argument("--seed", type=int, default=0, help="Seed of the random batch (default: %(default)s).")
    args = parser.parse_args(argv)

    try:
        reference = export(args.source, args.output_dir, args.samples, args.seed)
    except FileNotFoundError as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1

    print(f"Input: {args.output_dir / INPUT_BINARY} {tuple(reference['input_shape'])} float32")
    print(f"Reference: {args.output_dir / REFERENCE_JSON}")
    for name, values in zip(reference["names"], reference["outputs"]):
        array = np.asarray(values, dtype=np.float32)
        print(f"  {name:<12} min {array.min():+.6f}  max {array.max():+.6f}")
    print("Next: node models/parity/compare_tfjs.mjs")
    return 0


if __name__ == "__main__":
    sys.exit(main())
