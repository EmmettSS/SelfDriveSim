"""Convert the extended model to TensorFlow.js with float16 weights (phase 4, task 3).

Run from selfdriving_sim/ after extend_model.py::

    python models/convert.py

The converter writes ``model.json`` and the weight shards into
``client/static/client/models/dave2/``. The script prints the size of every
file and fails when the total is not below ``MAX_WEB_MODEL_BYTES``.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

from dave2_config import EXTENDED_MODEL_PATH, MAX_WEB_MODEL_BYTES, WEB_MODEL_DIR

CONVERTER_NAME = "tensorflowjs_converter"
MODEL_JSON = "model.json"
WEIGHT_SHARD_GLOB = "group*-shard*.bin"


class ConversionError(RuntimeError):
    """A conversion failure with a message the user can act on."""


def find_converter() -> str:
    """Return the converter that belongs to the active environment."""
    beside_python = Path(sys.executable).with_name(CONVERTER_NAME)
    if beside_python.exists():
        return str(beside_python)
    on_path = shutil.which(CONVERTER_NAME)
    if on_path:
        return on_path
    raise ConversionError(
        f"{CONVERTER_NAME} was not found. Install requirements.txt into the active environment."
    )


def clear_previous_output(output_dir: Path) -> None:
    """Delete files from an earlier conversion so stale shards cannot inflate the total."""
    output_dir.mkdir(parents=True, exist_ok=True)
    for path in [output_dir / MODEL_JSON, *output_dir.glob(WEIGHT_SHARD_GLOB)]:
        path.unlink(missing_ok=True)


def build_command(converter: str, source: Path, output_dir: Path) -> list[str]:
    """Return the converter command: Keras HDF5 in, TF.js layers model out, float16 weights."""
    # --quantize_float16 takes an optional value (a node-name filter), so argparse would
    # consume the next positional argument. It must come after the two paths, and it
    # must not be given a value, because that would select no weights.
    return [
        converter,
        "--input_format=keras",
        "--output_format=tfjs_layers_model",
        str(source),
        str(output_dir),
        "--quantize_float16",
    ]


def artifact_sizes(output_dir: Path) -> dict[str, int]:
    """Return the size in bytes of every file in the output directory."""
    return {path.name: path.stat().st_size for path in sorted(output_dir.iterdir()) if path.is_file()}


def check_budget(sizes: dict[str, int], limit: int | None = None) -> int:
    """Return the total size, or raise ConversionError when it is not below the limit."""
    if limit is None:
        limit = MAX_WEB_MODEL_BYTES
    if MODEL_JSON not in sizes:
        raise ConversionError(f"The converter did not write {MODEL_JSON}.")
    total = sum(sizes.values())
    if total >= limit:
        raise ConversionError(f"Converted model is {total:,} bytes; the limit is {limit:,} bytes.")
    return total


def convert(source: Path, output_dir: Path) -> dict[str, int]:
    """Run the converter and return the size of each produced file.

    The output folder is only cleared once the input and converter are known to
    exist. If the conversion fails or the result is over budget, the folder is
    cleared again, so a partial or oversized model cannot be committed by accident.
    """
    if not source.exists():
        raise ConversionError(f"{source} does not exist. Run models/extend_model.py first.")
    converter = find_converter()

    clear_previous_output(output_dir)
    try:
        subprocess.run(build_command(converter, source, output_dir), check=True)
    except subprocess.CalledProcessError as exc:
        clear_previous_output(output_dir)
        raise ConversionError(f"The converter exited with status {exc.returncode}.") from exc

    sizes = artifact_sizes(output_dir)
    try:
        check_budget(sizes)
    except ConversionError:
        clear_previous_output(output_dir)
        raise
    return sizes


def main(argv: list[str] | None = None) -> int:
    """Command-line entry point. Returns the process exit code."""
    parser = argparse.ArgumentParser(description="Convert the extended DAVE-2-GRU model to TensorFlow.js.")
    parser.add_argument("--source", type=Path, default=EXTENDED_MODEL_PATH, help="Extended .h5 model.")
    parser.add_argument("--output-dir", type=Path, default=WEB_MODEL_DIR, help="TF.js output folder.")
    args = parser.parse_args(argv)

    try:
        sizes = convert(args.source, args.output_dir)
    except ConversionError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    for name, size in sizes.items():
        print(f"{name:<32} {size:>12,} bytes")
    total = sum(sizes.values())
    print(f"{'total':<32} {total:>12,} bytes (limit {MAX_WEB_MODEL_BYTES:,})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
