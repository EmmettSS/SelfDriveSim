"""End-to-end test: extend a checkpoint, convert it, and compare it in TensorFlow.js.

This is the strongest check available without a browser. The converted model is
loaded through the JavaScript the phone runs (``loadModel``, ``realInference`` and
``toCommand`` from ``client/static/client/js/inference.js``, with the custom
reset-after GRU) and must return the commands Keras returns. The stand-in has the
published architecture at the real frame size, because the browser code checks the
model against that contract before running it.

Node.js and the project's ``node_modules`` are needed; without them the test is
skipped.
"""

import json
import shutil
import subprocess
from pathlib import Path

import pytest

import convert
import extend_model
from dave2_config import INPUT_SHAPE, OUTPUT_NAMES
from models.tests.helpers import build_reference_checkpoint
from parity.export_reference import export

PROJECT_ROOT = Path(__file__).resolve().parents[2]
COMPARE_SCRIPT = PROJECT_ROOT / "models" / "parity" / "compare_tfjs.mjs"
NODE = shutil.which("node")
TFJS = PROJECT_ROOT / "node_modules" / "@tensorflow" / "tfjs"

pytestmark = pytest.mark.skipif(
    NODE is None or not TFJS.exists(),
    reason="needs Node.js and 'npm ci' in selfdriving_sim/",
)


@pytest.fixture(scope="module")
def converted_model(tmp_path_factory):
    """Return (model folder, reference folder) for an extended and converted stand-in."""
    root = tmp_path_factory.mktemp("parity")
    extended, _ = extend_model.extend_checkpoint(build_reference_checkpoint())
    source = root / "dave2_gru_extended.h5"
    extended.save(source)

    model_dir = root / "web"
    convert.convert(source, model_dir)
    reference_dir = root / "reference"
    export(source, reference_dir, samples=2, seed=0)
    return model_dir, reference_dir


def test_the_converted_model_computes_the_keras_outputs(converted_model):
    model_dir, reference_dir = converted_model

    result = subprocess.run(
        [NODE, str(COMPARE_SCRIPT), str(model_dir), str(reference_dir)],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        timeout=900,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert "OK: max |tfjs - keras|" in result.stdout
    for name in OUTPUT_NAMES:
        assert name in result.stdout


def test_the_reference_holds_the_contract_batch(converted_model):
    _, reference_dir = converted_model
    reference = json.loads((reference_dir / "reference.json").read_text())

    assert reference["input_shape"][1:] == list(INPUT_SHAPE)
    assert reference["pixel_range"] == [0.0, 255.0]
    assert reference["names"] == list(OUTPUT_NAMES)
    assert (reference_dir / "reference_input.bin").stat().st_size > 0
