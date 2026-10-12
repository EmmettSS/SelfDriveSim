"""Writes the Keras reference fixture for the browser GRU layer.

The fixture holds small GRU layers with fixed random weights and the outputs
Keras computes for them, for both reset_after variants. The Node test
client/tests/js/reset-after-gru.test.mjs replays the same weights through
client/static/client/js/reset-after-gru.js and compares the results.

Run from the project root after changing the layer:

    python models/tests/make_gru_fixture.py
"""

import json
import os
from pathlib import Path

import numpy as np

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import tf_keras as keras  # noqa: E402  (same Keras build as the checkpoint loader)

FIXTURE_PATH = Path(__file__).resolve().parents[2] / "client/tests/js/fixtures/gru_fixture.json"
INPUT_DIM = 5
STEPS = 3
BATCH = 2

# (case name, units, reset_after, return_sequences, seed)
CASES = (
    ("reset_after_true_last_state", 4, True, False, 1),
    ("reset_after_true_sequence", 3, True, True, 2),
    ("reset_after_false_last_state", 4, False, False, 3),
    ("reset_after_false_sequence", 3, False, True, 4),
)


def build_case(units, reset_after, return_sequences, seed):
    rng = np.random.default_rng(seed)
    inputs = keras.Input(shape=(STEPS, INPUT_DIM))
    outputs = keras.layers.GRU(
        units, reset_after=reset_after, return_sequences=return_sequences, name="gru"
    )(inputs)
    model = keras.Model(inputs, outputs)
    model.set_weights([rng.normal(0.0, 0.4, w.shape).astype("float32") for w in model.weights])
    x = rng.normal(0.0, 1.0, (BATCH, STEPS, INPUT_DIM)).astype("float32")
    expected = model.predict(x, verbose=0)

    weights = {}
    for variable in model.weights:
        key = variable.name.split("/")[-1].split(":")[0]
        weights[key] = variable.numpy().tolist()
    return {
        "units": units,
        "resetAfter": reset_after,
        "returnSequences": return_sequences,
        "weights": weights,  # keys follow Keras: kernel, recurrent_kernel, bias
        "input": x.tolist(),
        "expected": np.asarray(expected).tolist(),
    }


def build_fixture():
    return {
        "description": "Keras GRU reference outputs for client/static/client/js/reset-after-gru.js",
        "cases": {name: build_case(units, reset, seq, seed) for name, units, reset, seq, seed in CASES},
    }


def main():
    FIXTURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    FIXTURE_PATH.write_text(json.dumps(build_fixture(), indent=1) + "\n", encoding="utf-8")
    print(f"Wrote {FIXTURE_PATH} ({FIXTURE_PATH.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
