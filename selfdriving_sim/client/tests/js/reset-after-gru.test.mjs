import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { test } from "node:test";

import * as tf from "@tensorflow/tfjs";

import {
  DAVE_GRU_CLASS_NAME,
  createResetAfterGRU,
  registerResetAfterGRU,
  renameGRULayers,
} from "../../static/client/js/reset-after-gru.js";

// Reference outputs written by models/tests/make_gru_fixture.py (Keras, both reset_after variants).
const fixture = JSON.parse(readFileSync(new URL("./fixtures/gru_fixture.json", import.meta.url), "utf8"));
const TOLERANCE = 1e-5;
const Layer = createResetAfterGRU(tf);

function buildModel(spec) {
  const inputs = tf.input({ shape: [3, 5] });
  const layer = new Layer({
    units: spec.units,
    resetAfter: spec.resetAfter,
    returnSequences: spec.returnSequences,
    name: "gru",
  });
  const model = tf.model({ inputs, outputs: layer.apply(inputs) });
  // Keras weight order: kernel, recurrent_kernel, bias.
  const weights = [spec.weights.kernel, spec.weights.recurrent_kernel, spec.weights.bias];
  model.setWeights(weights.map((values) => tf.tensor(values)));
  return model;
}

for (const [name, spec] of Object.entries(fixture.cases)) {
  test(`matches Keras output for ${name}`, () => {
    const model = buildModel(spec);
    const predicted = model.predict(tf.tensor(spec.input));
    const actual = predicted.dataSync();
    const expected = spec.expected.flat(Infinity);
    assert.equal(actual.length, expected.length);
    const worst = expected.reduce((max, value, index) => Math.max(max, Math.abs(value - actual[index])), 0);
    assert.ok(worst < TOLERANCE, `max difference ${worst} exceeds ${TOLERANCE}`);
    predicted.dispose();
    model.dispose();
  });
}

test("rejects GRU settings that the layer does not implement", () => {
  assert.throws(() => new Layer({ units: 4, useBias: false }), /use_bias=false/);
  assert.throws(() => new Layer({ units: 4, activation: "relu" }), /activation=relu/);
  assert.throws(() => new Layer({ units: 4, recurrentActivation: "hard_sigmoid" }), /recurrent_activation=hard_sigmoid/);
  assert.throws(() => new Layer({ units: 4, goBackwards: true }), /stateful\/go_backwards\/return_state/);
  assert.throws(() => new Layer({ units: 0 }), /positive integer/);
});

test("renames only the GRU layers in a topology", () => {
  const artifacts = {
    modelTopology: {
      model_config: {
        class_name: "Functional",
        config: {
          layers: [
            { class_name: "InputLayer", name: "input" },
            { class_name: "GRU", name: "gru" },
            { class_name: "Dense", name: "steering" },
          ],
        },
      },
    },
  };
  renameGRULayers(artifacts);
  const classes = artifacts.modelTopology.model_config.config.layers.map((layer) => layer.class_name);
  assert.deepEqual(classes, ["InputLayer", DAVE_GRU_CLASS_NAME, "Dense"]);
});

test("registers the layer so a renamed topology deserializes", async () => {
  const registered = registerResetAfterGRU(tf);
  const model = await tf.models.modelFromJSON({
    modelTopology: {
      model_config: {
        class_name: "Sequential",
        config: {
          name: "probe",
          layers: [
            {
              class_name: DAVE_GRU_CLASS_NAME,
              config: { name: "gru", units: 4, reset_after: true, return_sequences: false, batch_input_shape: [null, 3, 5] },
            },
          ],
        },
      },
    },
  });
  assert.ok(model.layers[0] instanceof registered);
  assert.equal(model.layers[0].units, 4);
  model.dispose();
});
