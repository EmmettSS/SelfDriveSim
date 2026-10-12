/**
 * Compare the converted TensorFlow.js model with the Keras reference.
 *
 * Loads client/static/client/models/dave2/ through the same code path the phone
 * uses (loadModel, realInference and toCommand from inference.js, with the custom
 * reset-after GRU) and checks it against models/data/parity/reference.json, which
 * models/parity/export_reference.py writes from the extended Keras model.
 *
 * Run from selfdriving_sim/ after convert.py::
 *
 *     node models/parity/compare_tfjs.mjs [modelDir] [referenceDir]
 *
 * The remaining difference should be the float16 rounding of the weights, about
 * 1e-5 for this network. Exits with status 1 when a head differs by more than the
 * tolerance (PARITY_TOLERANCE, default 1e-3).
 */

import fs from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";

import * as tf from "@tensorflow/tfjs";

import { loadModel, realInference, toCommand } from "../../client/static/client/js/inference.js";

const here = path.dirname(fileURLToPath(import.meta.url));
const projectRoot = path.resolve(here, "../..");
const modelDir = path.resolve(projectRoot, process.argv[2] ?? "client/static/client/models/dave2");
const referenceDir = path.resolve(projectRoot, process.argv[3] ?? "models/data/parity");
const tolerance = Number(process.env.PARITY_TOLERANCE ?? 1e-3);

/** Reads a converted model from disk the way the browser reads it over HTTP. */
function fileHandler(directory) {
  return {
    load: async () => {
      const json = JSON.parse(await fs.readFile(path.join(directory, "model.json"), "utf8"));
      const paths = json.weightsManifest.flatMap((group) => group.paths);
      const chunks = await Promise.all(paths.map((file) => fs.readFile(path.join(directory, file))));
      const weightData = new Uint8Array(chunks.reduce((total, chunk) => total + chunk.length, 0));
      let offset = 0;
      for (const chunk of chunks) {
        weightData.set(chunk, offset);
        offset += chunk.length;
      }
      return {
        modelTopology: json.modelTopology,
        format: json.format,
        generatedBy: json.generatedBy,
        convertedBy: json.convertedBy,
        weightSpecs: json.weightsManifest.flatMap((group) => group.weights),
        weightData: weightData.buffer,
      };
    },
  };
}

const reference = JSON.parse(await fs.readFile(path.join(referenceDir, "reference.json"), "utf8"));
// A Node Buffer is a view into a pooled ArrayBuffer, so the float32 batch is read
// from a copy of exactly its own bytes.
const bytes = await fs.readFile(path.join(referenceDir, "reference_input.bin"));
const pixels = new Float32Array(bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength));

await tf.setBackend("cpu");
await tf.ready();

console.log(`Model:     ${modelDir}`);
console.log(`Reference: ${reference.source} (${reference.generated_utc})`);

const model = await loadModel(tf, fileHandler(modelDir));
const input = tf.tensor(pixels, reference.input_shape);
const predicted = model.predict(input);
const heads = Array.isArray(predicted) ? predicted : [predicted];
const values = await Promise.all(heads.map((head) => head.data()));

let worst = 0;
reference.names.forEach((name, index) => {
  const expected = reference.outputs[index].flat();
  const actual = Array.from(values[index]);
  const difference = Math.max(...actual.map((value, position) => Math.abs(value - expected[position])));
  worst = Math.max(worst, difference);
  console.log(
    `  ${name.padEnd(12)} tfjs ${actual[0].toFixed(6)}  keras ${expected[0].toFixed(6)}  max diff ${difference.toExponential(3)}`,
  );
});

// The command the phone would send, from the same frames, against the command the
// Keras outputs map to. Rounding and clamping must agree as well.
const sample = tf.unstack(input)[0];
const frames = tf.unstack(sample);
const command = await realInference(tf, model, frames);
const expectedCommand = toCommand({
  steering: reference.outputs[0][0][0],
  throttle: reference.outputs[1][0][0],
  brake: reference.outputs[2][0][0],
  turnProbabilities: reference.outputs[3][0],
});
frames.forEach((frame) => frame.dispose());
sample.dispose();
input.dispose();
heads.forEach((head) => head.dispose());

console.log(`  command      tfjs ${JSON.stringify(command)}`);
console.log(`  command      keras ${JSON.stringify(expectedCommand)}`);
const commandsMatch = JSON.stringify(command) === JSON.stringify(expectedCommand);

model.dispose();

if (worst > tolerance || !commandsMatch) {
  console.error(`FAIL: max diff ${worst.toExponential(3)} (tolerance ${tolerance}), commands match: ${commandsMatch}`);
  process.exit(1);
}
console.log(`OK: max |tfjs - keras| = ${worst.toExponential(3)} (tolerance ${tolerance})`);
