/**
 * GRU layer for the DAVE-2 checkpoint, which TensorFlow.js 4.22 cannot load.
 *
 * The checkpoint was trained with Keras 2 GRU(reset_after=True), the default.
 * tfjs-layers only implements reset_after=False and throws "GRUCell does not
 * support reset_after parameter set to true" when it sees the flag. Converting
 * to reset_after=False would change the recurrence (the reset gate would apply
 * before the recurrent matmul instead of after it), so the model would no longer
 * compute what it was trained to compute.
 *
 * This layer implements both Keras variants with the same weight layout, so the
 * converted weights load unchanged. The topology's "GRU" class name is renamed to
 * DAVE_GRU_CLASS_NAME at load time (see withResetAfterGRU), which leaves the
 * built-in GRU registration alone.
 */

export const DAVE_GRU_CLASS_NAME = "DaveResetAfterGRU";

const GATES = 3; // Keras order along the last axis: update (z), reset (r), candidate (h).

/**
 * Returns the layer class bound to a TensorFlow.js namespace. The class
 * matches the Keras 2 GRU in gate order, bias layout and output shape.
 */
export function createResetAfterGRU(tf) {
  class DaveResetAfterGRU extends tf.layers.Layer {
    constructor(config) {
      super(config);
      this.units = config.units;
      if (!Number.isInteger(this.units) || this.units < 1) {
        throw new Error(`GRU units must be a positive integer, found ${this.units}`);
      }
      // Keras 2 defaults: reset_after=true, bias on, tanh and sigmoid gates.
      this.resetAfter = config.resetAfter !== false;
      this.returnSequences = Boolean(config.returnSequences);
      const unsupported = [];
      if (config.useBias === false) unsupported.push("use_bias=false");
      if (config.activation && config.activation !== "tanh") unsupported.push(`activation=${config.activation}`);
      if (config.recurrentActivation && config.recurrentActivation !== "sigmoid") {
        unsupported.push(`recurrent_activation=${config.recurrentActivation}`);
      }
      if (config.stateful || config.goBackwards || config.returnState) unsupported.push("stateful/go_backwards/return_state");
      if (unsupported.length > 0) {
        throw new Error(`Unsupported GRU settings for the browser model: ${unsupported.join(", ")}`);
      }
    }

    build(inputShape) {
      const inputDim = inputShape[inputShape.length - 1];
      const width = GATES * this.units;
      const zeros = tf.initializers.zeros();
      this.kernel = this.addWeight("kernel", [inputDim, width], "float32", zeros);
      this.recurrentKernel = this.addWeight("recurrent_kernel", [this.units, width], "float32", zeros);
      // reset_after keeps separate input and recurrent biases: shape [2, 3 * units].
      this.bias = this.addWeight(
        "bias",
        this.resetAfter ? [2, width] : [width],
        "float32",
        zeros,
      );
      this.built = true;
    }

    computeOutputShape(inputShape) {
      return this.returnSequences
        ? [inputShape[0], inputShape[1], this.units]
        : [inputShape[0], this.units];
    }

    call(inputs) {
      return tf.tidy(() => {
        const sequence = Array.isArray(inputs) ? inputs[0] : inputs;
        const [batch, steps, inputDim] = sequence.shape;
        const kernel = this.kernel.read();
        const recurrentKernel = this.recurrentKernel.read();
        const [inputBias, recurrentBias] = this.resetAfter
          ? tf.unstack(this.bias.read(), 0)
          : [this.bias.read(), null];

        let state = tf.zeros([batch, this.units]);
        const outputs = [];
        for (let t = 0; t < steps; t += 1) {
          const step = sequence.slice([0, t, 0], [batch, 1, inputDim]).reshape([batch, inputDim]);
          state = this.resetAfter
            ? this.stepResetAfter(step, state, kernel, recurrentKernel, inputBias, recurrentBias)
            : this.stepClassic(step, state, kernel, recurrentKernel, inputBias);
          outputs.push(state);
        }
        return this.returnSequences ? tf.stack(outputs, 1) : state;
      });
    }

    /** reset_after=true: the reset gate multiplies the recurrent candidate after its matmul. */
    stepResetAfter(x, h, kernel, recurrentKernel, inputBias, recurrentBias) {
      const xProj = tf.add(tf.matMul(x, kernel), inputBias);
      const hProj = tf.add(tf.matMul(h, recurrentKernel), recurrentBias);
      const [xz, xr, xh] = tf.split(xProj, GATES, 1);
      const [hz, hr, hh] = tf.split(hProj, GATES, 1);
      const z = tf.sigmoid(tf.add(xz, hz));
      const r = tf.sigmoid(tf.add(xr, hr));
      const candidate = tf.tanh(tf.add(xh, tf.mul(r, hh)));
      return tf.add(tf.mul(z, h), tf.mul(tf.sub(1, z), candidate));
    }

    /** reset_after=false: the reset gate multiplies the state before the candidate matmul. */
    stepClassic(x, h, kernel, recurrentKernel, bias) {
      const xProj = tf.add(tf.matMul(x, kernel), bias);
      const [xz, xr, xh] = tf.split(xProj, GATES, 1);
      const [hz, hr] = tf.split(tf.matMul(h, recurrentKernel.slice([0, 0], [this.units, 2 * this.units])), 2, 1);
      const z = tf.sigmoid(tf.add(xz, hz));
      const r = tf.sigmoid(tf.add(xr, hr));
      const candidateRecurrent = tf.matMul(
        tf.mul(r, h),
        recurrentKernel.slice([0, 2 * this.units], [this.units, this.units]),
      );
      const candidate = tf.tanh(tf.add(xh, candidateRecurrent));
      return tf.add(tf.mul(z, h), tf.mul(tf.sub(1, z), candidate));
    }

    getConfig() {
      return {
        ...super.getConfig(),
        units: this.units,
        resetAfter: this.resetAfter,
        returnSequences: this.returnSequences,
      };
    }
  }
  DaveResetAfterGRU.className = DAVE_GRU_CLASS_NAME;
  return DaveResetAfterGRU;
}

let registered = null;

/** Registers the layer class once per TensorFlow.js namespace. */
export function registerResetAfterGRU(tf) {
  if (registered?.tf !== tf) {
    registered = { tf, layer: createResetAfterGRU(tf) };
    tf.serialization.registerClass(registered.layer);
  }
  return registered.layer;
}

/**
 * Renames every "GRU" layer in the topology to DAVE_GRU_CLASS_NAME. Other layers are
 * untouched. Returns the artifacts so the function can wrap an IO handler.
 */
export function renameGRULayers(artifacts) {
  const layers = artifacts?.modelTopology?.model_config?.config?.layers ?? [];
  for (const layer of layers) {
    if (layer.class_name === "GRU") layer.class_name = DAVE_GRU_CLASS_NAME;
  }
  return artifacts;
}

/**
 * Wraps a model source (a URL or an IO handler) so that it loads with the
 * reset-after GRU layer. The class is registered before the first load.
 */
export function withResetAfterGRU(tf, source) {
  registerResetAfterGRU(tf);
  const handler = typeof source === "string" ? tf.io.browserHTTPRequest(source) : source;
  return {
    load: async () => renameGRULayers(await handler.load()),
  };
}
