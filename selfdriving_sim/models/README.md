# models

Scripts that run outside the Django request path: fetch the DAVE-2-GRU checkpoint, extend it to the four controls the simulator protocol needs, and convert it to TensorFlow.js for the phone. Everything here is phase 4; the phone runs the converted model in `client/static/client/models/dave2/`.

All commands run from `selfdriving_sim/` with the virtual environment active.

## Where the model comes from

| | |
|---|---|
| Repository | [`maxim-igenbergs/dave2-gru`](https://huggingface.co/maxim-igenbergs/dave2-gru) on Hugging Face |
| Revision | `dddcf9fc24fc258bb809f276c021d4c66aa4ceb5` (pinned in `dave2_config.HF_REVISION`) |
| Default checkpoint | `genroads_20251215-174930/best_model.h5`, 33,328,880 bytes |
| SHA-256 | `fbee0740ce53b0ad7be35b708c0242fc98143699f8f7024af0d21d28719a32bf` |
| Alternative | `jungle_20251201-142321/best_model.h5` (`--checkpoint jungle`) |
| Training code | [`MaximIgenbergs/thesis-robust-e2e-vision-ads`](https://github.com/MaximIgenbergs/thesis-robust-e2e-vision-ads), `scripts/udacity/models/dave2_gru/` |

The file is a Keras 2 HDF5 model: architecture and weights together, written by TensorFlow 2.9. Its `meta.json` records the input as `[3, 120, 160, 3]`, `seq_len 3`, Adam at `1e-4` and `alpha_steer 0.7`. The model card's `66 × 200` diagram does not match the trained weights; `dave2_config.py` follows `meta.json`.

Both digests and sizes are pinned in `dave2_config.CHECKPOINTS`, and `download_model.py` verifies them. A 133-byte download is a Git LFS pointer, not a model; delete it and download again.

## What Git stores

| Path | In Git | Why |
|---|---|---|
| `models/*.py`, `models/parity/*`, `models/tests/*` | yes | the pipeline |
| `models/data/` (checkpoints, extended `.h5`, parity references) | no | large binaries, reproducible from the pinned revision |
| `client/static/client/models/dave2/` | **yes** | the converted model is what the phone downloads, about 5.5 MB |

`*.h5` and `models/data/` are ignored in `.gitignore`. The converted TensorFlow.js model is committed on purpose, so a clone can serve the dashboard without installing TensorFlow.

## Architecture and contract

As stored in the checkpoint:

```
seq_img_in (3, 120, 160, 3)          three consecutive RGB frames, raw 0..255 pixels
└─ td_backbone  TimeDistributed(dave2_backbone)
   ├─ norm      Lambda(t / 255.0)     pixel normalization
   ├─ conv2d_1..5  24, 32, 64, 64, 64 filters, relu, dropout 0.1 between them
   └─ flattened
├─ gru          128 units, reset_after=True, dropout 0.1
├─ fc_100, fc_50  relu
└─ outputs      Dense(2, linear)      [steering, throttle]
```

Input and output as the phone sees them, after `extend_model.py`:

- **Input** `(batch, 3, 120, 160, 3)`, RGB **0..255** floats. The frames of one window are consecutive: the training pipeline slides a window of three frames over the recorded stream, and its `stride` is the step between window starts, not a gap inside a window (`utils/data_stream.py: iter_windows`). At the dashboard's 20 Hz loop the window spans 150 ms.
- **Outputs**, in this order:

| Head | Activation | Range | Source |
|---|---|---|---|
| `steering` | linear | about −1..1, clamped on the phone | trained weights, column 0 |
| `throttle` | linear | about 0..1, clamped on the phone | trained weights, column 1 |
| `brake` | sigmoid | 0 | neutral constant, no training signal |
| `turn_signal` | softmax over 3 | class 0 = off | neutral constant, no training signal |

The checkpoint was trained on Udacity-simulator logs, whose labels are `user/angle` and `user/throttle`. There is no brake or turn-signal label in that data, so those two heads start from constant neutral outputs instead of random weights: a random brake head would slow the car at random. They are ready to be trained in a later phase; the losses are already attached.

## Commands

```bash
# 1. Fetch and verify the checkpoint (or verify the copy you placed by hand)
python models/download_model.py                     # genroads, the default
python models/download_model.py --checkpoint jungle

# 2. Extend it to four heads and check the result numerically
python models/extend_model.py
#    writes models/data/dave2_gru_extended.h5

# 3. Convert to TensorFlow.js with float16 weights
python models/convert.py
#    writes client/static/client/models/dave2/{model.json,group1-shard*.bin}

# 4. Optional: prove the browser model computes what Keras computes
python models/parity/export_reference.py
node models/parity/compare_tfjs.mjs
```

`download_model.py` needs no network when `models/data/<folder>/best_model.h5` already exists; it verifies size and SHA-256 and uses the local file. That is the way to work on a machine that cannot reach Hugging Face: download the file in a browser and copy it to that path.

`convert.py` prints the size of every artifact and fails when the total reaches `MAX_WEB_MODEL_BYTES` (10 MB). It clears the output folder before converting and again after a failure, so a partial or oversized model cannot be committed by accident. float16 is what keeps the model near 5.5 MB; the price is a weight rounding of about `1e-5` on the outputs, which the parity check reports.

## What the extension changes, and why

`extend_model.py` does more than add heads. Two things in the checkpoint's graph cannot run in a browser, and both are removed without changing a single computed value:

1. **The graph is flattened.** `tfjs-layers` has no `Lambda` layer, and it cannot deserialize a nested model inside a `TimeDistributed` wrapper — the checkpoint uses both (`TimeDistributed(dave2_backbone)` with `norm` first in the backbone). Every backbone layer therefore gets its own `TimeDistributed` wrapper.
2. **The normalization is folded into the first convolution.** `W·(s·x) + b == (W·s)·x + b`, so dividing `conv2d_1`'s kernel by 255 removes the Lambda and leaves the input contract unchanged: raw 0..255 pixels, exactly what the network was trained on. `client/static/client/js/inference.js` must therefore *not* divide by 255; dividing twice would darken every frame to a 255th of its value.

The scaling is measured from the checkpoint, not assumed: the normalization layer is fed constant frames and has to behave as a plain multiplication. A checkpoint that normalizes differently is rejected instead of converted.

`steering` and `throttle` keep the trained head's **linear** activation. The phase-4 brief suggested `tanh` and `sigmoid`, but the trained values are already the protocol's ranges, so squashing them again would shrink full lock to `tanh(1) == 0.76` and turn a trained `throttle == 0` into `sigmoid(0) == 0.5` — a car that always accelerates. The phone clamps the two values instead (`toCommand`).

Two numerical checks run before the model is saved, on random frames in the checkpoint's own pixel range:

- the flattened network must reproduce the checkpoint's `[steering, throttle]` (observed drift `~1e-8`, tolerance `1e-5`);
- `steering` and `throttle` must equal the trained outputs, `brake` must be `~0` and `turn_signal` must be class 0 with probability `~1`.

## Loading a checkpoint written by another Python version

Keras stores a `Lambda` as marshalled Python bytecode, and bytecode only unmarshals in the Python version that wrote it. A checkpoint saved under Python 3.10 (TensorFlow 2.9, the training environment) can therefore fail under a newer Python with:

```
ValueError: bad marshal data (unknown type code)
```

`checkpoint.py` handles that. The topology next to the bytecode is plain JSON, so `architecture.py` rebuilds the same layers from it — with a normalization lambda written here instead of the unreadable one — and `Model.load_weights` fills them from the untouched weights in the file. Before doing so it checks that the stored code really contains the constant 255; if it does not, it refuses to guess the pixel range and asks for `--normalization-scale`. Both paths were verified to produce bit-identical outputs.

## Tests

```bash
pytest models            # download pins, checkpoint loading, extension, conversion, parity
npm test                 # phone-side frame window, tensors, dashboard, converted model
```

The tests build a small stand-in with the published layout (normalization lambda, nested `TimeDistributed` backbone, GRU, linear `Dense(2)`) and a full-size stand-in for the end-to-end parity test, which converts it and compares the browser code path against Keras through Node. `models/tests/test_download_model.py` never touches the network.

## Files

| File | Purpose |
|---|---|
| `dave2_config.py` | The single source of truth: pins, contract, head names, paths, budget. |
| `download_model.py` | Fetch from Hugging Face, or verify a hand-copied file. |
| `checkpoint.py` | Load a checkpoint, rebuilding its topology when the stored lambda is unreadable. |
| `architecture.py` | Rebuild layers from the topology JSON inside the file. |
| `extend_model.py` | Flatten the graph, fold the normalization, attach and verify the four heads. |
| `convert.py` | Run `tensorflowjs_converter` with float16 weights and enforce the size budget. |
| `parity/` | Export Keras outputs and compare them with the converted model in TensorFlow.js. |
| `tests/` | The above, on stand-ins that run in seconds. |
