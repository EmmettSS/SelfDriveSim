# Vendored TensorFlow.js

`tf.min.js` is the UMD build of TensorFlow.js 4.22.0, copied here so that the
phone runs the DAVE-2-GRU model without a CDN. The phone only needs to reach
this server, which keeps the closed loop working on a LAN.

| Property | Value |
|---|---|
| Package | `@tensorflow/tfjs@4.22.0` (npm) |
| File | `dist/tf.min.js` |
| Version | 4.22.0 |
| Size | 1469843 bytes |
| SHA-256 | `300dfae273d20b4046f46a06d735688f03675a807561e9bcb5f664eb2f3d2831` |
| License | Apache-2.0 (the license header is kept inside the file) |

The file is used verbatim. Do not edit it.

The version matches the `tensorflowjs` converter in `requirements.txt`
(`tensorflowjs` 4.22.0). Keep the two in step: a model converted by one version
can fail to load in another.

## Why the UMD build

`dist/tf.min.js` is loaded with a classic `<script>` before `inference.js`. It
sets `window.tf`, and `inference.js` reads that global. The same package is
also a devDependency (`package.json`), so the Node tests in
`client/tests/js/` import the same version.

## Replacing it

```bash
npm pack @tensorflow/tfjs@4.22.0
tar -xzf tensorflow-tfjs-4.22.0.tgz package/dist/tf.min.js
mv package/dist/tf.min.js client/static/client/js/vendor/
```

Then update the size and the hash in the table above:

```bash
sha256sum client/static/client/js/vendor/tf.min.js
```
