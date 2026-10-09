# Vendored Three.js

`three.module.min.js` is the ES module build of Three.js r160, copied here so
that the scene works without a CDN. The phone and the PC only need to reach
this server, not the internet, which keeps the closed loop working on a LAN.

| Property | Value |
|---|---|
| Package | `three@0.160.1` (npm) |
| File | `build/three.module.min.js` |
| Revision | 160 |
| Size | 670681 bytes |
| SHA-256 | `3e690ac7d180b0aadf0891bea39eec643e29e2d3e75c99b18689518665f69ba6` |
| License | MIT |

The file is used verbatim. Do not edit it.

## Replacing it

```bash
npm pack three@0.160.1
tar -xzf three-0.160.1.tgz package/build/three.module.min.js
mv package/build/three.module.min.js simulation/static/simulation/js/vendor/
```

Then update the size and the hash in the table above:

```bash
sha256sum simulation/static/simulation/js/vendor/three.module.min.js
```

## Why the module build

`three@0.160.1` still ships the UMD bundles `build/three.js` and
`build/three.min.js`, but both print a deprecation warning on load and both are
gone from `three@0.161.0` onwards. Loading the ES module build keeps the scene
on a supported path, so `scene.js` is loaded with `<script type="module">` and
imports Three.js with a relative path. The same relative import resolves under
Node, which is what makes the tests in `simulation/tests/js/` possible.
