# Local Vazirmatn font

- Upstream: https://github.com/rastikerdar/vazirmatn
- Package: `vazirmatn@33.0.3` from the npm registry
- Original file: `fonts/webfonts/Vazirmatn[wght].woff2`
- Local name: `Vazirmatn.woff2`
- SHA-256: `4e3fa217d38fdafc1fea4414ceb58ca5e662cf0ab5fa735a8c8c20e8b42cad92`
- License: SIL Open Font License 1.1, reproduced in `OFL.txt`

The variable font contains Persian and Latin characters, with weights 100–900.
It is served by Django staticfiles so the phone does not contact a font CDN.
The binary is unmodified; only its filename is simplified.

To update, download the pinned npm tarball with `npm pack vazirmatn@VERSION`,
extract that webfont and the license, then update the version and checksum here.
Do not commit the tarball or the other font variants.
