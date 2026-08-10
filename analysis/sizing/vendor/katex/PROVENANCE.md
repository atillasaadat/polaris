# KaTeX 0.16.11 — vendored verbatim

The sizing report (`analysis/sizing/html.py`) typesets its formulae with KaTeX,
client-side, from assets **inlined into the page**. The page must open offline
and fetch nothing at runtime (`tests/analysis/test_sizing_html.py` asserts it),
so the library cannot come from a CDN at view time and is committed here instead,
byte-for-byte as upstream ships it — the repository convention for external
reference data and assets (root `CLAUDE.md`, design doc §3.7).

## What is here, and where it came from

Downloaded from the jsDelivr mirror of the npm package `katex@0.16.11`:

| File | Source URL |
| --- | --- |
| `katex.min.js` | <https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/katex.min.js> |
| `katex.min.css` | <https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/katex.min.css> |
| `fonts/KaTeX_*.woff2` | <https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/fonts/> |

Nothing is pre-processed: the `.js` and `.css` are the published minified
artifacts and the `.woff2` files are the published fonts. The base64 `@font-face`
rewrite that makes the page self-contained is computed **from these files at
render time** (`analysis/sizing/texmath.py`), never committed in place of them.

## Which fonts, and why only eight

Upstream ships 60 font files (WOFF2/WOFF/TTF across ten families and their
weights). Only the WOFF2 files are kept, and only the eight families the report's
mathematics actually reaches — every glyph the formulae, the inertia `bmatrix`
and the nomenclature table set. They are inlined into every generated page, so an
unused family is weight on every artifact:

`Main-Regular`, `Main-Bold`, `Main-Italic`, `Math-Italic`, `Size1-Regular`,
`Size2-Regular`, `AMS-Regular`, `Caligraphic-Regular`.

`tests/analysis/test_sizing_html.py` asserts the rendered page carries an inlined
`@font-face` for each of the eight and **no remaining relative `url(...)`** — the
inliner drops the `@font-face` rules for families that are not vendored rather
than leaving a reference the browser would try to fetch. A formula reaching a
ninth family therefore renders in the browser's fallback face, visibly wrong
rather than invisibly broken. Adding one means downloading it from the URL above
at the same version and extending the list here.

## Upgrading

Re-download every file above at the new version, update the version in this
file's title and URLs, re-run `uv run --group analysis pytest tests/analysis/`,
and open a generated page to confirm the typesetting is unchanged.

## License

KaTeX is MIT-licensed. Reproduced verbatim from
<https://github.com/KaTeX/KaTeX/blob/v0.16.11/LICENSE>:

```
The MIT License (MIT)

Copyright (c) 2013-2020 Khan Academy and other contributors

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```
