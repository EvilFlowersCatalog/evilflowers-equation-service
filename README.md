# evilflowers-equation-service

Equations of an extraction run (IP-018 Intelligent Viewer, catalog
`docs/proposals/posts/ip-018-intelligent-viewer.md`).

`evilflowers_equation.equations` (queue `evilflowers_equation`) receives the formula regions found by
`evilflowers-analyzer-service`'s Docling layout pass, for every page of the document: `run_id`,
`source`, `output_dir` and `regions: [{page, seq, bbox}]` (page fractions, top-left origin). For
each region it:

1. crops it to WebP at `CROP_DPI` into `<STORAGE_PATH>/<output_dir>/p041-eq02.webp`;
2. transcribes it to LaTeX with `LATEX_BACKEND`: `onnx` (LaTeX-OCR weights on ONNX Runtime, CPU,
   downloaded to `MODELS_PATH` on first use), `ollama` (a multimodal model behind an Ollama endpoint)
   or `none`;
3. normalizes the LaTeX with sympy where it parses; the model's LaTeX is stored in `latex`, sympy's
   form in the region's `provenance.sympy`.

Regions and the `equations` job are reported to the catalog's `/extraction/v1/runs/{run_id}/...`
(`src/catalog.py`). A failing region is skipped; the job fails only when the PDF cannot be opened.

The earlier prototype ran Nougat on one test page (`pages[1:2]`) with region detection left as a
TODO; detection is now the analyzer's and every page is processed.

See `.env.sample` for the settings. CI publishes
`ghcr.io/evilflowerscatalog/evilflowers-equation-service` (`:main` on `main`, semver tags on `v*.*.*`).
