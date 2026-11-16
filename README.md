# EvilFlowers Equation Service

The Celery worker that reads formulas. It has two callers:

- **The catalog.** For [EvilFlowersCatalog](https://github.com/EvilFlowersCatalog/EvilFlowersCatalog) it runs the
  `equations` job of an IP-018 extraction run. It takes each formula region the analyzer's layout pass found in a PDF
  and does three things:
  - crops it to WebP;
  - transcribes it to LaTeX;
  - normalizes the LaTeX with sympy where it parses.

  The results go back to the catalog as regions of kind `equation`.
- **The books digitalizer.** It runs the catalog-free task `standalone.formulas` (#8): regions of leaf images in,
  LaTeX or text in `formulas.json` out, transcribed by PaddleOCR-VL. See [Standalone formulas](#standalone-formulas).

The worker runtime comes from [evilflowers-worker](https://github.com/EvilFlowersCatalog/evilflowers-worker), pinned
to a commit in `pyproject.toml` (`[tool.uv.sources]`). The kit provides the message contract, the catalog client,
the job lifecycle (`LeafJob`), the Celery app, the scale-in shutdown, the delivery cap and the telemetry. Its design
is in its `docs/architecture.md`. This repository holds only the domain: crops, LaTeX recognition and its thread pool.

## Tasks

The task name, queue and argument shape below are a contract with the caller (`tests/test_worker.py` guards it).

| Task | Queue | Sent by | Does |
|---|---|---|---|
| `evilflowers_equation.equations` | `evilflowers_equation` | analyzer, kwargs `run_id, source, output_dir, regions`, optional `job`, `attempt` | For each region `{page, seq, bbox}` (page fractions, top-left origin), on every page: renders a crop at `CROP_DPI` into `<STORAGE_PATH>/<output_dir>/p041-eq02.webp`, transcribes it to LaTeX, and adds sympy's form where it parses and differs. Posts the regions to `/regions`, then reports the job. |
| `evilflowers_equation.standalone.formulas` | `evilflowers_equation.standalone` (bulk), `evilflowers_equation.standalone.interactive` | books digitalizer, the kit's `StandaloneJob` plus `regions` | See [Standalone formulas](#standalone-formulas). Writes `formulas.json` and the outcome file; no catalog call. |

- **`job`** (#5): sent when the analyzer dispatches one task per layout batch (`DISPATCH_PER_BATCH`). The task
  reports under that name (`/jobs/equations:0001-0008`) instead of `equations`. An empty batch (`regions: []`)
  still reports. Metrics keep the kind (`job="equations"`), so their labels do not grow with page count.
- **`attempt`** (EvilFlowersCatalog#140): sent whenever the catalog's `/start` returned one. Every ingestion call
  then carries `?attempt=N`. A 409 is final and never retried, whether its reason is `superseded` or `not_running`.

`evilflowers-analyzer-service` dispatches the job, and the job reports to the catalog's
`/extraction/v1/runs/{run_id}/…` API with `CATALOG_API_KEY`. That key belongs to a catalog user holding
`extraction.ingest_extraction`.

- **The job always reports `done` or `failed`.** A run finishes only when all of its declared jobs have reported
  (`LeafJob.run` in the kit). An unreadable PDF reports `failed`, and so do a catalog error that outlasts the
  retries, an `output_dir` that escapes `STORAGE_PATH`, and a message that does not validate (dropped, never retried).
- **A bad region is skipped, not fatal.** A region that cannot be cropped is logged and left out. If recognition
  fails, the crop is still kept, with `latex` empty.
- **A 409 is the exception.** It means the catalog no longer accepts the run, and the job then stops quietly
  (a `superseded` 409 is still reported as `{"status": "superseded"}`, which the catalog only counts).
- **The run is read first, and again while decoding.** `/regions` is not idempotent, so a copy of a task (a requeue,
  or an analyzer that dispatched again) must not post twice. Before the work the job calls
  `GET /extraction/v1/runs/<id>`, and repeats the call every `RUN_CHECK_INTERVAL` (300 s) before the next crop. It
  stops in three cases:
  - the run moved to a newer `attempt`, whether restarted or reset (`superseded`; reported as
    `{"status": "superseded"}`);
  - the run is not `running` (`not_running`);
  - the run no longer waits for this job (`already_done`).

  If the run cannot be read mid-book, the job logs it and decodes on. It also needs the catalog to post the results.
  `RUN_CHECK_INTERVAL=0` reads the run only once, before the work.
- **Late acknowledgement.** The job acknowledges its message late, so a worker that dies mid-book has the book
  redelivered.

### Stopping and killed workers

Both are the kit's (`evilflowers_worker.shutdown`, `evilflowers_worker.deliveries`, #7); the kit's own tests
prove them against a real Redis, including a kill the pool is denied.

- **SIGTERM requeues the book after a grace.** SIGTERM stays a warm shutdown. A book still running after
  `WORKER_SOFT_SHUTDOWN_TIMEOUT` (60 s) is cancelled, and Redis gets the message back at once instead of after
  `BROKER_VISIBILITY_TIMEOUT`. If that cancel is denied (the `PermissionError(13)` seen on dev), the worker restores
  the message itself after `WORKER_CANCEL_TIMEOUT` (30 s) and exits. Another pod then starts the book from its first
  crop. Keep `WORKER_SOFT_SHUTDOWN_TIMEOUT + WORKER_CANCEL_TIMEOUT` under the pod's `terminationGracePeriodSeconds`
  (3 h), and do not set `REMAP_SIGTERM`.
- **A book that kills the worker is capped (#6).** A killed pool child (OOM) is requeued at once
  (`reject_on_worker_lost`), and a killed container after the visibility timeout. Each redelivery adds one to
  `evilflowers:deliveries:equation:<task_id>` in Redis; a fresh delivery starts at 1 again, because the catalog
  re-sends a run under a stable task id. The delivery after `MAX_DELIVERIES` (3) reports the job `failed`
  (`Gave up after 3 deliveries…`) without decoding, and the run can finish. A scale-in requeue gives its delivery
  back, so it does not count. If Redis is unreachable, the delivery counts as the first; it never fails a book.
  The old keys `equation:deliveries:<task_id>` are no longer read and expire on their own.
- **Crops are bounded (#6).** A box that would render to more than `MAX_CROP_PIXELS` (2 MP) is rendered at a lower
  dpi so that it fits: a whole A4 page at 200 dpi (3.9 MP), or a page of a poster-sized PDF. A wide display formula
  stays well under the bound and keeps `CROP_DPI`. Each crop is logged with its page, box and
  pixel size *before* it is rendered: if one kills the worker, it is the last line of the task.

Chunked posting of a long book's regions (#4, second part) is not implemented. The catalog has no read-back of a
run's regions and `/regions` does not upsert, so a requeued book could not skip what an earlier delivery posted
without storing it twice. Per-batch tasks (#5) are the chunks instead: a requeue repeats one batch of
`LAYOUT_BATCH_PAGES` pages, not a whole book.

## Standalone formulas

`evilflowers_equation.standalone.formulas` (#8) serves the books digitalizer
(evilflowers-books-digitalizer, `docs/design/target-architecture.md` §2–§3, §6). The digitalizer is the system of
record: the task holds no catalog key and never calls the catalog. The diagram is
[docs/diagrams/sequence-standalone-formulas.puml](docs/diagrams/sequence-standalone-formulas.puml).

**Message.** The kit's `StandaloneJob` (`run_id`, `job`, `token`, `inputs`, `output_dir`, `outcome`) plus:

```json
{"regions": [{"key": "region-1734", "leaf": 3, "image": "digitized/…/0003.tif",
              "bbox": [0.08, 0.41, 0.62, 0.47], "task": "formula"}]}
```

- `bbox`: fractions of the leaf image, origin top-left, `x0, y0, x1, y1`.
- `task`: `formula` gives LaTeX. `ocr` gives text with inline math, for a table or a mixed line.
- Keys must be unique; a message with duplicates fails validation and raises with no outcome file (the kit's
  default). `image` must be one of `inputs`; otherwise the job writes a `failed` outcome.
- `bbox` is not range-checked: a box that overshoots the leaf by a rounding error is clamped to it.

**Work.** Each leaf is opened once. Each region is cropped with 10 px of context, the way the vlm service was
validated, and bounded to about 1 MP at once (a job holds its crops until it writes the rows). It is then transcribed by PaddleOCR-VL through the shared vlm service (`VLM_*` below). A `formula` answer goes
through `latex.clean`. An `ocr` answer is only stripped, because `clean` would eat its `$…$`.

**Output.** `formulas.json` in `output_dir` holds one row per region, in message order:

```json
{"key": "region-1734", "text": "E = mc^2", "model": "paddleocr-vl:PaddleOCR-VL-1.6@132c45f"}
```

- `text: null` means "no answer": the model gave none, or the box was empty. The digitalizer keeps such a region as
  an image crop.
- A row may carry `"disputed": true` and a `"reason"` (quality checks below).

The outcome file is written last and atomically: `{run_id, token, status, error, formulas}`, with `status` set to
`done` or `failed`.
- **Any error writes `failed` and raises,** so the orchestrator's `link_error` fires. That covers an unreadable leaf,
  an image that is not an input, an `output_dir` that escapes `STORAGE_PATH`, and a vlm still unavailable after
  `VLM_ATTEMPTS`.
- **Past `MAX_DELIVERIES` it does the same** without working.
- **The task returns `{"job", "answered"}`,** which is what the `link` callback receives.

**Quality checks** (`checks.py`, design §6). A check marks an answer `disputed` and never edits it:

- **Repetition** (always on): the answer ends in a loop (`latex.repeats`, span `LATEX_REPETITION_STOP`).
- **Render** (`FORMULAS_MIN_SIMILARITY`, 0 = off): the LaTeX is rendered with matplotlib's mathtext (the `check`
  extra) and compared with the crop by structural similarity. **The threshold is not calibrated,** so the check is
  off until the digitalizer's review screen has measured it. LaTeX that mathtext cannot render stays unchecked.
- **Cross-check** (`FORMULAS_CROSSCHECK=true`): LaTeX-OCR (`onnx`) reads the formula too. Readings whose sympy forms,
  or canonical LaTeX where sympy cannot parse, agree less than `FORMULAS_CROSSCHECK_MIN_RATIO` (0.8) are disputed.
  This doubles the CPU per crop.

**Deployment.** The standalone task runs as a second Deployment of the same image:
- `REDIS_URL` on db 8;
- `--queues=evilflowers_equation.standalone` for bulk (KEDA, 0..N), or `….standalone.interactive` for the one
  replica that is always on;
- no `CATALOG_API_KEY`;
- `STORAGE_PATH` on the digitalizer's library mount.

## LaTeX

`LATEX_BACKEND` picks the recognizer of the catalog task:

| Backend | What | `latex_model` in provenance |
|---|---|---|
| `onnx` (default) | LaTeX-OCR (pix2tex) on ONNX Runtime, CPU, through `rapid-latex-ocr`. The four weight files are downloaded into `MODELS_PATH` on first use and reused after that (in k8s, the `models` PVC). | `rapid-latex-ocr` |
| `ollama` | A multimodal model behind an Ollama `/api/chat` endpoint (`OLLAMA_ENDPOINT`, `LATEX_MODEL`, `LATEX_TIMEOUT`). | `ollama:<model>` |
| `paddleocr-vl` | PaddleOCR-VL behind the shared vlm service, `POST /v1/formulas` (`VLM_ENDPOINT`). It sends crops in batches of `VLM_BATCH_SIZE` (at most 32), with a timeout of `VLM_TIMEOUT` per crop. A `503` (model loading, queue full) or a lost connection is retried after the service's `Retry-After`, or with exponential backoff, capped at `VLM_MAX_WAIT`, up to `VLM_ATTEMPTS` tries. The service sends no model revision, so `VLM_MODEL_REVISION` names the one in the deployed image. The standalone task always uses this backend. | `paddleocr-vl:<VLM_MODEL_REVISION>` |
| `none` | Crops and positions only. | `none` |

Set `LATEX_THREADS` to the pod's CPU limit (`LATEX_THREADS=4`). rapid-latex-ocr does not set ONNX Runtime's
thread count itself, so by default each of its three sessions starts one thread per core of the *node*. The decoder
makes up to 512 tiny `run()` calls per crop, and 16 threads on a 2–4 core quota spend that quota on synchronisation:
the container is throttled in nearly every CFS period, and a crop takes 30 s instead of about one. The value is
recorded as `latex_threads` in the job's provenance. `LATEX_ALLOW_SPINNING=false` additionally stops idle pool threads
from busy-waiting between calls.

The service runs its own decode loop (`latex.GreedyDecoder`) over rapid-latex-ocr's ONNX sessions (#3):

- **Greedy.** It takes the argmax. Upstream samples at temperature 1e-5, which is the same thing with more work. On
  56 rendered images both gave identical LaTeX, and greedy was ~10 % faster per token.
- **Loops are stopped.** Decoding stops once the last `LATEX_REPETITION_STOP` (64) tokens repeat one unit of 1 to 8
  tokens (`\qquad\qquad…`). What was decoded is kept, so the output is a prefix of the full decode. Before this,
  5.6 % of crops ran to the 512-token cap, and they took 41 % of decode time on dev. `LATEX_MAX_TOKENS` (512) is the
  hard cap. A crop cut short has `provenance.latex_stop` (`repetition` or `max_tokens`).
- **Crops in parallel.** `LATEX_WORKERS` crops of a book are decoded at once, one thread each. The task thread
  renders the crops, because PyMuPDF is not thread-safe. With `LATEX_THREADS=1`, set `LATEX_WORKERS` to the CPU
  limit. Locally, 3 workers × 1 thread did 1.4× the crops/s of 1 worker × 3 threads. That was on a 10-core Mac with
  short formulas, not under the pod's quota.

The decoder graph has no KV cache, so each step re-runs the whole prefix and time grows with the square of the
length. Each crop logs `LaTeX-OCR: N tokens in S s`. There is no span per crop (#5).

The model's LaTeX is what readers see (`latex`). sympy's canonical form goes into the region's `provenance.sympy`,
and only when it differs.

## Formula kind (#9)

Layout marks text lines, list markers, chemical labels and noise as formulas. The catalog task asks decider-vision
whether such a crop is a formula, but only for doubtful ones (`decisions.py`):

1. **Doubt, no model.** A kept formula is *doubtful* when sympy cannot parse its LaTeX, or it is shorter than 3
   tokens. A second backend's disagreeing reading would count too, but the catalog task runs one backend, so this
   check never fires there. A crop with no LaTeX is `unread` and is not judged.
2. **Judge, doubtful only.** One `formula_kind` question goes to decider-vision through the kit's `judges`.
   Categories: `formula`, `text`, `figure`, `table`, `noise`. The crop is sent as PNG, longest side
   `JUDGE_MAX_SIDE` (448). The answer goes on the region as `decision`, `p` and `decider_model`
   (EvilFlowersCatalog#189). The crop's doubts go into `signals.doubts`. `band` is posted only once
   `JUDGE_THRESHOLDS` has `formula_kind` (knowledge#65); then `text` and `noise` are the hiding answers.
3. **Never fatal.** With judges off, decider-vision down, the budget spent or any error, the region is posted as
   before.

Off by default. `DECIDER_ENABLED=true` and `DECIDER_VISION_URL` switch it on. `DECIDER_VISION_REVISION` and
`JUDGE_BUDGET_*` come from the kit. decider-vision is scarce (~17 s a call, one replica). Read
`equation_formulas_total{outcome}` first to learn the doubtful share, then enable it (knowledge#77).

## Layout

```
src/evilflowers_equation/
├── worker.py          Celery app (`celery -A evilflowers_equation.worker worker`): the kit's `create_app("equation")`
├── tasks.py           `equations` (`runtime.run_job`) and `standalone.formulas` (`runtime.run_task`)
├── jobs.py            `EquationsJob(LeafJob[RegionsJob, list[Region]])`: crops, the LaTeX pool, provenance
├── standalone.py      `FormulasJob(StandaloneJob)`: leaf crops, the vlm, `formulas.json`, the outcome file
├── checks.py          quality checks of an answer: repetition, rendered-vs-crop similarity, onnx cross-check
├── decisions.py       doubtful formulas (sympy, tokens, readings) and decider-vision's `formula_kind` (#9)
├── latex.py           recognizers (onnx, ollama, paddleocr-vl, none), cleanup of model output, sympy normalization
├── crops.py           page geometry (PDF pages and leaf images), rendering and WebP crops
├── metrics.py         equation's own instruments (recognition, regions) on the kit's `Instrument`
└── settings.py        `Settings(WorkerSettings, JudgeSettings)`: the queue, crop, LaTeX and judge fields
src/worker.py          deprecated shim for `celery --workdir src -A worker` (see Deployment)
```

## Development

The project uses [uv](https://docs.astral.sh/uv/) and Python 3.12. It has four extras: `onnx` adds LaTeX-OCR,
`check` adds matplotlib for the render check, `otel` adds the OpenTelemetry SDK, and `profiling` adds the Pyroscope
SDK (the last two are the kit's extras). The image installs all four. `tests/test_worker.py` runs the app as a real prefork worker against Redis at
`TEST_REDIS_URL` (default `redis://localhost:6379/15`) and is skipped without one, except in CI.

To move to a newer kit, change `rev` in `[tool.uv.sources]` and run `uv lock --upgrade-package evilflowers-worker`.

```bash
make install        # core + dev tools (ruff, pytest): enough for the tests
make check          # ruff check, ruff format --check, pytest
make install-all    # everything the image has
make worker         # a local worker (REDIS_URL etc. from .env)
docker compose up --build    # Redis + equation; weights in ./models, storage in ./data
```

Bump dependencies with `uv lock --upgrade-package <name>`.

## Configuration

All settings are environment variables. `.env.sample` lists them, and `src/evilflowers_equation/settings.py` has
their defaults and meaning. A deployment sets these:

- `REDIS_URL` (default db 7) and `EQUATION_QUEUE` (default `evilflowers_equation`)
- `STORAGE_PATH`, the catalog's storage root, mounted read/write
- `CATALOG_API_URL` and `CATALOG_API_KEY`
- `LATEX_BACKEND` and `MODELS_PATH` (a writable volume), with either `LATEX_THREADS` (the CPU limit) or
  `LATEX_THREADS=1` plus `LATEX_WORKERS` (the CPU limit); or `OLLAMA_*` and `LATEX_MODEL`
- `IMAGE_TAG`, built in by CI

## Deployment

CI (`.github/workflows/docker-publish.yml`) works like text-service's:

- every push runs ruff and pytest;
- the image is pushed to `ghcr.io/evilflowerscatalog/evilflowers-equation-service` as `:main`, `:develop`, semver
  and `:sha-<short>`;
- a push to `develop` also restarts the dev Deployment `equation` in `evilflowers-dev`, from the in-cluster runner
  `elvira-k3s`, and checks that the running pod uses the digest it just pushed.

The image starts the worker as `celery -A evilflowers_equation.worker worker --concurrency=1 -E
--queues=evilflowers_equation`. That is prefork with one child: the model loads once in the child, and Flower can
inspect the worker during a task.

For now the dev Deployment's `args` still copy the old command, `celery --workdir src -A worker`. `src/worker.py`
keeps that command working. Remove it once the knowledge repo switches the role. That issue also covers the
deploy prerequisites and the telemetry variables:
[EvilFlowersCatalog/knowledge#21](https://github.com/EvilFlowersCatalog/knowledge/issues/21).

## Observability

The service follows the cluster's telemetry contract (knowledge base, `Cookbooks/Deploy-a-Service` §3) through the
kit's `evilflowers_worker.observability`. All of it is off when its variables are unset.

- **Traces.** Every task gets a CONSUMER span that continues the caller's trace. Every catalog or Ollama call gets a
  CLIENT span carrying `traceparent`. Hand-made spans cover the extraction (`equations.extract`) and the model
  download and load (`latex.model.*`). Spans are per task and per phase, never per region (#5): with per-batch tasks
  one book's trace holds hundreds of tasks. The endpoint comes from
  `OTEL_EXPORTER_OTLP_ENDPOINT`, or from `HOST_IP` when only that is set; with neither, telemetry is off. The sampler
  is the SDK's standard `OTEL_TRACES_SAMPLER` / `OTEL_TRACES_SAMPLER_ARG`.
- **Metrics** (Mimir):
  - `equation_task_duration_seconds`
  - `equation_job_outcome_total{job,outcome}` (`job` is always `equations`; `outcome`: `done`, `failed`,
    `not_running`, `superseded`, `already_done`)
  - `equation_recognition_duration_seconds{model,outcome}`
  - `equation_regions_total{outcome}` (`kept`, `skipped`, `failed`)
  - `equation_document_regions`
  - `equation_formulas_total{outcome}` (`doubtful`, `clean`, `unread`; counted with judges on or off)
  - `equation_judge_total{question="formula_kind",outcome,band}` and `equation_judge_seconds` (the kit's judges)
  - `equation_catalog_retries_total`
  - `equation_deliveries_capped_total`
  - process CPU and memory
- **Logs.** Each log line is one JSON object on stdout. It carries `level`, `trace_id`/`span_id`,
  `task_name`/`task_id`, and the task's `run_id`, `job`, `attempt` and `source`. `task_id` is also on the lines of
  the LaTeX worker threads. Set `LOG_FORMAT=text` for a terminal.
- **Profiles** (Pyroscope). The node's eBPF profiler cannot unwind Python on k3s `control`, so the worker profiles
  itself with the Pyroscope SDK when `PYROSCOPE_SERVER_ADDRESS` is set. It reads `PYROSCOPE_APPLICATION_NAME`
  (`equation`) and `PYROSCOPE_TAGS` (`k=v` pairs separated by commas). The profiler runs in the process that runs the
  tasks, with `gil_only=False`, so ONNX Runtime's threads show too. Each trace's root span links to its profile.

The pod needs the label `elvira.stuba.sk/instrumentation: otel-sdk`, so Beyla stops tracing it a second time.

Queries:

```
Tempo:     {resource.service.name="equation" && kind=consumer}
Tempo:     {resource.service.name="equation" && span.evilflowers.extraction.run_id="<run uuid>"}
Loki:      {service_name="equation"} | json | run_id="<run uuid>"
Pyroscope: {service_name="equation", namespace="evilflowers-dev"}
```
