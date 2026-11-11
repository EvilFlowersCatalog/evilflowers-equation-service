# EvilFlowers Equation Service

The Celery worker that reads the formulas of the catalog's PDFs. For
[EvilFlowersCatalog](https://github.com/EvilFlowersCatalog/EvilFlowersCatalog) it runs the `equations` job of an
IP-018 extraction run. It takes each formula region the analyzer's layout pass found and does three things:

- crops it to WebP;
- transcribes it to LaTeX;
- normalizes the LaTeX with sympy where it parses.

The results go back to the catalog as regions of kind `equation`.

## Tasks

The task name, queue and argument shape below are a contract with the caller (`tests/test_worker.py` guards it).

| Task | Queue | Sent by | Does |
|---|---|---|---|
| `evilflowers_equation.equations` | `evilflowers_equation` | analyzer, kwargs `run_id, source, output_dir, regions` | For each region `{page, seq, bbox}` (page fractions, top-left origin), on every page: renders a crop at `CROP_DPI` into `<STORAGE_PATH>/<output_dir>/p041-eq02.webp`, transcribes it to LaTeX, and adds sympy's form where it parses and differs. Posts the regions to `/regions`, then reports the job. |

`evilflowers-analyzer-service` dispatches the job, and the job reports to the catalog's
`/extraction/v1/runs/{run_id}/…` API with `CATALOG_API_KEY`. That key belongs to a catalog user holding
`extraction.ingest_extraction`.

- **The job always reports `done` or `failed`.** A run finishes only when all of its declared jobs have reported
  (`jobs.EquationsJob.run`). An unreadable PDF reports `failed`, and so does a catalog error that outlasts the
  retries.
- **A bad region is skipped, not fatal.** A region that cannot be cropped is logged and left out. If recognition
  fails, the crop is still kept, with `latex` empty.
- **A 409 is the exception.** It means the catalog no longer accepts the run, and the job then stops quietly.
- **Late acknowledgement.** The job acknowledges its message late, so a worker that dies mid-book has the book
  redelivered.

## LaTeX

`LATEX_BACKEND` picks the recognizer:

| Backend | What | `latex_model` in provenance |
|---|---|---|
| `onnx` (default) | LaTeX-OCR (pix2tex) on ONNX Runtime, CPU, through `rapid-latex-ocr`. The four weight files are downloaded into `MODELS_PATH` on first use and reused after that (in k8s, the `models` PVC). | `rapid-latex-ocr` |
| `ollama` | A multimodal model behind an Ollama `/api/chat` endpoint (`OLLAMA_ENDPOINT`, `LATEX_MODEL`, `LATEX_TIMEOUT`). | `ollama:<model>` |
| `none` | Crops and positions only. | `none` |

Set `LATEX_THREADS` to the pod's CPU limit (`LATEX_THREADS=4`). rapid-latex-ocr does not set ONNX Runtime's
thread count itself, so by default each of its three sessions starts one thread per core of the *node*. The decoder
makes up to 512 tiny `run()` calls per crop, and 16 threads on a 2–4 core quota spend that quota on synchronisation:
the container is throttled in nearly every CFS period, and a crop takes 30 s instead of about one. The value is
recorded as `latex_threads` in the job's provenance. `LATEX_ALLOW_SPINNING=false` additionally stops idle pool threads
from busy-waiting between calls. Each `latex.recognize` span carries `latex.tokens`, the number of tokens decoded (at
most 512), and the same count is logged per crop.

The model's LaTeX is what readers see (`latex`). sympy's canonical form goes into the region's `provenance.sympy`,
and only when it differs.

## Layout

```
src/evilflowers_equation/
├── worker.py          Celery app (`celery -A evilflowers_equation.worker worker`), config and routes
├── tasks.py           the thin Celery adapter of the `equations` job
├── jobs.py            the IP-018 job protocol (extract → publish → report) and the equations job
├── latex.py           recognizers (onnx, ollama, none), cleanup of model output, sympy normalization
├── crops.py           page geometry, rendering and WebP crops
├── schemas.py         messages received and payloads the catalog ingests (mirror of its forms)
├── settings.py        typed settings from the environment (pydantic-settings)
├── clients/           catalog ingestion API, shared HTTP session
└── observability/     OpenTelemetry setup, spans, metrics, JSON logs, Pyroscope profiling, Celery signal wiring
src/worker.py          deprecated shim for `celery --workdir src -A worker` (see Deployment)
```

## Development

The project uses [uv](https://docs.astral.sh/uv/) and Python 3.12. It has three extras: `onnx` adds LaTeX-OCR,
`otel` adds the OpenTelemetry SDK, and `profiling` adds the Pyroscope SDK. The image installs all three.

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
- `LATEX_BACKEND` and `MODELS_PATH` (a writable volume) and `LATEX_THREADS` (the CPU limit), or `OLLAMA_*` and
  `LATEX_MODEL`
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

The service follows the cluster's telemetry contract (knowledge base, `Cookbooks/Deploy-a-Service` §3). All of it
is off when its variables are unset.

- **Traces.** Every task gets a CONSUMER span that continues the caller's trace. Every catalog or Ollama call gets a
  CLIENT span carrying `traceparent`. Hand-made spans cover the extraction (`equations.extract`), the model download
  and load (`latex.model.*`) and each recognition (`latex.recognize`). The endpoint comes from
  `OTEL_EXPORTER_OTLP_ENDPOINT`, or from `HOST_IP` when only that is set; with neither, telemetry is off. The sampler
  is the SDK's standard `OTEL_TRACES_SAMPLER` / `OTEL_TRACES_SAMPLER_ARG`.
- **Metrics** (Mimir):
  - `equation_task_duration_seconds`
  - `equation_job_outcome_total{job,outcome}`
  - `equation_recognition_duration_seconds{model,outcome}`
  - `equation_regions_total{outcome}` (`kept`, `skipped`, `failed`)
  - `equation_document_regions`
  - `equation_catalog_retries_total`
  - process CPU and memory
- **Logs.** Each log line is one JSON object on stdout. It carries `level`, `trace_id`/`span_id`,
  `task_name`/`task_id`, and the task's `run_id`, `job` and `source`. Set `LOG_FORMAT=text` for a terminal.
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
