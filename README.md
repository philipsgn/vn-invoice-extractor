# VN-Invoice-Extractor

> Enterprise-grade, high-resilience Vietnamese invoice ingestion platform combining multimodal OCR, cloud LLM extraction, local deep learning fallback, and server-rendered analytics for single-file and batch financial document processing.

![Python](https://img.shields.io/badge/Python-3.x-3776AB?logo=python&logoColor=white)
![Flask](https://img.shields.io/badge/Flask-Web%20API-000000?logo=flask&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.1.0-EE4C2C?logo=pytorch&logoColor=white)
![Gemini](https://img.shields.io/badge/Gemini-2.5%20Flash-4285F4?logo=google&logoColor=white)
![License](https://img.shields.io/badge/License-Not%20Specified-lightgrey)

## 1. Project Title & Hero

`VN-Invoice-Extractor` is a production-oriented document AI system built to parse highly variable Vietnamese invoices across VAT invoices, non-VAT invoices, and retail-style layouts. The repository combines OCR, layout-aware sequence labeling, deterministic business-rule validation, and interactive web delivery into a single ingestion platform.

What makes this project portfolio-worthy is not just the OCR accuracy story, but the systems engineering around it:

- It supports synchronous API extraction, asynchronous queue-backed extraction, multi-file extraction, and ZIP-driven dashboard analytics.
- It orchestrates a cloud-first extraction path with a local fallback path, preserving service continuity when Gemini is unavailable or rate-limited.
- It isolates long-running AI work from request-response rendering paths using background workers, thread-safe shared state, and pollable job contracts.

## 2. Architectural Overview & Problem Solved

### Why this project exists

Vietnamese invoices are structurally noisy. In the same batch, the system may encounter:

- VAT invoices and sales invoices with different header semantics.
- Dense line-item tables with inconsistent row boundaries and OCR drift.
- Vietnamese diacritics, tax codes, mixed date formats, and fragmented currency tokens.
- Image uploads and multi-page PDF uploads requiring very different preprocessing paths.

Traditional rule-only extraction is too brittle for this domain. Generic OCR-only extraction is also insufficient because invoice understanding depends on both text and spatial layout.

### Core solution

This repository implements a hybrid document AI pipeline:

1. **Cloud-first extraction path**
   - `inference/web_app/gemini_extractor.py` uses Gemini Flash (`gemini-2.5-flash`) as the primary multimodal extractor.
   - The prompt enforces strict JSON output, invoice typing, normalized dates, normalized financial totals, and line-item extraction.

2. **Local fallback path**
   - If Gemini is disabled, unavailable, or fails at runtime, the system falls back to a local path:
     - `PaddleOCR` for detection
     - `VietOCR` for text recognition
     - 3 specialized `LayoutLMv3` models for `header`, `table`, and `footer`
     - BIO repair and invoice reconstruction logic
     - validation and serialization layers

3. **Validation-first formatting**
   - Stage 5 attaches warnings, confidence summaries, formatted output, and business-rule validation so that downstream UI rendering can stay stable even when extraction quality varies.

### Asynchronous architecture

The repository uses two distinct non-blocking execution modes:

- **Async API queue**
  - `AsyncJobQueue` stores jobs in a `deque`, coordinates worker wakeups via `threading.Condition`, and executes extraction in daemon worker threads.
  - Jobs are pollable via `GET /api/v1/jobs/<job_id>`.

- **ZIP dashboard processing**
  - `POST /upload-zip` immediately creates a job entry in `JobManager`, then launches a background thread to process invoices in the ZIP.
  - The frontend polls progress while the main Flask thread remains free to serve status responses and UI traffic.

This is important from an engineering perspective: the system does not block the request thread on heavy OCR/model work, and it avoids front-end polling failures by making job state visible as soon as the request is accepted.

## 3. Key Engineering Features

### 🚦 Asynchronous non-blocking execution pool

The project implements a custom thread-based execution model rather than relying on `ThreadPoolExecutor`.

- `AsyncJobQueue` maintains in-memory job state and worker coordination using `deque`, `Lock`, and `Condition`.
- Polling endpoints return structured job state instead of tying clients to long-lived open requests.
- `JobManager` tracks ZIP workflow progress independently from the async API queue, enabling the same polling endpoint to serve both job types.
- Queue TTL eviction prevents unbounded in-memory growth for completed or stale jobs.

This is a strong systems design choice for CPU-bound, model-heavy inference under a Flask application where request isolation matters.

### 🧠 Multimodal AI fallback strategy

The extraction engine is intentionally hybrid.

- **Primary path:** Gemini Flash performs zero-shot multimodal extraction into strict JSON.
- **Fallback path:** OCR + LayoutLMv3 + post-processing reconstruct the same internal invoice schema locally.
- **Safety rails:** the pipeline validates subtotal/VAT/total relationships, normalizes invoice types, and preserves confidence metadata for low-certainty fields.
- **Operational resilience:** if Gemini fails, the service degrades gracefully instead of going offline.

### 💱 Resilient Jinja2 custom formatting engine

The dashboard renderer is hardened against inconsistent extraction output.

- `safe_currency` is registered as a custom template filter to safely render `None`, empty strings, already-formatted strings, or numeric values using Vietnamese-style thousands separators.
- Dashboard normalization helpers coerce missing totals, names, tax codes, line-item collections, and chart payloads into safe defaults before rendering.
- This eliminates brittle server-side template failures in the presence of partial or imperfect extraction results.

### 📦 Batch ZIP ingestion & live analytics

The UI supports three document ingestion modes:

- Single image/PDF extraction
- Multi-file extraction (`/extract-multi`)
- ZIP batch extraction with dashboard render (`/upload-zip` → `/dashboard/<job_id>`)

The ZIP workflow adds:

- real-time progress polling
- current-file tracking
- aggregate totals and VAT summaries
- invoice-type distribution charts
- timeline-style spending visualization
- per-invoice review in a modal editing experience

## 4. System Architecture & Data Flow

```text
Client Upload
   |
   +--> Single file UI/API
   |      |
   |      +--> /api/v1/extract
   |      |      |
   |      |      +--> Stage 0: Validate file extension, size, decodability
   |      |      +--> Stage 1: Normalize image / render PDF pages
   |      |      +--> Stage 3+4 Primary: Gemini Flash extraction
   |      |      |        |
   |      |      |        +--> If Gemini fails:
   |      |      |              Stage 2: PaddleOCR detect + VietOCR recognize
   |      |      |              Stage 3: LayoutLMv3 inference (header/table/footer)
   |      |      |              Stage 4: BIO repair + invoice reconstruction
   |      |      +--> Stage 5: Serialize, validate, enrich, format
   |      |      +--> JSON response
   |      |
   |      +--> /api/v1/extract/async
   |             |
   |             +--> Create Job -> enqueue in AsyncJobQueue
   |             +--> Worker thread executes pipeline
   |             +--> Frontend polls /api/v1/jobs/<job_id>
   |
   +--> ZIP batch upload
          |
          +--> /upload-zip
                 |
                 +--> JobManager state inserted immediately
                 +--> Background daemon thread processes ZIP entries
                 +--> Poll /api/v1/jobs/<job_id>
                 +--> Aggregate summary + normalize dashboard payload
                 +--> Render /dashboard/<job_id>
```

### Stage lifecycle

| Stage | Responsibility |
|---|---|
| Stage 0 | Input validation for extension, file size, and decode viability |
| Stage 1 | Image/PDF preprocessing and normalization |
| Stage 2 | OCR fallback path: PaddleOCR detection + VietOCR recognition |
| Stage 3 | LayoutLMv3 inference for header, table, and footer token labeling |
| Stage 4 | BIO repair, table reconstruction, conflict resolution, field assembly |
| Stage 5 | Validation, confidence scoring, serialization, warnings, render-safe formatting |

## 5. Tech Stack & Dependencies

| Layer | Technologies actually used in this repository |
|---|---|
| Core Backend | Python, Flask, Waitress (optional production server path), `threading`, `deque`, `Condition`, `Lock` |
| AI / ML | Gemini Flash (`gemini-2.5-flash`), PyTorch 2.1.0, Transformers, VietOCR, PaddleOCR, LayoutLMv3 |
| OCR / Image Processing | OpenCV, Pillow, PyMuPDF, pdf2image |
| Frontend / Presentation | Jinja2, TailwindCSS, Vanilla JavaScript polling, Chart.js |
| Data & Serialization | JSON, `pathlib`, ZIP processing, custom schema serialization, confidence wrappers |
| Observability | Structured JSON logging, in-memory metrics, health/readiness endpoints |

### Dependency files present in the repo

- Root OCR/ML dependencies: `requirements.txt`
- Web-app-specific dependencies: `inference/web_app/requirements.txt`

### Model assets expected by the code

- VietOCR weights:
  - `models/vietocr/vgg_transformer.pth`
- LayoutLMv3 model families:
  - `model_final_layoutlmv3/model_final_header/final_model`
  - `model_final_layoutlmv3/model_final_table/final_model`
  - `model_final_layoutlmv3/model_final_footer/final_model`

## 6. Installation & Local Deployment

### Option A: `venv`

```bash
python -m venv .venv
.venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
pip install -r inference/web_app/requirements.txt
pip install waitress
```

### Option B: `conda`

```bash
conda create -n vn-invoice-extractor python=3.10 -y
conda activate vn-invoice-extractor
pip install --upgrade pip
pip install -r requirements.txt
pip install -r inference/web_app/requirements.txt
pip install waitress
```

### Important SDK note for Gemini

`inference/web_app/gemini_extractor.py` imports:

```python
from google import genai
from google.genai import types
```

If your environment raises an import error after installing `inference/web_app/requirements.txt`, install the compatible Google GenAI SDK explicitly:

```bash
pip install google-genai
```

### Environment variables

The Flask app loads environment variables from:

```text
inference/web_app/.env
```

Recommended `.env` contents:

```env
GEMINI_API_KEY=your_gemini_api_key
DEVICE=cpu
PORT=5000
DEBUG=true
FLASK_DEBUG=false
RATE_LIMIT_RPM=20
RATE_LIMIT_ENABLED=true
```

### Start the web application

```bash
cd inference/web_app
python app.py
```

The application boot flow will:

- initialize upload/temp/log/result directories
- warm the model manager in a background thread
- expose the UI at `http://localhost:5000/`
- expose Swagger at `http://localhost:5000/docs`

### Production path

The app attempts to start with Waitress when not running in debug mode:

```bash
set FLASK_DEBUG=false
cd inference/web_app
python app.py
```

## 7. API Endpoints Specification

### Important route clarification

There is **no** `POST /api/v1/upload` route in the current repository. The actual upload/extraction routes are:

- `POST /api/v1/extract`
- `POST /api/v1/extract/async`
- `POST /upload-zip`

### Core endpoints

| Method | Route | Purpose |
|---|---|---|
| `POST` | `/api/v1/extract` | Synchronous single-invoice extraction |
| `POST` | `/api/v1/extract/async` | Async single-invoice extraction with pollable `job_id` |
| `GET` | `/api/v1/jobs/<job_id>` | Poll async API jobs and ZIP dashboard jobs |
| `GET` | `/api/v1/jobs/status/<job_id>` | Alternate ZIP progress/status endpoint |
| `POST` | `/upload-zip` | Submit ZIP archive for dashboard batch processing |
| `GET` | `/dashboard/<job_id>` | Render the final server-side dashboard for a completed ZIP batch |
| `POST` | `/extract-multi` | Extract multiple images/PDFs and render `results_multi.html` |
| `POST` | `/api/v1/update-result` | Persist edited extraction results back to `extracted_json/` |
| `GET` | `/api/v1/health` | Liveness and Gemini availability |
| `GET` | `/api/v1/ready` | Model readiness and loaded model names |
| `GET` | `/api/v1/metrics` | In-memory service metrics snapshot |
| `GET` | `/docs` | Swagger UI |

### `POST /api/v1/extract/async`

Accepts `multipart/form-data` with a single `file` field.

Example response:

```json
{
  "success": true,
  "request_id": "a1b2c3d4",
  "job_id": "8e7f0e6b9f0d4c26a9f97dcbdd1b8d73",
  "poll_url": "/api/v1/jobs/8e7f0e6b9f0d4c26a9f97dcbdd1b8d73",
  "status": "pending"
}
```

### `GET /api/v1/jobs/<job_id>`

This endpoint serves two job models:

- async API jobs from `AsyncJobQueue`
- ZIP dashboard jobs from `JobManager`

Observed states in the codebase include:

- Async API queue: `pending`, `processing`, `done`, `failed`
- ZIP dashboard jobs: `processing`, `completed`, `failed`

### `GET /dashboard/<job_id>`

Server-side renders `templates/dashboard.html` using:

- normalized `details`
- aggregated `summary`
- safe defaults for missing values
- custom Jinja formatting (`safe_currency`)

This is the final human-review surface for ZIP batch processing.

## 8. Production Lessons Learned & Optimizations

### Race condition resolution: job creation vs frontend polling

One of the more important operational fixes in this repository is the elimination of early polling failures.

What changed:

- `POST /upload-zip` inserts the job into `JobManager` **before** background processing begins.
- The frontend waits briefly before starting its polling loop, giving the server a stable initialization window.
- `GET /api/v1/jobs/<job_id>` now checks both the async API queue and the ZIP `JobManager`, so a batch dashboard job is still resolvable through the common polling contract.

Impact:

- fewer transient `404` responses during startup
- more stable UX for high-frequency polling
- cleaner separation between “job accepted” and “job finished”

### Memory management and large-model runtime control

This project makes several pragmatic choices to keep PyTorch-heavy inference manageable:

- `ModelManager` lazy-loads heavyweight OCR and LayoutLMv3 components instead of loading everything at import time.
- model warmup runs in a background thread, reducing first-request latency shock.
- `torch.set_num_threads(2)` and `torch.set_num_interop_threads(2)` cap thread pressure on CPU.
- `inference_lock` serializes the local fallback inference section, which is a practical guardrail against CPU thrashing and stability issues when multiple heavy jobs arrive close together.
- temporary directories are removed in `finally` blocks and `gc.collect()` is called after pipeline completion.
- queue size limits and job TTL settings bound long-lived memory growth in the in-memory job store.

### Additional engineering notes worth highlighting in a portfolio

- The OCR path intentionally discards Paddle text recognition and uses VietOCR as the canonical recognizer, improving text consistency.
- The local extraction path uses three specialized LayoutLMv3 models rather than one monolithic classifier, reflecting a deliberate decomposition of header/table/footer semantics.
- The dashboard layer is not a superficial frontend add-on; it is a validation interface for batch finance ingestion with summary metrics and visual review.

## Repository Layout

```text
invoice_ocr/
├── inference/
│   ├── web_app/
│   │   ├── app.py
│   │   ├── gemini_extractor.py
│   │   ├── templates/
│   │   ├── static/
│   │   ├── extracted_json/
│   │   └── .env
│   └── bio_repair_inference.py
├── model_final_layoutlmv3/
│   ├── model_final_header/final_model
│   ├── model_final_table/final_model
│   └── model_final_footer/final_model
├── models/
│   └── vietocr/vgg_transformer.pth
├── src/
│   ├── run_ocr.py
│   ├── engine/
│   ├── normalizers/
│   ├── validators/
│   └── invoice_schema.py
└── requirements.txt
```

## Why this repository stands out

This is not a toy OCR demo. It is a serious ingestion system that blends:

- multimodal LLM extraction
- deterministic OCR fallback
- model lifecycle management
- concurrency-safe job orchestration
- dashboard-grade presentation
- validation-aware document engineering

For a portfolio or CV, the value of `VN-Invoice-Extractor` is that it demonstrates end-to-end ownership across backend systems, applied AI, operational reliability, and product-facing data presentation in one coherent codebase.

