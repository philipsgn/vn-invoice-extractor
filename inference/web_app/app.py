# -*- coding: utf-8 -*-
"""
╔══════════════════════════════════════════════════════════════════╗
║           INVOICE EXTRACTION — PHASE 7 & 8 PRODUCTION           ║
║                    Flask API Service                             ║
╠══════════════════════════════════════════════════════════════════╣
║  Phase 7 — Inference Pipeline                                    ║
║    Stage 0  Input validation                                     ║
║    Stage 1  Image preprocessing (normalize → synthetic standard) ║
║    Stage 2  OCR  (PaddleOCR detect + VietOCR recognize)          ║
║    Stage 3  LayoutLMv3 model inference                           ║
║    Stage 4  BIO repair + InvoiceExtractionEngine                 ║
║    Stage 5  Output formatting + field validators                 ║
║                                                                  ║
║  Phase 8 — Deployment                                            ║
║    • Flask REST API                                              ║
║    • Sync  : POST /api/v1/extract                                ║
║    • Async : POST /api/v1/extract/async  →  GET /api/v1/jobs/:id ║
║    • Rate limiting (token-bucket per IP)                         ║
║    • Structured JSON logging                                     ║
║    • In-memory metrics (/api/v1/metrics)                         ║
║    • Swagger UI (/docs)                                          ║
║    • Health + readiness checks                                   ║
╚══════════════════════════════════════════════════════════════════╝

Run:
    python app.py                                      # development
    gunicorn -w 1 -b 0.0.0.0:5000 app:app             # production

Endpoints:
    POST   /api/v1/extract            Synchronous extraction
    POST   /api/v1/extract/async      Submit async job → returns job_id
    GET    /api/v1/jobs/<job_id>      Poll async job result
    GET    /api/v1/health             Liveness check
    GET    /api/v1/ready              Readiness check (model loaded?)
    GET    /api/v1/metrics            Runtime metrics
    GET    /docs                      Swagger UI
    GET    /                          Web UI (upload form)
"""

# ═══════════════════════════════════════════════════════════════════
# 0.  STDLIB  IMPORTS
# ═══════════════════════════════════════════════════════════════════
import copy
import gc
import json
import logging
import os
# ── Load .env FIRST — before any os.environ.get() calls ────────
from pathlib import Path as _Path
_ENV_FILE = _Path(__file__).parent / ".env"
try:
    from dotenv import load_dotenv as _load_dotenv
    _loaded = _load_dotenv(dotenv_path=_ENV_FILE, override=False)
except ImportError:
    _loaded = False
# ──────────────────────────────────────────────────────────────
import re
import shutil
import sys
import threading
import time
import traceback
import uuid
import zipfile
from collections import defaultdict, deque
from datetime import datetime, timezone
from functools import wraps
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ═══════════════════════════════════════════════════════════════════
# 1.  THIRD-PARTY  IMPORTS
# ═══════════════════════════════════════════════════════════════════
import cv2
import numpy as np
from PIL import Image

import torch
from transformers import LayoutLMv3ForTokenClassification, LayoutLMv3Processor

from flask import Flask, g, jsonify, render_template, request

# ═══════════════════════════════════════════════════════════════════
# 2.  PROJECT  ROOT  →  sys.path
# ═══════════════════════════════════════════════════════════════════
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.run_ocr import OCRRunner
from src.engine import InvoiceExtractionEngine
from src.utils.field_helpers import (
    safe_unwrap, 
    make_field, 
    make_field_found, 
    make_field_not_found,
    CONFIDENCE_NOT_FOUND,
    CONFIDENCE_NOT_COMPUTED
)
from src.invoice_schema import (
    InvoiceSchema, 
    serialize_invoice_result,
    compute_document_confidence,
)
from src.validators.invoice_validator import InvoiceValidator
from inference.bio_repair_inference import (
    prepare_words_and_bboxes,
    repair_bio_sequence,
    strip_nested_prefixes,
    clean_company_fields,
)
from inference.image_preprocessor import normalize_invoice_image

# ── Gemini Flash extractor (primary engine) ───────────────────
try:
    from gemini_extractor import build_extractor as _build_gemini
    _GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "").strip()
    if _GEMINI_API_KEY and len(_GEMINI_API_KEY) > 10:
        GEMINI_EXTRACTOR = _build_gemini(_GEMINI_API_KEY)
        GEMINI_ENABLED   = GEMINI_EXTRACTOR is not None
        print(f"[Gemini] Key loaded: {_GEMINI_API_KEY[:8]}...{_GEMINI_API_KEY[-4:]}")
    else:
        GEMINI_EXTRACTOR = None
        GEMINI_ENABLED   = False
        print("[Gemini] WARNING: No API key found in .env")
except ImportError as _e:
    GEMINI_EXTRACTOR = None
    GEMINI_ENABLED   = False
    print(f"[Gemini] gemini_extractor not found: {_e}")
# ──────────────────────────────────────────────────────────────


# ═══════════════════════════════════════════════════════════════════
# 3.  CONFIGURATION
# ═══════════════════════════════════════════════════════════════════
class Config:
    """
    Central configuration.
    All values can be overridden via environment variables at runtime.
    """

    BASE_DIR:    Path = Path(__file__).resolve().parent
    DEFAULT_MODEL_DIR: Path = PROJECT_ROOT / "model_final_layoutlmv3"

    MODEL_PATH_HEADER: str = os.getenv(
        "MODEL_PATH_HEADER",
        str(DEFAULT_MODEL_DIR / "model_final_header" / "final_model"),
    )
    MODEL_PATH_TABLE: str = os.getenv(
        "MODEL_PATH_TABLE",
        str(DEFAULT_MODEL_DIR / "model_final_table" / "final_model"),
    )
    MODEL_PATH_FOOTER: str = os.getenv(
        "MODEL_PATH_FOOTER",
        str(DEFAULT_MODEL_DIR / "model_final_footer" / "final_model"),
    )
    DEVICE: str = os.getenv("DEVICE", "cpu")

    HOST:  str  = os.getenv("HOST",  "0.0.0.0")
    PORT:  int  = int(os.getenv("PORT", 5000))
    DEBUG: bool = os.getenv("DEBUG", "true").lower() == "true"

    UPLOAD_DIR:  Path = BASE_DIR / "uploads"
    TEMP_DIR:    Path = BASE_DIR / "tmp_normalized"
    LOG_DIR:     Path = BASE_DIR / "logs"
    RESULTS_DIR: Path = BASE_DIR / "results"

    MAX_UPLOAD_BYTES:   int = 50 * 1024 * 1024
    ALLOWED_EXTENSIONS: set = {".png", ".jpg", ".jpeg", ".pdf"}

    RATE_LIMIT_RPM:     int  = int(os.getenv("RATE_LIMIT_RPM", 20))
    RATE_LIMIT_ENABLED: bool = os.getenv("RATE_LIMIT_ENABLED", "true").lower() == "true"

    JOB_TTL_SECONDS:   int = 3600
    JOB_QUEUE_MAXSIZE: int = 50

    PDF_DPI:        int   = 200
    VAT_TOLERANCE:  float = 0.05
    MIN_CONFIDENCE: float = 0.50
    OCR_LANG:       str   = "vi"
    VIETOCR_MODEL:  str   = "vgg_transformer"

    @classmethod
    def init_dirs(cls):
        for d in (cls.UPLOAD_DIR, cls.TEMP_DIR, cls.LOG_DIR, cls.RESULTS_DIR):
            d.mkdir(parents=True, exist_ok=True)


# ═══════════════════════════════════════════════════════════════════
# 4.  STRUCTURED  LOGGING
# ═══════════════════════════════════════════════════════════════════
class JSONFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: Dict[str, Any] = {
            "ts":     datetime.now(timezone.utc).isoformat(),
            "level":  record.levelname,
            "logger": record.name,
            "msg":    record.getMessage(),
            "module": record.module,
            "line":   record.lineno,
        }
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        reserved = set(logging.LogRecord.__dict__) | set(payload)
        for k, v in record.__dict__.items():
            if k not in reserved:
                payload[k] = v
        return json.dumps(payload, ensure_ascii=False, default=str)


def setup_logging(log_dir: Path) -> logging.Logger:
    log = logging.getLogger("invoice_api")
    log.setLevel(logging.DEBUG)

    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter(
        fmt="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    ))

    fh = logging.FileHandler(log_dir / "api.jsonl", encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(JSONFormatter())

    log.addHandler(ch)
    log.addHandler(fh)
    return log


# ═══════════════════════════════════════════════════════════════════
# 5.  METRICS  COLLECTOR
# ═══════════════════════════════════════════════════════════════════
class Metrics:
    def __init__(self):
        self._lock               = threading.Lock()
        self.started_at          = datetime.now(timezone.utc).isoformat()
        self.total_requests:  int = 0
        self.total_successes: int = 0
        self.total_errors:    int = 0
        self.total_pdfs:      int = 0
        self.total_images:    int = 0
        self.stage_times: Dict[str, List[float]] = defaultdict(list)
        self.recent_latencies: deque = deque(maxlen=100)
        self.error_counts: Dict[str, int] = defaultdict(int)

    def record_request(self, success: bool, latency_ms: float, file_type: str):
        with self._lock:
            self.total_requests += 1
            if success:
                self.total_successes += 1
            else:
                self.total_errors += 1
            if file_type == "pdf":
                self.total_pdfs += 1
            else:
                self.total_images += 1
            self.recent_latencies.append(latency_ms)

    def record_stage(self, stage: str, elapsed_ms: float):
        with self._lock:
            self.stage_times[stage].append(elapsed_ms)

    def record_error(self, error_type: str):
        with self._lock:
            self.error_counts[error_type] += 1

    def snapshot(self) -> Dict:
        def _stats(values: List[float]) -> Dict:
            if not values:
                return {"count": 0, "avg_ms": 0, "p50_ms": 0, "p95_ms": 0}
            arr = sorted(values)
            n   = len(arr)
            return {
                "count":  n,
                "avg_ms": round(sum(arr) / n, 1),
                "p50_ms": round(arr[n // 2], 1),
                "p95_ms": round(arr[min(int(n * 0.95), n - 1)], 1),
            }

        with self._lock:
            return {
                "uptime_since":    self.started_at,
                "total_requests":  self.total_requests,
                "total_successes": self.total_successes,
                "total_errors":    self.total_errors,
                "total_pdfs":      self.total_pdfs,
                "total_images":    self.total_images,
                "success_rate": (
                    round(self.total_successes / self.total_requests, 4)
                    if self.total_requests else 0
                ),
                "latency":       _stats(list(self.recent_latencies)),
                "stage_latency": {
                    stage: _stats(times)
                    for stage, times in self.stage_times.items()
                },
                "error_breakdown": dict(self.error_counts),
            }


# ═══════════════════════════════════════════════════════════════════
# 6.  RATE  LIMITER  —  token-bucket per IP
# ═══════════════════════════════════════════════════════════════════
class RateLimiter:
    def __init__(self, rpm: int):
        self.capacity  = rpm
        self.refill_ps = rpm / 60.0
        self._buckets: Dict[str, Tuple[float, float]] = {}
        self._lock     = threading.Lock()

    def is_allowed(self, ip: str) -> bool:
        now = time.monotonic()
        with self._lock:
            tokens, last = self._buckets.get(ip, (float(self.capacity), now))
            elapsed      = now - last
            tokens       = min(self.capacity, tokens + elapsed * self.refill_ps)
            if tokens >= 1.0:
                self._buckets[ip] = (tokens - 1.0, now)
                return True
            self._buckets[ip] = (tokens, now)
            return False


# ═══════════════════════════════════════════════════════════════════
# 7.  MODEL  MANAGER  —  singleton, thread-safe lazy load
# ═══════════════════════════════════════════════════════════════════
class ModelManager:
    _instance: Optional["ModelManager"] = None
    _class_lock = threading.Lock()

    def __init__(self):
        self._ready:       bool = False
        self._init_error:  Optional[str] = None
        self._init_lock    = threading.Lock()

        self.ocr_runner: Optional[OCRRunner] = None
        self.engine:     Optional[InvoiceExtractionEngine] = None

        # --- Lazy Load Private Storage ---
        self._header_processor: Optional[LayoutLMv3Processor]              = None
        self._header_model:     Optional[LayoutLMv3ForTokenClassification] = None
        self.header_label2id:   Dict[str, int] = {}
        self.header_id2label:   Dict[int, str] = {}

        self._table_processor:  Optional[LayoutLMv3Processor]              = None
        self._table_model:      Optional[LayoutLMv3ForTokenClassification] = None
        self.table_label2id:    Dict[str, int] = {}
        self.table_id2label:    Dict[int, str] = {}

        self._footer_processor: Optional[LayoutLMv3Processor]              = None
        self._footer_model:     Optional[LayoutLMv3ForTokenClassification] = None
        self.footer_label2id:   Dict[str, int] = {}
        self.footer_id2label:   Dict[int, str] = {}

        self.device: str = Config.DEVICE
        self._lazy_lock = threading.Lock()

    @classmethod
    def get(cls) -> "ModelManager":
        if cls._instance is None:
            with cls._class_lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    # ── Header Model Property ─────────────────────────────────────────
    @property
    def header_processor(self) -> LayoutLMv3Processor:
        if self._header_processor is None: self._ensure_header()
        return self._header_processor

    @property
    def header_model(self) -> LayoutLMv3ForTokenClassification:
        if self._header_model is None: self._ensure_header()
        return self._header_model

    def _ensure_header(self):
        with self._lazy_lock:
            if self._header_model is not None: return
            p = Path(Config.MODEL_PATH_HEADER)
            logger.info("⚡ Lazy-loading Header model from %s", p)
            self._header_processor = LayoutLMv3Processor.from_pretrained(str(p), apply_ocr=False)
            self._header_model = LayoutLMv3ForTokenClassification.from_pretrained(str(p))
            self._header_model.to(self.device)
            self._header_model.eval()
            with open(p / "label2id.json", encoding="utf-8") as f:
                self.header_label2id = json.load(f)
            with open(p / "id2label.json", encoding="utf-8") as f:
                self.header_id2label = {int(k): v for k, v in json.load(f).items()}

    # ── Table Model Property ──────────────────────────────────────────
    @property
    def table_processor(self) -> LayoutLMv3Processor:
        if self._table_processor is None: self._ensure_table()
        return self._table_processor

    @property
    def table_model(self) -> LayoutLMv3ForTokenClassification:
        if self._table_model is None: self._ensure_table()
        return self._table_model

    def _ensure_table(self):
        with self._lazy_lock:
            if self._table_model is not None: return
            p = Path(Config.MODEL_PATH_TABLE)
            logger.info("⚡ Lazy-loading Table model from %s", p)
            self._table_processor = LayoutLMv3Processor.from_pretrained(str(p), apply_ocr=False)
            self._table_model = LayoutLMv3ForTokenClassification.from_pretrained(str(p))
            self._table_model.to(self.device)
            self._table_model.eval()
            with open(p / "label2id.json", encoding="utf-8") as f:
                self.table_label2id = json.load(f)
            with open(p / "id2label.json", encoding="utf-8") as f:
                self.table_id2label = {int(k): v for k, v in json.load(f).items()}

    # ── Footer Model Property ──────────────────────────────────────────
    @property
    def footer_processor(self) -> LayoutLMv3Processor:
        if self._footer_processor is None: self._ensure_footer()
        return self._footer_processor

    @property
    def footer_model(self) -> LayoutLMv3ForTokenClassification:
        if self._footer_model is None: self._ensure_footer()
        return self._footer_model

    def _ensure_footer(self):
        with self._lazy_lock:
            if self._footer_model is not None: return
            p = Path(Config.MODEL_PATH_FOOTER)
            logger.info("⚡ Lazy-loading Footer model from %s", p)
            self._footer_processor = LayoutLMv3Processor.from_pretrained(str(p), apply_ocr=False)
            self._footer_model = LayoutLMv3ForTokenClassification.from_pretrained(str(p))
            self._footer_model.to(self.device)
            self._footer_model.eval()
            with open(p / "label2id.json", encoding="utf-8") as f:
                self.footer_label2id = json.load(f)
            with open(p / "id2label.json", encoding="utf-8") as f:
                self.footer_id2label = {int(k): v for k, v in json.load(f).items()}

    def load(self, log: logging.Logger):
        if self._ready:
            return
        with self._init_lock:
            if self._ready:
                return
            try:
                self._load_components(log)
                self._ready = True
                log.info("✅ ModelManager: Base components (OCR + Engine) loaded.")
                if not GEMINI_ENABLED:
                    log.info("   GEMINI_ENABLED=False -> Pre-warming local models...")
                    self._ensure_header()
                    self._ensure_table()
                    self._ensure_footer()
            except Exception as exc:
                self._init_error = str(exc)
                log.error("ModelManager.load() FAILED", exc_info=True)
                raise

    def _load_components(self, log: logging.Logger):
        # ── [1/5] OCR ─────────────────────────────────────────────
        log.info("[1/5] Loading OCR runner (PaddleOCR + VietOCR)...")
        t = time.monotonic()
        self.ocr_runner = OCRRunner(
            paddle_lang        = Config.OCR_LANG,
            vietocr_model_name = Config.VIETOCR_MODEL,
            device             = self.device,
        )
        log.info("      OCR ready in %.1fs", time.monotonic() - t)

        # ── [5/5] Engine ───────────────────────────────────────────
        log.info("[5/5] Initializing InvoiceExtractionEngine...")
        self.engine = InvoiceExtractionEngine(
            vat_tolerance  = Config.VAT_TOLERANCE,
            min_confidence = Config.MIN_CONFIDENCE,
        )

    # Backward-compat aliases
    @property
    def processor(self) -> LayoutLMv3Processor: return self.header_processor
    @property
    def model(self) -> LayoutLMv3ForTokenClassification: return self.header_model
    @property
    def id2label(self) -> Dict[int, str]: return self.header_id2label
    @property
    def label2id(self) -> Dict[str, int]: return self.header_label2id

    @property
    def ready(self) -> bool:
        return self._ready

    @property
    def init_error(self) -> Optional[str]:
        return self._init_error


# ═══════════════════════════════════════════════════════════════════
# 8.  ASYNC  JOB  QUEUE
# ═══════════════════════════════════════════════════════════════════
class JobStatus:
    PENDING    = "pending"
    PROCESSING = "processing"
    DONE       = "done"
    FAILED     = "failed"


class Job:
    __slots__ = ("job_id", "status", "created_at", "finished_at",
                 "result", "error", "filename")

    def __init__(self, job_id: str, filename: str):
        self.job_id:      str             = job_id
        self.status:      str             = JobStatus.PENDING
        self.created_at:  float           = time.time()
        self.finished_at: Optional[float] = None
        self.result:      Optional[Dict]  = None
        self.error:       Optional[str]   = None
        self.filename:    str             = filename

    def to_dict(self) -> Dict:
        return {
            "job_id":      self.job_id,
            "status":      self.status,
            "filename":    self.filename,
            "created_at":  datetime.fromtimestamp(self.created_at,  tz=timezone.utc).isoformat(),
            "finished_at": (
                datetime.fromtimestamp(self.finished_at, tz=timezone.utc).isoformat()
                if self.finished_at else None
            ),
            "result": self.result,
            "error":  self.error,
        }


class AsyncJobQueue:
    def __init__(self, num_workers: int = 1):
        self._jobs:  Dict[str, Job] = {}
        self._queue: deque          = deque()
        self._lock   = threading.Lock()
        self._cond   = threading.Condition(self._lock)
        for i in range(num_workers):
            t = threading.Thread(target=self._worker, daemon=True, name=f"job-worker-{i}")
            t.start()

    def submit(self, job: Job, image_bytes: bytes) -> str:
        with self._cond:
            if len(self._queue) >= Config.JOB_QUEUE_MAXSIZE:
                raise RuntimeError("Job queue is full — please try again later.")
            self._jobs[job.job_id] = job
            self._queue.append((job, image_bytes))
            self._cond.notify()
        return job.job_id

    def get_job(self, job_id: str) -> Optional[Job]:
        with self._lock:
            return self._jobs.get(job_id)

    def _worker(self):
        while True:
            with self._cond:
                while not self._queue:
                    self._cond.wait()
                job, image_bytes = self._queue.popleft()

            job.status = JobStatus.PROCESSING
            try:
                job.result = run_full_pipeline(image_bytes, job.filename, logger, metrics)
                job.status = JobStatus.DONE
            except Exception as exc:
                job.error  = str(exc)
                job.status = JobStatus.FAILED
                logger.error("Async job %s FAILED: %s", job.job_id, exc)
            finally:
                job.finished_at = time.time()
                self._evict_expired()

    def _evict_expired(self):
        cutoff = time.time() - Config.JOB_TTL_SECONDS
        with self._lock:
            stale = [jid for jid, j in self._jobs.items() if j.created_at < cutoff]
            for jid in stale:
                del self._jobs[jid]


# ═══════════════════════════════════════════════════════════════════
# 9.  PIPELINE  STAGES
# ═══════════════════════════════════════════════════════════════════

class ValidationError(Exception):
    def __init__(self, message: str, http_status: int = 400):
        super().__init__(message)
        self.http_status = http_status


class PipelineError(Exception):
    def __init__(self, stage: str, message: str, cause: Exception = None):
        self.stage = stage
        self.cause = cause
        super().__init__(f"[Stage {stage}] {message}")


# ──────────────────────────────────────────────────────────────────
# MODULE-LEVEL HELPER — Clean garbled footer currency values
# FIX 2: Defined here once at module level, NOT inside stage4_postprocess
# ──────────────────────────────────────────────────────────────────
def clean_footer_value(raw_val) -> Optional[str]:
    """
    Clean garbled multi-token footer extraction values.

    LayoutLMv3 engine may concatenate multiple tokens with the same label,
    producing strings like "2 1350000" or "0 0 0 6 8 0".
    This function keeps only the last token matching Vietnamese currency format.

    Examples:
        "2 1.350.000"  → "1.350.000"
        "0 0 0 6 8 0"  → None  (no valid currency pattern)
        "1.080.000"    → "1.080.000"
        None           → None
    """
    if raw_val is None:
        return None
    parts = str(raw_val).strip().split()
    currency_pattern = re.compile(r'^\d[\d\.]*\d$')
    valid_parts = [p for p in parts if currency_pattern.match(p)]
    if valid_parts:
        return valid_parts[-1]
    return parts[-1] if parts else None


def clean_tax_code(raw: str) -> str:
    """
    Clean OCR noise from Vietnamese tax codes.
    Preserves all valid formats: 10-digit, 13-digit, 10-3 branch.
    Only strips obvious OCR prefix noise (1-2 leading digits).
    """
    if not raw:
        return raw
    import re
    s = str(raw).strip()
    s = re.sub(r'^[\s:.\-]+', '', s)  # remove leading punctuation
    if not s:
        return raw.strip()

    digits_only = re.sub(r'\D', '', s)

    # Valid 10-digit standard company MST
    if re.fullmatch(r'\d{10}', digits_only):
        return digits_only

    # Valid 13-digit household/personal business MST
    if re.fullmatch(r'\d{13}', digits_only):
        return digits_only

    # Valid 10-3 branch format
    if re.fullmatch(r'\d{10}-\d{3}', s):
        return s

    # 11-12 digits: likely OCR added 1-2 char prefix → try trimming
    if len(digits_only) in (11, 12):
        for trim in range(1, 3):
            candidate = digits_only[trim:]
            if re.fullmatch(r'\d{10}', candidate):
                return candidate
            if re.fullmatch(r'\d{13}', candidate):
                return candidate

    # Fallback: return as-is if reasonable length (10-14 digits)
    if 10 <= len(digits_only) <= 14:
        return digits_only

    return s


def is_noise_item(item: dict) -> bool:
    import re

    def get_val(obj, default=""):
        return obj.get("value", default) if isinstance(obj, dict) else obj

    name     = str(get_val(item.get("name", ""))).strip()
    unit     = str(get_val(item.get("unit", ""))).strip()
    qty      = get_val(item.get("quantity", 0), 0)
    price    = get_val(item.get("unit_price", 0), 0)
    total    = get_val(item.get("total", 0), 0)

    def is_pos(v):
        try:
            return float(str(v).replace('.', '').replace(',', '')) > 0
        except:
            return False

    has_value = is_pos(qty) or is_pos(price) or is_pos(total)

    # Rule 1: MST tax code row
    if re.match(r'^MST\s+\d', name, re.IGNORECASE):
        return True

    # Rule 2: Pure long digit string as name (tax code/ID)
    digits = re.sub(r'\D', '', name)
    if len(digits) >= 10 and len(digits) >= len(name.replace(' ', '')) * 0.8:
        return True

    # Rule 3: Address fragment
    addr_keywords = ["địa chỉ", "ngõ", "phường", "quận",
                     "tỉnh", "huyện", "tp.", "p."]
    name_lower = name.lower()
    if any(kw in name_lower for kw in addr_keywords):
        return True

    # Rule 4: Too short with no value
    if len(name) < 4 and not has_value:
        return True

    # Rule 5: Date unit keywords → date fragment mislabeled
    date_units = ["ngày", "tháng", "năm", "day", "month", "year"]
    if unit.lower() in date_units:
        return True

    # Rule 6: Year-like quantity with empty name
    try:
        if 1990 <= int(float(str(qty))) <= 2099 and len(name) == 0:
            return True
    except:
        pass

    # Rule 7: Single word, no value, not a real product name
    if len(name.split()) == 1 and len(name) < 6 and not has_value:
        return True

    # Rule 8: No name AND no price AND qty is small number (not year)
    # (catches ghost items from OCR misreads)
    if len(name) == 0 and not has_value:
        return True
    if len(name) == 0 and is_pos(qty) and not is_pos(price) and not is_pos(total):
        try:
            qty_int = int(float(str(qty)))
            # Small quantities with no name = noise (not a real item row)
            if qty_int < 100:
                return True
        except:
            pass

    # Rule 9: Single Vietnamese word that appears in signature areas
    SIGNATURE_WORDS = {
        "người", "nguoi", "thanh", "ký", "ky", "đại", "dai",
        "diện", "dien", "giám", "giam", "đốc", "doc",
        "kế", "ke", "toán", "toan", "trưởng", "truong"
    }
    if len(name.split()) == 1 and name.lower() in SIGNATURE_WORDS:
        return True

    # Rule 10: Name is just "Thanh" prefix garbage
    # (happens when OCR merges row separator with next item)
    if name.lower().startswith("thanh ") and len(name.split()) <= 2:
        return True

    # Rule 11 — Phone number field leaked into items
    # Pattern: name starts with "điện thoại" OR contains
    # standalone 10-digit Vietnamese phone number
    PHONE_PREFIXES = [
        "điện thoại",   # full
        "thoại",        # truncated: "Điện | thoại 09..."
        "phone",
        "tel:",
        "đt:",
        "fax:",
    ]
    phone_pattern  = re.compile(r'\b0[35789]\d{8}\b')
    name_raw       = name.strip()
    if any(name_lower.startswith(p) for p in PHONE_PREFIXES):
        return True
    if phone_pattern.search(name_raw) and len(name_raw.split()) <= 3:
        # Short name containing only a phone number → noise
        # (avoid killing items like "Máy điện thoại Samsung" with qty>0)
        qty = item.get("quantity", {})
        qty_val = qty.get("value", 0) if isinstance(qty, dict) else qty
        if qty_val == 0:
            return True

    # Rule 12 — Payment method field leaked into items
    PAYMENT_FRAGMENTS = [
        "hình thức thanh toán",
        "hình thức tt",
        "payment method",
        "phương thức thanh toán",
    ]
    if any(frag in name_lower for frag in PAYMENT_FRAGMENTS):
        return True

    # Rule 13 — Bank account field leaked into items
    # Matches "số tài khoản XXXXXXXXX" with no real item data
    BANK_PREFIXES = ["số tài khoản", "stk:", "account number"]
    is_bank_prefix = any(name_lower.startswith(p) for p in BANK_PREFIXES)
    bank_number_only = re.compile(r'^(số tài khoản|tài khoản|stk:?)\s*\d+\s*$', re.I)
    if bank_number_only.match(name_raw):
        return True
    # If starts with bank prefix BUT has real text after → do NOT discard
    # Let clean_item_name() handle it instead (see FIX B)

    # Rule 14 — Buyer label fragments
    BUYER_LABELS = [
        "họ và tên người mua",
        "họ tên người mua",
        "tên đơn vị",
        "thông tin người mua",
        "buyer information",
        "mã số thuế người mua",
    ]
    if any(frag in name_lower for frag in BUYER_LABELS):
        return True

    # Rule 15 — VAT footer summary rows (GTGT invoices)
    # Patterns like "Thuế suất 10%:", "Không chịu thuế GTGT:", etc.
    # These are footer labels mislabeled as item names
    VAT_FOOTER_PATTERNS = [
        r'^thuế suất\s+\d+%',           # "Thuế suất 10%:", "Thuế suất 8%:"
        r'^không kê khai thuế',          # "Không kê khai thuế GTGT:"
        r'^không chịu thuế',             # "Không chịu thuế GTGT:"
        r'^hàng hóa không chịu thuế',   # variant
        r'^hàng hóa miễn thuế',         # variant
        r'^tổng tiền thanh toán',        # total label leaked
        r'^cộng tiền hàng',              # subtotal label leaked
        r'^thuế gtgt',                   # "Thuế GTGT:"
        r'^tổng cộng$',                  # just "Tổng cộng"
    ]
    for vat_pattern in VAT_FOOTER_PATTERNS:
        if re.search(vat_pattern, name_lower):
            return True

    # Rule 16 — Tax code label row (not the actual tax code value)
    # "Mã số thuế:" with no quantity/price = header label leaked
    TAX_LABEL_PATTERNS = [
        r'^mã số thuế\s*:',
        r'^mst\s*:',
        r'^tax\s+code\s*:',
    ]
    for tax_pat in TAX_LABEL_PATTERNS:
        if re.search(tax_pat, name_lower) and not has_value:
            return True

    # Rule 17 — Numeric-only names that are amounts (e.g. "1.218.000")
    # These are unit_price values leaked into name field
    # Pattern: contains only digits, dots, commas — no letters
    name_stripped = re.sub(r'[\d.,\s]', '', name)
    if len(name_stripped) == 0 and len(name) > 3:
        return True

    return False





def validate_and_fix_amounts(invoice: dict) -> dict:
    def parse_amount(val) -> int:
        if not val:
            return 0
        if isinstance(val, dict):
            val = val.get("value", 0)
        if val is None:
            return 0
        try:
            # Handle both int, float, and string formats
            cleaned = str(val).replace('.', '').replace(',', '').strip()
            # Remove any non-digit characters
            import re
            cleaned = re.sub(r'[^\d]', '', cleaned)
            return int(cleaned) if cleaned else 0
        except (ValueError, TypeError):
            return 0

    def set_field(invoice, field, value, confidence=0.5):
        if isinstance(invoice.get(field), dict):
            invoice[field]["value"] = value
            invoice[field]["confidence"] = confidence
            invoice[field]["_corrected"] = True
        else:
            invoice[field] = value

    total   = parse_amount(invoice.get("total_amount"))
    subtotal = parse_amount(invoice.get("subtotal"))
    vat     = parse_amount(invoice.get("vat_amount"))

    if total <= 0:
        return invoice  # Can't fix without total

    # ─── RULE 0 (NEW — must be FIRST) ────────────────────────────
    # If subtotal + vat ≈ total → already correct, exit immediately
    # PATCH: only exit early if vat is meaningful (> 1000)
    if subtotal > 0 and vat > 1_000:
        diff_ratio = abs((subtotal + vat) - total) / max(total, 1)
        if diff_ratio <= 0.01:
            return invoice  # ✅ Perfect — do NOT modify anything
    # ─────────────────────────────────────────────────────────────

    # Rule 1: VAT noise — impossibly small (< 1000 VND)
    if 0 < vat < 1_000:
        set_field(invoice, "vat_amount", 0)
        vat = 0

    # Rule 2: subtotal LARGER than total → clearly wrong
    if subtotal > total:
        # This invoice likely has no VAT
        set_field(invoice, "subtotal", total)
        set_field(invoice, "vat_amount", 0)
        return invoice

    # Rule 3: subtotal much smaller than total (< 50%) → wrong
    if 0 < subtotal < total * 0.5:
        if vat > 0:
            # subtotal = total - vat (if that makes sense)
            candidate = total - vat
            if candidate > 0 and candidate < total:
                set_field(invoice, "subtotal", candidate)
        else:
            set_field(invoice, "subtotal", total)
        return invoice

    # Rule 4: subtotal + vat ≈ total (within 1%) → OK
    if subtotal > 0:
        calc = subtotal + vat
        if abs(calc - total) / total <= 0.01:
            return invoice  # Correct

    # Rule 5: vat = 0, subtotal should equal total
    if vat == 0 and subtotal != total:
        set_field(invoice, "subtotal", total)

    return invoice


def _recover_vat_from_totals(invoice: dict,
                              items: list = None) -> dict:
    """
    Recovery heuristic for cases where footer model missed
    subtotal and/or vat_amount tokens.

    Failure modes handled:
      A. subtotal = 0       (model missed subtotal entirely)
      B. subtotal = total   (model mislabeled total as subtotal)
      C. subtotal present, vat = 0

    Disambiguation strategy (most reliable first):
      1. If vat_rate string present → exact arithmetic (conf=0.85)
      2. If items available → use items_sum to confirm VAT exists
      3. Rate heuristic: try 10% / 8% / 5%  (conf=0.70)
      4. Case D guard: skip recovery for confirmed no-VAT invoices
         using items_sum evidence, NOT invoice_type alone

    Only fires when vat_amount < 1000 VND (noise level).
    """

    # ── Helpers ──────────────────────────────────────────────────
    def _parse(obj) -> int:
        v = obj.get("value", 0) if isinstance(obj, dict) else obj
        try:
            return int(str(v).replace(".", "").replace(",", "").strip())
        except Exception:
            return 0

    def _set_field(field_obj, new_val: int, conf: float):
        if isinstance(field_obj, dict):
            field_obj["value"]      = new_val
            field_obj["confidence"] = conf
            field_obj["_recovered"] = True
        return field_obj

    def _apply(rate: float, conf: float):
        implied_subtotal = round(total / (1.0 + rate))
        implied_vat      = total - implied_subtotal
        math_ok  = abs((implied_subtotal + implied_vat) - total) / total <= 0.01
        range_ok = total * 0.50 < implied_subtotal < total * 0.99
        if math_ok and range_ok:
            _set_field(invoice.get("subtotal",   {}), implied_subtotal, conf)
            _set_field(invoice.get("vat_amount", {}), implied_vat,      conf)
            return True
        return False

    # ── Parse current values ──────────────────────────────────────
    total    = _parse(invoice.get("total_amount", {}))
    subtotal = _parse(invoice.get("subtotal",     {}))
    vat      = _parse(invoice.get("vat_amount",   {}))

    if total <= 0:
        return invoice

    # Guard: vat already real
    if vat >= 1_000:
        if abs((subtotal + vat) - total) / total <= 0.01:
            return invoice

    # Guard: already consistent
    if subtotal > 0 and vat > 0:
        if abs((subtotal + vat) - total) / total <= 0.01:
            return invoice

    # ── Compute items_sum from table model output ─────────────────
    items_sum = 0
    if items:
        for item in items:
            t = item.get("total", {})
            items_sum += _parse(t)

    TOLERANCE = 0.05  # 5% tolerance for items sum matching

    def _near(a: int, b: int) -> bool:
        if b == 0:
            return False
        return abs(a - b) / b <= TOLERANCE

    # ── Determine failure mode ────────────────────────────────────
    is_mode_a = (subtotal == 0)
    is_mode_b = (subtotal > 0 and subtotal == total)
    is_mode_c = (0 < subtotal < total and vat == 0)

    if not (is_mode_a or is_mode_b or is_mode_c):
        return invoice

    # ── Strategy 1: vat_rate string ──────────────────────────────
    vat_rate_obj = invoice.get("vat_rate", {})
    vat_rate_str = (
        vat_rate_obj.get("value", "")
        if isinstance(vat_rate_obj, dict)
        else str(vat_rate_obj)
    ).strip()

    rate_from_string = None
    if vat_rate_str:
        try:
            rate_from_string = float(
                vat_rate_str.replace("%", "").strip()
            ) / 100.0
        except ValueError:
            pass

    if rate_from_string and 0 < rate_from_string <= 0.25:
        if _apply(rate_from_string, conf=0.85):
            return invoice

    # ── Strategy 2: Items sum disambiguation ─────────────────────
    VAT_RATES = [0.10, 0.08, 0.05]

    if items_sum > 0:
        if _near(items_sum, total):
            # items_sum ≈ total → subtotal was set to total (wrong)
            # VAT definitely exists → recover
            for rate in VAT_RATES:
                if _apply(rate, conf=0.75):
                    return invoice

        elif _near(items_sum, subtotal):
            # items_sum ≈ subtotal → subtotal is CORRECT
            # This is a genuine no-VAT invoice → do NOT recover
            return invoice

        # items_sum doesn't match either → inconclusive
        # Fall through to rate heuristic below

    # ── Strategy 3: Rate heuristic (no items or inconclusive) ────
    # Case D guard: only skip recovery if confirmed no-VAT
    # via items_sum evidence above — do NOT use invoice_type alone
    # because "Bán hàng" invoices CAN have VAT.
    for rate in VAT_RATES:
        if _apply(rate, conf=0.65):
            return invoice

    return invoice   # no strategy matched


# ──────────────────────────────────────────────────────────────────
# STAGE 0 — Input Validation
# ──────────────────────────────────────────────────────────────────
def stage0_validate(filename: str, file_bytes: bytes) -> str:
    """
    Validate uploaded file before any processing.
    Returns file extension string (e.g. ".png").
    Raises ValidationError on failure.
    """
    ext = Path(filename).suffix.lower()

    if ext not in Config.ALLOWED_EXTENSIONS:
        raise ValidationError(
            f"Unsupported format '{ext}'. Accepted: {', '.join(sorted(Config.ALLOWED_EXTENSIONS))}"
        )

    size = len(file_bytes)
    if size == 0:
        raise ValidationError("Uploaded file is empty.")
    if size > Config.MAX_UPLOAD_BYTES:
        raise ValidationError(
            f"File too large ({size / 1_048_576:.1f} MB). "
            f"Maximum: {Config.MAX_UPLOAD_BYTES // 1_048_576} MB"
        )

    if ext != ".pdf":
        arr = np.frombuffer(file_bytes, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            raise ValidationError("Cannot decode image — file may be corrupted.")
        h, w = img.shape[:2]
        if w < 100 or h < 100:
            raise ValidationError(
                f"Image resolution too low ({w}×{h} px). Minimum: 100×100 px."
            )

    return ext


# ──────────────────────────────────────────────────────────────────
# STAGE 1 — Image Preprocessing
# ──────────────────────────────────────────────────────────────────
def stage1_preprocess(
    image_bytes: bytes,
    ext:         str,
    temp_dir:    Path,
    file_stem:   str,
    log:         logging.Logger,
) -> Tuple[List[Tuple[np.ndarray, int, int]], List[Path]]:
    """
    Normalize invoice image(s).
    For PDF: render each page at PDF_DPI, then normalize.
    For image: decode bytes, write temp PNG, normalize.
    Returns: (pages [(img_bgr, w, h)], temp_paths)
    """
    temp_paths: List[Path] = []
    pages:      List[Tuple[np.ndarray, int, int]] = []

    if ext == ".pdf":
        try:
            import fitz
        except ImportError:
            raise PipelineError("1_preprocess", "PyMuPDF not installed. Run: pip install pymupdf")

        pdf_path = temp_dir / f"{file_stem}.pdf"
        pdf_path.write_bytes(image_bytes)
        temp_paths.append(pdf_path)

        doc = fitz.open(str(pdf_path))
        log.info("  PDF: %d page(s) detected", len(doc))

        for page_num in range(len(doc)):
            page     = doc.load_page(page_num)
            mat      = fitz.Matrix(Config.PDF_DPI / 72, Config.PDF_DPI / 72)
            pix      = page.get_pixmap(matrix=mat)
            png_path = temp_dir / f"{file_stem}_p{page_num + 1}_raw.png"
            pix.save(str(png_path))
            temp_paths.append(png_path)

            img_bgr, w, h = normalize_invoice_image(image_path=png_path, output_path=None)
            pages.append((img_bgr, w, h))
            log.debug("  PDF page %d normalized → %d×%d", page_num + 1, w, h)

        doc.close()

    else:
        arr     = np.frombuffer(image_bytes, dtype=np.uint8)
        raw_img = cv2.imdecode(arr, cv2.IMREAD_COLOR)

        raw_path = temp_dir / f"{file_stem}_raw{ext}"
        cv2.imwrite(str(raw_path), raw_img)
        temp_paths.append(raw_path)

        img_bgr, w, h = normalize_invoice_image(image_path=raw_path, output_path=None)
        pages.append((img_bgr, w, h))
        log.debug("  Image normalized → %d×%d", w, h)

    return pages, temp_paths


# ──────────────────────────────────────────────────────────────────
# STAGE 2 — OCR (PaddleOCR + VietOCR)
# ──────────────────────────────────────────────────────────────────
def stage2_ocr(
    mm:        ModelManager,
    img_bgr:   np.ndarray,
    w:         int,
    h:         int,
    temp_dir:  Path,
    file_stem: str,
    log:       logging.Logger,
) -> Tuple[Dict, List[str], List[List[int]]]:
    """
    OCR pipeline: PaddleOCR detect → VietOCR recognize → flatten tokens.
    Returns: (ocr_result, words, bboxes[0–1000])
    """
    norm_path = temp_dir / f"{file_stem}_norm.png"
    cv2.imwrite(str(norm_path), img_bgr)

    dets = mm.ocr_runner.detect_text_paddleocr(img_bgr)
    ocr_result = mm.ocr_runner.merge_results(
        image_name = file_stem,
        width      = w,
        height     = h,
        detections = dets,
        img_bgr    = img_bgr,
    )

    if not ocr_result.get("lines"):
        raise PipelineError(
            "2_ocr",
            "OCR returned no text. Image may be blank, too noisy, or resolution too low."
        )

    log.debug("  OCR: %d lines detected", len(ocr_result["lines"]))

    words, bboxes = prepare_words_and_bboxes(ocr_result, w, h)

    if not words:
        raise PipelineError("2_ocr", "No tokens produced after OCR processing.")

    log.debug("  Tokens prepared: %d words", len(words))
    return ocr_result, words, bboxes


# ──────────────────────────────────────────────────────────────────
# STAGE 3a/3b — Single-pass LayoutLMv3 Inference (Header + Table)
# FIX 1: Aligned subword aggregation — use first subword label (authoritative)
#         and mean confidence across subwords (consistent with sliding_window)
# ──────────────────────────────────────────────────────────────────
def _run_single_model_inference(
    processor,
    model,
    id2label:   Dict[int, str],
    pil_image,
    words:      List[str],
    bboxes:     List[List[int]],
    device:     str,
    log:        logging.Logger,
    model_name: str = "model",
) -> Tuple[List[str], List[float]]:
    """
    Run one LayoutLMv3 model (single pass, no sliding window).
    Used for header and table — their fields appear near the top
    of the invoice and fit within the 512-token limit.

    Subword aggregation (FIX 1):
      - Label  : first subword is authoritative (LayoutLMv3 convention)
      - Confidence: mean across all subwords (for overlap resolution parity)
    """
    encoding = processor(
        images         = pil_image,
        text           = words,
        boxes          = bboxes,
        truncation     = True,
        padding        = True,
        return_tensors = "pt",
    )

    # word_ids BEFORE moving tensors to device
    word_ids        = encoding.word_ids(batch_index=0)
    encoding_device = {k: v.to(device) for k, v in encoding.items()}

    with torch.no_grad():
        outputs     = model(**encoding_device)
        logits      = outputs.logits
        predictions = logits.argmax(-1).squeeze(0)
        confidences = torch.softmax(logits, dim=-1).max(-1).values.squeeze(0)

    word_labels: List[str]   = []
    word_confs:  List[float] = []
    current_word             = None
    tmp_preds:   List[int]   = []
    tmp_confs:   List[float] = []

    for idx, word_id in enumerate(word_ids):
        if word_id is None:
            continue
        pred_id = predictions[idx].item()
        conf    = confidences[idx].item()

        if word_id != current_word:
            if current_word is not None:
                # FIX 1: first subword label is authoritative for LayoutLMv3
                word_labels.append(id2label[tmp_preds[0]])
                word_confs.append(float(np.mean(tmp_confs)))
            current_word = word_id
            tmp_preds    = [pred_id]
            tmp_confs    = [conf]
        else:
            tmp_preds.append(pred_id)
            tmp_confs.append(conf)

    # Flush last word
    if tmp_preds:
        # FIX 1 (flush): first subword label is authoritative
        word_labels.append(id2label[tmp_preds[0]])
        word_confs.append(float(np.mean(tmp_confs)))

    non_o = sum(1 for l in word_labels if l != "O")
    log.debug(
        "  [%s] %d words → %d non-O (%.0f%%)",
        model_name, len(word_labels), non_o,
        100.0 * non_o / max(len(word_labels), 1),
    )
    return word_labels, word_confs


# ──────────────────────────────────────────────────────────────────
# STAGE 3c — Sliding Window Inference (Footer)
# Handles LayoutLMv3 512-token limit for long Vietnamese invoices
# CHUNK_SIZE=150 words (~450 subwords), STRIDE=75 words (50% overlap)
# FIX 3: log param now used for per-chunk debug output
# ──────────────────────────────────────────────────────────────────
def sliding_window_inference(
    model,
    processor,
    id2label:   Dict[int, str],
    pil_image,
    words:      List[str],
    bboxes:     List[List[int]],
    device:     str,
    log:        logging.Logger,
    model_name: str = "footer",
) -> Tuple[List[str], List[float]]:
    """
    Sliding window inference bypassing LayoutLMv3 512-token limit.
    Footer fields appear at the bottom of long invoices and are
    typically truncated in a single-pass inference.

    Algorithm:
      1. Split word list into overlapping chunks (CHUNK_SIZE, STRIDE)
      2. Run inference independently per chunk
      3. Merge with overlap resolution:
           - non-O beats O
           - higher confidence wins on real-label conflicts
    """
    CHUNK_SIZE = 150   # ~450 subwords, safely under 512
    STRIDE     = 75    # 50% overlap
    MAX_CHUNKS = 8

    n_words = len(words)
    if n_words == 0:
        return [], []

    final_labels = ["O"] * n_words
    final_confs  = [0.0]  * n_words

    # Build chunk list
    chunks = []
    start  = 0
    # BUG 14: Add boundary overlap to prevent dropping items at chunk edges.
    # CHUNK_SIZE=200, STRIDE=100 (Balanced overlap for speed)
    STRIDE = 100 
    while start < n_words and len(chunks) < MAX_CHUNKS:
        end = min(start + CHUNK_SIZE, n_words)
        chunks.append((start, end))
        if end == n_words:
            break
        start += STRIDE

    non_o_total = 0
    for i, (start_idx, end_idx) in enumerate(chunks):
        chunk_words  = words [start_idx:end_idx]
        chunk_bboxes = bboxes[start_idx:end_idx]

        encoding = processor(
            images         = pil_image,
            text           = chunk_words,
            boxes          = chunk_bboxes,
            truncation     = True,
            padding        = True,
            return_tensors = "pt",
        )
        word_ids           = encoding.word_ids(batch_index=0)
        encoding_on_device = {k: v.to(device) for k, v in encoding.items()}

        with torch.no_grad():
            outputs     = model(**encoding_on_device)
            predictions = outputs.logits.argmax(-1).squeeze(0)
            probs       = torch.softmax(outputs.logits, dim=-1)
            confidences = probs.max(-1).values.squeeze(0)

        # Subword → word-level aggregation
        # FIX 1 (consistent): first subword label authoritative, mean confidence
        chunk_word_labels: List[str]   = []
        chunk_word_confs:  List[float] = []
        current_word = None
        temp_preds:  List[int]   = []
        temp_confs:  List[float] = []

        for idx, word_id in enumerate(word_ids):
            if word_id is None:
                continue
            pred_id = predictions[idx].item()
            conf    = confidences[idx].item()

            if word_id != current_word:
                if current_word is not None:
                    chunk_word_labels.append(id2label[temp_preds[0]])
                    chunk_word_confs.append(float(np.mean(temp_confs)))
                current_word = word_id
                temp_preds   = [pred_id]
                temp_confs   = [conf]
            else:
                temp_preds.append(pred_id)
                temp_confs.append(conf)

        if temp_preds:
            chunk_word_labels.append(id2label[temp_preds[0]])
            chunk_word_confs.append(float(np.mean(temp_confs)))

        non_o_chunk = sum(1 for l in chunk_word_labels if l != "O")
        non_o_total += non_o_chunk
        token_count  = sum(1 for wid in word_ids if wid is not None)

        # FIX 3: actual logging via log param (was silently unused before)
        log.debug(
            "  [%s CHUNK %d/%d] words[%d:%d] tokens≈%d non-O=%d",
            model_name, i + 1, len(chunks),
            start_idx, end_idx, token_count, non_o_chunk,
        )

        # Overlap resolution & merge
        conflicts = 0
        for w_i, (lbl, conf) in enumerate(zip(chunk_word_labels, chunk_word_confs)):
            global_i = start_idx + w_i
            if global_i >= n_words:
                break

            curr_lbl  = final_labels[global_i]
            curr_conf = final_confs [global_i]

            if curr_lbl == "O" and lbl != "O":
                # Non-O always wins over O
                final_labels[global_i] = lbl
                final_confs [global_i] = conf
            elif curr_lbl != "O" and lbl != "O" and curr_lbl != lbl:
                # Conflict: higher confidence wins
                conflicts += 1
                if conf > curr_conf:
                    final_labels[global_i] = lbl
                    final_confs [global_i] = conf
            elif curr_lbl == lbl:
                # Same label: keep higher confidence
                if conf > curr_conf:
                    final_confs[global_i] = conf
            elif curr_lbl == "O" and lbl == "O":
                if conf > curr_conf:
                    final_confs[global_i] = conf

        if conflicts:
            log.debug(
                "  [%s CHUNK %d/%d] %d conflict(s) resolved by confidence",
                model_name, i + 1, len(chunks), conflicts,
            )

    log.debug("  [%s] Sliding window done — total non-O: %d", model_name, non_o_total)
    return final_labels, final_confs


def table_sliding_window_inference(
    model,
    processor,
    id2label:   Dict[int, str],
    pil_image,
    words:      List[str],
    bboxes:     List[List[int]],
    device:     str,
    log:        logging.Logger,
    model_name: str = "table",
) -> Tuple[List[str], List[float]]:
    """
    Run table model inference with sliding window.
    CHUNK_SIZE=200, STRIDE=100, MAX_CHUNKS=10.
    """
    CHUNK_SIZE = 200
    STRIDE     = 120 # Balanced overlap (was 80)
    MAX_CHUNKS = 10

    n_words = len(words)
    if n_words == 0:
        return [], []

    final_labels = ["O"] * n_words
    final_confs  = [0.0]  * n_words

    chunks = []
    start  = 0
    while start < n_words and len(chunks) < MAX_CHUNKS:
        end = min(start + CHUNK_SIZE, n_words)
        chunks.append((start, end))
        if end == n_words:
            break
        start += STRIDE

    log.info(f"[table-SW] {n_words} words → {len(chunks)} chunks")

    for i, (start_idx, end_idx) in enumerate(chunks):
        chunk_words  = words [start_idx:end_idx]
        chunk_bboxes = bboxes[start_idx:end_idx]

        encoding = processor(
            images         = pil_image,
            text           = chunk_words,
            boxes          = chunk_bboxes,
            truncation     = True,
            padding        = True,
            return_tensors = "pt",
        )
        word_ids           = encoding.word_ids(batch_index=0)
        encoding_on_device = {k: v.to(device) for k, v in encoding.items()}

        with torch.no_grad():
            outputs     = model(**encoding_on_device)
            predictions = outputs.logits.argmax(-1).squeeze(0)
            probs       = torch.softmax(outputs.logits, dim=-1)
            confidences = probs.max(-1).values.squeeze(0)

        chunk_word_labels: List[str]   = []
        chunk_word_confs:  List[float] = []
        current_word = None
        temp_preds:  List[int]   = []
        temp_confs:  List[float] = []

        for idx, word_id in enumerate(word_ids):
            if word_id is None:
                continue
            pred_id = predictions[idx].item()
            conf    = confidences[idx].item()

            if word_id != current_word:
                if current_word is not None:
                    chunk_word_labels.append(id2label[temp_preds[0]])
                    chunk_word_confs.append(float(np.mean(temp_confs)))
                current_word = word_id
                temp_preds   = [pred_id]
                temp_confs   = [conf]
            else:
                temp_preds.append(pred_id)
                temp_confs.append(conf)

        if temp_preds:
            chunk_word_labels.append(id2label[temp_preds[0]])
            chunk_word_confs.append(float(np.mean(temp_confs)))

        non_o_count = sum(1 for l in chunk_word_labels if l != "O")
        token_count = sum(1 for wid in word_ids if wid is not None)
        log.info(f"[CHUNK {i+1}/{len(chunks)}] words[{start_idx}:{end_idx}] "
                 f"→ tokens≈{token_count} → non-O labels: {non_o_count}")

        for w_i, (lbl, conf) in enumerate(zip(chunk_word_labels, chunk_word_confs)):
            global_i = start_idx + w_i
            if global_i >= n_words:
                break
            
            curr_lbl  = final_labels[global_i]
            curr_conf = final_confs [global_i]

            if curr_lbl == "O" and lbl != "O":
                final_labels[global_i] = lbl
                final_confs [global_i] = conf
            elif curr_lbl != "O" and lbl != "O" and curr_lbl != lbl:
                if conf > curr_conf:
                    final_labels[global_i] = lbl
                    final_confs [global_i] = conf
            elif curr_lbl == lbl:
                if conf > curr_conf:
                    final_confs[global_i] = conf

    return final_labels, final_confs


def _item_has_content(item: dict) -> bool:
    """Item must have at least a name OR a price to be kept."""
    has_name = bool(item.get("name", {}).get("value", "").strip())
    has_price = any(
        item.get(f, {}).get("value") 
        for f in ["unit_price", "total", "quantity"]
    )
    return has_name or has_price



def _normalize_unit_price_value(price_val) -> float:
    """Normalize unit_price value: handle Vietnamese number format."""
    if not price_val and price_val != 0:
        return 0.0
    
    if isinstance(price_val, (int, float)):
        result = float(price_val)
        return result
    
    if isinstance(price_val, str):
        # Vietnamese format: "2.878.435" → 2878435
        price_str = price_val.strip()
        if not price_str:
            return 0.0
        
        # Fix: Remove OCR bleed - trailing space + "0" (VAT rate) from adjacent cell
        # Example: "79.185 0" (where "0" is from VAT column) → "79.185"
        price_str = re.sub(r'\s+0$', '', price_str)
        
        # Remove all dots and replace comma with dot (if present)
        price_str = price_str.replace('.', '').replace(',', '.')
        
        try:
            result = float(price_str)
            return result
        except (ValueError, TypeError):
            return 0.0
    
    return 0.0


def _get_fallback_unit(items_list: List[dict]) -> str:
    """Extract most common non-empty unit from all items for fallback."""
    from collections import Counter
    units = [item.get("unit", {}).get("value", "") 
             for item in items_list 
             if item.get("unit", {}).get("value", "").strip()]
    if units:
        most_common = Counter(units).most_common(1)[0][0]
        return most_common
    return ""  # Be honest if no units found


def _clean_reconstructed_name(name: str) -> str:
    """
    Remove leading bank-account / phone-number prefix from item names.

    Contamination pattern (sliding window overlap with buyer info):
      "tài khoản 10028280217981 Ổ cứng SSD Samsung 870 EVO 1TB"
      "số tài khoản 1558329110900 Nước ngọt Pepsi 330ml"
      "thoại 0928089778 Sản phẩm X"

    Strategy: find first 10-18 digit block (bank account or phone).
    Strip everything before it. Return remainder as clean name.

    Guards:
      - prefix must be ≤ 55 chars (label, not product description)
      - remainder must be > 5 chars (real product name)
      - if no long digit block → return as-is (no change)
      - if no remainder → return as-is (is_noise_item() will remove)

    Unicode-safe: uses only digit detection, no Vietnamese matching.
    No external dependencies: pure re + str operations.
    """
    import re

    if not name or not name.strip():
        return name

    # Find: optional non-digit prefix (≤80 chars) + 10-18 digit block
    m = re.search(r'^\D{0,80}(\d{10,18})', name)
    if not m:
        return ' '.join(name.split())   # normalize whitespace, return

    digits_start = m.start(1)
    digits_end   = m.end(1)

    prefix    = name[:digits_start].strip(' :,-\u2013\u2014\t')
    remainder = name[digits_end:].lstrip(' :,-\u2013\u2014\t')

    # Guard: prefix too long → digits are part of product code, not label
    if len(prefix) > 55:
        return ' '.join(name.split())

    rem_cleaned = ' '.join(remainder.split())

    # Guard: remainder too short → pure noise, let is_noise_item() remove
    if len(rem_cleaned) <= 5:
        return name

    return rem_cleaned


UNIT_PATTERNS = {
    # ── Tools / Electronics (prioritize size indicator → Cái) ───
    r'\b\d+\s*(w|watt)\b':             "Cái",    # 100W
    r'\b\d+\s*inch\b':                 "Cái",    # 24 inch
    r'\b\d+\s*(cổng|port)\b':          "Cái",    # 24 cổng
    r'\b\d+\s*tầng\b':                 "Cái",    # 4 tầng
    r'\b\d+\s*tấn\b':                  "Cái",    # 3 tấn (xe nâng)
    r'\b\d+\s*kg\b':                   "Cái",    # 30kg scale
    
    # ── OCR Corrections ──────────────────────────────
    r'\blkg\b':                        "Kg",     # OCR misread "1kg"
    
    # ── Liquid / Volume ──────────────────────────────
    r'\b\d+\s*ml\b':                   "Chai",   
    r'\b\d+\s*lít\b':                  "Lít",    
    r'\b\d+\s*l\b':                    "Lít",    
    r'\b(lít|lit)\b':                  "Lít",    
    r'\bml\b':                         "Chai",
    
    # ── Weight (Pure units) ──────────────────────────
    r'\bkg\b':                         "Kg",
    r'\b(gram|tấn)\b':                 "Kg",     # general weight category
    
    # ── Length / Area ────────────────────────────────
    r'\b(ống|pipe|tube)\b':            "m",      
    r'\b(dây|cáp|cable|wire)\b':       "m",      
    r'\b\d+\s*mm\b':                   "Cái",    
    r'\b\d+\s*cm\b':                   "Cái",    
    r'\b\d+\s*m\b(?!\d)':             "Cái",    
    r'\b(m2|m²|mét vuông)\b':         "m²",
    r'\bm\b':                          "m",
    
    # ── Containers / Packaging ───────────────────────
    r'\b(hộp|hop)\b':                  "Hộp",
    r'\b(chai|bình|can)\b':            "Chai",
    r'\b(túi|gói|bao|pack)\b':         "Gói",
    r'\b(thùng|carton)\b':             "Thùng",
    r'\b(cuộn|cuon|roll)\b':           "Cuộn",
    r'\b(tấm|tam)\b':                  "Tấm",
    
    # ── Sets / Units ─────────────────────────────────
    r'\b(bộ|bo|set)\b':                "Bộ",
    r'\b(đôi|pair)\b':                 "Đôi",
    r'\b(viên|vien)\b':                "Viên",
    r'\b(tờ|sheet)\b':                 "Tờ",
    r'\b(cái|chiếc|chiec)\b':          "Chiếc",
    r'\b(cây|cay)\b':                  "Cây",
    r'\b(lốc|loc)\b':                  "Lốc",
    r'\b(vỉ|vi)\b':                    "Vỉ",
}

def _clean_reconstructed_unit(unit_str: str) -> str:
    """Clean reconstructed unit field with pattern matching heuristic."""
    if not unit_str or not isinstance(unit_str, str):
        return ""
    
    val = unit_str.strip().lower()
    if not val:
        return ""

    # Strategy: try pattern matches first
    for pattern, normalized in UNIT_PATTERNS.items():
        if re.search(pattern, val, re.IGNORECASE):
            return normalized

    # Fallback: if it's a short word, return capitalized
    if len(val) <= 10:
        return val.capitalize()

    return ""


def _safe_numeric(v):
    """Safely convert any value to numeric (handles str, int, float)."""
    if isinstance(v, (int, float)):
        return float(v)
    if isinstance(v, str):
        cleaned = v.replace('.', '').replace(',', '').strip()
        try:
            return float(cleaned)
        except (ValueError, TypeError):
            return 0.0
    return 0.0


def _ensure_unit_and_price(items: list) -> list:
    """Phase 16 SAFETY NET: Guarantee unit & unit_price ALWAYS present and valid."""
    if not items:
        return items
    
    fallback_unit = _get_fallback_unit(items) or "Cái"
    
    for item in items:
        if not isinstance(item, dict):
            continue
        
        # ── Unit: guarantee always has value ──
        if not isinstance(item.get("unit"), dict):
            item["unit"] = make_field(fallback_unit, 0.6)
        elif not str(item["unit"].get("value", "")).strip():
            item["unit"]["value"] = fallback_unit
            item["unit"]["confidence"] = 0.6
        
        # ── Unit_price: calculate from total/qty if missing ──
        qty = _safe_numeric(item.get("quantity", {}).get("value", 0) if isinstance(item.get("quantity"), dict) else item.get("quantity", 0))
        total = _safe_numeric(item.get("total", {}).get("value", 0) if isinstance(item.get("total"), dict) else item.get("total", 0))
        price = _safe_numeric(item.get("unit_price", {}).get("value", 0) if isinstance(item.get("unit_price"), dict) else item.get("unit_price", 0))
        
        if qty > 0 and total > 0 and price == 0:
            item["unit_price"]["value"] = round(total / qty, 2)
            item["unit_price"]["confidence"] = 0.75
    
    return items


def _reconstruct_items_from_labels(labels: List[str], confs: List[float], words: List[str]) -> List[dict]:
    """Reconstruct item rows from BIO labels. Phase 15: Robust unit & unit_price extraction."""
    items = []
    current_item = {}
    
    # Phase 15 FIX: Complete field mapping with all variations
    # NOTE: ITEM_UNIT label has low prediction rate (~5% coverage).
    # The table model rarely predicts ITEM_UNIT tokens due to insufficient
    # training annotations. Unit values are recovered via heuristic in
    # _normalize_sw_items() as a fallback.
    # TODO: Add more ITEM_UNIT annotations to training data and retrain.
    field_map = {
        "ITEM_NAME":        "name",
        "ITEM_UNIT":        "unit",
        "ITEM_QUANTITY":    "quantity",
        "ITEM_UNIT_PRICE":  "unit_price",
        "ITEM_TOTAL_PRICE": "total",
        "ITEM_TOTAL":       "total",      # Legacy alias
        "ITEM_DISCOUNT":    "discount",
        "ITEM_VAT_RATE":    "vat_rate",
        "ITEM_LINE_TAX":    "line_tax",
        "ITEM_ROW_TOTAL":   "row_total",
    }

    def _normalize_numeric(val_str: str):
        if not val_str:
            return 0
        val_str = str(val_str).strip()
        # Fix: Remove OCR bleed - trailing space + "0" (VAT rate) from adjacent cell
        # Example: "1000 0" → "1000"
        val_str = re.sub(r'\s+0$', '', val_str)
        cleaned = val_str.replace('.', '').replace(',', '')
        try:
            val = int(cleaned)
            return val
        except ValueError:
            try:
                return float(cleaned)
            except:
                return 0

    for i, (label, conf, text) in enumerate(zip(labels, confs, words)):
        if label.startswith("B-ITEM_NAME"):
            if current_item and _item_has_content(current_item):
                # Phase 15: Clean name + unit before finalization
                if "name" in current_item and isinstance(current_item.get("name"), dict):
                    current_item["name"]["value"] = _clean_reconstructed_name(
                        current_item["name"].get("value", "")
                    )
                if "unit" in current_item and isinstance(current_item.get("unit"), dict):
                    current_item["unit"]["value"] = _clean_reconstructed_unit(
                        current_item["unit"].get("value", "")
                    )
                items.append(current_item)
            current_item = {
                "name": make_field(text, conf)
            }
        elif label.startswith("I-ITEM_NAME"):
            if "name" in current_item:
                current_item["name"]["value"] = \
                    current_item["name"]["value"] + " " + text
                current_item["name"]["confidence"] = max(
                    current_item["name"]["confidence"], conf
                )
        elif "ITEM_" in label:
            # Phase 15: Robust BIO label matching with proper continuation
            for label_key, field_name in field_map.items():
                if label_key in label:
                    prefix = label.split("-")[0]
                    # B- prefix: always start new field. I- prefix: continue or start
                    if prefix == "B" or field_name not in current_item:
                        current_item[field_name] = make_field(text, conf)
                    else:
                        # I- continuation: append with space
                        current_item[field_name]["value"] = \
                            str(current_item[field_name].get("value", "")) + " " + text
                        current_item[field_name]["confidence"] = max(
                            float(current_item[field_name].get("confidence", 0)), conf
                        )
                    break  # Stop after first field_map match

    if current_item and _item_has_content(current_item):
        # Phase 15: Clean name + unit before finalization
        if "name" in current_item and isinstance(current_item.get("name"), dict):
            current_item["name"]["value"] = _clean_reconstructed_name(
                current_item["name"].get("value", "")
            )
        if "unit" in current_item and isinstance(current_item.get("unit"), dict):
            current_item["unit"]["value"] = _clean_reconstructed_unit(
                current_item["unit"].get("value", "")
            )
        items.append(current_item)

    # Phase 15: Normalized unit fallback + field initialization
    expected_fields = {
        "name":       make_field("", CONFIDENCE_NOT_COMPUTED),
        "unit":       make_field("", CONFIDENCE_NOT_COMPUTED),
        "quantity":   make_field(0,  CONFIDENCE_NOT_COMPUTED),
        "unit_price": make_field(0,  CONFIDENCE_NOT_COMPUTED),
        "total":      make_field(0,  CONFIDENCE_NOT_COMPUTED),
        "discount":   make_field(0,  CONFIDENCE_NOT_COMPUTED),
        "line_tax":   make_field(0,  CONFIDENCE_NOT_COMPUTED),
        "row_total":  make_field(0,  CONFIDENCE_NOT_COMPUTED),
        "vat_rate":   make_field("", CONFIDENCE_NOT_COMPUTED),
    }
    
    # Get most common unit for fallback
    fallback_unit = _get_fallback_unit(items)
    
    for item_idx, itm in enumerate(items):
        for field, default in expected_fields.items():
            if field not in itm:
                itm[field] = copy.deepcopy(default)
        
        # Phase 15: Unit fallback logic
        if not itm["unit"]["value"] or not str(itm["unit"]["value"]).strip():
            itm["unit"]["value"] = fallback_unit
            itm["unit"]["confidence"] = 0.6  # Fallback confidence
        
        # Phase 15: Unit_price calculation fallback (using global _safe_numeric)
        _qty = _safe_numeric(itm["quantity"].get("value", 0))
        _total = _safe_numeric(itm["total"].get("value", 0))
        _price = _safe_numeric(itm["unit_price"].get("value", 0))
        
        if (_qty > 0 and _total > 0 and _price == 0):
            # Calculate unit_price = total / quantity
            calculated_price = _total / _qty
            itm["unit_price"]["value"] = round(calculated_price, 2)
            itm["unit_price"]["confidence"] = min(
                itm["quantity"].get("confidence", 0),
                itm["total"].get("confidence", 0)
            ) * 0.8  # Reduced confidence for calculated value

    # Phase 15: Numeric normalization with proper type handling
    for itm in items:
        # Quantity, total, discount: integer normalization
        for f in ["quantity", "total", "discount", "line_tax", "row_total"]:
            if f in itm and isinstance(itm[f], dict):
                val = itm[f].get("value", 0)
                itm[f]["value"] = _normalize_numeric(val)
        
        # Unit_price: float normalization with Vietnamese format handling
        if "unit_price" in itm and isinstance(itm["unit_price"], dict):
            val = itm["unit_price"].get("value", 0)
            itm["unit_price"]["value"] = _normalize_unit_price_value(val)

    return items


# ──────────────────────────────────────────────────────────────────
# STAGE 3 — Orchestrate triple inference
# ──────────────────────────────────────────────────────────────────
def stage3_inference(
    mm:      ModelManager,
    img_bgr: np.ndarray,
    words:   List[str],
    bboxes:  List[List[int]],
    log:     logging.Logger,
) -> Tuple[List[str], List[float], List[str], List[float], List[str], List[float], List[dict]]:
    """
    Run Header (single-pass), Table (single-pass), Footer (sliding window).
    Returns word-level labels and confidences for all three models.
    """
    pil_image = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))

    import concurrent.futures

    log.info("  Stage 3 — Multi-model Parallel Inference (Local CPU)")
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
        # 1. Header Task
        header_task = executor.submit(
            _run_single_model_inference,
            processor  = mm.header_processor,
            model      = mm.header_model,
            id2label   = mm.header_id2label,
            pil_image  = pil_image,
            words      = words,
            bboxes     = bboxes,
            device     = mm.device,
            log        = log,
            model_name = "header"
        )
        
        # 2. Table Task
        table_task = executor.submit(
            table_sliding_window_inference,
            model      = mm.table_model,
            processor  = mm.table_processor,
            id2label   = mm.table_id2label,
            pil_image  = pil_image,
            words      = words,
            bboxes     = bboxes,
            device     = mm.device,
            log        = log,
            model_name = "table"
        )
        
        # 3. Footer Task
        footer_task = executor.submit(
            sliding_window_inference,
            model      = mm.footer_model,
            processor  = mm.footer_processor,
            id2label   = mm.footer_id2label,
            pil_image  = pil_image,
            words      = words,
            bboxes     = bboxes,
            device     = mm.device,
            log        = log,
            model_name = "footer"
        )

        # Wait for all
        header_labels, header_confs = header_task.result()
        table_labels, table_confs   = table_task.result()
        footer_labels, footer_confs = footer_task.result()

    # Log reconstructed item count for validation
    sw_table_items = _reconstruct_items_from_labels(table_labels, table_confs, words)
    log.info(f"[TABLE SLIDING WINDOW DONE] Total items reconstructed: {len(sw_table_items)}")

    return header_labels, header_confs, table_labels, table_confs, footer_labels, footer_confs, sw_table_items


# ──────────────────────────────────────────────────────────────────
# STAGE 4 — Post-Processing (BIO repair + Engine + Merge)
# ──────────────────────────────────────────────────────────────────
def _should_force_vat_zero(invoice: dict) -> bool:
    """
    Only force VAT=0 when ALL 3 conditions are true:
    1. vat_amount confidence < 0.5 (model not confident)
    2. vat_amount value < 1000 VND (noise level)
    3. No vat_rate string found (e.g. "10%", "8%")
    """
    vat_obj  = invoice.get("vat_amount", {})
    vat_val  = vat_obj.get("value", 0)      if isinstance(vat_obj, dict) else vat_obj
    vat_conf = vat_obj.get("confidence", 0) if isinstance(vat_obj, dict) else 0

    vat_rate_obj = invoice.get("vat_rate", {})
    vat_rate = vat_rate_obj.get("value", "") if isinstance(vat_rate_obj, dict) else str(vat_rate_obj)

    try:
        # Regex to extract digits for numeric conversion
        import re
        vat_digits = re.sub(r'[^\d]', '', str(vat_val))
        vat_numeric = int(vat_digits) if vat_digits else 0
    except:
        vat_numeric = 0

    return (vat_conf < 0.5 and
            vat_numeric < 1_000 and
            not vat_rate.strip())


def _math_validate_items(items: list) -> None:
    """
    Validate mathematical consistency in normalized items.
    
    Check: quantity × unit_price - discount = total
    
    This is a post-normalization validation pass that logs warnings
    for inconsistencies but does NOT modify items.
    """
    for i, item in enumerate(items or []):
        if not isinstance(item, dict):
            continue
        
        try:
            qty   = item.get("quantity", {}).get("value", 0) or 0
            price = item.get("unit_price", {}).get("value", 0) or 0
            disc  = item.get("discount", {}).get("value", 0) or 0
            total = item.get("total", {}).get("value", 0) or 0
            
            # Only validate if we have numeric values
            if not (qty and price and total):
                continue
            
            expected = qty * price - disc
            
            # Check if math is consistent (allow 5% tolerance for rounding)
            if total > 0:
                ratio = expected / total
                if ratio < 0.95 or ratio > 1.05:
                    name = item.get("name", {}).get("value", f"item_{i}")
                    print(f"[MATH_WARN] {i}: '{name[:30]}' "
                          f"math_inconsist: {qty}×{price}-{disc}={expected} "
                          f"≠ total={total} (ratio={ratio:.2f})")
        except Exception as e:
            print(f"[MATH_WARN] Error validating item {i}: {e}")


def _normalize_sw_items(sw_items: list) -> list:
    """
    Normalize sw_items to standard pipeline schema.

    Handles ALL field formats:
      Format A: {"value": "clean string", "confidence": 0.999}   → use directly
      Format B: {"value": "{'value': '...', 'confidence': 0.999}", "confidence": 0}
                → parse inner dict, recover real value + confidence
      Format C: missing field → use zero/empty default

    Also computes total from qty × unit_price when total=0.
    """
    import ast, copy

    FIELD_DEFAULTS = {
        "name":       make_field("", CONFIDENCE_NOT_COMPUTED),
        "unit":       make_field("", CONFIDENCE_NOT_COMPUTED),
        "quantity":   make_field(0,  CONFIDENCE_NOT_COMPUTED),
        "unit_price": make_field(0,  CONFIDENCE_NOT_COMPUTED),
        "total":      make_field(0,  CONFIDENCE_NOT_COMPUTED),
        "discount":   make_field(0,  CONFIDENCE_NOT_COMPUTED),
        "line_tax":   make_field(0,  CONFIDENCE_NOT_COMPUTED),
        "row_total":  make_field(0,  CONFIDENCE_NOT_COMPUTED),
        "vat_rate":   make_field("", CONFIDENCE_NOT_COMPUTED),
    }

    def _unwrap(field_data, default):
        """Extract {value, confidence} using canonical safe_unwrap."""
        val, conf = safe_unwrap(field_data)
        if val is None:
            return copy.deepcopy(default)
        return make_field(val, conf)

    normalized = []
    for item in (sw_items or []):
        if not isinstance(item, dict):
            continue

        norm = {}
        for field, default in FIELD_DEFAULTS.items():
            norm[field] = _unwrap(item.get(field, default), default)

        # BUG 3 FIX: compute total from qty × price when total=0
        total_val = norm["total"]["value"]
        if (not total_val or total_val == 0):
            qty   = norm["quantity"]["value"]   or 0
            price = norm["unit_price"]["value"] or 0
            disc  = norm["discount"]["value"]   or 0
            if qty > 0 and price > 0:
                norm["total"]["value"] = int(qty * price) - int(disc)
                # Confidence = average of qty_conf and unit_price_conf
                # Reflects actual model quality, not a fixed guess
                _qty_conf   = norm["quantity"]["confidence"]   or 0.0
                _price_conf = norm["unit_price"]["confidence"] or 0.0
                if _qty_conf > 0 and _price_conf > 0:
                    norm["total"]["confidence"] = round((_qty_conf + _price_conf) / 2, 4)
                elif _qty_conf > 0:
                    norm["total"]["confidence"] = round(_qty_conf * 0.9, 4)
                elif _price_conf > 0:
                    norm["total"]["confidence"] = round(_price_conf * 0.9, 4)
                else:
                    norm["total"]["confidence"] = 0.0

        # ── BUG A FIX: detect unit/unit_price swap ──────────────
        # If unit.value looks like a number AND unit_price.value is 0,
        # the model mislabeled unit_price as unit → swap them back
        _unit_val   = norm["unit"]["value"]
        _price_val  = norm["unit_price"]["value"]
        _unit_conf  = norm["unit"]["confidence"]
        _price_conf = norm["unit_price"]["confidence"]

        def _looks_numeric(v) -> bool:
            """Return True if v looks like a price/number, not a unit label."""
            if isinstance(v, (int, float)) and v > 0:
                return True
            if isinstance(v, str):
                # Handle Vietnamese number formats:
                # "110.000,00" → strip dots and comma → "11000000" → numeric
                # "Cái" → not numeric
                cleaned = v.replace(".", "").replace(",", "").replace(":", "").strip()
                try:
                    n = int(cleaned)
                    return n > 100  # prices > 100, unit labels are short text
                except (ValueError, TypeError):
                    # Try float (e.g. "1.5")
                    try:
                        n = float(v.replace(",", "."))
                        return n > 100
                    except (ValueError, TypeError):
                        return False
            return False

        # ── Phase 15: ROBUST SWAP DETECTION ──
        if _looks_numeric(_unit_val):
            # Case A: unit.value is numeric (mislabeled price)
            if not _looks_numeric(_price_val) or _price_val == 0 or _price_conf == 0:
                _parsed_price = _normalize_unit_price_value(_unit_val)
                norm["unit_price"]["value"]      = _parsed_price
                norm["unit_price"]["confidence"] = _unit_conf
            # Always clear unit if it's numeric/price-like
            norm["unit"]["value"]            = ""
            norm["unit"]["confidence"]       = 0.0

        normalized.append(norm)

    # Phase 15: Final unit normalization
    for norm in normalized:
        if "unit" in norm and isinstance(norm["unit"], dict):
            norm["unit"]["value"] = _clean_reconstructed_unit(norm["unit"]["value"])
            
            # ── FIX 2: Default "Cái" for discrete items ────────────────
            if not norm["unit"]["value"]:
                name = (norm.get("name", {}).get("value", "") or "").lower()
                
                # Check expanded patterns against the name to recover missing units
                matched_heuristic = False
                for pattern, normalized_unit in UNIT_PATTERNS.items():
                    if re.search(pattern, name, re.IGNORECASE):
                        norm["unit"]["value"]      = normalized_unit
                        norm["unit"]["confidence"] = 0.5
                        matched_heuristic = True
                        break
                
                if not matched_heuristic:
                    # Only default "Cái" for clearly discrete countable items
                    price = _safe_numeric(norm.get("unit_price", {}).get("value", 0))
                    qty   = _safe_numeric(norm.get("quantity", {}).get("value", 0))
                    # Skip default if name suggests non-discrete item
                    non_discrete = any(w in name for w in ["lít","ml","kg","gram","m2","m²","tấn", "m", "lit"])
                    if price > 0 and qty > 0 and not non_discrete:
                        norm["unit"]["value"]      = "Cái"
                        norm["unit"]["confidence"] = 0.3

    
    # Phase 15: NUMERIC NORMALIZATION (like in _reconstruct_items_from_labels Part 3)
    for norm in normalized:
        # Quantity, total, discount: integer normalization
        for f in ["quantity", "total", "discount", "line_tax", "row_total"]:
            if f in norm and isinstance(norm[f], dict):
                val = norm[f].get("value", 0)
                # Inline normalization logic
                if isinstance(val, (int, float)):
                    norm[f]["value"] = int(val)
                elif isinstance(val, str):
                    cleaned = val.strip().replace('.', '').replace(',', '')
                    try:
                        norm[f]["value"] = int(cleaned)
                    except (ValueError, TypeError):
                        norm[f]["value"] = 0
                else:
                    norm[f]["value"] = 0
        # Unit_price: float normalization with Vietnamese format handling
        if "unit_price" in norm and isinstance(norm["unit_price"], dict):
            val = norm["unit_price"].get("value", 0)
            norm["unit_price"]["value"] = _normalize_unit_price_value(val)
    
    return normalized


def _merge_extraction_results(
    header_result: Dict,
    table_result:  Dict,
    footer_result: Dict,
    sw_items:      List[Dict] = None,
) -> Dict:
    """
    Deep-merge results from all three models with conflict detection (BUG 16).

    Priority (highest → lowest):
      header : seller, buyer, invoice metadata
      table  : items, totals (fills None header fields)
      footer : subtotal, vat_amount, total_amount
    """
    from src.invoice_schema import merge_with_conflict_log
    from src.utils.field_helpers import safe_unwrap
    merged = copy.deepcopy(header_result)
    conflicts = []

    # Fields to check for conflicts across all 3 models
    CONFLICT_FIELDS = [
        ("invoice.number",   "invoice", "number"),
        ("invoice.date",     "invoice", "date"),
        ("seller.tax_code",  "seller",  "tax_code"),
        ("seller.name",      "seller",  "name"),
        ("buyer.tax_code",   "buyer",   "tax_code"),
        ("buyer.name",       "buyer",   "name"),
    ]

    header_conf = header_result.get("confidence", {}).get("fields", {})
    table_conf  = table_result.get("confidence", {}).get("fields", {})
    footer_conf = footer_result.get("confidence", {}).get("fields", {})

    # 1. Compare Header vs Table
    for path, sec, key in CONFLICT_FIELDS:
        val_h, c_h = safe_unwrap((header_result.get(sec, {}) or {}).get(key))
        # If scalar in header, try confidence dict
        if c_h == 0 and path in header_conf: c_h = header_conf[path]
        
        val_t, c_t = safe_unwrap((table_result.get(sec, {}) or {}).get(key))
        if c_t == 0 and path in table_conf: c_t = table_conf[path]

        if val_t and val_h:
            _, _, conflict = merge_with_conflict_log(path, val_h, c_h, val_t, c_t)
            if conflict:
                conflicts.append(conflict)

    # 2. Merge table → header (non-destructive)
    if "invoice" in table_result:
        merged.setdefault("invoice", {})
        for k, v in table_result["invoice"].items():
            if v and merged["invoice"].get(k) in (None, "", 0):
                merged["invoice"][k] = v

    # 3. Items priority
    _items_source = sw_items if sw_items is not None else table_result.get("items", [])
    _normalize_sw_items(_items_source)
    _math_validate_items(_items_source)
    merged["items"] = _items_source

    if "confidence" in table_result:
        merged.setdefault("confidence", {"fields": {}})
        fields_conf = table_result.get("confidence", {}).get("fields", {})
        if isinstance(merged["confidence"].get("fields"), dict):
            merged["confidence"]["fields"].update(fields_conf)

    # 4. Compare Merged vs Footer
    for path, sec, key in CONFLICT_FIELDS:
        val_m, c_m = safe_unwrap((merged.get(sec, {}) or {}).get(key))
        # Confidence update logic from previous steps
        if c_m == 0 and path in merged.get("confidence", {}).get("fields", {}):
            c_m = merged["confidence"]["fields"][path]

        val_f, c_f = safe_unwrap((footer_result.get(sec, {}) or {}).get(key))
        if c_f == 0 and path in footer_conf: c_f = footer_conf[path]

        if val_f and val_m:
            _, _, conflict = merge_with_conflict_log(path, val_m, c_m, val_f, c_f)
            if conflict:
                conflicts.append(conflict)

    # 5. Merge footer → result (additive only)
    if "invoice" in footer_result:
        merged.setdefault("invoice", {})
        footer_inv = footer_result.get("invoice", {}) or {}
        for field in ["subtotal", "vat_amount", "total_amount", "vat_rate"]:
            current = merged["invoice"].get(field)
            current_val = current.get("value", 0) if isinstance(current, dict) else current
            footer_val = footer_inv.get(field)
            f_val = footer_val.get("value", 0) if isinstance(footer_val, dict) else footer_val
            if (not current_val or current_val == 0) and f_val and f_val != 0:
                merged["invoice"][field] = footer_val
        
        for k, v in footer_inv.items():
            if k not in ["subtotal", "vat_amount", "total_amount", "vat_rate"]:
                if v and merged["invoice"].get(k) in (None, "", 0):
                    merged["invoice"][k] = v

    if "confidence" in footer_result:
        merged.setdefault("confidence", {"fields": {}})
        f_conf = footer_result.get("confidence", {}).get("fields", {})
        if isinstance(merged["confidence"].get("fields"), dict):
            for k, v in f_conf.items():
                if k not in merged["confidence"]["fields"]:
                    merged["confidence"]["fields"][k] = v

    # Attach conflicts for serialization to pick up
    if conflicts:
        merged["_conflicts"] = conflicts

    return merged




def stage4_postprocess(
    mm:            ModelManager,
    words:         List[str],
    bboxes:        List[List[int]],
    header_labels: List[str],
    header_confs:  List[float],
    table_labels:  List[str],
    table_confs:   List[float],
    footer_labels: List[str],
    footer_confs:  List[float],
    log:           logging.Logger,
    sw_items:      List[Dict] = None,
) -> Dict:
    """
    Post-processing:
      1. BIO repair on all three label sequences
      2. InvoiceExtractionEngine on each sequence
      3. Apply clean_footer_value() to footer currency fields
      4. Three-way merge → strip_nested_prefixes → clean_company_fields
    """
    # ── Header ────────────────────────────────────────────────────
    repaired_header = repair_bio_sequence(header_labels, words=words, bboxes=bboxes)
    n_h = sum(1 for a, b in zip(header_labels, repaired_header) if a != b)
    if n_h:
        log.debug("  BIO repair (header): fixed %d label(s)", n_h)
    header_raw = mm.engine.process({
        "tokens":           words,
        "bboxes":           bboxes,
        "predicted_labels": repaired_header,
        "confidence":       header_confs,
    })

    # ── Table ─────────────────────────────────────────────────────
    repaired_table = repair_bio_sequence(table_labels, words=words, bboxes=bboxes)
    n_t = sum(1 for a, b in zip(table_labels, repaired_table) if a != b)
    if n_t:
        log.debug("  BIO repair (table): fixed %d label(s)", n_t)
    table_raw = mm.engine.process({
        "tokens":           words,
        "bboxes":           bboxes,
        "predicted_labels": repaired_table,
        "confidence":       table_confs,
    })

    # ── Footer ────────────────────────────────────────────────────
    repaired_footer = repair_bio_sequence(footer_labels, words=words, bboxes=bboxes)
    n_f = sum(1 for a, b in zip(footer_labels, repaired_footer) if a != b)
    if n_f:
        log.debug("  BIO repair (footer): fixed %d label(s)", n_f)
    footer_raw = mm.engine.process({
        "tokens":           words,
        "bboxes":           bboxes,
        "predicted_labels": repaired_footer,
        "confidence":       footer_confs,
    })

    # Apply clean_footer_value to footer currency fields
    # FIX 2: clean_footer_value() is now module-level, called directly
    if "invoice" in footer_raw:
        for k in ("subtotal", "vat_amount", "total_amount"):
            if k in footer_raw["invoice"]:
                footer_raw["invoice"][k] = clean_footer_value(footer_raw["invoice"][k])

    # ── Merge + clean ─────────────────────────────────────────────
    merged = _merge_extraction_results(header_raw, table_raw, footer_raw,
                                       sw_items=sw_items)
    result = strip_nested_prefixes(merged)
    result = clean_company_fields(result)

    # ### FIX 1 — tax_code cleaning
    for party in ("seller", "buyer"):
        if party in result and "tax_code" in result[party]:
            result[party]["tax_code"] = clean_tax_code(result[party]["tax_code"])

    if "items" in result:
        # _tokens already removed; just apply noise filter
        result["items"] = [
            item for item in result["items"]
            if not is_noise_item(item)
        ]

    # ### FIX 3 — amounts validation
    if result.get("invoice"):
        result["invoice"] = validate_and_fix_amounts(result["invoice"])
        result["invoice"] = _recover_vat_from_totals(
            result.get("invoice", {}),
            items=result.get("items", [])
        )

    # ### FIX 1 — Seller tax code contamination check
    def is_amount_not_taxcode(value: str, result: dict) -> bool:
        """Return True if this 'tax_code' is actually a money amount."""
        if not value:
            return False
        digits = value.replace('.', '').replace(',', '').replace('-', '')
        if not digits.isdigit():
            return False
        numeric = int(digits)
        # Check if this value matches any invoice amount
        inv = result.get("invoice", {})
        for field in ("subtotal", "total_amount", "vat_amount"):
            field_obj = inv.get(field, {})
            field_val = field_obj.get("value", 0) if isinstance(field_obj, dict) else field_obj
            try:
                if abs(int(str(field_val).replace('.','').replace(',','')) - numeric) < 10:
                    return True  # It's a money amount, not a tax code
            except:
                pass
        return False

    # After cleaning tax codes:
    for party in ("seller", "buyer"):
        tc_obj = result.get(party, {}).get("tax_code", {})
        if not tc_obj: continue
        tc_val = tc_obj.get("value", "") if isinstance(tc_obj, dict) else str(tc_obj)
        if is_amount_not_taxcode(tc_val, result):
            # Clear the contaminated tax_code
            if isinstance(tc_obj, dict):
                result[party]["tax_code"]["value"] = ""
                result[party]["tax_code"]["confidence"] = 0
            else:
                result[party]["tax_code"] = ""

    # ### FIX 4 — Forced zero VAT check (Revised BUG 1)
    if _should_force_vat_zero(result.get("invoice", {})):
        inv = result.get("invoice", {})
        total_obj = inv.get("total_amount", {})
        total_val = total_obj.get("value", 0) if isinstance(total_obj, dict) else total_obj
        
        # Force VAT to 0
        if isinstance(inv.get("vat_amount"), dict):
            result["invoice"]["vat_amount"]["value"] = 0
            result["invoice"]["vat_amount"]["_corrected"] = True
        else:
            result["invoice"]["vat_amount"] = 0
            
        # Align subtotal = total
        if total_val:
            if isinstance(inv.get("subtotal"), dict):
                result["invoice"]["subtotal"]["value"] = total_val
                result["invoice"]["subtotal"]["_corrected"] = True
            else:
                result["invoice"]["subtotal"] = total_val

    # ### FIX 2 — Clean company names from year/garbage prefixes
    def clean_company_name(name: str) -> str:
        if not name:
            return name
        # Remove leading year patterns: "1991:", "2005 -", "(2010)"
        name = re.sub(r'^\(?\d{4}\)?[\s:\-\.]+', '', name).strip()
        # Remove leading punctuation garbage
        name = re.sub(r'^[\s:;\-\.\|]+', '', name).strip()
        # Remove known label prefixes
        prefixes = [
            "VỊ:", "ĐƠN VỊ:", "SELLER:", "BUYER:",
            "NGƯỜI MUA:", "NGƯỜI BÁN:", "ĐƠN VỊ BÁN:",
            "ĐƠN VỊ MUA:", "THE ", "TEN:", "NAME:"
        ]
        name_upper = name.upper()
        for p in prefixes:
            if name_upper.startswith(p):
                name = name[len(p):].strip()
                break
        return name.strip(": ").strip()

    # Apply to seller and buyer names:
    for party in ("seller", "buyer"):
        for field in ("name", "full_name"):
            obj = result.get(party, {}).get(field, {})
            if isinstance(obj, dict) and obj.get("value"):
                obj["value"] = clean_company_name(obj["value"])
            elif isinstance(obj, str) and obj:
                result[party][field] = clean_company_name(obj)

    # ### FIX 1B — Subtotal picks up tax_code digits check
    def _subtotal_is_taxcode(result: dict) -> bool:
        """Check if subtotal value suspiciously matches a tax code."""
        inv = result.get("invoice", {})
        sub_obj = inv.get("subtotal", {})
        sub_val = str(sub_obj.get("value", "") if isinstance(sub_obj, dict) else sub_obj)
        sub_digits = re.sub(r'\D', '', sub_val)
        
        for party in ("seller", "buyer"):
            tc_obj = result.get(party, {}).get("tax_code", {})
            tc_val = str(tc_obj.get("value", "") if isinstance(tc_obj, dict) else tc_obj)
            tc_digits = re.sub(r'\D', '', tc_val)
            if tc_digits and tc_digits in sub_digits:
                return True
        return False

    if _subtotal_is_taxcode(result):
        total_obj = result.get("invoice", {}).get("total_amount", {})
        total_val = total_obj.get("value", 0) if isinstance(total_obj, dict) else total_obj
        if isinstance(result["invoice"].get("subtotal"), dict):
            result["invoice"]["subtotal"]["value"] = total_val
            result["invoice"]["subtotal"]["_corrected"] = True
        else:
            result["invoice"]["subtotal"] = total_val

    log.debug(
        "  Stage4 done — header_non_O=%d  table_non_O=%d  footer_non_O=%d",
        sum(1 for l in repaired_header if l != "O"),
        sum(1 for l in repaired_table  if l != "O"),
        sum(1 for l in repaired_footer if l != "O"),
    )
    
    return result


# ──────────────────────────────────────────────────────────────────
# STAGE 5 — Output Formatting + Field Validators
# ──────────────────────────────────────────────────────────────────
def _normalize_date(value: Any) -> Optional[str]:
    """Normalize date string → ISO 8601 (YYYY-MM-DD)."""
    if not value:
        return None
    s = str(value).strip()
    if not s:
        return None

    vi_pattern = re.compile(
        r"""(?:ng[aà]y\s+)?
            (\d{1,2})\s+
            th[aá]ng\s+
            (\d{1,2})\s+
            n[aă]m\s+
            (\d{4})
        """,
        re.IGNORECASE | re.VERBOSE,
    )
    m = vi_pattern.search(s)
    if m:
        try:
            d, mo, y = int(m.group(1)), int(m.group(2)), int(m.group(3))
            return datetime(y, mo, d).strftime("%Y-%m-%d")
        except ValueError:
            pass

    for fmt in ("%d/%m/%Y", "%d-%m-%Y", "%d.%m.%Y", "%Y-%m-%d", "%d/%m/%y"):
        try:
            return datetime.strptime(s, fmt).strftime("%Y-%m-%d")
        except ValueError:
            continue

    return s


def _normalize_amount(value: Any) -> Optional[float]:
    """Parse Vietnamese number format. "5.000.000" → 5000000.0"""
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    s = str(value).strip()
    s = re.sub(r"[^\d.,-]", "", s)
    s = s.replace(".", "").replace(",", ".")
    try:
        return float(s)
    except ValueError:
        return None


def _normalize_tax_code(value: Any) -> Optional[str]:
    if not value:
        return None
    # FIX: handle confidence-field dict — extract inner value only
    if isinstance(value, dict) and "value" in value:
        inner = value.get("value", "")
        return str(inner).strip() if inner else None
    s = str(value).strip()
    return s if s else None

def _normalize_vat_rate(value: Any) -> Optional[str]:
    if value is None:
        return None
    s = str(value).strip()
    if not s:
        return None
    s_clean = s.rstrip("%").strip()
    m = re.search(r"(\d+(?:[.,]\d+)?)", s_clean)
    if m:
        num_str = m.group(1).replace(",", ".")
        try:
            num = float(num_str)
            if num == int(num):
                return str(int(num))
            return str(num)
        except ValueError:
            pass
    return s_clean if s_clean else None


def stage5_format(serialized: Dict, filename: str, elapsed_ms: float, validation: Optional[Dict] = None) -> Dict:
    """Format, normalize, and enrich the final extraction output."""
    result = copy.deepcopy(serialized)

    if "invoice" in result and isinstance(result["invoice"], dict):
        _date_field = result["invoice"].get("date")
        _date_conf  = _date_field.get("confidence", 0.95) \
                    if isinstance(_date_field, dict) else 0.95
        _date_norm  = _normalize_date(_date_field)
        result["invoice"]["date"] = make_field(_date_norm, _date_conf) \
                                if _date_norm else make_field(None, None)
    if "invoice" in result and isinstance(result["invoice"], dict):
        for amount_field in ("total_amount", "vat_amount", "subtotal"):
            if amount_field in result["invoice"]:
                field_obj = result["invoice"][amount_field]
                # Extract value from dict format if needed
                if isinstance(field_obj, dict):
                    raw_val = field_obj.get("value", 0)
                    normalized = _normalize_amount(raw_val)
                    if normalized is not None:
                        field_obj["value"] = normalized
                else:
                    normalized = _normalize_amount(field_obj)
                    if normalized is not None:
                        result["invoice"][amount_field] = normalized

    if "invoice" in result and isinstance(result["invoice"], dict):
        if "vat_rate" in result["invoice"]:
            result["invoice"]["vat_rate"] = _normalize_vat_rate(
                result["invoice"].get("vat_rate")
            )

    for item in result.get("items", []):
        if not isinstance(item, dict):
            continue
        for item_field in ("quantity", "unit_price", "discount", "line_tax", "total", "row_total"):
            if item_field in item:
                field_obj = item[item_field]
                # Extract value from dict format if needed
                if isinstance(field_obj, dict):
                    raw_val = field_obj.get("value", 0)
                    normalized = _normalize_amount(raw_val)
                    if normalized is not None:
                        # Update only the "value" field, preserve confidence
                        field_obj["value"] = normalized
                else:
                    # Raw value, normalize directly
                    normalized = _normalize_amount(field_obj)
                    if normalized is not None:
                        item[item_field] = normalized
        if "vat_rate" in item:
            item["vat_rate"] = _normalize_vat_rate(item.get("vat_rate"))

    for party in ("seller", "buyer"):
        if party in result and isinstance(result[party], dict):
            _tc_field = result[party].get("tax_code")
            _tc_conf  = _tc_field.get("confidence", 0.95) \
                        if isinstance(_tc_field, dict) else 0.95
            _tc_norm  = _normalize_tax_code(_tc_field)
            result[party]["tax_code"] = make_field(_tc_norm, _tc_conf) \
                                        if _tc_norm else make_field(None, None)

    final = result

    final["_validation"] = validation or {}

    # Bug 3: Derive status from _validation before returning
    if not final.get("status"):
        _v = final.get("_validation", {})
        _errors   = _v.get("errors",   [])
        _warnings = _v.get("warnings", [])
        if _errors:
            final["status"] = "partial"
        elif _warnings:
            final["status"] = "partial"
        else:
            final["status"] = "ok"

    return final


# ──────────────────────────────────────────────────────────────────
# Pipeline Orchestrator
# ──────────────────────────────────────────────────────────────────
def run_full_pipeline(
    image_bytes: bytes,
    filename:    str,
    log:         logging.Logger,
    met:         Metrics,
) -> Dict:
    """Orchestrate all 5 pipeline stages for a single invoice file."""
    t_start  = time.monotonic()
    uid      = uuid.uuid4().hex[:10]
    temp_dir = Config.TEMP_DIR / uid
    temp_dir.mkdir(parents=True, exist_ok=True)

    mm = ModelManager.get()

    try:
        t   = time.monotonic()
        ext = stage0_validate(filename, image_bytes)
        met.record_stage("0_validate", (time.monotonic() - t) * 1000)
        log.info("Stage 0 ✅ validate  ext=%s  size=%dB", ext, len(image_bytes))

        t = time.monotonic()
        pages, _ = stage1_preprocess(image_bytes, ext, temp_dir, uid, log)
        met.record_stage("1_preprocess", (time.monotonic() - t) * 1000)
        log.info("Stage 1 ✅ preprocess  pages=%d", len(pages))

        img_bgr, w, h = pages[0]

        # ── Stage 3+4 (PRIMARY): Gemini Flash ──────────────────────────────
        raw_result = None
        
        if GEMINI_ENABLED:
            try:
                log.info("Attempting extraction with Gemini Flash...")
                import cv2 as _cv2
                _encode_success, _jpeg_buf = _cv2.imencode(
                    ".jpg", img_bgr,
                    [_cv2.IMWRITE_JPEG_QUALITY, 92]
                )
                if not _encode_success:
                    raise RuntimeError("Failed to encode img_bgr to JPEG for Gemini")
                
                _gemini_bytes = _jpeg_buf.tobytes()
                
                # Internal retry logic for Gemini transient errors (429, etc)
                MAX_RETRIES = 2
                RETRY_DELAYS = [5, 15]
                for attempt in range(MAX_RETRIES + 1):
                    try:
                        raw_result = GEMINI_EXTRACTOR.extract(
                            image_bytes=_gemini_bytes,
                            ocr_words=None, # Defer OCR context — strictly multimodal
                            mime_type="image/jpeg",
                        )
                        if raw_result:
                            log.info("Stage 3+4 ✅ Gemini done  items=%d", len(raw_result.get("items", [])))
                            break
                    except Exception as gemini_err:
                        err_str = str(gemini_err)
                        is_rate_limit = "429" in err_str or "RESOURCE_EXHAUSTED" in err_str
                        if is_rate_limit and attempt < MAX_RETRIES:
                            delay = RETRY_DELAYS[attempt]
                            log.warning("[Gemini] Rate limit (attempt %d) — retrying in %ds", attempt + 1, delay)
                            time.sleep(delay)
                            continue
                        raise
                
            except Exception as e:
                log.warning(f"⚠️ Gemini extraction failed ({e}). Falling back to local LayoutLMv3 models...")
                raw_result = None

        # --- Local Model Extraction Logic (Fallback / Default) ---
        if raw_result is None:
            log.info("Executing local LayoutLMv3 pipeline (Stage 2, 3, 4)...")

            # ── Stage 2: OCR (Only if LayoutLMv3 fallback required) ─────
            t = time.monotonic()
            _, words, bboxes = stage2_ocr(mm, img_bgr, w, h, temp_dir, uid, log)
            met.record_stage("2_ocr", (time.monotonic() - t) * 1000)
            log.info("Stage 2 ✅ ocr  words=%d", len(words))

            # ── Stage 3: Inference ──────────────────────────────────────
            t = time.monotonic()
            (header_labels, header_confs,
             table_labels,  table_confs,
             footer_labels, footer_confs,
             sw_items) = stage3_inference(mm, img_bgr, words, bboxes, log)
            met.record_stage("3_inference", (time.monotonic() - t) * 1000)
            
            log.debug(
                "  Stage 3 ✅ inference  header_non_O=%d  table_non_O=%d  footer_non_O=%d",
                sum(1 for l in header_labels if l != "O"),
                sum(1 for l in table_labels  if l != "O"),
                sum(1 for l in footer_labels if l != "O"),
            )

            # ── Stage 4: Post-processing ───────────────────────────────
            t = time.monotonic()
            raw_result = stage4_postprocess(
                mm, words, bboxes,
                header_labels, header_confs,
                table_labels,  table_confs,
                footer_labels, footer_confs,
                log,
                sw_items=sw_items,
            )
            met.record_stage("4_postprocess", (time.monotonic() - t) * 1000)
            log.info("Stage 4 ✅ postprocess done")
        # ── end stage 3+4 ───────────────────────────────────────────────────

        # ── Stage 5: Final Serialization & Validation ──────────────────────
        # Canonical entry point for consistent JSON structure (Bug 3 fixed)

        # 1. Base Serialization (Financial logic + Base Confidence)
        final = serialize_invoice_result(raw_result)
        
        # 2. Business Rule Validation
        validator  = InvoiceValidator(
            vat_tolerance  = Config.VAT_TOLERANCE,
            min_confidence = Config.MIN_CONFIDENCE,
        )
        validation = validator.validate_all(final)

        # 3. Merge Orchestrator findings (conflicts from model merging)
        # raw_result here is the output of Gemini or Stage 4 (merged raw)
        conflicts = (raw_result or {}).get("_conflicts") or []
        for conflict in conflicts:
            validation["warnings"].append(conflict)

        if validation["errors"]:
            log.warning("Validation errors: %s", validation["errors"])
        if validation["warnings"]:
            log.debug("Validation warnings: %s", validation["warnings"])

        # 4. Attach validation results to schema
        final["_validation"] = validation
        
        # 5. Penalize Document Confidence (Bug 9 & 20)
        # Re-compute based on base confidence AND validation hits
        final["document_confidence"] = compute_document_confidence(final)

        elapsed_ms = (time.monotonic() - t_start) * 1000
        # Final formatting and enrichment (metadata, etc)
        final = stage5_format(final, filename, elapsed_ms, validation=validation)
        log.info("Stage 5 ✅ format  total=%.0fms  status=%s",
                 elapsed_ms, final.get("status", "?"))

        return final

    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)
        gc.collect()


# ═══════════════════════════════════════════════════════════════════
# 10.  FLASK  APPLICATION  BOOTSTRAP
# ═══════════════════════════════════════════════════════════════════
Config.init_dirs()

logger    = setup_logging(Config.LOG_DIR)
metrics   = Metrics()
limiter   = RateLimiter(rpm=Config.RATE_LIMIT_RPM)
job_queue = AsyncJobQueue(num_workers=1)

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = Config.MAX_UPLOAD_BYTES

# Thư mục lưu ảnh preview cho Dashboard batch
PREVIEW_FOLDER = Config.BASE_DIR / "static" / "temp_previews"
PREVIEW_FOLDER.mkdir(parents=True, exist_ok=True)


@app.before_request
def _before():
    g.request_id = uuid.uuid4().hex[:8]
    g.t_start    = time.monotonic()


@app.after_request
def _after(response):
    response.headers["X-Request-ID"]       = getattr(g, "request_id", "-")
    response.headers["X-Pipeline-Version"] = "phase8-v4-triple-active"
    return response


def rate_limited(fn):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        if Config.RATE_LIMIT_ENABLED:
            ip = request.remote_addr or "unknown"
            if not limiter.is_allowed(ip):
                logger.warning("Rate limited | ip=%s", ip)
                return jsonify({
                    "error":      "Too many requests — please wait before retrying.",
                    "code":       429,
                    "request_id": getattr(g, "request_id", "-"),
                }), 429
        return fn(*args, **kwargs)
    return wrapper


def requires_model(fn):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        mm = ModelManager.get()
        if not mm.ready:
            try:
                mm.load(logger)
            except Exception as exc:
                return jsonify({
                    "error":      f"Model failed to initialize: {exc}",
                    "code":       503,
                    "request_id": getattr(g, "request_id", "-"),
                }), 503
        return fn(*args, **kwargs)
    return wrapper


# ═══════════════════════════════════════════════════════════════════
# 11.  ROUTES
# ═══════════════════════════════════════════════════════════════════

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/v1/extract", methods=["POST"])
@rate_limited
@requires_model
def extract_sync():
    req_id = getattr(g, "request_id", "-")

    if "file" not in request.files:
        return jsonify({"error": "Missing 'file' field in multipart/form-data.", "request_id": req_id}), 400

    f = request.files["file"]
    if not f or not f.filename:
        return jsonify({"error": "Empty filename.", "request_id": req_id}), 400

    filename    = f.filename
    image_bytes = f.read()
    file_type   = Path(filename).suffix.lower().lstrip(".")

    logger.info("SYNC | req=%s  file=%s  size=%dB", req_id, filename, len(image_bytes))

    t0 = time.monotonic()
    try:
        result  = run_full_pipeline(image_bytes, filename, logger, metrics)
        latency = (time.monotonic() - t0) * 1000
        metrics.record_request(success=True, latency_ms=latency, file_type=file_type)
        logger.info("SYNC OK | req=%s  %.0fms", req_id, latency)

        return jsonify({
            "success":    True,
            "request_id": req_id,
            "latency_ms": round(latency, 1),
            "data":       result,
        })

    except ValidationError as exc:
        metrics.record_request(success=False, latency_ms=0, file_type=file_type)
        metrics.record_error("validation_error")
        logger.warning("SYNC validation error | req=%s  %s", req_id, exc)
        return jsonify({"error": str(exc), "request_id": req_id}), exc.http_status

    except PipelineError as exc:
        latency = (time.monotonic() - t0) * 1000
        metrics.record_request(success=False, latency_ms=latency, file_type=file_type)
        metrics.record_error(f"pipeline_{exc.stage}")
        logger.error("SYNC pipeline error | req=%s  stage=%s  %s", req_id, exc.stage, exc)
        return jsonify({"error": str(exc), "stage": exc.stage, "request_id": req_id}), 422

    except Exception as exc:
        latency = (time.monotonic() - t0) * 1000
        metrics.record_request(success=False, latency_ms=latency, file_type=file_type)
        metrics.record_error(type(exc).__name__)
        logger.error("SYNC FAILED | req=%s  %s", req_id, exc, exc_info=True)
        tb = traceback.format_exc() if app.debug else None
        return jsonify({"error": str(exc), "traceback": tb, "request_id": req_id}), 500


@app.route("/api/v1/extract/async", methods=["POST"])
@rate_limited
@requires_model
def extract_async():
    req_id = getattr(g, "request_id", "-")

    if "file" not in request.files:
        return jsonify({"error": "Missing 'file' field.", "request_id": req_id}), 400

    f = request.files["file"]
    if not f or not f.filename:
        return jsonify({"error": "Empty filename.", "request_id": req_id}), 400

    filename    = f.filename
    image_bytes = f.read()

    try:
        stage0_validate(filename, image_bytes)
    except ValidationError as exc:
        return jsonify({"error": str(exc), "request_id": req_id}), exc.http_status

    try:
        job    = Job(job_id=uuid.uuid4().hex, filename=filename)
        job_id = job_queue.submit(job, image_bytes)
    except RuntimeError as exc:
        return jsonify({"error": str(exc), "request_id": req_id}), 503

    logger.info("ASYNC submitted | req=%s  job=%s  file=%s", req_id, job_id, filename)

    return jsonify({
        "success":    True,
        "request_id": req_id,
        "job_id":     job_id,
        "poll_url":   f"/api/v1/jobs/{job_id}",
        "status":     JobStatus.PENDING,
    }), 202


@app.route("/api/v1/jobs/<job_id>", methods=["GET"])
def get_job(job_id: str):
    job = job_queue.get_job(job_id)
    if job is None:
        return jsonify({
            "error": f"Job '{job_id}' not found or expired (TTL: {Config.JOB_TTL_SECONDS}s)."
        }), 404
    return jsonify(job.to_dict())


@app.route("/api/v1/health")
def health():
    return jsonify({
        "status": "ok", 
        "ts": datetime.now(timezone.utc).isoformat(),
        "gemini": {
            "enabled": GEMINI_ENABLED,
            "model":   "gemini-2.5-flash" if GEMINI_ENABLED else "disabled",
        }
    })


@app.route("/api/v1/ready")
def ready():
    mm = ModelManager.get()
    if mm.ready:
        return jsonify({
            "status":            "ready",
            "model_header":      Path(Config.MODEL_PATH_HEADER).name,
            "model_table":       Path(Config.MODEL_PATH_TABLE).name,
            "model_footer":      Path(Config.MODEL_PATH_FOOTER).name,
            "device":            Config.DEVICE,
            "api_version":       "phase8-v4-triple-active",
        })
    elif mm.init_error:
        return jsonify({"status": "error", "error": mm.init_error}), 503
    else:
        return jsonify({"status": "loading"}), 503


@app.route("/api/v1/metrics")
def get_metrics():
    return jsonify(metrics.snapshot())


# ──────────────────────────────────────────────────────────────────
# UI — Multi-image/PDF Upload (không dashboard)
# ──────────────────────────────────────────────────────────────────
def _flatten_result(pipeline_result: dict, filename: str) -> dict:
    """Chuyển output run_full_pipeline() thành dict phẳng cho template."""
    def _val(obj, key, default=None):
        field = (obj or {}).get(key, {})
        if isinstance(field, dict):
            return field.get("value", default)
        return field if field is not None else default

    invoice = pipeline_result.get("invoice") or {}
    seller  = pipeline_result.get("seller")  or {}
    buyer   = pipeline_result.get("buyer")   or {}
    items   = pipeline_result.get("items")   or []

    return {
        "filename":      filename,
        "invoice_type":  invoice.get("type", "BAN_HANG"),
        "invoice_date":  _val(invoice, "date", ""),
        "vendor_name":   _val(seller, "name", "Không xác định"),
        "buyer_name":    _val(buyer,  "name", "Không xác định"),
        "vendor_tax_id": _val(seller, "tax_code", ""),
        "net_amount":    float(_val(invoice, "subtotal",     0) or 0),
        "vat_amount":    float(_val(invoice, "vat_amount",   0) or 0),
        "total_amount":  float(_val(invoice, "total_amount", 0) or 0),
        "status":        pipeline_result.get("status", "ok"),
        "confidence":    round(float(pipeline_result.get("document_confidence", 0) or 0), 3),
        "line_items":    items,   # "items" conflicts with dict.items() in Jinja2
        "image_url":     f"/static/temp_previews/{filename}",
    }


def _aggregate_batch(results: list) -> dict:
    """Tổng hợp thống kê cho Dashboard Chart.js."""
    grand_total = sum(r.get("total_amount", 0) for r in results)
    total_vat   = sum(r.get("vat_amount",   0) for r in results)
    timeline_map: dict = {}
    for r in results:
        date = r.get("invoice_date") or "Unknown"
        if not date or date in ("None", "null", ""):
            date = "Unknown"
        timeline_map[date] = timeline_map.get(date, 0) + r.get("total_amount", 0)
    return {
        "count_total":   len(results),
        "count_gtgt":    sum(1 for r in results if r.get("invoice_type") == "GTGT"),
        "count_banhang": sum(1 for r in results if r.get("invoice_type") != "GTGT"),
        "grand_total":   grand_total,
        "total_vat":     total_vat,
        "chart_timeline": [
            {"date": d, "total": timeline_map[d]}
            for d in sorted(k for k in timeline_map if k != "Unknown")
        ],
    }


@app.route("/extract-multi", methods=["POST"])
@requires_model
def extract_multi():
    """
    Upload nhiều ảnh/PDF cùng lúc → trích xuất từng file →
    trả về trang kết quả (không phải dashboard, không có chart).
    """
    req_id = getattr(g, "request_id", "-")
    files  = request.files.getlist("files")
    if not files:
        return jsonify({"error": "Không có file nào được upload."}), 400

    results = []
    errors  = []

    for f in files[:20]:  # tối đa 20 file một lần
        if not f or not f.filename:
            continue
        filename    = f.filename
        image_bytes = f.read()

        # Lưu ảnh preview
        try:
            preview_path = PREVIEW_FOLDER / filename
            preview_path.write_bytes(image_bytes)
        except Exception:
            pass

        try:
            pipeline_result = run_full_pipeline(image_bytes, filename, logger, metrics)
            flat = _flatten_result(pipeline_result, filename)
            results.append(flat)
            logger.info("MULTI | %s → status=%s  total=%.0f",
                        filename, flat["status"], flat["total_amount"])
        except ValidationError as ve:
            errors.append({"filename": filename, "error": str(ve)})
        except Exception as exc:
            errors.append({"filename": filename, "error": str(exc)})
            logger.error("MULTI error | %s: %s", filename, exc, exc_info=True)

    if not results and errors:
        return jsonify({"error": "Không trích xuất được file nào.", "details": errors}), 400

    return render_template("results_multi.html", results=results, errors=errors)


# ──────────────────────────────────────────────────────────────────
# UI — ZIP Batch Upload → Dashboard
# ──────────────────────────────────────────────────────────────────
@app.route("/upload-zip", methods=["POST"])
@requires_model
def upload_zip():
    """
    Upload file ZIP chứa các ảnh hóa đơn (VD: hóa đơn tháng) →
    trích xuất từng ảnh → render Dashboard với Chart.js.
    """
    if "file" not in request.files:
        return jsonify({"error": "Thiếu file ZIP."}), 400

    zip_file = request.files["file"]
    if not zip_file or not zip_file.filename:
        return jsonify({"error": "Tên file rỗng."}), 400

    # Dọn preview cũ
    for old in PREVIEW_FOLDER.iterdir():
        try:
            if old.is_file(): old.unlink()
        except Exception:
            pass

    results = []
    errors  = []

    try:
        with zipfile.ZipFile(zip_file, "r") as zf:
            image_entries = [
                name for name in zf.namelist()
                if name.lower().endswith((".png", ".jpg", ".jpeg"))
                and not name.startswith("__MACOSX")
            ][:20]

            if not image_entries:
                return jsonify({"error": "Không tìm thấy ảnh (.png/.jpg/.jpeg) trong ZIP."}), 400

            logger.info("[ZIP] %d ảnh từ: %s", len(image_entries), zip_file.filename)

            for entry in image_entries:
                clean_name = Path(entry).name
                if not clean_name:
                    continue
                try:
                    img_bytes = zf.read(entry)
                    # Lưu ảnh preview
                    (PREVIEW_FOLDER / clean_name).write_bytes(img_bytes)
                    # Chạy pipeline y hệt /api/v1/extract
                    pipeline_result = run_full_pipeline(
                        image_bytes = img_bytes,
                        filename    = clean_name,
                        log         = logger,
                        met         = metrics,
                    )
                    flat = _flatten_result(pipeline_result, clean_name)
                    results.append(flat)
                    logger.info("[ZIP] ✅ %s → total=%.0f  status=%s",
                                clean_name, flat["total_amount"], flat["status"])
                except ValidationError as ve:
                    errors.append(str(ve))
                    logger.warning("[ZIP] ⚠️  %s: %s", clean_name, ve)
                except Exception as exc:
                    errors.append(f"{clean_name}: {exc}")
                    logger.error("[ZIP] ❌ %s: %s", clean_name, exc, exc_info=True)

    except zipfile.BadZipFile:
        return jsonify({"error": "File không phải định dạng ZIP hợp lệ."}), 400
    except Exception as exc:
        logger.error("[ZIP] System error: %s", exc, exc_info=True)
        return jsonify({"error": f"Lỗi hệ thống: {exc}"}), 500

    if not results:
        return jsonify({"error": "Không trích xuất được kết quả nào.", "details": errors}), 400

    summary = _aggregate_batch(results)
    summary["errors"] = errors
    logger.info("[ZIP] Hoàn tất: %d/%d ảnh", len(results), len(image_entries))

    return render_template("dashboard.html", summary=summary, details=results)



# ──────────────────────────────────────────────────────────────────
# Swagger UI
# ──────────────────────────────────────────────────────────────────
SWAGGER_SPEC = {
    "openapi": "3.0.0",
    "info": {
        "title":       "Invoice Extraction API",
        "version":     "1.0.0",
        "description": "Phase 7 & 8 — LayoutLMv3 Vietnamese invoice field extraction (triple-model)",
    },
    "paths": {
        "/api/v1/extract": {
            "post": {
                "summary": "Extract invoice fields (synchronous)",
                "requestBody": {
                    "required": True,
                    "content": {
                        "multipart/form-data": {
                            "schema": {
                                "type": "object",
                                "properties": {"file": {"type": "string", "format": "binary"}},
                                "required": ["file"],
                            }
                        }
                    },
                },
                "responses": {
                    "200": {"description": "Extraction result"},
                    "400": {"description": "Validation error"},
                    "422": {"description": "Pipeline error"},
                    "429": {"description": "Rate limit exceeded"},
                    "500": {"description": "Server error"},
                    "503": {"description": "Model not ready"},
                },
            }
        },
        "/api/v1/extract/async": {
            "post": {
                "summary": "Submit async extraction job",
                "requestBody": {
                    "required": True,
                    "content": {
                        "multipart/form-data": {
                            "schema": {
                                "type": "object",
                                "properties": {"file": {"type": "string", "format": "binary"}},
                                "required": ["file"],
                            }
                        }
                    },
                },
                "responses": {
                    "202": {"description": "Job submitted"},
                    "400": {"description": "Validation error"},
                    "503": {"description": "Queue full"},
                },
            }
        },
        "/api/v1/jobs/{job_id}": {
            "get": {
                "summary": "Poll async job result",
                "parameters": [
                    {"name": "job_id", "in": "path", "required": True, "schema": {"type": "string"}}
                ],
                "responses": {
                    "200": {"description": "Job object"},
                    "404": {"description": "Job not found or expired"},
                },
            }
        },
        "/api/v1/health":  {"get": {"summary": "Liveness probe"}},
        "/api/v1/ready":   {"get": {"summary": "Readiness probe"}},
        "/api/v1/metrics": {"get": {"summary": "Runtime metrics"}},
    },
}


@app.route("/docs")
def swagger_ui():
    spec_json = json.dumps(SWAGGER_SPEC)
    html = f"""<!DOCTYPE html>
<html>
<head>
  <title>Invoice API Docs</title>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <link rel="stylesheet"
        href="https://cdnjs.cloudflare.com/ajax/libs/swagger-ui/5.17.14/swagger-ui.min.css"/>
</head>
<body>
<div id="swagger-ui"></div>
<script src="https://cdnjs.cloudflare.com/ajax/libs/swagger-ui/5.17.14/swagger-ui-bundle.min.js"></script>
<script>
SwaggerUIBundle({{
  spec: {spec_json},
  dom_id: '#swagger-ui',
  deepLinking: true,
  layout: 'BaseLayout',
}});
</script>
</body>
</html>"""
    return html, 200, {"Content-Type": "text/html"}


# ═══════════════════════════════════════════════════════════════════
# 12.  GLOBAL  ERROR  HANDLERS
# ═══════════════════════════════════════════════════════════════════
@app.errorhandler(400)
def err_400(e):
    return jsonify({"error": "Bad Request", "detail": str(e)}), 400

@app.errorhandler(404)
def err_404(e):
    return jsonify({"error": "Endpoint not found."}), 404

@app.errorhandler(405)
def err_405(e):
    return jsonify({"error": "Method not allowed."}), 405

@app.errorhandler(413)
def err_413(e):
    mb = Config.MAX_UPLOAD_BYTES // 1_048_576
    return jsonify({"error": f"File too large. Maximum: {mb} MB."}), 413

@app.errorhandler(500)
def err_500(e):
    logger.error("Unhandled 500: %s", e, exc_info=True)
    return jsonify({"error": "Internal server error."}), 500


# ═══════════════════════════════════════════════════════════════════
# 13.  STARTUP
# ═══════════════════════════════════════════════════════════════════
def _warmup_model():
    try:
        logger.info("⏳ Warming up models in background thread...")
        ModelManager.get().load(logger)
        logger.info("🔥 All 3 models warm — ready to serve!")
    except Exception as exc:
        logger.error("❌ Background model warmup FAILED: %s", exc)


if __name__ == "__main__":
    banner = f"""
╔══════════════════════════════════════════════════════╗
║  🧾  Invoice Extraction API  —  Phase 7 & 8          ║
╠══════════════════════════════════════════════════════╣
║  Header : {Path(Config.MODEL_PATH_HEADER).name:<41}║
║  Table  : {Path(Config.MODEL_PATH_TABLE).name:<41}║
║  Footer : {Path(Config.MODEL_PATH_FOOTER).name:<41}║
║  Device : {Config.DEVICE:<41}║
║  Rate   : {str(Config.RATE_LIMIT_RPM) + ' req/min per IP':<41}║
╠══════════════════════════════════════════════════════╣
║  UI      http://localhost:{Config.PORT}/                    ║
║  Sync    POST /api/v1/extract                        ║
║  Async   POST /api/v1/extract/async                  ║
║  Docs    http://localhost:{Config.PORT}/docs                ║
║  Health  http://localhost:{Config.PORT}/api/v1/health       ║
║  Metrics http://localhost:{Config.PORT}/api/v1/metrics      ║
╚══════════════════════════════════════════════════════╝"""
    try:
        print(banner)
    except UnicodeEncodeError:
        # Fallback for Windows CP1252 consoles
        print("--- INVOICE OCR SERVER STARTING ---")

    threading.Thread(target=_warmup_model, daemon=True).start()

    app.run(
        host         = Config.HOST,
        port         = Config.PORT,
        debug        = Config.DEBUG,
        use_reloader = False,   # ⚠️ MUST be False — reloader causes double model load
        threaded     = True,
    )
