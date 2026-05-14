# -*- coding: utf-8 -*-
"""
Phase 6 - Test Trained Model on Real Invoices
Complete Inference + Evaluation Pipeline

Supports:
- PNG images (direct)
- PDF files (converted to images first)
- Vietnamese invoice extraction
- Automatic evaluation against ground truth
"""

import json
import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import cv2
import numpy as np
from PIL import Image
import torch
from transformers import LayoutLMv3Processor, LayoutLMv3ForTokenClassification

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Import from your project
from src.run_ocr import OCRRunner
from src.engine import InvoiceExtractionEngine
from bio_repair_inference import (
    prepare_words_and_bboxes,
    repair_bio_sequence,
    strip_nested_prefixes,   # ✅ dùng nested-aware version
    clean_company_fields,
)

from src.engine.conflict_resolver import ConflictResolver
from src.engine.validator import NumericValidator


# ==========================================================
# PDF UTILITIES
# ==========================================================

def pdf_to_images(pdf_path: Path, output_dir: Path) -> List[Path]:
    """
    Convert PDF to images (one per page) at 200 DPI.

    Args:
        pdf_path:   Path to PDF file
        output_dir: Directory to save images

    Returns:
        List of image paths
    """
    try:
        import fitz  # PyMuPDF
    except ImportError:
        raise ImportError("PyMuPDF not installed. Install: pip install pymupdf")

    output_dir.mkdir(parents=True, exist_ok=True)
    doc         = fitz.open(pdf_path)
    image_paths = []

    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        mat  = fitz.Matrix(200 / 72, 200 / 72)   # 200 DPI
        pix  = page.get_pixmap(matrix=mat)

        image_path = output_dir / f"{pdf_path.stem}_page{page_num + 1}.png"
        pix.save(str(image_path))
        image_paths.append(image_path)

    doc.close()
    return image_paths


# ==========================================================
# HELPER — FLAT/NESTED NAVIGATE
# ==========================================================

def _detect_nested(sample: Dict) -> bool:
    """Detect nếu engine output là nested dict."""
    return any(isinstance(v, dict) for v in sample.values())


def _get_field(d: Dict, *keys: str, is_nested: bool = True):
    """
    Navigate dict bằng keys.
    nested : {"seller": {"name": ...}}       → _get_field(d, "seller", "name")
    flat   : {"seller_name": ...}            → _get_field(d, "seller", "name")
    """
    if is_nested:
        cur = d
        for k in keys:
            cur = cur.get(k, {}) if isinstance(cur, dict) else None
        return cur
    else:
        return d.get("_".join(keys))


# ==========================================================
# HELPER — CLEAN FOOTER CURRENCY VALUES
# FIX 3: Module-level function, NOT inside loop
# ==========================================================

def clean_footer_value(raw_value) -> Optional[str]:
    """
    Clean garbled footer values by isolating monetary patterns.
    Prevents tax codes or IDs from merging with total amounts.
    """
    import re
    if not raw_value:
        return None
    val_str = str(raw_value).strip()
    
    # 1. Regex to find monetary strings: matches "1.080.000", "550,000", "150000"
    monetary_pattern = re.compile(r'(\d{1,3}([\.,]\d{3})+|\d{4,})')
    matches = monetary_pattern.findall(val_str)
    matches = [m[0] if isinstance(m, tuple) else m for m in matches]
    
    if not matches:
        return val_str.split()[-1] if val_str.split() else val_str
        
    # 2. Filter out matches that look like tax codes (10+ digits without separators)
    valid_amounts = []
    for m in matches:
        digits_only = re.sub(r'\D', '', m)
        if len(digits_only) >= 10 and '.' not in m and ',' not in m:
            continue
        valid_amounts.append(m)
        
    return valid_amounts[-1] if valid_amounts else matches[-1]


# ==========================================================
# SLIDING WINDOW INFERENCE
# Handles LayoutLMv3 512-token limit for long Vietnamese invoices
# CHUNK_SIZE=150 words (~450 subwords), STRIDE=75 words (50% overlap)
# FIX 4: Module-level function, placed ONCE above RealInvoiceTester
# ==========================================================

def sliding_window_inference(
    model,
    processor,
    id2label: Dict,
    words: List[str],
    bboxes: List[List[int]],
    device: str,
    pil_image,
) -> Tuple[List[str], List[float]]:
    """
    Sliding window inference bypassing the LayoutLMv3 512-token limit.
    Ensures long Vietnamese invoices do not have their footer truncated.

    Args:
        model:      LayoutLMv3ForTokenClassification
        processor:  LayoutLMv3Processor
        id2label:   {int → label_str} mapping
        words:      OCR word list
        bboxes:     Normalized bboxes [0, 1000]
        device:     "cpu" or "cuda"
        pil_image:  PIL.Image of the invoice

    Returns:
        (labels, confidences) — word-level, length == len(words)
    """
    CHUNK_SIZE = 150   # ~450 subwords, safely under 512
    STRIDE     = 75    # 50% overlap
    MAX_CHUNKS = 8     # hard limit

    n_words = len(words)
    if n_words == 0:
        return [], []

    final_labels = ["O"] * n_words
    final_confs  = [0.0]  * n_words

    # ── Build chunk index list ────────────────────────────────
    chunks = []
    start  = 0
    while start < n_words and len(chunks) < MAX_CHUNKS:
        end = min(start + CHUNK_SIZE, n_words)
        chunks.append((start, end))
        if end == n_words:
            break
        start += STRIDE

    # ── Inference per chunk ───────────────────────────────────
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

        # ── Aggregate subwords → word level ──────────────────
        # FIX 1: Use FIRST subword label (authoritative in LayoutLMv3)
        #        Use MEAN confidence for overlap resolution
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
                    # FIX 1: first subword is authoritative
                    chunk_word_labels.append(id2label[temp_preds[0]])
                    chunk_word_confs.append(float(np.mean(temp_confs)))
                current_word = word_id
                temp_preds   = [pred_id]
                temp_confs   = [conf]
            else:
                temp_preds.append(pred_id)
                temp_confs.append(conf)

        # Flush last word
        if temp_preds:
            # FIX 1 (flush): first subword is authoritative
            chunk_word_labels.append(id2label[temp_preds[0]])
            chunk_word_confs.append(float(np.mean(temp_confs)))

        non_o_chunk = sum(1 for l in chunk_word_labels if l != "O")
        non_o_total += non_o_chunk
        token_count  = sum(1 for wid in word_ids if wid is not None)
        print(f"   [CHUNK {i+1}/{len(chunks)}] words[{start_idx}:{end_idx}]"
              f" → tokens≈{token_count} → non-O labels: {non_o_chunk}")

        # ── Overlap Resolution & Merge ────────────────────────
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
                # Conflict: different real labels → higher confidence wins
                conflicts += 1
                if conf > curr_conf:
                    final_labels[global_i] = lbl
                    final_confs [global_i] = conf

            elif curr_lbl == lbl:
                # Same label: keep higher confidence
                if conf > curr_conf:
                    final_confs[global_i] = conf

            # Both O: keep higher confidence (cosmetic, no functional impact)
            elif curr_lbl == "O" and lbl == "O":
                if conf > curr_conf:
                    final_confs[global_i] = conf

        if conflicts:
            print(f"   [MERGE] Chunk {i+1}: {conflicts} overlap conflict(s) resolved by confidence")

    print(f"   [SLIDING WINDOW DONE] Total non-O labels: {non_o_total}")
    return final_labels, final_confs


# ==========================================================
# HELPER — DATE NORMALIZATION
# ==========================================================

def _normalize_date(value: Optional[str]) -> Optional[str]:
    """
    Normalize date string to ISO 8601 (YYYY-MM-DD).
    Handles Vietnamese format, standard formats, and noisy strings.
    """
    if not value:
        return value
    import re
    from datetime import datetime
    s = str(value).strip()

    # 1. Vietnamese: "Ngày 20 tháng 09 năm 2022"
    vi_pattern = re.compile(r'(\d{1,2})\s+th[aá]ng\s+(\d{1,2})\s+n[aă]m\s+(\d{4})', re.IGNORECASE)
    match = vi_pattern.search(s)
    if match:
        d, m, y = match.groups()
        try:
            return f"{int(y):04d}-{int(m):02d}-{int(d):02d}"
        except: pass

    # 2. Standard formats
    for fmt in ("%d/%m/%Y", "%d-%m-%Y", "%d.%m.%Y", "%Y-%m-%d", "%d/%m/%y"):
        try:
            return datetime.strptime(s, fmt).strftime("%Y-%m-%d")
        except ValueError:
            continue

    # 3. Noise recovery: extract first valid 4-digit year
    year_match = re.search(r'\b(19|20)\d{2}\b', s)
    if year_match:
        return year_match.group(0)

    return s


# ==========================================================
# HELPER — CONFLICT RESOLUTION
# ==========================================================

def resolve_model_conflicts(
    words: List[str],
    header_labels: List[str], header_confs: List[float],
    table_labels:  List[str], table_confs:  List[float],
) -> Tuple[List[str], List[str]]:
    """
    Resolves overlaps where Header and Table models predict different entities
    for the same token. Prioritizes global context for metadata.
    """
    final_header = list(header_labels)
    final_table  = list(table_labels)
    
    for i, (hl, hc, tl, tc) in enumerate(zip(header_labels, header_confs, table_labels, table_confs)):
        if hl != "O" and tl != "O":
            h_entity = hl.split("-")[-1]
            t_entity = tl.split("-")[-1]
            
            # 1. Header wins for INVOICE metadata (Number, Date, etc.)
            if h_entity in ("INVOICE_NUMBER", "INVOICE_DATE", "INVOICE_SYMBOL", "INVOICE_TYPE"):
                if hc > 0.6:
                    final_table[i] = "O"
                elif tc > hc + 0.35: # Table model needs huge confidence advantage to override
                    final_header[i] = "O"
                else:
                    final_table[i] = "O"
            
            # 2. Table wins for ITEM fields if Header prediction is low confidence
            elif t_entity.startswith("ITEM_"):
                if tc > 0.8:
                    final_header[i] = "O"
                else:
                    if hc > tc: final_table[i] = "O"
                    else: final_header[i] = "O"
    
    return final_header, final_table


# ==========================================================
# HELPER — DATA QUALITY & FILTERING
# ==========================================================

def is_valid_currency(val) -> bool:
    if val is None:
        return False
    cleaned = str(val).strip().replace('.', '').replace(',', '')
    if not cleaned.isdigit():
        return False
    numeric = int(cleaned)
    # Valid VND range: 1,000 to 999,999,999,999
    return 1_000 <= numeric <= 999_999_999_999


def _is_positive(val) -> bool:
    """Safely check if a value (str or number) is > 0."""
    if not val:
        return False
    try:
        cleaned = str(val).replace('.', '').replace(',', '').strip()
        return float(cleaned) > 0
    except (ValueError, TypeError):
        return False


def filter_noise_items(items: list) -> list:
    import re
    cleaned = []
    for itm in items:
        name = str(itm.get("name", "")).strip()
        
        # Keep if has prices/qty
        has_vals = any(
            _is_positive(itm.get(k)) 
            for k in ["quantity", "unit_price", "total"]
        )

        # Filter single-word items that are clearly noise
        if len(name.split()) == 1 and not has_vals:
            continue

        # Rule 1: Too short
        if len(name) < 4:
            continue
        
        # Rule 2: Contains tax code pattern (10-15 digits)
        digits_only = re.sub(r'\D', '', name)
        if len(digits_only) >= 10:
            continue
        
        # Rule 3: Looks like an address fragment
        address_keywords = ["ngõ", "số ", "đường", "phường", 
                             "quận", "tỉnh", "tp.", "p."]
        is_address = any(kw in name.lower() for kw in address_keywords)
        if is_address:
            continue
        
        # Rule 4: Label-only garbage (ends with colon)
        if name.endswith(":"):
            continue
        
        if has_vals:
            cleaned.append(itm)
            continue

        # Rule 5: Relaxation — Keep long descriptive names (service descriptions)
        if name_conf >= 0.80 and len(name) >= 10:
            # Must be at least 2 words to avoid single-word fragments
            if len(name.split()) >= 2:
                cleaned.append(itm)
    
    return cleaned


# ==========================================================
# REAL INVOICE TESTER
# FIX 4: Single class definition (removed duplicate empty shell)
# ==========================================================

class RealInvoiceTester:
    """
    Test trained LayoutLMv3 models on real Vietnamese invoices.

    Triple-model pipeline:
    1. Load invoice (PNG or PDF)
    2. Run OCR (PaddleOCR + VietOCR)
    3. Header model inference  → seller / buyer / invoice-header fields
    4. Table model inference   → item rows / totals / VAT
    5. Footer model inference  → subtotal / vat_amount / total_amount (sliding window)
    6. Merge results + field extraction (Engine)
    7. Evaluation (vs ground truth)
    """

    def __init__(
        self,
        header_model_path: str,
        table_model_path:  str,
        footer_model_path: str,
        device: str = "cpu",
    ):
        self.header_model_path = Path(header_model_path)
        self.table_model_path  = Path(table_model_path)
        self.footer_model_path = Path(footer_model_path)
        self.device            = device

        # ── OCR ──────────────────────────────────────────────
        print("🔄 Initializing OCR...")
        self.ocr_runner = OCRRunner(
            paddle_lang        = "vi",
            vietocr_model_name = "vgg_transformer",
            device             = device,
        )

        # ── Header model ──────────────────────────────────────
        print(f"🔄 Loading header model from {header_model_path}...")
        self.header_processor = LayoutLMv3Processor.from_pretrained(
            str(self.header_model_path), apply_ocr=False,
        )
        self.header_model = LayoutLMv3ForTokenClassification.from_pretrained(
            str(self.header_model_path)
        )
        self.header_model.to(device)
        self.header_model.eval()

        with open(self.header_model_path / "label2id.json", encoding="utf-8") as f:
            self.header_label2id = json.load(f)
        with open(self.header_model_path / "id2label.json", encoding="utf-8") as f:
            self.header_id2label = {int(k): v for k, v in json.load(f).items()}
        print(f"   Header classes: {len(self.header_id2label)}")

        # ── Table model ───────────────────────────────────────
        print(f"🔄 Loading table model from {table_model_path}...")
        self.table_processor = LayoutLMv3Processor.from_pretrained(
            str(self.table_model_path), apply_ocr=False,
        )
        self.table_model = LayoutLMv3ForTokenClassification.from_pretrained(
            str(self.table_model_path)
        )
        self.table_model.to(device)
        self.table_model.eval()

        with open(self.table_model_path / "label2id.json", encoding="utf-8") as f:
            self.table_label2id = json.load(f)
        with open(self.table_model_path / "id2label.json", encoding="utf-8") as f:
            self.table_id2label = {int(k): v for k, v in json.load(f).items()}
        print(f"   Table classes: {len(self.table_id2label)}")

        # ── Footer model ──────────────────────────────────────
        print(f"🔄 Loading footer model from {footer_model_path}...")
        self.footer_processor = LayoutLMv3Processor.from_pretrained(
            str(self.footer_model_path), apply_ocr=False,
        )
        self.footer_model = LayoutLMv3ForTokenClassification.from_pretrained(
            str(self.footer_model_path)
        )
        self.footer_model.to(device)
        self.footer_model.eval()

        with open(self.footer_model_path / "label2id.json", encoding="utf-8") as f:
            self.footer_label2id = json.load(f)
        with open(self.footer_model_path / "id2label.json", encoding="utf-8") as f:
            self.footer_id2label = {int(k): v for k, v in json.load(f).items()}
        print(f"   Footer classes: {len(self.footer_id2label)}")

        # ── Engine ────────────────────────────────────────────
        print("🔄 Initializing extraction engine...")
        self.engine = InvoiceExtractionEngine(
            vat_tolerance  = 0.05,
            min_confidence = 0.5,
        )

        print("✅ Tester initialized successfully!\n")

    # ----------------------------------------------------------
    def _infer_single_model(
        self,
        processor,
        model,
        id2label: Dict,
        image,
        words:      List[str],
        bboxes:     List[List[int]],
        model_name: str = "model",
    ) -> Tuple[List[str], List[float]]:
        """
        Run one LayoutLMv3 model (single pass, no sliding window).
        Used for header and table models — invoices fit in 512 tokens
        for these fields since they appear near the top of the document.

        FIX 2: Aligned subword aggregation with sliding_window_inference:
                Uses temp_preds[0] (first subword) as authoritative label.
        """
        encoding = processor(
            images         = image,
            text           = words,
            boxes          = bboxes,
            truncation     = True,
            padding        = True,
            return_tensors = "pt",
        )
        word_ids           = encoding.word_ids(batch_index=0)
        encoding_on_device = {k: v.to(self.device) for k, v in encoding.items()}

        with torch.no_grad():
            outputs     = model(**encoding_on_device)
            predictions = outputs.logits.argmax(-1).squeeze(0)
            probs       = torch.softmax(outputs.logits, dim=-1)
            confidences = probs.max(-1).values.squeeze(0)

        word_level_labels: List[str]   = []
        word_level_confs:  List[float] = []
        current_word                   = None
        temp_preds:        List[int]   = []
        temp_confs:        List[float] = []

        for idx, word_id in enumerate(word_ids):
            if word_id is None:
                continue
            pred_id = predictions[idx].item()
            conf    = confidences[idx].item()

            if word_id != current_word:
                if current_word is not None:
                    # FIX 2: first subword is authoritative (aligned with sliding_window)
                    word_level_labels.append(id2label[temp_preds[0]])
                    word_level_confs.append(float(np.mean(temp_confs)))
                current_word = word_id
                temp_preds   = [pred_id]
                temp_confs   = [conf]
            else:
                temp_preds.append(pred_id)
                temp_confs.append(conf)

        # Flush last word
        if temp_preds:
            # FIX 2 (flush): first subword is authoritative
            word_level_labels.append(id2label[temp_preds[0]])
            word_level_confs.append(float(np.mean(temp_confs)))

        non_o = sum(1 for l in word_level_labels if l != "O")
        print(f"   [{model_name}] {len(word_level_labels)} words → {non_o} non-O labels")
        return word_level_labels, word_level_confs

    # ----------------------------------------------------------
    def _merge_results(
        self,
        header_result: Dict,
        table_result:  Dict,
        footer_result: Dict,
    ) -> Dict:
        """
        Deep-merge header, table, and footer extraction results.

        Priority (highest → lowest):
          header fields : seller, buyer, invoice metadata
          table  fields : items, totals (overrides header where None)
          footer fields : subtotal, vat_amount, total_amount
                          ONLY fills fields still None/empty/0 after header+table
        """
        import copy
        merged = copy.deepcopy(header_result)

        # ── Merge table into header ───────────────────────────
        if "invoice" in table_result:
            if "invoice" not in merged:
                merged["invoice"] = {}
            for k, v in table_result["invoice"].items():
                if v is not None and merged["invoice"].get(k) is None:
                    merged["invoice"][k] = v

        merged["items"] = table_result.get("items", [])

        if "confidence" in table_result:
            merged.setdefault("confidence", {"fields": {}})
            merged["confidence"]["fields"].update(
                table_result.get("confidence", {}).get("fields", {})
            )

        # ── Merge footer into result (additive only) ──────────
        footer_invoice = footer_result.get("invoice", {})
        for field in ("subtotal", "vat_amount", "total_amount"):
            current = merged.get("invoice", {}).get(field)
            if not current or current == 0:
                candidate = footer_invoice.get(field)
                if candidate and is_valid_currency(candidate):
                    # Apply cleaning before assignment
                    candidate = clean_footer_value(candidate)
                    if "invoice" not in merged:
                        merged["invoice"] = {}
                    merged["invoice"][field] = candidate

        if "confidence" in footer_result:
            merged.setdefault("confidence", {"fields": {}})
            for k, v in footer_result.get("confidence", {}).get("fields", {}).items():
                if k not in merged["confidence"]["fields"]:
                    merged["confidence"]["fields"][k] = v

        return merged

    # ----------------------------------------------------------
    def process_image(self, image_path: Path, temp_dir: Path = None) -> Dict:
        print(f"📄 Processing: {image_path.name}")

        # ── 0. Normalize image ────────────────────────────────
        from image_preprocessor import normalize_invoice_image

        print("   0/5 Normalizing image...")
        img_bgr, w, h = normalize_invoice_image(
            image_path  = image_path,
            output_path = None,
        )

        if temp_dir is None:
            temp_dir = image_path.parent / "_normalized_tmp"
        temp_dir = Path(temp_dir)
        temp_dir.mkdir(parents=True, exist_ok=True)

        normalized_path = temp_dir / f"norm_{image_path.name}"
        cv2.imwrite(str(normalized_path), img_bgr)

        # ── 1. OCR ───────────────────────────────────────────
        print("   1/5 Running OCR...")
        dets = self.ocr_runner.detect_text_paddleocr(img_bgr)
        ocr_result = self.ocr_runner.merge_results(
            image_name = image_path.name,
            width      = w,
            height     = h,
            detections = dets,
            img_bgr    = img_bgr,
        )

        if not ocr_result["lines"]:
            return {"error": "No OCR text detected"}

        # ── 2. Prepare words + bbox ───────────────────────────
        print("   2/5 Preparing input tokens...")
        words, bboxes = prepare_words_and_bboxes(ocr_result, w, h)

        if not words:
            return {"error": "No tokens after OCR"}

        pil_image = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))

        # ── 3a. Header model inference ────────────────────────
        print("   3a/5 Header model inference...")
        header_labels, header_confs = self._infer_single_model(
            processor  = self.header_processor,
            model      = self.header_model,
            id2label   = self.header_id2label,
            image      = pil_image,
            words      = words,
            bboxes     = bboxes,
            model_name = "header",
        )
        header_labels = repair_bio_sequence(header_labels, words=words, bboxes=bboxes)

        # ── 3b. Table model inference ─────────────────────────
        print("   3b/5 Table model inference...")
        table_labels, table_confs = self._infer_single_model(
            processor  = self.table_processor,
            model      = self.table_model,
            id2label   = self.table_id2label,
            image      = pil_image,
            words      = words,
            bboxes     = bboxes,
            model_name = "table",
        )
        table_labels = repair_bio_sequence(table_labels, words=words, bboxes=bboxes)

        # ── 3c. Footer model inference (Sliding Window) ───────
        print("   3c/5 Footer model inference (Sliding Window)...")
        footer_labels, footer_confs = sliding_window_inference(
            model     = self.footer_model,
            processor = self.footer_processor,
            id2label  = self.footer_id2label,
            words     = words,
            bboxes    = bboxes,
            device    = self.device,
            pil_image = pil_image,
        )
        footer_labels = repair_bio_sequence(footer_labels, words=words, bboxes=bboxes)

        # ── 3d. Conflict Resolution ───────────────────────────
        print("   3d/5 Resolving model conflicts...")
        header_labels, table_labels = resolve_model_conflicts(
            words, header_labels, header_confs, table_labels, table_confs
        )

        # ── Debug label sample ────────────────────────────────
        print("\n===== DEBUG LABELS SAMPLE (first 40 tokens) =====")
        for w_tok, h_lbl, t_lbl, f_lbl in list(
            zip(words, header_labels, table_labels, footer_labels)
        )[:40]:
            print(f"  {w_tok:25} | H:{h_lbl:20} | T:{t_lbl:20} | F:{f_lbl}")
        print("==================================================\n")

        # ── 4. Engine extraction ──────────────────────────────
        print("   4/5 Extracting structured fields...")
        header_raw = self.engine.process({
            "tokens":           words,
            "bboxes":           bboxes,
            "predicted_labels": header_labels,
            "confidence":       header_confs,
        })
        table_raw = self.engine.process({
            "tokens":           words,
            "bboxes":           bboxes,
            "predicted_labels": table_labels,
            "confidence":       table_confs,
        })
        footer_raw = self.engine.process({
            "tokens":           words,
            "bboxes":           bboxes,
            "predicted_labels": footer_labels,
            "confidence":       footer_confs,
        })

        # DEBUG: print raw footer result
        print(f"[DEBUG FOOTER RAW] {json.dumps(footer_raw.get('invoice', {}), ensure_ascii=False)}")

        # ── 5. Merge + clean ──────────────────────────────────
        print("   5/5 Merging and cleaning results...")
        merged = self._merge_results(header_raw, table_raw, footer_raw)
        result = strip_nested_prefixes(merged)
        result = clean_company_fields(result)

        # DEBUG: print after merge result
        print(f"[DEBUG AFTER MERGE] {json.dumps(result.get('invoice', {}), ensure_ascii=False)}")

        # FIX 3: Strip buyer.name prefixes
        prefixes = ["VỊ:", "KẾ TỔNG:", "BUYER:", "(BUYER)", "SELLER:",
                    "ĐƠN VỊ MUA HÀNG:", "ĐƠN VỊ:", "NGƯỜI MUA:"]
        for party in ["seller", "buyer"]:
            if party in result and "name" in result[party]:
                name = str(result[party]["name"])
                for p in prefixes:
                    if name.upper().startswith(p):
                        name = name[len(p):].strip()
                result[party]["name"] = name.strip(": ")

        # FIX 4: Truncate invoice.type
        if result.get("invoice", {}).get("type"):
            val = result["invoice"]["type"]
            for keyword in ["Mẫu", "Form", "Ký", "Số:"]:
                if keyword in val:
                    val = val.split(keyword)[0].strip()
            result["invoice"]["type"] = val

        # FIX 5: Filter noise items
        if "items" in result:
            result["items"] = filter_noise_items(result["items"])

        # BUG 2 FIX: Normalize date field
        if result.get("invoice", {}).get("date"):
            result["invoice"]["date"] = _normalize_date(
                result["invoice"]["date"]
            )

        # Fix A — Strip buyer.full_name garbage
        GARBAGE_NAMES = {"(buyer)", "buyer", "seller", "(seller)", 
                         "người mua", "người bán", "đơn vị mua"}
        for party in ["seller", "buyer"]:
            full_name = result.get(party, {}).get("name", "") # evaluation uses 'name'
            if full_name and str(full_name).strip().lower().strip("()") in GARBAGE_NAMES:
                result[party]["name"] = None
            
            # Also check 'full_name' if present
            fn_key = result.get(party, {}).get("full_name", "")
            if fn_key and str(fn_key).strip().lower().strip("()") in GARBAGE_NAMES:
                result[party]["full_name"] = None

        # Fix B — Strip leading punctuation from payment_method
        pm = result.get("invoice", {}).get("payment_method", "")
        if pm:
            result["invoice"]["payment_method"] = str(pm).lstrip(": ").strip()

        print("\n===== ENGINE OUTPUT =====")
        # Final UTF-8 Encoding Fix for Terminal
        try:
            output = json.dumps(result, ensure_ascii=False, indent=2)
            if sys.platform == "win32":
                import io
                sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
            print(output)
        except Exception as e:
            print(f"Error printing: {e}")
            print(json.dumps(result, indent=2))
        print("=========================\n")

        print("   ✅ Done!\n")
        return result

    # ----------------------------------------------------------
    def process_pdf(self, pdf_path: Path, temp_dir: Path) -> List[Dict]:
        """
        Process PDF invoice (may have multiple pages).

        Returns:
            List of extracted data (one per page)
        """
        print(f"📄 Converting PDF: {pdf_path.name}")
        image_paths = pdf_to_images(pdf_path, temp_dir)
        print(f"   Converted to {len(image_paths)} page(s)\n")
        return [self.process_image(p) for p in image_paths]

    # ----------------------------------------------------------
    def evaluate(
        self,
        predictions:   List[Dict],
        ground_truths: List[Dict],
    ) -> Dict:
        """
        Evaluate predictions against ground truth.

        ✅ Tự detect engine output format (nested vs flat) và navigate đúng.
        """
        print("=" * 70)
        print("📊 EVALUATION")
        print("=" * 70)

        sample    = next((p for p in predictions if "error" not in p), {})
        is_nested = _detect_nested(sample)
        print(f"\n   Engine output format : {'nested' if is_nested else 'flat'}")

        fields = [
            ("seller.name",          "seller",  "name"),
            ("seller.tax_code",      "seller",  "tax_code"),
            ("buyer.name",           "buyer",   "name"),
            ("buyer.tax_code",       "buyer",   "tax_code"),
            ("invoice.number",       "invoice", "number"),
            ("invoice.date",         "invoice", "date"),
            ("invoice.total_amount", "invoice", "total_amount"),
            # ✅ Footer fields now included in evaluation
            ("invoice.subtotal",     "invoice", "subtotal"),
            ("invoice.vat_amount",   "invoice", "vat_amount"),
        ]

        results = {
            "total_invoices": len(predictions),
            "engine_format":  "nested" if is_nested else "flat",
            "per_field":      {},
            "overall":        {},
            "failures":       [],
        }

        for field_path, *keys in fields:
            correct = 0
            total   = 0

            for i, (pred, gt) in enumerate(zip(predictions, ground_truths)):
                if "error" in pred:
                    continue

                pred_val = _get_field(pred, *keys, is_nested=is_nested)
                gt_val   = _get_field(gt,   *keys, is_nested=is_nested)

                # Skip footer fields if ground truth doesn't have them
                if gt_val is None:
                    continue

                total += 1
                if self._compare_values(pred_val, gt_val):
                    correct += 1
                else:
                    results["failures"].append({
                        "invoice_index": i,
                        "field":         field_path,
                        "predicted":     str(pred_val),
                        "ground_truth":  str(gt_val),
                    })

            acc = correct / total if total > 0 else 0
            results["per_field"][field_path] = {
                "correct":  correct,
                "total":    total,
                "accuracy": acc,
            }

        total_correct = sum(f["correct"] for f in results["per_field"].values())
        total_fields  = sum(f["total"]   for f in results["per_field"].values())
        results["overall"] = {
            "accuracy": total_correct / total_fields if total_fields > 0 else 0,
            "correct":  total_correct,
            "total":    total_fields,
        }

        print(f"\n📊 Overall Accuracy : {results['overall']['accuracy']:.2%}")
        print(f"   Correct / Total  : {results['overall']['correct']} / {results['overall']['total']}")

        print(f"\n📋 Per-field Accuracy:")
        for field, m in results["per_field"].items():
            status = "✅" if m["accuracy"] >= 0.9 else "⚠️" if m["accuracy"] >= 0.7 else "❌"
            print(f"   {status} {field:35} {m['accuracy']:>6.1%}  ({m['correct']}/{m['total']})")

        if results["failures"]:
            print(f"\n❌ Failures ({len(results['failures'])} total) — showing first 5:")
            for failure in results["failures"][:5]:
                print(f"   Invoice {failure['invoice_index']}: {failure['field']}")
                print(f"      Predicted : {failure['predicted']}")
                print(f"      Expected  : {failure['ground_truth']}")

        print("\n" + "=" * 70)
        return results

    # ----------------------------------------------------------
    def _compare_values(self, pred, gt) -> bool:
        """Compare predicted and ground truth values."""
        if pred is None or gt is None:
            return False
        if isinstance(pred, str) and isinstance(gt, str):
            pred_c = " ".join(pred.strip().split()).lower()
            gt_c   = " ".join(gt.strip().split()).lower()
            return pred_c == gt_c
        if isinstance(pred, (int, float)) and isinstance(gt, (int, float)):
            tolerance = max(1, abs(gt) * 0.01)
            return abs(pred - gt) <= tolerance
        return pred == gt


# ==========================================================
# MAIN
# ==========================================================

def main():
    import sys, io
    if sys.platform == "win32":
        sys.stdout = io.TextIOWrapper(
            sys.stdout.buffer, encoding='utf-8', errors='replace'
        )

    # Resolve paths relative to project root (portable for Docker/Cloud)
    project_root = PROJECT_ROOT

    MODEL_PATH_HEADER = str(
        project_root / "model_final_layoutlmv3" / "model_final_header" / "final_model"
    )
    MODEL_PATH_TABLE = str(
        project_root / "model_final_layoutlmv3" / "model_final_table" / "final_model"
    )
    MODEL_PATH_FOOTER = str(
        project_root / "model_final_layoutlmv3" / "model_final_footer" / "final_model"
    )
    REAL_INVOICES_DIR = str(project_root / "inference" / "real_invoices")
    GROUND_TRUTH_FILE = str(project_root / "inference" / "labels" / "labels.json")
    OUTPUT_FILE = str(project_root / "inference" / "results" / "real_test_results.json")
    TEMP_DIR = str(project_root / "inference" / "tmp")

    # Optional overrides via environment variables for deployment
    MODEL_PATH_HEADER = str(Path(os.getenv("MODEL_PATH_HEADER", MODEL_PATH_HEADER)))
    MODEL_PATH_TABLE = str(Path(os.getenv("MODEL_PATH_TABLE", MODEL_PATH_TABLE)))
    MODEL_PATH_FOOTER = str(Path(os.getenv("MODEL_PATH_FOOTER", MODEL_PATH_FOOTER)))
    REAL_INVOICES_DIR = str(Path(os.getenv("REAL_INVOICES_DIR", REAL_INVOICES_DIR)))
    GROUND_TRUTH_FILE = str(Path(os.getenv("GROUND_TRUTH_FILE", GROUND_TRUTH_FILE)))
    OUTPUT_FILE = str(Path(os.getenv("OUTPUT_FILE", OUTPUT_FILE)))
    TEMP_DIR = str(Path(os.getenv("TEMP_DIR", TEMP_DIR)))

    Path(REAL_INVOICES_DIR).mkdir(parents=True, exist_ok=True)
    Path(TEMP_DIR).mkdir(parents=True, exist_ok=True)
    Path(OUTPUT_FILE).parent.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("PHASE 6 — REAL INVOICE TESTING (Triple-Model Pipeline)")
    print("  Models: header ✅  table ✅  footer ✅")
    print("=" * 70)

    tester = RealInvoiceTester(
        header_model_path = MODEL_PATH_HEADER,
        table_model_path  = MODEL_PATH_TABLE,
        footer_model_path = MODEL_PATH_FOOTER,
        device            = "cpu",   # đổi "cuda" nếu có GPU
    )

    print("📂 Loading ground truth labels...")

    if not Path(GROUND_TRUTH_FILE).exists():
        print(f"⚠️  Ground truth file not found: {GROUND_TRUTH_FILE}")
        print("   Creating template...\n")

        template = {
            "invoices": [
                {
                    "file": "invoice_001.png",
                    "seller": {
                        "name":     "CÔNG TY KẾ TOÁN THIÊN ƯNG",
                        "tax_code": "0108892073"
                    },
                    "buyer": {
                        "name":     "Công ty TNHH Dịch Vụ Bảo An",
                        "tax_code": "0102256321"
                    },
                    "invoice": {
                        "number":       "00002486",
                        "date":         "2026-04-16",
                        "subtotal":     "1.000.000",
                        "vat_amount":   "80.000",
                        "total_amount": "1.080.000",
                    }
                }
            ]
        }

        with open(GROUND_TRUTH_FILE, "w", encoding="utf-8") as f:
            json.dump(template, f, ensure_ascii=False, indent=2)

        print(f"✅ Template created: {GROUND_TRUTH_FILE}")
        print("   Hãy điền labels thực tế vào file này rồi chạy lại!")
        return

    with open(GROUND_TRUTH_FILE, "r", encoding="utf-8") as f:
        gt_data = json.load(f)

    ground_truths = gt_data.get("invoices", [])
    print(f"✅ Loaded {len(ground_truths)} ground truth labels\n")

    predictions = []
    for gt in ground_truths:
        file_name = gt.get("file")
        if not file_name:
            continue

        file_path = Path(REAL_INVOICES_DIR) / file_name

        if not file_path.exists():
            print(f"⚠️  File not found: {file_name}")
            predictions.append({"error": "File not found"})
            continue

        try:
            if file_path.suffix.lower() == ".pdf":
                results = tester.process_pdf(file_path, Path(TEMP_DIR))
                pred    = results[0] if results else {"error": "PDF processing failed"}
            else:
                pred = tester.process_image(file_path)

            predictions.append(pred)

        except Exception as e:
            print(f"❌ Error processing {file_name}: {e}\n")
            predictions.append({"error": str(e)})

    if predictions and ground_truths:
        results = tester.evaluate(predictions, ground_truths)

        output = {
            "predictions":   predictions,
            "ground_truths": ground_truths,
            "metrics":       results,
        }

        with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
            json.dump(output, f, ensure_ascii=False, indent=2)

        print(f"\n✅ Results saved: {OUTPUT_FILE}")
    else:
        print("\n⚠️  No predictions to evaluate!")


if __name__ == "__main__":
    main()
