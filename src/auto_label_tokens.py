# -*- coding: utf-8 -*-
"""
auto_label_tokens.py  —  Approach B: 3 Separate Models
=======================================================
Chạy với --mode header, --mode table, hoặc --mode footer.

MODEL 1 — HEADER  : 12 entity types → 25 labels (B/I×12 + O)
MODEL 2 — TABLE   : 12 entity types → 25 labels (B/I×12 + O)
MODEL 3 — FOOTER  :  3 entity types →  7 labels (B/I×3  + O)

ZONE ISOLATION (KEY DESIGN):
──────────────────────────────────────────────────────────────
Mỗi mode CHỈ nhận tokens thuộc vùng không gian của mode đó.
Zone = [min_GT_y1 - padding, max_GT_y2 + padding]

  HEADER zone : từ đầu trang → dưới payment_method row
  TABLE zone  : từ row đầu item → dưới grand_total row
  FOOTER zone : từ subtotal row → cuối trang

Lý do: Mỗi model được train trên đúng vùng → không bị nhiễu
token từ vùng khác → accuracy cao hơn, ít ambiguity hơn.

BUG FIX HISTORY:
──────────────────────────────────────────────────────────────
v1 → v2:
  BUG 1 [run_ocr.py]: OCR tokens là plain strings, không có per-token bbox
  BUG 2 [CRITICAL]:   filter_tokens_by_mode dùng hardcoded pixel threshold
  BUG 3 [HIGH]:       token budget trim không ưu tiên labeled tokens
  BUG 4 [MEDIUM]:     fallback tokens không có is_fallback flag

v2 → v3:
  BUG 5 [SCHEMA]:  seller_tax_authority_code thiếu trong HEADER schema
  BUG 6 [CRASH]:   num_labels_expected hardcoded → crash khi thêm/bớt entity
  BUG 7 [DATA LOSS]: trim dùng Manhattan distance → drop labeled tokens
  BUG 8 [SILENT]:  logger.debug() → không thấy trong production log

v3 → v4 (bản này):
  BUG 9  [FOOTER FILTER MISSING]:  filter_tokens_by_mode không có nhánh footer
           → log "Unknown filter mode" và pass ALL tokens → model học sai vùng
  BUG 10 [FOOTER SCHEMA LOGIC ERROR]: vat_x_total → VAT_AMOUNT là SAI
           → vat10_total = subtotal_10% + tax_10% = GROUP TOTAL, không phải tiền thuế
           → Đổi thành GRAND_TOTAL (đúng với TABLE schema) hoặc bỏ hẳn (footer model)
  BUG 11 [ZONE NOT ISOLATED]: Mỗi mode nhận tokens cả trang → model học nhiễu
           → Thống nhất filter thành zone [min_GT_y1 - pad, max_GT_y2 + pad]
           → HEADER/TABLE/FOOTER đều dùng cùng logic, chỉ khác GT field set

USAGE:
  python auto_label_tokens.py --mode header \\
      --gt_dir   output/annotations \\
      --ocr_dir  output/ocr_results \\
      --output_dir output/labeled_tokens_header

  python auto_label_tokens.py --mode table \\
      --gt_dir   output/annotations \\
      --ocr_dir  output/ocr_results \\
      --output_dir output/labeled_tokens_table

  python auto_label_tokens.py --mode footer \\
      --gt_dir   output/annotations \\
      --ocr_dir  output/ocr_results \\
      --output_dir output/labeled_tokens_footer
"""

import json
import argparse
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Set
from dataclasses import dataclass
from difflib import SequenceMatcher
from multiprocessing import Pool, cpu_count
from tqdm import tqdm

try:
    import yaml
except Exception:
    yaml = None

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════
# HEADER SCHEMA  —  12 entities, 25 labels
# ══════════════════════════════════════════════════════════════════════
#
# ZONE: [top-of-page → max(buyer_address, payment_method).y2 + padding]
# Covers: invoice meta, seller block, buyer block, payment row
#
HEADER_FIELD_TO_LABEL: Dict[str, str] = {

    # ── Invoice meta ─────────────────────────────────────────────────
    "invoice_type":               "INVOICE_TYPE",
    "invoice_date":               "INVOICE_DATE",
    "invoice_symbol":             "INVOICE_SYMBOL",
    "invoice_number":             "INVOICE_NUMBER",
    # "invoice_form" → O  (Mẫu số: 01GTKT0/001 — decorative text)

    # ── Seller ───────────────────────────────────────────────────────
    "seller_name":                "SELLER_NAME",
    "seller_tax_code":            "SELLER_TAX_CODE",
    "seller_tax_authority_code":  "SELLER_TAX_AUTHORITY_CODE",  # FIX BUG 5
    "seller_address":             "SELLER_ADDRESS",
    # "seller_phone"        → O  (ngoài 12-entity trained model)
    # "seller_bank_account" → O
    # "seller_bank_name"    → O
    # "seller_email"        → O  (layout_07)
    # "seller_website"      → O  (layout_07)
    # "seller_name_sign"    → O  (layout_07)

    # ── Buyer — 3 aliases → BUYER_NAME ──────────────────────────────
    #   buyer_full_name     = "Họ tên người mua hàng"  (cá nhân)
    #   buyer_customer_name = "Họ và tên người mua"    (legacy alias gtgt_02,03)
    #   buyer_name          = "Tên đơn vị mua hàng"    (công ty, ALL layouts)
    #   field_extractor._split_buyer_name_spans() tách full_name vs company name
    "buyer_full_name":            "BUYER_NAME",   # gtgt_01,04,05,07 | banhang_02,03,04
    "buyer_customer_name":        "BUYER_NAME",   # gtgt_02,03
    "buyer_name":                 "BUYER_NAME",   # ALL layouts
    "buyer_tax_code":             "BUYER_TAX_CODE",
    "buyer_address":              "BUYER_ADDRESS",
    # "buyer_phone"        → O
    # "buyer_bank_account" → O

    # ── Payment ──────────────────────────────────────────────────────
    "payment_method":             "PAYMENT_METHOD",
    # "payment_currency"   → O  (layout_07)
}
# Verify: 12 entities × 2 + 1 = 25 labels
# Entities: INVOICE_TYPE, INVOICE_DATE, INVOICE_SYMBOL, INVOICE_NUMBER,
#           SELLER_NAME, SELLER_TAX_CODE, SELLER_TAX_AUTHORITY_CODE, SELLER_ADDRESS,
#           BUYER_NAME, BUYER_TAX_CODE, BUYER_ADDRESS, PAYMENT_METHOD


# ══════════════════════════════════════════════════════════════════════
# TABLE SCHEMA  —  12 entities, 25 labels
# ══════════════════════════════════════════════════════════════════════
#
# ZONE: [min(item_rows).y1 - padding → max(grand_total).y2 + padding]
# Covers: item rows + footer aggregates (subtotal / vat / grand_total)
# NOTE: TABLE mode includes footer aggregates vì model cần học cả 2 pattern
#       → FOOTER model là specialist chỉ focus vào aggregate rows
#
TABLE_FIELD_TO_LABEL: Dict[str, str] = {

    # ════════════════════════════════════════════════════════════════
    # A. ITEM ROW FIELDS  (per-row, all layouts)
    # ════════════════════════════════════════════════════════════════

    "item_name":              "ITEM_NAME",
    "item_unit":              "ITEM_UNIT",
    "item_quantity":          "ITEM_QUANTITY",
    "item_unit_price":        "ITEM_UNIT_PRICE",

    # Thành tiền — 2 aliases
    "item_total_price":       "ITEM_TOTAL_PRICE",   # gtgt_01–07 | banhang_01–03
    "item_discounted_total":  "ITEM_TOTAL_PRICE",   # banhang_04 (sau chiết khấu)

    # Chiết khấu per-item (banhang_04 only)
    "item_discount_amount":   "ITEM_DISCOUNT",

    # Thuế suất GTGT per-item (gtgt_04,05,07 | banhang_04 display)
    "item_vat_rate":          "ITEM_VAT_RATE",

    # Tiền thuế GTGT per-item — 2 aliases
    "item_line_tax":          "ITEM_LINE_TAX",      # gtgt_04, gtgt_05
    "item_vat_amount":        "ITEM_LINE_TAX",      # gtgt_07 alias

    # Thành tiền đã gồm VAT per-item (gtgt_07 — cột 8)
    "item_total_with_vat":    "ITEM_ROW_TOTAL",     # gtgt_07 only

    # ════════════════════════════════════════════════════════════════
    # B. FOOTER VAT RATE DISPLAY (inline label, không phải per-item)
    #    Ví dụ: "Thuế GTGT (10%)" → feed vào invoice.vat_rate
    # ════════════════════════════════════════════════════════════════
    "vat_rate":               "ITEM_VAT_RATE",      # gtgt_02,03 | banhang_02,04

    # ════════════════════════════════════════════════════════════════
    # C. FOOTER AGGREGATES — SUBTOTAL
    # ════════════════════════════════════════════════════════════════

    "subtotal":                   "SUBTOTAL",        # gtgt_01–05 | banhang_02
    "subtotal_after_discount":    "SUBTOTAL",        # banhang_04

    # gtgt_07: per VAT-rate group subtotals (trước thuế)
    "vat5_subtotal":              "SUBTOTAL",
    "vat5b_subtotal":             "SUBTOTAL",
    "vat8_subtotal":              "SUBTOTAL",
    "vat10_subtotal":             "SUBTOTAL",
    "no_vat_subtotal":            "SUBTOTAL",
    "no_declaration_subtotal":    "SUBTOTAL",

    # ════════════════════════════════════════════════════════════════
    # D. FOOTER AGGREGATES — VAT_AMOUNT
    # ════════════════════════════════════════════════════════════════

    "vat_amount":                 "VAT_AMOUNT",      # gtgt_01–05 | banhang_02
    "vat_amount_after_discount":  "VAT_AMOUNT",      # banhang_04

    # gtgt_07: per VAT-rate group tax amounts (tiền thuế)
    "vat5_tax":                   "VAT_AMOUNT",
    "vat5b_tax":                  "VAT_AMOUNT",
    "vat8_tax":                   "VAT_AMOUNT",
    "vat10_tax":                  "VAT_AMOUNT",
    "no_vat_tax":                 "VAT_AMOUNT",
    "no_declaration_tax":         "VAT_AMOUNT",

    # ════════════════════════════════════════════════════════════════
    
    # "total_discount" → O  (banhang_04 — không có TOTAL_DISCOUNT entity)
}
# Verify: 12 entities × 2 + 1 = 25 labels
# Entities: ITEM_NAME, ITEM_UNIT, ITEM_QUANTITY, ITEM_UNIT_PRICE,
#           ITEM_TOTAL_PRICE, ITEM_DISCOUNT, ITEM_VAT_RATE, ITEM_LINE_TAX,
#           ITEM_ROW_TOTAL, SUBTOTAL, VAT_AMOUNT, GRAND_TOTAL


# ══════════════════════════════════════════════════════════════════════
# FOOTER SCHEMA  —  3 entities, 7 labels
# ══════════════════════════════════════════════════════════════════════
#
# ZONE: [min(subtotal row).y1 - padding → image_height]
# Covers: ONLY the aggregate summary rows below item rows
#
# DESIGN DECISIONS:
#   1. vat_x_total KHÔNG có trong footer schema (BUG 10 FIX + BUG 9)
#      Lý do: vat5_total = subtotal_5% + vat5_tax = group GRAND_TOTAL
#             → nếu map → VAT_AMOUNT sẽ train model sai hoàn toàn
#             → nếu map → GRAND_TOTAL thì footer model cần phân biệt
#               6 per-group totals + 1 final total → quá phức tạp cho 3-entity model
#             → SOLUTION: bỏ hẳn, footer model chỉ nhận grand_total_layout07 cho layout_07
#             → Các per-group values sẽ được TABLE model học (TABLE schema có đủ aliases)
#
#   2. vat_x_subtotal, vat_x_tax vẫn giữ
#      Lý do: Chúng là SUBTOTAL và VAT_AMOUNT thực sự (đúng về mặt kế toán)
#             → footer model học pattern aggregate rows → benefit khi inference
#
#   3. id2label_footer.json cần được tạo riêng với 7 labels:
#      B-SUBTOTAL, I-SUBTOTAL, B-VAT_AMOUNT, I-VAT_AMOUNT,
#      B-GRAND_TOTAL, I-GRAND_TOTAL, O
#
FOOTER_FIELD_TO_LABEL: Dict[str, str] = {

    # ── Standard aggregates (all layouts) ────────────────────────────
    "subtotal":                   "SUBTOTAL",
    "subtotal_after_discount":    "SUBTOTAL",        # banhang_04

    "vat_amount":                 "VAT_AMOUNT",
    "vat_amount_after_discount":  "VAT_AMOUNT",      # banhang_04

    "grand_total":                "GRAND_TOTAL",
    "grand_total_after_discount": "GRAND_TOTAL",     # banhang_04
    "grand_total_layout07":       "GRAND_TOTAL",     # gtgt_07 final total

    # ── gtgt_07: per VAT-rate group subtotals (tiền chưa thuế theo nhóm) ──
    "vat5_subtotal":              "SUBTOTAL",
    "vat5b_subtotal":             "SUBTOTAL",
    "vat8_subtotal":              "SUBTOTAL",
    "vat10_subtotal":             "SUBTOTAL",
    "no_vat_subtotal":            "SUBTOTAL",
    "no_declaration_subtotal":    "SUBTOTAL",

    # ── gtgt_07: per VAT-rate group tax amounts (tiền thuế theo nhóm) ──
    "vat5_tax":                   "VAT_AMOUNT",
    "vat5b_tax":                  "VAT_AMOUNT",
    "vat8_tax":                   "VAT_AMOUNT",
    "vat10_tax":                  "VAT_AMOUNT",
    "no_vat_tax":                 "VAT_AMOUNT",
    "no_declaration_tax":         "VAT_AMOUNT",

    # ── KHÔNG CÓ vat_x_total (BUG 10 FIX) ────────────────────────────
    # "vat5_total"  → BỎQUA — là group grand total, TABLE model xử lý
    # "vat10_total" → BỎQUA — tương tự
    # Nếu cần dùng footer model detect các giá trị này, map → GRAND_TOTAL
    # nhưng hiện tại footer model chỉ cần grand_total_layout07 là đủ
}
# Verify: 3 entities × 2 + 1 = 7 labels
# Entities: SUBTOTAL, VAT_AMOUNT, GRAND_TOTAL


# ══════════════════════════════════════════════════════════════════════
# KNOWN_O_FIELDS — Fields được assign "O" có chủ ý (không phải bug)
# load_gt_annotations() sẽ suppress false-alert warning cho các field này
# ══════════════════════════════════════════════════════════════════════
KNOWN_O_FIELDS: frozenset = frozenset({
    # ── Structural ────────────────────────────────────────────────
    "item_index",                    # STT column

    # ── Invoice meta decorative ───────────────────────────────────
    "invoice_form",                  # "Mẫu số: 01GTKT0/001"

    # ── Seller contact (ngoài 12-entity header model) ─────────────
    "seller_phone",
    "seller_bank_account",
    "seller_bank_name",
    "seller_email",
    "seller_website",
    "seller_name_sign",

    # ── Buyer contact ─────────────────────────────────────────────
    "buyer_phone",
    "buyer_bank_account",
    "buyer_signature",
    "seller_signature",

    # ── Payment extras (layout_07) ───────────────────────────────
    "payment_currency",
    "retrieval_code",
    "retrieval_url",

    # ── Footer text representations ───────────────────────────────
    "total_discount",
    "amount_in_words",
    "amount_in_words_after_discount",

    # ── gtgt_07 per-group totals (Footer model bỏ qua) ───────────
    # TABLE model xử lý các field này với mapping GRAND_TOTAL
    "vat5_total",
    "vat5b_total",
    "vat8_total",
    "vat10_total",
    "no_vat_total",
    "no_declaration_total",
})


# ══════════════════════════════════════════════════════════════════════
# SCHEMA REGISTRY
# ══════════════════════════════════════════════════════════════════════
_SCHEMAS = {
    "header": {
        "field_to_label": HEADER_FIELD_TO_LABEL,
        "schema_version": "header_v4_production",
        "default_out":    "output/labeled_tokens_header",
    },
    "table": {
        "field_to_label": TABLE_FIELD_TO_LABEL,
        "schema_version": "table_v4_production",
        "default_out":    "output/labeled_tokens_table",
    },
    "footer": {
        "field_to_label": FOOTER_FIELD_TO_LABEL,
        "schema_version": "footer_v4_production",
        "default_out":    "output/labeled_tokens_footer",
    },
}


# ══════════════════════════════════════════════════════════════════════
# TUNABLE DEFAULTS
# ══════════════════════════════════════════════════════════════════════
DEFAULT_OVERLAP_THRESHOLD:      float = 0.35
DEFAULT_LINE_Y_THRESHOLD:       float = 0.55
DEFAULT_TEXT_SIM_THRESHOLD:     float = 0.70
DEFAULT_TEXT_SIM_BONUS:         float = 0.25
DEFAULT_CONTAINMENT_BONUS:      float = 0.15
DEFAULT_FALLBACK_OVERLAP_SCALE: float = 0.60
DEFAULT_ZONE_PADDING_RATIO:     float = 0.05   # 5% of image height

_TOKEN_BUDGET_WARN  = 400
_TOKEN_BUDGET_LIMIT = 510
_MAX_SAFE_TOKENS    = 500


# ======================================================================
# LOGGING
# ======================================================================
_STANDARD_LOG_RECORD_KEYS = set(logging.LogRecord(
    name="", level=0, pathname="", lineno=0, msg="", args=(), exc_info=None
).__dict__.keys())


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        base = {
            "time":    self.formatTime(record, "%H:%M:%S"),
            "level":   record.levelname,
            "name":    record.name,
            "message": record.getMessage(),
        }
        extras = {k: v for k, v in record.__dict__.items()
                  if k not in _STANDARD_LOG_RECORD_KEYS}
        base.update(extras)
        return json.dumps(base, ensure_ascii=False)


def configure_logging(json_log: bool) -> None:
    handler = logging.StreamHandler()
    if json_log:
        handler.setFormatter(JsonFormatter())
    else:
        handler.setFormatter(logging.Formatter(
            fmt="%(asctime)s [%(levelname)s] %(message)s",
            datefmt="%H:%M:%S",
        ))
    root = logging.getLogger()
    root.handlers = [handler]
    root.setLevel(logging.INFO)


# ======================================================================
# SCHEMA BUILDER + VALIDATOR  (FIX BUG 6: dynamic expected count)
# ======================================================================
def build_valid_labels(field_to_label: Dict[str, str]) -> set:
    return {
        f"{bio}-{label}"
        for label in set(field_to_label.values())
        for bio in ("B", "I")
    } | {"O"}


def validate_schema(mode: str, field_to_label: Dict[str, str]) -> None:
    valid_labels = build_valid_labels(field_to_label)
    n_entity     = len(set(field_to_label.values()))
    n_labels     = len(valid_labels)
    expected     = n_entity * 2 + 1  # FIX BUG 6: tính động, không hardcode

    logger.info(f"[{mode.upper()}] Entities : {n_entity}  |  Labels : {n_labels}  (expected {expected})")

    if n_labels != expected:
        raise RuntimeError(
            f"[{mode.upper()}] Schema mismatch! expected={expected}, got={n_labels}. "
            f"Check for alias mapping errors."
        )

    b_set = {l[2:] for l in valid_labels if l.startswith("B-")}
    i_set = {l[2:] for l in valid_labels if l.startswith("I-")}
    if b_set != i_set:
        raise RuntimeError(f"B/I asymmetry: {b_set.symmetric_difference(i_set)}")

    logger.info(f"[{mode.upper()}] Schema ✅  entities={sorted(set(field_to_label.values()))}")


# ======================================================================
# DATA STRUCTURES
# ======================================================================
@dataclass
class BBox:
    x1: int
    y1: int
    x2: int
    y2: int

    @property
    def area(self) -> int:
        return max(0, self.x2 - self.x1) * max(0, self.y2 - self.y1)

    @property
    def height(self) -> int:
        return max(0, self.y2 - self.y1)

    @classmethod
    def from_xywh(cls, x, y, w, h) -> "BBox":
        return cls(int(x), int(y), int(x + w), int(y + h))

    @classmethod
    def from_xyxy(cls, coords) -> "BBox":
        return cls(int(coords[0]), int(coords[1]), int(coords[2]), int(coords[3]))

    def __repr__(self) -> str:
        return f"BBox({self.x1},{self.y1},{self.x2},{self.y2})"


@dataclass
class GTField:
    field_key: str
    label:     str
    text:      str
    bbox:      BBox


@dataclass
class OCRToken:
    token_id:    int
    text:        str
    bbox:        BBox
    line_id:     int
    confidence:  float
    is_fallback: bool = False  # FIX BUG 4: track bbox approximation


@dataclass
class LabeledToken:
    token_id:    int
    text:        str
    bbox:        BBox
    label:       str
    field_key:   Optional[str]
    confidence:  float
    is_fallback: bool = False


@dataclass
class MatchConfig:
    overlap_threshold:      float = DEFAULT_OVERLAP_THRESHOLD
    line_y_threshold:       float = DEFAULT_LINE_Y_THRESHOLD
    text_sim_threshold:     float = DEFAULT_TEXT_SIM_THRESHOLD
    text_sim_bonus:         float = DEFAULT_TEXT_SIM_BONUS
    containment_bonus:      float = DEFAULT_CONTAINMENT_BONUS
    fallback_overlap_scale: float = DEFAULT_FALLBACK_OVERLAP_SCALE
    zone_padding_ratio:     float = DEFAULT_ZONE_PADDING_RATIO


@dataclass
class AppConfig:
    match:              MatchConfig
    token_budget_warn:  int
    token_budget_limit: int
    max_safe_tokens:    int
    workers:            int
    dry_run:            bool
    stats_only:         bool
    force:              bool
    skip_exist:         bool
    json_log:           bool


# ======================================================================
# BBOX UTILITIES
# ======================================================================
def overlap_ratio(token: BBox, gt: BBox) -> float:
    """Intersection / token_area — token nhỏ nằm trong GT lớn → ratio cao."""
    ix1 = max(token.x1, gt.x1)
    iy1 = max(token.y1, gt.y1)
    ix2 = min(token.x2, gt.x2)
    iy2 = min(token.y2, gt.y2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    return inter / token.area if token.area > 0 else 0.0


def y_overlap_ratio(token: BBox, gt: BBox) -> float:
    """Line-level vertical overlap — cheap early reject."""
    iy1 = max(token.y1, gt.y1)
    iy2 = min(token.y2, gt.y2)
    inter_y = max(0, iy2 - iy1)
    denom = min(token.height, gt.height)
    return inter_y / denom if denom > 0 else 0.0


def compute_iou(a: BBox, b: BBox) -> float:
    ix1 = max(a.x1, b.x1)
    iy1 = max(a.y1, b.y1)
    ix2 = min(a.x2, b.x2)
    iy2 = min(a.y2, b.y2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    if inter == 0:
        return 0.0
    union = a.area + b.area - inter
    return inter / union if union > 0 else 0.0


def text_similarity(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    return SequenceMatcher(None, a.strip().lower(), b.strip().lower()).ratio()


def parse_bbox(b: Any) -> Optional[BBox]:
    if isinstance(b, dict):
        if all(k in b for k in ("x", "y", "width", "height")):
            return BBox.from_xywh(b["x"], b["y"], b["width"], b["height"])
    elif isinstance(b, (list, tuple)) and len(b) == 4:
        return BBox.from_xyxy(b)
    return None


# ======================================================================
# FALLBACK TOKEN SPLITTING  (backward-compat with old OCR format)
# ======================================================================
def split_line_bbox(line_bbox: List[int], tokens: List[str]) -> List[BBox]:
    """
    FALLBACK: Chia bbox dòng theo tỉ lệ ký tự.
    Chỉ dùng khi OCR không cung cấp per-token bbox (OCR data cũ).
    Với run_ocr.py đã fix (Bug 1), path này không bao giờ trigger.
    """
    x1_line, y1_line, x2_line, y2_line = line_bbox
    line_width = x2_line - x1_line
    if not tokens or line_width <= 0:
        return []
    char_counts = [max(1, len(t)) for t in tokens]
    total_chars = sum(char_counts)
    bboxes: List[BBox] = []
    cursor = x1_line
    for i, n_chars in enumerate(char_counts):
        is_last = (i == len(tokens) - 1)
        x2 = x2_line if is_last else cursor + int(line_width * n_chars / total_chars)
        bboxes.append(BBox(cursor, y1_line, x2, y2_line))
        cursor = x2
    return bboxes


# ======================================================================
# ZONE COMPUTATION  (BUG 11 FIX — unified across all modes)
# ======================================================================
def compute_zone(
    gt_fields:    List[GTField],
    image_height: int,
    mode:         str,
    padding_ratio: float = DEFAULT_ZONE_PADDING_RATIO,
) -> Tuple[int, int]:
    """
    Tính zone [y_start, y_end] chứa TẤT CẢ GT fields của mode này.

    Zone = [min_GT_y1 - padding, max_GT_y2 + padding]

    Logic thống nhất cho cả 3 mode:
      HEADER : GT = invoice meta + seller + buyer + payment
               → zone bao phủ header block
      TABLE  : GT = item rows + subtotal + vat + grand_total
               → zone bao phủ toàn bộ table section
      FOOTER : GT = subtotal + vat_amount + grand_total
               → zone bao phủ aggregate rows ở cuối invoice

    Args:
        gt_fields:    GT fields đã được filter theo mode schema
        image_height: pixel height của invoice image
        mode:         "header" | "table" | "footer"
        padding_ratio: padding = image_height * padding_ratio (min 20px)

    Returns:
        (y_start, y_end) cả hai đã clip vào [0, image_height]
    """
    if not gt_fields:
        # Không có GT → trả về toàn trang (sẽ được log ở caller)
        return 0, image_height

    padding = max(20, int(image_height * padding_ratio))

    min_y1 = min(f.bbox.y1 for f in gt_fields)
    max_y2 = max(f.bbox.y2 for f in gt_fields)

    # Header: bắt đầu từ top-of-page (không clip top)
    y_start = 0 if mode == "header" else max(0, min_y1 - padding)

    # Footer: kéo dài đến bottom-of-page (không clip bottom)
    y_end = image_height if mode == "footer" else min(image_height, max_y2 + padding)

    return y_start, y_end


# ======================================================================
# TOKEN REGION FILTER  (BUG 9 + BUG 11 FIX — unified zone logic)
# ======================================================================
def filter_tokens_by_mode(
    tokens:       List[OCRToken],
    gt_fields:    List[GTField],
    mode:         str,
    image_height: int,
    padding_ratio: float = DEFAULT_ZONE_PADDING_RATIO,
) -> List[OCRToken]:
    """
    Giữ ONLY tokens thuộc zone của mode này.

    BUG 9  FIX: Thêm nhánh footer (trước đây → "Unknown mode" warning + pass all)
    BUG 11 FIX: Unified zone = [min_GT_y1 - pad, max_GT_y2 + pad] cho cả 3 modes

    A token được giữ nếu: y_start <= tok.y1 AND tok.y2 <= y_end
    (cần cả y1 VÀ y2 nằm trong zone để tránh cross-boundary tokens)
    """
    if not tokens:
        return tokens

    if image_height <= 0:
        logger.warning(f"[{mode}] Invalid image_height={image_height} → skip zone filter")
        return tokens

    if not gt_fields:
        logger.warning(
            f"[{mode}] 0 GT fields loaded → skip zone filter, ALL tokens pass. "
            f"Check schema mapping covers annotation field keys."
        )
        return tokens

    y_start, y_end = compute_zone(gt_fields, image_height, mode, padding_ratio)

    filtered = [
        t for t in tokens
        if t.bbox.y1 >= y_start and t.bbox.y2 <= y_end
    ]

    if not filtered:
        # Fallback: thử relaxed — chỉ cần y1 nằm trong zone
        filtered_relaxed = [
            t for t in tokens
            if y_start <= t.bbox.y1 <= y_end
        ]
        logger.warning(
            f"[{mode}] Strict zone filter → 0 tokens. "
            f"Fallback to relaxed filter → {len(filtered_relaxed)} tokens. "
            f"Zone y=[{y_start},{y_end}], img_h={image_height}"
        )
        return filtered_relaxed

    logger.debug(
        f"[{mode}] Zone y=[{y_start},{y_end}] (pad={int(image_height * padding_ratio)}px) "
        f"kept={len(filtered)}/{len(tokens)} tokens"
    )
    return filtered


# ======================================================================
# SPATIAL MATCHING
# ======================================================================
def find_best_gt(
    token:     OCRToken,
    gt_fields: List[GTField],
    cfg:       MatchConfig,
) -> Optional[GTField]:
    """
    Tìm GT field phù hợp nhất cho một OCR token.

    Ranking (DESC):
      1. spatial_score = overlap_ratio + bonuses  (PRIMARY — FIX BUG 3)
      2. IoU                                       (tiebreaker 1)
      3. text_similarity + containment_bonus       (tiebreaker 2)
      4. GT area ASC                               (prefer smaller GT)

    Fallback tokens dùng relaxed threshold (× fallback_overlap_scale) — FIX BUG 4.
    """
    effective_overlap_th = (
        cfg.overlap_threshold * cfg.fallback_overlap_scale
        if token.is_fallback else cfg.overlap_threshold
    )

    candidates: List[Tuple[float, float, float, float, GTField]] = []

    for gt in gt_fields:
        # 1. Line alignment (cheap reject)
        if y_overlap_ratio(token.bbox, gt.bbox) < cfg.line_y_threshold:
            continue

        # 2. Spatial overlap
        ov = overlap_ratio(token.bbox, gt.bbox)
        if ov < effective_overlap_th:
            continue

        # 3. Text similarity
        sim = text_similarity(token.text, gt.text)

        # 4. Containment bonus (token text xuất hiện trong gt text)
        tok_lower = token.text.strip().lower()
        gt_lower  = gt.text.strip().lower()
        containment = cfg.containment_bonus if (
            tok_lower and len(tok_lower) >= 2 and tok_lower in gt_lower
        ) else 0.0

        # 5. Text sim bonus
        text_bonus = cfg.text_sim_bonus if sim >= cfg.text_sim_threshold else 0.0

        # 6. Composite spatial score (primary)
        spatial_score = ov + containment + text_bonus
        iou = compute_iou(token.bbox, gt.bbox)

        candidates.append((spatial_score, iou, sim + containment, -gt.bbox.area, gt))

    if not candidates:
        return None

    candidates.sort(key=lambda x: (x[0], x[1], x[2], x[3]), reverse=True)
    return candidates[0][4]


# ======================================================================
# BIO LABEL ASSIGNMENT
# ======================================================================
def assign_bio_labels(
    ocr_tokens: List[OCRToken],
    gt_fields:  List[GTField],
    cfg:        MatchConfig,
) -> Tuple[List[LabeledToken], Set[int]]:
    """
    2-pass BIO assignment:
      Pass 1: spatial matching → matches dict
      Pass 2: B- nếu GT mới, I- nếu cùng GT liên tiếp, O + reset nếu no match
    """
    matches: Dict[int, GTField] = {}
    matched_gt_ids: Set[int] = set()

    for tok in ocr_tokens:
        gt = find_best_gt(tok, gt_fields, cfg)
        if gt is not None:
            matches[tok.token_id] = gt
            matched_gt_ids.add(id(gt))

    labeled: List[LabeledToken] = []
    prev_gt_id: Optional[int] = None

    for tok in ocr_tokens:
        gt = matches.get(tok.token_id)
        if gt is not None:
            current_gt_id = id(gt)
            bio = "B" if current_gt_id != prev_gt_id else "I"
            prev_gt_id = current_gt_id
            labeled.append(LabeledToken(
                token_id=tok.token_id, text=tok.text, bbox=tok.bbox,
                label=f"{bio}-{gt.label}", field_key=gt.field_key,
                confidence=tok.confidence, is_fallback=tok.is_fallback,
            ))
        else:
            prev_gt_id = None
            labeled.append(LabeledToken(
                token_id=tok.token_id, text=tok.text, bbox=tok.bbox,
                label="O", field_key=None,
                confidence=tok.confidence, is_fallback=tok.is_fallback,
            ))

    return labeled, matched_gt_ids


# ======================================================================
# TOKEN BUDGET TRIMMING  (FIX BUG 3 + BUG 7)
# ======================================================================
def trim_tokens_to_budget(
    tokens:     List[OCRToken],
    gt_fields:  List[GTField],
    max_tokens: int,
    invoice_id: str,
    mode:       str,
) -> Tuple[List[OCRToken], int]:
    """
    Smart trim khi tokens > max_tokens TRƯỚC KHI labeling.

    FIX BUG 3: Trim trước labeling → LayoutLMv3 không truncate labeled tokens
    FIX BUG 7: 3-tier priority thay vì Manhattan distance thuần

    Tier 0 (score = -2.0): Token trong footer zone (y1 > max_item_y2)
                           → LUÔN giữ (GRAND_TOTAL / SUBTOTAL / VAT_AMOUNT)
    Tier 1 (score = -1.0): Token có spatial overlap với bất kỳ GT field nào
                           → Luôn giữ trước Tier 2
    Tier 2 (score ≥ 0.0):  Token không overlap GT → sort theo Manhattan distance
                           → Drop những token xa GT nhất

    Sau trim: re-sort theo reading order (y1, x1).
    """
    n_tokens = len(tokens)
    if n_tokens <= max_tokens:
        return tokens, 0

    if not gt_fields:
        logger.warning(f"[{invoice_id}] {n_tokens} tokens, no GT → keep first {max_tokens}")
        return tokens[:max_tokens], n_tokens - max_tokens

    # Footer zone boundary (for Tier-0 guard)
    max_gt_y2 = max(gt.bbox.y2 for gt in gt_fields)

    def trim_score(tok: OCRToken) -> float:
        # Tier 0: footer aggregate zone — NEVER drop
        if tok.bbox.y1 > max_gt_y2:
            return -2.0

        # Tier 1: overlaps any GT field — always keep
        for gt in gt_fields:
            if (max(tok.bbox.x1, gt.bbox.x1) < min(tok.bbox.x2, gt.bbox.x2) and
                    max(tok.bbox.y1, gt.bbox.y1) < min(tok.bbox.y2, gt.bbox.y2)):
                return -1.0  # FIX BUG 7: Tier 1 < Tier 2

        # Tier 2: Manhattan distance to nearest GT center
        tok_cx = (tok.bbox.x1 + tok.bbox.x2) / 2
        tok_cy = (tok.bbox.y1 + tok.bbox.y2) / 2
        return min(
            abs(tok_cx - (gt.bbox.x1 + gt.bbox.x2) / 2) +
            abs(tok_cy - (gt.bbox.y1 + gt.bbox.y2) / 2)
            for gt in gt_fields
        )

    scored_indices = sorted(range(n_tokens), key=lambda i: trim_score(tokens[i]))
    kept_indices = set(scored_indices[:max_tokens])

    # Restore reading order
    trimmed = [
        tokens[i]
        for i in sorted(kept_indices, key=lambda i: (tokens[i].bbox.y1, tokens[i].bbox.x1))
    ]

    n_dropped = n_tokens - max_tokens
    logger.warning(
        f"[{invoice_id}] ⚠️ TRIMMED {n_dropped} tokens "
        f"({n_tokens} → {max_tokens}) mode={mode}. "
        f"Giảm items/invoice hoặc dùng --max-tokens."
    )
    return trimmed, n_dropped


# ======================================================================
# LOADERS
# ======================================================================
def safe_json_load(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON: {path} ({e})")


def load_gt_annotations(
    gt_path:        Path,
    field_to_label: Dict[str, str],
    mode:           str,
) -> Tuple[str, int, int, List[GTField], int]:
    data       = safe_json_load(gt_path)
    invoice_id = data["invoice_id"]
    width      = int(data["width"])
    height     = int(data["height"])

    gt_fields:      List[GTField] = []
    skipped_known:  List[str]     = []
    skipped_unknown: List[str]    = []

    for ann in data.get("annotations", []):
        field_key = ann.get("field", "").strip().lower()

        if field_key not in field_to_label:
            if field_key in KNOWN_O_FIELDS:
                skipped_known.append(field_key)
            else:
                skipped_unknown.append(field_key)
            continue

        label = field_to_label[field_key]
        bbox  = parse_bbox(ann.get("bbox"))
        if bbox is None or bbox.area <= 0:
            logger.warning(f"[{invoice_id}][{mode}] '{field_key}' bbox invalid, skip.")
            continue

        gt_fields.append(GTField(
            field_key=field_key, label=label,
            text=ann.get("text", ""), bbox=bbox,
        ))

    # FIX BUG 8: logger.warning() thay vì logger.debug()
    if skipped_unknown:
        unique_unknown = sorted(set(skipped_unknown))
        logger.warning(
            f"[{invoice_id}][{mode}] {len(unique_unknown)} UNKNOWN field(s) → label O: "
            f"{unique_unknown}  (add to schema or KNOWN_O_FIELDS)"
        )

    n_skipped_total = len(skipped_known) + len(skipped_unknown)

    if not gt_fields:
        logger.warning(
            f"[{invoice_id}][{mode}] No GT fields matched schema! "
            f"unknown={sorted(set(skipped_unknown))}  known_O={sorted(set(skipped_known))}"
        )

    return invoice_id, width, height, gt_fields, n_skipped_total


def load_ocr_results(
    invoice_id: str,
    ocr_path:   Path,
) -> Tuple[str, int, int, List[OCRToken], int, int]:
    data       = safe_json_load(ocr_path)
    image_name = data.get("image", "")
    width      = int(data.get("width",  0))
    height     = int(data.get("height", 0))

    ocr_tokens:        List[OCRToken] = []
    token_id           = 0
    fallback_lines     = 0
    invalid_token_bbox = 0

    for line in data.get("lines", []):
        line_id    = line.get("id", 0)
        confidence = float(line.get("confidence", 1.0))
        tokens_raw = line.get("tokens", [])
        line_bbox  = line.get("bbox", [0, 0, 0, 0])

        if not tokens_raw:
            continue

        token_texts:  List[str]   = []
        token_bboxes: List[BBox]  = []
        used_real_bbox = False

        # Case 1: tokens là dicts với per-token bbox (run_ocr.py đã fix)
        if isinstance(tokens_raw[0], dict):
            missing = False
            for t in tokens_raw:
                text = str(t.get("text", ""))
                token_texts.append(text)
                bbox = parse_bbox(t.get("bbox") or t.get("box"))
                if bbox is None:
                    missing = True
                    break
                token_bboxes.append(bbox)
            if not missing and len(token_bboxes) == len(token_texts):
                used_real_bbox = True

        # Case 2: strings + separate token_bboxes key
        if not used_real_bbox and isinstance(tokens_raw[0], str):
            token_texts = [str(t) for t in tokens_raw]
            token_bbox_raw = line.get("token_bboxes")
            if isinstance(token_bbox_raw, list) and len(token_bbox_raw) == len(token_texts):
                parsed = [parse_bbox(b) for b in token_bbox_raw]
                if all(p is not None for p in parsed):
                    token_bboxes = [p for p in parsed if p is not None]
                    used_real_bbox = True

        # Case 3 (FALLBACK): OCR data cũ — không có per-token bbox
        if not used_real_bbox:
            fallback_lines += 1
            token_texts  = [t.get("text", "") if isinstance(t, dict) else str(t)
                            for t in tokens_raw]
            token_bboxes = split_line_bbox(line_bbox, token_texts)

        is_fallback_line = not used_real_bbox

        for tok_text, tok_bbox in zip(token_texts, token_bboxes):
            if not tok_text.strip():
                continue
            if tok_bbox is None or tok_bbox.area <= 0:
                invalid_token_bbox += 1
                continue
            ocr_tokens.append(OCRToken(
                token_id=token_id, text=tok_text, bbox=tok_bbox,
                line_id=line_id, confidence=confidence,
                is_fallback=is_fallback_line,
            ))
            token_id += 1

    if fallback_lines > 0:
        logger.warning(
            f"[{invoice_id}] {fallback_lines} fallback lines (OCR thiếu per-token bbox). "
            f"Upgrade run_ocr.py để fix upstream."
        )
    if invalid_token_bbox > 0:
        logger.warning(f"[{invoice_id}] {invalid_token_bbox} token bbox invalid, skipped.")

    return image_name, width, height, ocr_tokens, fallback_lines, invalid_token_bbox


# ======================================================================
# TOKEN BUDGET WARNING
# ======================================================================
def check_token_budget(
    invoice_id: str,
    n_tokens:   int,
    n_labeled:  int,
    mode:       str,
    warn_th:    int,
    limit_th:   int,
) -> None:
    if mode == "header":
        if n_labeled > 200:
            logger.warning(
                f"[{invoice_id}][header] {n_labeled} labeled tokens — bất thường! "
                f"Zone filter có đang bao phủ vùng table không?"
            )
    elif mode == "footer":
        if n_labeled > 100:
            logger.warning(
                f"[{invoice_id}][footer] {n_labeled} labeled tokens — bất thường! "
                f"Footer zone quá rộng?"
            )
    else:
        if n_tokens >= limit_th:
            logger.error(
                f"[{invoice_id}][table] 🚨 {n_tokens} tokens VẪN VƯỢT {limit_th} sau trim!"
            )
        elif n_tokens >= warn_th:
            logger.info(f"[{invoice_id}][table] {n_tokens} tokens — gần giới hạn 512.")


# ======================================================================
# CONFIDENCE SCORE
# ======================================================================
def compute_confidence_score(
    labeled:   List[LabeledToken],
    n_tokens:  int,
    n_labeled: int,
) -> float:
    if n_tokens <= 0:
        return 0.0
    bi_tokens = [t for t in labeled if t.label != "O"]
    if not bi_tokens:
        return 0.0
    avg_conf = sum(
        t.confidence * (0.8 if t.is_fallback else 1.0) for t in bi_tokens
    ) / len(bi_tokens)
    label_ratio = n_labeled / n_tokens
    return round((label_ratio + avg_conf) / 2, 4)


# ======================================================================
# PROCESS SINGLE INVOICE
# ======================================================================
def process_single_invoice(
    gt_path:        Path,
    ocr_path:       Path,
    output_path:    Path,
    field_to_label: Dict[str, str],
    valid_labels:   set,
    schema_version: str,
    mode:           str,
    cfg:            AppConfig,
) -> Dict[str, Any]:

    invoice_id = gt_path.stem

    if not ocr_path.exists():
        return {"status": "skipped", "invoice_id": invoice_id, "reason": "missing_ocr"}
    if output_path.exists() and cfg.skip_exist:
        return {"status": "skipped", "invoice_id": invoice_id, "reason": "output_exists"}
    if output_path.exists() and not cfg.force and not cfg.skip_exist:
        return {"status": "skipped", "invoice_id": invoice_id, "reason": "output_exists_no_force"}

    invoice_id, gt_w, gt_h, gt_fields, gt_skipped = load_gt_annotations(
        gt_path, field_to_label, mode
    )
    image_name, ocr_w, ocr_h, ocr_tokens, fallback_lines, invalid_bbox = load_ocr_results(
        invoice_id, ocr_path
    )

    if not gt_fields:
        return {"status": "skipped", "invoice_id": invoice_id, "reason": "no_gt_fields"}
    if not ocr_tokens:
        return {"status": "skipped", "invoice_id": invoice_id, "reason": "no_ocr_tokens"}

    image_height = ocr_h or gt_h or 1000

    # BUG 9/11 FIX: Unified zone filter — ONLY tokens in this mode's zone
    ocr_tokens = filter_tokens_by_mode(
        ocr_tokens, gt_fields, mode, image_height, cfg.match.zone_padding_ratio
    )

    # BUG 3/7 FIX: Smart trim BEFORE labeling
    n_before_trim = len(ocr_tokens)
    ocr_tokens, n_dropped = trim_tokens_to_budget(
        ocr_tokens, gt_fields, cfg.max_safe_tokens, invoice_id, mode
    )

    labeled, matched_gt_ids = assign_bio_labels(ocr_tokens, gt_fields, cfg.match)

    for tok in labeled:
        if tok.label not in valid_labels:
            raise ValueError(f"[{invoice_id}][{mode}] Invalid label: '{tok.label}'")

    n_labeled = sum(1 for t in labeled if t.label != "O")
    n_tokens  = len(labeled)

    check_token_budget(invoice_id, n_tokens, n_labeled, mode,
                       cfg.token_budget_warn, cfg.token_budget_limit)

    confidence_score = compute_confidence_score(labeled, n_tokens, n_labeled)

    output = {
        "invoice_id":            invoice_id,
        "image":                 image_name,
        "width":                 ocr_w,
        "height":                ocr_h,
        "mode":                  mode,
        "schema_version":        schema_version,
        "confidence_score":      confidence_score,
        "matched_gt_count":      len(matched_gt_ids),
        "unmatched_token_count": n_tokens - n_labeled,
        "n_tokens_before_trim":  n_before_trim,
        "n_dropped_by_trim":     n_dropped,
        "tokens": [
            {
                "id":          t.token_id,
                "text":        t.text,
                "bbox":        [t.bbox.x1, t.bbox.y1, t.bbox.x2, t.bbox.y2],
                "label":       t.label,
                "field":       t.field_key,
                "confidence":  round(t.confidence, 4),
                "is_fallback": t.is_fallback,
            }
            for t in labeled
        ],
    }

    if not cfg.dry_run:
        output_path.write_text(
            json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    return {
        "status":                "processed",
        "invoice_id":            invoice_id,
        "n_tokens":              n_tokens,
        "n_labeled":             n_labeled,
        "n_o":                   n_tokens - n_labeled,
        "gt_skipped":            gt_skipped,
        "fallback_lines":        fallback_lines,
        "invalid_token_bbox":    invalid_bbox,
        "matched_gt_count":      len(matched_gt_ids),
        "unmatched_token_count": n_tokens - n_labeled,
        "n_dropped_by_trim":     n_dropped,
    }


# ======================================================================
# STATS ONLY
# ======================================================================
def compute_stats_only(gt_path: Path, ocr_path: Path) -> Dict[str, Any]:
    invoice_id = gt_path.stem
    if not ocr_path.exists():
        return {"status": "skipped", "invoice_id": invoice_id, "reason": "missing_ocr"}
    try:
        _, _, _, ocr_tokens, fallback_lines, invalid_bbox = load_ocr_results(invoice_id, ocr_path)
    except Exception as e:
        return {"status": "skipped", "invoice_id": invoice_id, "reason": str(e)}
    return {
        "status":             "processed",
        "invoice_id":         invoice_id,
        "n_tokens":           len(ocr_tokens),
        "fallback_lines":     fallback_lines,
        "invalid_token_bbox": invalid_bbox,
    }


# ======================================================================
# SCHEMA SUMMARY PRINT
# ======================================================================
def print_schema_summary(mode: str, field_to_label: Dict[str, str], cfg: MatchConfig) -> None:
    from collections import defaultdict

    groups: Dict[str, List[str]] = defaultdict(list)
    for fk, label in sorted(field_to_label.items()):
        groups[label].append(fk)

    n_entity = len(groups)
    n_labels = n_entity * 2 + 1

    ORDER = {
        "header": [
            "INVOICE_TYPE", "INVOICE_DATE", "INVOICE_SYMBOL", "INVOICE_NUMBER",
            "SELLER_NAME", "SELLER_TAX_CODE", "SELLER_TAX_AUTHORITY_CODE", "SELLER_ADDRESS",
            "BUYER_NAME", "BUYER_TAX_CODE", "BUYER_ADDRESS", "PAYMENT_METHOD",
        ],
        "table": [
            "ITEM_NAME", "ITEM_UNIT", "ITEM_QUANTITY",
            "ITEM_UNIT_PRICE", "ITEM_TOTAL_PRICE", "ITEM_DISCOUNT",
            "ITEM_VAT_RATE", "ITEM_LINE_TAX", "ITEM_ROW_TOTAL",
            "SUBTOTAL", "VAT_AMOUNT", "GRAND_TOTAL",
        ],
        "footer": ["SUBTOTAL", "VAT_AMOUNT", "GRAND_TOTAL"],
    }

    zone_desc = {
        "header": "top-of-page → max(header GT).y2 + padding",
        "table":  "min(item GT).y1 - padding → max(grand_total GT).y2 + padding",
        "footer": "min(footer GT).y1 - padding → bottom-of-page",
    }

    print("\n" + "=" * 70)
    print(f"📋  AUTO_LABEL_TOKENS v4 — MODE: {mode.upper()}")
    print("=" * 70)
    print(f"  Entities      : {n_entity}")
    print(f"  Labels        : {n_labels}  (B/I × {n_entity} + O)")
    print(f"  Zone          : {zone_desc.get(mode, '?')}")
    print(f"  Overlap th    : {cfg.overlap_threshold}  "
          f"(fallback ×{cfg.fallback_overlap_scale})")
    print(f"  Y-overlap th  : {cfg.line_y_threshold}")
    print(f"  Text sim bonus: sim ≥ {cfg.text_sim_threshold} → +{cfg.text_sim_bonus}")
    print(f"  Contain bonus : tok⊂gt.text (len≥2) → +{cfg.containment_bonus}")
    print(f"  Zone padding  : {cfg.zone_padding_ratio*100:.0f}% of image height")
    print("-" * 70)

    for label in ORDER.get(mode, sorted(groups.keys())):
        if label not in groups:
            continue
        keys = groups[label]
        line = f"  {label:<30} ← {keys[0]}"
        if len(keys) > 1:
            line += f"  [+{len(keys)-1} alias: {', '.join(keys[1:])}]"
        print(line)

    print("=" * 70 + "\n")


# ======================================================================
# CONFIG LOADING
# ======================================================================
def load_yaml_config(path: Optional[str]) -> Dict[str, Any]:
    if not path:
        return {}
    if yaml is None:
        raise RuntimeError("PyYAML not installed. Install it or remove --config.")
    cfg_path = Path(path)
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config not found: {cfg_path}")
    data = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    return data or {}


def apply_schema_override(
    field_to_label: Dict[str, str],
    cfg_data:       Dict[str, Any],
    mode:           str,
) -> Dict[str, str]:
    override = cfg_data.get("schema_mappings", {}).get(mode, {})
    if not isinstance(override, dict):
        return field_to_label
    updated = dict(field_to_label)
    updated.update({k.strip().lower(): v for k, v in override.items()})
    return updated


# ======================================================================
# MAIN
# ======================================================================
def process_worker(args: Tuple[Any, ...]) -> Dict[str, Any]:
    try:
        return process_single_invoice(*args)
    except Exception as e:
        return {"status": "skipped", "invoice_id": args[0].stem, "reason": str(e)}


def main():
    p = argparse.ArgumentParser(
        description=(
            "Auto-label OCR tokens với BIO labels — v4 Production\n"
            "3 separate models, each with STRICT zone isolation:\n\n"
            "  --mode header : 12 entities, 25 labels\n"
            "                  Zone: invoice meta + seller + buyer + payment\n\n"
            "  --mode table  : 12 entities, 25 labels\n"
            "                  Zone: item rows + footer aggregates\n\n"
            "  --mode footer :  3 entities,  7 labels\n"
            "                  Zone: subtotal / vat_amount / grand_total rows ONLY\n\n"
            "Zone = [min_GT_y1 - padding, max_GT_y2 + padding]\n"
            "Tokens outside the zone are EXCLUDED before labeling."
        ),
        formatter_class=argparse.RawTextHelpFormatter,
    )
    p.add_argument("--mode",        required=True, choices=["header", "table", "footer"])
    p.add_argument("--gt_dir",      default=None)
    p.add_argument("--ocr_dir",     default=None)
    p.add_argument("--output_dir",  default=None)
    p.add_argument("--show_schema", action="store_true")
    p.add_argument("--config",      type=str,   default=None)
    p.add_argument("--dry-run",     action="store_true")
    p.add_argument("--stats-only",  action="store_true")
    p.add_argument("--min-overlap", type=float, default=None)
    p.add_argument("--max-tokens",  type=int,   default=None)
    p.add_argument("--zone-padding",type=float, default=None,
                   help="Zone padding ratio (0.0–0.2). Default: 0.06")
    p.add_argument("--force",       action="store_true")
    p.add_argument("--skip-exist",  action="store_true")
    p.add_argument("--workers",     type=int,   default=None)
    p.add_argument("--json-log",    action="store_true")
    args = p.parse_args()

    configure_logging(args.json_log)

    mode       = args.mode
    schema_cfg = _SCHEMAS[mode]
    cfg_data   = load_yaml_config(args.config)

    field_to_label = apply_schema_override(schema_cfg["field_to_label"], cfg_data, mode)
    schema_version = schema_cfg["schema_version"]
    valid_labels   = build_valid_labels(field_to_label)

    overlap_th          = cfg_data.get("OVERLAP_THRESHOLD",      DEFAULT_OVERLAP_THRESHOLD)
    line_y_th           = cfg_data.get("LINE_Y_THRESHOLD",       DEFAULT_LINE_Y_THRESHOLD)
    text_sim_th         = cfg_data.get("TEXT_SIM_THRESHOLD",     DEFAULT_TEXT_SIM_THRESHOLD)
    text_sim_bonus      = cfg_data.get("TEXT_SIM_BONUS",         DEFAULT_TEXT_SIM_BONUS)
    containment_bonus   = cfg_data.get("CONTAINMENT_BONUS",      DEFAULT_CONTAINMENT_BONUS)
    fallback_ovlp_scale = cfg_data.get("FALLBACK_OVERLAP_SCALE", DEFAULT_FALLBACK_OVERLAP_SCALE)
    zone_padding        = cfg_data.get("ZONE_PADDING_RATIO",     DEFAULT_ZONE_PADDING_RATIO)

    if args.min_overlap  is not None: overlap_th   = float(args.min_overlap)
    if args.zone_padding is not None: zone_padding = float(args.zone_padding)

    token_budget_warn  = int(cfg_data.get("TOKEN_BUDGET_WARN",  _TOKEN_BUDGET_WARN))
    token_budget_limit = int(cfg_data.get("TOKEN_BUDGET_LIMIT", _TOKEN_BUDGET_LIMIT))
    max_safe_tokens    = args.max_tokens or int(cfg_data.get("MAX_SAFE_TOKENS", _MAX_SAFE_TOKENS))

    paths_cfg  = cfg_data.get("paths", {})
    gt_dir     = Path(args.gt_dir     or paths_cfg.get("gt_dir")     or "output/annotations")
    ocr_dir    = Path(args.ocr_dir    or paths_cfg.get("ocr_dir")    or "output/ocr_results")
    output_dir = Path(args.output_dir or paths_cfg.get("output_dir") or schema_cfg["default_out"])

    workers = args.workers if args.workers is not None else min(8, cpu_count())

    app_cfg = AppConfig(
        match=MatchConfig(
            overlap_threshold      = float(overlap_th),
            line_y_threshold       = float(line_y_th),
            text_sim_threshold     = float(text_sim_th),
            text_sim_bonus         = float(text_sim_bonus),
            containment_bonus      = float(containment_bonus),
            fallback_overlap_scale = float(fallback_ovlp_scale),
            zone_padding_ratio     = float(zone_padding),
        ),
        token_budget_warn  = token_budget_warn,
        token_budget_limit = token_budget_limit,
        max_safe_tokens    = max_safe_tokens,
        workers            = workers,
        dry_run            = args.dry_run,
        stats_only         = args.stats_only,
        force              = args.force,
        skip_exist         = args.skip_exist,
        json_log           = args.json_log,
    )

    if args.show_schema:
        print_schema_summary(mode, field_to_label, app_cfg.match)
        return

    validate_schema(mode, field_to_label)
    print_schema_summary(mode, field_to_label, app_cfg.match)

    output_dir.mkdir(parents=True, exist_ok=True)

    gt_files = sorted(gt_dir.glob("*.json"))
    if not gt_files:
        logger.error(f"Không tìm thấy GT JSON nào trong '{gt_dir}'")
        return

    logger.info(f"Found {len(gt_files)} GT files  mode={mode}  → '{output_dir}'")

    total = {
        "processed": 0, "skipped": 0, "n_tokens": 0, "n_labeled": 0, "n_o": 0,
        "gt_skipped": 0, "fallback_lines": 0, "invalid_token_bbox": 0,
        "matched_gt_count": 0, "unmatched_token_count": 0, "n_dropped_by_trim": 0,
    }

    if app_cfg.stats_only:
        with tqdm(total=len(gt_files), desc=f"[{mode}] Stats", unit="inv") as pbar:
            for gt_path in gt_files:
                res = compute_stats_only(gt_path, ocr_dir / f"{gt_path.stem}.json")
                if res["status"] == "processed":
                    total["processed"]          += 1
                    total["n_tokens"]           += res.get("n_tokens", 0)
                    total["fallback_lines"]     += res.get("fallback_lines", 0)
                    total["invalid_token_bbox"] += res.get("invalid_token_bbox", 0)
                else:
                    total["skipped"] += 1
                pbar.update(1)
        _print_summary(mode, total, output_dir, stats_only=True)
        return

    worker_args = [
        (
            gt_path,
            ocr_dir    / f"{gt_path.stem}.json",
            output_dir / f"{gt_path.stem}.json",
            field_to_label, valid_labels, schema_version, mode, app_cfg,
        )
        for gt_path in gt_files
    ]

    if app_cfg.workers <= 1:
        with tqdm(total=len(worker_args), desc=f"[{mode}] Labeling", unit="inv") as pbar:
            for arg in worker_args:
                res = process_worker(arg)
                _accumulate(total, res)
                pbar.update(1)
    else:
        with Pool(processes=app_cfg.workers) as pool:
            with tqdm(total=len(worker_args), desc=f"[{mode}] Labeling", unit="inv") as pbar:
                for res in pool.imap_unordered(process_worker, worker_args):
                    _accumulate(total, res)
                    pbar.update(1)

    _print_summary(mode, total, output_dir)


def _accumulate(total: Dict, res: Dict) -> None:
    if res["status"] == "processed":
        total["processed"]            += 1
        total["n_tokens"]             += res.get("n_tokens", 0)
        total["n_labeled"]            += res.get("n_labeled", 0)
        total["n_o"]                  += res.get("n_o", 0)
        total["gt_skipped"]           += res.get("gt_skipped", 0)
        total["fallback_lines"]       += res.get("fallback_lines", 0)
        total["invalid_token_bbox"]   += res.get("invalid_token_bbox", 0)
        total["matched_gt_count"]     += res.get("matched_gt_count", 0)
        total["unmatched_token_count"]+= res.get("unmatched_token_count", 0)
        total["n_dropped_by_trim"]    += res.get("n_dropped_by_trim", 0)
    else:
        total["skipped"] += 1
        logger.warning(
            f"[{res.get('invoice_id')}] skipped: {res.get('reason')}",
            extra={"event": "skip", "reason": res.get("reason")},
        )


def _print_summary(
    mode:       str,
    total:      Dict,
    output_dir: Path,
    stats_only: bool = False,
) -> None:
    n           = max(1, total["n_tokens"])
    label_ratio = total["n_labeled"] / n

    tag = "STATS ONLY" if stats_only else "COMPLETE"
    print("\n" + "=" * 70)
    print(f"✅  {tag} v4 — mode: {mode.upper()}")
    print("=" * 70)
    print(f"  Processed      : {total['processed']}")
    print(f"  Skipped        : {total['skipped']}")
    print(f"  Total tokens   : {total['n_tokens']:,}")
    if not stats_only:
        print(f"  Labeled B/I    : {total['n_labeled']:,}  ({label_ratio:.1%})")
        print(f"  O (outside)    : {total['n_o']:,}  ({1-label_ratio:.1%})")
        print(f"  GT skipped     : {total['gt_skipped']:,}")
        print(f"  Matched GT     : {total['matched_gt_count']:,}")
        print(f"  Dropped (trim) : {total['n_dropped_by_trim']:,}")
    print(f"  Fallback lines : {total['fallback_lines']:,}")
    print(f"  Invalid bbox   : {total['invalid_token_bbox']:,}")
    print(f"  Output dir     : {output_dir}")
    print("=" * 70)

    if not stats_only:
        expected_range = {
            "header": (0.10, 0.50),
            "table":  (0.15, 0.60),
            "footer": (0.40, 0.90),
        }.get(mode, (0.05, 0.80))

        lo, hi = expected_range
        if label_ratio < lo:
            logger.warning(
                f"⚠️  Labeled ratio {label_ratio:.1%} thấp hơn kỳ vọng ({lo:.0%}). "
                f"Kiểm tra: (1) GT bbox có đúng vùng? "
                f"(2) OVERLAP_THRESHOLD có quá cao? "
                f"(3) Schema có đủ aliases?"
            )
        elif label_ratio > hi:
            logger.warning(
                f"⚠️  Labeled ratio {label_ratio:.1%} cao hơn kỳ vọng ({hi:.0%}). "
                f"Zone filter có đang quá rộng không?"
            )
        else:
            logger.info(f"✅  Labeled ratio {label_ratio:.1%} — trong kỳ vọng ({lo:.0%}–{hi:.0%}).")

        if total["skipped"] > 0:
            logger.warning(f"⚠️  {total['skipped']} invoice bị skip.")
        if total["fallback_lines"] > 0:
            logger.warning(
                f"⚠️  {total['fallback_lines']:,} fallback lines còn tồn tại. "
                f"Upgrade run_ocr.py → re-run OCR."
            )


if __name__ == "__main__":
    main()