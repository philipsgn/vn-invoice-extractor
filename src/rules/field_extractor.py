# -*- coding: utf-8 -*-
from typing import Any, Dict, List, Optional, Tuple
from ..normalizers.ocr_normalizer import normalize_ocr_text
from ..engine.table_row_rebuilder import TableRowRebuilder

# ======================================================
# LABEL → FIELD PATH MAPPING
# ======================================================

ENTITY_TO_FIELD: Dict[str, Tuple[str, ...]] = {
    "SELLER_NAME":               ("seller",  "name"),
    "SELLER_TAX_CODE":           ("seller",  "tax_code"),
    "SELLER_TAX_AUTHORITY_CODE": ("seller",  "tax_authority_code"),
    "SELLER_ADDRESS":            ("seller",  "address"),
    "BUYER_NAME":      ("buyer",   "name"),
    "BUYER_TAX_CODE":  ("buyer",   "tax_code"),
    "BUYER_ADDRESS":   ("buyer",   "address"),
    "INVOICE_NUMBER":  ("invoice", "number"),
    "INVOICE_DATE":    ("invoice", "date"),
    "INVOICE_SYMBOL":  ("invoice", "symbol"),
    "INVOICE_TYPE":    ("invoice", "type"),
    "PAYMENT_METHOD":  ("invoice", "payment_method"),
    "GRAND_TOTAL":   ("invoice", "total_amount"),
    "VAT_AMOUNT":    ("invoice", "vat_amount"),
    "SUBTOTAL":      ("invoice", "subtotal"),
    "ITEM_VAT_RATE": ("invoice", "vat_rate"),
    "VAT_RATE":      ("invoice", "vat_rate"),
}

ITEM_ENTITIES = {
    "ITEM_NAME", "ITEM_UNIT", "ITEM_QUANTITY",
    "ITEM_UNIT_PRICE", "ITEM_TOTAL_PRICE",
    "ITEM_DISCOUNT", "ITEM_LINE_TAX",
    "ITEM_ROW_TOTAL", "ITEM_VAT_RATE", "VAT_RATE",
}

ITEM_FIELD_MAP = {
    "ITEM_UNIT":        "unit",
    "ITEM_QUANTITY":    "quantity",
    "ITEM_UNIT_PRICE":  "unit_price",
    "ITEM_TOTAL_PRICE": "total",
    "ITEM_DISCOUNT":    "discount",
    "ITEM_VAT_RATE":    "vat_rate",
    "VAT_RATE":         "vat_rate",
    "ITEM_LINE_TAX":    "line_tax",
    "ITEM_ROW_TOTAL":   "row_total",
}

FOOTER_Y_THRESHOLD = 870

_COMPANY_KEYWORDS = (
    "công ty", "tnhh", "cổ phần", "doanh nghiệp", "tập đoàn",
    "chi nhánh", "văn phòng", "co.", "ltd", "corp", "inc",
    "joint", "enterprise",
)

# ======================================================
# SPAN DATACLASS
# ======================================================

class Span:
    __slots__ = ("entity", "tokens", "bboxes", "confs")
    def __init__(self, entity: str):
        self.entity: str             = entity
        self.tokens: List[str]       = []
        self.bboxes: List[List[int]] = []
        self.confs:  List[float]     = []

    def add(self, token: str, bbox: List[int], conf: float):
        self.tokens.append(token)
        self.bboxes.append(bbox)
        self.confs.append(conf)

    @property
    def text(self) -> str:
        # Normalize double spaces that might occur during merging
        return " ".join(" ".join(self.tokens).split())

    @property
    def avg_conf(self) -> float:
        return sum(self.confs) / len(self.confs) if self.confs else 0.0

    @property
    def n_tokens(self) -> int:
        return len(self.tokens)

    @property
    def y_center(self) -> float:
        if not self.bboxes: return 0.0
        return sum((b[1] + b[3]) / 2 for b in self.bboxes) / len(self.bboxes)

    @property
    def is_footer(self) -> bool:
        if not self.bboxes: return False
        avg_y1 = sum(b[1] for b in self.bboxes) / len(self.bboxes)
        return avg_y1 > FOOTER_Y_THRESHOLD

    def score(self) -> Tuple[int, float]:
        return (self.n_tokens, self.avg_conf)

def extract_spans(ner_output: Dict[str, Any]) -> Dict[str, List[Span]]:
    tokens  = ner_output.get("tokens", [])
    bboxes  = ner_output.get("bboxes", [])
    labels  = ner_output.get("predicted_labels", [])
    confs   = ner_output.get("confidence", [])
    n = len(tokens)
    if len(bboxes) < n: bboxes = bboxes + [[0,0,0,0]] * (n - len(bboxes))
    if len(confs) < n: confs = confs + [1.0] * (n - len(confs))
    if len(labels) < n: labels = labels + ["O"] * (n - len(labels))

    spans_by_entity: Dict[str, List[Span]] = {}
    current_span: Optional[Span] = None
    for tok, bbox, lbl, conf in zip(tokens, bboxes, labels, confs):
        if "-" not in lbl:
            if current_span: spans_by_entity.setdefault(current_span.entity, []).append(current_span)
            current_span = None
            continue
        tag, entity = lbl.split("-", 1)
        if tag == "B":
            if current_span: spans_by_entity.setdefault(current_span.entity, []).append(current_span)
            current_span = Span(entity)
            current_span.add(tok, bbox, float(conf))
        elif tag == "I" and current_span and current_span.entity == entity:
            current_span.add(tok, bbox, float(conf))
    if current_span: spans_by_entity.setdefault(current_span.entity, []).append(current_span)
    return spans_by_entity

def merge_spans(spans: List[Span]) -> Optional[Span]:
    if not spans: return None
    # Sort by Y (line) then X (horizontal) to handle multi-word fields correctly.
    # Grouping by rough Y center (rounded to 10px) helps with line jitter.
    def sort_key(s):
        first_bbox = s.bboxes[0] if s.bboxes else [0,0,0,0]
        return (round(s.y_center / 8), first_bbox[0])

    candidates = [s for s in spans if not s.is_footer] or spans
    sorted_spans = sorted(candidates, key=sort_key)
    merged = Span(candidates[0].entity)
    for s in sorted_spans:
        for t, b, c in zip(s.tokens, s.bboxes, s.confs):
            merged.add(normalize_ocr_text(t), b, c)
    return merged

def _is_company_name(text: str) -> bool:
    return any(kw in text.lower() for kw in _COMPANY_KEYWORDS)

def _split_buyer_name_spans(spans: List[Span]) -> Tuple[Optional[str], Optional[str]]:
    if not spans: return None, None
    candidates = [s for s in spans if not s.is_footer] or spans
    if len(candidates) == 1:
        text = normalize_ocr_text(candidates[0].text)
        return (None, text) if _is_company_name(text) else (text, None)
    
    company_spans = [s for s in candidates if _is_company_name(s.text)]
    personal_spans = [s for s in candidates if not _is_company_name(s.text)]
    
    comp = max(company_spans, key=lambda s: s.score()).text if company_spans else None
    pers = max(personal_spans, key=lambda s: s.score()).text if personal_spans else None
    return normalize_ocr_text(pers) if pers else None, normalize_ocr_text(comp) if comp else None

# ======================================================
# ITEM ROW ASSEMBLY
# ======================================================

def _group_item_spans_into_rows(spans_by_entity: Dict[str, List[Span]], ner_output: Dict[str, Any]) -> List[Dict[str, Any]]:
    tokens = []
    word_list, bbox_list, label_list, conf_list = ner_output["tokens"], ner_output["bboxes"], ner_output["predicted_labels"], ner_output["confidence"]
    for i, (text, bbox, lbl, conf) in enumerate(zip(word_list, bbox_list, label_list, conf_list)):
        entity = lbl.split("-")[-1]
        if lbl != "O" and (entity in ITEM_ENTITIES or entity == "ITEM_NAME"):
            tokens.append({"text": text, "bbox": bbox, "label": lbl, "conf": conf, "index": i})
    
    if not tokens: return []
    rebuilder = TableRowRebuilder()
    clustered_rows = rebuilder.rebuild_rows(tokens)
    raw_rows = []
    for cluster in clustered_rows:
        row = {"name": "", "unit": "", "quantity": "", "unit_price": "", "total": "", "discount": "", "vat_rate": "", "line_tax": "", "row_total": "", "_conf": {}}
        for t in cluster:
            lbl = t["label"].split("-")[-1]
            text = normalize_ocr_text(t["text"])
            if lbl == "ITEM_NAME": row["name"] = (row["name"] + " " + text).strip()
            elif lbl in ITEM_FIELD_MAP: row[ITEM_FIELD_MAP[lbl]] = text
            fk = lbl.replace("ITEM_", "").lower()
            if fk == "row_total": fk = "total"
            row["_conf"][fk] = round(t["conf"], 4)
        if row["name"] or any(row[k] for k in ["quantity", "unit_price", "total"]): raw_rows.append(row)
    return rebuilder.merge_multiline_items(raw_rows)

class FieldExtractor:
    def extract_all(self, ner_output: Dict[str, Any]) -> Dict[str, Any]:
        spans_by_entity = extract_spans(ner_output)
        result = {"seller": {}, "buyer": {}, "invoice": {}, "items": [], "confidence": {}}
        field_conf = {}
        for entity, path in ENTITY_TO_FIELD.items():
            if entity == "BUYER_NAME": continue
            merged = merge_spans(spans_by_entity.get(entity, []))
            if merged:
                sec, *sub = path
                result[sec][sub[0]] = merged.text
                field_conf[f"{sec}.{sub[0]}"] = merged.avg_conf
        
        fn, cn = _split_buyer_name_spans(spans_by_entity.get("BUYER_NAME", []))
        if fn: result["buyer"]["full_name"] = fn
        if cn: result["buyer"]["name"] = cn
        elif fn: result["buyer"]["name"] = fn

        result["items"] = _group_item_spans_into_rows(spans_by_entity, ner_output)
        result["confidence"] = {"fields": field_conf}
        return result
