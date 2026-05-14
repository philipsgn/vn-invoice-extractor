# -*- coding: utf-8 -*-
"""
bio_repair_inference.py
======================================================
POST-PROCESSING PATCH CHO INFERENCE (test_real_invoices.py)
======================================================
DÃ¹ng ngay bÃ¢y giá» mÃ  KHÃ”NG Cáº¦N RETRAIN model.

Váº¥n Ä‘á» cáº§n fix táº¡i inference:
  1. BIO sequence bá»‹ broken (I-tag khÃ´ng cÃ³ B-tag trÆ°á»›c)
  2. Prefix tokens bá»‹ label nháº§m vÃ o entity (vd "hÃ ng:" â†’ B-SELLER_NAME)
  3. Token bbox chia sai
  4. Field extraction láº¥y cáº£ prefix text vÃ o value
  5. OCR merge label+value thÃ nh 1 token (vd "thuÃª:0108892073")

CÃCH DÃ™NG trong test_real_invoices.py:
  from bio_repair_inference import (
      repair_bio_sequence,
      strip_field_prefixes,
      strip_all_field_prefixes,
      strip_nested_prefixes,
      prepare_words_and_bboxes,
  )
======================================================
"""

import re
import unicodedata
from typing import List, Dict, Optional, Tuple


# ======================================================
# VIETNAMESE PREFIX PATTERNS
# ======================================================

FIELD_PREFIXES: Dict[str, List[str]] = {
    "seller_name": [
        "Ä‘Æ¡n vá»‹ bÃ¡n hÃ ng", "bÃ¡n hÃ ng", "tÃªn Ä‘Æ¡n vá»‹", "tÃªn cÃ´ng ty",
    ],
    "seller_tax_code": [
        "mÃ£ sá»‘ thuáº¿", "mst", "sá»‘ thuáº¿",
    ],
    "seller_address": [
        "Ä‘á»‹a chá»‰ cÆ¡ sá»Ÿ", "Ä‘á»‹a chá»‰", "Ä‘ia chi",
    ],
    "seller_phone": [
        "Ä‘iá»‡n thoáº¡i", "tel", "phone", "Ä‘t",
    ],
    "buyer_name": [
        "tÃªn ngÆ°á»i mua hÃ ng", "ngÆ°á»i mua hÃ ng", "tÃªn ngÆ°á»i mua",
        "ngÆ°á»i mua", "tÃªn Ä‘Æ¡n vá»‹", "tÃªn cÃ´ng ty",
    ],
    "buyer_tax_code": [
        "mÃ£ sá»‘ thuáº¿", "mst", "sá»‘ thuáº¿",
    ],
    "buyer_address": [
        "Ä‘á»‹a chá»‰", "dia chi",
    ],
    "invoice_number": [
        "sá»‘ hÃ³a Ä‘Æ¡n", "sá»‘ hÄ‘", "hÃ³a Ä‘Æ¡n sá»‘",
        # âœ… KHÃ”NG thÃªm "sá»‘" Ä‘Æ¡n thuáº§n â€” trÃ¡nh cáº¯t nháº§m
    ],
    "invoice_symbol": [
        "kÃ½ hiá»‡u", "kÃ­ hiá»‡u",
    ],
    "invoice_date": [
        "ngÃ y láº­p",
        # âœ… KHÃ”NG thÃªm "ngÃ y" Ä‘Æ¡n thuáº§n â€” "ngÃ y 01 thÃ¡ng 10" cáº§n giá»¯ nguyÃªn
    ],
    "payment_method": [
        "hÃ¬nh thá»©c thanh toÃ¡n", "phÆ°Æ¡ng thá»©c thanh toÃ¡n",
        "hÃ¬nh thá»©c", "thanh toÃ¡n",
    ],
    "total_amount": [
        "tá»•ng tiá»n thanh toÃ¡n", "tá»•ng cá»™ng", "tá»•ng tiá»n", "cá»™ng tiá»n hÃ ng",
    ],
    "vat_rate": [
        "thuáº¿ suáº¥t thuáº¿ gtgt", "thuáº¿ suáº¥t", "thuáº¿ gtgt", "gtgt",
    ],
    "vat_amount": [
        "tiá»n thuáº¿ gtgt", "tiá»n thuáº¿", "thuáº¿ gtgt",
    ],
}

# âœ… Chá»‰ giá»¯ token CÃ“ Dáº¤U : hoáº·c abbreviation cháº¯c cháº¯n lÃ  label header
# KHÃ”NG bao gá»“m: "sá»‘", "ngÃ y", "thÃ¡ng", "nÄƒm", "mÃ£", "thanh", "toÃ¡n"
PURE_LABEL_TOKENS = {
    "stt",
    "Ä‘vt",
    "mst:",     # cÃ³ dáº¥u : â†’ cháº¯c cháº¯n lÃ  label header
    "cqt:",
    "hÃ ng:",    # "hÃ ng:" cÃ³ dáº¥u : â†’ label header
    "thuáº¿:",
    "chá»‰",      # "Äá»‹a chá»‰" â€” "chá»‰" standalone khÃ´ng bao giá» lÃ  value
}


# ======================================================
# TEXT NORMALIZATION  (internal only)
# ======================================================

def _normalize(text: str) -> str:
    """Normalize cho prefix matching."""
    text = unicodedata.normalize("NFC", text)
    return text.lower().strip().rstrip(":")


# ======================================================
# 1. BIO SEQUENCE REPAIR
# ======================================================

def repair_bio_sequence(
    labels:    List[str],
    words:     Optional[List[str]]       = None,
    bboxes:    Optional[List[List[int]]] = None,
    max_y_gap: int = 30,
) -> List[str]:
    """
    Sá»­a BIO sequence sau model inference.

    CÃ¡c vi pháº¡m Ä‘Æ°á»£c fix:
      (a) I-TAG khÃ´ng cÃ³ B-TAG trÆ°á»›c â†’ promote thÃ nh B-TAG
      (b) I-FIELD_A sau B/I-FIELD_B (field khÃ¡c) â†’ promote thÃ nh B-FIELD_A
      (c) Token lÃ  PURE_LABEL_TOKEN + B-tag â†’ force O
      (d) Spatial y-gap lá»›n â†’ force B (new span)

    Args:
        labels:     list labels tá»« model
        words:      optional â€” list token text tÆ°Æ¡ng á»©ng
        bboxes:     optional â€” list bbox [x1,y1,x2,y2] normalized [0,1000]
        max_y_gap:  khoáº£ng cÃ¡ch y tá»‘i Ä‘a (normalized) Ä‘á»ƒ coi lÃ  cÃ¹ng entity

    Returns:
        labels Ä‘Ã£ Ä‘Æ°á»£c sá»­a
    """
    if not labels:
        return labels

    repaired     = list(labels)
    prev_entity: Optional[str]       = None
    prev_bbox:   Optional[List[int]] = None

    for i, label in enumerate(repaired):
        word = words[i]  if words  and i < len(words)  else ""
        bbox = bboxes[i] if bboxes and i < len(bboxes) else None

        # â”€â”€ O: reset state â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        if label == "O":
            prev_entity = None
            prev_bbox   = None
            continue

        # â”€â”€ Format khÃ´ng há»£p lá»‡ â†’ O â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        if "-" not in label:
            repaired[i] = "O"
            prev_entity = None
            prev_bbox   = None
            continue

        tag, entity = label.split("-", 1)

        # â”€â”€ (c) Pure label token + B-tag â†’ force O â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        if tag == "B" and word and _normalize(word) in PURE_LABEL_TOKENS:
            repaired[i] = "O"
            prev_entity = None
            prev_bbox   = None
            continue

        # â”€â”€ B-tag: báº¯t Ä‘áº§u entity má»›i â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        if tag == "B":
            prev_entity = entity
            prev_bbox   = bbox
            continue

        # â”€â”€ I-tag: kiá»ƒm tra tÃ­nh há»£p lá»‡ â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        if tag == "I":
            # (a) Orphan I â†’ promote B
            if prev_entity is None:
                repaired[i] = f"B-{entity}"
                prev_entity = entity
                prev_bbox   = bbox
                continue

            # (b) I cá»§a entity khÃ¡c â†’ promote B
            if entity != prev_entity:
                repaired[i] = f"B-{entity}"
                prev_entity = entity
                prev_bbox   = bbox
                continue

            # (d) Spatial gap break
            if bbox is not None and prev_bbox is not None:
                y_curr = (bbox[1]      + bbox[3])      / 2
                y_prev = (prev_bbox[1] + prev_bbox[3]) / 2
                if abs(y_curr - y_prev) > max_y_gap:
                    repaired[i] = f"B-{entity}"
                    prev_entity = entity
                    prev_bbox   = bbox
                    continue

            prev_bbox = bbox

    return repaired


# ======================================================
# 2. STRIP FIELD PREFIXES
# ======================================================

def strip_field_prefixes(text: str, field_name: str) -> str:
    """
    Loáº¡i bá» prefix phá»• biáº¿n cá»§a field khá»i text Ä‘Ã£ extract.

    VÃ­ dá»¥:
        "hÃ ng: CÃ”NG TY Káº¾ TOÃN THIÃŠN Æ¯NG" â†’ "CÃ”NG TY Káº¾ TOÃN THIÃŠN Æ¯NG"
        "MÃ£ sá»‘ thuáº¿: 0108892073"           â†’ "0108892073"
    """
    if not text:
        return text

    prefixes_sorted = sorted(
        FIELD_PREFIXES.get(field_name, []),
        key=len, reverse=True,
    )

    text_norm = _normalize(text)

    for prefix in prefixes_sorted:
        if text_norm.startswith(_normalize(prefix)):
            result = text[len(prefix):].lstrip(": \t")
            if result:
                return result.strip()

    return text.strip()


def strip_all_field_prefixes(extracted: Dict) -> Dict:
    """Strip prefix cho flat dict."""
    return {
        field: strip_field_prefixes(value, field) if isinstance(value, str) else value
        for field, value in extracted.items()
    }


def strip_nested_prefixes(data: Dict) -> Dict:
    """
    Strip prefix cho nested dict va flat dict tu engine output.
    """
    result = {}
    for key, value in data.items():
        if isinstance(value, dict):
            result[key] = {
                sub_k: strip_field_prefixes(
                    sub_v, 
                    f"{key}_{sub_k}" if f"{key}_{sub_k}" in FIELD_PREFIXES else sub_k
                ) if isinstance(sub_v, str) else sub_v
                for sub_k, sub_v in value.items()
            }
        elif isinstance(value, str):
            result[key] =                 strip_field_prefixes(value, key)
        else:
            result[key] = value
    return result


# ======================================================
# 3. PROPORTIONAL TOKEN BBOXES
# ======================================================

def proportional_token_bboxes(
    line_bbox:    List[int],
    tokens:       List[str],
    image_width:  int,
    image_height: int,
) -> List[List[int]]:
    """
    Chia line bbox thÃ nh per-token bboxes theo tá»‰ lá»‡ sá»‘ kÃ½ tá»±,
    normalize vá» [0, 1000] cho LayoutLMv3.
    """
    if not tokens:
        return []

    lx1, ly1, lx2, ly2 = line_bbox
    lx1, lx2   = min(lx1, lx2), max(lx1, lx2)
    ly1, ly2   = min(ly1, ly2), max(ly1, ly2)
    line_width = max(1, lx2 - lx1)

    char_counts = [max(1, len(t)) for t in tokens]
    total_chars = sum(char_counts)

    result = []
    cursor = float(lx1)

    for i, (token, chars) in enumerate(zip(tokens, char_counts)):
        tok_x1 = int(cursor)
        tok_x2 = lx2 if i == len(tokens) - 1 else int(cursor + line_width * chars / total_chars)

        norm = [
            max(0, min(1000, int(1000 * tok_x1 / max(1, image_width)))),
            max(0, min(1000, int(1000 * ly1    / max(1, image_height)))),
            max(0, min(1000, int(1000 * tok_x2 / max(1, image_width)))),
            max(0, min(1000, int(1000 * ly2    / max(1, image_height)))),
        ]
        result.append(norm)
        cursor += line_width * chars / total_chars

    return result


# ======================================================
# 4. SPLIT OCR-MERGED TOKEN  (vd "thuÃª:0108892073")
# ======================================================

_LABEL_VALUE_RE = re.compile(r"^([^\d:]+:)(\S{3,})$")


def _split_label_value_token(token) -> List[str]:
    """
    Split merged label:value tokens safely.
    Example: "MST:0108892073" -> ["MST:", "0108892073"]
    """
    if token is None:
        return []

    if not isinstance(token, str):
        try:
            token = str(token)
        except Exception:
            return []

    token = token.strip()
    if not token:
        return []

    match = _LABEL_VALUE_RE.match(token)
    if match:
        return [match.group(1), match.group(2)]

    return [token]


# ======================================================
# 5. PREPARE WORDS & BBOXES - Drop-in cho inference
# ======================================================

def prepare_words_and_bboxes(
    ocr_result: Dict,
    image_width: int,
    image_height: int,
) -> Tuple[List[str], List[List[int]]]:
    """
    Chuẩn bị words + bboxes từ OCR output cho LayoutLMv3.

    Supports BOTH token formats:
      - List[str]
      - List[{"text": str, "bbox": [x1,y1,x2,y2]}]
    """
    words: List[str] = []
    bboxes: List[List[int]] = []

    def _safe_line_bbox(raw) -> Optional[List[int]]:
        if not isinstance(raw, (list, tuple)) or len(raw) != 4:
            return None
        try:
            x1, y1, x2, y2 = (float(raw[0]), float(raw[1]), float(raw[2]), float(raw[3]))
        except Exception:
            return None
        if image_width <= 0 or image_height <= 0:
            return None
        x1, x2 = (x1, x2) if x1 <= x2 else (x2, x1)
        y1, y2 = (y1, y2) if y1 <= y2 else (y2, y1)
        return [int(x1), int(y1), int(x2), int(y2)]

    def _normalize_bbox(raw) -> Optional[List[int]]:
        if not isinstance(raw, (list, tuple)) or len(raw) != 4:
            return None
        try:
            x1, y1, x2, y2 = (float(raw[0]), float(raw[1]), float(raw[2]), float(raw[3]))
        except Exception:
            return None
        if image_width <= 0 or image_height <= 0:
            return None
        x1, x2 = (x1, x2) if x1 <= x2 else (x2, x1)
        y1, y2 = (y1, y2) if y1 <= y2 else (y2, y1)
        return [
            max(0, min(1000, int(1000 * x1 / image_width))),
            max(0, min(1000, int(1000 * y1 / image_height))),
            max(0, min(1000, int(1000 * x2 / image_width))),
            max(0, min(1000, int(1000 * y2 / image_height))),
        ]

    def _coerce_text(raw) -> Optional[str]:
        if raw is None:
            return None
        if isinstance(raw, dict):
            raw = raw.get("text", "")
        elif isinstance(raw, (list, tuple)):
            raw = raw[0] if raw else ""
        if not isinstance(raw, str):
            try:
                raw = str(raw)
            except Exception:
                return None
        raw = raw.strip()
        return raw if raw else None

    for line in ocr_result.get("lines", []) if isinstance(ocr_result, dict) else []:
        line_bbox = _safe_line_bbox(line.get("bbox") if isinstance(line, dict) else None)
        tokens = line.get("tokens", []) if isinstance(line, dict) else []
        if not isinstance(tokens, (list, tuple)) or not tokens:
            continue

        expanded_texts: List[str] = []
        override_bboxes: List[Optional[List[int]]] = []

        for tok in tokens:
            tok_text = _coerce_text(tok)
            if not tok_text:
                continue

            parts = _split_label_value_token(tok_text)
            if not parts:
                continue

            tok_bbox = None
            if isinstance(tok, dict):
                tok_bbox = _normalize_bbox(tok.get("bbox"))

            for part in parts:
                part = part.strip()
                if not part:
                    continue
                expanded_texts.append(part)
                override_bboxes.append(tok_bbox)

        if not expanded_texts:
            continue

        if line_bbox is not None:
            prop_bboxes = proportional_token_bboxes(
                line_bbox=line_bbox,
                tokens=expanded_texts,
                image_width=image_width,
                image_height=image_height,
            )
        else:
            prop_bboxes = [[0, 0, 0, 0] for _ in expanded_texts]

        for text, prop_bbox, override_bbox in zip(expanded_texts, prop_bboxes, override_bboxes):
            words.append(text)
            bboxes.append(override_bbox if override_bbox is not None else prop_bbox)

    return words, bboxes

# ======================================================
# 6. COMPANY NAME NORMALIZATION
# ======================================================

COMMON_OCR_CONFUSIONS = {
    "Æ¯NG": "ƯNG",
    "Æ¯": "Ư",
}

def normalize_company_name(name: str) -> str:
    if not name:
        return name

    name = unicodedata.normalize("NFC", str(name))
    name = name.strip()

    # Uppercase toàn bộ (để evaluation match GT)
    name = name.upper()

    # Fix common OCR confusions
    for garbled, correct in COMMON_OCR_CONFUSIONS.items():
        name = name.replace(garbled.upper(), correct.upper())

    # Remove duplicated spaces
    name = re.sub(r"\s+", " ", name)

    return name.strip()

def clean_buyer_name(buyer_name: str, seller_name: str) -> str:
    if not buyer_name:
        return buyer_name

    buyer = buyer_name.upper().strip()
    seller = seller_name.upper().strip() if seller_name else ""

    # 1ï¸âƒ£ Remove seller full náº¿u cÃ³
    if seller and seller in buyer:
        buyer = buyer.replace(seller, "").strip()

    # 2ï¸âƒ£ Remove seller suffix (3-4 tá»« cuá»‘i)
    if seller:
        seller_words = seller.split()
        for n in range(3, min(6, len(seller_words)) + 1):
            suffix = " ".join(seller_words[-n:])
            if buyer.endswith(suffix):
                buyer = buyer[:-len(suffix)].strip()
                break

    # 3ï¸âƒ£ Remove trailing orphan word (vd: "KÃŠ")
    tokens = buyer.split()
    if len(tokens) > 3:
        last = tokens[-1]

        # Náº¿u tá»« cuá»‘i ngáº¯n vÃ  khÃ´ng pháº£i loáº¡i hÃ¬nh doanh nghiá»‡p
        if len(last) <= 3 and last not in {"TNHH", "CP", "MTV"}:
            buyer = " ".join(tokens[:-1])

    # 4ï¸âƒ£ Cleanup
    buyer = re.sub(r"\s+", " ", buyer)

    return buyer.strip()
# ======================================================
# 7. CLEAN COMPANY FIELDS (drop-in for inference)
# ======================================================

def clean_company_fields(data: Dict) -> Dict:
    """
    Apply:
      - normalize_company_name
      - clean_buyer_name
    """
    if "seller" not in data or "buyer" not in data:
        return data

    seller_name = data.get("seller", {}).get("name", "")
    buyer_name  = data.get("buyer", {}).get("name", "")

    # Normalize seller
    seller_name = normalize_company_name(seller_name)

    # Normalize buyer trÆ°á»›c
    buyer_name = normalize_company_name(buyer_name)

    # Remove seller from buyer náº¿u bá»‹ dÃ­nh
    buyer_name = clean_buyer_name(buyer_name, seller_name)

    data["seller"]["name"] = seller_name
    data["buyer"]["name"]  = buyer_name

    return data


# ======================================================
# QUICK TEST
# ======================================================

if __name__ == "__main__":
    print("=" * 60)
    print("TEST: repair_bio_sequence")
    print("=" * 60)

    labels_broken = [
        "B-INVOICE_TYPE", "I-INVOICE_TYPE",
        "B-INVOICE_SYMBOL",
        "I-INVOICE_TYPE", "I-INVOICE_TYPE",   # Orphan I â†’ B
        "O",
        "B-SELLER_TAX_CODE", "I-SELLER_TAX_CODE",
        "I-INVOICE_DATE", "I-INVOICE_DATE",    # Orphan I â†’ B
        "O",
        "B-SELLER_NAME",
        "I-SELLER_NAME", "I-SELLER_NAME",
    ]
    words_sample = [
        "HÃ“A", "ÄÆ N",
        "hiá»‡u:",
        "GIÃ", "TRá»Š",
        "MÃƒ",
        "cqt:", "596SRE",
        "16", "thÃ¡ng",
        "MÃƒ",
        "hÃ ng:",
        "cÃ´ng", "ty",
    ]

    repaired = repair_bio_sequence(labels_broken, words=words_sample)
    print(f"{'Word':<20} {'Original':<28} {'Repaired':<28} {'Changed?'}")
    print("-" * 85)
    for w, orig, rep in zip(words_sample, labels_broken, repaired):
        changed = "âš ï¸  CHANGED" if orig != rep else ""
        print(f"{w:<20} {orig:<28} {rep:<28} {changed}")

    print("\n" + "=" * 60)
    print("TEST: strip_field_prefixes")
    print("=" * 60)
    for raw, field in [
        ("hÃ ng: CÃ”NG TY Káº¾ TOÃN THIÃŠN Æ¯NG",   "seller_name"),
        ("MÃ£ sá»‘ thuáº¿:0108892073",               "seller_tax_code"),
        ("HÃ¬nh thá»©c thanh toÃ¡n: Chuyá»ƒn khoáº£n", "payment_method"),
        ("CÃ”NG TY Káº¾ TOÃN THIÃŠN Æ¯NG",          "seller_name"),
    ]:
        stripped = strip_field_prefixes(raw, field)
        status   = "âœ…" if stripped != raw else "âž¡ï¸ "
        print(f"  {status} [{field}]  {raw!r} â†’ {stripped!r}")

    print("\n" + "=" * 60)
    print("TEST: _split_label_value_token")
    print("=" * 60)
    for tok in ["thuÃª:0108892073", "MST:0108892073", "Sá»‘:", "596SRE"]:
        print(f"  {tok!r:25} â†’ {_split_label_value_token(tok)}")
