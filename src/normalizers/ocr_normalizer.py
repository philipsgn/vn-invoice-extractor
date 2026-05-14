# -*- coding: utf-8 -*-
import unicodedata
import re

def normalize_ocr_text(text: str) -> str:
    """
    Production-grade OCR text normalization for Vietnamese invoices.
    
    Fixes:
    - Unicode NFC normalization
    - Common Vietnamese OCR artifacts (mangled characters)
    - Repeated whitespace
    - Specific common misrecognitions (TNHHI, VU -> VỤ, etc.)
    """
    if not text:
        return ""

    # 1. Unicode NFC Normalization
    text = unicodedata.normalize("NFC", text)

    # 2. Rule-based Corrections for Vietnamese OCR Artifacts
    corrections = {
        "Æ¯": "Ư",
        "Ã": "Á",
        "TNHHI": "TNHH",
        "VU ": "VỤ ",
        "CO PHAN": "CỔ PHẦN",
        "LIEN": "LIÊN",
        "KHOAN": "KHOẢN",
        "NGAY": "NGÀY",
        "THANG": "THÁNG",
        "NAM": "NĂM",
    }
    
    for garbled, correct in corrections.items():
        text = text.replace(garbled, correct)
        # Also try uppercase mapping if text is uppercase
        text = text.replace(garbled.upper(), correct.upper())

    # 3. Collapse repeated whitespace
    text = re.sub(r'\s+', ' ', text)

    return text.strip()
