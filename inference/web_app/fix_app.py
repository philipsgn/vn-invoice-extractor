import sys
from pathlib import Path

def main():
    app_path = Path('c:/Users/TanPhat/Documents/DEEPLEARNING/invoice_ocr/inference/web_app/app.py')
    with open(app_path, 'r', encoding='utf-8') as f:
        content = f.read()

    bad_table_block = """        "predicted_labels": repaired_header,
    repaired_footer = repair_bio_sequence(footer_labels, words=words, bboxes=bboxes)"""
    
    table_raw_block = """        "predicted_labels": repaired_header,
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
    repaired_footer = repair_bio_sequence(footer_labels, words=words, bboxes=bboxes)"""
    
    if bad_table_block in content:
        content = content.replace(bad_table_block, table_raw_block)
        print("Fixed stage4_postprocess!")

    # Fix the syntax error block
    import re
    pattern = re.compile(r'def _normalize_amount\(value: Any\).*?def stage5_format\(raw: Dict', re.DOTALL)
    
    rep = '''def _normalize_amount(value: Any) -> Optional[float]:
    """Parse Vietnamese number format. "5.000.000" -> 5000000.0"""
    if value is None:
        return None
    val, conf = safe_unwrap(value)
    if val is None:
        return None
    if isinstance(val, (int, float)):
        return float(val)
    s = str(val).strip()
    s = re.sub(r"[^\\d.,-]", "", s)
    s = s.replace(".", "").replace(",", ".")
    try:
        return float(s)
    except ValueError:
        return None


def _normalize_tax_code(value: Any) -> Any:
    if not value:
        return value
    val, conf = safe_unwrap(value)
    if not val:
        return value
    s = str(val).strip()
    n = s if s else None

    if n is not None:
        if isinstance(value, dict):
            value["value"] = n
            if "confidence" not in value:
                value["confidence"] = conf
            return value
        return n
    return value


def _normalize_vat_rate(value: Any) -> Any:
    if value is None:
        return value
    val, conf = safe_unwrap(value)
    if val is None:
        return value
    s = str(val).strip()
    if not s:
        return value
    s_clean = s.rstrip("%").strip()
    n = s_clean if s_clean else None

    m = re.search(r"(\\d+(?:[.,]\\d+)?)", s_clean)
    if m:
        num_str = m.group(1).replace(",", ".")
        try:
            num = float(num_str)
            if num == int(num):
                n = str(int(num))
            else:
                n = str(num)
        except ValueError:
            pass

    if n is not None:
        if isinstance(value, dict):
            value["value"] = n
            if "confidence" not in value:
                value["confidence"] = conf
            return value
        return n
    return value


def stage5_format(raw: Dict'''

    match = pattern.search(content)
    if match:
        content = content[:match.start()] + rep + content[match.end():]
        print("Fixed normalizers via string slicing!")
        
    with open(app_path, 'w', encoding='utf-8') as f:
        f.write(content)
    print("Done")

if __name__ == '__main__':
    main()
