# -*- coding: utf-8 -*-
"""
invoice_schema.py
=================
Final schema authority for Vietnamese invoice extraction.

Every field outputs: {"value": ..., "confidence": ...}
Top-level: status, document_confidence, _validation, _meta

FIXES in this version (v2 — GTGT layouts):
  BUG A — _seller(): Added phone, bank_account, bank_name fields
           (model cannot extract these yet, but schema must include placeholders
            for web app to display them consistently across all layouts)

  BUG B — _buyer(): Added full_name field
           (personal name of buyer, e.g. "Nguyễn Văn A" — populated by
            field_extractor.py _split_buyer_name_spans() BUG C fix)

  Both are non-breaking additions: field_extractor must set these keys
  for them to appear with real values; otherwise they default to "" / 0.0.
"""

from typing import Any, Dict, List, Optional


class InvoiceSchema:

    # ──────────────────────────────────────────────────────────────
    # Safe coercions
    # ──────────────────────────────────────────────────────────────

    @staticmethod
    def _safe_int(v: Any, default: int = 0) -> int:
        try:
            if isinstance(v, str):
                v = v.replace(".", "").replace(",", "")
            return int(float(v))
        except Exception:
            return default

    @staticmethod
    def _safe_float(v: Any, default: float = 0.0) -> float:
        try:
            if isinstance(v, str):
                v = v.replace(".", "").replace(",", ".")
            return float(v)
        except Exception:
            return default

    @staticmethod
    def _safe_str(v: Any, default: str = "") -> str:
        try:
            if v is None:
                return default
            return str(v).strip()
        except Exception:
            return default

    # ──────────────────────────────────────────────────────────────
    # Wrap helper
    # ──────────────────────────────────────────────────────────────

    @staticmethod
    def _wrap(value: Any, confidence: float = 0.0) -> Dict[str, Any]:
        return {
            "value":      value,
            "confidence": round(max(0.0, min(1.0, confidence)), 4),
        }

    # ──────────────────────────────────────────────────────────────
    # Section builders
    # ──────────────────────────────────────────────────────────────

    @staticmethod
    def _seller(data: Dict[str, Any], field_confs: Dict[str, float]) -> Dict[str, Any]:
        seller = data.get("seller", {}) or {}

        def _sf(key: str, default: str = "") -> Dict[str, Any]:
            """Extract seller field — handles both Gemini and LayoutLMv3 format."""
            raw = seller.get(key, default)
            val, c = InvoiceSchema._extract_field(raw, default, field_confs, f"seller.{key}")
            return InvoiceSchema._wrap(val, c)

        return {
            "name":               _sf("name"),
            "tax_code":           _sf("tax_code"),
            "address":            _sf("address"),
            "phone":              _sf("phone"),
            "bank_account":       _sf("bank_account"),
            "bank_name":          _sf("bank_name"),
            "tax_authority_code": _sf("tax_authority_code"),
        }

    @staticmethod
    def _buyer(data: Dict[str, Any], field_confs: Dict[str, float]) -> Dict[str, Any]:
        buyer = data.get("buyer", {}) or {}

        def _bf(key: str, default: str = "") -> Dict[str, Any]:
            raw = buyer.get(key, default)
            val, c = InvoiceSchema._extract_field(raw, default, field_confs, f"buyer.{key}")
            return InvoiceSchema._wrap(val, c)

        return {
            "name":      _bf("name"),
            "tax_code":  _bf("tax_code"),
            "address":   _bf("address"),
            "full_name": _bf("full_name"),
        }

    @staticmethod
    def _vat_breakdown(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Build vat_breakdown from item-level vat_rate fields.

        Groups items by their vat_rate, sums taxable_amount (total) and
        vat_amount (line_tax) per group.  Returns a list only when 2+
        distinct non-empty rates are found; otherwise returns [].

        Each entry:
          {"rate": "10%",
           "taxable_amount": {"value": ..., "confidence": ...},
           "vat_amount":     {"value": ..., "confidence": ...}}
        """
        if not items:
            return []

        from collections import defaultdict

        # Collect per-rate groups from item-level data
        groups: Dict[str, Dict] = defaultdict(lambda: {
            "taxable_sum": 0, "vat_sum": 0,
            "taxable_confs": [], "vat_confs": [],
        })

        for it in items:
            if not isinstance(it, dict):
                continue

            # Extract vat_rate — may be {value, confidence} or raw
            vr_obj = it.get("vat_rate", {})
            if isinstance(vr_obj, dict):
                rate_str = str(vr_obj.get("value", "")).strip()
            else:
                rate_str = str(vr_obj).strip() if vr_obj else ""

            # Normalize: strip trailing %, ignore empty/zero
            rate_clean = rate_str.rstrip("%").strip()
            if not rate_clean or rate_clean == "0":
                continue

            # Get taxable amount (item total before tax)
            total_obj = it.get("total", {})
            if isinstance(total_obj, dict):
                t_val = InvoiceSchema._safe_int(total_obj.get("value", 0), 0)
                t_conf = float(total_obj.get("confidence", 0))
            else:
                t_val = InvoiceSchema._safe_int(total_obj, 0)
                t_conf = 0.0

            # Get line tax (VAT amount per item)
            tax_obj = it.get("line_tax", {})
            if isinstance(tax_obj, dict):
                lt_val = InvoiceSchema._safe_int(tax_obj.get("value", 0), 0)
                lt_conf = float(tax_obj.get("confidence", 0))
            else:
                lt_val = InvoiceSchema._safe_int(tax_obj, 0)
                lt_conf = 0.0

            rate_key = rate_clean  # e.g. "10", "8"
            groups[rate_key]["taxable_sum"] += t_val
            groups[rate_key]["vat_sum"] += lt_val
            if t_conf > 0:
                groups[rate_key]["taxable_confs"].append(t_conf)
            if lt_conf > 0:
                groups[rate_key]["vat_confs"].append(lt_conf)

        # Only produce breakdown when 2+ distinct rates
        if len(groups) < 2:
            return []

        breakdown = []
        for rate_key in sorted(groups.keys(), key=lambda r: -float(r)):
            g = groups[rate_key]
            t_conf_avg = (
                sum(g["taxable_confs"]) / len(g["taxable_confs"])
                if g["taxable_confs"] else 0.0
            )
            v_conf_avg = (
                sum(g["vat_confs"]) / len(g["vat_confs"])
                if g["vat_confs"] else 0.0
            )
            breakdown.append({
                "rate": f"{rate_key}%",
                "taxable_amount": InvoiceSchema._wrap(g["taxable_sum"], t_conf_avg),
                "vat_amount":     InvoiceSchema._wrap(g["vat_sum"],     v_conf_avg),
            })

        return breakdown

    @staticmethod
    def _invoice(
        data: Dict[str, Any],
        field_confs: Dict[str, float],
        items: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        inv  = data.get("invoice", {}) or {}

        def _if_str(key: str, default: str = "") -> Dict[str, Any]:
            """Invoice string field extractor."""
            raw = inv.get(key, default)
            val, c = InvoiceSchema._extract_field(raw, default, field_confs, f"invoice.{key}")
            return InvoiceSchema._wrap(val if val is not None else default, c)

        def _if_num(key: str, default: int = 0) -> Dict[str, Any]:
            """Invoice numeric field extractor — converts to int."""
            raw = inv.get(key, default)
            val, c = InvoiceSchema._extract_field(raw, default, field_confs, f"invoice.{key}")
            return InvoiceSchema._wrap(InvoiceSchema._safe_int(val, default), c)

        # Build vat_breakdown from item-level rates
        vat_breakdown = InvoiceSchema._vat_breakdown(items or [])

        # Base result
        result = {
            "type":           _if_str("type"),
            "symbol":         _if_str("symbol"),
            "number":         _if_str("number"),
            "date":           _if_str("date"),
            "payment_method": _if_str("payment_method"),
            "currency":       _if_str("currency", "VND"),
            "vat_rate":       _if_str("vat_rate"),
            "subtotal":       _if_num("subtotal"),
            "vat_amount":     _if_num("vat_amount"),
            "total_amount":   _if_num("total_amount"),
        }

        if vat_breakdown:
            # Multiple VAT rates detected → override vat_rate / vat_amount
            rate_str = ",".join(entry["rate"] for entry in vat_breakdown)
            combined_vat = sum(
                entry["vat_amount"]["value"] for entry in vat_breakdown
            )
            # Use average confidence across breakdown entries
            vat_confs = [
                entry["vat_amount"]["confidence"]
                for entry in vat_breakdown
                if entry["vat_amount"]["confidence"] > 0
            ]
            avg_vat_conf = (
                sum(vat_confs) / len(vat_confs) if vat_confs else 0.0
            )

            result["vat_rate"]   = InvoiceSchema._wrap(rate_str, avg_vat_conf)
            result["vat_amount"] = InvoiceSchema._wrap(combined_vat, avg_vat_conf)

        # vat_breakdown (new field for dual VAT support)
        result["vat_breakdown"] = vat_breakdown

        return result

    @staticmethod
    def _extract_field(raw_field, default_value, conf_dict=None, conf_key=None):
        """
        Universal field extractor — handles ALL formats from both
        Gemini output and legacy LayoutLMv3 output.

        Format A: {"value": "clean string", "confidence": 0.95}  ← Gemini
        Format B: raw scalar + confidence in separate conf_dict   ← LayoutLMv3
        Format C: None/missing → use default_value
        Format D: {"value": "{'value': '...', 'confidence': ...}"} ← double-wrapped
        """
        import ast

        # Format A/D: field is a {value, confidence} dict
        if isinstance(raw_field, dict) and "value" in raw_field:
            val  = raw_field.get("value", default_value)
            conf = raw_field.get("confidence", 0.0)

            # Format D: value is itself a stringified dict
            if isinstance(val, str) and val.strip().startswith("{"):
                try:
                    inner = ast.literal_eval(val)
                    if isinstance(inner, dict) and "value" in inner:
                        return inner.get("value", default_value), inner.get("confidence", conf)
                except Exception:
                    pass

            return val, conf

        # Format B: raw scalar with confidence in separate dict
        if conf_dict is not None and conf_key is not None:
            conf = conf_dict.get(conf_key, 0.0)
        else:
            conf = 0.0

        if raw_field is None:
            return default_value, conf

        return raw_field, conf

    @staticmethod
    def _items(data: Dict[str, Any]) -> List[Dict[str, Any]]:
        items = data.get("items", []) or []
        result = []
        for it in items:
            if not isinstance(it, dict):
                continue
            item_conf = it.get("_conf", {}) or {}
            print(f"[SCHEMA_IN] quantity={repr(it.get('quantity'))} "
                  f"unit_price={repr(it.get('unit_price'))} "
                  f"name={repr(str(it.get('name', ''))[:30])}")

            _n_val,  _n_conf  = InvoiceSchema._extract_field(it.get("name"),       "", item_conf, "name")
            _u_val,  _u_conf  = InvoiceSchema._extract_field(it.get("unit"),       "", item_conf, "unit")
            _q_val,  _q_conf  = InvoiceSchema._extract_field(it.get("quantity"),    0, item_conf, "quantity")
            _p_val,  _p_conf  = InvoiceSchema._extract_field(it.get("unit_price"),  0, item_conf, "unit_price")
            _t_val,  _t_conf  = InvoiceSchema._extract_field(it.get("total"),       0, item_conf, "total")
            _d_val,  _d_conf  = InvoiceSchema._extract_field(it.get("discount"),    0, item_conf, "discount")
            _lt_val, _lt_conf = InvoiceSchema._extract_field(it.get("line_tax"),    0, item_conf, "line_tax")
            _rt_val, _rt_conf = InvoiceSchema._extract_field(it.get("row_total"),   0, item_conf, "row_total")
            _vr_val, _vr_conf = InvoiceSchema._extract_field(it.get("vat_rate"),   "", item_conf, "vat_rate")

            # Ensure numeric fields are actually numeric
            _q_val  = InvoiceSchema._safe_int(_q_val,   0)
            _p_val  = InvoiceSchema._safe_int(_p_val,   0)
            _t_val  = InvoiceSchema._safe_int(_t_val,   0)
            _d_val  = InvoiceSchema._safe_int(_d_val,   0)
            _lt_val = InvoiceSchema._safe_int(_lt_val,  0)
            _rt_val = InvoiceSchema._safe_int(_rt_val,  0)

            result.append({
                "name":       InvoiceSchema._wrap(_n_val,  _n_conf),
                "unit":       InvoiceSchema._wrap(_u_val,  _u_conf),
                "quantity":   InvoiceSchema._wrap(_q_val,  _q_conf),
                "unit_price": InvoiceSchema._wrap(_p_val,  _p_conf),
                "total":      InvoiceSchema._wrap(_t_val,  _t_conf),
                "discount":   InvoiceSchema._wrap(_d_val,  _d_conf),
                "line_tax":   InvoiceSchema._wrap(_lt_val, _lt_conf),
                "row_total":  InvoiceSchema._wrap(_rt_val, _rt_conf),
                "vat_rate":   InvoiceSchema._wrap(_vr_val, _vr_conf),
            })
        return result

    # ──────────────────────────────────────────────────────────────
    # Status determination
    # ──────────────────────────────────────────────────────────────

    @staticmethod
    def _determine_status(
        data: Dict[str, Any],
        validation: Optional[Dict[str, Any]],
        doc_confidence: float,
        error: Optional[str],
    ) -> str:
        if error:
            return "error"
        if validation and validation.get("errors"):
            return "partial"
        seller  = data.get("seller", {}) or {}
        invoice = data.get("invoice", {}) or {}
        key_fields_ok = bool(seller.get("name") and invoice.get("number"))
        if doc_confidence >= 0.6 and key_fields_ok:
            return "success"
        return "partial"

    # ──────────────────────────────────────────────────────────────
    # Final build
    # ──────────────────────────────────────────────────────────────

    @staticmethod
    def build(
        data: Dict[str, Any],
        validation: Optional[Dict[str, Any]] = None,
        error: Optional[str] = None,
    ) -> Dict[str, Any]:
        try:
            if not isinstance(data, dict):
                data = {}

            # Unwrap nested "fields" structure if present
            if "fields" in data and isinstance(data["fields"], dict):
                flat_data = data["fields"].copy()
                flat_data["items"]      = data.get("items", [])
                flat_data["confidence"] = data.get("confidence", {})
            else:
                flat_data = data

            conf = flat_data.get("confidence", {}) or {}
            field_confs = (
                conf.get("fields", {})
                if isinstance(conf.get("fields"), dict)
                else {}
            )

            conf_values   = [float(v) for v in field_confs.values() if v is not None]
            doc_confidence = (
                sum(conf_values) / len(conf_values) if conf_values else 0.0
            )
            doc_confidence = max(0.0, min(1.0, doc_confidence))

            status = InvoiceSchema._determine_status(
                flat_data, validation, doc_confidence, error,
            )

            result = {
                "status":              status,
                "document_confidence": round(doc_confidence, 4),
                "seller":              InvoiceSchema._seller(flat_data, field_confs),
                "buyer":               InvoiceSchema._buyer(flat_data, field_confs),
                "items":               InvoiceSchema._items(flat_data),
                "invoice":             InvoiceSchema._invoice(
                    flat_data, field_confs,
                    items=flat_data.get("items", []),
                ),
                "_validation": {
                    "status":   status,
                    "errors":   (validation or {}).get("errors",   []),
                    "warnings": (validation or {}).get("warnings", []),
                },
            }

            if error:
                result["_error"] = InvoiceSchema._safe_str(error)

            return result

        except Exception as exc:
            _w = InvoiceSchema._wrap
            return {
                "status":              "error",
                "document_confidence": 0.0,
                "seller": {
                    "name":               _w(""), "tax_code":           _w(""),
                    "tax_authority_code": _w(""), "address":            _w(""),
                    "phone":              _w(""), "bank_account":       _w(""),
                    "bank_name":          _w(""),
                },
                "buyer": {
                    "full_name": _w(""), "name":     _w(""),
                    "tax_code":  _w(""), "address":  _w(""),
                },
                "invoice": {
                    "type": _w(""), "symbol": _w(""), "number": _w(""),
                    "date": _w(""), "payment_method": _w(""),
                    "subtotal":    _w(0), "vat_rate":   _w(""),
                    "vat_amount":  _w(0), "vat_breakdown": [],
                    "total_amount": _w(0),
                    "currency":    _w("VND", 1.0),
                },
                "items":      [],
                "_validation": {"status": "error", "errors": [str(exc)], "warnings": []},
                "_error":     f"Schema build failure: {exc}",
            }


def serialize_invoice_result(raw_result: Dict[str, Any]) -> Dict[str, Any]:
    """
    Canonical serialization entry point for the web app.

    Wraps InvoiceSchema.build() to ensure a consistent {value, confidence} schema.
    """
    if not isinstance(raw_result, dict):
        raw_result = {}
    return InvoiceSchema.build(raw_result)


def compute_document_confidence(final: Dict[str, Any]) -> float:
    """
    Compute document_confidence from field confidences with validation penalties.

    This is used after business-rule validation in app.py.
    """
    if not isinstance(final, dict):
        return 0.0

    # Base confidence: prefer existing value, else compute from fields
    base = final.get("document_confidence", None)
    if base is None:
        confs = []

        def _collect_conf(obj: Any) -> None:
            if isinstance(obj, dict):
                if "confidence" in obj:
                    try:
                        c = float(obj.get("confidence", 0.0))
                        if c >= 0:
                            confs.append(c)
                    except Exception:
                        pass
                for v in obj.values():
                    _collect_conf(v)
            elif isinstance(obj, list):
                for v in obj:
                    _collect_conf(v)

        _collect_conf(final.get("seller", {}))
        _collect_conf(final.get("buyer", {}))
        _collect_conf(final.get("invoice", {}))
        _collect_conf(final.get("items", []))

        if confs:
            base = sum(confs) / len(confs)
        else:
            base = 0.0

    try:
        base = float(base)
    except Exception:
        base = 0.0

    base = max(0.0, min(1.0, base))

    # Penalties from validation
    vld = final.get("_validation", {}) or {}
    errors = vld.get("errors", []) or []
    warnings = vld.get("warnings", []) or []

    # Small penalties per issue, capped to avoid wiping out confidence entirely
    penalty = 0.0
    penalty += min(0.4, len(errors) * 0.15)
    penalty += min(0.2, len(warnings) * 0.05)

    doc_conf = base - penalty
    doc_conf = max(0.0, min(1.0, doc_conf))
    return round(doc_conf, 4)
