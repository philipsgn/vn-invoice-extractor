# -*- coding: utf-8 -*-
"""
invoice_validator.py
======================================================
Production business rule validation for extracted invoice fields.

Checks:
  1. VAT math:  vat_amount ≈ subtotal × vat_rate
  2. Total math: subtotal + vat_amount ≈ total_amount
  3. Row math:   quantity × unit_price ≈ total
  4. Row VAT:    total + line_tax ≈ row_total
  5. Tax code format: 10 or 13 digits (10-3 format)
  6. Confidence thresholds
======================================================
"""

import re
from typing import Any, Dict, List, Optional


class InvoiceValidator:

    def __init__(
        self,
        vat_tolerance:  float = 0.05,
        min_confidence: float = 0.5,
        amount_tolerance: float = 1.0,  # Allow ±1 VND for rounding
    ):
        self.vat_tolerance    = vat_tolerance
        self.min_confidence   = min_confidence
        self.amount_tolerance = amount_tolerance

    # ── Safe coerce ──────────────────────────────────────────

    @staticmethod
    def _to_float(v: Any) -> Optional[float]:
        if v is None:
            return None
        # Handle dict format: {"value": ..., "confidence": ...}
        if isinstance(v, dict):
            v = v.get("value")
            if v is None:
                return None
        if isinstance(v, (int, float)):
            return float(v)
        try:
            s = str(v).strip()
            # Remove % sign
            s = s.replace("%", "").strip()
            # Vietnamese format: "5.000.000,50" -> "5000000.50"
            s = s.replace(".", "")
            s = s.replace(",", ".")
            return float(s) if s else None
        except Exception:
            return None

    # ── Individual checks ─────────────────────────────────────

    def _check_vat_rate(self, data: Dict[str, Any]) -> List[str]:
        """
        Check: vat_amount ≈ subtotal × vat_rate

        When vat_breakdown is present (multi-VAT), validate each breakdown
        entry individually: entry.vat_amount ≈ entry.taxable_amount × rate.
        Otherwise fall back to the single-rate check.
        """
        errors = []
        inv = data.get("invoice", {}) or {}

        # ── Multi-VAT: per-entry validation ──────────────────────────
        vat_breakdown = inv.get("vat_breakdown", [])
        if vat_breakdown and isinstance(vat_breakdown, list):
            for entry in vat_breakdown:
                if not isinstance(entry, dict):
                    continue
                rate_str = str(entry.get("rate", "")).rstrip("%").strip()
                rate = self._to_float(rate_str)
                taxable = self._to_float(entry.get("taxable_amount"))
                vat_amt = self._to_float(entry.get("vat_amount"))

                if None in (rate, taxable, vat_amt):
                    continue
                if rate == 0 or taxable == 0:
                    continue

                # Normalize: 10 → 0.10
                if rate > 1:
                    rate = rate / 100.0

                expected_vat = taxable * rate
                if expected_vat > 0:
                    diff_ratio = abs(vat_amt - expected_vat) / expected_vat
                    if diff_ratio > self.vat_tolerance:
                        errors.append(
                            f"VAT breakdown mismatch ({entry.get('rate', '?')}): "
                            f"vat_amount={vat_amt:.0f} "
                            f"expected≈{expected_vat:.0f} "
                            f"(diff={diff_ratio:.1%} > tolerance={self.vat_tolerance:.0%})"
                        )
            return errors

        # ── Single-VAT: original logic ───────────────────────────────
        vat_rate   = self._to_float(inv.get("vat_rate"))
        vat_amount = self._to_float(inv.get("vat_amount"))
        subtotal   = self._to_float(inv.get("subtotal"))

        if None in (vat_rate, vat_amount, subtotal):
            return errors
        if subtotal == 0 or vat_rate == 0:
            return errors

        # Normalize vat_rate: 10 → 0.10
        rate = vat_rate
        if rate > 1:
            rate = rate / 100.0

        expected_vat = subtotal * rate
        if expected_vat > 0:
            diff_ratio = abs(vat_amount - expected_vat) / expected_vat
            if diff_ratio > self.vat_tolerance:
                errors.append(
                    f"VAT rate mismatch: vat_amount={vat_amount:.0f} "
                    f"expected≈{expected_vat:.0f} "
                    f"(diff={diff_ratio:.1%} > tolerance={self.vat_tolerance:.0%})"
                )

        return errors

    def _check_total_sum(self, data: Dict[str, Any]) -> List[str]:
        """
        Check: subtotal + vat_amount ≈ total_amount

        When vat_breakdown is present, sums VAT amounts from all breakdown
        entries instead of using the single invoice-level vat_amount.
        """
        errors = []
        inv = data.get("invoice", {}) or {}

        subtotal    = self._to_float(inv.get("subtotal"))
        total       = self._to_float(inv.get("total_amount"))

        # Determine effective vat_amount: sum from breakdown if present
        vat_breakdown = inv.get("vat_breakdown", [])
        if vat_breakdown and isinstance(vat_breakdown, list):
            vat_amount = 0.0
            for entry in vat_breakdown:
                if not isinstance(entry, dict):
                    continue
                v = self._to_float(entry.get("vat_amount"))
                if v is not None:
                    vat_amount += v
        else:
            vat_amount = self._to_float(inv.get("vat_amount"))

        if None in (subtotal, vat_amount, total):
            return errors
        if subtotal == 0 and vat_amount == 0 and total == 0:
            return errors

        expected_total = subtotal + vat_amount
        diff = abs(total - expected_total)

        if diff > max(self.amount_tolerance, abs(total) * self.vat_tolerance):
            errors.append(
                f"Total sum mismatch: subtotal({subtotal:.0f}) + vat_amount({vat_amount:.0f}) "
                f"= {expected_total:.0f}, but total_amount = {total:.0f} "
                f"(diff={diff:.0f})"
            )

        return errors

    def _check_item_rows(self, data: Dict[str, Any]) -> List[str]:
        """
        Check per-item:
          quantity × unit_price ≈ total
          total + line_tax ≈ row_total
        """
        warnings = []
        items = data.get("items", []) or []

        for idx, item in enumerate(items):
            if not isinstance(item, dict):
                continue

            qty        = self._to_float(item.get("quantity"))
            unit_price = self._to_float(item.get("unit_price"))
            total      = self._to_float(item.get("total"))
            line_tax   = self._to_float(item.get("line_tax"))
            row_total  = self._to_float(item.get("row_total"))

            # Check qty × unit_price ≈ total
            if qty is not None and unit_price is not None and total is not None:
                if qty > 0 and unit_price > 0:
                    discount = self._to_float(item.get("discount")) or 0
                    calc = (qty * unit_price) - discount
                    tolerance = max(total * 0.02, 500)  # 2% or 500 VND rounding
                    if abs(calc - total) > tolerance:
                        warnings.append(  # WARNING not ERROR — OCR rounding is normal
                            f"Item[{idx}] line total mismatch: "
                            f"{qty}×{unit_price}-{discount}={calc:.0f} ≠ total={total:.0f}"
                        )

            # Check total + line_tax ≈ row_total
            if total is not None and line_tax is not None and row_total is not None:
                if total > 0:
                    expected_row = total + line_tax
                    if row_total > 0 and abs(row_total - expected_row) > max(1, abs(row_total) * 0.05):
                        warnings.append(
                            f"Item[{idx}] row total mismatch: "
                            f"total({total:.0f})+line_tax({line_tax:.0f})"
                            f"={expected_row:.0f} ≠ row_total={row_total:.0f}"
                        )

        return warnings

    def _check_tax_codes(self, data: Dict[str, Any]) -> List[str]:
        """Check tax code format: 10, 13, or 10-3 digits."""
        errors  = []
        # Valid Vietnamese MST patterns:
        VALID_TAX_PATTERNS = [
            r'\d{10}',        # Standard company MST
            r'\d{10}-\d{3}',  # Branch office MST
            r'\d{13}',        # Household / personal business MST
        ]

        for party in ("seller", "buyer"):
            tc_obj = (data.get(party, {}) or {}).get("tax_code", "")
            if not tc_obj:
                continue
            
            # Handle dict format
            tc = tc_obj.get("value", "") if isinstance(tc_obj, dict) else str(tc_obj)
            if not tc:
                continue

            tc_clean = str(tc).replace(" ", "").replace(".", "")
            
            if not any(re.fullmatch(p, tc_clean) for p in VALID_TAX_PATTERNS):
                errors.append(
                    f"{party}.tax_code '{tc}' invalid "
                    f"(must be 10 digits, 13 digits, or 10-3 format)"
                )

        return errors

    def _check_confidence(self, data: Dict[str, Any]) -> List[str]:
        """Check fields with confidence below threshold."""
        warnings = []
        conf     = data.get("confidence", {}) or {}
        fields   = conf.get("fields", {}) or {}

        low_conf = [
            f"{field}={score:.2f}"
            for field, score in fields.items()
            if isinstance(score, (int, float)) and score < self.min_confidence
        ]

        if low_conf:
            warnings.append(f"Low confidence fields: {', '.join(low_conf)}")

        return warnings

    def _check_missing_fields(self, data: Dict[str, Any]) -> List[str]:
        """Check for critical missing fields."""
        warnings = []
        seller = data.get("seller", {}) or {}
        buyer  = data.get("buyer", {}) or {}
        inv    = data.get("invoice", {}) or {}

        if not seller.get("name"):
            warnings.append("Missing seller.name")
        if not seller.get("tax_code"):
            warnings.append("Missing seller.tax_code")
        if not inv.get("number"):
            warnings.append("Missing invoice.number")
        if not inv.get("date"):
            warnings.append("Missing invoice.date")

        return warnings

    # ── Main validation ───────────────────────────────────────

    def validate_all(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run all validations.

        Returns:
            {
                "errors":   List[str],   # business logic violations
                "warnings": List[str],   # non-blocking warnings
            }
        """
        errors   = []
        warnings = []

        errors   += self._check_vat_rate(data)
        errors   += self._check_total_sum(data)
        errors   += self._check_tax_codes(data)
        warnings += self._check_item_rows(data)
        warnings += self._check_confidence(data)
        warnings += self._check_missing_fields(data)

        return {
            "errors":   errors,
            "warnings": warnings,
        }