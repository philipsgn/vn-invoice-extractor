# -*- coding: utf-8 -*-
"""
Date normalizer for Vietnamese invoice dates.

Handles formats:
  - dd/mm/yyyy, dd-mm-yyyy, dd.mm.yyyy
  - dd/mm/yy  (converts to 20yy)
  - Ngày DD tháng MM năm YYYY   ← FIX: Vietnamese long-form (all GTGT layouts)
  - ngày DD/MM/YYYY             ← FIX: mixed prefix+separator variant

Validates dates and rejects future dates.
"""

import re
from datetime import datetime
from typing import Optional, Tuple


class DateNormalizer:
    """
    Normalize and validate date strings from Vietnamese invoices.

    Supports formats:
    - dd/mm/yyyy, dd-mm-yyyy, dd.mm.yyyy
    - dd/mm/yy  (converts to 20yy)
    - Ngày DD tháng MM năm YYYY   ← GTGT + banhang all layouts
    - ngày DD/MM/YYYY             ← mixed variant

    Rejects:
    - Future dates (configurable)
    - Invalid dates (e.g., 32/13/2024)
    - Dates before 2000 (assumed invalid for modern Vietnamese invoices)
    """

    # ── Vietnamese long-form: "Ngày 20 tháng 09 năm 2022"
    # Handles: optional "ngày"/"Ngày"/"NGÀY", tháng/THÁNG, năm/NĂM
    # Also handles OCR noise: "Ngay 20 thang 09 nam 2022"
    _VI_LONG_PATTERN = re.compile(
        r"""(?:ng[aà]y\s+)?         # Optional prefix "ngày " (diacritics optional)
            (\d{1,2})\s+            # DD
            th[aá]ng\s+             # "tháng" (diacritics optional)
            (\d{1,2})\s+            # MM
            n[aă]m\s+               # "năm" (diacritics optional)
            (\d{4})                 # YYYY
        """,
        re.IGNORECASE | re.VERBOSE,
    )

    # ── Standard separator: dd/mm/yyyy, dd-mm-yyyy, dd.mm.yyyy
    _SEP_PATTERN = re.compile(
        r"\b(\d{1,2})[\/\-\.](\d{1,2})[\/\-\.](\d{2,4})\b"
    )

    def __init__(self, max_future_days: int = 0):
        """
        Args:
            max_future_days: Allow dates up to N days in future (0 = no future)
        """
        self.max_future_days = max_future_days

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_date(self, day: int, month: int, year: int) -> Optional[datetime]:
        """Validate and build datetime. Returns None on invalid input."""
        if year < 100:
            year += 2000 if year < 50 else 1900
        if year < 2000 or year > 2100:
            return None
        if not (1 <= month <= 12):
            return None
        if not (1 <= day <= 31):
            return None
        try:
            return datetime(year, month, day)
        except ValueError:
            return None

    def _within_time_bounds(self, dt: datetime) -> bool:
        """Check date is not in the future beyond allowed window."""
        today = datetime.now()
        if self.max_future_days == 0:
            return dt <= today
        from datetime import timedelta
        return dt <= today + timedelta(days=self.max_future_days)

    def _normalize_dt(self, dt: datetime, full_match: bool) -> Tuple[str, float]:
        """Format datetime to ISO 8601 with confidence score."""
        normalized = dt.strftime("%Y-%m-%d")
        confidence = 0.95 if full_match else 0.85
        return normalized, confidence

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def extract_date(self, text: str) -> Optional[Tuple[str, float]]:
        """
        Extract and normalize date from text.

        Priority:
          1. Vietnamese long-form "Ngày DD tháng MM năm YYYY"
          2. Standard separator "DD/MM/YYYY"

        Returns:
            (normalized_date "YYYY-MM-DD", confidence) or None
        """
        if not text or not isinstance(text, str):
            return None
        text = text.strip()
        if not text:
            return None

        # ── Priority 1: Vietnamese long-form ───────────────────────
        m = self._VI_LONG_PATTERN.search(text)
        if m:
            day, month, year = int(m.group(1)), int(m.group(2)), int(m.group(3))
            dt = self._build_date(day, month, year)
            if dt and self._within_time_bounds(dt):
                full = m.group(0).strip() == text
                return self._normalize_dt(dt, full_match=full)

        # ── Priority 2: Standard separator ─────────────────────────
        m = self._SEP_PATTERN.search(text)
        if m:
            # Detect order: if group3 is 4-digit it's always YYYY
            g1, g2, g3 = int(m.group(1)), int(m.group(2)), int(m.group(3))
            if g3 > 31:
                # dd / mm / yyyy  (Vietnamese standard)
                day, month, year = g1, g2, g3
            elif g1 > 31:
                # yyyy / mm / dd  (ISO variant)
                year, month, day = g1, g2, g3
            else:
                # Assume dd / mm / yy(yy)
                day, month, year = g1, g2, g3

            dt = self._build_date(day, month, year)
            if dt and self._within_time_bounds(dt):
                full = m.group(0).strip() == text
                return self._normalize_dt(dt, full_match=full)

        return None

    def normalize(self, text: str) -> Optional[str]:
        """
        Normalize date text → "YYYY-MM-DD" or None.

        Usage:
            dn = DateNormalizer()
            dn.normalize("Ngày 20 tháng 09 năm 2022")  # → "2022-09-20"
            dn.normalize("20/09/2022")                  # → "2022-09-20"
        """
        result = self.extract_date(text)
        return result[0] if result else None


# ------------------------------------------------------------------
# Module-level convenience instance
# ------------------------------------------------------------------
_default = DateNormalizer(max_future_days=0)


def normalize_invoice_date(text: str) -> Optional[str]:
    """
    Convenience function — normalize a single date string.

    Examples:
        normalize_invoice_date("Ngày 20 tháng 09 năm 2022") → "2022-09-20"
        normalize_invoice_date("20/09/2022")                 → "2022-09-20"
        normalize_invoice_date("ngay 04 thang 01 nam 2023")  → "2023-01-04"
    """
    return _default.normalize(text)


# ------------------------------------------------------------------
# Quick smoke test
# ------------------------------------------------------------------
if __name__ == "__main__":
    dn = DateNormalizer()
    tests = [
        ("Ngày 20 tháng 09 năm 2022",  "2022-09-20"),
        ("Ngày 04 tháng 01 năm 2023",  "2023-01-04"),
        ("Ngày 22 tháng 08 năm 2022",  "2022-08-22"),
        ("Ngày 26 tháng 01 năm 2026",  "2026-01-26"),   # future date allowed by test data
        ("ngay 04 thang 01 nam 2023",  "2023-01-04"),   # OCR noise
        ("20/09/2022",                 "2022-09-20"),
        ("20-09-2022",                 "2022-09-20"),
        ("20.09.2022",                 "2022-09-20"),
        ("invalid text",               None),
    ]
    all_pass = True
    for text, expected in tests:
        result = dn.normalize(text)
        ok = result == expected
        if not ok:
            all_pass = False
        print(f"  {'✅' if ok else '❌'} {text!r:45} → {result!r}  (expected {expected!r})")
    print(f"\n{'✅ ALL PASS' if all_pass else '❌ FAILURES'}")