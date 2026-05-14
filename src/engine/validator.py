# -*- coding: utf-8 -*-
import re
from typing import Optional, Dict, Any

class NumericValidator:
    """
    Validates monetary values and performs accounting cross-checks.
    """
    
    # Strict monetary regex: matches patterns like 1.000.000 or 50,819,028
    MONETARY_RE = re.compile(r'\b\d{1,3}(?:[.,]\d{3})+\b|\b\d{4,}\b')

    def extract_monetary_value(self, raw_text: str) -> Optional[str]:
        """
        Extracts only valid monetary patterns from raw string.
        Rejects long numeric sequences (tax codes/IDs).
        """
        if not raw_text:
            return None
            
        # 1. Reject continuous digits >= 10 (likely Tax Code or ID)
        if re.search(r'\d{10,}', raw_text.replace(" ", "")):
            # If there are NO dots/commas and long digits, reject
            if not any(c in raw_text for c in ".,"):
                return None
        
        matches = self.MONETARY_RE.findall(raw_text)
        if not matches:
            return None
            
        # Pick the most likely monetary value (usually the last one in a contaminated string)
        return matches[-1]

    def parse_float(self, val: Any) -> float:
        if not val:
            return 0.0
        try:
            s = str(val).replace(".", "").replace(",", "")
            return float(s)
        except:
            return 0.0

    def validate_totals(self, subtotal: Any, vat: Any, grand_total: Any) -> Dict[str, Any]:
        """
        Validates: subtotal + vat = grand_total.
        Returns corrected values using majority vote/recalculation.
        """
        s = self.parse_float(subtotal)
        v = self.parse_float(vat)
        g = self.parse_float(grand_total)

        # Check relationship
        if abs((s + v) - g) < 2.0:  # Allow 2.0 tolerance for rounding
            return {"subtotal": s, "vat_amount": v, "total_amount": g, "valid": True}

        # If mismatch, recalculate based on extracted values
        # Preference: If G matches S+V, great. 
        # If not, check if one of them is likely wrong.
        if s > 0 and v >= 0:
            return {"subtotal": s, "vat_amount": v, "total_amount": s + v, "valid": False, "corrected": True}
        
        return {"subtotal": s, "vat_amount": v, "total_amount": g, "valid": False}
