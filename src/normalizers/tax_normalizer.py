"""
Tax code (MST) normalizer for Vietnamese invoices.

Handles Vietnamese tax codes (Mã số thuế):
- 10 digits (standard)
- 13 digits (with branch code)
"""

import re
from typing import Optional, Tuple


class TaxNormalizer:
    """
    Normalize and validate Vietnamese tax codes (Mã số thuế / MST).
    
    Vietnamese tax codes:
    - Standard: 10 digits
    - With branch: 13 digits (10 + 3 branch digits)
    
    Common patterns:
    - MST: 0123456789
    - Mã số thuế: 0123456789
    - Tax code: 0123456789
    """

    # Tax code patterns
    TAX_PATTERNS = [
        r"\b(?:MST|Mã\s*số\s*thuế|TAX\s*CODE|Mã\s*ST)\s*[:\-]?\s*(\d{10}(?:\d{3})?)\b",
        r"\b(\d{10}(?:\d{3})?)\b",  # Standalone 10 or 13 digit number
    ]

    # Keywords that should be near tax code
    TAX_KEYWORDS = [
        "MST", "Mã số thuế", "TAX", "Mã ST", "Mã số", "Tax Code"
    ]

    def __init__(self, require_keyword: bool = False):
        """
        Args:
            require_keyword: If True, require tax keyword near the number
        """
        self.require_keyword = require_keyword
        self.compiled_patterns = [re.compile(p, re.IGNORECASE | re.UNICODE) for p in self.TAX_PATTERNS]
        self.keyword_pattern = re.compile(
            "|".join(re.escape(k) for k in self.TAX_KEYWORDS),
            re.IGNORECASE | re.UNICODE
        )

    def extract_tax_code(self, text: str) -> Optional[Tuple[str, float]]:
        """
        Extract and normalize tax code from text.
        
        Args:
            text: Input text containing tax code
            
        Returns:
            Tuple of (normalized_tax_code, confidence) or None
            Confidence: 0.0-1.0 based on pattern match quality
        """
        if not text or not isinstance(text, str):
            return None

        text = text.strip()
        if not text:
            return None

        # Check for keyword if required
        has_keyword = bool(self.keyword_pattern.search(text))
        if self.require_keyword and not has_keyword:
            return None

        candidates = []

        # Try patterns with keyword context first
        for i, pattern in enumerate(self.compiled_patterns):
            matches = pattern.finditer(text)
            for match in matches:
                tax_code = match.group(1) if match.lastindex else match.group(0)
                
                # Validate length
                if len(tax_code) not in [10, 13]:
                    continue
                
                # Validate all digits
                if not tax_code.isdigit():
                    continue
                
                # Confidence scoring
                confidence = 0.8
                
                # Higher confidence if keyword present
                if has_keyword:
                    confidence = 0.95
                
                # Higher confidence for pattern with keyword
                if i == 0:  # First pattern includes keyword
                    confidence = 0.95
                
                # Slightly lower for standalone numbers
                if i == 1 and not has_keyword:
                    confidence = 0.7
                
                candidates.append((tax_code, confidence, match.start(), match.end()))

        if not candidates:
            return None

        # Prefer highest confidence, then first occurrence
        candidates.sort(key=lambda x: (x[1], -x[2]), reverse=True)
        tax_code, confidence, _, _ = candidates[0]

        return (tax_code, confidence)

    def normalize(self, text: str) -> Optional[str]:
        """
        Normalize tax code text.
        
        Args:
            text: Input text
            
        Returns:
            Normalized tax code string or None
        """
        result = self.extract_tax_code(text)
        return result[0] if result else None
