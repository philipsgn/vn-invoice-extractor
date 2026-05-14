"""
Money normalizer for Vietnamese invoice amounts.

Handles formats:
- 1.250.000đ
- 1,250,000 VND
- 1250000
- 1.250.000,00
"""

import re
from typing import Optional, Tuple


class MoneyNormalizer:
    """
    Normalize and extract money amounts from Vietnamese invoice text.
    
    Supports Vietnamese money formats:
    - 1.250.000đ
    - 1,250,000 VND
    - 1.250.000,00
    - 1250000
    """

    # Money patterns for Vietnamese invoices
    MONEY_PATTERNS = [
        r"\b(\d{1,3}(?:[.,]\d{3})+(?:[.,]\d{2})?)\s*[đĐVNDvnd]?\b",  # 1.250.000đ, 1,250,000 VND
        r"\b(\d{1,3}(?:\.\d{3})+(?:,\d{2})?)\b",  # 1.250.000,00
        r"\b(\d{4,})\b",  # Plain numbers >= 1000
    ]

    def __init__(self):
        """Initialize money normalizer."""
        self.compiled_patterns = [re.compile(p, re.IGNORECASE) for p in self.MONEY_PATTERNS]

    def extract_money(self, text: str) -> Optional[Tuple[int, float]]:
        """
        Extract and normalize money amount from text.
        
        Args:
            text: Input text containing money amount
            
        Returns:
            Tuple of (amount_in_cents_or_smallest_unit, confidence) or None
            Confidence: 0.0-1.0 based on pattern match quality
        """
        if not text or not isinstance(text, str):
            return None

        text = text.strip()
        if not text:
            return None

        candidates = []

        # Try each pattern
        for pattern in self.compiled_patterns:
            matches = pattern.finditer(text)
            for match in matches:
                amount_str = match.group(1)
                
                # Normalize: remove separators
                # Vietnamese uses . as thousands separator
                normalized = amount_str.replace(".", "").replace(",", "")
                
                # Check if there's a decimal part (last 2 digits after comma)
                # For Vietnamese, if there's a comma with 2 digits, it's usually decimal
                if "," in amount_str:
                    parts = amount_str.split(",")
                    if len(parts) == 2 and len(parts[1]) == 2:
                        # Has decimal part, keep as is
                        normalized = amount_str.replace(".", "").replace(",", "")
                    else:
                        # Comma is thousands separator
                        normalized = amount_str.replace(",", "").replace(".", "")
                else:
                    # Only dots, treat as thousands separator
                    normalized = amount_str.replace(".", "").replace(",", "")

                try:
                    amount = int(normalized)
                    
                    # Filter out unrealistic amounts
                    if amount < 0:
                        continue
                    if amount > 10_000_000_000:  # 10 billion (unrealistic for single invoice)
                        continue
                    
                    # Confidence based on pattern and context
                    confidence = 0.9
                    if "đ" in match.group(0).lower() or "vnd" in match.group(0).lower():
                        confidence = 0.95  # Explicit currency marker
                    elif amount < 1000:
                        confidence = 0.7  # Small amounts might be quantities
                    
                    candidates.append((amount, confidence, match.start(), match.end()))
                    
                except (ValueError, TypeError):
                    continue

        if not candidates:
            return None

        # Prefer the last match (usually the total/amount)
        # Or the one with highest confidence
        candidates.sort(key=lambda x: (x[1], x[2]), reverse=True)
        amount, confidence, _, _ = candidates[0]

        return (amount, confidence)

    def normalize(self, text: str) -> Optional[int]:
        """
        Normalize money text to integer amount.
        
        Args:
            text: Input text
            
        Returns:
            Amount as integer or None
        """
        result = self.extract_money(text)
        return result[0] if result else None
