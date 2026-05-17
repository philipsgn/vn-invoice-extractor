"""
Gemini Flash Invoice Extractor — Vietnamese Invoice OCR
Replaces LayoutLMv3 stages 3+4 as primary extraction engine.

Model: gemini-2.0-flash (retire June 2026) → upgrade gemini-2.5-flash
Free tier: 1500 requests/day → $0 for < 100 invoices/day
"""

import json
import logging
import re
import time
from typing import Optional

from google import genai
from google.genai import types

log = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────
# gemini-2.0-flash has limit:0 on this account's free tier
# gemini-2.5-flash is confirmed working (tested 2026-03-16)
GEMINI_MODEL_PRIMARY  = "gemini-2.5-flash"
GEMINI_MODEL_FALLBACK = "gemini-2.5-flash"   # same — only one works
GEMINI_MODEL          = GEMINI_MODEL_PRIMARY

TEMPERATURE  = 0.0    # 0 = deterministic, tối quan trọng cho extraction
MAX_TOKENS   = 8192  # Gemini Flash có thể trả về JSON rất dài nếu hóa đơn phức tạp, nhiều dòng
TIMEOUT_SEC  = 30
# ──────────────────────────────────────────────────────────────────────────────

# ── Extraction Prompt (Senior OCR Vietnamese Invoice Expert) ──────────────────
INVOICE_EXTRACTION_PROMPT = """# ROLE & OBJECTIVE
You are a world-class Document AI Agent specializing in Vietnamese Financial Documents (Hóa đơn Giá trị gia tăng & Hóa đơn bán hàng). Your absolute objective is to extract ALL information from the provided invoice image/PDF with 100% accuracy and output a strict, valid JSON object matching the defined schema. Do not include any markdown formatting (like ```json) or conversational text outside the JSON.

# INVOICE TYPE LOGIC (CRITICAL)
- If the invoice title contains "GIÁ TRỊ GIA TĂNG" or "VAT INVOICE": Set "invoice_type" to "HOA_DON_GTGT". You MUST extract "vat_rate" and "vat_amount".
- If the invoice title contains "BÁN HÀNG" or does not specify VAT: Set "invoice_type" to "HOA_DON_BAN_HANG". Set "vat_rate" to null and "vat_amount" to 0.

# DATA EXTRACTION EXTRA RULES
- Numbers: Extract all monetary amounts, quantities, and prices as numeric values (float/int), removing any currency symbols (đ, VND) or thousands separators (.) from the final JSON values.
- Dates: Standardize all dates to "YYYY-MM-DD" format.
- Fallback: If a field does not exist or is completely unreadable, return null (or 0 for numeric financial fields if applicable).

# OUTPUT JSON SCHEMA
{
  "invoice_type": "HOA_DON_GTGT" | "HOA_DON_BAN_HANG",
  "invoice_metadata": {
    "form_number": string or null,
    "serial_number": string or null,
    "invoice_number": string or null,
    "invoice_date": "YYYY-MM-DD" or null
  },
  "seller": {
    "company_name": string or null,
    "tax_code": string or null,
    "address": string or null
  },
  "buyer": {
    "company_name": string or null,
    "tax_code": string or null,
    "address": string or null
  },
  "line_items": [
    {
      "sequence_number": int or null,
      "item_name": string or null,
      "unit": string or null,
      "quantity": float or null,
      "unit_price": float or null,
      "total_amount": float or null
    }
  ],
  "financial_summary": {
    "subtotal_amount": float, 
    "vat_rate_percentage": float or null,
    "vat_amount": float,
    "total_amount": float
  }
}

# STRICT ENFORCEMENT
- Double check the mathematical alignment: financial_summary.subtotal_amount + financial_summary.vat_amount MUST equal financial_summary.total_amount.
- Output ONLY the raw JSON string."""
# ──────────────────────────────────────────────────────────────────────────────


class GeminiExtractor:
    """
    Gemini Flash-based Vietnamese Invoice Extractor.
    Primary extraction engine replacing LayoutLMv3 stages 3+4.
    """

    def __init__(self, api_key: str):
        self.client = genai.Client(api_key=api_key)
        self._config = types.GenerateContentConfig(
            temperature=TEMPERATURE,
            max_output_tokens=MAX_TOKENS,
            response_mime_type="application/json",  # force JSON output
        )
        log.info("[Gemini] Initialized model: %s", GEMINI_MODEL)

    def extract(
        self,
        image_bytes: bytes,
        ocr_words:   Optional[list] = None,
        mime_type:   str = "image/jpeg",
    ) -> dict:
        """
        Extract invoice fields from image using Gemini Vision.

        Args:
            image_bytes: Raw image bytes (JPEG/PNG)
            ocr_words:   Optional list of OCR words for additional context
            mime_type:   Image MIME type

        Returns:
            Normalized dict matching pipeline schema with confidence scores
        """
        t0 = time.monotonic()

        # Build content parts
        image_part = types.Part.from_bytes(
            data=image_bytes,
            mime_type=mime_type,
        )

        prompt = INVOICE_EXTRACTION_PROMPT
        if ocr_words:
            ocr_text = " ".join(str(w) for w in ocr_words[:500])  # cap at 500 words
            prompt += f"\n\n═══ VĂN BẢN OCR THAM KHẢO ═══\n{ocr_text}"

        # Call Gemini
        try:
            response = self.client.models.generate_content(
                model=GEMINI_MODEL,
                contents=[image_part, prompt],
                config=self._config,
            )
            raw_text = response.text
        except Exception as e:
            raise RuntimeError(f"Gemini API error: {e}") from e

        elapsed = (time.monotonic() - t0) * 1000
        log.info("[Gemini] Extraction done in %.0fms", elapsed)

        # Parse JSON
        raw_data = self._parse_json(raw_text)

        # Normalize to pipeline schema
        return self._normalize(raw_data)

    def _parse_json(self, raw: str) -> dict:
        """Parse JSON from Gemini response, handle edge cases."""
        raw = raw.strip()

        # Strip markdown fences if present (shouldn't happen with response_mime_type)
        if raw.startswith("```"):
            match = re.search(r"```(?:json)?\s*([\s\S]*?)```", raw)
            if match:
                raw = match.group(1).strip()

        # Find JSON object
        start = raw.find("{")
        end   = raw.rfind("}") + 1
        if start == -1 or end == 0:
            raise ValueError(f"No JSON object found in response: {raw[:200]}")

        try:
            return json.loads(raw[start:end])
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON from Gemini: {e}\nRaw: {raw[:300]}") from e

    def _wrap(self, value, confidence: float = 0.95) -> dict:
        """Wrap value with confidence score — matches pipeline schema."""
        if value is None or value == "" or value == 0:
            return {"value": value if value is not None else "", "confidence": 0.0}
        return {"value": value, "confidence": round(confidence, 4)}

    def _safe_int(self, value) -> int:
        """Convert value to int safely."""
        if isinstance(value, (int, float)):
            return int(value)
        if isinstance(value, str):
            cleaned = re.sub(r"[.,\s]", "", value)
            try:
                return int(float(cleaned))
            except (ValueError, TypeError):
                return 0
        return 0

    def _normalize(self, raw: dict) -> dict:
        """
        Normalize Gemini output to exact pipeline schema.
        Maps the new English schema back to the internal Vietnamese-compatible schema.
        """
        result = {
            "seller":  {},
            "buyer":   {},
            "invoice": {},
            "items":   [],
        }

        inv_meta = raw.get("invoice_metadata", {}) or {}
        seller_raw = raw.get("seller", {}) or {}
        buyer_raw  = raw.get("buyer", {}) or {}
        summary    = raw.get("financial_summary", {}) or {}

        # ── Seller ────────────────────────────────────────────────────────────
        result["seller"]["name"]     = self._wrap(seller_raw.get("company_name") or "")
        result["seller"]["tax_code"] = self._wrap(seller_raw.get("tax_code") or "")
        result["seller"]["address"]  = self._wrap(seller_raw.get("address") or "")
        # Empty fallbacks for compatibility
        for f in ["phone", "bank_account", "bank_name", "tax_authority_code"]:
            result["seller"][f] = self._wrap("")

        # ── Buyer ─────────────────────────────────────────────────────────────
        result["buyer"]["name"]     = self._wrap(buyer_raw.get("company_name") or "")
        result["buyer"]["tax_code"] = self._wrap(buyer_raw.get("tax_code") or "")
        result["buyer"]["address"]  = self._wrap(buyer_raw.get("address") or "")
        result["buyer"]["full_name"] = self._wrap("")

        # ── Invoice ───────────────────────────────────────────────────────────
        result["invoice"]["number"] = self._wrap(inv_meta.get("invoice_number") or "")
        result["invoice"]["date"]   = self._wrap(inv_meta.get("invoice_date") or "")
        result["invoice"]["symbol"] = self._wrap(inv_meta.get("serial_number") or "")
        
        # Type mapping for UI consistency
        raw_type = raw.get("invoice_type", "")
        mapped_type = "GTGT" if "GTGT" in str(raw_type) else "BAN_HANG"
        result["invoice"]["type"] = self._wrap(mapped_type)

        subtotal     = self._safe_int(summary.get("subtotal_amount", 0))
        vat_amount   = self._safe_int(summary.get("vat_amount", 0))
        total_amount = self._safe_int(summary.get("total_amount", 0))
        vat_rate     = str(summary.get("vat_rate_percentage") or "")

        # Math validation for confidence
        if subtotal > 0 and total_amount > 0:
            footer_conf = 0.99 if abs((subtotal + vat_amount) - total_amount) < 2 else 0.80
        else:
            footer_conf = 0.85

        result["invoice"]["subtotal"]     = self._wrap(subtotal,     footer_conf)
        result["invoice"]["vat_amount"]   = self._wrap(vat_amount,   footer_conf)
        result["invoice"]["total_amount"] = self._wrap(total_amount, footer_conf)
        result["invoice"]["vat_rate"]     = self._wrap(vat_rate)
        
        for f in ["payment_method", "currency"]:
            result["invoice"][f] = self._wrap("")

        # ── Items ─────────────────────────────────────────────────────────────
        for item in raw.get("line_items", []) or []:
            name  = str(item.get("item_name") or "").strip()
            unit  = str(item.get("unit") or "").strip()
            qty   = item.get("quantity") or 0
            price = self._safe_int(item.get("unit_price", 0))
            total = self._safe_int(item.get("total_amount", 0))
            
            # Math validation
            expected = (qty or 0) * (price or 0)
            item_conf = 0.99 if abs(expected - total) < 2 else 0.80
            
            result["items"].append({
                "name":       self._wrap(name, 0.99),
                "unit":       self._wrap(unit, 0.95 if unit else 0.0),
                "quantity":   self._wrap(qty, item_conf),
                "unit_price": self._wrap(price, item_conf),
                "total":      self._wrap(total, item_conf),
                "discount":   self._wrap(0, 0.0),
                "vat_rate":   self._wrap("", 0.0),
                "line_tax":   self._wrap(0, 0.0),
                "row_total":  self._wrap(0, 0.0),
                "discount_rate": self._wrap("", 0.0),
            })

        return result

    def is_available(self) -> bool:
        """Check if Gemini API is accessible."""
        try:
            self.client.models.generate_content(
                model=GEMINI_MODEL,
                contents="test",
                config=types.GenerateContentConfig(max_output_tokens=10)
            )
            return True
        except Exception:
            return False


def build_extractor(api_key: str) -> Optional[GeminiExtractor]:
    """Factory function — returns None if API key not set."""
    if not api_key:
        log.warning("[Gemini] No API key provided — Gemini extraction disabled")
        return None
    try:
        extractor = GeminiExtractor(api_key)
        log.info("[Gemini] Extractor initialized successfully")
        return extractor
    except Exception as e:
        log.error("[Gemini] Failed to initialize: %s", e)
        return None
