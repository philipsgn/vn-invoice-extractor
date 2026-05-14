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
INVOICE_EXTRACTION_PROMPT = """Bạn là chuyên gia OCR trích xuất hóa đơn Việt Nam với 10 năm kinh nghiệm.
Phân tích hóa đơn trong ảnh và trả về JSON theo schema bên dưới.

═══ QUY TẮC BẮT BUỘC ═══

1. CHỈ trả về JSON thuần túy — KHÔNG có markdown, KHÔNG có ```json, KHÔNG có giải thích
2. Số tiền là số NGUYÊN, KHÔNG có dấu chấm/phẩy (ví dụ: 1500000 không phải 1.500.000)
3. Mã số thuế: 10 số, 13 số, hoặc định dạng 10-3 (ví dụ: 0123456789-001)
4. Ngày tháng: định dạng YYYY-MM-DD
5. unit_price × quantity − discount = total (kiểm tra toán học mỗi dòng)
6. ĐVT (unit) lấy CHÍNH XÁC từ hóa đơn: Cái/Hộp/Kg/Lít/m/Chiếc/Chai/Gói/Bộ/Cuộn/Tấm/Thùng/Cây/Viên/...
7. Nếu trường không có trong hóa đơn: string → "" (rỗng), number → 0
8. Phân biệt rõ:
   - Hóa đơn GTGT: có "Thuế GTGT", mã số thuế cả bên bán và bên mua
   - Hóa đơn bán hàng: không có dòng thuế GTGT riêng
9. subtotal (Cộng tiền hàng/Tổng tiền trước thuế) KHÁC total_amount (Tổng thanh toán)
10. Khi có 2 mức thuế (10% và 8%): ghi vat_rate = "10%,8%", tính tổng vat_amount

═══ JSON SCHEMA ═══

{
  "invoice_type": "HÓA ĐƠN BÁN HÀNG hoặc HÓA ĐƠN GIÁ TRỊ GIA TĂNG",
  "seller": {
    "name":              "tên công ty/hộ kinh doanh bên bán",
    "tax_code":          "mã số thuế bên bán",
    "address":           "địa chỉ đầy đủ bên bán",
    "phone":             "số điện thoại hoặc rỗng",
    "bank_account":      "số tài khoản ngân hàng hoặc rỗng",
    "bank_name":         "tên ngân hàng hoặc rỗng",
    "tax_authority_code":"mã cơ quan thuế (ví dụ T01N/1234567) hoặc rỗng"
  },
  "buyer": {
    "name":      "tên công ty/cá nhân bên mua",
    "tax_code":  "mã số thuế bên mua hoặc rỗng",
    "address":   "địa chỉ đầy đủ bên mua",
    "full_name": "họ tên người mua hàng hoặc rỗng"
  },
  "invoice": {
    "symbol":         "ký hiệu hóa đơn (ví dụ: 1K17KC, 2C26MY)",
    "number":         "số hóa đơn",
    "date":           "YYYY-MM-DD",
    "payment_method": "Tiền mặt / Chuyển khoản / TM/CK",
    "currency":       "VND",
    "subtotal":       số nguyên (Cộng tiền hàng/Tổng tiền chưa thuế),
    "vat_rate":       "10% hoặc 8% hoặc 0% hoặc 10%,8% nếu có 2 mức hoặc rỗng",
    "vat_amount":     số nguyên (tổng tiền thuế GTGT),
    "total_amount":   số nguyên (Tổng cộng thanh toán cuối cùng)
  },
  "items": [
    {
      "name":       "tên hàng hóa/dịch vụ đầy đủ",
      "unit":       "đơn vị tính chính xác từ hóa đơn",
      "quantity":   số nguyên hoặc số thực,
      "unit_price": số nguyên (đơn giá trước chiết khấu),
      "discount":   số nguyên (tiền chiết khấu, 0 nếu không có),
      "total":      số nguyên (thành tiền sau chiết khấu),
      "vat_rate":   "thuế suất dòng này nếu có, rỗng nếu không"
    }
  ]
}

═══ LƯU Ý ĐẶC BIỆT ═══

- "Cộng tiền hàng" hoặc "Tổng tiền hàng" = subtotal (KHÔNG phải total_amount)
- "Tổng cộng thanh toán" hoặc "Tổng thanh toán" = total_amount
- Tiền chiết khấu: có thể là "CK", "Chiết khấu", "Giảm giá" trong bảng items
- Mã CQT có thể nằm ở vị trí bất kỳ trên hóa đơn, thường format: T01N/XXXXXXX
- Số tài khoản ngân hàng: thường sau "STK:", "Số TK:", "Tài khoản:"
- Nếu thấy 2 bảng thuế (10% và 8%), liệt kê tổng cả hai vào vat_amount"""
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
        Adds confidence scores based on field presence and math validation.
        """
        result = {
            "seller":  {},
            "buyer":   {},
            "invoice": {},
            "items":   [],
        }

        # ── Seller ────────────────────────────────────────────────────────────
        seller = raw.get("seller", {}) or {}
        for field in ["name", "tax_code", "address", "phone",
                      "bank_account", "bank_name", "tax_authority_code"]:
            val = seller.get(field, "")
            result["seller"][field] = self._wrap(val or "")

        # ── Buyer ─────────────────────────────────────────────────────────────
        buyer = raw.get("buyer", {}) or {}
        for field in ["name", "tax_code", "address", "full_name"]:
            val = buyer.get(field, "")
            result["buyer"][field] = self._wrap(val or "")

        # ── Invoice ───────────────────────────────────────────────────────────
        inv = raw.get("invoice", {}) or {}
        for field in ["symbol", "number", "date", "payment_method",
                      "currency", "vat_rate"]:
            val = inv.get(field, "")
            result["invoice"][field] = self._wrap(val or "")

        result["invoice"]["type"] = self._wrap(
            raw.get("invoice_type", inv.get("type", "HÓA ĐƠN BÁN HÀNG"))
        )

        subtotal     = self._safe_int(inv.get("subtotal",     0))
        vat_amount   = self._safe_int(inv.get("vat_amount",   0))
        total_amount = self._safe_int(inv.get("total_amount", 0))

        # Validate footer math
        if subtotal > 0 and vat_amount >= 0 and total_amount > 0:
            expected = subtotal + vat_amount
            footer_conf = 0.99 if abs(expected - total_amount) / max(total_amount, 1) < 0.02 else 0.75
        else:
            footer_conf = 0.85

        result["invoice"]["subtotal"]     = self._wrap(subtotal,     footer_conf)
        result["invoice"]["vat_amount"]   = self._wrap(vat_amount,   footer_conf)
        result["invoice"]["total_amount"] = self._wrap(total_amount, footer_conf)
        result["invoice"]["vat_breakdown"] = []

        # ── Items ─────────────────────────────────────────────────────────────
        for item in raw.get("items", []) or []:
            name      = str(item.get("name",       "") or "").strip()
            unit      = str(item.get("unit",       "") or "").strip()
            qty       = item.get("quantity",   0) or 0
            price     = self._safe_int(item.get("unit_price", 0))
            disc      = self._safe_int(item.get("discount",   0))
            total     = self._safe_int(item.get("total",      0))
            vat_rate  = str(item.get("vat_rate",   "") or "").strip()

            # Math validation for confidence
            if qty > 0 and price > 0 and total > 0:
                expected  = qty * price - disc
                ratio     = expected / total if total > 0 else 0
                math_ok   = abs(ratio - 1.0) < 0.02   # 2% tolerance
                item_conf = 0.99 if math_ok else 0.75

                # Auto-correct unit_price if math wrong and total is reliable
                if not math_ok and total > 0 and qty > 0:
                    price = round((total + disc) / qty)
                    item_conf = 0.85  # computed, slightly less confident
            else:
                item_conf = 0.90

            result["items"].append({
                "name":       self._wrap(name,   0.99),
                "unit":       self._wrap(unit,   0.95 if unit else 0.0),
                "quantity":   self._wrap(qty,    item_conf),
                "unit_price": self._wrap(price,  item_conf),
                "discount":   self._wrap(disc,   0.95 if disc > 0 else 0.0),
                "total":      self._wrap(total,  item_conf),
                "vat_rate":   self._wrap(vat_rate, 0.90 if vat_rate else 0.0),
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
