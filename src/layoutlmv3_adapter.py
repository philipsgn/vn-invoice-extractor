import re
from typing import Dict, Any

def _clean_number(val: Any) -> float:
    if not val:
        return 0.0
    if isinstance(val, (int, float)):
        return float(val)
    if isinstance(val, str):
        # Remove currency symbols and spaces
        cleaned = re.sub(r'[^\d.,]', '', val)
        if not cleaned: return 0.0
        # Replace dots/commas safely
        cleaned = cleaned.replace('.', '').replace(',', '.')
        try:
            return float(cleaned)
        except ValueError:
            return 0.0
    return 0.0

class LayoutLMv3Adapter:
    """
    Adapter to transform LayoutLMv3 raw output (via FieldExtractor) 
    into the unified JSON schema used by the Gemini parser.
    """
    def __init__(self):
        from src.rules.field_extractor import FieldExtractor
        self.extractor = FieldExtractor()

    def convert_to_unified_schema(self, ner_output: Dict[str, Any]) -> Dict[str, Any]:
        """
        Takes raw LayoutLMv3 output (tokens, bboxes, labels) and maps it
        to the strict Gemini-compatible JSON schema.
        """
        # Step 1: Group coordinates and text into fields using existing logic
        raw_fields = self.extractor.extract_all(ner_output)
        
        # Step 2: Extract base sections
        invoice = raw_fields.get("invoice", {})
        seller = raw_fields.get("seller", {})
        buyer = raw_fields.get("buyer", {})
        items = raw_fields.get("items", [])
        
        # Rule 1: Invoice Type
        raw_type = str(invoice.get("type", "")).upper()
        if "GIÁ TRỊ GIA TĂNG" in raw_type or "GTGT" in raw_type or "VAT" in raw_type:
            invoice_type = "HOA_DON_GTGT"
        else:
            invoice_type = "HOA_DON_BAN_HANG"

        # Rule 2: Clean strings to pure numeric values
        subtotal = _clean_number(invoice.get("subtotal"))
        vat_amount = _clean_number(invoice.get("vat_amount"))
        total_amount = _clean_number(invoice.get("total_amount"))
        
        vat_rate_str = invoice.get("vat_rate", "")
        vat_rate_percentage = None
        if vat_rate_str:
            match = re.search(r'(\d+)', str(vat_rate_str))
            if match:
                vat_rate_percentage = float(match.group(1))

        # Enforce HOA_DON_BAN_HANG logic
        if invoice_type == "HOA_DON_BAN_HANG":
            vat_rate_percentage = None
            vat_amount = 0.0

        # Mathematical alignment: subtotal + vat_amount MUST equal total_amount
        if invoice_type == "HOA_DON_GTGT":
            # Simple reconciliation
            if abs((subtotal + vat_amount) - total_amount) > 1.0:
                if total_amount > 0 and vat_amount > 0 and subtotal == 0:
                    subtotal = total_amount - vat_amount
                elif subtotal > 0 and vat_amount > 0 and total_amount == 0:
                    total_amount = subtotal + vat_amount

        # Rule 3: Line Items Grouping
        line_items = []
        for i, item in enumerate(items):
            line_items.append({
                "sequence_number": i + 1,
                "item_name": item.get("name") or None,
                "unit": item.get("unit") or None,
                "quantity": _clean_number(item.get("quantity")) if item.get("quantity") else None,
                "unit_price": _clean_number(item.get("unit_price")) if item.get("unit_price") else None,
                "total_amount": _clean_number(item.get("total")) if item.get("total") else None,
            })

        # Final unified schema mapping
        return {
            "invoice_type": invoice_type,
            "invoice_metadata": {
                "form_number": invoice.get("form") or None,
                "serial_number": invoice.get("symbol") or None,
                "invoice_number": invoice.get("number") or None,
                "invoice_date": invoice.get("date") or None
            },
            "seller": {
                "company_name": seller.get("name") or None,
                "tax_code": seller.get("tax_code") or None,
                "address": seller.get("address") or None
            },
            "buyer": {
                "company_name": buyer.get("name") or None,
                "tax_code": buyer.get("tax_code") or None,
                "address": buyer.get("address") or None
            },
            "line_items": line_items,
            "financial_summary": {
                "subtotal_amount": subtotal,
                "vat_rate_percentage": vat_rate_percentage,
                "vat_amount": vat_amount,
                "total_amount": total_amount
            }
        }
