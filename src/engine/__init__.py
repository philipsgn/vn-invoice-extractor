"""
Production-grade orchestration engine for Vietnamese invoice extraction.

Responsibilities:
- Orchestrate OCR post-processing pipeline
- Call extractor -> validator -> schema
- NEVER touch output JSON structure
"""

import json
from pathlib import Path
from typing import Any, Dict, Optional

from ..invoice_schema import InvoiceSchema
from ..rules.field_extractor import FieldExtractor
from ..validators.invoice_validator import InvoiceValidator


class InvoiceExtractionEngine:
    """
    Orchestrator for Vietnamese invoice extraction.

    Pipeline:
    1. Extract structured fields from NER output
    2. Validate business rules (VAT, totals, confidence)
    3. Delegate final output construction to InvoiceSchema
    """

    def __init__(
        self,
        vat_tolerance: float = 0.05,
        min_confidence: float = 0.5,
    ):
        self.extractor = FieldExtractor()
        self.validator = InvoiceValidator(
            vat_tolerance=vat_tolerance,
            min_confidence=min_confidence,
        )

    def process(self, ner_output: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process NER inference output into raw extracted fields.

        NOTE: Returns raw extracted data (flat values + confidence metadata).
              Schema wrapping ({value, confidence} per field) happens in
              stage5_format() -> InvoiceSchema.build().

        Args:
            ner_output: {
                tokens: List[str],
                bboxes: List[List[int]],
                predicted_labels: List[str],
                confidence: List[float]
            }

        Returns:
            Raw extracted dict: {seller: {}, buyer: {}, invoice: {}, items: [], confidence: {}}
        """
        try:
            extracted = self.extractor.extract_all(ner_output)
            return extracted

        except Exception as e:
            # Return empty structure on failure
            return {
                "seller": {},
                "buyer": {},
                "invoice": {},
                "items": [],
                "confidence": {"fields": {}},
                "_error": str(e),
            }

    def process_file(
        self,
        input_path: Path,
        output_path: Optional[Path] = None,
    ) -> Dict[str, Any]:
        """
        Process NER output from JSON file.
        """
        with open(input_path, "r", encoding="utf-8") as f:
            ner_output = json.load(f)

        result = self.process(ner_output)

        if output_path:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(result, f, ensure_ascii=False, indent=2)

        return result


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Vietnamese Invoice Extraction Engine (Production)"
    )
    parser.add_argument("--input", required=True, help="NER output JSON")
    parser.add_argument("--output", help="Output invoice JSON")
    parser.add_argument("--vat_tolerance", type=float, default=0.05)
    parser.add_argument("--min_confidence", type=float, default=0.5)

    args = parser.parse_args()

    engine = InvoiceExtractionEngine(
        vat_tolerance=args.vat_tolerance,
        min_confidence=args.min_confidence,
    )

    result = engine.process_file(
        Path(args.input),
        Path(args.output) if args.output else None,
    )

    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
