"""
Phase 6 - Test on Real Invoices
Inference + Evaluation
"""

import json
from pathlib import Path
from typing import Dict, List

import cv2
from PIL import Image
import torch
from transformers import LayoutLMv3Processor, LayoutLMv3ForTokenClassification

# Import OCR
from src.run_ocr import OCRRunner
from src.engine import InvoiceExtractionEngine


class RealInvoiceTester:
    """Test trained model on real invoices"""
    
    def __init__(
        self,
        model_path: str,
        ocr_runner: OCRRunner,
        engine: InvoiceExtractionEngine,
    ):
        """
        Args:
            model_path: Path to trained model
            ocr_runner: OCR pipeline
            engine: Extraction engine
        """
        self.model_path = Path(model_path)
        
        # Load model
        print(f"📥 Loading model from {model_path}")
        self.processor = LayoutLMv3Processor.from_pretrained(model_path)
        self.model = LayoutLMv3ForTokenClassification.from_pretrained(model_path)
        self.model.eval()
        
        # Load label mappings
        with open(self.model_path / "label2id.json") as f:
            self.label2id = json.load(f)
        
        with open(self.model_path / "id2label.json") as f:
            self.id2label = {int(k): v for k, v in json.load(f).items()}
        
        self.ocr_runner = ocr_runner
        self.engine = engine
        
        print("✅ Model loaded successfully")
    
    def infer_single(self, image_path: Path) -> Dict:
        """Run inference on single invoice"""
        
        # Step 1: OCR
        print(f"\n🔄 Processing {image_path.name}")
        print("   1/3 Running OCR...")
        
        img_bgr, w, h = self.ocr_runner.load_image(image_path)
        dets = self.ocr_runner.detect_text_paddleocr(img_bgr)
        ocr_result = self.ocr_runner.merge_results(
            image_name=image_path.name,
            width=w,
            height=h,
            detections=dets,
            img_bgr=img_bgr,
        )
        
        # Step 2: Model inference
        print("   2/3 Running model inference...")
        
        # Prepare input
        words = [line["text"] for line in ocr_result["lines"]]
        bboxes = [self._normalize_bbox(line["bbox"], w, h) 
                  for line in ocr_result["lines"]]
        
        # Load image
        image = Image.open(image_path).convert("RGB")
        
        # Tokenize
        encoding = self.processor(
            images=image,
            text=words,
            boxes=bboxes,
            truncation=True,
            padding="max_length",
            max_length=512,
            return_tensors="pt",
        )
        
        # Predict
        with torch.no_grad():
            outputs = self.model(**encoding)
            predictions = outputs.logits.argmax(-1).squeeze().tolist()
        
        # Decode predictions
        predicted_labels = [
            self.id2label.get(pred, "O") 
            for pred in predictions
        ]
        
        # Step 3: Post-processing with engine
        print("   3/3 Extracting structured fields...")
        
        # Build NER output format
        ner_output = {
            "tokens": words,
            "bboxes": bboxes,
            "predicted_labels": predicted_labels,
            "confidence": [1.0] * len(words),  # Placeholder
        }
        
        # Extract fields
        result = self.engine.process(ner_output)
        
        print(f"   ✅ Done!")
        
        return result
    
    def _normalize_bbox(self, bbox: List[int], w: int, h: int) -> List[int]:
        """Normalize bbox to [0, 1000]"""
        x1, y1, x2, y2 = bbox
        return [
            int(1000 * x1 / w),
            int(1000 * y1 / h),
            int(1000 * x2 / w),
            int(1000 * y2 / h),
        ]
    
    def evaluate(
        self,
        predictions: List[Dict],
        ground_truths: List[Dict],
    ) -> Dict:
        """Calculate metrics"""
        
        results = {
            "total": len(predictions),
            "per_field": {},
            "overall": {},
        }
        
        # Define fields to evaluate
        fields = [
            "seller.name",
            "seller.tax_code",
            "buyer.name",
            "buyer.tax_code",
            "invoice.number",
            "invoice.date",
            "invoice.total_amount",
            "invoice.vat_rate",
            "invoice.vat_amount",
        ]
        
        for field in fields:
            correct = 0
            total = 0
            
            for pred, gt in zip(predictions, ground_truths):
                # Navigate nested dict
                pred_val = self._get_nested(pred, field)
                gt_val = self._get_nested(gt, field)
                
                total += 1
                if self._compare_values(pred_val, gt_val):
                    correct += 1
            
            accuracy = correct / total if total > 0 else 0
            results["per_field"][field] = {
                "correct": correct,
                "total": total,
                "accuracy": accuracy,
            }
        
        # Overall accuracy
        total_correct = sum(f["correct"] for f in results["per_field"].values())
        total_fields = sum(f["total"] for f in results["per_field"].values())
        
        results["overall"]["accuracy"] = (
            total_correct / total_fields if total_fields > 0 else 0
        )
        results["overall"]["correct"] = total_correct
        results["overall"]["total"] = total_fields
        
        return results
    
    def _get_nested(self, d: Dict, path: str):
        """Get nested dict value by path"""
        keys = path.split(".")
        val = d
        for k in keys:
            val = val.get(k, None)
            if val is None:
                return None
        return val
    
    def _compare_values(self, pred, gt):
        """Compare predicted and ground truth values"""
        if pred is None or gt is None:
            return False
        
        # String comparison (case-insensitive)
        if isinstance(pred, str) and isinstance(gt, str):
            return pred.strip().lower() == gt.strip().lower()
        
        # Numeric comparison (with tolerance)
        if isinstance(pred, (int, float)) and isinstance(gt, (int, float)):
            return abs(pred - gt) < max(1, abs(gt) * 0.01)  # 1% tolerance
        
        # Exact match
        return pred == gt


# ==========================================================
# MAIN TESTING SCRIPT
# ==========================================================

def main():
    # Paths
    model_path = "C:/Users/TanPhat/Documents/DEEPLEARNING/invoice_ocr/trained_model"
    real_invoices_dir = "C:/Users/TanPhat/Documents/DEEPLEARNING/invoice_ocr/real_invoices"
    ground_truth_file = "C:/Users/TanPhat/Documents/DEEPLEARNING/invoice_ocr/real_invoices/labels.json"
    output_file = "C:/Users/TanPhat/Documents/DEEPLEARNING/invoice_ocr/real_test_results.json"
    
    # Initialize
    ocr_runner = OCRRunner()
    engine = InvoiceExtractionEngine()
    tester = RealInvoiceTester(model_path, ocr_runner, engine)
    
    # Load ground truth
    with open(ground_truth_file, 'r', encoding='utf-8') as f:
        ground_truths = json.load(f)
    
    # Run inference on all invoices
    predictions = []
    
    for gt in ground_truths:
        image_path = Path(real_invoices_dir) / gt["image_file"]
        
        try:
            pred = tester.infer_single(image_path)
            predictions.append(pred)
        except Exception as e:
            print(f"❌ Error processing {image_path}: {e}")
            predictions.append({})
    
    # Evaluate
    print("\n" + "="*70)
    print("📊 EVALUATION RESULTS")
    print("="*70)
    
    results = tester.evaluate(predictions, ground_truths)
    
    # Print results
    print(f"\nOverall Accuracy: {results['overall']['accuracy']:.2%}")
    print(f"Correct: {results['overall']['correct']}/{results['overall']['total']}")
    
    print("\nPer-field accuracy:")
    for field, metrics in results["per_field"].items():
        acc = metrics["accuracy"]
        status = "✅" if acc > 0.9 else "⚠️" if acc > 0.7 else "❌"
        print(f"   {status} {field:30} {acc:.2%} ({metrics['correct']}/{metrics['total']})")
    
    # Save results
    output = {
        "predictions": predictions,
        "ground_truths": ground_truths,
        "metrics": results,
    }
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    
    print(f"\n✅ Results saved to {output_file}")


if __name__ == "__main__":
    main()
