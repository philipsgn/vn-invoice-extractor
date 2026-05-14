import sys
import json
import os
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
import torch
import re
from transformers import LayoutLMv3Processor, LayoutLMv3ForTokenClassification

# Ensure correct imports from project
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.run_ocr import OCRRunner
from src.engine import InvoiceExtractionEngine
from inference.bio_repair_inference import (
    prepare_words_and_bboxes,
    repair_bio_sequence,
    strip_nested_prefixes,
)
from inference.image_preprocessor import normalize_invoice_image

# Resolve paths relative to project root (portable for Docker/Cloud)
MODEL_PATH_FOOTER = str(
    PROJECT_ROOT / "model_final_layoutlmv3" / "model_final_footer" / "final_model"
)
TEST_IMAGE = str(PROJECT_ROOT / "inference" / "real_invoices" / "invoice_001.png")

# Optional overrides via environment variables for deployment
MODEL_PATH_FOOTER = str(Path(os.getenv("MODEL_PATH_FOOTER", MODEL_PATH_FOOTER)))
TEST_IMAGE = str(Path(os.getenv("TEST_IMAGE", TEST_IMAGE)))

def sliding_window_inference(model, processor, id2label, words, bboxes, device, pil_image):
    CHUNK_SIZE = 150
    STRIDE = 75
    MAX_CHUNKS = 8

    n_words = len(words)
    if n_words == 0:
        return [], []

    final_labels = ["O"] * n_words
    final_confs = [0.0] * n_words

    chunks = []
    start = 0
    while start < n_words and len(chunks) < MAX_CHUNKS:
        end = min(start + CHUNK_SIZE, n_words)
        chunks.append((start, end))
        if end == n_words:
            break
        start += STRIDE

    conflicts_resolved = 0

    for i, (start_idx, end_idx) in enumerate(chunks):
        chunk_words = words[start_idx:end_idx]
        chunk_bboxes = bboxes[start_idx:end_idx]

        encoding = processor(
            images=pil_image, text=chunk_words, boxes=chunk_bboxes,
            truncation=True, padding=True, return_tensors="pt"
        )
        
        n_tokens = len(encoding.input_ids[0])
        word_ids = encoding.word_ids(batch_index=0)
        encoding_on_device = {k: v.to(device) for k, v in encoding.items()}

        with torch.no_grad():
            outputs = model(**encoding_on_device)
            predictions = outputs.logits.argmax(-1).squeeze(0)
            probs = torch.softmax(outputs.logits, dim=-1)
            confidences = probs.max(-1).values.squeeze(0)

        chunk_word_labels = []
        chunk_word_confs = []
        current_word = None
        temp_preds = []
        temp_confs = []

        for idx, word_id in enumerate(word_ids):
            if word_id is None:
                continue
            pred_id = predictions[idx].item()
            conf = confidences[idx].item()
            if word_id != current_word:
                if current_word is not None:
                    # First subword label is authoritative for LayoutLMv3
                    chunk_word_labels.append(id2label[temp_preds[0]])
                    # Use mean confidence across all subwords for overlap resolution
                    chunk_word_confs.append(float(np.mean(temp_confs)))
                current_word = word_id
                temp_preds = [pred_id]
                temp_confs = [conf]
            else:
                temp_preds.append(pred_id)
                temp_confs.append(conf)

        if temp_preds:
            # First subword label is authoritative for LayoutLMv3
            chunk_word_labels.append(id2label[temp_preds[0]])
            # Use mean confidence across all subwords for overlap resolution
            chunk_word_confs.append(float(np.mean(temp_confs)))

        # Overlap Resolution & Merge
        non_o_count = 0
        for w_i, (lbl, conf) in enumerate(zip(chunk_word_labels, chunk_word_confs)):
            global_i = start_idx + w_i
            if global_i >= len(final_labels):
                break
                
            if lbl != "O":
                non_o_count += 1
                
            curr_lbl = final_labels[global_i]
            curr_conf = final_confs[global_i]
            
            if curr_lbl == "O" and lbl != "O":
                final_labels[global_i] = lbl
                final_confs[global_i] = conf
            elif curr_lbl != "O" and lbl != "O" and curr_lbl != lbl:
                conflicts_resolved += 1
                if conf > curr_conf:
                    final_labels[global_i] = lbl
                    final_confs[global_i] = conf
            elif curr_lbl == "O" and lbl == "O":
                if conf > curr_conf:
                    final_labels[global_i] = lbl
                    final_confs[global_i] = conf
            elif curr_lbl == lbl:
                if conf > curr_conf:
                    final_labels[global_i] = lbl
                    final_confs[global_i] = conf

        print(f"   [CHUNK {i+1}/{len(chunks)}] words[{start_idx}:{end_idx}] -> tokens={n_tokens} -> non-O labels found: {non_o_count}")

    print(f"   [MERGE] Overlap conflicts resolved: {conflicts_resolved}")
    
    return final_labels, final_confs

def clean_footer_value(raw_value: str) -> str:
    if raw_value is None:
        return None
    # If value contains spaces, it's multiple OCR tokens joined
    # Keep only the LAST numeric token (most likely the actual amount)
    parts = str(raw_value).strip().split()
    # Filter to parts that look like Vietnamese currency amounts
    # Pattern: digits with optional dots (1.000.000 or 550.000)
    currency_pattern = re.compile(r'^\d[\d\.]*\d$')
    valid_parts = [p for p in parts if currency_pattern.match(p)]
    if valid_parts:
        return valid_parts[-1]  # Take the last valid amount
    return parts[-1] if parts else None

def main():
    print(f"🔄 Loading Footer Model from: {MODEL_PATH_FOOTER}")
    device = "cpu"
    
    # 1. Load processor & model
    processor = LayoutLMv3Processor.from_pretrained(MODEL_PATH_FOOTER, apply_ocr=False)
    model = LayoutLMv3ForTokenClassification.from_pretrained(MODEL_PATH_FOOTER)
    model.to(device)
    model.eval()

    with open(Path(MODEL_PATH_FOOTER) / "label2id.json", encoding="utf-8") as f:
        label2id = json.load(f)
    with open(Path(MODEL_PATH_FOOTER) / "id2label.json", encoding="utf-8") as f:
        id2label = {int(k): v for k, v in json.load(f).items()}

    print(f"\n📄 Normalizing image: {TEST_IMAGE}")
    
    if not Path(TEST_IMAGE).exists():
        print(f"⚠️ Image not found: {TEST_IMAGE}")
        print("Please edit the TEST_IMAGE variable to point to a valid image.")
        return

    img_bgr, w, h = normalize_invoice_image(Path(TEST_IMAGE), output_path=None)
    
    print("🔄 Running OCR...")
    ocr_runner = OCRRunner(paddle_lang="vi", vietocr_model_name="vgg_transformer", device=device)
    dets = ocr_runner.detect_text_paddleocr(img_bgr)
    ocr_result = ocr_runner.merge_results(Path(TEST_IMAGE).name, w, h, dets, img_bgr)
    
    words, bboxes = prepare_words_and_bboxes(ocr_result, w, h)
    
    if not words:
        print("❌ No OCR tokens found.")
        return

    print("🧠 Running Inference with Sliding Window...")
    pil_image = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
    
    full_labels, full_confs = sliding_window_inference(model, processor, id2label, words, bboxes, device, pil_image)

    print("\n--- HYPOTHESIS 1: RAW BIO LABELS (BEFORE REPAIR) ---")
    print("Format: word_token | predicted_label | confidence")
    for w_tok, lbl, conf in list(zip(words, full_labels, full_confs))[-30:]:
        print(f"   {w_tok:<25} | {lbl:<15} | {conf:.4f}")
    print("----------------------------------------------------")
    
    repaired_labels = repair_bio_sequence(full_labels, words=words, bboxes=bboxes)
    
    print("\n📝 Raw BIO Output Sample (non-O):")
    for w_tok, lbl in zip(words, repaired_labels):
        if lbl != "O":
            print(f"   {w_tok:<25} | {lbl}")
            
    print("\n⚙️ Engine Extraction:")
    engine = InvoiceExtractionEngine(vat_tolerance=0.05, min_confidence=0.5)
    footer_ner = {
        "tokens": words,
        "bboxes": bboxes,
        "predicted_labels": repaired_labels,
        "confidence": full_confs,
    }
    raw_result = engine.process(footer_ner)
    
    extracted = strip_nested_prefixes(raw_result)
    
    print("\n✅ Extracted Footer Fields AFTER prefix stripping & cleaning:")
    invoice_data = extracted.get("invoice", {})
    subtotal = clean_footer_value(invoice_data.get('subtotal'))
    vat_amount = clean_footer_value(invoice_data.get('vat_amount'))
    total_amount = clean_footer_value(invoice_data.get('total_amount'))
    
    print(f"   Subtotal     (subtotal)    : {subtotal}")
    print(f"   VAT Amount   (vat_amount)  : {vat_amount}")
    print(f"   Grand Total  (total_amount): {total_amount}")
    
    # NOTE: Footer model may confuse SUBTOTAL vs GRAND_TOTAL on 
    # invoices where amounts are visually similar.
    # Fix: add more SUBTOTAL training examples to kaggle dataset.
    # This does NOT affect total_amount extraction correctness.

if __name__ == "__main__":
    main()
