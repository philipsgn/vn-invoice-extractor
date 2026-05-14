import os
import sys
import time
import logging
from pathlib import Path
from dotenv import load_dotenv
from gemini_extractor import build_extractor

# Setup logging
logging.basicConfig(level=logging.INFO)
log = logging.getLogger("test_gemini")

def test_extraction():
    load_dotenv()
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key or api_key == "your_api_key_here":
        log.error("GEMINI_API_KEY not found in .env or environment")
        return

    extractor = build_extractor(api_key)
    if not extractor:
        log.error("Failed to initialize Gemini extractor")
        return

    project_root = Path(__file__).resolve().parent.parent.parent
    default_images_dir = project_root / "synthetic_invoices" / "output" / "images"

    env_files = os.getenv("GEMINI_TEST_FILES", "").strip()
    if env_files:
        env_files = env_files.replace(",", os.pathsep)
        test_files = [p for p in env_files.split(os.pathsep) if p.strip()]
    else:
        test_files = [
            str(default_images_dir / "banhang_layout_04_1998.png"),
            str(default_images_dir / "banhang_layout_01_1201.png"),
            str(default_images_dir / "banhang_layout_02_1598.png"),
        ]

    for img_path in test_files:
        if not os.path.exists(img_path):
            log.warning(f"File not found: {img_path}")
            continue

        log.info(f"Testing: {os.path.basename(img_path)}")
        with open(img_path, "rb") as f:
            img_bytes = f.read()

        try:
            result = extractor.extract(img_bytes)
            # Print a concise summary of items
            items = result.get("items", [])
            log.info(f"Success! Found {len(items)} items.")
            for i, item in enumerate(items[:2]):  # show first 2 items
                name = item.get("name", {}).get("value", "N/A")
                qty = item.get("quantity", {}).get("value", 0)
                price = item.get("unit_price", {}).get("value", 0)
                total = item.get("total", {}).get("value", 0)
                log.info(f"  Item {i+1}: {name} | Qty: {qty} | Price: {price} | Total: {total}")
            
            if len(items) > 2:
                log.info(f"  ... and {len(items)-2} more items")
                
        except Exception as e:
            log.error(f"Failed to extract {img_path}: {e}")
        
        # Rate limit protection (always wait)
        log.info("Waiting 15s for quota...")
        time.sleep(15)

if __name__ == "__main__":
    test_extraction()
