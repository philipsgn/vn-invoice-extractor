# -*- coding: utf-8 -*-
"""
preprocessing.py
======================================================
PRODUCTION INPUT PREPROCESSING FOR INVOICE OCR
======================================================
Normalize user uploads (any size/format) to 200 DPI standard
for consistent OCR performance.

WHY PREPROCESSING?
- User uploads vary: 72 DPI photos, 300 DPI scans, PDFs, etc.
- Model trained on 200 DPI → production must match
- Ensures consistent OCR accuracy regardless of input

USAGE:
    from preprocessing import preprocess_invoice
    
    # Process user upload
    normalized_img = preprocess_invoice("user_upload.jpg")
    
    # Save for OCR
    normalized_img.save("normalized.png")
======================================================
"""

from pathlib import Path
from typing import Union, Tuple
import numpy as np
from PIL import Image, ImageEnhance, ImageOps


# ======================================================
# TARGET CONFIGURATION (MATCH TRAINING DATA)
# ======================================================
TARGET_DPI = 200
TARGET_WIDTH = 1588   # 794px viewport × 2.0 device scale
TARGET_HEIGHT = 2246  # 1123px viewport × 2.0 device scale

# Acceptable input size range (DPI estimation)
MIN_WIDTH = 600   # ~80 DPI
MAX_WIDTH = 3000  # ~400 DPI


# ======================================================
# PDF SUPPORT
# ======================================================
def pdf_to_image(pdf_path: Path, dpi: int = TARGET_DPI) -> Image.Image:
    """
    Convert PDF first page to image.
    
    Args:
        pdf_path: Path to PDF file
        dpi: Target DPI for conversion
        
    Returns:
        PIL Image object
    """
    try:
        from pdf2image import convert_from_path
    except ImportError:
        raise RuntimeError(
            "pdf2image not installed. Install: pip install pdf2image\n"
            "Also requires poppler: https://poppler.freedesktop.org/"
        )
    
    images = convert_from_path(pdf_path, dpi=dpi)
    return images[0]  # First page only


# ======================================================
# IMAGE ENHANCEMENT
# ======================================================
def enhance_image(img: Image.Image) -> Image.Image:
    """
    Apply basic image enhancements for better OCR.
    
    Enhancements:
    - Auto-contrast (improve text/background separation)
    - Sharpness boost (improve character edges)
    - Convert to RGB if needed
    
    Args:
        img: Input image
        
    Returns:
        Enhanced image
    """
    # Convert to RGB if needed
    if img.mode != 'RGB':
        img = img.convert('RGB')
    
    # Auto-contrast (improve text visibility)
    img = ImageOps.autocontrast(img, cutoff=1)
    
    # Slight sharpness boost (improve OCR accuracy)
    enhancer = ImageEnhance.Sharpness(img)
    img = enhancer.enhance(1.2)
    
    return img


# ======================================================
# RESIZE WITH ASPECT RATIO
# ======================================================
def resize_to_target(
    img: Image.Image,
    target_width: int = TARGET_WIDTH,
    target_height: int = TARGET_HEIGHT,
    maintain_aspect: bool = True
) -> Image.Image:
    """
    Resize image to target dimensions.
    
    Args:
        img: Input image
        target_width: Target width
        target_height: Target height
        maintain_aspect: If True, maintain aspect ratio (may not fill exactly)
        
    Returns:
        Resized image
    """
    width, height = img.size
    
    if maintain_aspect:
        # Calculate resize ratio (fit within target)
        ratio = min(target_width / width, target_height / height)
        
        new_width = int(width * ratio)
        new_height = int(height * ratio)
    else:
        # Stretch to exact size (may distort)
        new_width = target_width
        new_height = target_height
    
    # High-quality resize
    resized = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
    
    return resized


# ======================================================
# PAD TO EXACT SIZE
# ======================================================
def pad_to_size(
    img: Image.Image,
    target_width: int = TARGET_WIDTH,
    target_height: int = TARGET_HEIGHT,
    bg_color: Tuple[int, int, int] = (255, 255, 255)  # White
) -> Image.Image:
    """
    Pad image to exact size (center placement).
    
    Args:
        img: Input image (must be <= target size)
        target_width: Target width
        target_height: Target height
        bg_color: Background color (default: white)
        
    Returns:
        Padded image
    """
    width, height = img.size
    
    # Create blank canvas
    canvas = Image.new('RGB', (target_width, target_height), bg_color)
    
    # Calculate center position
    paste_x = (target_width - width) // 2
    paste_y = (target_height - height) // 2
    
    # Paste image in center
    canvas.paste(img, (paste_x, paste_y))
    
    return canvas


# ======================================================
# MAIN PREPROCESSING FUNCTION
# ======================================================
def preprocess_invoice(
    file_path: Union[str, Path],
    enhance: bool = True,
    target_width: int = TARGET_WIDTH,
    target_height: int = TARGET_HEIGHT,
    return_metadata: bool = False
) -> Union[Image.Image, Tuple[Image.Image, dict]]:
    """
    Preprocess invoice image/PDF for OCR.
    
    Pipeline:
    1. Load image (or convert PDF)
    2. Validate size
    3. Enhance (optional)
    4. Resize to target (maintain aspect ratio)
    5. Pad to exact size
    
    Args:
        file_path: Path to image or PDF file
        enhance: Apply image enhancements
        target_width: Target width (default: 1587 @ 200 DPI)
        target_height: Target height (default: 2246 @ 200 DPI)
        return_metadata: If True, return (image, metadata) tuple
        
    Returns:
        Preprocessed PIL Image, or (image, metadata) if return_metadata=True
        
    Raises:
        ValueError: If image too small or too large
        FileNotFoundError: If file doesn't exist
    """
    file_path = Path(file_path)
    
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    
    metadata = {
        "original_file": str(file_path),
        "file_type": file_path.suffix.lower()
    }
    
    # -------------------- LOAD --------------------
    if file_path.suffix.lower() == '.pdf':
        img = pdf_to_image(file_path, dpi=TARGET_DPI)
        metadata["source"] = "pdf"
    else:
        img = Image.open(file_path)
        metadata["source"] = "image"
    
    orig_width, orig_height = img.size
    metadata["original_size"] = (orig_width, orig_height)
    
    # -------------------- VALIDATE --------------------
    if orig_width < MIN_WIDTH:
        raise ValueError(
            f"Image too small: {orig_width}px wide (min: {MIN_WIDTH}px)\n"
            f"Please provide higher resolution image."
        )
    
    if orig_width > MAX_WIDTH:
        print(f"⚠️  Warning: Very large image ({orig_width}px), may be slow")
    
    # -------------------- ENHANCE --------------------
    if enhance:
        img = enhance_image(img)
        metadata["enhanced"] = True
    else:
        # Still convert to RGB if needed
        if img.mode != 'RGB':
            img = img.convert('RGB')
        metadata["enhanced"] = False
    
    # -------------------- RESIZE --------------------
    img = resize_to_target(img, target_width, target_height, maintain_aspect=True)
    resized_width, resized_height = img.size
    metadata["resized_size"] = (resized_width, resized_height)
    
    # -------------------- PAD --------------------
    img = pad_to_size(img, target_width, target_height)
    metadata["final_size"] = (target_width, target_height)
    metadata["target_dpi"] = TARGET_DPI
    
    if return_metadata:
        return img, metadata
    else:
        return img


# ======================================================
# BATCH PROCESSING
# ======================================================
def preprocess_batch(
    input_dir: Path,
    output_dir: Path,
    file_pattern: str = "*.png"
):
    """
    Preprocess all images in a directory.
    
    Args:
        input_dir: Directory containing input files
        output_dir: Directory to save preprocessed images
        file_pattern: Glob pattern for files (default: *.png)
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    files = list(input_dir.glob(file_pattern))
    print(f"Found {len(files)} files in {input_dir}")
    
    success = 0
    failed = 0
    
    for file_path in files:
        try:
            img = preprocess_invoice(file_path, enhance=True)
            
            output_path = output_dir / file_path.name
            img.save(output_path, "PNG", optimize=True)
            
            success += 1
            print(f"✅ {file_path.name}")
            
        except Exception as e:
            failed += 1
            print(f"❌ {file_path.name}: {e}")
    
    print(f"\n✅ Success: {success}/{len(files)}")
    if failed > 0:
        print(f"❌ Failed: {failed}/{len(files)}")


# ======================================================
# QUICK TEST FUNCTION
# ======================================================
def test_preprocessing(image_path: str):
    """
    Test preprocessing on a single image and show results.
    
    Args:
        image_path: Path to test image
    """
    print("Testing preprocessing pipeline...")
    print(f"Input: {image_path}\n")
    
    img, metadata = preprocess_invoice(image_path, return_metadata=True)
    
    print("Metadata:")
    for key, value in metadata.items():
        print(f"  {key}: {value}")
    
    print(f"\nFinal image size: {img.size}")
    print(f"Expected: ({TARGET_WIDTH}, {TARGET_HEIGHT})")
    print("✅ Test complete!")
    
    return img


# ======================================================
# CLI ENTRY POINT
# ======================================================
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Preprocess invoice images/PDFs for OCR"
    )
    
    parser.add_argument(
        "input",
        help="Input file or directory"
    )
    
    parser.add_argument(
        "--output",
        help="Output file or directory (default: preprocessed/)",
        default="preprocessed"
    )
    
    parser.add_argument(
        "--batch",
        action="store_true",
        help="Batch process directory"
    )
    
    parser.add_argument(
        "--no-enhance",
        action="store_true",
        help="Skip image enhancement"
    )
    
    parser.add_argument(
        "--test",
        action="store_true",
        help="Test mode (show metadata)"
    )
    
    args = parser.parse_args()
    
    input_path = Path(args.input)
    
    if args.test:
        # Test mode
        img = test_preprocessing(str(input_path))
        output_path = Path("test_preprocessed.png")
        img.save(output_path)
        print(f"\nSaved to: {output_path}")
        
    elif args.batch:
        # Batch mode
        preprocess_batch(
            input_dir=input_path,
            output_dir=Path(args.output)
        )
        
    else:
        # Single file mode
        img = preprocess_invoice(
            input_path,
            enhance=not args.no_enhance
        )
        
        output_path = Path(args.output)
        if output_path.is_dir():
            output_path = output_path / input_path.name
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        img.save(output_path, "PNG", optimize=True)
        
        print(f"✅ Preprocessed: {output_path}")