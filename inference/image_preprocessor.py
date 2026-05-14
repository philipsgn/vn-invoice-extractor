# -*- coding: utf-8 -*-
"""
image_preprocessor.py
======================================================
Normalize real invoice images về chuẩn synthetic data.

PHÂN TÍCH SYNTHETIC DATA (800 samples):
  Width  : LUÔN 1588px  ← cố định
  Height : 123 sizes khác nhau (2246~2748+) ← thay đổi theo items

KẾT LUẬN:
  ✅ Chỉ normalize WIDTH về 1588px
  ✅ Height tự scale theo aspect ratio  ← KHÔNG pad/crop cứng
  ✅ Bbox proportion được bảo toàn hoàn toàn
======================================================
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Tuple

# ── Chỉ fix WIDTH — đây là hằng số duy nhất trong synthetic data ──
TARGET_W = 1588

MIN_DPI_EQUIVALENT = 150   # px/inch minimum để OCR hoạt động tốt
A4_INCH_W          = 8.27
MIN_W              = int(A4_INCH_W * MIN_DPI_EQUIVALENT)   # ~1240 px


def normalize_invoice_image(
    image_path: Path,
    output_path: Path = None,
    target_w:   int   = TARGET_W,
    pad_color:  Tuple[int, int, int] = (255, 255, 255),
) -> Tuple[np.ndarray, int, int]:
    """
    Normalize real invoice image về chuẩn synthetic data.

    Pipeline:
      1. Load ảnh
      2. Nếu landscape → rotate về portrait
      3. Upscale nếu width < MIN_W
      4. Scale WIDTH về target_w, HEIGHT theo aspect ratio  ← KEY CHANGE
      5. Optional: save output

    Args:
        image_path:  Path đến ảnh gốc
        output_path: Nếu có → save ảnh đã normalize
        target_w:    Target width (default 1588 — khớp synthetic)
        pad_color:   Không dùng nữa nhưng giữ lại để backward compat

    Returns:
        (img_bgr, final_w, final_h) — width=1588, height tự scale
    """
    # ── 1. Load ───────────────────────────────────────────────
    img = cv2.imread(str(image_path))
    if img is None:
        raise ValueError(f"Không đọc được ảnh: {image_path}")

    h, w = img.shape[:2]
    print(f"   Original size : {w} × {h} px  (ratio {w/h:.3f})")

    # ── 2. Auto-rotate landscape → portrait ──────────────────
    if w > h:
        img  = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
        h, w = img.shape[:2]
        print(f"   Rotated       : {w} × {h} px (landscape → portrait)")

    # ── 3. Upscale nếu quá nhỏ ───────────────────────────────
    if w < MIN_W:
        scale = MIN_W / w
        new_w = int(w * scale)
        new_h = int(h * scale)
        img   = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
        h, w  = img.shape[:2]
        print(f"   Upscaled      : {w} × {h} px (scale={scale:.2f})")

    # ── 4. Scale WIDTH → target_w, HEIGHT theo aspect ratio ──
    # ✅ KHÔNG pad/crop cứng height — synthetic height cũng variable
    scale  = target_w / w
    new_w  = target_w
    new_h  = int(h * scale)

    interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC
    img    = cv2.resize(img, (new_w, new_h), interpolation=interp)
    h, w   = img.shape[:2]

    print(f"   Normalized    : {w} × {h} px  (width fixed={target_w}, height scaled)")

    # ── 5. Save nếu cần ──────────────────────────────────────
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(output_path), img)
        print(f"   Saved         : {output_path}")

    return img, w, h


def normalize_batch(
    input_dir:  Path,
    output_dir: Path,
    target_w:   int = TARGET_W,
) -> None:
    """Normalize toàn bộ ảnh trong input_dir → output_dir."""
    input_dir  = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    images = sorted(
        list(input_dir.glob("*.png")) +
        list(input_dir.glob("*.jpg")) +
        list(input_dir.glob("*.jpeg"))
    )

    print(f"\n🔄 Normalizing {len(images)} images → {output_dir}")
    print(f"   Target width: {target_w}px (height auto-scaled)\n")

    ok = 0
    for img_path in images:
        print(f"📄 {img_path.name}")
        try:
            normalize_invoice_image(
                image_path  = img_path,
                output_path = output_dir / img_path.name,
                target_w    = target_w,
            )
            ok += 1
        except Exception as e:
            print(f"   ❌ Error: {e}")
        print()

    print(f"✅ Done: {ok}/{len(images)} images normalized")