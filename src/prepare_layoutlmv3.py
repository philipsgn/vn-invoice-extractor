# -*- coding: utf-8 -*-
"""
prepare_layoutlmv3.py  —  Approach B: 2 Models
================================================
Chuyển đổi labeled_tokens/*.json → HuggingFace Dataset
format chuẩn cho LayoutLMv3 training.

Chạy riêng cho từng model:

  # Header model (23 labels)
  python prepare_layoutlmv3.py --mode header \
      --labeled_dir output/labeled_tokens_header \
      --images_dir  output/images \
      --output_dir  output/hf_dataset_header \
      --label2id    label2id_header.json

  # Table model (25 labels)
  python prepare_layoutlmv3.py --mode table \
      --labeled_dir output/labeled_tokens_table \
      --images_dir  output/images \
      --output_dir  output/hf_dataset_table \
      --label2id    label2id_table.json

INPUT:
  labeled_tokens_{mode}/*.json  ← từ auto_label_tokens.py --mode {mode}
  images/*.png                  ← ảnh invoice gốc
  label2id_{mode}.json          ← schema mapping

OUTPUT:
  hf_dataset_{mode}/{train,val,test}/  ← Arrow format

SCHEMA VERSIONS:
  header_v2 : 12 entity types, 25 labels (B/I×12 + O)  — added SELLER_TAX_AUTHORITY_CODE
  table_v2  : 12 entity types, 25 labels (B/I×12 + O)
"""

import json
import argparse
import random
import logging
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Set

from PIL import Image
from tqdm import tqdm
from transformers import LayoutLMv3Processor

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ======================================================================
# MODE CONFIGS — khớp 1:1 với auto_label_tokens.py _SCHEMAS dict
# ======================================================================

_MODE_CONFIGS = {

    "header": {
        "schema_version":      "header_v2",
        "expected_n_labels":   25,  # B/I × 12 + O
        "default_labeled_dir": "output/labeled_tokens_header",
        "default_output_dir":  "output/hf_dataset_header",
        "default_label2id":    "label2id_header.json",

        "entity_groups": [

            ("INVOICE META", [
                "INVOICE_TYPE",
                "INVOICE_DATE",
                "INVOICE_SYMBOL",
                "INVOICE_NUMBER",
            ]),

            ("SELLER", [
                "SELLER_NAME",
                "SELLER_TAX_CODE",
                "SELLER_TAX_AUTHORITY_CODE",
                "SELLER_ADDRESS",
            ]),

            ("BUYER", [
                "BUYER_NAME",
                "BUYER_TAX_CODE",
                "BUYER_ADDRESS",
            ]),

            ("PAYMENT", [
                "PAYMENT_METHOD",
            ]),
        ],
    },


    "table": {
        "schema_version":      "table_v2",
        "expected_n_labels":   19,  # B/I × 9 + O
        "default_labeled_dir": "output/labeled_tokens_table",
        "default_output_dir":  "output/hf_dataset_table",
        "default_label2id":    "label2id_table.json",

        "entity_groups": [

            ("TABLE ITEMS", [
                "ITEM_NAME",
                "ITEM_UNIT",
                "ITEM_QUANTITY",
                "ITEM_UNIT_PRICE",
                "ITEM_TOTAL_PRICE",
                "ITEM_DISCOUNT",
                "ITEM_VAT_RATE",
                "ITEM_LINE_TAX",
                "ITEM_ROW_TOTAL",
            ]),

        ],
    },


    "footer": {
        "schema_version":      "footer_v1",
        "expected_n_labels":   7,  # B/I × 3 + O
        "default_labeled_dir": "output/labeled_tokens_footer",
        "default_output_dir":  "output/hf_dataset_footer",
        "default_label2id":    "label2id_footer.json",

        "entity_groups": [

            ("TOTALS BLOCK", [
                "SUBTOTAL",
                "VAT_AMOUNT",
                "GRAND_TOTAL",
            ]),

        ],
    },

}

MAX_SEQ_LENGTH = 512
DEFAULT_TRAIN  = 0.8
DEFAULT_VAL    = 0.1


# ======================================================================
# LOAD & VALIDATE LABEL MAP
# ======================================================================

def load_label2id(label2id_path: Path, mode: str) -> Dict[str, int]:
    """
    Load label2id.json với full validation:
      1. File tồn tại
      2. Có key "O"
      3. Số lượng labels đúng với mode (23 | 25)
      4. Không có duplicate ID
      5. B/I symmetry check
    """
    if not label2id_path.exists():
        raise FileNotFoundError(
            f"Không tìm thấy: {label2id_path}\n"
            f"Hint: generate bằng label2id_{mode}.json từ schema audit."
        )

    data: Dict[str, int] = json.loads(label2id_path.read_text(encoding="utf-8"))
    cfg  = _MODE_CONFIGS[mode]
    expected = cfg["expected_n_labels"]

    # 1. "O" tồn tại
    if "O" not in data:
        raise ValueError("label2id.json thiếu key 'O' — kiểm tra schema.")

    # 2. Số lượng labels
    n = len(data)
    if n != expected:
        raise ValueError(
            f"[{mode.upper()}] label2id có {n} labels, kỳ vọng {expected}.\n"
            f"Hint: Dùng đúng label2id_{mode}.json chưa?"
        )

    # 3. Duplicate ID
    id_counts = Counter(data.values())
    dups = {lbl: idx for lbl, idx in data.items() if id_counts[idx] > 1}
    if dups:
        raise ValueError(f"Duplicate IDs trong label2id: {dups}")

    # 4. B/I symmetry
    b_fields = {k[2:] for k in data if k.startswith("B-")}
    i_fields = {k[2:] for k in data if k.startswith("I-")}
    if b_fields != i_fields:
        diff = b_fields.symmetric_difference(i_fields)
        raise ValueError(f"B/I asymmetry: {diff}")

    logger.info(
        f"[{mode.upper()}] label2id OK — "
        f"{n} labels, O={data['O']}, "
        f"{len(b_fields)} entity types"
    )
    return data


# ======================================================================
# SCHEMA VALIDATION (PRE-FLIGHT)
# ======================================================================

def validate_schema_against_files(
    labeled_files: List[Path],
    label2id:      Dict[str, int],
    mode:          str,
    sample_size:   int = 50,
) -> bool:
    """
    Scan sample_size files để phát hiện sớm mismatch TRƯỚC khi process
    toàn bộ dataset — fail fast thay vì corrupt dataset im lặng.

    Kiểm tra:
      1. schema_version khớp với mode config
      2. Mọi label trong file đều có trong label2id
    """
    expected_sv = _MODE_CONFIGS[mode]["schema_version"]
    files_to_check = labeled_files[:sample_size]

    logger.info(
        f"[{mode.upper()}] 🔍 Pre-flight validation "
        f"({min(sample_size, len(labeled_files))} files)..."
    )

    unknown_labels:      Set[str] = set()
    wrong_version_count: int      = 0

    for f in files_to_check:
        try:
            data = json.loads(f.read_text(encoding="utf-8"))

            # Check schema_version
            sv = data.get("schema_version", "UNKNOWN")
            if sv != expected_sv:
                wrong_version_count += 1
                if wrong_version_count == 1:
                    logger.warning(
                        f"⚠️  '{f.name}': schema_version='{sv}', "
                        f"kỳ vọng '{expected_sv}'. "
                        f"Có thể chạy nhầm mode ở auto_label_tokens.py."
                    )

            # Check labels tồn tại trong label2id
            for tok in data.get("tokens", []):
                lbl = tok.get("label", "O")
                if lbl not in label2id:
                    unknown_labels.add(lbl)

        except Exception as e:
            logger.warning(f"Không đọc được '{f.name}': {e}")

    # ── Report ────────────────────────────────────────────────────────
    is_ok = True

    if unknown_labels:
        logger.error(
            f"🔴 SCHEMA MISMATCH — {len(unknown_labels)} labels KHÔNG có "
            f"trong label2id_{mode}.json:\n"
            f"   {sorted(unknown_labels)}\n"
            f"   Fix: (1) Dùng đúng label2id_{mode}.json\n"
            f"        (2) Chạy lại: auto_label_tokens.py --mode {mode}"
        )
        is_ok = False
    else:
        logger.info(f"[{mode.upper()}] ✅ All labels present in label2id")

    if wrong_version_count > 0:
        pct = wrong_version_count / len(files_to_check) * 100
        logger.warning(
            f"⚠️  {wrong_version_count}/{len(files_to_check)} files ({pct:.0f}%) "
            f"có schema_version sai. Chạy lại auto_label_tokens.py --mode {mode}."
        )
        # version sai nhưng labels OK → vẫn cho tiếp tục với warning
        # version sai + labels sai → is_ok đã False ở trên

    return is_ok


# ======================================================================
# NORMALIZE BBOX
# ======================================================================

def normalize_bbox(bbox: List[int], width: int, height: int) -> List[int]:
    """
    Normalize [x1,y1,x2,y2] → [0, 1000] theo yêu cầu LayoutLMv3.
    Clamp về [0, 1000] + đảm bảo x2>=x1, y2>=y1.
    """
    x1, y1, x2, y2 = bbox
    x1 = min(max(0, int(x1 * 1000 / width)),  1000)
    y1 = min(max(0, int(y1 * 1000 / height)), 1000)
    x2 = min(max(0, int(x2 * 1000 / width)),  1000)
    y2 = min(max(0, int(y2 * 1000 / height)), 1000)
    if x2 < x1: x2 = x1
    if y2 < y1: y2 = y1
    return [x1, y1, x2, y2]


# ======================================================================
# LOAD LABELED TOKEN FILE
# ======================================================================

def load_labeled_tokens(labeled_path: Path) -> Dict[str, Any]:
    """
    Load 1 labeled token JSON → dict chuẩn để pass vào prepare_sample.

    Token JSON format (từ auto_label_tokens.py):
    {
      "invoice_id": "...", "image": "...",
      "width": …, "height": …,
      "schema_version": "header_v2" | "table_v2",
      "tokens": [
        {"id":…, "text":"…", "bbox":[x1,y1,x2,y2],
         "label":"B-SELLER_NAME", "field":"seller_name", "confidence":0.97}
      ]
    }
    """
    data = json.loads(labeled_path.read_text(encoding="utf-8"))

    words:  List[str]       = []
    bboxes: List[List[int]] = []
    labels: List[str]       = []

    for tok in data.get("tokens", []):
        text = tok.get("text", "").strip()
        if not text:
            continue
        words.append(text)
        bboxes.append(tok.get("bbox", [0, 0, 0, 0]))
        labels.append(tok.get("label", "O"))

    return {
        "invoice_id":     data.get("invoice_id", labeled_path.stem),
        "image_path_rel": data.get("image", ""),
        "width":          max(1, int(data.get("width",  1))),
        "height":         max(1, int(data.get("height", 1))),
        "schema_version": data.get("schema_version", "UNKNOWN"),
        "words":          words,
        "bboxes":         bboxes,
        "labels":         labels,
    }


# ======================================================================
# PREPARE SINGLE SAMPLE → LayoutLMv3 encoding
# ======================================================================

def prepare_sample(
    sample:     Dict[str, Any],
    processor:  LayoutLMv3Processor,
    label2id:   Dict[str, int],
    images_dir: Path,
    mode:       str,
) -> Optional[Dict[str, Any]]:
    """
    Convert 1 invoice sample → LayoutLMv3 encoding dict.

    STRICT label lookup:
      - Label có trong label2id → dùng đúng ID
      - Label KHÔNG có → WARNING + fallback O (không silent)
        Trường hợp này rất hiếm sau validate_schema_against_files()
        nhưng vẫn handle gracefully thay vì crash toàn batch.

    Image resolution:
      Try 1: images_dir / image_path_rel.name
      Try 2: images_dir / {invoice_id}.png
    """
    words      = sample["words"]
    bboxes_raw = sample["bboxes"]
    labels_str = sample["labels"]
    width      = sample["width"]
    height     = sample["height"]
    invoice_id = sample["invoice_id"]

    if not words:
        logger.debug(f"[{invoice_id}] Empty tokens, skip.")
        return None

    # ── Resolve image ─────────────────────────────────────────────────
    image_path = images_dir / Path(sample["image_path_rel"]).name
    if not image_path.exists():
        alt = images_dir / f"{invoice_id}.png"
        if alt.exists():
            image_path = alt
        else:
            logger.warning(f"[{invoice_id}] Image không tìm thấy → skip")
            return None

    image = Image.open(image_path).convert("RGB")

    # ── Normalize bboxes ─────────────────────────────────────────────
    bboxes_norm = [normalize_bbox(bb, width, height) for bb in bboxes_raw]

    # ── STRICT label → ID (không silent fallback) ────────────────────
    o_id = label2id["O"]
    word_label_ids:    List[int] = []
    unknown_in_sample: List[str] = []

    for lbl in labels_str:
        if lbl in label2id:
            word_label_ids.append(label2id[lbl])
        else:
            unknown_in_sample.append(lbl)
            word_label_ids.append(o_id)

    if unknown_in_sample:
        logger.warning(
            f"[{invoice_id}] {len(unknown_in_sample)} tokens có label KHÔNG "
            f"trong label2id_{mode}.json → fallback O. "
            f"Labels: {sorted(set(unknown_in_sample))}"
        )

    # ── LayoutLMv3 encoding ──────────────────────────────────────────
    encoding = processor(
        images=image,
        text=words,
        boxes=bboxes_norm,
        word_labels=word_label_ids,
        truncation=True,
        max_length=MAX_SEQ_LENGTH,
        padding="max_length",
        return_tensors=None,   # list → compatible với datasets, không phải tensor
    )

    return {
        "id":             invoice_id,
        "input_ids":      encoding["input_ids"],
        "attention_mask": encoding["attention_mask"],
        "bbox":           encoding["bbox"],
        "labels":         encoding["labels"],
        "pixel_values":   encoding["pixel_values"][0],   # float32 [3,224,224] — required by LayoutLMv3ForTokenClassification
    }


# ======================================================================
# SPLIT DATASET
# ======================================================================

def split_files(
    files:       List[Path],
    train_ratio: float,
    val_ratio:   float,
    seed:        int = 42,
) -> Tuple[List[Path], List[Path], List[Path]]:
    """Reproducible shuffle + split → train / val / test."""
    files = files.copy()
    random.seed(seed)
    random.shuffle(files)

    n       = len(files)
    n_train = int(n * train_ratio)
    n_val   = int(n * val_ratio)

    return (
        files[:n_train],
        files[n_train: n_train + n_val],
        files[n_train + n_val:],
    )


# ======================================================================
# BUILD SPLIT
# ======================================================================

def build_split(
    labeled_files: List[Path],
    processor:     LayoutLMv3Processor,
    label2id:      Dict[str, int],
    images_dir:    Path,
    split_name:    str,
    mode:          str,
) -> Tuple[List[Dict[str, Any]], Counter]:
    """
    Process danh sách labeled files → list samples + label Counter.
    Counter dùng để báo cáo label distribution sau khi xong.
    """
    samples:       List[Dict[str, Any]] = []
    label_counter: Counter              = Counter()
    skipped = 0

    for lf in tqdm(labeled_files, desc=f"  [{split_name:5s}]", unit="inv"):
        try:
            raw = load_labeled_tokens(lf)

            if not raw["words"]:
                logger.debug(f"[{lf.stem}] No tokens, skip.")
                skipped += 1
                continue

            label_counter.update(raw["labels"])

            sample = prepare_sample(raw, processor, label2id, images_dir, mode)
            if sample is None:
                skipped += 1
                continue

            samples.append(sample)

        except Exception as e:
            logger.error(f"[{lf.stem}] {e}")
            skipped += 1

    logger.info(f"  [{split_name}] {len(samples)} OK  /  {skipped} skipped")
    return samples, label_counter


# ======================================================================
# SAVE AS HF DATASET
# ======================================================================

def save_hf_dataset(samples: List[Dict[str, Any]], output_dir: Path) -> None:
    """
    Lưu samples thành HuggingFace Dataset (Arrow format).

    Features khai báo tường minh:
      labels:       Value("int64")                      → chứa được -100 (HF Trainer ignore_index)
      pixel_values: Array3D(float32, [3,224,224])       → preprocessed image tensor cho LayoutLMv3

    Warn nếu overwrite dataset cũ để tránh nhầm.
    """
    try:
        from datasets import Dataset, Features, Sequence, Value, Array3D
    except ImportError:
        raise ImportError("pip install datasets")

    if not samples:
        logger.warning(f"Không có samples → bỏ qua {output_dir}")
        return

    features = Features({
        "id":             Value("string"),
        "input_ids":      Sequence(Value("int64")),
        "attention_mask": Sequence(Value("int64")),
        "bbox":           Sequence(Sequence(Value("int64"))),
        "labels":         Sequence(Value("int64")),
        "pixel_values":   Array3D(dtype="float32", shape=(3, 224, 224)),
    })

    output_dir.mkdir(parents=True, exist_ok=True)

    if any(output_dir.iterdir()):
        logger.warning(f"⚠️  {output_dir} đã có nội dung → sẽ overwrite")

    dataset = Dataset.from_list(samples, features=features)
    dataset.save_to_disk(str(output_dir))
    logger.info(f"✅ Saved {len(samples)} samples → {output_dir}")


# ======================================================================
# LABEL COVERAGE REPORT
# ======================================================================

def print_label_report(
    counters:      Dict[str, Counter],
    label2id:      Dict[str, int],
    mode:          str,
) -> None:
    """
    In phân phối label sau khi process xong.

    Giúp phát hiện:
      - Entity type có 0 tokens → thiếu GT annotation
      - O ratio > 90% → spatial matching có vấn đề
      - O ratio < 20% → GT bbox quá rộng, bao phủ cả label prefix
    """
    total: Counter = Counter()
    for c in counters.values():
        total.update(c)

    if not total:
        return

    n_total = sum(total.values())

    # Aggregate theo entity type (bỏ B-/I- prefix)
    entity_totals: Dict[str, int] = defaultdict(int)
    for lbl, count in total.items():
        entity_totals["O" if lbl == "O" else lbl[2:]] += count

    o_count = entity_totals.get("O", 0)
    o_pct   = o_count / n_total * 100

    print("\n" + "=" * 70)
    print(f"📊  LABEL COVERAGE REPORT  —  MODE: {mode.upper()}")
    print("=" * 70)
    print(f"  Total tokens processed : {n_total:,}")
    print(f"  O (outside entities)   : {o_count:,}  ({o_pct:.1f}%)")
    print(f"  Labeled (B + I)        : {n_total - o_count:,}  ({100 - o_pct:.1f}%)")
    print("-" * 70)

    entity_groups = _MODE_CONFIGS[mode]["entity_groups"]

    for group_name, entities in entity_groups:
        print(f"\n  {group_name}")
        for ent in entities:
            count   = entity_totals.get(ent, 0)
            pct     = count / n_total * 100 if n_total else 0
            b_count = total.get(f"B-{ent}", 0)   # số entity instances
            i_count = total.get(f"I-{ent}", 0)
            # Warn nếu entity có trong schema mà 0 tokens
            in_schema = f"B-{ent}" in label2id
            if in_schema and count == 0:
                status = "⚠️ "
            elif not in_schema:
                status = "N/A"   # entity không trong schema mode này
                continue
            else:
                status = "   "
            print(
                f"  {status}  {ent:<22}"
                f"  {count:>8,} tokens  ({pct:5.2f}%)"
                f"  [{b_count} B / {i_count} I]"
            )

    print("=" * 70)

    # ── Health checks ─────────────────────────────────────────────────
    if o_pct > 90:
        logger.warning(
            f"⚠️  O tokens {o_pct:.1f}% — spatial matching có thể kém. "
            f"Kiểm tra OVERLAP_THRESHOLD trong auto_label_tokens.py."
        )
    elif o_pct < 20:
        logger.warning(
            f"⚠️  O tokens {o_pct:.1f}% — GT bbox quá rộng? "
            f"Có thể bao phủ cả label text thay vì chỉ value."
        )
    else:
        logger.info(f"✅ O ratio {o_pct:.1f}% — trong ngưỡng bình thường.")

    # Warn entity có 0 tokens
    all_expected = [e for _, ents in entity_groups for e in ents]
    missing = [e for e in all_expected if entity_totals.get(e, 0) == 0
               and f"B-{e}" in label2id]
    if missing:
        logger.warning(
            f"⚠️  {len(missing)} entity types có 0 tokens: {missing}\n"
            f"   Kiểm tra GT annotations có field này không."
        )


# ======================================================================
# MAIN PIPELINE
# ======================================================================

def prepare_dataset(
    mode:          str,
    labeled_dir:   Path,
    images_dir:    Path,
    output_dir:    Path,
    label2id_path: Path,
    model_name:    str   = "microsoft/layoutlmv3-base",
    train_ratio:   float = DEFAULT_TRAIN,
    val_ratio:     float = DEFAULT_VAL,
    seed:          int   = 42,
    skip_validate: bool  = False,
) -> None:

    cfg = _MODE_CONFIGS[mode]

    print("=" * 70)
    print(f"🚀  PREPARE LAYOUTLMV3 DATASET  —  MODE: {mode.upper()}")
    print("=" * 70)

    # ── Load & validate label2id ──────────────────────────────────────
    label2id   = load_label2id(label2id_path, mode)
    num_labels = len(label2id)

    # ── Load processor ────────────────────────────────────────────────
    print(f"\n📦 Loading processor: {model_name}")
    processor = LayoutLMv3Processor.from_pretrained(
        model_name,
        apply_ocr=False,
    )
    print("✅ Processor loaded")

    # ── Scan files ────────────────────────────────────────────────────
    labeled_files = sorted(f for f in labeled_dir.glob("*.json")
                           if not f.name.startswith("_"))

    if not labeled_files:
        logger.error(f"Không tìm thấy labeled token files trong '{labeled_dir}'")
        return

    print(f"\n📂 Labeled tokens  : {len(labeled_files)} files  ({labeled_dir})")
    print(f"📂 Images dir      : {images_dir}")
    print(f"📂 Output dir      : {output_dir}")
    print(f"🏷️  num_labels      : {num_labels}  (O={label2id['O']})")
    print(f"📐 max_seq_length  : {MAX_SEQ_LENGTH}")
    print(f"🔖 schema_version  : {cfg['schema_version']}")

    # ── Pre-flight schema validation ──────────────────────────────────
    if not skip_validate:
        is_valid = validate_schema_against_files(
            labeled_files, label2id, mode
        )
        if not is_valid:
            logger.error(
                f"❌ Schema validation FAILED — dừng để tránh corrupt dataset.\n"
                f"   Fix:\n"
                f"     1. Dùng đúng label2id_{mode}.json\n"
                f"     2. Chạy lại: auto_label_tokens.py --mode {mode}\n"
                f"     3. Hoặc thêm --skip_validate để bỏ qua (không khuyến nghị)"
            )
            return
    else:
        logger.warning("⚠️  Schema validation bị bỏ qua (--skip_validate).")

    # ── Split ─────────────────────────────────────────────────────────
    train_files, val_files, test_files = split_files(
        labeled_files, train_ratio, val_ratio, seed
    )
    test_ratio = max(0.0, round(1.0 - train_ratio - val_ratio, 4))
    print(
        f"\n📊 Split (seed={seed}): "
        f"train={len(train_files)} ({train_ratio:.0%}) | "
        f"val={len(val_files)} ({val_ratio:.0%}) | "
        f"test={len(test_files)} ({test_ratio:.0%})"
    )

    # ── Process & save ────────────────────────────────────────────────
    label_counters: Dict[str, Counter] = {}

    for split_name, files in [("train", train_files), ("val", val_files), ("test", test_files)]:
        if not files:
            logger.warning(f"{split_name}: 0 files, skip.")
            continue

        print(f"\n🔄 Processing {split_name} ({len(files)} invoices)...")
        samples, counter = build_split(
            files, processor, label2id, images_dir, split_name, mode
        )
        label_counters[split_name] = counter

        if samples:
            save_hf_dataset(samples, output_dir / split_name)

    # ── Label coverage report ─────────────────────────────────────────
    print_label_report(label_counters, label2id, mode)

    # ── Final summary ─────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print(f"✅  DATASET READY  —  {mode.upper()}")
    print("=" * 70)
    print(f"📂 Output: {output_dir}")
    print(f"""
Sử dụng trong training:

    from datasets import load_from_disk
    train_ds = load_from_disk("{output_dir}/train")
    val_ds   = load_from_disk("{output_dir}/val")

    from transformers import LayoutLMv3ForTokenClassification
    model = LayoutLMv3ForTokenClassification.from_pretrained(
        "microsoft/layoutlmv3-base",
        num_labels={num_labels},
    )
    # num_labels = {num_labels}  ({'header: 25 (v2)' if mode == 'header' else 'table: 25'})
""")
    print("=" * 70)


# ======================================================================
# CLI
# ======================================================================

def main():
    p = argparse.ArgumentParser(
        description=(
            "Chuẩn bị HuggingFace Dataset cho LayoutLMv3 — Approach B\n"
            "  --mode header : header entities\n"
            "  --mode table  : table entities\n"
            "  --mode footer : totals/payment entities"
        ),
        formatter_class=argparse.RawTextHelpFormatter,
    )
    p.add_argument(
        "--mode",
        required=True,
        choices=["header", "table", "footer"],
        help=(
            "--mode header : header entities\n"
            "--mode table  : table entities\n"
            "--mode footer : totals/payment entities"
        ),
    )
    p.add_argument("--labeled_dir",  default=None,
                   help="Default: output/labeled_tokens_{mode}")
    p.add_argument("--images_dir",   default="output/images")
    p.add_argument("--output_dir",   default=None,
                   help="Default: output/hf_dataset_{mode}")
    p.add_argument("--label2id",     default=None,
                   help="Default: label2id_{mode}.json")
    p.add_argument("--model_name",   default="microsoft/layoutlmv3-base")
    p.add_argument("--train_ratio",  type=float, default=DEFAULT_TRAIN)
    p.add_argument("--val_ratio",    type=float, default=DEFAULT_VAL)
    p.add_argument("--seed",         type=int,   default=42)
    p.add_argument(
        "--skip_validate",
        action="store_true",
        help="Bỏ qua pre-flight schema validation (không khuyến nghị)",
    )

    args = p.parse_args()

    if args.train_ratio + args.val_ratio > 1.0:
        p.error("train_ratio + val_ratio phải <= 1.0")

    mode = args.mode
    cfg  = _MODE_CONFIGS[mode]

    prepare_dataset(
        mode          = mode,
        labeled_dir   = Path(args.labeled_dir  or cfg["default_labeled_dir"]),
        images_dir    = Path(args.images_dir),
        output_dir    = Path(args.output_dir   or cfg["default_output_dir"]),
        label2id_path = Path(args.label2id     or cfg["default_label2id"]),
        model_name    = args.model_name,
        train_ratio   = args.train_ratio,
        val_ratio     = args.val_ratio,
        seed          = args.seed,
        skip_validate = args.skip_validate,
    )


if __name__ == "__main__":
    main()