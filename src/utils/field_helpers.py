# -*- coding: utf-8 -*-
"""
Common field helpers used across the web app and inference pipeline.

These helpers normalize the mixed field formats produced by different
models and post-processing stages.
"""

from __future__ import annotations

from typing import Any, Dict, Tuple
import ast

# Confidence sentinels
CONFIDENCE_NOT_FOUND = 0.0
# Distinct sentinel for "not computed yet" (kept negative to preserve intent).
CONFIDENCE_NOT_COMPUTED = -1.0


def _coerce_conf(conf: Any) -> float:
    """Coerce confidence to float while preserving negative sentinel values."""
    try:
        if conf is None:
            return 0.0
        conf_f = float(conf)
    except Exception:
        return 0.0

    # Preserve negative sentinel values (e.g., -1.0 for not computed)
    if conf_f < 0:
        return conf_f

    # Clamp normal confidences into [0, 1]
    if conf_f > 1.0:
        conf_f = 1.0
    return round(conf_f, 4)


def make_field(value: Any, confidence: Any) -> Dict[str, Any]:
    """Return canonical field dict: {value, confidence}."""
    return {
        "value": value,
        "confidence": _coerce_conf(confidence),
    }


def make_field_found(value: Any, confidence: Any = 0.95) -> Dict[str, Any]:
    """Return a field marked as found."""
    return make_field(value, confidence)


def make_field_not_found(default: Any = None) -> Dict[str, Any]:
    """Return a field explicitly marked as not found."""
    return make_field(default, CONFIDENCE_NOT_FOUND)


def safe_unwrap(field_data: Any) -> Tuple[Any, float]:
    """
    Extract (value, confidence) from different field formats.

    Supported:
      - {"value": ..., "confidence": ...}
      - {"value": "{'value': ..., 'confidence': ...}", "confidence": ...}
      - (value, confidence) or [value, confidence]
      - raw scalar -> (scalar, 0.0)
      - None -> (None, 0.0)
    """
    if field_data is None:
        return None, 0.0

    # Tuple/list form
    if isinstance(field_data, (tuple, list)) and len(field_data) >= 2:
        return field_data[0], _coerce_conf(field_data[1])

    # Dict form
    if isinstance(field_data, dict):
        if "value" in field_data:
            val = field_data.get("value")
            conf = _coerce_conf(field_data.get("confidence", 0.0))

            # Handle double-wrapped stringified dict
            if isinstance(val, str) and val.strip().startswith("{"):
                try:
                    inner = ast.literal_eval(val)
                    if isinstance(inner, dict) and "value" in inner:
                        return inner.get("value"), _coerce_conf(inner.get("confidence", conf))
                except Exception:
                    pass

            return val, conf

        # Fallback: if dict but no "value", return as-is with zero conf
        return field_data, 0.0

    # Raw scalar
    return field_data, 0.0

