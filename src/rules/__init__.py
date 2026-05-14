"""
Extraction rules for Vietnamese invoice fields.

Rules combine NER labels, spatial relationships, and regex patterns.
"""

from .field_extractor import FieldExtractor

__all__ = ["FieldExtractor"]
