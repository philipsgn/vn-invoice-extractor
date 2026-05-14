# -*- coding: utf-8 -*-
from typing import List, Dict, Any, Tuple
import numpy as np

class TableRowRebuilder:
    """
    Ultimate Table Row Rebuilder for Invoice Extraction.
    Uses Y-center clustering to group tokens into rows and implements 
    multiline merging for long service descriptions.
    """
    
    def __init__(self, proximity_threshold_ratio: float = 0.8):
        self.ratio = proximity_threshold_ratio

    def rebuild_rows(self, tokens: List[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:
        """
        Groups tokens into horizontal rows based on vertical proximity.
        """
        if not tokens:
            return []

        # 1. Compute Y-center for each token
        for t in tokens:
            bbox = t["bbox"]
            t["y_center"] = (bbox[1] + bbox[3]) / 2
            t["height"] = bbox[3] - bbox[1]

        # 2. Dynamic threshold based on median token height
        heights = [t["height"] for t in tokens]
        median_h = np.median(heights)
        threshold = self.ratio * median_h

        # 3. Cluster tokens using vertical proximity
        # Sort by Y-center first
        sorted_tokens = sorted(tokens, key=lambda x: x["y_center"])
        
        rows: List[List[Dict[str, Any]]] = []
        if not sorted_tokens:
            return []

        current_row = [sorted_tokens[0]]
        for i in range(1, len(sorted_tokens)):
            curr = sorted_tokens[i]
            prev = sorted_tokens[i-1]
            
            if abs(curr["y_center"] - prev["y_center"]) <= threshold:
                current_row.append(curr)
            else:
                rows.append(current_row)
                current_row = [curr]
        rows.append(current_row)

        # 4. Sort tokens inside rows by X coordinate
        for row in rows:
            row.sort(key=lambda x: x["bbox"][0])

        return rows

    def merge_multiline_items(self, raw_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Merge adjacent rows that follow the (Description-only) -> (Numeric-data) pattern.
        """
        if not raw_rows:
            return []

        merged_rows = []
        i = 0
        while i < len(raw_rows):
            curr = raw_rows[i]
            
            # Numeric fields check
            has_numeric = any(str(curr.get(k, "")).strip() for k in ["quantity", "unit_price", "total"])
            
            # If it's just text and there's a row below it
            if not has_numeric and i + 1 < len(raw_rows):
                nxt = raw_rows[i+1]
                has_next_numeric = any(str(nxt.get(k, "")).strip() for k in ["quantity", "unit_price", "total"])
                
                # Heuristic: If current is likely an anchor/description and next has values
                if has_next_numeric:
                    # Merge name
                    nxt["name"] = f"{curr.get('name', '').strip()} {nxt.get('name', '').strip()}".strip()
                    # Also merge confidence (take mean or max)
                    i += 1
                    continue
            
            merged_rows.append(curr)
            i += 1
            
        return merged_rows
