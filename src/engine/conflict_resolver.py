# -*- coding: utf-8 -*-
from typing import List, Tuple

class ConflictResolver:
    """
    Resolves label conflicts between Header, Table, and Footer models.
    Priority: HEADER > TABLE > FOOTER.
    """
    
    HEADER_DOMINANT = {
        "INVOICE_NUMBER", "INVOICE_DATE", "INVOICE_SYMBOL", 
        "INVOICE_TYPE", "SELLER_NAME", "BUYER_NAME",
        "SELLER_TAX_CODE", "BUYER_TAX_CODE", "SELLER_ADDRESS", "BUYER_ADDRESS"
    }
    
    TABLE_DOMINANT = {
        "ITEM_NAME", "ITEM_QUANTITY", "ITEM_UNIT_PRICE", "ITEM_UNIT", "ITEM_TOTAL_PRICE"
    }
    
    FOOTER_DOMINANT = {
        "SUBTOTAL", "VAT_AMOUNT", "GRAND_TOTAL", "VAT_RATE"
    }

    def resolve(
        self,
        words: List[str],
        header_labels: List[str],
        table_labels: List[str],
        footer_labels: List[str]
    ) -> List[str]:
        """
        Applies priority rules to select the best label for each token.
        """
        resolved_labels = []
        
        for h, t, f in zip(header_labels, table_labels, footer_labels):
            # Strip BIO prefix for mapping
            h_entity = h.split("-")[-1] if "-" in h else h
            t_entity = t.split("-")[-1] if "-" in t else t
            f_entity = f.split("-")[-1] if "-" in f else f

            # Priority 1: Header
            if h != "O" and h_entity in self.HEADER_DOMINANT:
                resolved_labels.append(h)
                continue
            
            # Priority 2: Table
            if t != "O" and t_entity in self.TABLE_DOMINANT:
                resolved_labels.append(t)
                continue
                
            # Priority 3: Footer
            if f != "O" and f_entity in self.FOOTER_DOMINANT:
                resolved_labels.append(f)
                continue
            
            # Default to Header if not in dominant lists but still predicted
            if h != "O":
                resolved_labels.append(h)
            elif t != "O":
                resolved_labels.append(t)
            elif f != "O":
                resolved_labels.append(f)
            else:
                resolved_labels.append("O")
                
        return resolved_labels
