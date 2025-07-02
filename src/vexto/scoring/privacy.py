# src/vexto/scoring/privacy.py (NY)

import re

__all__ = ["strip_pii"]

def strip_pii(text: str) -> str:
    """
    A placeholder function to strip Personally Identifiable Information (PII) from text.
    This should be expanded with more robust rules.
    
    Example: Will be used on scraped Google reviews.
    """
    # Simpel regex til at fjerne e-mailadresser som et eksempel
    email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
    text = re.sub(email_pattern, '[EMAIL REDACTED]', text)
    
    # TODO: Tilføj regler for at fjerne navne, telefonnumre etc.
    
    return text