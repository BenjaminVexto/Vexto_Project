# src/vexto/scoring/trust_signal_fetcher.py

import logging
from bs4 import BeautifulSoup
from .schemas import ConversionMetrics

log = logging.getLogger(__name__)

# Nøgleord vi leder efter i links, billeders alt-tekst eller class/id navne
TRUST_KEYWORDS = [
    'trustpilot',
    'e-mærket',
    'emaerket', # Almindelig variation
    'anmeldelser',
    'google reviews',
    'certifikat',
    'certificate'
]

def find_trust_signals(soup: BeautifulSoup) -> ConversionMetrics:
    """
    Analyserer et BeautifulSoup-objekt for at finde troværdighedssignaler.
    """
    if not soup:
        return {'trust_signals_found': []}

    found_signals = set()
    
    try:
        # Tjek alle links
        for link in soup.find_all('a', href=True):
            for keyword in TRUST_KEYWORDS:
                if keyword in link['href'].lower():
                    found_signals.add(keyword)

        # Tjek alle billeder (alt-tekst og filnavn)
        for img in soup.find_all('img'):
            img_text = f"{img.get('alt', '')} {img.get('src', '')}".lower()
            for keyword in TRUST_KEYWORDS:
                if keyword in img_text:
                    found_signals.add(keyword)
        
        return {'trust_signals_found': sorted(list(found_signals))}

    except Exception as e:
        log.error(f"Fejl under detektion af trust signals: {e}", exc_info=True)
        return {'trust_signals_found': []}