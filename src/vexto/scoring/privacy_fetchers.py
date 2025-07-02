# src/vexto/scoring/privacy_fetchers.py
import logging
from bs4 import BeautifulSoup
from .schemas import PrivacyMetrics

log = logging.getLogger(__name__)

# Høj-konfidens (næsten altid bannere) vs. lav-konfidens (kan være false positives)
HIGH_CONFIDENCE_KEYWORDS = ['cookie', 'consent', 'gdpr', 'accept', 'samtykke', 'cmp']
LOW_CONFIDENCE_KEYWORDS = ['privacy', 'persondata']

def detect_cookie_banner(soup: BeautifulSoup) -> PrivacyMetrics:
    """
    Returnerer:
      - cookie_banner_detected: True/False/None
      - detection_method: Hvordan det blev fundet eller 'none'/'no_html'/'error'
    """
    if not soup:
        return {'cookie_banner_detected': None, 'detection_method': 'no_html'}
    try:
        # 1) ID/class-heuristik
        all_keywords = HIGH_CONFIDENCE_KEYWORDS + LOW_CONFIDENCE_KEYWORDS
        for kw in all_keywords:
            sel = f'[id*="{kw}" i], [class*="{kw}" i]'
            if soup.select_one(sel):
                return {'cookie_banner_detected': True, 'detection_method': f'selector:{kw}'}

        # 2) Tekst-heuristik
        text = soup.get_text(separator=' ').lower()

        # 2a) Høj-konfidens for tekst: hvis et af disse ord optræder → banner
        for kw in HIGH_CONFIDENCE_KEYWORDS:
            if kw in text:
                return {'cookie_banner_detected': True, 'detection_method': f'text:{kw}'}

        # 2b) Lav-konfidens for 'privacy': kun hvis ikke 'privacy policy' eller 'privatlivspolitik'
        if 'privacy' in text:
            if 'privacy policy' not in text and 'privatlivspolitik' not in text:
                return {'cookie_banner_detected': True, 'detection_method': 'text:privacy'}

        # Intet fundet
        return {'cookie_banner_detected': False, 'detection_method': 'none'}

    except Exception as e:
        log.error("Fejl under cookie-banner-detektion: %s", e, exc_info=True)
        return {'cookie_banner_detected': None, 'detection_method': 'error'}