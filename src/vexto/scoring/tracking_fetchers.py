# src/vexto/scoring/tracking_fetchers.py
import logging
from bs4 import BeautifulSoup
from .schemas import ConversionMetrics

log = logging.getLogger(__name__)

# Unikke tekststrenge, der indikerer tilstedeværelsen af de respektive scripts
GA4_IDENTIFIER = 'googletagmanager.com/gtag/js?id=G-'
META_PIXEL_IDENTIFIER = 'connect.facebook.net/en_US/fbevents.js'

def detect_tracking_scripts(soup: BeautifulSoup) -> ConversionMetrics:
    """
    Analyserer et BeautifulSoup-objekt for at detektere GA4 og Meta Pixel scripts.
    """
    if not soup:
        return {'has_ga4': None, 'has_meta_pixel': None}

    try:
        # Konverter hele HTML-suppen til en enkelt tekststreng for nem søgning
        html_text = str(soup).lower()

        has_ga4 = GA4_IDENTIFIER.lower() in html_text
        has_meta_pixel = META_PIXEL_IDENTIFIER.lower() in html_text

        tracking_data: ConversionMetrics = {
            'has_ga4': has_ga4,
            'has_meta_pixel': has_meta_pixel,
        }
        return tracking_data

    except Exception as e:
        log.error(f"Fejl under detektion af tracking scripts: {e}", exc_info=True)
        return {'has_ga4': None, 'has_meta_pixel': None}