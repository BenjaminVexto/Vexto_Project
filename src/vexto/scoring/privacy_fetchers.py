import logging
from bs4 import BeautifulSoup
from .schemas import PrivacyMetrics

log = logging.getLogger(__name__)

# Høj-konfidens (næsten altid bannere) vs. lav-konfidens (kan være false positives)
HIGH_CONFIDENCE_KEYWORDS = ['cookie', 'consent', 'gdpr', 'accept', 'samtykke', 'cmp']
LOW_CONFIDENCE_KEYWORDS = ['privacy', 'persondata']

# Til trust signals: Keywords for badges, certifikater, reviews etc.
TRUST_KEYWORDS = ['trustpilot', 'certified', 'secure', 'ssl', 'reviews', 'badge', 'guarantee', 'verified', 'award', 'partner']

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

def detect_trust_signals(soup: BeautifulSoup) -> dict:
    """
    Scraper for troværdighedssignaler som badges, certifikater og reviews-widgets.
    Returnerer {'trust_signals_found': list[str]} til ConversionMetrics.
    """
    if not soup:
        return {'trust_signals_found': []}
    try:
        signals = set()
        # 1) ID/class-heuristik (genbrug fra banner-logik, men med trust keywords)
        for kw in TRUST_KEYWORDS:
            sel = f'[id*="{kw}" i], [class*="{kw}" i], [alt*="{kw}" i]'
            if soup.select_one(sel):
                signals.add(kw)

        # 2) Tekst-heuristik (høj-konfidens matches i tekst)
        text = soup.get_text(separator=' ').lower()
        for kw in TRUST_KEYWORDS:
            if kw in text:
                signals.add(kw)

        # 3) Specifikke checks (f.eks. Trustpilot-widget eller SSL-badge)
        if soup.find('script', src=lambda s: s and 'trustpilot' in s.lower()):
            signals.add('trustpilot_widget')
        if soup.find('img', alt=lambda a: a and 'ssl' in a.lower()):
            signals.add('ssl_badge')

        return {'trust_signals_found': list(signals)}

    except Exception as e:
        log.error("Fejl under trust-signal-detektion: %s", e, exc_info=True)
        return {'trust_signals_found': []}