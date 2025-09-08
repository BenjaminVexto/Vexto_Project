import logging
from bs4 import BeautifulSoup
import re
from typing import Dict, Any


def _cmp_script_detect(soup) -> str:
    try:
        for s in soup.find_all("script", src=True):
            src = (s.get("src") or "").lower()
            if re.search(r"cookieinformation|cookiebot|onetrust|osano|trustarc|cookieyes|quantcast|usercentrics|cookiepro|iubenda|consentmanager", src):
                v = "cookieinformation" if "cookieinformation" in src else \
                    "cookiebot" if "cookiebot" in src else \
                    "onetrust" if "onetrust" in src else \
                    "osano" if "osano" in src else \
                    "trustarc" if "trustarc" in src else \
                    "cookieyes" if "cookieyes" in src else "cmp"
                return v
    except Exception:
        pass
    return ""

from .schemas import PrivacyMetrics

log = logging.getLogger(__name__)

# Høj-konfidens (næsten altid bannere) vs. lav-konfidens (kan være false positives)
HIGH_CONFIDENCE_KEYWORDS = ['cookie', 'consent', 'gdpr', 'accept', 'samtykke', 'cmp']
LOW_CONFIDENCE_KEYWORDS = ['privacy', 'persondata']

# Til trust signals: Keywords for badges, certifikater, reviews etc.
TRUST_KEYWORDS = ['trustpilot', 'certified', 'secure', 'ssl', 'reviews', 'badge', 'guarantee', 'verified', 'award', 'partner']


def detect_cookie_banner(soup) -> Dict[str, Any]:
    result: Dict[str, Any] = {
        'cookie_banner_detected': False,
        'detection_method': 'none',
        'cookie_banner_confidence': 0.0,
        'personal_data_redacted': None,
    }
    try:
        if not soup:
            return result

        # 0) Vendor-script (høj konfidens)
        vendor = _cmp_script_detect(soup)
        if vendor:
            result['cookie_banner_detected'] = True
            result['detection_method'] = 'script'
            result['cookie_banner_confidence'] = 0.95
            return result

        # 1) Selectors/role-dialog (middel konfidens)
        selectors = [
            '[id*="cookie"]','[class*="cookie"]','[id*="consent"]','[class*="consent"]',
            '[data-testid*="cookie"]','[data-test*="cookie"]','[data-testid*="consent"]','[data-test*="consent"]',
        ]
        for sel in selectors:
            if soup.select_one(sel):
                result['cookie_banner_detected'] = True
                result['detection_method'] = 'selector'
                result['cookie_banner_confidence'] = 0.7
                return result

        for dlg in soup.select('[role="dialog"], [role="alertdialog"]'):
            t = (dlg.get_text(" ", strip=True) or "").lower()
            if any(x in t for x in ("cookie","cookies","samtykke","consent","gdpr")):
                result['cookie_banner_detected'] = True
                result['detection_method'] = 'selector'
                result['cookie_banner_confidence'] = 0.7
                return result

        # 2) Tekst-hints (lav konfidens)
        txt = soup.get_text(" ", strip=True).lower()
        if any(h in txt for h in ("cookie","cookies","samtykke","consent","gdpr")):
            result['cookie_banner_detected'] = True
            result['detection_method'] = 'text'
            result['cookie_banner_confidence'] = 0.4
            return result

        return result
    except Exception:
        return result


def detect_trust_signals(soup: BeautifulSoup) -> dict:
    """
    Returnér DICT med nøgle 'trust_signals_found' (liste af strings).
    Eksempler: ['trustpilot', 'partner', 'iso', 'ssl', 'betalingslogo', 'rating_schema', 'kunde_logo_vaeg'].
    Matcher analyzer/scorer-forventningen og undgår AttributeError.
    """
    signals: list[str] = []
    if not soup:
        return {"trust_signals_found": []}

    txt = soup.get_text(" ", strip=True).lower()

    # 1) Tekstlige mønstre
    text_markers = [
        "trustpilot", "anmeldelser", "kundeanmeldelser", "certificeret", "certified",
        "partner", "partnerprogram", "partner logo", "verified", "verificeret",
        "iso", "iso9001", "iso 9001", "iso14001", "iso 14001",
        "ssl", "secure", "sikker betaling", "betaling", "betalinger",
        "garanti", "warranty", "kvalitetssikring",
    ]
    if any(m in txt for m in text_markers):
        for m in text_markers:
            if m in txt:
                signals.append(m.split()[0])  # kort tag

    # 2) Klassiske badge-/widget-selectors
    selectors = [
        "[class*='trustpilot']",
        "[id*='trustpilot']",
        "[class*='verified']",
        "[class*='partner']",
        "[class*='badge']",
        "[class*='certificate']",
        "[class*='secure']",
        "img[alt*='trustpilot' i], img[src*='trustpilot' i]",
        "img[alt*='verified' i],  img[src*='verified' i]",
        "img[alt*='ssl' i],       img[src*='ssl' i]",
        "img[alt*='dankort' i],   img[src*='dankort' i]",
        "img[alt*='visa' i],      img[src*='visa' i]",
        "img[alt*='mastercard' i],img[src*='mastercard' i]",
        "img[alt*='mobilepay' i], img[src*='mobilepay' i]",
    ]
    for sel in selectors:
        if soup.select_one(sel):
            tag = sel.split("[", 1)[0] or "badge"
            signals.append("betalingslogo" if "img[" in sel else tag.strip() or "badge")

    # 3) Schema.org AggregateRating (rating fra data markup)
    for agg in soup.select('[itemtype*="AggregateRating" i], script[type="application/ld+json"]'):
        try:
            if agg.name == "script":
                import json as _json
                data = _json.loads(agg.get_text(strip=True))
                # simple scan
                if isinstance(data, dict):
                    blocks = [data]
                elif isinstance(data, list):
                    blocks = data
                else:
                    blocks = []
                for b in blocks:
                    if isinstance(b, dict) and ("aggregateRating" in b or b.get("@type") == "AggregateRating"):
                        signals.append("rating_schema")
                        break
            else:
                signals.append("rating_schema")
        except Exception:
            # Støj tolereres; schema-parsing er best-effort
            pass

    # 4) Kunde-logo-væg (mange logoer i grid/slider)
    possible_logo_blocks = soup.select("[class*='logo'] ul, [class*='logo'] .swiper, [class*='customer'] [class*='logo']")
    if possible_logo_blocks:
        signals.append("kunde_logo_vaeg")

    # Normalisér og deduplicér
    out = []
    seen = set()
    for s in signals:
        s = (s or "").strip().lower()
        if not s:
            continue
        if s not in seen:
            seen.add(s); out.append(s)

    return {"trust_signals_found": out}


# ---- VEXTO HOTFIX 2025-08-11: cookie/consent detection via scripts ----

_COOKIE_HINTS = (
    "cookieinformation",
    "cookiebot",
    "iubenda",
    "consentmanager",
    "truste"
)

def detect_cookie_solutions_from_html(html: str) -> dict:
    low = (html or "").lower()
    found = [k for k in _COOKIE_HINTS if k in low]
    return {"cookie_banner_detected": 1 if found else 0, "vendors": found}
