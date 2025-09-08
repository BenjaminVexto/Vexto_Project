import logging
from bs4 import BeautifulSoup

from .schemas import ConversionMetrics  # For type hinting
from .http_client import AsyncHtmlClient

log = logging.getLogger(__name__)

CTA_KEYWORDS = ['kontakt', 'køb', 'book', 'tilmeld', 'abonner', 'start', 'prøv', 'få', 'læs mere', 'se mere']

async def fetch_cta_data(client: AsyncHtmlClient, soup: BeautifulSoup) -> dict:
    """
    Scraper for CTA'er (buttons/links med opfordringer).
    Beregn cta_score (0-100) baseret på antal og placering.
    Returnerer dict til ConversionMetrics.
    """
    if not soup:
        return {'cta_score': 0}

    try:
        cta_count = 0
        score = 0

        # Søg i <button> og <a>
        for tag in soup.find_all(['button', 'a']):
            text = tag.get_text().lower().strip()
            if any(kw in text for kw in CTA_KEYWORDS):
                cta_count += 1
                # Bonus for placering (f.eks. i header/nav)
                if tag.find_parent(['header', 'nav']) or 'cta' in tag.get('class', []) or 'button' in tag.get('class', []):
                    score += 20  # Højere vægt for fremtrædende
                else:
                    score += 10

        # Normaliser score (max 100)
        cta_score = min(score, 100)

        return {'cta_score': cta_score}

    except Exception as e:
        log.error(f"Fejl under CTA-scraping: {e}", exc_info=True)
        return {'cta_score': 0}