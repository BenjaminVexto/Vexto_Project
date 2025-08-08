import logging
from bs4 import BeautifulSoup
from .schemas import ConversionMetrics  # For type hinting
from .http_client import AsyncHtmlClient

log = logging.getLogger(__name__)

async def fetch_tracking_data(client: AsyncHtmlClient, soup: BeautifulSoup) -> dict:
    """
    Tjekker for GA4 og Meta Pixel ved at scrape <script> tags og src.
    Returnerer dict med bool-værdier til ConversionMetrics.
    """
    if not soup:
        return {'has_ga4': False, 'has_meta_pixel': False}

    try:
        # Tjek for GA4: Søg efter gtag.js eller analytics.google.com
        has_ga4 = any(
            'google-analytics.com/ga.js' in str(script) or 
            'analytics.google.com/g/collect' in str(script) or 
            'gtag.js' in str(script).lower()
            for script in soup.find_all('script')
        )

        # Tjek for Meta Pixel: Søg efter connect.facebook.net og pixel
        has_meta_pixel = any(
            'connect.facebook.net' in str(script) and 'pixel' in str(script).lower()
            for script in soup.find_all('script')
        )

        return {'has_ga4': has_ga4, 'has_meta_pixel': has_meta_pixel}

    except Exception as e:
        log.error(f"Fejl under tracking-scraping: {e}", exc_info=True)
        return {'has_ga4': False, 'has_meta_pixel': False}