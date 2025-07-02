# src/vexto/scoring/website_fetchers.py

from bs4 import BeautifulSoup
import logging
from typing import Optional
from urllib.parse import urlparse

from .schemas import BasicSEO, TechnicalSEO
from .http_client import AsyncHtmlClient

log = logging.getLogger(__name__)

def _parse_basic_seo_from_soup(soup: BeautifulSoup) -> BasicSEO:
    """
    Helper function that extracts Basic SEO data from a BeautifulSoup object.
    Image-related logic has been moved to image_fetchers.py.
    """
    # Hvis der slet ikke er noget soup-objekt, returner en tom struktur.
    if not soup:
        return {
            'h1': None, 'h1_count': None, 'h1_texts': None,
            'meta_description': None, 'meta_description_length': None,
            'title_text': None, 'title_length': None,
            'word_count': None,
            'canonical_url': None, 'canonical_error': "No HTML content to parse",
            'schema_markup_found': False,
        }
    
    try:
        # --- H1, Meta Description, Title ---
        h1_tags = soup.find_all('h1')
        h1_count = len(h1_tags)
        h1_texts = [h1.get_text(strip=True) for h1 in h1_tags] if h1_tags else []
        h1_first_text = h1_texts[0] if h1_texts else None

        meta_tag = soup.find('meta', attrs={'name': lambda x: x and x.lower() == 'description'})
        meta_description_text = meta_tag['content'].strip() if meta_tag and 'content' in meta_tag.attrs else None
        meta_description_length = len(meta_description_text) if meta_description_text else 0

        title_tag = soup.find('title')
        title_text = title_tag.get_text(separator=' ', strip=True) if title_tag else None
        title_length = len(title_text) if title_text else 0

        # --- Ordtælling ---
        word_count = len(soup.get_text(' ', strip=True).split())

        # --- Alt-tekst logik er FJERNET herfra ---

        # --- Canonical URL ---
        canonical_tags = soup.find_all('link', attrs={'rel': 'canonical'})
        canonical_url = None
        canonical_error = None
        if len(canonical_tags) == 1:
            canonical_url = canonical_tags[0].get('href')
        elif len(canonical_tags) > 1:
            canonical_error = "Multiple canonical tags found"
        else:
            canonical_error = "No canonical tag found"

        # -- Schema Markup ---
        json_ld_tag = soup.find('script', attrs={'type': 'application/ld+json'})
        microdata_tag = soup.find(attrs={'itemscope': True})
        schema_markup_found = bool(json_ld_tag or microdata_tag)

        seo_data: BasicSEO = {
            'h1': h1_first_text,
            'h1_count': h1_count,
            'h1_texts': h1_texts,
            'meta_description': meta_description_text,
            'meta_description_length': meta_description_length,
            'title_text': title_text,
            'title_length': title_length,
            'word_count': word_count,
            'canonical_url': canonical_url,
            'canonical_error': canonical_error,
            'schema_markup_found': schema_markup_found,
        }
        return seo_data

    except Exception as e:
        log.error(f"Error during basic SEO parsing: {e}", exc_info=True)
        return {
            'h1': None, 'h1_count': None, 'h1_texts': None,
            'meta_description': None, 'meta_description_length': None,
            'title_text': None, 'title_length': None,
            'word_count': None,
            'canonical_url': None, 'canonical_error': "Parsing failed with exception",
            'schema_markup_found': False,
        }

async def fetch_basic_seo_data(client: AsyncHtmlClient, url: str) -> BasicSEO:
    """
    Fetches the HTML for a URL and uses the helper function to parse basic SEO data.
    """
    soup = await client.get_soup(url)
    return _parse_basic_seo_from_soup(soup)

async def fetch_technical_seo_data(client: AsyncHtmlClient, url: str) -> TechnicalSEO:
    # Denne funktion er urørt, men inkluderet for fuldstændighed
    # ... (din eksisterende kode for denne funktion) ...
    pass