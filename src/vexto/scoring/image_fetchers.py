# src/vexto/scoring/image_fetchers.py
import asyncio
import logging
from urllib.parse import urljoin, urlparse
from bs4 import BeautifulSoup

from .http_client import AsyncHtmlClient
from .schemas import BasicSEO  # Vi genbruger BasicSEO til at returnere billeddata

log = logging.getLogger(__name__)

async def fetch_image_stats(client: AsyncHtmlClient, soup: BeautifulSoup, base_url: str) -> dict:
    """
    Finder alle billeder, analyserer alt-tekster og beregner samlet/gennemsnitlig filstørrelse.
    """
    if not soup:
        return {
            'image_count': 0,
            'image_alt_count': 0,
            'image_alt_pct': 0,
            'avg_image_size_kb': 0
        }

    images = soup.find_all('img')
    total_images = len(images)
    
    # 1. Alt-tekst analyse
    images_with_alt_count = sum(1 for img in images if img.get('alt', '').strip())
    image_alt_pct = round((images_with_alt_count / total_images) * 100) if total_images > 0 else 100
    
    # 2. Filstørrelses-analyse
    async def get_size(src: str):
        try:
            absolute_url = urljoin(base_url, src)
            parsed = urlparse(absolute_url)
            if parsed.scheme not in ('http', 'https'):
                log.debug(f"Skipping non-HTTP image URL: {absolute_url}")
                return 0
            response = await client.head(absolute_url, timeout=5, follow_redirects=True)
            if response and response.status_code == 405:  # Fallback to GET for servers that block HEAD
                response = await client.httpx_get(absolute_url, timeout=5, follow_redirects=True)
            if response and 'content-length' in response.headers:
                return int(response.headers['content-length'])
        except Exception as e:
            log.debug(f"Failed to fetch image size for {absolute_url}: {e}")
            return 0
        return 0

    image_sources = [tag['src'] for tag in images if tag.get('src')]
    tasks = [get_size(src) for src in image_sources]
    sizes = await asyncio.gather(*tasks)
    
    total_size_bytes = sum(sizes)
    avg_size_kb = round((total_size_bytes / total_images) / 1024) if total_images > 0 else 0
    
    # Returnerer felter relateret til billeder
    return {
        'image_count': total_images,
        'image_alt_count': images_with_alt_count,
        'image_alt_pct': image_alt_pct,
        'avg_image_size_kb': avg_size_kb,
    }