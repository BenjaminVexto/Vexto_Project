# src/vexto/scoring/image_fetchers.py
import asyncio
import logging
from urllib.parse import urljoin
from bs4 import BeautifulSoup

from .http_client import AsyncHtmlClient
from .schemas import BasicSEO # Vi genbruger BasicSEO til at returnere billeddata

log = logging.getLogger(__name__)

async def fetch_image_stats(client: AsyncHtmlClient, soup: BeautifulSoup, base_url: str) -> dict:
    """
    Finder alle billeder, analyserer alt-tekster og beregner samlet/gennemsnitlig filstørrelse.
    """
    if not soup:
        return {}

    images = soup.find_all('img')
    total_images = len(images)
    
    # 1. Alt-tekst analyse (logik flyttet hertil)
    images_with_alt_count = sum(1 for img in images if img.get('alt', '').strip())
    image_alt_pct = round((images_with_alt_count / total_images) * 100) if total_images > 0 else 100
    
    # 2. Filstørrelses-analyse (ny logik)
    async def get_size(src: str):
        try:
            absolute_url = urljoin(base_url, src)
            response = await client.head(absolute_url)
            if response and 'content-length' in response.headers:
                return int(response.headers['content-length'])
        except Exception:
            return 0
        return 0

    image_sources = [tag['src'] for tag in images if tag.get('src')]
    tasks = [get_size(src) for src in image_sources]
    sizes = await asyncio.gather(*tasks)
    
    total_size_bytes = sum(sizes)
    avg_size_kb = round((total_size_bytes / total_images) / 1024) if total_images > 0 else 0
    
    # Vi returnerer kun de felter, der relaterer til billeder
    return {
        'image_count': total_images,
        'image_alt_count': images_with_alt_count,
        'image_alt_pct': image_alt_pct,
        'avg_image_size_kb': avg_size_kb,
    }