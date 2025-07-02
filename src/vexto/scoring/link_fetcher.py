# src/vexto/scoring/link_fetcher.py

import asyncio
import logging
from urllib.parse import urljoin, urlparse
from bs4 import BeautifulSoup

from .http_client import AsyncHtmlClient
from .schemas import TechnicalSEO # Vi kan genbruge/udvide denne til resultatet

log = logging.getLogger(__name__)

async def check_links_on_page(client: AsyncHtmlClient, soup: BeautifulSoup, base_url: str) -> dict:
    """
    Finder alle links på en side og tjekker deres HTTP-status for at finde 'broken links'.
    """
    if not soup:
        return {'broken_links_internal_pct': None}

    links = soup.find_all('a', href=True)
    internal_links = set()
    base_domain = urlparse(base_url).netloc

    # Find og normaliser alle unikke, interne links
    for link in links:
        href = link['href']
        if not href or href.startswith('#') or href.startswith('mailto:') or href.startswith('tel:'):
            continue
        
        absolute_url = urljoin(base_url, href)
        link_domain = urlparse(absolute_url).netloc

        if link_domain == base_domain:
            internal_links.add(absolute_url)

    if not internal_links:
        return {'broken_links_internal_pct': 0}

    # Tjek status for hvert unikt internt link parallelt
    async def get_status(url: str):
        try:
            response = await client.head(url)
            return response.status_code if response else 999 # Returner en fejl-kode, hvis intet svar
        except Exception:
            return 999 # Returner en fejl-kode ved exception

    tasks = [get_status(link) for link in internal_links]
    statuses = await asyncio.gather(*tasks)

    # Tæl antallet af "døde" links (statuskode 400-499)
    broken_count = sum(1 for status in statuses if 400 <= status < 500)
    
    broken_links_pct = round((broken_count / len(internal_links)) * 100)
    
    return {'broken_links_internal_pct': broken_links_pct}