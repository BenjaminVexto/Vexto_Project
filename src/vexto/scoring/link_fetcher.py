# src/vexto/scoring/link_fetcher.py

import asyncio
import logging
from urllib.parse import urljoin, urlparse
from bs4 import BeautifulSoup
from .http_client import AsyncHtmlClient
from .schemas import TechnicalSEO  # Vi kan genbruge/udvide denne til resultatet

log = logging.getLogger(__name__)

async def check_links_on_page(client: AsyncHtmlClient, soup: BeautifulSoup, base_url: str) -> dict:
    """
    Finder alle interne links på en side og tjekker deres endelige HTTP-status
    med redirects og GET-fallback. Returnerer både procent, antal og liste.
    - Følger redirects (3xx).
    - Fallback til GET ved 401/403/405 eller når HEAD ikke giver mening.
    - Tæller 4xx og 5xx (samt netværksfejl) som brudte links.
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
        return {
            'broken_links_internal_pct': 0,
            'broken_links_pct': 0,
            'broken_links_count': 0,
            'broken_links_list': [],
            'links_checked': 0,
        }

    async def _resolve_status(url: str) -> int:
        """
        Returnerer endelig statuskode efter redirects.
        Fallback'er til GET hvis HEAD returnerer 401/403/405 eller en 3xx.
        """
        try:
            resp = await client.head(url, follow_redirects=True)
            if not resp:
                return 0
            sc = getattr(resp, "status_code", 0)
            final_url = getattr(resp, "url", url)

            if sc in (401, 403, 405) or (300 <= sc < 400):
                g = await client.get(final_url, follow_redirects=True)
                if g:
                    sc = getattr(g, "status_code", sc)
            return sc or 0
        except Exception:
            return 0  # netværks-/timeout-fejl: betragt som brudt

    async def _status_tuple(url: str):
        sc = await _resolve_status(url)
        return (url, sc)

    # Tjek status for hvert unikt internt link parallelt (begræns evt. med semaphore hvis nødvendigt)
    tasks = [_status_tuple(link) for link in internal_links]
    results = await asyncio.gather(*tasks)

    # Brudte links: 4xx/5xx eller 0 (fejl)
    broken = [(u, sc) for (u, sc) in results if sc == 0 or sc >= 400]
    broken_count = len(broken)
    links_checked = len(internal_links)
    broken_links_pct = round((broken_count / links_checked) * 100, 1)

    return {
        'broken_links_internal_pct': broken_links_pct,  # behold eksisterende nøgle for bagudkompatibilitet
        'broken_links_pct': broken_links_pct,
        'broken_links_count': broken_count,
        'broken_links_list': [u for (u, sc) in broken],
        'links_checked': links_checked,
    }