# src/vexto/scoring/robots_sitemap_fetcher.py

import logging
import asyncio                              # ← tilføjet
from urllib.parse import urljoin
from typing import List, Set

from .http_client import AsyncHtmlClient

log = logging.getLogger(__name__)

# En udvidet liste af standard-stier at gætte på
FALLBACK_SITEMAP_PATHS = [
    "/sitemap.xml",
    "/sitemap_index.xml",
    "/sitemap.xml.gz",
    "/sitemaps.xml",
    "/wp-sitemap.xml",
]

async def _fetch_robots_txt(client: AsyncHtmlClient, base_url: str) -> str | None:
    """Henter indholdet af robots.txt-filen."""
    robots_url = urljoin(base_url, "/robots.txt")
    return await client.get_raw_html(robots_url)

def _parse_sitemaps_from_robots(robots_txt: str) -> Set[str]:
    """Udtrækker alle 'Sitemap:'-linjer fra en robots.txt-streng."""
    sitemaps: Set[str] = set()
    for line in robots_txt.splitlines():
        clean_line = line.strip().lower()
        if clean_line.startswith("sitemap:"):
            url = line.split(":", 1)[1].strip()
            if url:
                sitemaps.add(url)
    return sitemaps

async def discover_sitemap_locations(client: AsyncHtmlClient, base_url: str) -> Set[str]:
    """
    Returnerer et sæt af sitemap-URL'er fundet for et domæne.
    1. Læser robots.txt og finder alle 'Sitemap:'-direktiver.
    2. Hvis intet findes, falder den tilbage til at gætte på en liste af standard-stier.
    """
    found_sitemaps: Set[str] = set()

    # Trin 1: Hent og parse robots.txt
    robots_txt = await _fetch_robots_txt(client, base_url)
    if robots_txt:
        found_sitemaps = _parse_sitemaps_from_robots(robots_txt)
        if found_sitemaps:
            log.info("Sitemap(s) fundet via robots.txt for %s", base_url)
            return found_sitemaps
    
    # Trin 2: Fallback - gæt på standard-stier, hvis intet blev fundet
    log.info("Ingen sitemaps fundet i robots.txt for %s. Gætter på standard-stier.", base_url)
    fallback_urls = {urljoin(base_url, path) for path in FALLBACK_SITEMAP_PATHS}
    
    # Tjek hvilke af de gættede URLs der rent faktisk eksisterer
    coroutines = [client.head(url) for url in fallback_urls]               # ← ændret
    results    = await asyncio.gather(*coroutines)                         # ← ændret
    
    existing_sitemaps = {
        url for url, resp in zip(fallback_urls, results)                   # ← ændret
        if resp and resp.status_code == 200
    }
    
    if existing_sitemaps:
        log.info("Sitemap(s) fundet via fallback-gæt for %s", base_url)
        return existing_sitemaps

    return set()
