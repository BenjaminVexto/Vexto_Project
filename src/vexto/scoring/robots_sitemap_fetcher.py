# src/vexto/scoring/robots_sitemap_fetcher.py

import logging
import asyncio
from urllib.parse import urljoin
from typing import Set
from bs4 import BeautifulSoup

# src/vexto/scoring/robots_sitemap_fetcher.py


from .http_client import AsyncHtmlClient

log = logging.getLogger(__name__)

FALLBACK_SITEMAP_PATHS = [
    "/sitemap.xml",
    "/sitemap_index.xml",
    "/sitemap.xml.gz",
    "/sitemaps.xml",
    "/wp-sitemap.xml",
]


async def _fetch_robots_txt(client: AsyncHtmlClient, base_url: str) -> str | None:
    """Henter indholdet af robots.txt og renser det for eventuel HTML-indpakning."""
    robots_url = urljoin(base_url, "/robots.txt")
    content = await client.get_raw_html(robots_url)

    if content and content.strip().lower().startswith('<html'):
        log.info("HTML detekteret i robots.txt. Forsøger at udtrække ren tekst.")
        soup = BeautifulSoup(content, 'lxml')
        pre_tag = soup.find('pre')
        if pre_tag:
            return pre_tag.get_text()
        return soup.body.get_text() if soup.body else content

    return content


def _parse_sitemaps_from_robots(robots_txt: str) -> Set[str]:
    """Udtrækker alle 'Sitemap:'-linjer fra en robots.txt-streng (case-insensitiv)."""
    sitemaps: Set[str] = set()
    key = "sitemap:"
    for line in robots_txt.splitlines():
        clean_line = line.strip()
        if clean_line.lower().startswith(key):
            url = clean_line[len(key):].strip()
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
            log.info("Sitemap(s) fundet via robots.txt for %s: %s", base_url, found_sitemaps)
            return found_sitemaps

    # Trin 2: Fallback
    log.info("Ingen sitemaps fundet i robots.txt for %s. Gætter på standard-stier.", base_url)
    fallback_urls = {urljoin(base_url, path) for path in FALLBACK_SITEMAP_PATHS}

    coroutines = [client.head(url) for url in fallback_urls]
    results = await asyncio.gather(*coroutines)

    existing_sitemaps = {
        url for url, resp in zip(fallback_urls, results)
        if resp and resp.status_code == 200
    }

    if existing_sitemaps:
        log.info("Sitemap(s) fundet via fallback-gæt for %s: %s", base_url, existing_sitemaps)
        return existing_sitemaps

    return set()