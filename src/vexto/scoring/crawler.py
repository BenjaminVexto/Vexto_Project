# src/vexto/scoring/crawler.py
import asyncio
import logging
from urllib.parse import urljoin, urlparse
from typing import Set

from .http_client import AsyncHtmlClient
# schema-import er ikke strengt nødvendig her, men god praksis for klarhed
from .schemas import TechnicalSEO 

log = logging.getLogger(__name__)

async def crawl_site_for_links(client: AsyncHtmlClient, start_url: str, max_pages: int = 10) -> dict:
    """
    Crawls a website starting from a given URL to find all unique internal links
    and check their status, but in a more controlled manner.
    """
    base_domain = urlparse(start_url).netloc
    urls_to_visit = {start_url}
    visited_urls = set()
    all_found_links = set()
    
    # Loop so long as there are pages to visit and we haven't hit our limit
    while urls_to_visit and len(visited_urls) < max_pages:
        current_url = urls_to_visit.pop()
        if current_url in visited_urls:
            continue

        visited_urls.add(current_url)
        log.info(f"Crawling: {current_url} (Visited: {len(visited_urls)}/{max_pages})")

        soup = await client.get_soup(current_url)
        if not soup:
            continue

        # Find all links on the current page
        links_on_page = soup.find_all('a', href=True)
        for link in links_on_page:
            href = link.get('href')
            if not href or href.startswith(('#', 'mailto:', 'tel:')):
                continue

            absolute_url = urljoin(current_url, href)
            all_found_links.add(absolute_url) # Add all links for status check later

            # If the link is on the same domain, add it to the list to crawl
            if urlparse(absolute_url).netloc == base_domain:
                if absolute_url not in visited_urls:
                    urls_to_visit.add(absolute_url)

    # --- OPDATERET: Mere høflig link-checking ---

    # 1. Tilføj en Semaphore til at begrænse samtidige kald
    link_sema = asyncio.Semaphore(15) # Tjekker max 15 links ad gangen

    async def get_status(url: str):
        async with link_sema: # Brug semaphoren her
            try:
                # Tilføj en kortere, specifik timeout for link-tjek
                response = await client.head(url, timeout=10)
                return url, response.status_code if response else 999
            except Exception:
                return url, 999

    log.info(f"Fandt {len(all_found_links)} unikke links. Tjekker nu status for dem alle...")

    # 2. Tilføj en grænse på det samlede antal links, der skal tjekkes
    links_to_check = list(all_found_links)[:200]
    if len(all_found_links) > len(links_to_check):
        log.info(f"Begrænser tjek til de første {len(links_to_check)} links for at undgå overbelastning.")

    tasks = [get_status(link) for link in links_to_check]
    results = await asyncio.gather(*tasks)

    broken_links = {url: status for url, status in results if 400 <= status < 500}
    
    # Beregn procentdel baseret på de tjekkede links, ikke alle fundne
    total_checked = len(links_to_check)
    broken_links_pct = round((len(broken_links) / total_checked) * 100) if total_checked > 0 else 0

    return {
        'total_pages_crawled': len(visited_urls),
        'total_links_found': len(all_found_links),
        'broken_links_count': len(broken_links),
        'broken_links_pct': broken_links_pct, # Tilføjet for KPI #16
        'broken_links_list': broken_links,
    }