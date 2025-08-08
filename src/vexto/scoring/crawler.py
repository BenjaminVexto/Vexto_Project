# src/vexto/scoring/crawler.py
import asyncio
import logging
from urllib.parse import urljoin, urlparse
from typing import Set, Optional
from bs4 import BeautifulSoup

from .http_client import AsyncHtmlClient

log = logging.getLogger(__name__)

# Tilføj denne funktion for at undgå NameError
def log_metric_status(metric, value, status):
    log.info(f"[{status}] {metric}: {value}")

async def crawl_site_for_links(
    client: AsyncHtmlClient,
    start_url: str,
    max_pages: int = 50,
    root_soup: Optional[BeautifulSoup] = None,
) -> dict:
    parsed_start = urlparse(start_url)
    base_domain = parsed_start.netloc
    urls_to_visit = {start_url}
    visited_urls = set()
    all_found_links: Set[str] = set()
    internal_links: Set[str] = set()
    exclude_extensions = {'.pdf', '.jpg', '.jpeg', '.png', '.gif', '.xml'}

    if root_soup:
        footer = root_soup.find('footer') or root_soup.find(id=lambda x: x and 'footer' in x.lower()) or root_soup.find(class_=lambda x: x and 'footer' in x.lower())
        if footer:
            for a in footer.find_all('a', href=True):
                href = a['href']
                if href.startswith('/'):
                    absolute_url = urljoin(start_url, href)
                    if not any(absolute_url.lower().endswith(ext) for ext in exclude_extensions):
                        urls_to_visit.add(absolute_url)

    while urls_to_visit and len(visited_urls) < max_pages:
        current_url = urls_to_visit.pop()
        if current_url in visited_urls or any(current_url.lower().endswith(ext) for ext in exclude_extensions):
            continue
        visited_urls.add(current_url)

        log.info(f"Crawling: {current_url} ({len(visited_urls)}/{max_pages})")

        if current_url == start_url and root_soup is not None:
            soup = root_soup
        else:
            try:
                resp = await client.httpx_get(current_url, follow_redirects=True)
                soup = BeautifulSoup(resp.text, "lxml") if resp and resp.status_code < 400 else None
            except Exception as e:
                log.debug(f"httpx_get failed for {current_url}: {e}")
                continue

        if not soup:
            continue

        for link in soup.find_all('a', href=True):
            href = link.get('href')
            if not href or href.startswith(('#', 'mailto:', 'tel:', 'javascript:', 'data:', 'callto:')):
                continue
            absolute_url = urljoin(current_url, href)
            all_found_links.add(absolute_url)
            if urlparse(absolute_url).netloc == base_domain and not any(absolute_url.lower().endswith(ext) for ext in exclude_extensions):
                internal_links.add(absolute_url)
                if absolute_url not in visited_urls:
                    urls_to_visit.add(absolute_url)

    internal_link_score = round((len(internal_links) / len(all_found_links) * 100) if all_found_links else 0)

    link_sema = asyncio.Semaphore(15)
    async def get_status(url: str):
        async with link_sema:
            parsed = urlparse(url)
            if parsed.scheme not in ('http', 'https'):
                log.debug(f"Skipping non-HTTP URL: {url}")
                return url, 999
            try:
                response = await client.head(url, timeout=10, follow_redirects=True)
                status = response.status_code if response else 999
                if status == 405:
                    response = await client.httpx_get(url, timeout=10, follow_redirects=True)
                    status = response.status_code if response else 999
                return url, status
            except Exception as e:
                log.warning(f"HEAD request fejlede for {url}: {e}")
                return url, 999

    log.info(f"Found {len(all_found_links)} unique links (internal: {len(internal_links)}). Checking status...")
    links_to_check = [l for l in all_found_links if urlparse(l).scheme in ('http', 'https')][:200]
    if len(all_found_links) > len(links_to_check):
        log.info(f"Limiting check to {len(links_to_check)} links (of {len(all_found_links)}).")

    results = await asyncio.gather(*(get_status(l) for l in links_to_check))
    broken_links = {url: status for url, status in results if 400 <= status < 500}
    broken_links_pct = round((len(broken_links) / len(links_to_check)) * 100) if links_to_check else 0

    log_metric_status("total_pages_crawled", len(visited_urls), "ok")
    log_metric_status("total_links_found", len(all_found_links), "ok")
    log_metric_status("broken_links_count", len(broken_links), "ok")
    log_metric_status("internal_link_score", internal_link_score, "ok")

    return {
        'total_pages_crawled': len(visited_urls),
        'total_links_found': len(all_found_links),
        'broken_links_count': len(broken_links),
        'broken_links_pct': broken_links_pct,
        'broken_links_list': broken_links,
        'internal_link_score': internal_link_score,
        'visited_urls': visited_urls
    }
