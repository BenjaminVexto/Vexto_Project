# src/vexto/scoring/technical_fetchers.py

import logging
from urllib.parse import urlparse, urlunparse, urljoin   # ← tilføjet urljoin

from .schemas import TechnicalSEO
from .http_client import AsyncHtmlClient
from .robots_sitemap_fetcher import discover_sitemap_locations

log = logging.getLogger(__name__)

async def fetch_technical_seo_data(client: AsyncHtmlClient, url: str) -> TechnicalSEO:
    """
    Fetches simple Technical SEO data points using the new intelligent sitemap discoverer.
    """
    # -- KIRURGISK INDGREB: Bruger den nye 'resolve_final_url' for at finde den korrekte base-URL --
    final_url = await client.resolve_final_url(url)
    parsed_url = urlparse(final_url)
    base_url = urlunparse((parsed_url.scheme, parsed_url.netloc, '', '', '', ''))

    result: TechnicalSEO = {"is_https": parsed_url.scheme == "https"}

    # HEAD-request for status-kode
    head_resp = await client.head(final_url)
    if head_resp:
        result["status_code"] = head_resp.status_code

    # Tjek for robots.txt
    robots_resp = await client.head(urljoin(base_url, "/robots.txt"))
    result["robots_txt_found"] = robots_resp is not None and robots_resp.status_code == 200

    # -- KIRURGISK INDGREB START --
    # Erstatter den gamle, simple sitemap-logik med et kald til den nye, intelligente finder
    sitemap_locs = await discover_sitemap_locations(client, base_url)
    result["sitemap_xml_found"] = bool(sitemap_locs)
    result["sitemap_locations"] = sorted(list(sitemap_locs))  # Gemmer en sorteret liste
    # -- KIRURGISK INDGREB SLUT --
            
    return result
