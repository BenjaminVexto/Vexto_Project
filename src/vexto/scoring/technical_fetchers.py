# src/vexto/scoring/technical_fetchers.py

import logging
from urllib.parse import urljoin, urlparse, urlunparse
from datetime import datetime, timezone
from bs4 import BeautifulSoup

# Sørg for at importere de korrekte typer
from .schemas import TechnicalSEO
from .http_client import AsyncHtmlClient
from .robots_sitemap_fetcher import discover_sitemap_locations

log = logging.getLogger(__name__)

async def fetch_technical_seo_data(client: AsyncHtmlClient, url: str) -> TechnicalSEO:
    
    # 1. Start med at lave et enkelt, effektivt HEAD-kald.
    head_resp = await client.head(url, follow_redirects=True)
    
    # Hvis det indledende kald fejler, kan vi ikke fortsætte sikkert.
    if not head_resp:
        log.warning(f"Indledende HEAD request for {url} fejlede. Returnerer tomme tekniske data.")
        # Returnerer en tom, men valid, dictionary, så programmet ikke crasher.
        return {
            "status_code": None, "is_https": url.startswith("https://"), "robots_txt_found": False,
            "sitemap_xml_found": False, "sitemap_is_fresh": False, "sitemap_locations": [],
            "response_time_ms": None, "total_pages_crawled": None, "total_links_found": None,
            "broken_links_count": None, "broken_links_pct": None, "broken_links_list": None,
            "schema_markup_found": False, "canonical_url": None,
        }

    # 2. Udtræk al information fra det succesfulde kald.
    final_url = str(head_resp.url)
    parsed_url = urlparse(final_url)
    base_url = urlunparse((parsed_url.scheme, parsed_url.netloc, '', '', '', ''))

    result: TechnicalSEO = {
        "status_code": head_resp.status_code,
        "is_https": parsed_url.scheme == "https",
        "response_time_ms": getattr(head_resp, 'elapsed_time_ms', None),
        "robots_txt_found": False,
        "sitemap_xml_found": False,
        "sitemap_is_fresh": False,
        "sitemap_locations": [],
        "total_pages_crawled": None, # Udfyldes af crawler-modulet
        "total_links_found": None,   # Udfyldes af crawler-modulet
        "broken_links_count": None,  # Udfyldes af crawler-modulet
        "broken_links_pct": None,    # Udfyldes af crawler-modulet
        "broken_links_list": None,   # Udfyldes af crawler-modulet
        "schema_markup_found": False, # Udfyldes af basic_seo_fetcher
        "canonical_url": None,     # Udfyldes af basic_seo_fetcher
    }

    # 3. Fortsæt med resten af de tekniske tjek.
    robots_resp = await client.head(urljoin(base_url, "/robots.txt"))
    result["robots_txt_found"] = robots_resp is not None and robots_resp.status_code == 200

    sitemap_locs = await discover_sitemap_locations(client, base_url)
    if sitemap_locs:
        result["sitemap_xml_found"] = True
        result["sitemap_locations"] = sorted(list(sitemap_locs))

        latest_date_found = None
        sitemaps_to_check = set(result["sitemap_locations"])
        processed_sitemaps = set()
        
        while sitemaps_to_check:
            sitemap_url = sitemaps_to_check.pop()
            if sitemap_url in processed_sitemaps: continue
            processed_sitemaps.add(sitemap_url)

            # --- Fejlhåndtering er nu KUN omkring den del, der kan fejle ---
            try:
                sitemap_content = await client.get_raw_html(sitemap_url)
                if not sitemap_content: continue
                
                soup = BeautifulSoup(sitemap_content, features="xml")
                
                if soup.find('sitemapindex'):
                    for sitemap_tag in soup.find_all('sitemap'):
                        loc_tag = sitemap_tag.find('loc')
                        if loc_tag and loc_tag.string:
                            sitemaps_to_check.add(loc_tag.string.strip())
                    continue

                for url_tag in soup.find_all('url'):
                    lastmod_tag = url_tag.find('lastmod')
                    if lastmod_tag and lastmod_tag.string:
                        date_str = lastmod_tag.string.strip()
                        current_date = datetime.fromisoformat(date_str.split('T')[0]).date()
                        if latest_date_found is None or current_date > latest_date_found:
                            latest_date_found = current_date
            
            except Exception as e:
                log.warning(f"Kunne ikke hente/parse sitemap {sitemap_url}: {e}")

        if latest_date_found:
            days_diff = (datetime.now(timezone.utc).date() - latest_date_found).days
            if days_diff <= 90:
                result["sitemap_is_fresh"] = True
            
    return result