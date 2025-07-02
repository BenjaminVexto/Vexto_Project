# src/vexto/scoring/analyzer.py

import asyncio
import logging
from typing import List, Optional, Dict, Any

from .http_client import AsyncHtmlClient
from .schemas import (
    UrlAnalysisData, BasicSEO, TechnicalSEO, PerformanceMetrics, AuthorityMetrics,
    ContentMetrics, SocialAndReputationMetrics, ConversionMetrics, PrivacyMetrics,
    SecurityMetrics
)
from .website_fetchers import _parse_basic_seo_from_soup
from .technical_fetchers import fetch_technical_seo_data
from .performance_fetcher import get_performance, calculate_js_size
from .authority_fetcher import get_authority
from .security_fetchers import fetch_security_headers
from .privacy_fetchers import detect_cookie_banner
from .tracking_fetchers import detect_tracking_scripts
from .social_fetchers import find_social_media_links
from .contact_fetchers import find_contact_info
from .image_fetchers import fetch_image_stats
from .crawler import crawl_site_for_links
from .form_fetcher import analyze_forms
from .trust_signal_fetcher import find_trust_signals
from .gmb_fetcher import fetch_gmb_data # <-- NYT: Import

from bs4 import BeautifulSoup
from urllib.parse import urlparse

log = logging.getLogger(__name__)

# --- Default Dictionaries ---
DEFAULT_BASIC_SEO: BasicSEO = {'h1': None, 'h1_count': None, 'h1_texts': None, 'meta_description': None, 'meta_description_length': None, 'title_text': None, 'title_length': None, 'word_count': None, 'image_count': None, 'image_alt_count': None, 'image_alt_pct': None, 'avg_image_size_kb': None, 'canonical_url': None, 'canonical_error': None, 'schema_markup_found': False}
DEFAULT_TECHNICAL_SEO: TechnicalSEO = {'status_code': None, 'is_https': False, 'robots_txt_found': False, 'sitemap_xml_found': False, 'sitemap_locations': [], 'broken_links_count': None, 'broken_links_list': None, 'total_pages_crawled': None, 'total_links_found': None, 'schema_markup_found': False, 'response_time_ms': None, 'canonical_url': None}
DEFAULT_PERFORMANCE_METRICS: PerformanceMetrics = {'lcp_ms': None, 'cls': None, 'inp_ms': None, 'is_mobile_friendly': None, 'performance_score': None, 'psi_status': "not_run", 'total_js_size_kb': None, 'js_file_count': None}
DEFAULT_AUTHORITY_METRICS: AuthorityMetrics = {'domain_authority': None, 'page_authority': None, 'global_rank': None, 'authority_status': "not_run"}
DEFAULT_CONTENT_METRICS: ContentMetrics = {'latest_post_date': None, 'keywords_in_content': {}, 'internal_link_score': None}
DEFAULT_SOCIAL_AND_REPUTATION_METRICS: SocialAndReputationMetrics = {'gmb_review_count': None, 'gmb_average_rating': None, 'gmb_profile_complete': None, 'social_media_links': []}
DEFAULT_CONVERSION_METRICS: ConversionMetrics = {'has_ga4': None, 'has_meta_pixel': None, 'emails_found': [], 'phone_numbers_found': [], 'cta_analysis': {}, 'form_field_counts': [], 'trust_signals_found': []}
DEFAULT_PRIVACY_METRICS: PrivacyMetrics = {'cookie_banner_detected': None, 'detection_method': None, 'personal_data_redacted': None}
DEFAULT_SECURITY_METRICS: SecurityMetrics = {'hsts_enabled': False, 'csp_enabled': False, 'x_content_type_options_enabled': False, 'x_frame_options_enabled': False}


# --- Main Analysis Functions ---
async def analyze_single_url(client: AsyncHtmlClient, url: str) -> UrlAnalysisData:
    log.info(f"Starter analyse for: {url}")
    fetch_method = "httpx_initial"
    soup = await client.get_soup(url)
    
    # Kør synkrone parsere
    basic_seo_result = _parse_basic_seo_from_soup(soup) if soup else DEFAULT_BASIC_SEO.copy()
    privacy_result = detect_cookie_banner(soup) if soup else DEFAULT_PRIVACY_METRICS.copy()
    social_result = find_social_media_links(soup) if soup else DEFAULT_SOCIAL_AND_REPUTATION_METRICS.copy()
    tracking_data = detect_tracking_scripts(soup) if soup else {}
    contact_data = find_contact_info(soup) if soup else {}
    form_data = analyze_forms(soup) if soup else {}
    trust_data = find_trust_signals(soup) if soup else {}
    conversion_result = DEFAULT_CONVERSION_METRICS.copy()
    conversion_result.update(tracking_data)
    conversion_result.update(contact_data)
    conversion_result.update(form_data)
    conversion_result.update(trust_data)

    if not soup or all(basic_seo_result.get(field) is None for field in ("title_text", "h1", "meta_description")):
        log.warning("Første HTML-parse gav ingen essentiel SEO-data for %s. Forsøger Playwright-fallback.", url)
        fetch_method = "playwright_fallback"
        client._playwright_only_domains.add(urlparse(url).netloc)
        soup = await client.get_soup(url)
        # Gen-kør parsere
        basic_seo_result = _parse_basic_seo_from_soup(soup) if soup else DEFAULT_BASIC_SEO.copy()
        privacy_result = detect_cookie_banner(soup) if soup else DEFAULT_PRIVACY_METRICS.copy()
        social_result = find_social_media_links(soup) if soup else DEFAULT_SOCIAL_AND_REPUTATION_METRICS.copy()
        tracking_data = detect_tracking_scripts(soup) if soup else {}
        contact_data = find_contact_info(soup) if soup else {}
        form_data = analyze_forms(soup) if soup else {}
        trust_data = find_trust_signals(soup) if soup else {}
        conversion_result = DEFAULT_CONVERSION_METRICS.copy()
        conversion_result.update(tracking_data)
        conversion_result.update(contact_data)
        conversion_result.update(form_data)
        conversion_result.update(trust_data)

        if not soup or all(basic_seo_result.get(field) is None for field in ("title_text", "h1", "meta_description")):
            error_msg = f"Kunne ikke hente eller parse brugbar HTML for {url} selv med Playwright-fallback."
            log.error(error_msg)
            fetch_method = "fetch_failed"
            return {"url": url, "fetch_error": error_msg, "fetch_method": fetch_method, "basic_seo": basic_seo_result, "technical_seo": DEFAULT_TECHNICAL_SEO.copy(), "performance": DEFAULT_PERFORMANCE_METRICS.copy(), "authority": DEFAULT_AUTHORITY_METRICS.copy(), "security": DEFAULT_SECURITY_METRICS.copy(), "content": DEFAULT_CONTENT_METRICS.copy(), "social_and_reputation": social_result, "conversion": conversion_result, "privacy": privacy_result}

    try:
        final_url_for_fetchers = basic_seo_result.get('canonical_url') or url
        # Asynkrone tasks
        tasks = {
            "tech": fetch_technical_seo_data(client, final_url_for_fetchers),
            "psi": get_performance(client, final_url_for_fetchers),
            "auth": get_authority(client, final_url_for_fetchers),
            "security": fetch_security_headers(client, final_url_for_fetchers),
            "js_size": calculate_js_size(client, soup, url) if soup else asyncio.sleep(0, result={}),
            "images": fetch_image_stats(client, soup, url) if soup else asyncio.sleep(0, result={}),
            "crawl": crawl_site_for_links(client, url),
            "gmb": fetch_gmb_data(client, url), # <-- NYT: Tilføjet GMB-task
        }
        results = await asyncio.gather(*tasks.values(), return_exceptions=True)
        results_map = dict(zip(tasks.keys(), results))

        def get_result(key: str, default_value: Dict):
            res = results_map.get(key)
            if isinstance(res, Exception):
                log.warning(f"Fetcher for '{key}' on {url} failed with exception: {res}")
                return default_value.copy()
            return res

        # Sammensæt de endelige resultater
        image_stats_result = get_result("images", {})
        basic_seo_result.update(image_stats_result)

        performance_result = get_result("psi", DEFAULT_PERFORMANCE_METRICS)
        js_size_result = get_result("js_size", {})
        performance_result.update(js_size_result)
        
        tech_result = get_result("tech", DEFAULT_TECHNICAL_SEO)
        crawl_result = get_result("crawl", {})
        tech_result.update(crawl_result)
        
        gmb_result = get_result("gmb", {}) # <-- NYT: Hent GMB-resultat
        social_result.update(gmb_result) # <-- OPDATERET: Flet GMB ind i social

        final_data: UrlAnalysisData = {
            "url": url,
            "fetch_method": fetch_method,
            "basic_seo": basic_seo_result,
            "technical_seo": tech_result,
            "performance": performance_result,
            "authority": get_result("auth", DEFAULT_AUTHORITY_METRICS),
            "security": get_result("security", DEFAULT_SECURITY_METRICS),
            "content": DEFAULT_CONTENT_METRICS.copy(),
            "social_and_reputation": social_result,
            "conversion": conversion_result,
            "privacy": privacy_result,
        }
        log.info(f"Analyse færdig for: {url}")
        return final_data
    except Exception as e:
        log.error(f"Kritisk fejl under analyse-fasen for {url}: {e}", exc_info=True)
        return {"url": url, "fetch_error": str(e), "fetch_method": "analysis_failed", "basic_seo": basic_seo_result, "technical_seo": DEFAULT_TECHNICAL_SEO.copy(), "performance": DEFAULT_PERFORMANCE_METRICS.copy(), "authority": DEFAULT_AUTHORITY_METRICS.copy(), "security": DEFAULT_SECURITY_METRICS.copy(), "content": DEFAULT_CONTENT_METRICS.copy(), "social_and_reputation": social_result, "conversion": conversion_result, "privacy": privacy_result}

async def analyze_multiple_urls(urls: List[str], num_workers: int = 10, verify_ssl: bool = True, proxy: Optional[str] = None) -> List[UrlAnalysisData]:
    """
    Orkestrerer analysen af flere URLs med én enkelt, delt hybrid-klient.
    """
    log.info("Fandt %d URLs. Starter analyse med %d workers...", len(urls), num_workers)
    async with AsyncHtmlClient(max_connections=num_workers, verify_ssl=verify_ssl, proxy=proxy) as client:
        tasks = [analyze_single_url(client, url) for url in urls]
        all_data = await asyncio.gather(*tasks)
        return all_data