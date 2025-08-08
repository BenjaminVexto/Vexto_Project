# src/vexto/scoring/analyzer.py

import asyncio
import logging
from typing import List, Optional, Dict, Any
from bs4 import BeautifulSoup
from urllib.parse import urlparse, urljoin
from collections import Counter

from . import (
    http_client, website_fetchers, technical_fetchers, performance_fetcher,
    authority_fetcher, security_fetchers, privacy_fetchers, tracking_fetchers,
    social_fetchers, contact_fetchers, image_fetchers, form_fetcher,
    gmb_fetcher, content_fetcher, cta_fetcher
)
from .schemas import (
    BasicSEO, TechnicalSEO, SocialAndReputationMetrics, PerformanceMetrics,
    AuthorityMetrics, ContentMetrics, ConversionMetrics, PrivacyMetrics,
    SecurityMetrics, UrlAnalysisData
)

log = logging.getLogger(__name__)
print("Loading analyzer.py from:", __file__)

# Default dictionaries remain the same
DEFAULT_BASIC_SEO: BasicSEO = {
    'h1': None, 'h1_count': 0, 'h1_texts': None,
    'meta_description': None, 'meta_description_length': 0,
    'title_text': None, 'title_length': 0,
    'word_count': 0,
    'image_count': 0, 'image_alt_count': 0, 'image_alt_pct': 0, 'avg_image_size_kb': 0,
    'canonical_url': None, 'canonical_error': None,
    'schema_markup_found': False
}

DEFAULT_TECHNICAL_SEO: TechnicalSEO = {
    'status_code': None, 'is_https': False,
    'robots_txt_found': False, 'sitemap_xml_found': False, 'sitemap_locations': [],
    'sitemap_is_fresh': True,
    'response_time_ms': 0,
    'total_pages_crawled': 0, 'total_links_found': 0,
    'broken_links_count': 0, 'broken_links_pct': 0, 'broken_links_list': [],
    'schema_markup_found': False, 'canonical_url': None,
    'internal_link_score': 0
}

DEFAULT_SOCIAL_AND_REPUTATION_METRICS: SocialAndReputationMetrics = {
    'gmb_review_count': 0, 'gmb_average_rating': None,
    'gmb_profile_complete': None, 'social_media_links': [], 'social_share_links': []
}

DEFAULT_PERFORMANCE_METRICS: PerformanceMetrics = {
    'lcp_ms': None, 'cls': None, 'inp_ms': None,
    'viewport_score': 0,
    'performance_score': 0,
    'psi_status': "not_run", 'total_js_size_kb': 0, 'js_file_count': 0
}

DEFAULT_AUTHORITY_METRICS: AuthorityMetrics = {
    'domain_authority': 0, 'page_authority': 0,
    'global_rank': None, 'authority_status': "not_run"
}

DEFAULT_CONTENT_METRICS: ContentMetrics = {
    'latest_post_date': None, 'days_since_last_post': 99999,
    'keywords_in_content': {}, 'internal_link_score': 0,
    'keyword_relevance_score': 0, 'average_word_count': 0
}

DEFAULT_CONVERSION_METRICS: ConversionMetrics = {
    'has_ga4': None, 'has_meta_pixel': None,
    'emails_found': [], 'phone_numbers_found': [],
    'form_field_counts': [], 'trust_signals_found': [],
    'cta_analysis': {}, 'cta_score': 0
}

DEFAULT_PRIVACY_METRICS: PrivacyMetrics = {
    'cookie_banner_detected': None, 'detection_method': None, 'personal_data_redacted': None
}

DEFAULT_SECURITY_METRICS: SecurityMetrics = {
    'hsts_enabled': False, 'csp_enabled': False,
    'x_content_type_options_enabled': False, 'x_frame_options_enabled': False
}

def log_metric_status(metric: str, value: any, status: str = "ok"):
    log.info(f"[{status}] {metric}: {value}")

def _extract_html_and_canonical_data(raw_result):
    """
    Robust extraction af HTML og canonical data fra forskellige return formater.
    
    Returns:
        tuple: (html_content: str, canonical_data: dict)
    """
    if raw_result is None:
        return None, {}
    
    # Format 1: Dict format fra Playwright
    if isinstance(raw_result, dict):
        html_content = raw_result.get('html')
        canonical_data = raw_result.get('canonical_data', {})
        return html_content, canonical_data
    
    # Format 2: Tuple format (html, canonical_data)
    elif isinstance(raw_result, tuple) and len(raw_result) == 2:
        html_content, canonical_data = raw_result
        # Tjek om anden element faktisk er canonical_data (dict) eller HTML string
        if isinstance(canonical_data, str):
            # Dette er faktisk HTML, canonical_data er tom
            return html_content, {}
        return html_content, canonical_data if isinstance(canonical_data, dict) else {}
    
    # Format 3: Bare HTML string
    elif isinstance(raw_result, str):
        return raw_result, {}
    
    # Ukendt format
    else:
        log.warning(f"Unknown raw_result format: {type(raw_result)}")
        return None, {}

async def analyze_single_url(client: http_client.AsyncHtmlClient, url: str, max_pages: int = 50) -> UrlAnalysisData:
    # Valider URL
    if url.startswith('https://https://') or url.startswith('http://http://'):
        url = url.replace('https://https://', 'https://').replace('http://http://', 'http://')
        log.warning(f"Fixed invalid URL: {url}")

    log.info(f"Starting analysis for: {url}")
    client.headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}

    # Crawl flere sider
    from .crawler import crawl_site_for_links
    crawled_urls = await crawl_site_for_links(client, url, max_pages=max_pages)
    log.info(f"Crawled {crawled_urls['total_pages_crawled']} pages")

    analysis_data = {
        "url": url,
        "fetch_method": client.last_fetch_method or "unknown",
        "basic_seo": DEFAULT_BASIC_SEO.copy(),
        "technical_seo": DEFAULT_TECHNICAL_SEO.copy(),
        "performance": DEFAULT_PERFORMANCE_METRICS.copy(),
        "authority": DEFAULT_AUTHORITY_METRICS.copy(),
        "security": DEFAULT_SECURITY_METRICS.copy(),
        "content": DEFAULT_CONTENT_METRICS.copy(),
        "social_and_reputation": DEFAULT_SOCIAL_AND_REPUTATION_METRICS.copy(),
        "conversion": DEFAULT_CONVERSION_METRICS.copy(),
        "privacy": DEFAULT_PRIVACY_METRICS.copy(),
        "benchmark_complete": False,
        "ux_ui_score": 0,
        "niche_score": 0,
    }
    
    # FLYTTET UD AF LOOP: Kald GMB og Authority én gang for hoveddomænet
    parsed_url = urlparse(url)
    base_url = f"{parsed_url.scheme}://{parsed_url.netloc}"
    
    # GMB data - kun én gang
    gmb_data = await gmb_fetcher.fetch_gmb_data(client, url)
    analysis_data['social_and_reputation'].update(gmb_data)
    log.info(f"GMB data fetched once: review_count={gmb_data.get('gmb_review_count', 0)}, rating={gmb_data.get('gmb_average_rating', 0)}")
    
    # Authority data - kun én gang
    authority_data = await authority_fetcher.get_authority(client, url)
    analysis_data['authority'].update(authority_data)
    log.info(f"Authority data fetched once: DA={authority_data.get('domain_authority', 0)}, PA={authority_data.get('page_authority', 0)}")
    
    page_count = 0
    h1_texts = []
    keyword_scores = []
    emails = set()
    phones = set()
    form_counts = []
    trust_signals = set()
    word_counts = []

    main_page_data = None
    canonical_found = False

    parsed_url = urlparse(url)
    base_url = f"{parsed_url.scheme}://{parsed_url.netloc}"
    raw_result = await client.get_raw_html(base_url, force=True)
    html_content, canonical_data = _extract_html_and_canonical_data(raw_result)
    if html_content:
        soup = BeautifulSoup(html_content, "lxml")
    
        # Technical SEO (inkl. sitemap/robots.txt)
        tech_result = await technical_fetchers.fetch_technical_seo_data(client, base_url)
        analysis_data['technical_seo'].update(tech_result)
    
        # Content metrics (inkl. blog-detection)
        content_data = await content_fetcher.fetch_content_data(client, soup, base_url)
        analysis_data['content'].update(content_data)
    
    for crawled_url in crawled_urls.get('visited_urls', {url}):
        raw_result = await client.get_raw_html(crawled_url, force=True)
        html_content, canonical_data = _extract_html_and_canonical_data(raw_result)
        
        if not html_content:
            log.error(f"Could not fetch usable HTML for {crawled_url}")
            continue

        soup = BeautifulSoup(html_content, "lxml")
        parsed_crawled_url = urlparse(crawled_url)
        crawled_base_url = f"{parsed_crawled_url.scheme}://{parsed_crawled_url.netloc}"

        # Sync parsers
        basic_seo_result = website_fetchers.parse_basic_seo_from_soup(soup, crawled_base_url)
        
        # Forbedret canonical håndtering
        if not basic_seo_result.get('canonical_url') and canonical_data and isinstance(canonical_data, dict):
            log.debug(f"Fallback to Playwright canonical_data for {crawled_url}: {canonical_data}")
            
            # Prioriter link canonical
            if canonical_data.get('link_canonical'):
                basic_seo_result['canonical_url'] = urljoin(crawled_base_url, canonical_data['link_canonical'])
                basic_seo_result['canonical_source'] = 'playwright_link'
                basic_seo_result['canonical_error'] = None
                canonical_found = True
                log.info(f"Used Playwright link canonical: {basic_seo_result['canonical_url']}")
            
            # Fallback til canonical fields
            elif canonical_data.get('canonical_fields') and isinstance(canonical_data['canonical_fields'], dict):
                for path, value in canonical_data['canonical_fields'].items():
                    if value and str(value).strip() not in ('None', 'null', '', 'false', 'False'):
                        if str(value).lower() == 'true':
                            basic_seo_result['canonical_url'] = crawled_url
                        else:
                            basic_seo_result['canonical_url'] = urljoin(crawled_base_url, str(value))
                        basic_seo_result['canonical_source'] = 'playwright_runtime_state'
                        basic_seo_result['canonical_error'] = None
                        canonical_found = True
                        log.info(f"Used Playwright canonical: {basic_seo_result['canonical_url']} from {path}")
                        break

        # Gem hovedsiden eller første side med gyldig canonical
        if crawled_url == url:
            main_page_data = basic_seo_result
        elif not main_page_data or (not main_page_data.get('canonical_url') and basic_seo_result.get('canonical_url')):
            main_page_data = basic_seo_result
            log.info(f"Updated main_page_data with canonical from {crawled_url}: {basic_seo_result.get('canonical_url')}")
            
        # Saml data fra siden
        if basic_seo_result.get('h1_texts'):
            h1_texts.extend(basic_seo_result['h1_texts'])
        if basic_seo_result.get('word_count'):
            word_counts.append(basic_seo_result['word_count'])
            
        social_result = social_fetchers.find_social_media_links(soup)
        privacy_result = privacy_fetchers.detect_cookie_banner(soup)
        trust_data = privacy_fetchers.detect_trust_signals(soup)
        trust_signals.update(trust_data.get('trust_signals_found', []))
        contact_data = await contact_fetchers.find_contact_info(soup, crawled_base_url)
        emails.update(contact_data.get('emails_found', []))
        phones.update(contact_data.get('phone_numbers_found', []))
        form_data = form_fetcher.analyze_forms(soup)
        form_counts.extend(form_data.get('form_field_counts', []))
        tracking_data = await tracking_fetchers.fetch_tracking_data(client, soup)
        if content_data.get('keyword_relevance_score'):
            keyword_scores.append(content_data['keyword_relevance_score'])
        cta_data = await cta_fetcher.fetch_cta_data(client, soup)

        # Kun de væsentligste async tasks per side
        tasks = {
            "tech": technical_fetchers.fetch_technical_seo_data(client, crawled_url),
            "psi": performance_fetcher.get_performance(client, crawled_url) if crawled_url == url else None,  # Kun hovedsiden
            "security": security_fetchers.fetch_security_headers(client, crawled_url) if crawled_url == url else None,  # Kun hovedsiden
            "js_size": performance_fetcher.calculate_js_size(client, soup, crawled_url),
            "images": image_fetchers.fetch_image_stats(client, soup, crawled_url),
        }
        
        # Filter None tasks
        filtered_tasks = {k: v for k, v in tasks.items() if v is not None}
        results = await asyncio.gather(*(filtered_tasks.values()), return_exceptions=True)
        results_map = dict(zip(filtered_tasks.keys(), results))

        def get_result(key: str, default_value: Dict):
            res = results_map.get(key)
            if isinstance(res, Exception):
                log.warning(f"Fetcher for '{key}' on {crawled_url} failed with exception: {res}", exc_info=True)
                return default_value.copy()
            return res or default_value.copy()

        basic_seo_result.update(get_result("images", {}))
        if crawled_url == url:  # Kun opdater performance og security for hovedsiden
            performance_result = get_result("psi", DEFAULT_PERFORMANCE_METRICS)
            performance_result.update(get_result("js_size", {}))
            analysis_data['performance'].update(performance_result)
            analysis_data['security'].update(get_result("security", DEFAULT_SECURITY_METRICS))
            
        # Aggregér data
        for section, data in [
            ("basic_seo", basic_seo_result), 
            ("technical_seo", tech_result),
            ("content", content_data),
            ("social_and_reputation", social_result), 
            ("conversion", {
                **tracking_data, **contact_data, **form_data, **trust_data, **cta_data
            }), 
            ("privacy", privacy_result)
        ]:
            for key, value in data.items():
                if isinstance(value, (int, float)) and key not in ['keyword_relevance_score']:
                    current = analysis_data[section].get(key, 0)
                    if current is None:
                        current = 0
                    analysis_data[section][key] = current + value
                elif isinstance(value, list):
                    current = analysis_data[section].get(key, [])
                    if current is None:
                        current = []
                    analysis_data[section][key] = list(set(current + value))
                else:
                    # Don't overwrite with None values
                    if value is not None or analysis_data[section].get(key) is None:
                        analysis_data[section][key] = value
        page_count += 1

    # Update crawled URLs info
    analysis_data['technical_seo'].update(crawled_urls)

    # Smart aggregation efter loop
    if main_page_data:
        # Canonical håndtering - FORBEDRET
        if main_page_data.get('canonical_url'):
            analysis_data['basic_seo']['canonical_url'] = main_page_data['canonical_url']
            analysis_data['basic_seo']['canonical_source'] = main_page_data.get('canonical_source', 'unknown')
            analysis_data['basic_seo']['canonical_error'] = None
            log.info(f"[canonical][set] Used canonical from main_page_data: {main_page_data['canonical_url']}")
        else:
            analysis_data['basic_seo']['canonical_error'] = "No canonical found on any page"
            log.info(f"[canonical][missing] No canonical found on any crawled page")
            
        analysis_data['basic_seo']['schema_markup_found'] = main_page_data.get('schema_markup_found', False)
        
        # Use main page title and meta description
        analysis_data['basic_seo']['title_text'] = main_page_data.get('title_text')
        analysis_data['basic_seo']['title_length'] = main_page_data.get('title_length', 0)
        analysis_data['basic_seo']['meta_description'] = main_page_data.get('meta_description')
        analysis_data['basic_seo']['meta_description_length'] = main_page_data.get('meta_description_length', 0)
        
    # Log canonical status
    log_metric_status("canonical_url", analysis_data['basic_seo']['canonical_url'], 
                     "ok" if analysis_data['basic_seo']['canonical_url'] else "missing")
    log_metric_status("schema_markup_found", analysis_data['basic_seo']['schema_markup_found'], 
                     "ok" if analysis_data['basic_seo']['schema_markup_found'] else "missing")
    
    # H1 aggregation
    if h1_texts:
        most_common_h1 = Counter(h1_texts).most_common(1)[0][0]
        analysis_data['basic_seo']['h1'] = most_common_h1
        analysis_data['basic_seo']['h1_count'] = len(h1_texts) / max(page_count, 1)  # Average
        analysis_data['basic_seo']['h1_texts'] = list(set(h1_texts))[:5]  # Max 5 unique
        log_metric_status("h1", most_common_h1, "ok")
    else:
        log_metric_status("h1", None, "missing")
        
    # Keyword relevance score
    if keyword_scores:
        analysis_data['content']['keyword_relevance_score'] = max(keyword_scores)
        log_metric_status("keyword_relevance_score", analysis_data['content']['keyword_relevance_score'], "ok")
        
    # Average word count
    if word_counts:
        analysis_data['content']['average_word_count'] = sum(word_counts) // len(word_counts)
        analysis_data['basic_seo']['word_count'] = analysis_data['content']['average_word_count']
        
    # Contact and trust signals
    analysis_data['conversion']['emails_found'] = list(emails)
    analysis_data['conversion']['phone_numbers_found'] = list(phones)
    analysis_data['conversion']['form_field_counts'] = form_counts
    analysis_data['conversion']['trust_signals_found'] = list(trust_signals)
    
    # Calculate CTA score based on collected data
    cta_score = 0
    if emails:
        cta_score += 35
    if phones:
        cta_score += 35
    if form_counts:
        cta_score += 30
    if trust_signals:
        cta_score += 30
    if analysis_data['conversion'].get('cta_analysis'):
        cta_score += 30
    # Cap at reasonable maximum
    analysis_data['conversion']['cta_score'] = min(cta_score, 60)
    
    log_metric_status("emails_found", analysis_data['conversion']['emails_found'], 
                     "ok" if emails else "missing")
    log_metric_status("phone_numbers_found", analysis_data['conversion']['phone_numbers_found'], 
                     "ok" if phones else "missing")
    log_metric_status("form_field_counts", analysis_data['conversion']['form_field_counts'], 
                     "ok" if form_counts else "missing")
    log_metric_status("trust_signals_found", analysis_data['conversion']['trust_signals_found'], 
                     "ok" if trust_signals else "missing")
    
    # Average numeric values hvis vi har flere sider
    if page_count > 1:
        for section in ['basic_seo', 'performance', 'technical_seo']:
            for key, value in analysis_data[section].items():
                if isinstance(value, (int, float)) and key not in ['status_code', 'is_https', 'total_pages_crawled', 
                                                                   'total_links_found', 'broken_links_count']:
                    analysis_data[section][key] = value / page_count
                    
    # Default values for missing data
    if not analysis_data['authority']['domain_authority']:
        analysis_data['authority']['domain_authority'] = 10
        log_metric_status("domain_authority", 10, "missing")
    else:
        log_metric_status("domain_authority", analysis_data['authority']['domain_authority'], "ok")
        
    # Copy GMB score to authority
    analysis_data['authority']['gmb_profile_complete'] = analysis_data['social_and_reputation'].get('gmb_profile_complete', 0)
    log_metric_status("gmb_profile_complete", analysis_data['authority']['gmb_profile_complete'], 
                     "ok" if analysis_data['authority']['gmb_profile_complete'] else "missing")
    
    # Content metrics
    analysis_data['content']['days_since_last_post'] = 99999 if not analysis_data['content'].get('latest_post_date') else analysis_data['content']['days_since_last_post']
    analysis_data['content']['internal_link_score'] = analysis_data['technical_seo'].get('internal_link_score', 0)
    
    log_metric_status("days_since_last_post", analysis_data['content']['days_since_last_post'], 
                     "missing" if analysis_data['content']['days_since_last_post'] == 99999 else "ok")
    log_metric_status("internal_link_score", analysis_data['content']['internal_link_score'], "ok")

    log.info(f"Analysis complete for: {url}")
    return analysis_data