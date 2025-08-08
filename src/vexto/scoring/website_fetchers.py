# website_fetchers.py

from __future__ import annotations
from bs4 import BeautifulSoup
import logging, asyncio
import os
from typing import Dict, Any, List, Tuple, Union
from urllib.parse import urlparse, urljoin 
import httpx
import re
import json

from .schemas import BasicSEO, TechnicalSEO
from .http_client import AsyncHtmlClient

log = logging.getLogger(__name__)

# ---------- Parser med forbedret JSON-håndtering og fejlhåndtering -----------------------------
def _parse_basic_seo_from_soup(soup: BeautifulSoup, base_url: str | None = None) -> BasicSEO:
# I website_fetchers.py, inde i def _parse_basic_seo_from_soup(soup: BeautifulSoup, base_url: str | None = None) -> BasicSEO:

    if not soup:
        return {
            'h1': None, 'h1_count': None, 'h1_texts': None,
            'meta_description': None, 'meta_description_length': None,
            'title_text': None, 'title_length': None,
            'word_count': None,
            'canonical_url': None, 'canonical_error': "No HTML content to parse",
            'schema_markup_found': False,
            'canonical_source': 'none'
        }
    try:
        h1_tags = soup.find_all('h1')
        h1_count = len(h1_tags)
        h1_texts = [h1.get_text(strip=True) for h1 in h1_tags] if h1_tags else []
        h1_first_text = h1_texts[0] if h1_texts else None

        # Forbedret meta description håndtering
        meta_tag = soup.find('meta', attrs={'name': lambda x: x and x.lower() == 'description'})
        meta_description_text = None
        if meta_tag and 'content' in meta_tag.attrs:
            content = meta_tag['content']
            # Robust håndtering af forskellige datatyper
            if isinstance(content, str):
                meta_description_text = content.strip()
            elif isinstance(content, (list, tuple)) and content:
                # Tag den første ikke-tomme værdi
                for item in content:
                    if isinstance(item, str) and item.strip():
                        meta_description_text = item.strip()
                        break
            elif content is not None:
                # Forsøg at konvertere andre typer til string
                try:
                    meta_description_text = str(content).strip()
                    log.debug(f"Converted meta content from {type(content)} to string: {meta_description_text[:50]}...")
                except Exception as e:
                    log.warning(f"Failed to convert meta content: {e}")

        meta_description_length = len(meta_description_text) if meta_description_text else 0

        # Forbedret title håndtering
        title_tag = soup.find('title')
        title_text = None
        if title_tag:
            try:
                content = title_tag.get_text(separator=' ', strip=True)
                # Robust håndtering af forskellige datatyper
                if isinstance(content, str):
                    title_text = content
                elif isinstance(content, (list, tuple)) and content:
                    # Tag den første ikke-tomme værdi
                    for item in content:
                        if isinstance(item, str) and item.strip():
                            title_text = item.strip()
                            break
                elif content is not None:
                    try:
                        title_text = str(content).strip()
                        log.debug(f"Converted title content from {type(content)} to string: {title_text[:50]}...")
                    except Exception as e:
                        log.warning(f"Failed to convert title content: {e}")
            except Exception as e:
                log.warning(f"Error extracting title text: {e}")
        
        title_length = len(title_text) if title_text else 0

        # Robust word count
        try:
            word_count = len(soup.get_text(' ', strip=True).split())
        except Exception as e:
            log.warning(f"Error calculating word count: {e}")
            word_count = 0

        # Canonical URL parsing
        canonical_tags = soup.find_all('link', attrs={'rel': 'canonical'})
        canonical_url = None
        canonical_error = None
        canonical_source = 'none'

        if len(canonical_tags) == 1:
            href = canonical_tags[0].get('href')
            if href and isinstance(href, str):
                canonical_url = urljoin(base_url, href) if base_url else href
                canonical_source = 'link'
                log.debug(f"Found canonical tag: {canonical_url} (source: link)")
        elif len(canonical_tags) > 1:
            canonical_error = "Multiple canonical tags found"
        else:
            # Fallback: Tjek __INITIAL_STATE__ for custom_canonical_url
            script_tags = soup.find_all('script')
            for script in script_tags:
                script_content = script.get_text() or ""
                if 'window.__INITIAL_STATE__' in script_content:
                    try:
                        json_str_match = re.search(r'window\.__INITIAL_STATE__\s*=\s*(\{[\s\S]*?\})(;|$)', script_content, re.DOTALL)
                        if json_str_match:
                            json_str = json_str_match.group(1)
                            log.debug(f"Extracted JSON string: {json_str[:500]}... (length: {len(json_str)})")
                            
                            # Forbedret JSON rengøring
                            json_str_clean = json_str
                            # Fjern trailing commas
                            json_str_clean = re.sub(r',\s*}', '}', json_str_clean)
                            json_str_clean = re.sub(r',\s*]', ']', json_str_clean)
                            # Håndter JavaScript værdier
                            json_str_clean = re.sub(r':\s*undefined', ': null', json_str_clean)
                            json_str_clean = re.sub(r':\s*!0\b', ': true', json_str_clean)
                            json_str_clean = re.sub(r':\s*!1\b', ': false', json_str_clean)
                            # Håndter funktioner
                            json_str_clean = re.sub(r':\s*function\s*\([^)]*\)\s*\{[^}]*\}', ': null', json_str_clean)
                            
                            initial_state = json.loads(json_str_clean)

                            # Rekursiv søgning efter canonical URLs
                            canonicals = []
                            def find_canonicals(obj, path=""):
                                if isinstance(obj, dict):
                                    for k, v in obj.items():
                                        current_path = f"{path}.{k}" if path else k
                                        if k in ("custom_canonical_url", "canonical_url", "canonical") and v not in (None, "null", "", False):
                                            if isinstance(v, str) and v.strip():
                                                canonicals.append((v.strip(), current_path))
                                            elif isinstance(v, bool) and v:
                                                canonicals.append((base_url or "", current_path))
                                        else:
                                            find_canonicals(v, current_path)
                                elif isinstance(obj, list):
                                    for i, item in enumerate(obj):
                                        find_canonicals(item, f"{path}[{i}]" if path else f"[{i}]")

                            find_canonicals(initial_state)
                            
                            # Vælg første gyldige canonical
                            for canonical_value, path in canonicals:
                                if canonical_value and canonical_value not in ("null", ""):
                                    if canonical_value == base_url:
                                        canonical_url = canonical_value
                                        canonical_source = 'custom_self'
                                    else:
                                        canonical_url = urljoin(base_url, canonical_value) if base_url else canonical_value
                                        canonical_source = 'custom'
                                    log.debug(f"Found custom_canonical_url in __INITIAL_STATE__: {canonical_url} from path: {path}")
                                    break
                            
                            if canonical_url:
                                break
                    except json.JSONDecodeError as e:
                        log.warning(f"Failed to parse __INITIAL_STATE__ JSON: {e}")
                    except Exception as e:
                        log.warning(f"Unexpected error parsing __INITIAL_STATE__: {e}")

        if not canonical_url:
            canonical_error = "No canonical tag or CMS canonical found"

        # Forbedret schema markup detection
        schema_markup_found = False
        schema_types = []  # Ny: Saml types for logging/analyse

        # Find alle JSON-LD scripts
        json_ld_scripts = soup.find_all('script', attrs={'type': 'application/ld+json'})
        for script in json_ld_scripts:
            try:
                # Ekstraher og parse JSON
                script_content = script.string
                if script_content:
                    schema_data = json.loads(script_content)
                    
                    # Håndter single dict eller list af dicts
                    if isinstance(schema_data, dict) and '@type' in schema_data:
                        schema_types.append(schema_data['@type'])
                        schema_markup_found = True
                    elif isinstance(schema_data, list):
                        for item in schema_data:
                            if isinstance(item, dict) and '@type' in item:
                                schema_types.append(item['@type'])
                                schema_markup_found = True
            except json.JSONDecodeError as e:
                log.warning(f"Invalid JSON-LD script on {base_url}: {e}")
            except Exception as e:
                log.warning(f"Error parsing JSON-LD: {e}")

        # Fallback: Tjek for microdata (bevar eksisterende, men udvid hvis nødvendigt)
        microdata_tag = soup.find(attrs={'itemscope': True})
        if microdata_tag and not schema_markup_found:
            schema_markup_found = True
            schema_types.append('Microdata')  # Generisk label, da microdata ikke har @type

        # Initialiser result dict før opdatering
        result = {
            'h1': h1_first_text, 'h1_count': h1_count, 'h1_texts': h1_texts,
            'meta_description': meta_description_text, 'meta_description_length': meta_description_length,
            'title_text': title_text, 'title_length': title_length,
            'word_count': word_count,
            'canonical_url': canonical_url, 'canonical_error': canonical_error,
            'schema_markup_found': schema_markup_found,
            'canonical_source': canonical_source
        }

        # Opdater resultatet med schema (redundant nu, da det er sat i dict)
        result['schema_markup_found'] = schema_markup_found

        # Log for bedre indsigt (kun hvis fundet eller fejl)
        if schema_markup_found:
            log.info(f"Schema types found on {base_url}: {schema_types}")
        elif json_ld_scripts or microdata_tag:
            log.info(f"Schema tags found but invalid/empty on {base_url}")
        else:
            log.info(f"No schema markup found on {base_url}")

        return result
    except Exception as e:
        log.error("Error during basic SEO parsing: %s", e, exc_info=True)
        return {
            'h1': None, 'h1_count': None, 'h1_texts': None,
            'meta_description': None, 'meta_description_length': None,
            'title_text': None, 'title_length': None,
            'word_count': None,
            'canonical_url': None, 'canonical_error': "Parsing failed with exception",
            'schema_markup_found': False,
            'canonical_source': 'none'
        }

# ---------- Offentlig wrapper -----------------------------------------------
def parse_basic_seo_from_soup(soup: BeautifulSoup, base_url: str | None = None) -> BasicSEO:
    """
    Offentlig wrapper omkring _parse_basic_seo_from_soup.
    Sender base_url til parsing for at håndtere relative canonical URLs.
    """
    return _parse_basic_seo_from_soup(soup, base_url)

# ---------- Hjælpere --------------------------------------------------------
ESSENTIAL_FIELDS = ("title_text", "h1", "meta_description")

def _is_essential_missing(basic: BasicSEO) -> bool:
    return all(basic.get(f) in (None, "", []) for f in ESSENTIAL_FIELDS)

async def _check_sitemap_in_robots(client: AsyncHtmlClient, robots_url: str) -> List[str]:
    sitemaps = []
    try:
        response = await client.get(robots_url, follow_redirects=True)
        if response.status_code == 200:
            lines = response.text.splitlines()
            for line in lines:
                if line.lower().startswith('sitemap:'):
                    sitemaps.append(line.split(':', 1)[1].strip())
    except Exception as e:
        log.debug("Kunne ikke læse sitemap fra %s: %s", robots_url, e)
    return sitemaps

# ---------- Fetchers --------------------------------------------------------
async def fetch_basic_seo_data(
    client: AsyncHtmlClient,
    url: str,
    *,
    timeout: int = 25,
) -> Tuple[Dict[str, Any], str | None]:
    """
    Hent HTML + Basic SEO (to værdier).
    1) Forsøg billig SSR/cached fetch (httpx).
    2) Parse. Hvis essentielle felter mangler -> eskalér til Playwright (én gang).
    3) Returnér basic-dict + rå HTML-streng, hvor canonical merges/fallbacker.
    
    VIGTIGT: Returnerer altid (basic_seo_dict, html_content) - IKKE canonical_data separat
    """
    
    basic_final = {}
    html_out = None
    
    # --- Første forsøg: SSR / cache ---
    try:
        ssr_result = await client.get_raw_html(
            url,
            force=False,
            return_soup=False,
            force_playwright=False
        )
        
        # Robust håndtering af return format
        html_content_ssr = None
        ssr_canonical_data = {}
        
        if ssr_result is None:
            html_content_ssr, ssr_canonical_data = None, {}
        elif isinstance(ssr_result, str):
            html_content_ssr, ssr_canonical_data = ssr_result, {}
        elif isinstance(ssr_result, tuple) and len(ssr_result) == 2:
            html_content_ssr, ssr_canonical_data = ssr_result
        elif isinstance(ssr_result, dict):
            # Hvis det er returneret som dict (fra Playwright)
            html_content_ssr = ssr_result.get('html')
            ssr_canonical_data = ssr_result.get('canonical_data', {})
        else:
            log.warning(f"Unexpected SSR result format: {type(ssr_result)}")
            html_content_ssr, ssr_canonical_data = None, {}
            
    except Exception as e:
        log.warning(f"SSR fetch failed for {url}: {e}")
        html_content_ssr, ssr_canonical_data = None, {}

    soup_ssr = BeautifulSoup(html_content_ssr, "lxml") if html_content_ssr else None
    basic_ssr = _parse_basic_seo_from_soup(soup_ssr, url) if soup_ssr else {}
    
    html_out = html_content_ssr
    need_render = (html_content_ssr is None) or _is_essential_missing(basic_ssr)
    
    # Initialiser med SSR data
    basic_final = basic_ssr.copy() if basic_ssr else {}
    
    if need_render:
        log.info(f"Eskalerer {url} til Playwright (dynamic / mangler essentials).")
        try:
            playwright_result = await client.get_raw_html(
                url,
                force_playwright=True,
                return_soup=False,
                force=False
            )
            
            # Robust håndtering af Playwright return format
            rendered_html = None
            pw_canonical_data = {}
            
            if playwright_result is None:
                rendered_html, pw_canonical_data = None, {}
            elif isinstance(playwright_result, str):
                rendered_html, pw_canonical_data = playwright_result, {}
            elif isinstance(playwright_result, tuple) and len(playwright_result) == 2:
                rendered_html, pw_canonical_data = playwright_result
            elif isinstance(playwright_result, dict):
                # Dette er det forventede format fra http_client
                rendered_html = playwright_result.get('html')
                pw_canonical_data = playwright_result.get('canonical_data', {})
            else:
                log.warning(f"Unexpected Playwright result format: {type(playwright_result)}")
                rendered_html, pw_canonical_data = None, {}
                
        except Exception as e:
            log.error(f"Playwright fetch failed for {url}: {e}")
            rendered_html, pw_canonical_data = None, {}
        
        if rendered_html:
            soup_playwright = BeautifulSoup(rendered_html, "lxml")
            basic_playwright = _parse_basic_seo_from_soup(soup_playwright, url)
            html_out = rendered_html
            
            # Merge data - prioriter Playwright for ikke-tomme værdier
            for field in basic_playwright:
                pw_value = basic_playwright.get(field)
                if pw_value not in (None, "", [], 0) and field != 'canonical_source':
                    basic_final[field] = pw_value
            
            # Særlig håndtering af canonical URL - prioriter Playwright fund
            canonical_url_final = None
            canonical_source_final = basic_final.get('canonical_source', 'none')
            
            # 1. Prioriter parsed canonical fra Playwright HTML
            if basic_playwright.get('canonical_url') not in (None, "", []):
                canonical_url_final = basic_playwright['canonical_url']
                canonical_source_final = basic_playwright.get('canonical_source', 'playwright_parsed')
                log.info(f"Using Playwright parsed canonical: {canonical_url_final}")
            
            # 2. Fallback til Playwright runtime data
            elif pw_canonical_data and isinstance(pw_canonical_data, dict):
                # Tjek link canonical
                if pw_canonical_data.get('link_canonical'):
                    canonical_url_final = urljoin(url, pw_canonical_data['link_canonical'])
                    canonical_source_final = 'playwright_link'
                    log.info(f"Using Playwright link canonical: {canonical_url_final}")
                
                # Tjek runtime state canonical fields
                elif pw_canonical_data.get('canonical_fields'):
                    canonical_fields = pw_canonical_data['canonical_fields']
                    if isinstance(canonical_fields, dict):
                        for path, value in canonical_fields.items():
                            if value and str(value).strip() not in ('None', 'null', '', 'false', 'False'):
                                if str(value).lower() == 'true':
                                    canonical_url_final = url  # Brug den aktuelle URL
                                else:
                                    canonical_url_final = urljoin(url, str(value))
                                canonical_source_final = 'playwright_runtime_state'
                                log.info(f"Using Playwright runtime canonical: {canonical_url_final} from {path}")
                                break
            
            # 3. Fallback til SSR hvis intet fundet
            if not canonical_url_final and basic_ssr.get('canonical_url'):
                canonical_url_final = basic_ssr['canonical_url']
                canonical_source_final = basic_ssr.get('canonical_source', 'ssr_fallback')
                log.info(f"Using SSR fallback canonical: {canonical_url_final}")
            
            # Opdater final data
            basic_final['canonical_url'] = canonical_url_final
            basic_final['canonical_source'] = canonical_source_final
            basic_final['canonical_error'] = None if canonical_url_final else "No canonical found after all attempts"
            
    else:
        log.info(f"Using SSR data for {url} (essentials found)")
    
    # Sikr at alle nødvendige felter findes
    required_fields = {
        'h1': None, 'h1_count': 0, 'h1_texts': [],
        'meta_description': None, 'meta_description_length': 0,
        'title_text': None, 'title_length': 0,
        'word_count': 0, 'canonical_url': None, 'canonical_error': None,
        'schema_markup_found': False, 'canonical_source': 'none'
    }
    
    for field, default_value in required_fields.items():
        if field not in basic_final:
            basic_final[field] = default_value

    log.info(f"Final canonical for {url}: {basic_final.get('canonical_url')} (source: {basic_final.get('canonical_source')})")
    
    # VIGTIGT: Returner altid tuple format som forventet af analyzer.py
    return basic_final, html_out

# ---------- Teknisk SEO -----------------------------------------------------
async def fetch_technical_seo_data(client: AsyncHtmlClient, url: str) -> TechnicalSEO:
    data: TechnicalSEO = {
        'status_code': None, 'is_https': False, 'robots_txt_found': False,
        'sitemap_xml_found': False, 'sitemap_locations': [], 'broken_links_count': 0,
        'broken_links_list': [], 'total_pages_crawled': 0, 'total_links_found': 0,
        'schema_markup_found': False, 'response_time_ms': None, 'canonical_url': None
    }

    parsed_url = urlparse(url)
    base_url = f"{parsed_url.scheme}://{parsed_url.netloc}"
    data['is_https'] = parsed_url.scheme == 'https'

    try:
        response = await client.get_response(url, follow_redirects=True)
        data['status_code'] = response.status_code
        data['response_time_ms'] = int(response.elapsed.total_seconds() * 1000)
        data['total_pages_crawled'] = 1

        soup = BeautifulSoup(response.text, 'html.parser')

        basic_data = _parse_basic_seo_from_soup(soup, base_url)
        data['canonical_url'] = basic_data.get('canonical_url')
        data['schema_markup_found'] = basic_data.get('schema_markup_found', False)

        robots_url = urljoin(base_url, '/robots.txt')
        sitemap_locations = set()

        try:
            robots_response = await client.head(robots_url, follow_redirects=True)
            if robots_response.status_code == 200:
                data['robots_txt_found'] = True
                sitemaps_from_robots = await _check_sitemap_in_robots(client, robots_url)
                for sitemap_url in sitemaps_from_robots:
                    sitemap_locations.add(sitemap_url)
        except Exception as e:
            log.debug("Kunne ikke tjekke robots.txt for %s: %s", url, e)

        default_sitemap_url = urljoin(base_url, '/sitemap.xml')
        try:
            sitemap_response = await client.head(default_sitemap_url, follow_redirects=True)
            if sitemap_response.status_code == 200:
                sitemap_locations.add(default_sitemap_url)
        except Exception as e:
            log.debug("Kunde ikke tjekke default sitemap for %s: %s", url, e)

        if sitemap_locations:
            data['sitemap_xml_found'] = True
            data['sitemap_locations'] = list(sitemap_locations)

        links_to_check = set()
        for link_tag in soup.find_all('a', href=True):
            href = link_tag['href']
            if href.startswith(('mailto:', 'tel:', '#', 'javascript:')):
                continue
            absolute_link = urljoin(base_url, href)
            links_to_check.add(absolute_link)

        data['total_links_found'] = len(links_to_check)

        if links_to_check:
            tasks = [client.head(link, follow_redirects=True) for link in links_to_check]
            results = await asyncio.gather(*tasks, return_exceptions=True)

            broken_links = []
            for i, result in enumerate(results):
                link = list(links_to_check)[i]
                if isinstance(result, Exception):
                    log.debug("Fejl ved tjek af link %s: %s", link, result)
                    broken_links.append(link)
                elif isinstance(result, httpx.Response) and result.status_code >= 400:
                    broken_links.append(link)

            data['broken_links_list'] = broken_links
            data['broken_links_count'] = len(broken_links)

    except Exception as e:
        log.error("fetch_technical_seo_data fejlede for %s: %s", url, e, exc_info=True)
        return data

    return data