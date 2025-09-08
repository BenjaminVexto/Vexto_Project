# src/vexto/scoring/website_fetchers.py

from __future__ import annotations

import asyncio
import json
import logging
import re
from os import getenv
from typing import Dict, Any, List, Tuple
from urllib.parse import urlparse, urljoin

import httpx
from bs4 import BeautifulSoup

from .schemas import BasicSEO, TechnicalSEO
from .http_client import AsyncHtmlClient

log = logging.getLogger(__name__)

# ---------- Parser med forbedret JSON-håndtering og fejlhåndtering -----------------------------
def _parse_basic_seo_from_soup(soup: BeautifulSoup, base_url: str | None = None) -> BasicSEO:
    """
    Udtrækker title, H1 (filtrerer skjulte/tomme H1), meta description,
    word_count, canonical (link + CMS state fallback) og schema presence.
    """
    if not soup:
        return {
            "h1": None,
            "h1_count": 0,
            "h1_texts": [],
            "meta_description": None,
            "meta_description_length": 0,
            "title_text": None,
            "title_length": 0,
            "word_count": 0,
            "canonical_url": None,
            "canonical_error": "No HTML content to parse",
            "schema_markup_found": False,
            "schema_types": [],
            "canonical_source": "none",
        }

    try:
        # --- H1 (filtrér skjulte/tomme) ---
        def _is_visually_hidden(el) -> bool:
            if el.get("aria-hidden") == "true":
                return True
            style = (el.get("style") or "").lower()
            return ("display:none" in style) or ("visibility:hidden" in style)

        visible_h1_texts: List[str] = []
        for h in soup.find_all("h1"):
            if _is_visually_hidden(h):
                continue
            txt = h.get_text(" ", strip=True)
            if txt:
                visible_h1_texts.append(txt)

        h1_count = len(visible_h1_texts)
        h1_texts = visible_h1_texts
        h1_first_text = visible_h1_texts[0] if visible_h1_texts else None

        # --- Meta description ---
        meta_description_text = None
        meta_tag = soup.find("meta", attrs={"name": lambda x: x and x.lower() == "description"})
        if meta_tag and "content" in meta_tag.attrs:
            content = meta_tag["content"]
            if isinstance(content, str):
                meta_description_text = content.strip()
            elif isinstance(content, (list, tuple)) and content:
                for item in content:
                    if isinstance(item, str) and item.strip():
                        meta_description_text = item.strip()
                        break
            elif content is not None:
                try:
                    meta_description_text = str(content).strip()
                except Exception as e:
                    log.warning(f"Failed to convert meta content: {e}")

        meta_description_length = len(meta_description_text) if meta_description_text else 0

        # --- Title ---
        title_text = None
        title_tag = soup.find("title")
        if title_tag:
            try:
                content = title_tag.get_text(separator=" ", strip=True)
                if isinstance(content, str):
                    title_text = content
                elif isinstance(content, (list, tuple)) and content:
                    for item in content:
                        if isinstance(item, str) and item.strip():
                            title_text = item.strip()
                            break
                elif content is not None:
                    try:
                        title_text = str(content).strip()
                    except Exception as e:
                        log.warning(f"Failed to convert title content: {e}")
            except Exception as e:
                log.warning(f"Error extracting title text: {e}")

        title_length = len(title_text) if title_text else 0

        # --- Word count (robust) ---
        try:
            word_count = len(soup.get_text(" ", strip=True).split())
        except Exception as e:
            log.warning(f"Error calculating word count: {e}")
            word_count = 0

        # --- Canonical (link → CMS/runtime fallback) ---
        canonical_tags = soup.find_all("link", attrs={"rel": "canonical"})
        canonical_url = None
        canonical_error = None
        canonical_source = "none"

        if len(canonical_tags) == 1:
            href = canonical_tags[0].get("href")
            if href and isinstance(href, str):
                canonical_url = urljoin(base_url, href) if base_url else href
                canonical_source = "link"
        elif len(canonical_tags) > 1:
            canonical_error = "Multiple canonical tags found"
        else:
            # Fallback: Tjek __INITIAL_STATE__ for canonical-felter
            try:
                for script in soup.find_all("script"):
                    script_content = script.get_text() or ""
                    if "window.__INITIAL_STATE__" not in script_content:
                        continue
                    m = re.search(
                        r"window\.__INITIAL_STATE__\s*=\s*(\{[\s\S]*?\})(;|$)",
                        script_content,
                        re.DOTALL,
                    )
                    if not m:
                        continue
                    json_str = m.group(1)

                    # Rens JavaScript → JSON
                    js = json_str
                    js = re.sub(r",\s*}", "}", js)
                    js = re.sub(r",\s*]", "]", js)
                    js = re.sub(r":\s*undefined\b", ": null", js)
                    js = re.sub(r":\s*!0\b", ": true", js)
                    js = re.sub(r":\s*!1\b", ": false", js)
                    js = re.sub(r":\s*function\s*\([^)]*\)\s*\{[^}]*\}", ": null", js)

                    initial_state = json.loads(js)

                    # Find canonical felter rekursivt
                    canonicals: List[tuple[str, str]] = []

                    def _walk(obj, path=""):
                        if isinstance(obj, dict):
                            for k, v in obj.items():
                                p = f"{path}.{k}" if path else k
                                if k in ("custom_canonical_url", "canonical_url", "canonical") and v not in (None, "null", "", False):
                                    if isinstance(v, str) and v.strip():
                                        canonicals.append((v.strip(), p))
                                    elif isinstance(v, bool) and v:
                                        canonicals.append(((base_url or ""), p))
                                else:
                                    _walk(v, p)
                        elif isinstance(obj, list):
                            for i, item in enumerate(obj):
                                _walk(item, f"{path}[{i}]" if path else f"[{i}]")

                    _walk(initial_state)

                    for value, p in canonicals:
                        if value and value not in ("null", ""):
                            if base_url and (value == base_url):
                                canonical_url = value
                                canonical_source = "custom_self"
                            else:
                                canonical_url = urljoin(base_url, value) if base_url else value
                                canonical_source = "custom"
                            log.debug(f"Found custom canonical in state at {p}: {canonical_url}")
                            break
                    if canonical_url:
                        break
            except json.JSONDecodeError as e:
                log.warning(f"Failed to parse __INITIAL_STATE__ JSON: {e}")
            except Exception as e:
                log.warning(f"Unexpected error parsing __INITIAL_STATE__: {e}")

        if not canonical_url and not canonical_error:
            canonical_error = "No canonical tag or CMS canonical found"

        # --- Schema markup detection (JSON-LD + microdata) ---
        schema_markup_found = False
        schema_types: List[str] = []

        # NYT: separate flags
        schema_jsonld_valid = False     # mindst ét JSON-LD script parsed OK
        jsonld_repaired = False         # mindst ét script krævede "reparation" før parse
        schema_microdata_found = False  # Microdata til stede (itemscope)

        def _sanitize_jsonld(txt: str) -> str:
            # Fjern JS-kommentarer og trailing commas før json.loads
            txt = re.sub(r"/\*.*?\*/", "", txt or "", flags=re.S)
            txt = re.sub(r"//.*?$", "", txt, flags=re.M)
            txt = re.sub(r",\s*([}\]])", r"\1", txt)
            return txt

        # JSON-LD (lenient)
        jsonld_scripts = soup.find_all("script", attrs={"type": "application/ld+json"})
        for sc in jsonld_scripts:
            try:
                content = sc.string or sc.get_text() or ""
                if not content.strip():
                    continue
                sanitized = _sanitize_jsonld(content)
                if sanitized != (content or ""):
                    jsonld_repaired = True
                data = json.loads(sanitized)
                items = data if isinstance(data, list) else [data]
                for it in items:
                    if isinstance(it, dict) and "@type" in it:
                        t = it["@type"]
                        if isinstance(t, list):
                            schema_types.extend([str(x) for x in t])
                        else:
                            schema_types.append(str(t))
                        schema_markup_found = True
                        schema_jsonld_valid = True
            except json.JSONDecodeError as e:
                log.warning(f"Invalid JSON-LD script on {base_url}: {e}")
            except Exception as e:
                log.warning(f"Error parsing JSON-LD on {base_url}: {e}")

        # Microdata fallback (uændret logik, men sæt separat flag)
        if not schema_markup_found:
            if soup.find(attrs={"itemscope": True}):
                schema_markup_found = True
                schema_microdata_found = True
                schema_types.append("Microdata")
        else:
            # JSON-LD fandt noget; tjek stadig om microdata også er til stede
            if soup.find(attrs={"itemscope": True}):
                schema_microdata_found = True

        if schema_markup_found:
            log.info(f"Schema types found on {base_url}: {schema_types}")
        else:
            log.info(f"No schema markup found on {base_url}")

        # --- Result ---
        result: BasicSEO = {
            "h1": h1_first_text,
            "h1_count": h1_count,
            "h1_texts": h1_texts,
            "meta_description": meta_description_text,
            "meta_description_length": meta_description_length,
            "title_text": title_text,
            "title_length": title_length,
            "word_count": word_count,
            "canonical_url": canonical_url,
            "canonical_error": canonical_error,
            "schema_markup_found": schema_markup_found,
            "schema_types": schema_types,
            "canonical_source": canonical_source,
            # NYT: tydelige flags til rapport/scorer
            "schema_jsonld_valid": schema_jsonld_valid,
            "jsonld_repaired": jsonld_repaired,
            "schema_microdata_found": schema_microdata_found,
        }
        return result
    except Exception as e:
        log.error("Error during basic SEO parsing: %s", e, exc_info=True)
        return {
            "h1": None,
            "h1_count": 0,
            "h1_texts": [],
            "meta_description": None,
            "meta_description_length": 0,
            "title_text": None,
            "title_length": 0,
            "word_count": 0,
            "canonical_url": None,
            "canonical_error": "Parsing failed with exception",
            "schema_markup_found": False,
            "schema_types": [],
            "canonical_source": "none",
        }


# ---------- Offentlig wrapper -----------------------------------------------
def parse_basic_seo_from_soup(soup: BeautifulSoup, base_url: str | None = None) -> BasicSEO:
    """
    Offentlig wrapper omkring _parse_basic_seo_from_soup.
    Sender base_url til parsing for at håndtere relative canonical URLs.
    """
    return _parse_basic_seo_from_soup(soup, base_url)


# ---------- Hjælpere --------------------------------------------------------
ESSENTIAL_FIELDS = ("title_text", "h1_texts", "meta_description")


def _is_essential_missing(basic: BasicSEO) -> bool:
    return all(basic.get(f) in (None, "", []) for f in ESSENTIAL_FIELDS)


async def _check_sitemap_in_robots(client: AsyncHtmlClient, robots_url: str) -> List[str]:
    sitemaps: List[str] = []
    try:
        response = await client.get(robots_url, follow_redirects=True)
        if response.status_code == 200:
            lines = response.text.splitlines()
            for line in lines:
                if line.lower().startswith("sitemap:"):
                    sitemaps.append(line.split(":", 1)[1].strip())
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
    3) Returnér basic-dict + rå HTML-streng.
    """
    basic_final: Dict[str, Any] = {}
    html_out: str | None = None

    # --- Første forsøg: SSR / cache ---
    try:
        ssr_result = await client.get_raw_html(
            url,
            force=False,
            return_soup=False,
            force_playwright=False,
        )

        html_content_ssr = None
        if ssr_result is None:
            html_content_ssr = None
        elif isinstance(ssr_result, str):
            html_content_ssr = ssr_result
        elif isinstance(ssr_result, tuple) and len(ssr_result) == 2:
            html_content_ssr = ssr_result[0]
        elif isinstance(ssr_result, dict):
            html_content_ssr = ssr_result.get("html")
        else:
            log.warning(f"Unexpected SSR result format: {type(ssr_result)}")
            html_content_ssr = None
    except Exception as e:
        log.warning(f"SSR fetch failed for {url}: {e}")
        html_content_ssr = None

    soup_ssr = BeautifulSoup(html_content_ssr, "lxml") if html_content_ssr else None
    basic_ssr = _parse_basic_seo_from_soup(soup_ssr, url) if soup_ssr else {}
    html_out = html_content_ssr

    _fast = getenv("VEXTO_FAST_MODE", "0") == "1"
    need_render = (
        (html_content_ssr is None)
        or (_is_essential_missing(basic_ssr) and not (_fast and basic_ssr.get("title_text") and basic_ssr.get("h1_texts")))
    )

    # Initialiser med SSR data
    basic_final = basic_ssr.copy() if basic_ssr else {}

    if need_render:
        log.info(f"Eskalér {url} til Playwright (dynamic / mangler essentials).")
        try:
            playwright_result = await client.get_raw_html(
                url,
                force_playwright=True,
                return_soup=False,
                force=False,
            )

            rendered_html = None
            pw_canonical_data: Dict[str, Any] = {}
            if playwright_result is None:
                rendered_html = None
            elif isinstance(playwright_result, str):
                rendered_html = playwright_result
            elif isinstance(playwright_result, tuple) and len(playwright_result) == 2:
                rendered_html, pw_canonical_data = playwright_result
            elif isinstance(playwright_result, dict):
                rendered_html = playwright_result.get("html")
                pw_canonical_data = playwright_result.get("canonical_data", {}) or {}
            else:
                log.warning(f"Unexpected Playwright result format: {type(playwright_result)}")
                rendered_html = None

        except Exception as e:
            log.error(f"Playwright fetch failed for {url}: {e}")
            rendered_html = None
            pw_canonical_data = {}

        if rendered_html:
            soup_playwright = BeautifulSoup(rendered_html, "lxml")
            basic_playwright = _parse_basic_seo_from_soup(soup_playwright, url)
            html_out = rendered_html

            # Merge data - prioriter Playwright for ikke-tomme værdier
            for field, pw_value in basic_playwright.items():
                if field == "canonical_source":
                    continue
                if pw_value not in (None, "", [], 0):
                    basic_final[field] = pw_value

            # Canonical-prioritet
            canonical_url_final = basic_final.get("canonical_url")
            canonical_source_final = basic_final.get("canonical_source", "none")

            if basic_playwright.get("canonical_url"):
                canonical_url_final = basic_playwright["canonical_url"]
                canonical_source_final = basic_playwright.get("canonical_source", "playwright_parsed")
                log.info(f"Using Playwright parsed canonical: {canonical_url_final}")
            elif isinstance(pw_canonical_data, dict) and pw_canonical_data:
                # link rel=canonical i runtime
                link_can = pw_canonical_data.get("link_canonical")
                if link_can:
                    canonical_url_final = urljoin(url, link_can)
                    canonical_source_final = "playwright_link"
                    log.info(f"Using Playwright link canonical: {canonical_url_final}")
                else:
                    # Felter fra runtime-state
                    for path, value in (pw_canonical_data.get("canonical_fields") or {}).items():
                        if value and str(value).strip().lower() not in ("none", "null", "", "false"):
                            canonical_url_final = urljoin(url, str(value)) if str(value).lower() != "true" else url
                            canonical_source_final = "playwright_runtime_state"
                            log.info(f"Using Playwright runtime canonical: {canonical_url_final} from {path}")
                            break

            if not canonical_url_final and basic_ssr.get("canonical_url"):
                canonical_url_final = basic_ssr["canonical_url"]
                canonical_source_final = basic_ssr.get("canonical_source", "ssr_fallback")
                log.info(f"Using SSR fallback canonical: {canonical_url_final}")

            basic_final["canonical_url"] = canonical_url_final
            basic_final["canonical_source"] = canonical_source_final
            basic_final["canonical_error"] = None if canonical_url_final else "No canonical found after all attempts"

    else:
        log.info(f"Using SSR data for {url} (essentials found)")

    # Sikr at alle nødvendige felter findes
    required_fields = {
        "h1": None,
        "h1_count": 0,
        "h1_texts": [],
        "meta_description": None,
        "meta_description_length": 0,
        "title_text": None,
        "title_length": 0,
        "word_count": 0,
        "canonical_url": None,
        "canonical_error": None,
        "schema_markup_found": False,
        "schema_types": [],
        "canonical_source": "none",
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
        "status_code": None,
        "is_https": False,
        "robots_txt_found": False,
        "sitemap_xml_found": False,
        "sitemap_locations": [],
        "broken_links_count": 0,
        "broken_links_list": [],
        "total_pages_crawled": 0,
        "total_links_found": 0,
        "schema_markup_found": False,
        "response_time_ms": None,
        "canonical_url": None,
        # NYT: render-telemetri
        "render_status": None,                # "content" | "empty"
        "rendered_content_length": 0,         # int (bytes-ish via len(html))
        "soft_404_suspected": False,          # bool
    }

    parsed_url = urlparse(url)
    base_url = f"{parsed_url.scheme}://{parsed_url.netloc}"
    data["is_https"] = parsed_url.scheme == "https"

    try:
        response = await client.get_response(url, follow_redirects=True)
        data["status_code"] = response.status_code
        data["response_time_ms"] = int(response.elapsed.total_seconds() * 1000)
        data["total_pages_crawled"] = 1

        html_text = response.text or ""
        data["rendered_content_length"] = len(html_text)
        data["render_status"] = "content" if data["rendered_content_length"] > 0 else "empty"
        # soft 404: 404 m. reelt indhold (threshold kan justeres)
        _SOFT_404_LEN = 500
        data["soft_404_suspected"] = (data["status_code"] == 404 and data["rendered_content_length"] > _SOFT_404_LEN)

        soup = BeautifulSoup(html_text, "html.parser")

        basic_data = _parse_basic_seo_from_soup(soup, base_url)
        data["canonical_url"] = basic_data.get("canonical_url")
        data["schema_markup_found"] = basic_data.get("schema_markup_found", False)

        robots_url = urljoin(base_url, "/robots.txt")
        sitemap_locations = set()

        try:
            robots_response = await client.head(robots_url, follow_redirects=True)
            if isinstance(robots_response, httpx.Response) and robots_response.status_code == 200:
                data["robots_txt_found"] = True
                sitemaps_from_robots = await _check_sitemap_in_robots(client, robots_url)
                for sitemap_url in sitemaps_from_robots:
                    sitemap_locations.add(sitemap_url)
        except Exception as e:
            log.debug("Kunne ikke tjekke robots.txt for %s: %s", url, e)

        sitemap_paths = ["/sitemap.xml", "/sitemap_index.xml", "/post-sitemap.xml", "/page-sitemap.xml", "/category-sitemap.xml"]
        for spath in sitemap_paths:
            sm_url = urljoin(base_url, spath)
            try:
                sitemap_response = await client.head(sm_url, follow_redirects=True)
                ok = isinstance(sitemap_response, httpx.Response) and sitemap_response.status_code == 200
                if not ok:
                    # Fallback: GET hvis HEAD ikke støttes
                    sitemap_get = await client.get(sm_url, follow_redirects=True)
                    ok = isinstance(sitemap_get, httpx.Response) and sitemap_get.status_code == 200
                if ok:
                    sitemap_locations.add(sm_url)
            except Exception as e:
                log.debug("Kunne ikke tjekke sitemap for %s: %s", sm_url, e)

        if sitemap_locations:
            data["sitemap_xml_found"] = True
            data["sitemap_locations"] = list(sitemap_locations)

        # --- Link-udtræk & normalisering ---
        links_to_check: set[str] = set()
        discarded = 0
        discarded_links: List[Dict[str, str]] = []

        def _push_discard(href: str, abs_url: str, reason: str) -> None:
            nonlocal discarded
            discarded += 1
            if len(discarded_links) < 20:
                discarded_links.append({"href": href, "abs": abs_url, "reason": reason})

        for link_tag in soup.find_all("a", href=True):
            raw_href = link_tag.get("href") or ""
            href = raw_href.strip()
            if not href:
                _push_discard(raw_href, "", "empty")
                continue

            # Autorepair: 'ttps://...' -> 'https://...'
            if href.lower().startswith("ttps://"):
                href = "h" + href  # præfikser 'h'

            # Drop åbenlyst ubrugelige skemaer/fragmenter
            if href.startswith(("#", "mailto:", "tel:", "javascript:", "data:", "callto:")):
                if href and not href.startswith("#"):
                    log.debug("[links][discard] href=%s reason=non_http_schema", href)
                _push_discard(href, "", "non_http_schema_or_fragment")
                continue

            # Absolutér og valider
            absolute_url = urljoin(base_url, href)
            try:
                parsed = urlparse(absolute_url)
            except Exception:
                log.debug("[links][discard] href=%s abs=%s reason=parse_error", href, absolute_url)
                _push_discard(href, absolute_url, "parse_error")
                continue

            if parsed.scheme not in ("http", "https"):
                log.debug("[links][discard] href=%s abs=%s reason=invalid_scheme", href, absolute_url)
                _push_discard(href, absolute_url, "invalid_scheme")
                continue
            if not parsed.netloc:
                log.debug("[links][discard] href=%s abs=%s reason=no_host", href, absolute_url)
                _push_discard(href, absolute_url, "no_host")
                continue
            # Domæne-fragmenter (fx netloc der starter med ".")
            if parsed.netloc.startswith(".") or absolute_url.startswith("/."):
                log.debug("[links][discard] href=%s abs=%s reason=domain_fragment", href, absolute_url)
                _push_discard(href, absolute_url, "domain_fragment")
                continue

            # Normaliseret URL tilbage i sættet (uden query for at reducere dubletter)
            links_to_check.add(parsed.geturl().split("?", 1)[0])

        data["total_links_found"] = len(links_to_check)
        if discarded:
            log.debug("[links] discarded_count=%d", discarded)
        if discarded_links:
            data["discarded_links"] = discarded_links  # kræver felt i schemas (Patch 15)

        # --- Broken link check: HEAD → GET fallback ---
        broken_links: List[str] = []
        if links_to_check:
            tasks = [client.head(link, follow_redirects=True) for link in links_to_check]
            results = await asyncio.gather(*tasks, return_exceptions=True)

            links_seq = list(links_to_check)
            for i, result in enumerate(results):
                link = links_seq[i]
                try:
                    if isinstance(result, Exception):
                        # Fallback: prøv GET
                        g = await client.get(link, follow_redirects=True)
                        if not isinstance(g, httpx.Response) or g.status_code >= 400:
                            broken_links.append(link)
                    elif isinstance(result, httpx.Response):
                        if result.status_code >= 400:
                            # Fallback: GET hvis f.eks. 405 Method Not Allowed på HEAD
                            g = await client.get(link, follow_redirects=True)
                            if not isinstance(g, httpx.Response) or g.status_code >= 400:
                                broken_links.append(link)
                        # ellers OK
                    else:
                        # Ukendt svar-type → konservativt tjek med GET
                        g = await client.get(link, follow_redirects=True)
                        if not isinstance(g, httpx.Response) or g.status_code >= 400:
                            broken_links.append(link)
                except Exception as e:
                    log.debug("Fejl ved tjek af link %s (fallback GET): %s", link, e)
                    broken_links.append(link)

        data["broken_links_count"] = len(broken_links)
        data["broken_links_list"] = broken_links

    except Exception as e:
        log.error("fetch_technical_seo_data fejlede for %s: %s", url, e, exc_info=True)
        return data

    return data
