# src/vexto/scoring/analyzer.py

import asyncio
import logging
import re
import json
import os.path as op
from datetime import timezone
import datetime as _dt
from typing import List, Optional, Dict, Any, Tuple
from bs4 import BeautifulSoup
from urllib.parse import urlparse, urljoin
from collections import Counter
from vexto.utils.paths import deep_get as _deep_get


from .contact_fetchers import find_contact_info, detect_forms
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

# -------------------- DEFAULTS --------------------
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
    'gmb_review_count': None,
    'gmb_average_rating': None,
    'gmb_profile_complete': False,  # bool (ikke None/0)
    'social_media_links': [],
    'social_share_links': []
}

DEFAULT_AUTHORITY_METRICS: AuthorityMetrics = {
    'domain_authority': 0,
    'page_authority': 0,
    'global_rank': None,
    'authority_status': "not_run",
    'gmb_profile_complete': False  # til scoring_rules.yml -> authority.gmb_profile
}

DEFAULT_PERFORMANCE_METRICS: PerformanceMetrics = {
    'lcp_ms': None, 'cls': None, 'inp_ms': None,
    'viewport_score': 0,
    'performance_score': 0,
    'psi_status': "not_run", 'total_js_size_kb': 0, 'js_file_count': 0
}

DEFAULT_CONTENT_METRICS: ContentMetrics = {
    'latest_post_date': None,
    'days_since_last_post': None,   # bevares som tal – fallback sættes senere til 999
    'keywords_in_content': {},
    'internal_link_score': 0,
    'keyword_relevance_score': 0,
    'average_word_count': 0
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

def log_metric_status(metric: str, value: Any, status: str = "ok"):
    log.info(f"[{status}] {metric}: {value}")

# -------------------- HELPERS --------------------

ASSET_EXTENSIONS = (
    ".jpg",".jpeg",".png",".gif",".webp",".svg",".ico",
    ".css",".js",".pdf",".xml",".mp4",".webm",".woff",".woff2",".ttf",".eot",".zip",".doc",".docx"
)
CDN_BLOCKED_PREFIXES = ("m2.","cdn.","media.","assets.")

def is_asset_url(u: str) -> bool:
    if not u:
        return False
    base = u.split("?", 1)[0].lower()
    return base.endswith(ASSET_EXTENSIONS) or "/img/160/160/resize/" in base or "/img/360/360/resize/" in base

def _normalize_url_for_compare(u: str) -> tuple[str, str, str]:
    try:
        p = urlparse(u)
        path = p.path or "/"
        if path != "/" and path.endswith("/"):
            path = path.rstrip("/")
        return (p.scheme.lower(), p.netloc.lower(), path)
    except Exception:
        return ("", "", u or "")

def _urls_equivalent(a: str, b: str) -> bool:
    return _normalize_url_for_compare(a) == _normalize_url_for_compare(b)

def _normalize_canonical_out(u: Optional[str]) -> Optional[str]:
    """
    Normaliser endelig canonical til stabil output:
    - Fjern trailing slash på roden (https://domæne/ -> https://domæne)
    - Behold query hvis det findes
    - Dropper default-port i netloc
    """
    if not u:
        return u
    try:
        p = urlparse(u)
        scheme = (p.scheme or "https").lower()
        host = (p.hostname or "").lower()
        netloc = host
        if p.port and not ((scheme == "http" and p.port == 80) or (scheme == "https" and p.port == 443)):
            netloc = f"{host}:{p.port}"
        path = p.path or "/"
        if path == "/":
            path = ""  # fjern trailing slash på roden
        query = f"?{p.query}" if p.query else ""
        return f"{scheme}://{netloc}{path}{query}"
    except Exception:
        return u

def detect_schema_microdata(html: str) -> tuple[bool, list[str]]:
    try:
        soup = BeautifulSoup(html or "", "html.parser")
        types = []
        for el in soup.find_all(attrs={"itemscope": True}):
            t = el.get("itemtype") or ""
            if isinstance(t, str) and t.strip():
                # itemtype kan være fuld URL – tag sidste led
                last = t.strip().split("/")[-1]
                if last:
                    types.append(last)
        return (len(types) > 0, sorted(set(types)))
    except Exception:
        return (False, [])

def is_blocked_cdn(u: str) -> bool:
    try:
        host = urlparse(u).netloc.lower()
    except Exception:
        return False
    return any(host.startswith(p) for p in CDN_BLOCKED_PREFIXES)

def should_accept_runtime_canonical(source_key: str) -> bool:
    """
    Afvis nav/menu-afledte 'canonicals' fra global state.
    Tillad velkendte page-/produkt-/kategori-felder.
    """
    key = (source_key or "").lower()
    reject = ("menucategories", "categoriesmap", "navigation", "menu", "header", "footer", "sidebar")
    if any(m in key for m in reject):
        return False
    accept = ("product.current", "category.current", "page.current", "canonical_primary_url", "custom_canonical_url", "link_canonical")
    return any(a in key for a in accept)

def _extract_runtime_state_from_html(html: str) -> Optional[dict]:
    """Prøv at finde __INITIAL_STATE__ i scripts."""
    soup = BeautifulSoup(html or "", "html.parser")
    for s in soup.find_all("script"):
        txt = (s.string or s.get_text() or "").strip()
        if "__INITIAL_STATE__" in txt:
            try:
                start = txt.find("{")
                end = txt.rfind("}")
                if start != -1 and end != -1 and end > start:
                    return json.loads(txt[start:end+1])
            except Exception:
                pass
    return None

def _local_deep_get(obj, dotted_path):
    cur = obj
    for key in dotted_path.split('.'):
        if isinstance(cur, dict):
            cur = cur.get(key)
        elif isinstance(cur, list) and key.isdigit():
            idx = int(key)
            cur = cur[idx] if 0 <= idx < len(cur) else None
        else:
            return None
        if cur is None:
            return None
    return cur

def detect_schema_from_runtime(html_content: str, canonical_data: dict):
    """
    1) Prøv eksisterende JSON-LD i HTML
    2) Kig i Nuxt runtime_state.*.script[].innerHTML for ld+json
    3) Fald tilbage til Microdata/RDFa (hvis du har helper), ellers OG-inferens
    Return: (found, types, structured)
    """
    types, structured = [], {}

    # 1) HTML → JSON-LD
    try:
        soup = BeautifulSoup(html_content or "", "html.parser")
        for tag in soup.find_all("script", type=lambda v: v and "ld+json" in str(v).lower()):
            raw = (tag.string or tag.get_text() or "").strip()
            if not raw:
                continue
            try:
                data = json.loads(raw)
            except Exception:
                continue
            bucket = data if isinstance(data, list) else [data]
            for node in bucket:
                if isinstance(node, dict) and node.get("@type"):
                    t = node["@type"]
                    if isinstance(t, list): types.extend(map(str, t))
                    else: types.append(str(t))
                    structured[str(t)] = node
        if types:
            return True, sorted(set(types)), structured
    except Exception:
        pass

    # 2) Nuxt runtime_state → head.script[].innerHTML med ld+json
    try:
        rs = canonical_data.get('runtime_state', {}) if isinstance(canonical_data, dict) else {}
        schema_paths = [
            "head.script", "seo.schema", "page.schema", "meta.schema",
            "__schema", "structuredData", "nuxtI18n.head.script", "page.head.script"
        ]
        for path in schema_paths:
            val = _local_deep_get(rs, path)
            if not val:
                continue
            items = val if isinstance(val, list) else [val]
            for it in items:
                inner = None
                if isinstance(it, dict):
                    inner = it.get('innerHTML') or it.get('innerHtml') or it.get('text') or None
                elif isinstance(it, str):
                    inner = it
                if not inner:
                    continue
                inner = inner.strip()
                try:
                    data = json.loads(inner)
                except Exception:
                    continue
                bucket = data if isinstance(data, list) else [data]
                for node in bucket:
                    if isinstance(node, dict) and node.get("@type"):
                        t = node["@type"]
                        if isinstance(t, list): types.extend(map(str, t))
                        else: types.append(str(t))
                        structured[str(t)] = node
        if types:
            return True, sorted(set(types)), structured
    except Exception:
        pass

    # 3) (Valgfrit) Microdata/RDFa, hvis du har en helper
    try:
        if 'detect_schema_microdata' in globals():
            md_found, md_types = detect_schema_microdata(html_content or "")
            if md_found and md_types:
                return True, sorted(set(md_types)), {}
    except Exception:
        pass

    # 4) OG/meta-inferens (sidste udvej)
    try:
        soup = BeautifulSoup(html_content or "", "html.parser")
        og_type = soup.find("meta", property="og:type")
        if og_type and og_type.get("content") in ("website", "product", "article"):
            mapping = {"website": "WebSite", "product": "Product", "article": "Article"}
            return True, [mapping[og_type.get("content")]], {}
        if soup.find("meta", attrs={"property": "og:site_name"}) or soup.find(string=re.compile(r"CVR|VAT|Copyright", re.I)):
            return True, ["Organization"], {}
    except Exception:
        pass

    return False, [], {}

def _search_nested_urls(obj, key_patterns=None, include_non_http=True, yield_paths=False, _path=()):
    """
    DFS i vilkårlige dict/list-strukturer.
    - key_patterns: liste af substrings (lowercase) der SKAL indgå i nøglen for at yield'e (narrow-mode).
    - include_non_http: hvis True, accepteres også "/path" som vi kan urljoine.
    - yield_paths: hvis True, yield'er (value, "a.b[2].c") til debug.
    """
    def key_ok(k: str) -> bool:
        if key_patterns is None:
            return True
        k = str(k).lower()
        return any(p in k for p in key_patterns)

    if isinstance(obj, dict):
        for k, v in obj.items():
            p = (*_path, k)
            if isinstance(v, (dict, list, tuple)):
                yield from _search_nested_urls(v, key_patterns, include_non_http, yield_paths, p)
            else:
                if key_ok(k) and isinstance(v, str):
                    sv = v.strip()
                    if not sv or sv.lower() in ("none", "null", "false"):
                        continue
                    if sv.startswith("http://") or sv.startswith("https://"):
                        yield (sv, ".".join(map(str, p))) if yield_paths else sv
                    elif include_non_http and sv.startswith("/"):
                        yield (sv, ".".join(map(str, p))) if yield_paths else sv
    elif isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            p = (*_path, i)
            if isinstance(v, (dict, list, tuple)):
                yield from _search_nested_urls(v, key_patterns, include_non_http, yield_paths, p)
            else:
                if isinstance(v, str):
                    sv = v.strip()
                    if not sv or sv.lower() in ("none", "null", "false"):
                        continue
                    if sv.startswith("http://") or sv.startswith("https://"):
                        yield (sv, ".".join(map(str, p))) if yield_paths else sv
                    elif include_non_http and sv.startswith("/"):
                        yield (sv, ".".join(map(str, p))) if yield_paths else sv

def _is_valid_canonical(candidate: str, page_url: str) -> bool:
    """Samme host, og rimelig sti-match med den aktuelle side."""
    try:
        cu = urlparse(page_url)
        ca = urlparse(urljoin(page_url, candidate))

        if not ca.scheme or not ca.netloc:
            return False
        if (ca.scheme, ca.netloc) != (cu.scheme, cu.netloc):
            return False

        if ca.path == cu.path:
            return True

        ca_parts = [p for p in ca.path.strip("/").split("/") if p]
        cu_parts = [p for p in cu.path.strip("/").split("/") if p]
        shared = 0
        for a, b in zip(ca_parts, cu_parts):
            if a == b:
                shared += 1
            else:
                break
        return shared >= 1
    except Exception:
        return False

def resolve_canonical_enhanced(base_url: str, rendered_html: str, runtime_state: Optional[dict]) -> Tuple[Optional[str], str]:
    """Enhanced canonical resolution med bedre fallbacks.
    Prioritet:
      1) <link rel="canonical">
      2) Runtime state (kendte felter)
      2.5) og:url / twitter:url
      3) self-canonical
    Return: (canonical_url, source)
    """
    # 1) <link rel="canonical">
    if rendered_html:
        soup = BeautifulSoup(rendered_html, "html.parser")
        canonical_link = soup.find("link", rel=lambda v: v and "canonical" in v.lower())
        if canonical_link and canonical_link.get("href"):
            href = canonical_link.get("href").strip()
            if href.startswith("http"):
                return href, "dom_link"
            return urljoin(base_url, href), "dom_link"

    # 2) Runtime state
    state = runtime_state if runtime_state is not None else _extract_runtime_state_from_html(rendered_html)
    if state:
        candidate_paths = [
            # Nuxt/Vue patterns
            "category-next.menuCategories.0.custom_canonical_url",
            "category-next.menuCategories.1.custom_canonical_url",
            "category-next.menuCategories.2.custom_canonical_url",
            "category.current_path.0.custom_canonical_url",
            "category.list.0.custom_canonical_url",
            "category-next.categoriesMap.0.custom_canonical_url",
            "category-next.categoriesMap.1.custom_canonical_url",
            "category-next.categoriesMap.2.custom_canonical_url",
            # Generic patterns
            "page.canonical", "meta.canonical", "seo.canonical", "head.canonical", "route.canonical",
            # WordPress/WooCommerce patterns
            "yoast.canonical", "seo.canonical_url",
            # Shopify patterns
            "template.canonical_url", "current.canonical_url",
        ]
        for path in candidate_paths:
            value = _deep_get(state, path)
            if isinstance(value, str) and value.strip():
                canonical = value.strip()
                if canonical.startswith("http"):
                    return canonical, f"runtime_{path}"
                if canonical.startswith("/"):
                    return urljoin(base_url, canonical), f"runtime_{path}"
                return urljoin(base_url, "/" + canonical), f"runtime_{path}"

    # 2.5) Fallback via OG/Twitter meta (hvis ingen DOM-link og ingen runtime-hit)
    if rendered_html:
        soup = BeautifulSoup(rendered_html, "html.parser")
        og = soup.find("meta", attrs={"property": "og:url"})
        tw = soup.find("meta", attrs={"name": "twitter:url"})
        og_url = (og.get("content").strip() if og and og.get("content") else None)
        tw_url = (tw.get("content").strip() if tw and tw.get("content") else None)

        for cand, src in ((og_url, "og_url"), (tw_url, "twitter_url")):
            if not cand:
                continue
            # Acceptér fuld URL eller relativ → join med base
            cand_url = cand if cand.startswith("http") else urljoin(base_url, cand)
            # Filtrér åbenlyst forkerte cross-domain canonicals
            try:
                bu = urlparse(base_url); cu = urlparse(cand_url)
                if (bu.scheme, bu.netloc) == (cu.scheme, cu.netloc):
                    return cand_url, src
            except Exception:
                # Hvis parsing fejler, prøv stadig at returnere best-effort join
                return urljoin(base_url, cand), src

    # 3) Self-canonical (sidste udvej)
    parsed = urlparse(base_url)
    clean_url = f"{parsed.scheme}://{parsed.netloc}{parsed.path}"
    if clean_url.endswith("/index.html"):
        clean_url = clean_url.replace("/index.html", "/")
    return clean_url, "self_canonical"

# --- Schema detection (enhanced) ---
def detect_schema_enhanced(rendered_html: str, url: str = "") -> Tuple[bool, List[str], Dict[str, Any]]:
    """
    Enhanced schema detection med bedre håndtering af forskellige schema typer.
    Returns: (found: bool, types: List[str], structured_data: Dict)
    """
    types: List[str] = []
    found = False
    structured_data = {}
    
    if not rendered_html:
        return False, [], {}
    
    soup = BeautifulSoup(rendered_html, "html.parser")
    
    # 1. JSON-LD Detection (most common)
    json_ld_scripts = soup.find_all("script", type=lambda v: v and "ld+json" in v.lower())
    
    for script_tag in json_ld_scripts:
        try:
            # Get script content
            script_content = script_tag.string or script_tag.get_text()
            if not script_content:
                continue
            
            # Clean up the content
            script_content = script_content.strip()
            
            # Try to parse JSON (robust med sanitation + greedy JSON-udtræk)
            try:
                data = json.loads(script_content)
            except json.JSONDecodeError:
                # 1) Fjern BOM/zero-width chars + trim
                cleaned = script_content.replace('\u200b', '').replace('\ufeff', '').strip()
                # fjern evt. afsluttende semikolon fra inlined JS
                if cleaned.endswith(';'):
                    cleaned = cleaned[:-1]
                
                # 2) Hvis strengen ikke starter med { eller [, forsøg at udtrække "bredeste" JSON-blok
                candidate = cleaned
                if not (candidate.startswith('{') or candidate.startswith('[')):
                    lb_obj, rb_obj = candidate.find('{'), candidate.rfind('}')
                    lb_arr, rb_arr = candidate.find('['), candidate.rfind(']')
                    spans = []
                    if lb_obj != -1 and rb_obj != -1 and rb_obj > lb_obj:
                        spans.append((rb_obj - lb_obj, lb_obj, rb_obj + 1))
                    if lb_arr != -1 and rb_arr != -1 and rb_arr > lb_arr:
                        spans.append((rb_arr - lb_arr, lb_arr, rb_arr + 1))
                    if spans:
                        _, s, e = max(spans)  # vælg bredeste spænd
                        candidate = candidate[s:e]
                    try:
                        data = json.loads(candidate)
                    except Exception as e:
                        log.debug(f"Failed to parse JSON-LD: {e}")
                        continue

            # Parsing lykkedes
            found = True

            # Handle different JSON-LD structures (uændret logik)
            if isinstance(data, list):
                for item in data:
                    if isinstance(item, dict):
                        schema_type = item.get("@type")
                        if schema_type:
                            if isinstance(schema_type, list):
                                types.extend([str(t) for t in schema_type])
                            else:
                                types.append(str(schema_type))
                        if "@type" in item:
                            type_key = item["@type"] if isinstance(item["@type"], str) else item["@type"][0]
                            structured_data[type_key] = item
            elif isinstance(data, dict):
                schema_type = data.get("@type")
                if schema_type:
                    if isinstance(schema_type, list):
                        types.extend([str(t) for t in schema_type])
                    else:
                        types.append(str(schema_type))
                
                if "@graph" in data:
                    for graph_item in data["@graph"]:
                        if isinstance(graph_item, dict) and "@type" in graph_item:
                            graph_type = graph_item["@type"]
                            if isinstance(graph_type, list):
                                types.extend([str(t) for t in graph_type])
                            else:
                                types.append(str(graph_type))
                
                if "@type" in data:
                    type_key = data["@type"] if isinstance(data["@type"], str) else data["@type"][0]
                    structured_data[type_key] = data
        except Exception as e:
            log.debug(f"Error processing JSON-LD script: {e}")
            continue
    
    # 2. Microdata Detection
    if not found:
        # Check for itemscope/itemtype attributes
        microdata_elements = soup.find_all(attrs={"itemscope": True})
        if microdata_elements:
            found = True
            for element in microdata_elements:
                itemtype = element.get("itemtype")
                if itemtype:
                    # Extract type from schema.org URL
                    if "schema.org/" in itemtype:
                        type_name = itemtype.split("schema.org/")[-1]
                        types.append(type_name)
    
    # 3. RDFa Detection  
    if not found:
        rdfa_elements = soup.find_all(attrs={"typeof": True})
        if rdfa_elements:
            found = True
            for element in rdfa_elements:
                typeof = element.get("typeof")
                if typeof:
                    types.append(typeof)
    
    # 4. Check for common e-commerce/CMS schema patterns
    if not found:
        # WooCommerce, Shopify, Magento patterns
        commerce_indicators = [
            soup.find(attrs={"class": lambda x: x and "schema" in x.lower()}),
            soup.find(attrs={"data-schema": True}),
            soup.find("meta", attrs={"property": lambda x: x and x.startswith("product:")})
        ]
        
        if any(commerce_indicators):
            found = True
            types.append("Product")  # Assume product schema for e-commerce sites
    
    # Remove duplicates and sort
    types = sorted(set(types))
    
    # Log results
    if found:
        log.info(f"Schema markup found on {url}: Types={types}")
    else:
        log.debug(f"No schema markup found on {url}")
    
    return found, types, structured_data

def detect_schema_for_nuxt(rendered_html: str, runtime_state: Optional[dict] = None) -> Tuple[bool, List[str], Dict[str, Any]]:
    """
    Ekstra schema-detektion rettet mod Nuxt/Vue.
    Finder JSON-LD i script-tags, forsøg på @graph, og kigger i runtime_state stier.
    """
    types: List[str] = []
    found = False
    structured: Dict[str, Any] = {}

    if not rendered_html:
        return False, [], {}

    soup = BeautifulSoup(rendered_html, "html.parser")

    # 1) Alm. JSON-LD i script-tags
    for s in soup.find_all("script", type=lambda v: v and "ld+json" in str(v).lower()):
        try:
            raw = (s.string or s.get_text() or "").strip()
            if not raw:
                continue
            # Fjern evt. CDATA wrappers
            raw = re.sub(r'^//<!\[CDATA\[|\]\]>$', '', raw)
            data = json.loads(raw)

            def _collect(d: dict):
                nonlocal found
                if "@type" in d:
                    t = d["@type"]
                    if isinstance(t, list):
                        types.extend([str(x) for x in t])
                        structured[str(t[0])] = d
                    else:
                        types.append(str(t))
                        structured[str(t)] = d
                    found = True

            if isinstance(data, dict):
                _collect(data)
                if "@graph" in data and isinstance(data["@graph"], list):
                    for item in data["@graph"]:
                        if isinstance(item, dict):
                            _collect(item)
            elif isinstance(data, list):
                for item in data:
                    if isinstance(item, dict):
                        _collect(item)
        except Exception:
            continue

    # 2) Runtime state (Nuxt/Vue)
    if runtime_state:
        candidate_paths = [
            "head.script", "seo.schema", "page.schema", "meta.schema",
            "__schema", "structuredData", "app.head.script"
        ]
        for p in candidate_paths:
            val = _deep_get(runtime_state, p)
            if not val:
                continue
            try:
                if isinstance(val, list):
                    for it in val:
                        inner = None
                        if isinstance(it, dict) and 'innerHTML' in it:
                            inner = it.get('innerHTML')
                        elif isinstance(it, str):
                            inner = it
                        if inner:
                            data = json.loads(inner)
                            if isinstance(data, dict) and "@type" in data:
                                t = data["@type"]
                                types.append(t if isinstance(t, str) else str(t[0]))
                                structured[str(t if isinstance(t, str) else t[0])] = data
                                found = True
                elif isinstance(val, str):
                    data = json.loads(val)
                    if isinstance(data, dict) and "@type" in data:
                        t = data["@type"]
                        types.append(t if isinstance(t, str) else str(t[0]))
                        structured[str(t if isinstance(t, str) else t[0])] = data
                        found = True
            except Exception:
                continue

    types = sorted(set(types))
    return found, types, structured


def _forms_fallback_from_dom(soup: BeautifulSoup) -> List[int]:
    """Fallback-detektor for formularer på dynamiske/JS-heavy sider.
    Finder både <form> og 'form-lignende' containere (role, class/id/aria-mønstre)
    og tæller synlige felter (input/textarea/select). Returnerer en liste
    med felt-antal pr. fundet formular/sektion.
    """
    if not soup:
        return []

    import re

    HIDDEN_CLASS_RX = re.compile(r"(hidden|sr-only|visually-hidden|d-none|is-hidden|v-hidden)", re.I)
    CTA_TEXT_RX = re.compile(
        r"(send|submit|gem|næste|next|fortsæt|continue|tilmeld|apply|request|book|bestil|kontakt|sign\s*up|signup)",
        re.I
    )
    CONTAINER_RX = re.compile(
        r"(form|kontakt|contact|checkout|subscribe|newsletter|tilmeld|signup|sign[-_ ]?up|lead|booking|book|quote|request|forespørg|bestil|support)",
        re.I
    )

    def _is_node_hidden(node) -> bool:
        # type="hidden"
        t = (node.get("type") or "").lower()
        if t == "hidden":
            return True
        # attribute/aria
        if node.has_attr("hidden") or (node.get("aria-hidden") or "").lower() in ("true", "1"):
            return True
        # inline style
        style = (node.get("style") or "").lower().replace(" ", "")
        if "display:none" in style or "visibility:hidden" in style or "opacity:0" in style:
            return True
        # class-baseret
        classes = " ".join(node.get("class", [])).lower()
        if HIDDEN_CLASS_RX.search(classes or ""):
            return True
        # tjek for skjulte forfædre
        parent = node.parent
        hops = 0
        while parent is not None and hops < 6:  # begræns dybde
            if _has_hidden_marker(parent):
                return True
            parent = parent.parent
            hops += 1
        return False

    def _has_hidden_marker(el) -> bool:
        if el.has_attr("hidden") or (el.get("aria-hidden") or "").lower() in ("true", "1"):
            return True
        style = (el.get("style") or "").lower().replace(" ", "")
        if "display:none" in style or "visibility:hidden" in style or "opacity:0" in style:
            return True
        classes = " ".join(el.get("class", [])).lower()
        return bool(HIDDEN_CLASS_RX.search(classes or ""))

    def _visible_fields(container) -> List:
        fields = container.find_all(["input", "textarea", "select"])
        vis = []
        for f in fields:
            t = (f.get("type") or "").lower()
            if t in ("submit", "reset", "button"):
                continue
            if _is_node_hidden(f):
                continue
            vis.append(f)
        return vis

    def _has_submit_affordance(container) -> bool:
        # Knapper med type/role + CTA-tekst
        if container.find("button", attrs={"type": re.compile(r"^(submit|button)$", re.I)}, string=CTA_TEXT_RX):
            return True
        if container.find("input", attrs={"type": re.compile(r"^(submit|image)$", re.I)}):
            return True
        # Links der agerer CTA i SPA'er
        if container.find("a", string=CTA_TEXT_RX):
            return True
        # Generic role="button" med CTA-tekst eller data-action
        if container.find(attrs={"role": "button"}, string=CTA_TEXT_RX):
            return True
        if container.find(attrs={"data-action": re.compile(r"(submit|send|apply|book|request|bestil)", re.I)}):
            return True
        return False

    field_counts: List[int] = []
    processed_nodes: list = []  # behold referencer for ancestor/descendant-skip

    def _already_covered(node) -> bool:
        # undgå dobbelt-tælling af indlejrede containere
        for pn in processed_nodes:
            try:
                if pn is node:
                    return True
                # node inde i pn
                if node in getattr(pn, "descendants", []):
                    return True
                # pn inde i node
                if pn in getattr(node, "descendants", []):
                    return True
            except Exception:
                # BeautifulSoup kan kaste på visse relationer – ignorer
                continue
        return False

    # 1) Ægte <form>-elementer først
    for f in soup.find_all("form"):
        if _already_covered(f):
            continue
        vis = _visible_fields(f)
        if vis:
            field_counts.append(len(vis))
            processed_nodes.append(f)

    # 2) role="form"
    for f in soup.select('[role="form"]'):
        if _already_covered(f):
            continue
        vis = _visible_fields(f)
        if vis:
            field_counts.append(len(vis))
            processed_nodes.append(f)

    # 3) Class/ID/data-* kandidater
    candidates = (
        list(soup.find_all(attrs={"class": CONTAINER_RX})) +
        list(soup.find_all(id=CONTAINER_RX)) +
        list(soup.select("[data-component*=form i], [data-testid*=form i], [data-form]")) +
        list(soup.find_all(attrs={"aria-label": re.compile(r"(kontakt|contact|form)", re.I)}))
    )

    # Dedup (identiske BS-noder kan dukke op flere gange)
    seen_ids = set()
    uniq_candidates = []
    for c in candidates:
        cid = id(c)
        if cid not in seen_ids:
            uniq_candidates.append(c)
            seen_ids.add(cid)

    for c in uniq_candidates:
        if _already_covered(c):
            continue
        vis = _visible_fields(c)
        # Kræv mindst 2 felter, eller 1 felt + submit-affordance
        if len(vis) >= 2 or (len(vis) >= 1 and _has_submit_affordance(c)):
            field_counts.append(len(vis))
            processed_nodes.append(c)

    # 4) Eksterne/form-embeds (Typeform, HubSpot, Google Forms, Jotform, Mailchimp, ActiveCampaign)
    #    Hvis vi ser dem uden DOM-felter, registrér en "syntetisk" form med felt-antal 3.
    iframes = soup.find_all("iframe", src=True)
    scripts = soup.find_all("script", src=True)
    EMBED_RX = re.compile(
        r"(typeform\.com|hsforms\.net|hubspotforms|forms\.gle|docs\.google\.com/forms|jotform\.com|tfaforms\.com|list-manage\.com|activehosted\.com)",
        re.I,
    )
    has_external_form = any(EMBED_RX.search(i["src"]) for i in iframes) or any(EMBED_RX.search(s["src"]) for s in scripts)
    if has_external_form and not field_counts:
        field_counts.append(3)

    return field_counts


_DATE_REGEXES = [
    re.compile(r"\b\d{4}-\d{2}-\d{2}\b"),                      # 2025-08-14
    re.compile(r"\b\d{2}/\d{2}/\d{4}\b"),                      # 14/08/2025
    re.compile(r"\b\d{1,2}\.\s*[A-Za-zæøåÆØÅ]+\s*\d{4}\b"),    # 14. august 2025
    re.compile(r"\b\d{1,2}\s*[A-Za-zæøåÆØÅ]+\s*\d{4}\b"),      # 14 august 2025
    re.compile(r"\b[A-Za-zæøåÆØÅ]+\s*\d{1,2},\s*\d{4}\b"),     # August 14, 2025
]

def _parse_date_str(s: str) -> Optional[_dt.datetime]:
    s = s.strip()
    for fmt in ("%Y-%m-%d", "%d/%m/%Y"):
        try:
            return _dt.datetime.strptime(s, fmt).replace(tzinfo=timezone.utc)
        except Exception:
            pass
    try:
        months = {
            'januar':'January','februar':'February','marts':'March','april':'April',
            'maj':'May','juni':'June','juli':'July','august':'August',
            'september':'September','oktober':'October','november':'November','december':'December'
        }
        parts = s.lower()
        for dk,en in months.items():
            parts = parts.replace(dk, en.lower())
        parts = re.sub(r'(\d{1,2})\.\s*', r'\1 ', parts)
        dt = _dt.datetime.strptime(parts.title(), "%d %B %Y").replace(tzinfo=timezone.utc)
        return dt
    except Exception:
        return None

def _freshness_fallback_from_html(soup: BeautifulSoup, html: str) -> Dict[str, Any]:
    import re
    from typing import Optional, Dict, Any, List

    latest: Optional[_dt.datetime] = None
    candidates: List[str] = []

    # --- Hjælper: normaliser DK-måneder til ENG for bedre parsing ---
    dk_month_map = {
        "januar": "january", "jan": "jan",
        "februar": "february", "feb": "feb",
        "marts": "march", "mar": "mar",
        "april": "april", "apr": "apr",
        "maj": "may",
        "juni": "june", "jun": "jun",
        "juli": "july", "jul": "jul",
        "august": "august", "aug": "aug",
        "september": "september", "sep": "sep",
        "oktober": "october", "okt": "oct",
        "november": "november", "nov": "nov",
        "december": "december", "dec": "dec",
    }

    def _norm_danish_months(s: str) -> str:
        out = s
        for dk, eng in dk_month_map.items():
            out = re.sub(rf"\b{dk}\b", eng, out, flags=re.IGNORECASE)
        return out

    # --- Ekstra dato-regex'er (supplerer dine globale _DATE_REGEXES) ---
    _EXTRA_DATE_REGEXES = [
        # ISO og ISO med tid
        re.compile(r"\b\d{4}-\d{2}-\d{2}(?:[T ]\d{2}:\d{2}:\d{2}(?:Z|[+\-]\d{2}:\d{2})?)?\b"),
        # DD/MM/YYYY eller DD.MM.YYYY eller DD-MM-YYYY (også enkeltdigit dag/måned)
        re.compile(r"\b\d{1,2}[./-]\d{1,2}[./-]\d{2,4}\b"),
        # 14. august 2025 / 14 august 2025 / 14. aug 2025
        re.compile(r"(?:opdateret den\s+)?\b\d{1,2}\.?\s*(jan(?:uar)?|feb(?:ruar)?|mar(?:ts)?|apr(?:il)?|maj|jun(?:i)?|jul(?:i)?|aug(?:ust)?|sep(?:tember)?|okt(?:ober)?|nov(?:ember)?|dec(?:ember)?)\.?\s*\d{2,4}\b", re.IGNORECASE),
        # august 14, 2025 / aug 14, 2025
        re.compile(r"\b(jan(?:uar)?|feb(?:ruar)?|mar(?:ts)?|apr(?:il)?|maj|jun(?:i)?|jul(?:i)?|aug(?:ust)?|sep(?:tember)?|okt(?:ober)?|nov(?:ember)?|dec(?:ember)?)\s*\d{1,2},?\s*\d{2,4}\b", re.IGNORECASE),
        # 14/8-2025 el. 14-8-2025 (bindestreg i år)
        re.compile(r"\b\d{1,2}[./-]\d{1,2}-\d{2,4}\b"),
    ]

    # --- 1) META-tags: name/property/itemprop/http-equiv der indikerer dato ---
    for m in soup.find_all("meta"):
        for key in ("name", "property", "itemprop", "http-equiv"):
            k = (m.get(key) or "").lower()
            if any(x in k for x in ("date", "modified", "updated", "published", "article:")):
                val = (m.get("content") or "").strip()
                if val:
                    candidates.append(val)

    # --- 2) <time> tags: datetime-attr eller tekst ---
    for t in soup.find_all("time"):
        val = (t.get("datetime") or t.get("content") or t.get_text(" ", strip=True) or "").strip()
        if val:
            candidates.append(val)

    # --- 3) itemprop hints direkte i DOM ---
    for el in soup.select('[itemprop="datePublished"], [itemprop="dateModified"]'):
        val = (el.get("content") or el.get("datetime") or el.get_text(" ", strip=True) or "").strip()
        if val:
            candidates.append(val)

    # --- 4) Klasse-baserede hints (da/en) ---
    class_hints = [
        '[class*="date"]', '[class*="dato"]', '[class*="publish"]',
        '[class*="updated"]', '[class*="modified"]', '.last-modified',
        '.published', '.opdateret'
    ]
    for sel in class_hints:
        for el in soup.select(sel):
            val = (el.get("content") or el.get("datetime") or el.get_text(" ", strip=True) or "").strip()
            if val:
                candidates.append(val)

    # --- 5) Fallback: hele teksten (sidst) ---
    text = soup.get_text(" ", strip=True) if soup else (html or "")
    if text:
        candidates.append(text)

    # Kombinér globale regex'er (hvis defineret) med ekstra
    combined_patterns = []
    try:
        combined_patterns.extend(list(_DATE_REGEXES))  # type: ignore[name-defined]
    except Exception:
        pass
    combined_patterns.extend(_EXTRA_DATE_REGEXES)

    # --- Parse alle kandidater med alle mønstre ---
    for source in candidates:
        if not source:
            continue
        src_norm = _norm_danish_months(source)
        for rx in combined_patterns:
            for mo in rx.finditer(src_norm):
                token = mo.group(0)
                # prøv først uden normalisering, derefter med
                dt = _parse_date_str(token) or _parse_date_str(_norm_danish_months(token))  # type: ignore[name-defined]
                if dt and (latest is None or dt > latest):
                    latest = dt

    if latest:
        days = int((_dt.datetime.now(timezone.utc) - latest).total_seconds() // 86400)
        return {"latest_post_date": latest.isoformat(), "days_since_last_post": days}

    return {}

# -------- Ny, robust kilde-detektion for seneste post-dato --------

_DK_MONTHS = {
    "januar": 1, "februar": 2, "marts": 3, "april": 4, "maj": 5, "juni": 6,
    "juli": 7, "august": 8, "september": 9, "oktober": 10, "november": 11, "december": 12
}

def _parse_any_date_isoaware(s: str) -> Optional[_dt.datetime]:
    if not s:
        return None
    s = s.strip()

    # ISO8601 (med Z/offset)
    iso = s.replace("Z", "+00:00")
    try:
        dt = _dt.datetime.fromisoformat(iso)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    except Exception:
        pass

    # Simple formater
    for fmt in ("%Y-%m-%d", "%Y-%m-%d %H:%M", "%Y-%m-%d %H:%M:%S",
                "%d-%m-%Y", "%d/%m/%Y", "%Y/%m/%d"):
        try:
            return _dt.datetime.strptime(s, fmt).replace(tzinfo=timezone.utc)
        except Exception:
            continue

    # Dansk “8. august 2025” / “8 august 2025”
    m = re.search(r"(\d{1,2})\.?\s*(januar|februar|marts|april|maj|juni|juli|august|september|oktober|november|december)\s+(\d{4})", s, re.I)
    if m:
        d = int(m.group(1)); month = _DK_MONTHS[m.group(2).lower()]; y = int(m.group(3))
        try:
            return _dt.datetime(y, month, d, tzinfo=timezone.utc)
        except Exception:
            return None
    return None

def _best_max_tuple(current: Optional[tuple], candidate: Optional[tuple]) -> Optional[tuple]:
    # tuples er (dt, source, url, snippet)
    if not candidate:
        return current
    if not current:
        return candidate
    return candidate if candidate[0] > current[0] else current

def _extract_date_from_jsonld(html: str, page_url: str) -> Optional[tuple]:
    hits = []
    for m in re.finditer(r'<script[^>]+type=["\']application/ld\+json["\'][^>]*>(.*?)</script>', html or "", re.S | re.I):
        raw = (m.group(1) or "").strip()
        snippet = raw[:240].replace("\n", " ")
        try:
            data = json.loads(raw)
        except Exception:
            continue

        def scan(obj):
            if isinstance(obj, dict):
                for k in ("datePublished", "dateCreated", "uploadDate"):
                    v = obj.get(k)
                    if isinstance(v, str):
                        dt = _parse_any_date_isoaware(v)
                        if dt:
                            hits.append((dt, "jsonld", page_url, f'{k}="{v}"'))
                for v in obj.values():
                    scan(v)
            elif isinstance(obj, list):
                for it in obj:
                    scan(it)

        scan(data)
        if hits:
            hits.sort(key=lambda x: x[0], reverse=True)
            dt, src, url, _ = hits[0]
            return (dt, src, url, snippet)
    return None

def _extract_date_from_meta_time(html: str, page_url: str) -> Optional[tuple]:
    # OpenGraph / article meta
    for prop in ("article:published_time", "og:updated_time"):
        m = re.search(rf'<meta[^>]+property=["\']{prop}["\'][^>]+content=["\']([^"\']+)["\']', html or "", re.I)
        if m:
            val = m.group(1); dt = _parse_any_date_isoaware(val)
            if dt:
                return (dt, f"meta:{prop}", page_url, f'content="{val}"')

    # <time datetime="...">
    m = re.search(r'<time[^>]+datetime=["\']([^"\']+)["\'][^>]*>.*?</time>', html or "", re.I | re.S)
    if m:
        val = m.group(1); dt = _parse_any_date_isoaware(val)
        if dt:
            return (dt, "time@datetime", page_url, f'datetime="{val}"')

    # Tekst i <time>…</time>
    m = re.search(r'<time[^>]*>([^<]{4,80})</time>', html or "", re.I | re.S)
    if m:
        text = re.sub(r"\s+", " ", m.group(1)).strip()
        dt = _parse_any_date_isoaware(text)
        if dt:
            return (dt, "time@text", page_url, text[:120])

    # Generisk ISO i markup
    m = re.search(r'(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+\-]\d{2}:?\d{2}))', html or "", re.I)
    if m:
        val = m.group(1); dt = _parse_any_date_isoaware(val)
        if dt:
            return (dt, "iso-in-html", page_url, val)
    return None

def _extract_date_from_sitemaps(xml_texts: Optional[list]) -> Optional[tuple]:
    # Valgfri — kan bruges hvis du senere loader sitemap-XML (her no-op hvis None)
    if not xml_texts:
        return None
    prefer = ("nyheder", "artikler", "blog", "news")
    best = None
    url_block_re = re.compile(r"<url>(.*?)</url>", re.S | re.I)
    loc_re = re.compile(r"<loc>\s*([^<]+)\s*</loc>", re.I)
    lastmod_re = re.compile(r"<lastmod>\s*([^<]+)\s*</lastmod>", re.I)
    for xml in xml_texts:
        for block in url_block_re.findall(xml or ""):
            lm = lastmod_re.search(block); loc_m = loc_re.search(block)
            if not (lm and loc_m):
                continue
            loc = loc_m.group(1).strip()
            if not any(p in loc.lower() for p in prefer):
                continue
            dt = _parse_any_date_isoaware(lm.group(1))
            if dt:
                snippet = re.sub(r"\s+", " ", block.strip())[:240]
                best = _best_max_tuple(best, (dt, "sitemap", loc, snippet))
    return best

def detect_latest_post_date(page_html: str, page_url: str, sitemaps_xml: Optional[list] = None) -> tuple[Optional[str], Optional[str], Optional[str], Optional[str]]:
    """
    Returnerer (iso_date, source, url, snippet) eller (None, None, None, None).
    Prioritet: JSON-LD → meta/time → sitemap.
    """
    best = _extract_date_from_jsonld(page_html or "", page_url)
    best = _best_max_tuple(best, _extract_date_from_meta_time(page_html or "", page_url))
    best = _best_max_tuple(best, _extract_date_from_sitemaps(sitemaps_xml))
    if not best:
        return (None, None, None, None)
    dt, source, url, snippet = best
    iso = dt.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")
    return (iso, source, url, snippet)


# -------------------- RAW RESULT PARSER --------------------
def _extract_html_and_canonical_data(raw_result):
    """
    Robust extraction af HTML og canonical data fra forskellige return formater.
    Returns:
      tuple: (html_content: str, canonical_data: dict)
    """
    if raw_result is None:
        return None, {}

    if isinstance(raw_result, dict):
        html_content = raw_result.get('html')
        canonical_data = raw_result.get('canonical_data', {})
        return html_content, canonical_data

    elif isinstance(raw_result, tuple) and len(raw_result) == 2:
        html_content, canonical_data = raw_result
        if isinstance(canonical_data, str):
            return html_content, {}
        return html_content, canonical_data if isinstance(canonical_data, dict) else {}

    elif isinstance(raw_result, str):
        return raw_result, {}

    else:
        log.warning(f"Unknown raw_result format: {type(raw_result)}")
        return None, {}

# -------------------- LOGGING HELPERS --------------------
IMPORTANT_SCHEMA_TYPES = {
    'Product','Offer','AggregateOffer','ProductGroup','Vehicle',
    'BreadcrumbList','ItemList','SiteNavigationElement',
    'WebPage','Article','NewsArticle','BlogPosting','Recipe',
    'Organization','LocalBusiness','Store','Restaurant',
    'Event','BusinessEvent','EducationEvent',
    'Review','AggregateRating','Rating',
    'FAQPage','Question','Answer',
    'JobPosting','EmployerAggregateRating',
    'RealEstateListing','Accommodation'
}

def log_schema_findings(url: str, schema_found: bool, schema_types: List[str]) -> None:
    if schema_types:
        important = [t for t in schema_types if t in IMPORTANT_SCHEMA_TYPES]
        if important:
            log.info(f"Schema types found on {url}: {important}")
        else:
            log.debug(f"Generic schema types on {url}: {schema_types}")
    elif schema_found:
        log.debug(f"Schema markup detected on {url} but no specific types identified")
    else:
        if any(ind in url.lower() for ind in ['product','item','/p/','/shop','/buy','/produkt']):
            log.info(f"No schema markup found on product page: {url}")
        else:
            log.debug(f"No schema markup found on {url}")

# -------------------- HOVEDANALYSE --------------------
async def analyze_single_url(client: http_client.AsyncHtmlClient, url: str, max_pages: int = 50) -> UrlAnalysisData:
    # Valider URL
    if url.startswith('https://https://') or url.startswith('http://http://'):
        url = url.replace('https://https://', 'https://').replace('http://http://', 'http://')
        log.warning(f"Fixed invalid URL: {url}")

    log.info(f"Starting analysis for: {url}")
    start_page_seen = False  # <- kun når crawled_url == url sætter vi BASIC_SEO + main_page_data
    client.user_agent = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36'

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

    try:
        # Crawl flere sider
        from .crawler import crawl_site_for_links
        crawled_urls = await crawl_site_for_links(client, url, max_pages=max_pages)
        log.info(f"Crawled {crawled_urls['total_pages_crawled']} pages")

        # GMB + Authority kun én gang
        parsed_url = urlparse(url)
        base_url = f"{parsed_url.scheme}://{parsed_url.netloc}"

        gmb_data = await gmb_fetcher.fetch_gmb_data(client, url)
        analysis_data['social_and_reputation'].update(gmb_data)

        # Propagér bool til authority (scorerens data_key peger her)
        analysis_data['authority']['gmb_profile_complete'] = bool(gmb_data.get('gmb_profile_complete', False))

        log.info(
            f"GMB data fetched once: review_count={gmb_data.get('gmb_review_count', 0)}, "
            f"rating={gmb_data.get('gmb_average_rating', 0)}"
        )

        authority_data = await authority_fetcher.get_authority(client, url)
        analysis_data['authority'].update(authority_data or {})
        log.info(f"Authority data fetched once: DA={analysis_data['authority'].get('domain_authority', 0)}, PA={analysis_data['authority'].get('page_authority', 0)}")

        page_count = 0
        h1_texts: List[str] = []
        keyword_scores: List[float] = []
        emails, phones = set(), set()
        form_counts: List[int] = []
        trust_signals = set()
        word_counts: List[int] = []

        main_page_data: Optional[dict] = None
        start_page_data: Optional[dict] = None

        # Technical + Content for base (bemærk: vi kører teknisk fetch KUN én gang her)
        raw_result = await client.get_raw_html(base_url, force=True)
        html_content, _ = _extract_html_and_canonical_data(raw_result)
        if html_content:
            soup_base = BeautifulSoup(html_content, "lxml")
            tech_result = await technical_fetchers.fetch_technical_seo_data(client, base_url)
            analysis_data['technical_seo'].update(tech_result)

            # Indlæs content-data – og suppler med robust kilde-detektion (JSON-LD → meta/time → sitemap)
            content_data = await content_fetcher.fetch_content_data(client, soup_base, base_url)

            if not content_data.get('latest_post_date'):
                iso_date, date_source, date_url, snippet = detect_latest_post_date(
                    page_html=html_content,
                    page_url=base_url,
                    sitemaps_xml=None  # giv evt. liste af sitemap-XML-strings her, hvis du har dem
                )
                if iso_date:
                    dt_obj = _dt.datetime.fromisoformat(iso_date.replace("Z", "+00:00"))
                    if dt_obj.tzinfo is None:
                        dt_obj = dt_obj.replace(tzinfo=timezone.utc)
                    days = (_dt.datetime.now(timezone.utc) - dt_obj).days
                    content_data["latest_post_date"] = iso_date
                    content_data["days_since_last_post"] = days
                    content_data["latest_post_date_source"] = date_source
                    content_data["latest_post_date_url"] = date_url
                    content_data["latest_post_date_snippet"] = snippet
                    log.info(f"[content] Latest post: date={iso_date}, days_since={days}, source={date_source}, url={date_url}")
                else:
                    # sidste udvej: din eksisterende heuristik
                    fallback = _freshness_fallback_from_html(soup_base, html_content)
                    if fallback:
                        content_data.update(fallback)

            analysis_data['content'].update(content_data)
        else:
            content_data = DEFAULT_CONTENT_METRICS.copy()
        
        # --- PRE-PARSE: Sørg for at start-URL altid har BASIC_SEO, også hvis crawleren ikke besøger den ---
        start_page_data = None
        main_page_data = None
        if html_content:
            try:
                start_basic = website_fetchers.parse_basic_seo_from_soup(soup_base, base_url)

                # Canonical fra DOM/OG (evt. runtime hvis tilgængelig i raw_result)
                try:
                    runtime_state = None  # vi har typisk ingen canonical_data her
                    resolved_can, can_src = resolve_canonical_enhanced(base_url=base_url,
                                                                       rendered_html=html_content or "",
                                                                       runtime_state=runtime_state)
                    if resolved_can:
                        start_basic['canonical_url'] = resolved_can
                        start_basic['canonical_source'] = can_src
                        start_basic['canonical_error'] = None
                except Exception:
                    pass

                # Schema-detektion på start-URL
                try:
                    s_found, s_types, s_struct = detect_schema_from_runtime(html_content or "", {})
                except Exception:
                    s_found, s_types, s_struct = False, [], {}
                start_basic['schema_markup_found'] = bool(s_found)
                if s_types:
                    start_basic['schema_types'] = sorted(set(s_types))
                if isinstance(s_struct, dict) and s_struct:
                    start_basic['schema_type_keys'] = sorted(s_struct.keys())

                # Lås som start/main
                start_page_data = start_basic
                main_page_data = start_basic

            except Exception as e:
                log.warning(f"Pre-parse of start URL failed: {e}")

        # Loop over crawlede URLs
        visited = crawled_urls.get('visited_urls', {url})
        try:
            visited_list = list(visited)
        except Exception:
            visited_list = [url]

        # Find start-URL via ækvivalens (robust mod trailing slash, query mm.)
        start_idx = next((i for i, u in enumerate(visited_list) if _urls_equivalent(u, url)), None)
        if start_idx is not None:
            start_url_norm = visited_list[start_idx]
            ordered_urls = [start_url_norm] + [u for i, u in enumerate(visited_list) if i != start_idx]
        else:
            log.warning("Start URL missing from visited_urls; forcing it to be processed first.")
            ordered_urls = [url] + visited_list

        for crawled_url in ordered_urls:
            raw_result = await client.get_raw_html(crawled_url)
            html_content, canonical_data = _extract_html_and_canonical_data(raw_result)
            if not html_content:
                log.error(f"Could not fetch usable HTML for {crawled_url}")
                continue

            # --- NYT: Render-telemetri & soft-404 på start-URL ---
            # Vi måler faktisk renderet indhold (DOM-længde) uafhængigt af HTTP-status.
            try:
                rendered_len = len(html_content or "")
            except Exception:
                rendered_len = 0

            parsed_crawled_url = urlparse(crawled_url)
            crawled_base_url = f"{parsed_crawled_url.scheme}://{parsed_crawled_url.netloc}"
            is_start = _urls_equivalent(crawled_url, url)

            if is_start:
                tech = analysis_data.get('technical_seo', {}) or {}
                tech['rendered_content_length'] = rendered_len
                tech['render_status'] = 'content' if rendered_len > 1000 else 'empty'
                http_sc = tech.get('status_code')
                tech['soft_404_suspected'] = bool(http_sc == 404 and rendered_len > 800)
                analysis_data['technical_seo'] = tech
                if tech['soft_404_suspected']:
                    log.info("Soft-404 suspected: HTTP 404 men DOM har indhold (>800 chars).")

            soup = BeautifulSoup(html_content, "lxml")

            # Sync parsers
            basic_seo_result = website_fetchers.parse_basic_seo_from_soup(soup, crawled_base_url)

            # -------- Canonical håndtering (forbedret) --------
            if not basic_seo_result.get('canonical_url') and canonical_data and isinstance(canonical_data, dict):
                log.debug(f"Fallback to Playwright canonical_data for {crawled_url}: {canonical_data}")
                if canonical_data.get('link_canonical'):
                    # Oprindelig assignment
                    candidate_url = urljoin(crawled_base_url, canonical_data['link_canonical'])
                    basic_seo_result['canonical_url'] = candidate_url
                    basic_seo_result['canonical_source'] = 'playwright_link'
                    basic_seo_result['canonical_error'] = None
                    log.info(f"Used Playwright link canonical: {basic_seo_result['canonical_url']}")

                    # --- NYT: Valider canonical (samme host + https + HEAD==200) ---
                    try:
                        cu, uu = urlparse(candidate_url), urlparse(crawled_url)
                        same_host = (cu.hostname or '').lower() == (uu.hostname or '').lower()
                        https_ok = (cu.scheme or '').lower() == 'https'
                        status_ok = False
                        if same_host and https_ok:
                            head_resp = await client.head(candidate_url, follow_redirects=True)
                            status_ok = bool(head_resp and getattr(head_resp, "status_code", 0) == 200)
                        if not (same_host and https_ok and status_ok):
                            basic_seo_result['canonical_error'] = 'invalid_cross_host_or_non_https_or_non_200'
                            basic_seo_result['canonical_status'] = 'rejected'
                            basic_seo_result['canonical_url'] = None
                        else:
                            basic_seo_result['canonical_status'] = 'valid'
                            basic_seo_result['canonical_validated_on'] = 'HEAD'
                    except Exception as _e:
                        basic_seo_result['canonical_error'] = f'validation_error: {str(_e)[:120]}'
                        basic_seo_result['canonical_status'] = 'validation_error'

                elif canonical_data.get('canonical_fields') and isinstance(canonical_data['canonical_fields'], dict):
                    best_canonical = None
                    best_score = -1
                    best_path = None

                    for path, value in canonical_data['canonical_fields'].items():
                        if not should_accept_runtime_canonical(path):
                            log.debug(f"[canonical][reject] runtime path: {path}")
                            continue

                        if value is None:
                            continue
                        sval = str(value).strip()
                        if sval in ('None', 'null', '', 'false', 'False'):
                            continue

                        candidate = crawled_url if sval.lower() == 'true' else sval
                        if not _is_valid_canonical(candidate, crawled_url):
                            log.debug(f"[canonical][reject] invalid candidate: {candidate} (path={path})")
                            continue

                        # Scor kilde – foretræk produkt/kategori-specifikke felter
                        s = 0
                        pl = path.lower()
                        if 'product.current' in pl or 'canonical_primary_url' in pl:
                            s = 15
                        elif 'category.current' in pl or 'custom_canonical_url' in pl:
                            s = 12
                        else:
                            s = 6

                        if s > best_score:
                            best_score = s
                            best_path = path
                            best_canonical = urljoin(crawled_base_url, candidate)

                    if best_canonical:
                        # Oprindelig assignment
                        basic_seo_result['canonical_url'] = best_canonical
                        basic_seo_result['canonical_source'] = 'playwright_runtime_state'
                        basic_seo_result['canonical_error'] = None
                        log.info(f"Used best runtime canonical: {best_canonical} from {best_path}")

                        # --- NYT: Valider canonical (samme host + https + HEAD==200) ---
                        try:
                            cu, uu = urlparse(best_canonical), urlparse(crawled_url)
                            same_host = (cu.hostname or '').lower() == (uu.hostname or '').lower()
                            https_ok = (cu.scheme or '').lower() == 'https'
                            status_ok = False
                            if same_host and https_ok:
                                head_resp = await client.head(best_canonical, follow_redirects=True)
                                status_ok = bool(head_resp and getattr(head_resp, "status_code", 0) == 200)
                            if not (same_host and https_ok and status_ok):
                                basic_seo_result['canonical_error'] = 'invalid_cross_host_or_non_https_or_non_200'
                                basic_seo_result['canonical_status'] = 'rejected'
                                basic_seo_result['canonical_url'] = None
                            else:
                                basic_seo_result['canonical_status'] = 'valid'
                                basic_seo_result['canonical_validated_on'] = 'HEAD'
                        except Exception as _e:
                            basic_seo_result['canonical_error'] = f'validation_error: {str(_e)[:120]}'
                            basic_seo_result['canonical_status'] = 'validation_error'

            # --- NYT: 404-special på start-URL ---
            try:
                if is_start and (analysis_data.get('technical_seo', {}) or {}).get('status_code') == 404:
                    basic_seo_result['canonical_status'] = 'invalid_on_404'
            except Exception:
                pass
            # 2) Hvis stadig ingen canonical → DOM/OG/runtime-bred
            if not basic_seo_result.get('canonical_url'):
                runtime_state = canonical_data.get('runtime_state') if isinstance(canonical_data, dict) else None
                resolved, src = resolve_canonical_enhanced(
                    base_url=crawled_url,
                    rendered_html=html_content or "",
                    runtime_state=runtime_state
                )
                if resolved:
                    basic_seo_result['canonical_url'] = resolved
                    basic_seo_result['canonical_source'] = src
                    basic_seo_result['canonical_error'] = None
                    log.info(f"[canonical][fallback] Using resolved canonical ({src}): {resolved}")

            # -------- Schema / JSON-LD detektion (Nuxt/runtime-aware) --------
            try:
                schema_found, schema_types, schema_struct = detect_schema_from_runtime(
                    html_content or "",
                    canonical_data if isinstance(canonical_data, dict) else {}
                )
            except Exception as e:
                log.warning(f"Schema runtime detection failed for {crawled_url}: {e}", exc_info=True)
                schema_found, schema_types, schema_struct = False, [], {}

            basic_seo_result['schema_markup_found'] = bool(schema_found)
            if schema_types:
                basic_seo_result['schema_types'] = sorted(set(schema_types))
            if isinstance(schema_struct, dict) and schema_struct:
                basic_seo_result['schema_type_keys'] = sorted(schema_struct.keys())

            # Én samlet log (undgå dublet-logs)
            log_schema_findings(crawled_url, schema_found, basic_seo_result.get('schema_types', []))

            # --- Freshness fra blog/nyheder (oversigt + artikler) ---
            try:
                url_l = (crawled_url or "").lower()
                looks_like_blog = any(seg in url_l for seg in ("/nyheder", "/blog", "/news", "/artikler"))
                if looks_like_blog:
                    fresh = _freshness_fallback_from_html(soup, html_content) or {}
                    new_iso = fresh.get("latest_post_date")
                    if new_iso:
                        # løft KUN hvis den er nyere end det vi allerede har
                        def _parse_iso(s):
                            try:
                                s = s.replace("Z", "+00:00")
                                return _dt.datetime.fromisoformat(s)
                            except Exception:
                                return None
                        cur_iso = analysis_data["content"].get("latest_post_date")
                        cur_dt = _parse_iso(cur_iso) if cur_iso else None
                        new_dt = _parse_iso(new_iso)
                        if new_dt and (cur_dt is None or new_dt > cur_dt):
                            analysis_data["content"]["latest_post_date"] = new_iso
                            analysis_data["content"]["days_since_last_post"] = (
                                _dt.datetime.now(timezone.utc) - new_dt.replace(tzinfo=new_dt.tzinfo or timezone.utc)
                            ).days
                            log.info(f"[freshness] Updated latest_post_date from {crawled_url}: {new_iso}")
            except Exception as e:
                log.debug(f"[freshness] blog/nyheder fallback failed on {crawled_url}: {e}")

            # Lås start-side metadata
            if is_start:
                main_page_data = basic_seo_result
                start_page_data = basic_seo_result

            # Saml data kun fra start-siden til visse felter
            if is_start:
                if basic_seo_result.get('h1_texts'):
                    # start-URL definerer h1-listen (ingen aggregering på tværs af sider)
                    h1_texts = list(basic_seo_result['h1_texts'])
                if basic_seo_result.get('word_count'):
                    # word_count for BASIC_SEO afspejler start-URL
                    word_counts = [basic_seo_result['word_count']]

            # Social/Privacy/Trust/Contact
            social_result = social_fetchers.find_social_media_links(soup, crawled_base_url)
            privacy_result = privacy_fetchers.detect_cookie_banner(soup)
            trust_data = privacy_fetchers.detect_trust_signals(soup)
            if isinstance(trust_data, dict):
                trust_signals.update(trust_data.get('trust_signals_found', []))
            elif isinstance(trust_data, list):
                trust_signals.update(trust_data)

            contact_data = await find_contact_info(soup, crawled_base_url)
            emails.update(contact_data.get('emails_found', []))
            phones.update(contact_data.get('phone_numbers_found', []))

            # === Forms (forbedret + fallback) ===
            # 1) Legacy/DOM-detektor (giver meta-struktur)
            forms_meta = detect_forms(str(soup))
            derived_counts = [f.get("input_count", 0) for f in (forms_meta.get("forms") or [])]

            # 2) Ny forbedret detektor
            try:
                form_data = form_fetcher.analyze_forms(soup) or {}
            except Exception as e:
                log.debug(f"Form analysis failed: {e}")
                form_data = {}

            # 3) Fallbacks: brug legacy counts eller heuristik direkte fra DOM
            if not form_data.get('form_field_counts'):
                if derived_counts:
                    form_data['form_field_counts'] = derived_counts
                else:
                    ff = _forms_fallback_from_dom(soup)
                    if ff:
                        form_data['form_field_counts'] = ff

            # Medtag detaljeret meta fra legacy-scanneren
            form_data['forms_meta'] = forms_meta

            if form_data.get('form_field_counts'):
                form_counts.extend(form_data['form_field_counts'])

            # Tracking / CTA
            tracking_data = await tracking_fetchers.fetch_tracking_data(client, soup)
            
            # --- Runtime-fallback (Playwright) for tracking IDs ---
            try:
                if isinstance(canonical_data, dict):
                    runtime = canonical_data
                    analytics = runtime.get("analytics") or {}

                    def pick(*keys):
                        for k in keys:
                            if k in runtime and runtime.get(k):
                                return runtime.get(k)
                            if k in analytics and analytics.get(k):
                                return analytics.get(k)
                        return None

                    # GA4
                    ga4_id_runtime = pick("ga4Id", "ga_measurement_id")
                    if ga4_id_runtime and not tracking_data.get("ga4_measurement_id"):
                        tracking_data["ga4_measurement_id"] = ga4_id_runtime
                        tracking_data["has_ga4"] = True

                    # Meta Pixel
                    fb_id_runtime = pick("fbPixelId", "metaPixelId", "facebookPixelId")
                    if fb_id_runtime and not tracking_data.get("meta_pixel_id"):
                        tracking_data["meta_pixel_id"] = fb_id_runtime
                        tracking_data["has_meta_pixel"] = True

                    # GTM
                    if pick("hasGTM", "gtm", "gtmEnabled"):
                        tracking_data["has_gtm"] = True
                    gtm_id_runtime = pick("gtmContainerId", "gtmId")
                    if gtm_id_runtime and not tracking_data.get("gtm_container_id"):
                        tracking_data["gtm_container_id"] = gtm_id_runtime

                    # TikTok
                    tt_id = pick("ttPixelId", "tiktokPixelId")
                    if tt_id and not tracking_data.get("tiktok_pixel_id"):
                        tracking_data["tiktok_pixel_id"] = tt_id
                        tracking_data["has_tiktok_pixel"] = True

                    # Pinterest
                    pin_id = pick("pinterestTagId", "pinTagId")
                    if pin_id and not tracking_data.get("pinterest_tag_id"):
                        tracking_data["pinterest_tag_id"] = pin_id
                        tracking_data["has_pinterest_tag"] = True

                    # Snapchat
                    snap_id = pick("snapPixelId", "snapchatPixelId")
                    if snap_id and not tracking_data.get("snap_pixel_id"):
                        tracking_data["snap_pixel_id"] = snap_id
                        tracking_data["has_snap_pixel"] = True

                    # LinkedIn
                    li_id = pick("linkedinPartnerId", "linkedinInsightId")
                    if li_id and not tracking_data.get("linkedin_partner_id"):
                        tracking_data["linkedin_partner_id"] = li_id
                        tracking_data["has_linkedin_insight"] = True

                    # Bing UET
                    uet_id = pick("bingUetId", "msUetId")
                    if uet_id and not tracking_data.get("bing_uet_id"):
                        tracking_data["bing_uet_id"] = uet_id
                        tracking_data["has_bing_uet"] = True

                    # Google Ads (AdWords)
                    gad_ids = pick("googleAdsConversionIds", "adwordsConversionIds")
                    if isinstance(gad_ids, list):
                        tracking_data.setdefault("google_ads_conversion_ids", [])
                        tracking_data["google_ads_conversion_ids"] = list({*tracking_data["google_ads_conversion_ids"], *gad_ids})
                    gad_labels = pick("googleAdsLabels", "adwordsLabels")
                    if isinstance(gad_labels, list):
                        tracking_data.setdefault("google_ads_labels", [])
                        tracking_data["google_ads_labels"] = list({*tracking_data["google_ads_labels"], *gad_labels})

                    # Twitter / X
                    tw_id = pick("twitterPixelId", "xPixelId")
                    if tw_id and not tracking_data.get("twitter_pixel_id"):
                        tracking_data["twitter_pixel_id"] = tw_id
                        tracking_data["has_twitter_pixel"] = True

            except Exception:
                pass
            
            if analysis_data['content'].get('keyword_relevance_score'):
                keyword_scores.append(analysis_data['content']['keyword_relevance_score'])
            cta_data = await cta_fetcher.fetch_cta_data(client, soup)

            # Fallback: sæt viewport_score fra DOM hvis PSI ikke kører
            try:
                if analysis_data['performance'].get('viewport_score') in (None, 0):
                    vp_tag = soup.find('meta', attrs={'name': 'viewport'})
                    if vp_tag:
                        content = (vp_tag.get('content') or '').lower()
                        if 'width=device-width' in content:
                            # Scoringsreglen forventer 1 for mobilvenlig
                            analysis_data['performance']['viewport_score'] = 1
            except Exception:
                pass

            # Async tasks — kun på start-URL for de tunge fetches
            tasks = {}
            if is_start:
                tasks.update({
                    "psi": performance_fetcher.get_performance(client, crawled_url),
                    "security": security_fetchers.fetch_security_headers(client, crawled_url),
                    "js_size": performance_fetcher.calculate_js_size(client, soup, crawled_url),
                    "images": image_fetchers.fetch_image_stats(client, soup, crawled_url),
                })

            def _ensure_coro(x):
                if asyncio.isfuture(x) or asyncio.iscoroutine(x):
                    return x
                # Pak ikke-awaitables ind i en færdig future
                return asyncio.sleep(0, result=x)

            results = await asyncio.gather(*(_ensure_coro(v) for v in tasks.values()), return_exceptions=True)
            results_map = dict(zip(tasks.keys(), results))

            def get_result(key: str, default_value: Dict):
                res = results_map.get(key)
                if isinstance(res, Exception):
                    log.warning(f"Fetcher for '{key}' on {crawled_url} failed with exception: {res}", exc_info=True)
                    return default_value.copy()
                return res or default_value.copy()

            # ---- Billeder (stabil image_count) ----
            images_res = get_result("images", {})

            # Bevar parserens egen <img>-tælling som sandheden
            parser_img_count = basic_seo_result.get("image_count")

            # (Debug) gem fetcherens alternative tælling uden at påvirke scorer
            if isinstance(images_res.get("image_count"), (int, float)):
                basic_seo_result["image_count_fetcher"] = int(images_res["image_count"])

            # Supplér kvalitetsfelter fra image-fetcher
            for k in ("avg_image_size_kb", "image_alt_count", "image_alt_pct"):
                v = images_res.get(k)
                if v is not None:
                    basic_seo_result[k] = v

            # Kun hvis parseren slet ikke kunne tælle, fallback til fetcherens
            if (not isinstance(parser_img_count, (int, float)) or parser_img_count is None) and isinstance(images_res.get("image_count"), (int, float)):
                basic_seo_result["image_count"] = int(images_res["image_count"])

            if is_start:
                performance_result = get_result("psi", DEFAULT_PERFORMANCE_METRICS)
                performance_result.update(get_result("js_size", {}))
                analysis_data['performance'].update(performance_result)
                analysis_data['security'].update(get_result("security", DEFAULT_SECURITY_METRICS))

            # Aggregér sektioner (med BOOL-OR i stedet for sum for bools)
            def _merge_section(sec_key: str, data: Dict[str, Any]):
                for k, v in (data or {}).items():
                    if isinstance(v, bool):
                        analysis_data[sec_key][k] = bool(analysis_data[sec_key].get(k)) or v
                    elif isinstance(v, (int, float)) and k not in ['keyword_relevance_score']:
                        # Særlige nøgler der IKKE må summeres
                        if k == 'cta_score':
                            cur = analysis_data[sec_key].get(k, 0) or 0
                            analysis_data[sec_key][k] = max(cur, v)
                        else:
                            cur = analysis_data[sec_key].get(k, 0) or 0
                            analysis_data[sec_key][k] = cur + v
                    elif isinstance(v, list):
                        cur = analysis_data[sec_key].get(k) or []
                        analysis_data[sec_key][k] = list({*cur, *v})
                    else:
                        if v is not None or analysis_data[sec_key].get(k) is None:
                            analysis_data[sec_key][k] = v

            # BASIC_SEO må ikke forurenes af undersider – kun start-URL må merge
            if is_start:
                _merge_section("basic_seo", basic_seo_result)
                start_page_seen = True

            # VIGTIGT: Undlad at merge content_data pr. underside (det er base-data og ellers fordobles tal)
            _merge_section("social_and_reputation", social_result)
            _merge_section("conversion", {**tracking_data, **contact_data, **form_data, **trust_data, **cta_data})
            _merge_section("privacy", privacy_result)

            page_count += 1

            # Update crawled URLs info
            analysis_data['technical_seo'].update(crawled_urls)
        

    except Exception as e:
        log.error(f"Error during analysis for {url}: {e}", exc_info=True)

        # Sørg for at vi har en gyldig sektion at opdatere
        if not isinstance(analysis_data.get('technical_seo'), dict):
            analysis_data['technical_seo'] = DEFAULT_TECHNICAL_SEO.copy()

        # Hvis noget fejlede før crawled_urls blev sat, så brug sikre defaults
        try:
            analysis_data['technical_seo'].update(crawled_urls)  # kan fejle hvis crawled_urls ikke findes
        except UnboundLocalError:
            analysis_data['technical_seo'].update({
                'total_pages_crawled': 0,
                'total_links_found': 0,
                'broken_links_count': 0,
                'broken_links_pct': 0.0,
                'broken_links_list': {},
                'internal_link_score': 0,
                'visited_urls': {url},
                'page_type_distribution': {'product': 0, 'category': 0, 'info': 0, 'blog': 0, 'other': 0},
                'links_checked': 0,
            })

    # Smart aggregation efter loop
    target_can = (start_page_data if (start_page_data and start_page_data.get('canonical_url')) else main_page_data)
    if target_can:
        # Canonical
        if target_can.get('canonical_url'):
            analysis_data['basic_seo']['canonical_url'] = target_can['canonical_url']
            analysis_data['basic_seo']['canonical_source'] = target_can.get('canonical_source', 'unknown')
            analysis_data['basic_seo']['canonical_error'] = None

            # --- Stabiliser endelig canonical output (fjern trailing slash på root) ---
            cu_norm = _normalize_canonical_out(analysis_data['basic_seo']['canonical_url'])
            if cu_norm:
                analysis_data['basic_seo']['canonical_url'] = cu_norm

            # ⇩⇩ spejl også ind i TECHNICAL_SEO
            analysis_data['technical_seo']['canonical_url'] = analysis_data['basic_seo']['canonical_url']
            log.info(f"[canonical][set] Used canonical from {'start_url' if target_can is start_page_data else 'main_page_data'}: {analysis_data['basic_seo']['canonical_url']}")
        else:
            analysis_data['basic_seo']['canonical_error'] = "No canonical found on any page"
            log.info(f"[canonical][missing] No canonical found on any crawled page")


        # Schema status (opsummeret)
        schema_found = bool(analysis_data['basic_seo'].get('schema_markup_found'))
        schema_types = analysis_data['basic_seo'].get('schema_types', [])
        if schema_found and schema_types:
            log_metric_status(
                "schema_markup_found",
                f"True ({', '.join(schema_types[:3])}{'...' if len(schema_types) > 3 else ''})",
                "ok"
            )
        elif schema_found:
            log_metric_status("schema_markup_found", "True (generic)", "ok")
        else:
            critical = any(x in url.lower() for x in ('/product', '/produkt', '/p/', '/shop', '/buy'))
            if critical:
                log_metric_status("schema_markup_found", False, "missing")
            else:
                log.debug("[missing] schema_markup_found: False")

        # Title/description
        target_meta = start_page_data or (main_page_data or {})
        analysis_data['basic_seo']['title_text'] = target_meta.get('title_text')
        analysis_data['basic_seo']['title_length'] = target_meta.get('title_length', 0)
        analysis_data['basic_seo']['meta_description'] = target_meta.get('meta_description')
        analysis_data['basic_seo']['meta_description_length'] = len(analysis_data['basic_seo']['meta_description'] or '')

    # --- Finalize aggregates (sikr data sættes korrekt efter loop) ---
    if start_page_data:
        analysis_data['basic_seo']['h1_texts'] = list(start_page_data.get('h1_texts') or [])
        analysis_data['basic_seo']['h1_count'] = len(analysis_data['basic_seo']['h1_texts'])
        analysis_data['basic_seo']['h1'] = (
            analysis_data['basic_seo']['h1_texts'][0] if analysis_data['basic_seo']['h1_texts'] else None
        )
        if 'word_count' in start_page_data:
            analysis_data['basic_seo']['word_count'] = start_page_data['word_count']
    
    if start_page_data:
        for k in ("image_count", "image_alt_count", "image_alt_pct", "avg_image_size_kb"):
            if start_page_data.get(k) is not None:
                analysis_data['basic_seo'][k] = start_page_data[k]  
    else:
        # Fallback hvis start-URL ikke kunne hentes – behold tidligere adfærd
        analysis_data['basic_seo']['h1_texts'] = sorted(set(h1_texts)) if h1_texts else []
        analysis_data['basic_seo']['h1_count'] = len(analysis_data['basic_seo']['h1_texts'])
        analysis_data['basic_seo']['h1'] = analysis_data['basic_seo']['h1_texts'][0] if analysis_data['basic_seo']['h1_texts'] else None
    
    analysis_data['content']['average_word_count'] = int(sum(word_counts) / len(word_counts)) if word_counts else 0
    if keyword_scores:
        analysis_data['content']['keyword_relevance_score'] = sum(keyword_scores) / len(keyword_scores)
    
    analysis_data['conversion']['emails_found'] = sorted(emails)
    analysis_data['conversion']['phone_numbers_found'] = sorted(phones)
    analysis_data['conversion']['form_field_counts'] = form_counts
    analysis_data['conversion']['trust_signals_found'] = sorted(trust_signals)
    
    # Canonical og schema fra main_page_data (hvis bedre)
    if main_page_data:
        # Lås canonical til start-URL, hvis den findes; ellers må main_page_data supplere
        if not (start_page_data and start_page_data.get('canonical_url')):
            analysis_data['basic_seo']['canonical_url'] = main_page_data.get('canonical_url') or analysis_data['basic_seo'].get('canonical_url')
        analysis_data['basic_seo']['schema_markup_found'] = analysis_data['basic_seo'].get('schema_markup_found', False) or main_page_data.get('schema_markup_found', False)
        if main_page_data.get('schema_types'):
            analysis_data['basic_seo']['schema_types'] = main_page_data['schema_types']

    
    # Sikr sitemap data (fra technical fetch eller client)
    if analysis_data['technical_seo'].get('sitemap_xml_found') is None:
        sitemap_urls = await technical_fetchers.find_sitemaps(client, base_url)  # Brug eksisterende fetcher hvis muligt
        if sitemap_urls:
            analysis_data['technical_seo']['sitemap_xml_found'] = True
            analysis_data['technical_seo']['sitemap_locations'] = list(sitemap_urls)
    
    analysis_data.setdefault('technical_seo', {})
    analysis_data.setdefault('content', {})
    analysis_data.setdefault('performance', {})

    # 1) Interne links: spejl fra technical_seo -> content
    if 'internal_link_score' in analysis_data['technical_seo']:
        analysis_data['content']['internal_link_score'] = analysis_data['technical_seo']['internal_link_score']

    # 1b) Canonical: spejl også til technical_seo for konsistent rapport
    if analysis_data['basic_seo'].get('canonical_url'):
        analysis_data['technical_seo']['canonical_url'] = analysis_data['basic_seo']['canonical_url']

    # 2) Mobilvenlighed: afled bool ud fra viewport_score (1/100 = mobilvenlig)
    vp = analysis_data['performance'].get('viewport_score')
    if isinstance(vp, (int, float)):
        is_mobile_friendly = (vp == 1 or vp == 100)  # hvis reglen forventer bool
        analysis_data['performance']['mobile_friendly_check'] = is_mobile_friendly
        analysis_data['performance']['mobile_friendly_score'] = 1 if is_mobile_friendly else 0

    # Valider final dict (min tilføjelse for robusthed)  
    required_sections = ['basic_seo', 'technical_seo', 'conversion', 'content', 'performance', 'authority', 'security', 'social_and_reputation', 'privacy']  # Tilpas til dine schemas  
    for section in required_sections:  
        if section not in analysis_data or not isinstance(analysis_data[section], dict):  
            default_key = f'DEFAULT_{section.upper().replace("_", "")}'  
            if default_key in globals():  
                analysis_data[section] = globals()[default_key].copy()  
            else:  
                analysis_data[section] = {}  
                log.warning(f"Reinitialized missing/invalid section: {section}")  

    # 2b) Freshness fallback fra sitemap_lastmod -> content.latest_post_date (med kilde)
    try:
        smap = analysis_data.get('technical_seo', {}).get('sitemap_lastmod')
        has_latest = analysis_data.get('content', {}).get('latest_post_date')
        if smap and not has_latest:
            analysis_data['content']['latest_post_date'] = smap
            analysis_data['content']['latest_post_date_source'] = 'sitemap_lastmod'
            analysis_data['content']['latest_post_date_url'] = None
            analysis_data['content']['latest_post_date_snippet'] = None
            try:
                iso = smap.replace('Z', '+00:00')
                dt2 = _dt.datetime.fromisoformat(iso)
                if dt2.tzinfo is None:
                    dt2 = dt2.replace(tzinfo=timezone.utc)
                days = (_dt.datetime.now(timezone.utc) - dt2).days
                analysis_data['content']['days_since_last_post'] = days
                log.info(f"[content] Latest post (fallback sitemap_lastmod): date={smap}, days_since={days}, source=sitemap_lastmod")
            except Exception:
                pass
    except Exception:
        pass

    # Return nu sikret gyldigt dict  
    return analysis_data
