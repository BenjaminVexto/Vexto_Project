# src/vexto/scoring/crawler.py

from __future__ import annotations
import re
import heapq
import json
import asyncio
import logging
from typing import Optional, List, Set, Dict, Tuple
from urllib.parse import urlparse, urljoin, urlunparse
from bs4 import BeautifulSoup

log = logging.getLogger(__name__)

def _debug_print_pq(pq, header: str = "", top_n: int = 10) -> None:
    """Logger top N fra heap-baseret prioriteringskø (negativ score i heap)."""
    try:
        snapshot = sorted(pq, key=lambda x: x[0])[:top_n]  # mindste (mest negative) = højest prio
        log.info("=" * 50)
        if header:
            log.info(f"📋 Queue snapshot — {header}")
        else:
            log.info("📋 Queue snapshot")
        for i, (neg_prio, url) in enumerate(snapshot, 1):
            pr = -int(neg_prio)
            tail = url[-80:]
            log.info(f"{i:2}. [P:{pr:3}] {tail}")
        log.info(f"Kø-størrelse: {len(pq)}")
        log.info("=" * 50)
    except Exception as e:
        log.debug(f"Kunne ikke printe queue snapshot: {e}")

# Filtrér assets/CDN for link-tjek (HEAD)
ASSET_EXTENSIONS = (
    ".jpg", ".jpeg", ".png", ".gif", ".webp", ".svg", ".ico",
    ".css", ".js", ".pdf", ".xml", ".mp4", ".webm", ".woff", ".woff2", ".ttf", ".eot", ".zip", ".doc", ".docx"
)
CDN_PREFIXES = ("m2.", "cdn.", "media.", "assets.")

# -------------------- HELPERS --------------------

def _is_asset(url: str) -> bool:
    if not url:
        return False
    base = url.split("?", 1)[0].lower()
    return base.endswith(ASSET_EXTENSIONS)

def _is_blocked_cdn(url: str) -> bool:
    try:
        host = urlparse(url).netloc.lower()
    except Exception:
        return False
    return any(host.startswith(p) for p in CDN_PREFIXES)

# --- URL normalisering / same-site helpers ---
def _norm_host(host: str) -> str:
    """Fjern www. og lav lowercase for stabile sammenligninger."""
    host = (host or "").strip().lower()
    return host[4:] if host.startswith("www.") else host

def is_same_site(url: str, base_domain: str) -> bool:
    """Accepter både www og naked-domain samt http/https som samme site."""
    try:
        u_host = _norm_host(urlparse(url).netloc)
        b_host = _norm_host(base_domain)
        return bool(u_host) and u_host == b_host
    except Exception:
        return False

def normalize_internal_url(url: str, start_url: str) -> str:
    """
    Normaliser interne URLs:
    - Prefix 'https:' for '//' og '://'
    - Tving https scheme
    - Brug start_url's host (bevar www-policy fra start-url)
    """
    if not url:
        return url
    if url.startswith("://"):      # f.eks. '://www.example.com/...'
        url = "https" + url
    elif url.startswith("//"):     # protokol-relative
        url = "https:" + url

    # Byg absolut ift. start_url og tving https + konsistent host
    base = urlparse(start_url)
    absu = urlparse(urljoin(start_url, url))
    scheme = "https"
    # Foretræk base.netloc (bevarer evt. 'www.') hvis samme registrerede host
    if _norm_host(absu.netloc) == _norm_host(base.netloc):
        netloc = base.netloc
    else:
        netloc = absu.netloc or base.netloc
    return urlunparse((scheme, netloc, absu.path or "/", absu.params, absu.query, absu.fragment))

def should_check_link_status(url: str) -> bool:
    """Kun HEAD-tjek for 'rigtige' sider – ikke assets, ikke CDN, ikke special schemes."""
    if not url or not url.startswith(("http://","https://")):
        return False
    if _is_asset(url) or _is_blocked_cdn(url):
        return False
    low = url.lower()
    if any(low.startswith(s) for s in ("mailto:","tel:","javascript:","data:","callto","#")):
        return False
    # site-specifik billedgenerator-mønstre (magento/resize mm.)
    if "/img/" in low and "/resize/" in low:
        return False
    return True

def _normalize_url(u: str) -> str:
    """Normalisér URL til stabil sammenligning/kø (samler trailing '/', fjerner fragment, normaliserer host/port)."""
    try:
        p = urlparse(u)
        scheme = (p.scheme or "https").lower()
        netloc = (p.netloc or "").lower()
        # Fjern default-porte
        if netloc.endswith(":80") and scheme == "http":
            netloc = netloc[:-3]
        if netloc.endswith(":443") and scheme == "https":
            netloc = netloc[:-4]
        # Saml paths: root => '/', ellers strip trailing '/'
        path = p.path or "/"
        if path != "/" and path.endswith("/"):
            path = path.rstrip("/")
        # behold query; drop fragment/params
        return f"{scheme}://{netloc}{path}" + (f"?{p.query}" if p.query else "")
    except Exception:
        return u    

async def _fetch_soup(client, url: str) -> Optional[BeautifulSoup]:
    """
    Hurtig, robust hentning af HTML -> BeautifulSoup.
    1) Prøv httpx (billigt). 2) Fallback til get_raw_html(return_soup=True) i alle de varianter jeres klient kan returnere.
    """
    # Billigt forsøg med httpx (render-fri)
    try:
        r = await client.httpx_get(url, follow_redirects=True)
        if r is not None and getattr(r, "status_code", 599) < 400 and r.text:
            return BeautifulSoup(r.text, "lxml")
    except Exception:
        pass

    # Fallback: get_raw_html (kan returnere tuple/dict/str/None)
    try:
        raw = await client.get_raw_html(url, return_soup=True)
    except Exception:
        raw = None

    if raw is None:
        return None
    # (soup, canonical_data)
    if isinstance(raw, tuple) and raw and isinstance(raw[0], BeautifulSoup):
        return raw[0]
    # direkte BeautifulSoup
    if isinstance(raw, BeautifulSoup):
        return raw
    # dict {"html": "..."}
    if isinstance(raw, dict) and raw.get("html"):
        return BeautifulSoup(raw["html"], "lxml")
    # rå HTML-streng
    if isinstance(raw, str):
        return BeautifulSoup(raw, "lxml")
    return None

def _segments(path: str) -> list[str]:
    return [s for s in (path or "").split('/') if s]

def analyze_url_structure(url: str) -> dict:
    """
    Multi-signal analyse af URL for at bestemme sidetype.
    Returnerer: {'type': str, 'confidence': float(0-1), 'scores': dict}
    """
    parsed = urlparse(url)
    path = (parsed.path or "").lower()
    query = (parsed.query or "").lower()
    segments = _segments(path)
    seg_count = len(segments)
    segs_lower = [s.lower() for s in segments]

    scores = {'product': 0, 'category': 0, 'info': 0, 'blog': 0, 'other': 0}

    # --- PRODUCT SIGNALS ---
    if any(p in query for p in ('sku=', 'id=', 'product=', 'item=', 'variant=', 'childsku=')):
        scores['product'] += 50
    if re.search(r'-\d{3,}(?:$|[/?#])', path):
        scores['product'] += 40
    if re.search(r'/p\d{3,}', path):
        scores['product'] += 30
    if re.search(r'/(dp|pd)/', path):
        scores['product'] += 25
    if re.search(r'\d{4,}', path):
        scores['product'] += 20
    if re.search(r'[a-z]\d{4,}', path):
        scores['product'] += 15
    if seg_count >= 3:
        scores['product'] += 10

    # --- CATEGORY SIGNALS ---
    category_words = {
        'category', 'categories', 'collection', 'collections',
        'shop', 'products', 'produkter', 'catalog', 'katalog', 'kategori', 'kategorier',
        'kontormobler', 'butiksinventar', 'kantine', 'lager', 'vaerksted',
        'kontor-tilbehor', 'tilbud', 'demo-og-restsalg'
    }
    if any(seg in category_words for seg in segments):
        scores['category'] += 40
    elif seg_count <= 2 and not re.search(r'\d{3,}', path):
        scores['category'] += 10

    # --- INFO SIGNALS ---
    info_map = {
        'about': 40, 'about-us': 40, 'om': 40, 'om-os': 40, 'om-inventarland': 40,
        'contact': 40, 'kontakt': 40, 'kontaktformular': 40, 'kontakt-os': 40,
        'case': 35, 'cases': 35, 'reference': 35, 'referencer': 35,
        'support': 35, 'help': 35, 'faq': 35,
        'terms': 30, 'privacy': 30, 'policy': 30,
        'privatlivspolitik': 30, 'cookies': 30, 'cookie-politik': 30,
        'vilkar': 30, 'betingelser': 30,
        'kundeservice': 35, 'service': 25,
        'projektafdelingen': 30
    }
    matched_info = False
    for key, weight in info_map.items():
        if key in segs_lower:
            scores['info'] += weight
            matched_info = True
            break
        if any(key in t for s in segs_lower for t in s.split('-')):
            scores['info'] += max(10, weight // 2)
            matched_info = True
            break
    
    # Hvis vi med rimelig sikkerhed har INFO, så nedprioritér produkt-score
    if matched_info and scores['info'] >= 40:
        scores['product'] = min(scores['product'], 10)

    # --- BLOG SIGNALS ---
    blog_set = {
        'blog', 'blogs', 'news', 'nyheder', 'article', 'articles', 'artikel', 'artikler', 'post', 'posts',
        'inspiration', 'inspiration-til-indretning'
    }
    if any(seg in blog_set for seg in segs_lower):
        scores['blog'] += 40
    if re.search(r'/\d{4}/\d{2}/', path) or re.search(r'/\d{4}-\d{2}/', path):
        scores['blog'] += 30

    # --- Afgør type ---
    page_type = max(scores, key=scores.get)
    max_score = scores[page_type]
    if max_score == 0:
        return {'type': 'other', 'confidence': 0.0, 'scores': scores}

    confidence = min(max_score / 100.0, 1.0)
    return {'type': page_type, 'confidence': confidence, 'scores': scores}


def is_likely_product_page(url: str, threshold: float = 0.5) -> bool:
    a = analyze_url_structure(url)
    return a['type'] == 'product' and a['confidence'] >= threshold

def _detect_product_dom_signals(soup: BeautifulSoup) -> dict:
    """
    Returnerer {'score': int, 'signals': [...], 'is_product': bool}
    Stærke signaler (>= 60) kræves for at kalde en side 'product'.
    """
    if not soup:
        return {'score': 0, 'signals': [], 'is_product': False}

    score = 0
    signals: list[str] = []

    # 1) JSON-LD: @type Product
    try:
        for sc in soup.find_all("script", attrs={"type": "application/ld+json"}):
            txt = sc.string or sc.get_text() or ""
            if not txt.strip():
                continue
            # lenient cleanup (trailing commas etc.)
            txt = re.sub(r"/\*.*?\*/", "", txt, flags=re.S)
            txt = re.sub(r"//.*?$", "", txt, flags=re.M)
            txt = re.sub(r",\s*([}\]])", r"\1", txt)
            data = json.loads(txt)
            nodes = data if isinstance(data, list) else [data]
            for n in nodes:
                if isinstance(n, dict):
                    t = n.get("@type")
                    if isinstance(t, list):
                        types = [str(x).lower() for x in t]
                    else:
                        types = [str(t).lower()] if t else []
                    if any(x == "product" for x in types):
                        score += 50
                        signals.append("jsonld_product")
                        raise StopIteration
    except StopIteration:
        pass
    except Exception:
        pass

    # 2) Pris-mønstre (DK/EN): kr, dkk, price, currency + tal
    body_text = (soup.get_text(" ", strip=True) or "")[:200000].lower()
    if re.search(r"\b(?:kr\.?|dkk)\s?\d{2,3}(?:[.\s]?\d{3})*(?:[.,]\d{2})?\b", body_text):
        score += 25; signals.append("price_dk")
    if re.search(r"\b(?:price|currency)\b.*?\d", body_text):
        score += 10; signals.append("price_en_hint")

    # 3) Køb/kurv-knapper
    btn_text = " ".join(el.get_text(" ", strip=True).lower() for el in soup.find_all(["button","a"]))
    if any(w in btn_text for w in ("køb", "læg i kurv", "tilføj til kurv", "add to cart", "add-to-cart")):
        score += 25; signals.append("add_to_cart")

    # 4) WooCommerce/CMS-artefakter
    classes = " ".join(" ".join(el.get("class", [])) for el in soup.find_all(attrs={"class": True})).lower()
    scripts = " ".join([s.get("src","") or "" for s in soup.find_all("script", src=True)]).lower()
    if any(k in classes for k in ("woocommerce", "wc-product", "single-product")) or \
       any(k in scripts for k in ("woocommerce.min.js", "add-to-cart", "wc-add-to-cart")):
        score += 20; signals.append("woocommerce_assets")

    is_product = score >= 60
    return {'score': score, 'signals': signals, 'is_product': is_product}

def classify_page_type(url: str) -> str:
    a = analyze_url_structure(url)
    if a['confidence'] < 0.3:
        log.debug(f"Low confidence page-type for {url}: {a}")
    return a['type']

def calculate_url_priority(u: str, base_domain: str | None = None) -> int:
    # base_domain er bevidst ubrugt her, men accepteres for bagud-kompatibilitet
    parsed = urlparse(u)
    analysis = analyze_url_structure(u)
    page_type = analysis['type']
    base = {"product": 10, "category": 50, "info": 80, "blog": 70, "other": 30}[page_type]
    priority = base

    segments = [s for s in parsed.path.split("/") if s]
    for seg in segments:
        s = seg.lower()
        if s in ("kontakt", "contact", "kundeservice") or "kontaktformular" in s or "kontakt-formular" in s:
            priority += 40
        if s in ("om", "om-inventarland", "about"):
            priority += 15
        if s.startswith("inspiration") or "inspiration" in s:
            priority += 25
        if any(k in s for k in ("case", "reference", "referencer", "kunde-case", "kundecase")):
            priority += 20

    return min(priority, 200)


def is_navigation_link(a_tag) -> bool:
    """Boost links i nav/header/footer."""
    for parent in a_tag.parents:
        if getattr(parent, "name", "").lower() in ('nav', 'header', 'footer'):
            return True
        if getattr(parent, "attrs", {}).get('role') == 'navigation':
            return True
        classes = parent.get('class', [])
        if isinstance(classes, list):
            classes = ' '.join(classes)
        if any(k in str(classes).lower() for k in ['nav', 'menu', 'footer', 'breadcrumb']):
            return True
    return False

def extract_navigation_links(soup: BeautifulSoup, base_url: str) -> List[str]:
    if not soup:
        return []
    out: Set[str] = set()
    selectors = [
        'nav', 'header', 'footer',
        '[role="navigation"]',
        '.navigation', '.nav', '.menu', '.footer', '.site-footer', '.global-footer',
        '#navigation', '#menu', '#footer'
    ]
    for sel in selectors:
        for el in soup.select(sel):
            for a in el.find_all('a', href=True):
                href = a.get('href')
                if href and not href.startswith(('#','mailto:','tel:','javascript:','data:','callto:')):
                    abs_url = urljoin(base_url, href)
                    out.add(normalize_internal_url(abs_url, base_url))
    return list(out)

def should_skip_page_type(
    url: str,
    counts: dict,
    max_products: int = 5,
    max_per_type=None,
) -> bool:
    """
    Returnerer True hvis siden skal springes over baseret på type-kvoter.
    """
    if max_per_type is None:
        max_per_type = {
            'product': max_products, 'category': 10, 'blog': 10, 'info': 20, 'other': 5,
        }
    elif isinstance(max_per_type, int):
        max_per_type = {
            'product': max_products, 'category': max_per_type, 'blog': 10, 'info': 20, 'other': 5,
        }

    analysis = analyze_url_structure(url)
    t = analysis['type']

    if analysis['confidence'] < 0.15 and counts.get(t, 0) > 5:
        return True

    current = counts.get(t, 0)
    limit = max_per_type.get(t, 10)
    if current >= limit:
        log.debug(f"Skipping {url} - quota reached for {t} ({current}/{limit})")
        return True

    return False

# -------------------- HOVED-CRAWL --------------------

async def crawl_site_for_links(
    client,
    start_url: str,
    max_pages: int = 50,
) -> dict:
    """
    Crawler et site for at finde links og analysere sidetyper og brudte links.
    """
    parsed = urlparse(start_url)
    base_domain = (parsed.netloc or "").lower()

    # Adaptiv kvote (kan tweakes)
    MAX_PRODUCTS = max(3, int(max_pages * 0.25))
    MAX_PER_CATEGORY = max(2, int(max_pages * 0.15))

    visited: Set[str] = set()
    queued: Set[str] = set()  # ny: for at undgå duplikater i køen
    all_found: Set[str] = set()
    internal_links: Set[str] = set()
    page_type_counts = {'product':0,'category':0,'info':0,'blog':0,'other':0}

    # Priority queue (neg-score -> max-heap adfærd)
    pq: List[Tuple[int,str]] = []
    start_n = _normalize_url(start_url)
    heapq.heappush(pq, (-100, start_n))
    queued.add(start_n)

    _debug_print_pq(pq, header="initial (after start)", top_n=10)
    # Sørg for at forsiden scannes først
    home_url = _normalize_url(start_url)
    if home_url not in visited and home_url not in queued:
        heapq.heappush(pq, (-9999, home_url))  # ekstrem høj prioritet
        queued.add(home_url)
    
    # Seed: hent forsiden som soup (med JS fallback via _fetch_soup) og træk nav/footer links
    try:
        root_soup = await _fetch_soup(client, start_url)
        if root_soup:
            seeded_nav = 0
            for link in extract_navigation_links(root_soup, start_url):
                ln = normalize_internal_url(link, start_url)
                if not is_same_site(ln, base_domain):
                    continue
                if _is_asset(ln):
                    continue
                pr = calculate_url_priority(ln, base_domain)
                heapq.heappush(pq, (-pr, ln))
                seeded_nav += 1
                log.debug(f"Seed link fra nav/footer: {ln} (prio {pr})")

            # Core path hints – sikre nøglesider, også hvis de ikke er i nav/footer
            CORE_PATH_HINTS = ["/kontakt", "/nyheder", "/nyheder/artikler"]
            forced = []
            for path in CORE_PATH_HINTS:
                forced_url = normalize_internal_url(urljoin(start_url, path), start_url)
                if is_same_site(forced_url, base_domain) and not _is_asset(forced_url):
                    pr = calculate_url_priority(forced_url, base_domain) + 30  # lille boost
                    heapq.heappush(pq, (-pr, forced_url))
                    forced.append(forced_url)

            if seeded_nav > 0:
                log.info(f"Seeded {seeded_nav} nav/footer-links (høj prioritet).")
            if forced:
                log.info(f"Seeded {len(forced)} core path-hints: {forced}")
            _debug_print_pq(pq, header=f"after seed(nav/footer, {seeded_nav} urls)", top_n=10)
    except Exception as e:
        log.debug(f"Kunne ikke seed'e nav/footer links: {e}")


    try:
        sm_url = await client.get_sitemap_url(start_url)
    except Exception:
        sm_url = None

    if sm_url:
        try:
            resp = await client.httpx_get(sm_url, follow_redirects=True, timeout=20)
            if resp and resp.status_code < 400 and resp.text:
                xsoup = BeautifulSoup(resp.text, "xml")
                locs = [loc.get_text(strip=True) for loc in xsoup.find_all("loc")]
                NEEDLES = ("kontakt", "kontaktformular", "blog", "nyheder", "inspiration", "case", "referencer", "reference")
                sitemap_seeded_count = 0
                for loc in locs:
                    if any(n in loc.lower() for n in NEEDLES):
                        pr = calculate_url_priority(loc) + 40  # sitemap-bonus
                        heapq.heappush(pq, (-pr, loc))
                        sitemap_seeded_count += 1
                        if sitemap_seeded_count >= 12:
                            break
                    if sitemap_seeded_count > 0:
                        log.info(f"Seeded {sitemap_seeded_count} vigtige URL’er fra sitemap.")
                        _debug_print_pq(pq, header=f"after seed(sitemap, {sitemap_seeded_count} urls)", top_n=10)
        except Exception as e:
            log.debug(f"Sitemap seed sprang over ({e})")

    exclude_ext = {'.pdf', '.jpg', '.jpeg', '.png', '.gif', '.xml', '.zip', '.doc', '.docx', '.webp', '.svg'}

    while pq and len(visited) < max_pages:
        try:
            _, current = heapq.heappop(pq)
        except IndexError:
            break

        current_n = _normalize_url(current)

        if current_n in visited:
            continue
        if _is_asset(current_n) or any(current_n.lower().endswith(ext) for ext in exclude_ext):
            continue
        if not is_same_site(current, base_domain):
            continue

        # Kvoter
        if should_skip_page_type(current_n, page_type_counts, max_products=MAX_PRODUCTS, max_per_type=MAX_PER_CATEGORY):
            log.debug(f"Skip (kvote nået) {current_n}")
            continue

        visited.add(current_n)

        # 1) URL-baseret første gæt
        t_initial = classify_page_type(current_n)

        # 2) Hent DOM og raffinér
        soup = await _fetch_soup(client, current)
        if not soup:
            # Kun hvis vi ikke kan hente DOM, tæller vi første gæt
            t_final = t_initial
        else:
            dom = _detect_product_dom_signals(soup)

            # Nedgrader "product" hvis DOM ikke viser stærke produkt-signaler
            if t_initial == "product" and not dom["is_product"]:
                # simpelt skel: hvis siden ligner blog/nyhed (år/måned eller 'article') → 'blog', ellers 'info'
                html_txt = soup.get_text(" ", strip=True).lower()
                looks_blog = bool(re.search(r"/\d{4}/\d{2}/", current_n) or "article" in html_txt or "blog" in html_txt or "nyhed" in html_txt or "artik" in html_txt)
                t_final = "blog" if looks_blog else "info"
                log.debug(f"[page-type] Demoted product → {t_final} (no strong DOM signals) for {current_n}; dom={dom}")
            # Opgrader til "product" hvis stærke signaler findes
            elif t_initial != "product" and dom["is_product"]:
                t_final = "product"
                log.debug(f"[page-type] Promoted to product (strong DOM signals) for {current_n}; dom={dom}")
            else:
                t_final = t_initial

        # 3) Tæl endelig type og log
        page_type_counts[t_final] = page_type_counts.get(t_final, 0) + 1
        log.info(f"Crawling: {current_n} ({len(visited)}/{max_pages}) [Type: {t_final}]")

        for a in soup.find_all('a', href=True):
            href = a.get('href')
            if not href or href.startswith(('#', 'mailto:', 'tel:', 'javascript:', 'data:', 'callto:')):
                continue

            ln = normalize_internal_url(href, current_n)
            norm_abs = _normalize_url(ln)
            all_found.add(norm_abs)

            if is_same_site(norm_abs, base_domain):
                internal_links.add(norm_abs)
                if norm_abs not in visited and norm_abs not in queued and not _is_asset(norm_abs):
                    pr = calculate_url_priority(norm_abs, base_domain)
                    if is_navigation_link(a):
                        pr += 20
                    heapq.heappush(pq, (-pr, norm_abs))
                    queued.add(norm_abs)

        if len(visited) > 0 and (len(visited) % 5 == 0):
            _debug_print_pq(pq, header=f"after add (visited={len(visited)})", top_n=10)

    internal_link_score = round((len(internal_links) / len(all_found) * 100) if all_found else 0)

    to_check = [u for u in internal_links if should_check_link_status(u)]
    CAP = min(60, max(20, max_pages * 3))
    to_check = to_check[:CAP]

    broken_links: Dict[str, int] = {}
    links_checked = 0
    try:
        from .http_client import batch_head_requests as _batch_head
        results = await _batch_head(client, to_check, timeout=5.0, concurrency=10, cap=CAP)
        links_checked = len(results)
        for item in results:
            status = item.get("status")
            url = item.get("url")
            if status is None:
                continue
            # Tæl både 4xx og 5xx som brudte
            if int(status) >= 400:
                broken_links[url] = int(status)
    except Exception as e:
        log.debug(f"HEAD batch fejlede: {e}")
        # Reset KUN hvis hele batchen fejler
        links_checked = 0
        broken_links = {}

    broken_links_count = len(broken_links)
    broken_links_pct = round((broken_links_count / links_checked * 100), 1) if links_checked else 0.0

    return {
        'total_pages_crawled': len(visited),
        'total_links_found': len(all_found),
        'broken_links_count': broken_links_count,
        'broken_links_pct': broken_links_pct,
        'broken_links_list': list(broken_links.keys()),
        'internal_link_score': internal_link_score,
        'visited_urls': list(visited),
        'page_type_distribution': page_type_counts,
        'links_checked': links_checked,
    }