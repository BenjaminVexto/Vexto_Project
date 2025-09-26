# src/vexto/scoring/http_client.py
from __future__ import annotations
import sys
import asyncio
import threading
import time
import logging
import ssl
import os
import json
import tempfile
import shutil
import re
import random
import certifi
import httpx
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, Optional, Union, Any, List, Tuple, Iterable
from urllib.parse import urlparse, urljoin
from bs4 import BeautifulSoup
from diskcache import Cache
from fake_useragent import FakeUserAgentError, UserAgent
from playwright.sync_api import (sync_playwright,Browser as PlaywrightBrowser)
from tenacity import (AsyncRetrying,RetryError,retry_if_exception,stop_after_attempt,wait_exponential_jitter)
from ipaddress import ip_address, ip_network
from urllib.parse import urlsplit


log = logging.getLogger(__name__)

# --- Optional stealth ---
try:
    from playwright_stealth import stealth_sync
    STEALTH_AVAILABLE = True 
except ImportError:
    STEALTH_AVAILABLE = False
    log.warning("playwright-stealth ikke tilgængelig - bruger manual stealth")

def _stealth_raw() -> str:
    return (os.getenv("VEXTO_STEALTH", "0") or "").strip().lower()

def _stealth_env_enabled() -> bool:
    raw = _stealth_raw()
    return raw in ("1", "true", "yes", "on")

_PRIVATE_NETS = (
    ip_network("10.0.0.0/8"),
    ip_network("172.16.0.0/12"),
    ip_network("192.168.0.0/16"),
    ip_network("127.0.0.0/8"),
    ip_network("169.254.0.0/16"),
)

def _is_private_ip_host(host: str) -> bool:
    if not host:
        return False
    try:
        ip = ip_address(host)
        return any(ip in net for net in _PRIVATE_NETS)
    except ValueError:
        # Ikke en IP-literal (domæne) – lad almindelig DNS/HTTP guard tage over
        return False



# --- Cache setup ---
CACHE_DIR = Path(".http_diskcache")
CACHE_DIR.mkdir(exist_ok=True)
html_cache = Cache(str(CACHE_DIR / "html"), size_limit=1 * 1024**3)

# --- User-Agent helpers ---
try:
    ua_generator = UserAgent()
except FakeUserAgentError:
    ua_generator = None

FALLBACK_USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/16.0 Safari/605.1.15",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:128.0) Gecko/20100101 Firefox/128.0",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Edge/126.0.0.0",
]

def get_random_user_agent() -> str:
    return ua_generator.random if ua_generator else random.choice(FALLBACK_USER_AGENTS)

try:
    import brotlicffi as _brotli
except Exception:
    try:
        import brotli as _brotli  # type: ignore
    except Exception:
        _brotli = None

def _accept_encoding() -> str:
    # gzip/deflate altid; tilføj br hvis lib er tilgængelig
    return "gzip, deflate" + (", br" if _brotli is not None else "")

def _get_headers(base_headers: Optional[Dict] = None, user_agent: Optional[str] = None) -> Dict[str, str]:
    headers = {
        "User-Agent": user_agent or get_random_user_agent(),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
        "Accept-Encoding": _accept_encoding(),
        "Connection": "keep-alive",
        "Sec-CH-UA": '"Not/A)Brand";v="8", "Chromium";v="126"',
        "Sec-CH-UA-Mobile": "?0",
        "Sec-CH-UA-Platform": '"Windows"',
    }
    if base_headers:
        headers.update(base_headers)
    return headers

# --- Asset/CDN filtering for link checks ---
ASSET_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".gif", ".webp", ".svg", ".ico",
    ".css", ".js", ".pdf", ".xml", ".mp4", ".webm",
    ".woff", ".woff2", ".ttf", ".eot", ".zip", ".doc", ".docx",
}
CDN_PREFIXES = ("m2.", "cdn.", "media.", "assets.", "static.")

def _is_asset_url(url: str) -> bool:
    if not url:
        return False
    base = url.split("?", 1)[0].lower()
    return any(base.endswith(ext) for ext in ASSET_EXTENSIONS)

def _is_blocked_cdn(url: str) -> bool:
    try:
        host = urlparse(url).netloc.lower()
    except Exception:
        return False
    return any(host.startswith(p) for p in CDN_PREFIXES)

def should_check_link_status(url: str) -> bool:
    """Filter links before HEAD checks."""
    if not url:
        return False

    try:
        host = (urlsplit(url).hostname or "").strip()
    except Exception:
        host = ""

    if host and _is_private_ip_host(host):
        log.debug("Private/net-local host blokeret: %s", host)
        return False

    u = url.lower().split("?", 1)[0]

    # Skip ikke-HTTP(S)
    if not (u.startswith("http://") or u.startswith("https://")):
        return False

    if _is_asset_url(u):
        return False
    if _is_blocked_cdn(u):
        return False
    if "/img/" in u and "/resize/" in u:
        return False

    return True

# --- Heuristikker ---
def needs_rendering(html: str, url: str | None = None, content_type: str | None = None) -> bool:
    """Heuristik til at afgøre om vi bør eskalere til Playwright.
    - Eskalér KUN ved stærke SPA-markører (Next/Nuxt/React root/Vite/INITIAL_STATE).
    - Hvis WordPress (generator/wp-content/wp-includes) → SKIP SPA-eskalering,
      medmindre stærke markører også findes.
    - Behold særregler for kontakt-URL'er og ikke-HTML endpoints.
    """
    u = (url or "").lower()
    ct = (content_type or "").lower()

    # Skip kendte ikke-HTML endpoints (robots/sitemaps/rss/xml/txt)
    if u.endswith((".xml", ".txt")) or "sitemap" in u or "robots" in u or "rss" in u:
        return False
    if "xml" in ct or "text/plain" in ct:
        return False

    # Kontakt-/formular-URL’er er ofte JS-drevne → eskalér
    if u and any(k in u for k in ("kontakt", "contact", "kontaktformular", "support", "kundeservice", "formular", "form")):
        log.info("Contact-like URL detected - rendering required.")
        return True

    if not html:
        return True

    lower = html.lower()

    # WordPress-indikatorer (gate)
    is_wordpress = (
        "wp-content/" in lower
        or "wp-includes/" in lower
        or re.search(r'<meta[^>]+name=["\']generator["\'][^>]+wordpress', lower, re.I) is not None
    )

    # Stærke SPA-markører (kræves for eskalering)
    strong_spa_markers = (
        "__next_data__",          # Next.js bootstrap
        "data-reactroot",         # React root
        "__initial_state__",      # Hydration state (generelt)
        "window.__",              # Hydration namespace
        "__nuxt__",               # Nuxt
        "id=\"__nuxt\"",          # eksplicit container
        "vite",                   # Vite boot
        "data-server-rendered",   # Nuxt/Vue SSR-hint
    )
    has_strong = any(m in lower for m in strong_spa_markers)

    # WP: skip SPA-eskalering hvis ikke stærke markører også findes
    if is_wordpress and not has_strong:
        return False

    if has_strong:
        log.info("Strong SPA markers detected - rendering required.")
        return True

    # Fjern generiske triggers (script-densitet/short-HTML) for at undgå false positives
    # Behold state-initialization som sidste stærke signal
    if "__initial_state__" in lower or "window.__" in lower:
        log.info("State initialization patterns found - rendering required.")
        return True

    return False


RETRY_POLICY = AsyncRetrying(
    retry=retry_if_exception(
        lambda e: isinstance(e, (httpx.RequestError, asyncio.TimeoutError))
        or (isinstance(e, httpx.HTTPStatusError) and e.response.status_code >= 500)
    ),
    wait=wait_exponential_jitter(initial=0.5, max=8),
    stop=stop_after_attempt(5),
    reraise=True,
)

def _ssl_context(verify: bool, ca_bundle: Optional[Path] = None) -> ssl.SSLContext | bool:
    if not verify:
        return False
    if sys.platform == "win32" and ca_bundle is None:
        try:
            return ssl.create_default_context()
        except ssl.SSLError:
            log.warning("Kunne ikke loade Windows system-store, falder tilbage til certifi.")
    try:
        ctx = ssl.create_default_context(cafile=certifi.where())
        if ca_bundle:
            ctx.load_verify_locations(cafile=str(ca_bundle))
        return ctx
    except Exception:
        log.warning("Kunne ikke loade 'certifi' bundle – bruger systemets trust store.")
        return True

ALWAYS_HTTPX = {"www.googleapis.com", "pagespeedonline.googleapis.com", "www.google.com"}

def _looks_like_placeholder(html: Optional[str], url: str = "") -> bool:
    # Spring statiske filer over
    u = (url or "").lower()
    if u.endswith(('.xml', '.txt')) or 'sitemap' in u or 'robots' in u:
        log.debug(f"Skipping placeholder check for static file: {url}")
        return False

    if not html:
        log.debug("Placeholder: html is None/empty")
        return True

    lowered = html.lower()

    # 1) Kort/tynd HTML UDEN åbenlyst kontaktindhold
    if len(html) < 1200 and ("mailto:" not in lowered and "tel:" not in lowered):
        log.debug(f"Placeholder: short html len={len(html)} (no mailto/tel)")
        return True

    # 2) Kendte CF/challenge-fraser (udvidet)
    challenge_patterns = (
        "cdn-cgi/challenge-platform",
        "just a moment", "just a moment...",
        "attention required", "attention required!",
        "verifying your browser", "checking your browser",
        "making sure you're not a bot", "not a bot",
        "enable javascript", "please enable javascript",
        "access denied",
    )
    if any(p in lowered for p in challenge_patterns):
        log.warning("Placeholder: matched challenge pattern")
        return True

    return False

def is_bot_detected(html: str, url: str = "") -> bool:
    # Skip non-HTML like sitemaps
    if url.lower().endswith('.xml') or 'sitemap' in url.lower():
        log.debug(f"Skipping bot detection for XML or sitemap URL: {url}")
        return False

    lower = (html or "").lower()

    # Strammere “challenge”-mønstre
    challenge_core = any(p in lower for p in (
        "just a moment", "attention required", "verifying your browser",
        "checking your browser", "datadome", "ddom-js", "bot detection",
        "unusual traffic", "access denied",
        "making sure you're not a bot", "not a bot",
        "enable javascript", "please enable javascript"
    ))

    # 'cloudflare' alene er ikke nok — kræv cf + challenge for at dømme
    has_cf = "cloudflare" in lower
    if challenge_core:
        return True
    if has_cf:
        # kræv yderligere stærke indikatorer før vi dømmer
        return any(p in lower for p in ("just a moment", "attention required", "verifying your browser"))

    return False

# --------------------------------------------------------------------
# Early-return heuristics (bruges af hydratoren)
# --------------------------------------------------------------------
MIN_CONTENT_LEN = int(os.getenv("VEXTO_MIN_CONTENT_LEN", "15000"))
PRODUCT_URL_HINTS = ("/produkt", "/product", "/p/", "/products/")

def _looks_like_product(url: str) -> bool:
    u = (url or "").lower()
    return any(h in u for h in PRODUCT_URL_HINTS)

def _extract_core_signals_from_html(html: str) -> dict:
    out = {"canonical": None, "h1_count": 0, "product_schema": False, "breadcrumbs": False}
    if not html:
        return out
    try:
        soup = BeautifulSoup(html, "html.parser")
        link = soup.find("link", rel=lambda v: v and "canonical" in v.lower())
        if link and link.get("href"):
            out["canonical"] = link.get("href")
        out["h1_count"] = len(soup.find_all("h1"))
        for s in soup.find_all("script", type=lambda t: t and "ld+json" in t.lower()):
            txt = (s.string or s.get_text() or "").lower()
            if '"@type"' in txt:
                if '"product"' in txt:
                    out["product_schema"] = True
                if '"breadcrumblist"' in txt:
                    out["breadcrumbs"] = True
                if out["product_schema"] and out["breadcrumbs"]:
                    break
    except Exception:
        pass
    return out

def _core_html_ok(html: str, expect_product: bool) -> bool:
    if not html or len(html) < MIN_CONTENT_LEN:
        return False
    sig = _extract_core_signals_from_html(html)
    if expect_product:
        return bool(sig["canonical"]) and sig["product_schema"]
    return bool(sig["canonical"]) and sig["h1_count"] > 0

def _verify_state_content(page) -> dict:
    try:
        return page.evaluate("""
        () => {
            if (window.__INITIAL_STATE__) {
                const s = JSON.stringify(window.__INITIAL_STATE__);
                return {
                    hasState: true,
                    size: s.length,
                    hasCanonical: s.includes('canonical'),
                    hasProduct: s.includes('product'),
                    hasCategory: s.includes('category'),
                    hasPrice: s.includes('price')
                };
            }
            return { hasState: false };
        }
        """)
    except Exception:
        return {'hasState': False}

def _missing_core_signals(page, html: str) -> bool:
    try:
        state = _verify_state_content(page)
        dom_canon = False
        try:
            dom_canon = bool(page.evaluate("() => !!document.querySelector('link[rel=\"canonical\"]')"))
        except Exception:
            pass
        if state.get('hasCanonical') or dom_canon:
            return False
        if html:
            sig = _extract_core_signals_from_html(html)
            if sig["canonical"] or sig["product_schema"] or sig["breadcrumbs"]:
                return False
        return True
    except Exception:
        return True

FOOTER_SELECTORS = "footer,[role='contentinfo'],.footer,.page-footer,#footer,.site-footer,.global-footer"

def _wait_for_footer(page, timeout: int = 8000) -> bool:
    try:
        page.wait_for_selector(FOOTER_SELECTORS, timeout=timeout)
        log.info("✅ Footer detected - page fully loaded")
        return True
    except Exception:
        log.debug(f"Footer not found (selectors: {FOOTER_SELECTORS})")
        return False

# --------------------------------------------------------------------
# Playwright thread fetcher
# --------------------------------------------------------------------
class _PlaywrightThreadFetcher:
    def __init__(self):
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="pw-thread")
        self._playwright_instance: Optional[sync_playwright] = None
        self._browser: Optional[PlaywrightBrowser] = None
        self._browser_type: str = "chromium"
        self._is_ready: bool = False
        self._bootstrap_lock = threading.Lock()
        self._active_fetches: int = 0
        self._fetch_lock = threading.Lock()
        self._http_client = httpx.Client(
            headers={"Accept-Language": "en-US,en;q=0.9,da;q=0.8"},
            timeout=httpx.Timeout(45.0, connect=10.0),
            follow_redirects=True,
            http2=True,
            verify=_ssl_context(True),
        )

    def _bootstrap(self):
        if self._is_ready:
            return
        with self._bootstrap_lock:
            if self._is_ready:
                return
            if sys.platform == "win32":
                asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
            try:
                log.info(f"Playwright-browser starter i baggrundstråd ({self._browser_type})…")
                self._playwright_instance = sync_playwright().start()
                launch_args = [
                    "--disable-blink-features=AutomationControlled",
                    "--disable-web-security",
                    "--disable-site-isolation-trials",
                    "--disable-features=IsolateOrigins,site-per-process",
                    "--no-sandbox",
                    "--disable-infobars",
                    "--start-maximized",
                    "--disable-gpu",
                    "--window-size=1920,1080",
                    "--ignore-certificate-errors",
                    "--enable-features=NetworkService,NetworkServiceInProcess",
                ]
                self._browser = self._playwright_instance.chromium.launch(
                    headless=False,
                    args=launch_args,
                    ignore_default_args=['--enable-automation'],
                )
                self._is_ready = True
                log.info(f"Playwright-browser ({self._browser_type}) startet succesfuldt.")
            except Exception as e:
                log.error(f"FATAL: Kunne ikke starte Playwright-browser: {e}", exc_info=True)
                self._is_ready = False

    def start(self):
        """Start Playwright browser i baggrundstråd."""
        self._bootstrap()

    def _detect_pwa_site(self, url: str) -> bool:
        pwa_indicators = [
            'inventarland.dk', 'magento', 'vue-storefront', 'nuxt', 'react', 'pwa', 'spa', 'ssr', 'csr', 'vue', 'angular', 'svelte'
        ]
        return any(indicator in url.lower() for indicator in pwa_indicators)

    def _handle_cookies_advanced(self, page):
        try:
            cookie_selectors = [
                "button:has-text('Accepter')", "button:has-text('Godkend')",
                "button:has-text('Tillad')", "button:has-text('OK')",
                "button:has-text('Accept')", "button:has-text('Allow')",
                "button:has-text('Agree')", "button:has-text('Continue')",
                "button[aria-label*='accept' i]", "button[aria-label*='consent' i]",
                "button[aria-label*='cookie' i]",
                "#onetrust-accept-btn-handler", ".cc-btn.cc-accept",
                ".cookie-accept", ".btn-consent", "button.accept-cookies",
                "[id*='cookie'] button", "[class*='cookie'] button",
                "[data-testid*='cookie']", "[data-cy*='cookie']",
                "button[class*='accept']", "button[id*='accept']",
            ]
            for selector in cookie_selectors:
                try:
                    element = page.locator(selector).first
                    if element.is_visible(timeout=2000):
                        element.click(timeout=3000)
                        log.info(f"✅ Clicked cookie banner: {selector}")
                        page.wait_for_timeout(1000)
                        return
                except Exception:
                    continue
            log.info("No clickable cookie button found; hiding overlays via JS.")
            page.evaluate("""
            [...document.querySelectorAll(
                "[id*='cookie' i], [class*='cookie' i], [id*='consent' i], [class*='consent' i], [class*='overlay'], [id*='modal']"
            )].forEach(el => {
                if (el.style) {
                    el.style.display = 'none';
                    el.style.visibility = 'hidden';
                    el.style.opacity = '0';
                }
            });
            """)
        except Exception as e:
            log.warning(f"Cookie handling failed: {e}")

    def _force_lazy_load_images(self, page) -> int:
        try:
            page.evaluate("""
            const scrollHeight = document.body.scrollHeight;
            const step = 500;
            let currentPosition = 0;
            const scrollInterval = setInterval(() => {
              window.scrollTo(0, currentPosition);
              currentPosition += step;
              if (currentPosition >= scrollHeight) {
                clearInterval(scrollInterval);
                window.scrollTo(0, 0);
              }
            }, 100);
            """)
            page.wait_for_timeout(3000)
            images_loaded = page.evaluate("""
            (() => {
                let count = 0;
                const images = document.querySelectorAll('img[data-src], img[data-lazy], img[loading="lazy"], img[data-original]');
                images.forEach(img => {
                    const lazySrc = img.getAttribute('data-src') ||
                                    img.getAttribute('data-lazy') ||
                                    img.getAttribute('data-original');
                    if (lazySrc && !img.src.includes(lazySrc)) {
                        img.src = lazySrc;
                        img.removeAttribute('loading');
                        count++;
                    }
                    img.dispatchEvent(new Event('load'));
                });
                const elementsWithLazyBg = document.querySelectorAll('[data-bg], [data-background]');
                elementsWithLazyBg.forEach(el => {
                    const bg = el.getAttribute('data-bg') || el.getAttribute('data-background');
                    if (bg) {
                        el.style.backgroundImage = `url(${bg})`;
                        count++;
                    }
                });
                return count;
            })()
            """)
            if images_loaded > 0:
                log.info(f"✅ Force-loaded {images_loaded} lazy images")
                page.wait_for_timeout(2000)
            page.evaluate("""
            if (window.IntersectionObserver) {
                window.dispatchEvent(new Event('scroll'));
                window.dispatchEvent(new Event('resize'));
            }
            """)
            return images_loaded
        except Exception as e:
            log.warning(f"Lazy loading handler failed: {e}")
            return 0

    def _simulate_user_interaction_advanced(self, page):
        try:
            entry_x, entry_y = random.randint(100, 500), random.randint(100, 400)
            page.mouse.move(entry_x, entry_y)
            page.wait_for_timeout(random.randint(300, 800))
            page.mouse.click(entry_x + random.randint(-50, 50), entry_y + random.randint(-50, 50))
            page.wait_for_timeout(random.randint(500, 1000))

            scroll_positions = [200, 400, 600, 900, 1200]
            for pos in scroll_positions:
                page.mouse.wheel(0, pos)
                page.wait_for_timeout(random.randint(400, 900))

            if random.random() < 0.3:
                hover_x, hover_y = random.randint(200, 800), random.randint(200, 600)
                page.mouse.move(hover_x, hover_y)
                page.wait_for_timeout(random.randint(200, 500))

            page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
            page.wait_for_timeout(1500)
            page.evaluate("window.scrollTo({top: 0, behavior: 'smooth'})")
            page.wait_for_timeout(1000)

            _ = self._force_lazy_load_images(page)
            final_check = page.evaluate("""
            () => {
              const images = document.querySelectorAll('img');
              let loaded = 0;
              let total = images.length;
              images.forEach(img => {
                if (img.complete && img.naturalHeight !== 0) loaded++;
              });
              return { loaded, total, percentage: (loaded/total * 100).toFixed(1) };
            }
            """)
            log.info(f" Image loading status: {final_check['loaded']}/{final_check['total']} ({final_check['percentage']}%)")
        except Exception as e:
            log.warning(f"Advanced user simulation failed: {e}")

    def _force_vue_refresh(self, page):
        try:
            page.evaluate("""
            if (window.$nuxt && window.$nuxt.$router) {
              const currentRoute = window.$nuxt.$router.currentRoute;
              window.$nuxt.$router.replace(currentRoute.fullPath);
            }
            window.dispatchEvent(new Event('resize'));
            window.dispatchEvent(new Event('scroll'));
            if (window.Vue && window.Vue.nextTick) {
              window.Vue.nextTick(() => { console.log('Vue nextTick executed'); });
            }
            window.scrollTo(0, 1);
            window.scrollTo(0, 0);
            """)
            page.wait_for_timeout(3000)
            return page.wait_for_function(
                "() => window.__INITIAL_STATE__ && Object.keys(window.__INITIAL_STATE__).length > 0", timeout=5000
            )
        except Exception as e:
            log.warning(f"Vue refresh failed: {e}")
            raise e

    def _handle_pwa_hydration_hybrid(self, page, url: str, hydration_requests: list) -> Optional[str]:
        log.info(" Starting hybrid PWA hydration...")
        try:
            page.wait_for_function("""
                () => window.Vue || window.$nuxt || window.__NUXT__ ||
                document.querySelector('[data-server-rendered]') ||
                document.querySelector('[id*="app"]') || document.querySelector('[class*="vue"]')
            """, timeout=15000)
            log.info("✅ Vue/Nuxt framework detected")
        except Exception:
            log.warning("⚠️ Framework not detected")

        self._simulate_user_interaction_advanced(page)
        if hydration_requests:
            log.info(f" Waiting for {len(hydration_requests)} API requests to complete...")
            try:
                page.wait_for_function("() => document.readyState === 'complete'", timeout=10000)
            except Exception:
                pass

        strategies = [
            lambda: page.wait_for_function(
                "() => window.__INITIAL_STATE__ && typeof window.__INITIAL_STATE__ === 'object' && Object.keys(window.__INITIAL_STATE__).length > 0",
                timeout=8000
            ),
            lambda: page.wait_for_selector("script:has-text('__INITIAL_STATE__')", timeout=8000),
            lambda: page.wait_for_function(
                "() => window.__INITIAL_STATE__ && JSON.stringify(window.__INITIAL_STATE__).includes('canonical')",
                timeout=8000
            ),
            lambda: self._force_vue_refresh(page),
        ]
        successful_strategy = None
        skip_2_3 = False

        for i, strategy in enumerate(strategies, 1):
            if skip_2_3 and i in (2, 3):
                log.info(f"⏭️ Skipping strategy {i} - already sufficient after Strategy 1")
                continue

            try:
                log.info(f" Hydration strategy {i}/4...")
                strategy()
                log.info(f"✅ Strategy {i} succeeded!")
                successful_strategy = i
                try:
                    state_check = _verify_state_content(page)
                    if state_check.get('hasState'):
                        log.info(f" State verified: {state_check.get('size', 0)} chars, canonical: {state_check.get('hasCanonical', False)}")
                        if state_check.get('hasCanonical'):
                            try:
                                page.wait_for_timeout(400)
                            except Exception:
                                pass
                            return page.content()
                    if i == 1:
                        html_s1 = page.content()
                        expect_product = _looks_like_product(url)
                        state_ok_by_size = state_check.get('size', 0) > 50000
                        state_ok_by_signals = state_check.get('hasCanonical') or sum([
                            state_check.get('hasProduct', False),
                            state_check.get('hasCategory', False),
                            state_check.get('hasPrice', False),
                        ]) >= 2
                        dom_ok = _core_html_ok(html_s1, expect_product=expect_product)
                        if state_ok_by_signals or state_ok_by_size or dom_ok:
                            if _missing_core_signals(page, html_s1):
                                log.info("Hydration: Strategy 1 ok → skipping 2/3, checking Strategy 4 (missing core signals)")
                                skip_2_3 = True
                                # NEW: Early-return hvis vi allerede har canonical + nok indhold
                                if state_check.get('hasCanonical') and state_check.get('size', 0) > 20000:
                                    log.info("Early-return: canonical present and content size > 20000 → skipping Strategy 4.")
                                    return html_s1
                            else:
                                log.info("Hydration: Strategy 1 ok → skipping Strategies 2/3 (core signals present)")
                                return html_s1
                except Exception as e:
                    log.warning(f"State verification failed: {e}")

            except Exception as e:
                log.warning(f"❌ Strategy {i} failed: {e}")
                continue
        # Fallback: even if no early return happened we still want the HTML content.
        try:
            fallback_html = page.content()
            if successful_strategy:
                log.info(
                    "Hydration fallback returning page.content() after strategy %s",
                    successful_strategy,
                )
            else:
                log.info("Hydration fallback returning page.content() without successful strategy")
            return fallback_html
        except Exception as e:
            log.warning(f"Hydration fallback failed to fetch page content: {e}")
            return None

    def _extract_canonical_hybrid(self, page, html_content: str, url: str) -> dict:
        """Kombiner runtime + HTML parsing til canonical hints, med filtering og normalisering."""
        canonical_data: Dict[str, Any] = {}

        # Method 1: Runtime state (filter navigation/breadcrumb/header/footer)
        try:
            runtime_state = page.evaluate("() => window.__INITIAL_STATE__")
        except Exception:
            runtime_state = None

        if runtime_state:
            canonical_data['runtime_state'] = runtime_state
            canonical_data['runtime_success'] = True

            blocked = ("menucategories", "categoriesmap", "navigation", "header", "footer", "breadcrumb", "menu")
            def walk(obj, path=""):
                found = {}
                if isinstance(obj, dict):
                    for k, v in obj.items():
                        p = f"{path}.{k}" if path else k
                        if isinstance(v, (dict, list)):
                            found.update(walk(v, p))
                        elif isinstance(v, str) and "canonical" in k.lower():
                            if any(b in p.lower() for b in blocked):
                                continue
                            s = v.strip()
                            if s and s.lower() not in ("none", "null", "false"):
                                found[p] = s
                elif isinstance(obj, list):
                    for i, v in enumerate(obj):
                        p = f"{path}[{i}]"
                        if isinstance(v, (dict, list)):
                            found.update(walk(v, p))
                return found

            raw_fields = walk(runtime_state)
            normalized_fields: Dict[str, str] = {}
            for pth, val in raw_fields.items():
                cand = val if val.startswith("http") else urljoin(url, val)
                # samme domæne
                try:
                    if urlparse(cand).netloc == urlparse(url).netloc:
                        normalized_fields[pth] = cand
                except Exception:
                    continue
            if normalized_fields:
                canonical_data['canonical_fields'] = normalized_fields

        else:
            canonical_data['runtime_success'] = False

        # Method 2: traditionel <link rel="canonical">
        try:
            canonical_link = page.get_attribute('link[rel="canonical"]', 'href', timeout=2000)
        except Exception:
            canonical_link = None
        if canonical_link:
            canonical_data['link_canonical'] = canonical_link if canonical_link.startswith("http") else urljoin(url, canonical_link)

        # Method 3: og:url
        try:
            og_url = page.get_attribute("meta[property='og:url']", "content", timeout=1000)
            if og_url:
                canonical_data['og_url'] = og_url if og_url.startswith("http") else urljoin(url, og_url)
        except Exception:
            pass

        # Method 4: Regex fallback for inline state (let)
        try:
            state_pattern = r'window\.__INITIAL_STATE__\s*=\s*({.+?});'
            matches = re.findall(state_pattern, html_content or "", re.DOTALL)
            canonical_data['html_matches'] = len(matches)
        except Exception:
            canonical_data['html_matches'] = 0

        return canonical_data

    def _sync_fetch(self, url: str, ua: str, stealth_flag: bool, retry_count: int = 0) -> Optional[Dict[str, Any]]:
        with self._fetch_lock:
            self._active_fetches += 1
        try:
            if not self._is_ready:
                log.error("Playwright not ready; returning None.")
                return {"html": None, "canonical_data": {}}
            if not url or not urlparse(url).netloc:
                log.error(f"Invalid URL provided: {url}")
                return {"html": None, "canonical_data": {}}

            state_path = os.path.join(tempfile.gettempdir(), "vexto-states", f"{self._browser_type}_{urlparse(url).netloc}.json")
            os.makedirs(os.path.dirname(state_path), exist_ok=True)
            storage_state: Dict[str, Any] = {}
            if os.path.exists(state_path):
                try:
                    with open(state_path, 'r') as f:
                        storage_state = json.load(f)
                except json.JSONDecodeError as e:
                    log.warning(f"Invalid storage state file for {url}: {e}")

            is_pwa_site = self._detect_pwa_site(url)

            java_script_enabled = True
            if not is_pwa_site:
                # Kun preflight GET for ikke-PWA (P0.3)
                try:
                    headers = _get_headers({"User-Agent": ua})
                    response = self._http_client.get(url, headers=headers)
                    html = response.text

                    # NEW: force rendering ved placeholder/CF, og support env override
                    placeholder = _looks_like_placeholder(html, url)
                    force_pw = os.getenv("VEXTO_FORCE_PW", "0").lower() in {"1","true","yes","on"}

                    java_script_enabled = force_pw or placeholder or needs_rendering(html) or is_pwa_site

                    if not java_script_enabled:
                        # Stadig statisk → returnér kun hvis vi IKKE ser bot/challenge
                        if is_bot_detected(html, url):
                            log.error(f"❌ Bot detection in static GET for {url}")
                            return {"html": None, "canonical_data": {}}
                        log.debug(f"Preflight static OK (len={len(html) if html else 0}) → no render: {url}")
                        return {"html": html, "canonical_data": {}}
                    else:
                        reason = "force_pw" if force_pw else ("placeholder" if placeholder else ("needs_rendering" if needs_rendering(html) else "pwa"))
                        log.info(f"Escalating to Playwright for {url}: reason={reason}")
                except Exception:
                    java_script_enabled = True  # eskaler

            viewport_config = {"width": 1366, "height": 768} if is_pwa_site else {"width": 1920, "height": 1080}
            with self._browser.new_context(
                storage_state=storage_state if storage_state else None,
                user_agent=ua,
                locale="da-DK",
                viewport=viewport_config,
                screen={"width": 1920, "height": 1080},
                java_script_enabled=java_script_enabled,
                bypass_csp=True,
                permissions=['geolocation'],
                geolocation={'latitude': 55.6761, 'longitude': 12.5683},
                timezone_id='Europe/Copenhagen',
            ) as context:
                page = context.new_page()

                # --- Call-time gate (log beslutningen)
                _raw = _stealth_raw()
                _enabled = _raw in ("1","true","yes","on")
                log.info(f"[stealth] gate: VEXTO_STEALTH={_raw!r} -> {'ON' if _enabled else 'OFF'}")

                # --- Stealth besluttet ved call-tid (override > env) ---
                resolved_stealth = bool(stealth_flag)
                log.info(f"[stealth] resolved={resolved_stealth} (override), env={os.getenv('VEXTO_STEALTH','?')!r}")
                if STEALTH_AVAILABLE and resolved_stealth:
                    stealth_sync(page)
                    log.info("✅ playwright-stealth applied (resolved=ON)")
                else:
                    log.info("playwright-stealth skipped (resolved=OFF or not available)")

                # Anti-bot init-script KUN når stealth er ON
                if resolved_stealth:
                    page.add_init_script("""
                    Object.defineProperty(navigator, 'webdriver', { get: () => undefined });
                    delete window.cdc_adoQpoasnfa76pfcZLmcfl_Array;
                    delete window.cdc_adoQpoasnfa76pfcZLmcfl_Promise;
                    delete window.cdc_adoQpoasnfa76pfcZLmcfl_Symbol;
                    window.chrome = { runtime: {}, loadTimes: function() {}, csi: function() {}, app: {} };
                    const originalQuery = window.navigator.permissions.query;
                    if (originalQuery) {
                    window.navigator.permissions.query = (parameters) => (
                        parameters.name === 'notifications' ?
                        Promise.resolve({ state: 'default' }) :
                        originalQuery(parameters)
                    );
                    }
                    Object.defineProperty(navigator, 'plugins', { get: () => [1, 2, 3, 4, 5], });
                    Object.defineProperty(navigator, 'languages', { get: () => ['da-DK', 'da', 'en-US', 'en'], });
                    window.outerHeight = screen.height;
                    window.outerWidth = screen.width;
                    """)

                hydration_requests: list[str] = []
                if is_pwa_site:
                    def handle_request(request):
                        request_url = request.url.lower()
                        if any(keyword in request_url for keyword in ['graphql', '/api/', 'ajax', 'json']):
                            hydration_requests.append(request.url)
                            log.debug(f" Detected API call: {request.url}")
                    page.on('request', handle_request)

                log.info(f" Playwright fetching {url} (PWA: {is_pwa_site}, retry: {retry_count})...")
                try:
                    timeout = 90000 if is_pwa_site else 60000
                    page.goto(url, wait_until="domcontentloaded", timeout=timeout)
                    try:
                        page.wait_for_load_state("networkidle", timeout=15000)
                    except Exception:
                        log.debug("Network idle timeout - continuing")
                except Exception as e:
                    log.warning(f"Navigation failed: {e}")
                    return {"html": None, "canonical_data": {}}

                self._handle_cookies_advanced(page)

                if is_pwa_site:
                    html_content = self._handle_pwa_hydration_hybrid(page, url, hydration_requests)
                else:
                    self._simulate_user_interaction_advanced(page)
                    _wait_for_footer(page, timeout=10000)
                    html_content = page.content()

                if not html_content:
                    # Grace: kort vent og nyt forsøg
                    try:
                        page.wait_for_timeout(1500)
                        html_content = page.content()
                    except Exception:
                        html_content = ""
                    # Last resort: hent hele DOM'en via JS
                    if not html_content:
                        try:
                            html_content = page.evaluate("() => document.documentElement.outerHTML") or ""
                        except Exception:
                            pass
                    if not html_content:
                        # Grace: kort vent og nyt forsøg
                        try:
                            page.wait_for_timeout(1500)
                            html_content = page.content()
                        except Exception:
                            html_content = ""
                        # Last resort: hent hele DOM'en via JS
                        if not html_content:
                            try:
                                html_content = page.evaluate("() => document.documentElement.outerHTML") or ""
                            except Exception:
                                pass
                        if not html_content:
                            return {"html": None, "canonical_data": {}}

                canonical_data = self._extract_canonical_hybrid(page, html_content, url)

                # --- NYT: Analytics runtime-hook ---
                try:
                    analytics = page.evaluate("""
                    () => {
                    const out = {
                        hasGA4: false, hasGTM: false, hasFB: false,
                        ga4Id: null, fbPixelId: null,
                        // ekstra id’er vi kan finde
                        gtmContainerId: null, ttPixelId: null, pinterestTagId: null,
                        snapPixelId: null, linkedinPartnerId: null, bingUetId: null
                    };

                    // dataLayer / GTM / GA4
                    try {
                        if (typeof dataLayer !== 'undefined' && Array.isArray(dataLayer)) {
                        out.hasGTM = true;
                        for (const item of dataLayer) {
                            if (!item || typeof item !== 'object') continue;
                            const s = JSON.stringify(item);
                            const g = s.match(/G-[A-Z0-9\-]{6,}/);
                            if (g && !out.ga4Id) out.ga4Id = g[0];
                            const sendTo = s.match(/send_to["']?\s*:\s*["'](G-[A-Z0-9\-]{6,})["']/i);
                            if (sendTo && !out.ga4Id) out.ga4Id = sendTo[1];
                        }
                        }
                    } catch(e){}

                    // gtag/gtag.js
                    try { if (typeof gtag !== 'undefined') out.hasGA4 = true; } catch(e){}
                    try {
                        const scripts = Array.from(document.scripts || []);
                        for (const s of scripts) {
                        const src = s.src || "";
                        const txt = s.innerHTML || "";
                        if (src.includes('googletagmanager.com/gtag/js')) {
                            out.hasGA4 = true;
                            const m = src.match(/id=(G-[A-Z0-9\-]{6,})/i);
                            if (m && !out.ga4Id) out.ga4Id = m[1];
                        }
                        if (src.includes('googletagmanager.com/gtm.js')) {
                            out.hasGTM = true;
                            const m2 = src.match(/id=(GTM-[A-Z0-9]+)/i);
                            if (m2 && !out.gtmContainerId) out.gtmContainerId = m2[1];
                        }
                        if (txt.includes("gtag(")) out.hasGA4 = true;
                        }
                    } catch(e){}

                    // Meta Pixel (fbq)
                    try {
                        if (typeof fbq !== 'undefined') out.hasFB = true;
                        const ns = Array.from(document.getElementsByTagName('noscript'));
                        for (const n of ns) {
                        const t = n.innerHTML || "";
                        if (t.includes('facebook.com/tr')) out.hasFB = true;
                        }
                        const scripts = Array.from(document.scripts || []);
                        for (const s of scripts) {
                        const txt = s.innerHTML || "";
                        const m = txt.match(/fbq\(['"]init['"]\s*,\s*['"](\d+)['"]\)/i);
                        if (m && !out.fbPixelId) out.fbPixelId = m[1];
                        }
                    } catch(e){}

                    // Andre udbredte pixels (let signaturjagt)
                    try {
                        const html = document.documentElement.outerHTML;
                        const find = (re) => { const m = html.match(re); return m ? m[1] || m[0] : null; };
                        // TikTok
                        out.ttPixelId = out.ttPixelId || find(/tiktok(?:Pixel)?Id["']?\s*[:=]\s*["']?([A-Z0-9_:-]{5,})/i);
                        // Pinterest
                        out.pinterestTagId = out.pinterestTagId || find(/pin(?:terest)?(?:Tag)?Id["']?\s*[:=]\s*["']?([A-Z0-9_:-]{4,})/i);
                        // Snapchat
                        out.snapPixelId = out.snapPixelId || find(/snap(?:chat)?PixelId["']?\s*[:=]\s*["']?([A-Z0-9_:-]{4,})/i);
                        // LinkedIn Insight
                        out.linkedinPartnerId = out.linkedinPartnerId || find(/linkedin(?:Insight)?Id["']?\s*[:=]\s*["']?(\d{4,})/i);
                        // Bing UET
                        out.bingUetId = out.bingUetId || find(/(UET-\w{6,}|bingUetId["']?\s*[:=]\s*["']?([A-Z0-9\-]{6,}))/i);
                    } catch(e){}

                    return out;
                    }
                    """)
                except Exception:
                    analytics = {}

                # Pak analytics ind i canonical_data som analyzer forventer
                try:
                    if isinstance(canonical_data, dict):
                        canonical_data["analytics"] = analytics
                        # Dupliker top-level hints for analyzer.pick(...)
                        # GA4
                        if analytics.get("ga4Id"):
                            canonical_data["ga4Id"] = analytics["ga4Id"]
                            canonical_data["ga_measurement_id"] = analytics["ga4Id"]
                        # GTM
                        if analytics.get("gtmContainerId"):
                            canonical_data["gtmContainerId"] = analytics["gtmContainerId"]
                            canonical_data["gtm"] = True
                            canonical_data["hasGTM"] = True
                        # Meta Pixel
                        if analytics.get("fbPixelId"):
                            canonical_data["fbPixelId"] = analytics["fbPixelId"]
                        # TikTok / Pinterest / Snapchat / LinkedIn / Bing UET
                        for k in ("ttPixelId","pinterestTagId","snapPixelId","linkedinPartnerId","bingUetId"):
                            if analytics.get(k):
                                canonical_data[k] = analytics[k]
                except Exception:
                    pass

                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                domain = urlparse(url).netloc.replace('.', '_')
                path = urlparse(url).path.replace('/', '_') or 'index'
                html_filename = f"debug_{domain}_{path}_{timestamp}.html"
                try:
                    with open(html_filename, "w", encoding="utf-8") as f:
                        f.write(html_content)
                    log.info(f" Saved debug HTML: {html_filename}")
                except Exception:
                    pass

                if is_bot_detected(html_content, url):
                    # Soft-fail: bevar PW-HTML hvis der er stærke signaler
                    try:
                        sig = _extract_core_signals_from_html(html_content)
                        strong = (len(html_content) >= max(20000, MIN_CONTENT_LEN // 2)) and (sig.get("canonical") or sig.get("h1_count", 0) > 0)
                    except Exception:
                        strong = False

                    if strong:
                        log.warning(f"Bot heuristics tripped on {url}, but core signals present (keeping Playwright HTML).")
                    else:
                        log.error(f"❌ Bot detection in final HTML for {url} (dropping)")
                        return {"html": None, "canonical_data": {}, "reason": "bot_detected"}

                try:
                    storage_state = context.storage_state()
                    with open(state_path, 'w') as f:
                        json.dump(storage_state, f)
                except Exception as e:
                    log.warning(f"Storage state save failed: {e}")

                # ← VIGTIGT: returnér ALTID Playwright-resultatet her
                return {
                    "html": html_content,
                    "canonical_data": canonical_data,
                    "stealth_applied": resolved_stealth
                }


        except Exception as e:
            log.error(f"Fatal error for {url}: {e}", exc_info=True)
            return {"html": None, "canonical_data": {}}
        finally:
            with self._fetch_lock:
                self._active_fetches -= 1

    async def fetch(self, url: str, ua: str, stealth_flag: bool) -> Optional[Dict[str, Any]]:
        if not self._is_ready:
            return {"html": None, "canonical_data": {}}
        return await asyncio.get_running_loop().run_in_executor(
            self._executor, self._sync_fetch, url, ua, stealth_flag
        )

    async def close(self):
        def _sync_close():
            if self._active_fetches > 0:
                log.warning(f"Waiting for {self._active_fetches} active fetches to complete before closing...")
            while self._active_fetches > 0:
                time.sleep(0.1)
            try:
                if self._browser:
                    self._browser.close()
                if self._playwright_instance:
                    self._playwright_instance.stop()
            finally:
                self._is_ready = False
                if hasattr(self, '_http_client'):
                    self._http_client.close()

        if getattr(self._executor, "_shutdown", False):
            return
        try:
            await asyncio.get_running_loop().run_in_executor(self._executor, _sync_close)
        finally:
            self._executor.shutdown(wait=True)
            profile_root = os.path.join(tempfile.gettempdir(), "vexto-states")
            try:
                if os.path.exists(profile_root):
                    for dir_path in os.listdir(profile_root):
                        full_path = os.path.join(profile_root, dir_path)
                        if os.path.getmtime(full_path) < time.time() - 7 * 86400:
                            shutil.rmtree(full_path, ignore_errors=True)
                            log.info(f"Cleaned up old profile: {dir_path}")
            except Exception as e:
                log.warning(f"Failed to clean up profiles: {e}")

# --------------------------------------------------------------------
# Top-level batch HEAD with filtering (bruges bl.a. af image_fetchers)
# --------------------------------------------------------------------
async def batch_head_requests(
    client: "AsyncHtmlClient",
    urls: List[str],
    *,
    timeout: float = 5.0,
    concurrency: int = 10,
    follow_redirects: bool = True,
    cap: int = 50,
) -> List[dict]:
    """
    Returnerer liste af dicts: {url, status, content_length, content_type, final_url}
    - Filtrerer assets/CDN/resize-URLs
    - Dedupper + cap'er
    """
    if not urls:
        return []

    # filter + dedup + cap
    seen = set()
    to_probe: List[str] = []
    for u in urls:
        if not u or not should_check_link_status(u):
            continue
        if u not in seen:
            seen.add(u)
            to_probe.append(u)
        if len(to_probe) >= cap:
            break

    sem = asyncio.Semaphore(concurrency)

    async def _probe(u: str) -> dict:
        async with sem:
            # 1) Prøv HEAD med eksponentiel backoff (3 forsøg)
            for attempt in range(3):
                try:
                    resp = await client.head(u, timeout=timeout, follow_redirects=follow_redirects)
                    if resp is not None and resp.status_code < 400:
                        cl = resp.headers.get("content-length")
                        ct = resp.headers.get("content-type")
                        return {
                            "url": u,
                            "status": resp.status_code,
                            "content_length": int(cl) if cl and str(cl).isdigit() else None,
                            "content_type": ct,
                            "final_url": str(getattr(resp, "url", u)),
                        }
                    # HEAD ikke tilladt eller 4xx/5xx → fald tilbage til GET
                    break
                except Exception:
                    # Backoff ved fejl (fx ConnectionTerminated)
                    await asyncio.sleep(0.5 * (2 ** attempt))

            # 2) Fallback: let GET med Range-header (minimér payload)
            try:
                r2 = await client.httpx_get(
                    u,
                    timeout=timeout,
                    follow_redirects=follow_redirects,
                    headers={"Range": "bytes=0-0"},
                )
                cl2 = r2.headers.get("content-length")
                ct2 = r2.headers.get("content-type")
                return {
                    "url": u,
                    "status": r2.status_code,
                    "content_length": int(cl2) if cl2 and str(cl2).isdigit() else None,
                    "content_type": ct2,
                    "final_url": str(getattr(r2, "url", u)),
                }
            except Exception:
                return {"url": u, "status": None, "content_length": None, "content_type": None, "final_url": None}

    return await asyncio.gather(*(_probe(u) for u in to_probe))

# --------------------------------------------------------------------
# Public HTTP/HTML client (async)
# --------------------------------------------------------------------
class AsyncHtmlClient:
    def __init__(
        self,
        *,
        max_connections: int = 10,
        total_timeout: float = 45.0,
        verify_ssl: bool = True,
        ca_bundle: Optional[Path] = None,
        proxy: Optional[str] = None,
        stealth: Optional[bool] = None,   # <- NY: per-klient override
    ):
        self._sem = asyncio.Semaphore(max_connections)
        proxy = proxy or os.getenv("HTTP_PROXY")
        self._httpx_client = httpx.AsyncClient(
            headers={"Accept-Language": "en-US,en;q=0.9,da;q=0.8"},
            timeout=httpx.Timeout(total_timeout, connect=10.0, read=total_timeout, write=total_timeout),
            follow_redirects=True,
            http2=True,
            verify=_ssl_context(verify_ssl, ca_bundle),
            proxy=proxy,
            limits=httpx.Limits(max_connections=30, max_keepalive_connections=10),
        )
        self._pw_thread = _PlaywrightThreadFetcher()
        self._url_locks: Dict[str, asyncio.Lock] = {}
        self.last_fetch_method: Optional[str] = None
        self._is_closed: bool = False
        self.user_agent = get_random_user_agent()
        self._stealth_override: Optional[bool] = stealth
        self.last_stealth_resolved: Optional[bool] = None

        # P0.2: Domæne-cache for robots/sitemap
        self._domain_cache: Dict[str, Dict[str, Optional[str]]] = {}  # {domain: {"robots": str|None, "sitemap": str|None}}

    def _check_analytics_runtime(self, page) -> dict:
        """
        Kører i Playwright-runtime og detekterer GA4/GTM/Meta Pixel + forsøger at udtrække ID'er.
        Returnerer et sikkert dict (fail-closed).
        """
        import logging
        log = logging.getLogger(__name__)
        try:
            analytics = page.evaluate("""
                () => {
                    const out = {
                        hasGA4: false,
                        hasGTM: false,
                        hasFB: false,
                        ga4Id: null,
                        fbPixelId: null,
                        details: []
                    };

                    try {
                        if (typeof gtag !== 'undefined') { out.hasGA4 = true; out.details.push('gtag found'); }
                    } catch(e) {}

                    try {
                        if (typeof ga !== 'undefined')   { out.hasGA4 = true; out.details.push('ga (UA) found'); }
                    } catch(e) {}

                    try {
                        if (typeof dataLayer !== 'undefined' && Array.isArray(dataLayer)) {
                            out.hasGTM = true; out.details.push('dataLayer found');
                            for (const item of dataLayer) {
                                if (item && typeof item === 'object') {
                                    const s = JSON.stringify(item);
                                    const m = s.match(/G-[A-Z0-9]+/);
                                    if (m && !out.ga4Id) out.ga4Id = m[0];
                                }
                            }
                        }
                    } catch(e) {}

                    try {
                        if (typeof fbq !== 'undefined')  { out.hasFB  = true; out.details.push('fbq found'); }
                    } catch(e) {}
                    try {
                        if (typeof _fbq !== 'undefined') { out.hasFB  = true; out.details.push('_fbq found'); }
                    } catch(e) {}

                    try {
                        const scripts = Array.from(document.getElementsByTagName('script'));
                        for (const s of scripts) {
                            const src = s.src || '';
                            const txt = s.innerHTML || '';

                            if (src.includes('googletagmanager.com/gtag/js')) {
                                out.hasGA4 = true;
                                const idm = src.match(/id=(G-[A-Z0-9]+)/);
                                if (idm) out.ga4Id = out.ga4Id || idm[1];
                            }
                            if (src.includes('googletagmanager.com/gtm.js')) {
                                out.hasGTM = true;
                            }
                            if (src.includes('connect.facebook.net') && src.includes('fbevents.js')) {
                                out.hasFB = true;
                            }

                            if (txt.includes('gtag(')) out.hasGA4 = true;

                            const pxm = txt.match(/fbq\\('init'\\s*,\\s*'(\\d+)'\\)/);
                            if (pxm && !out.fbPixelId) { out.hasFB = true; out.fbPixelId = pxm[1]; }
                        }
                    } catch(e) {}

                    try {
                        const nos = Array.from(document.getElementsByTagName('noscript'));
                        for (const n of nos) {
                            const t = n.innerHTML || '';
                            if (t.includes('facebook.com/tr')) out.hasFB = true;
                            if (t.includes('googletagmanager.com')) out.hasGTM = true;
                        }
                    } catch(e) {}

                    return out;
                }
            """)

            # 🔧 Korrekt Python-hale (ingen JS-try/catch her)
            if not isinstance(analytics, dict):
                analytics = {}

            return analytics

        except Exception as e:
            log.warning(f"Analytics runtime check failed: {e}")
            return {
                'hasGA4': False,
                'hasGTM': False,
                'hasFB': False,
                'ga4Id': None,
                'fbPixelId': None,
                'details': []
            }



    async def startup(self):
        await asyncio.get_running_loop().run_in_executor(self._pw_thread._executor, self._pw_thread.start)

    async def __aenter__(self) -> "AsyncHtmlClient":
        await self.startup()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()

    async def _check_if_closed(self, url: str) -> bool:
        if self._is_closed:
            log.warning(f"Kald forsøgt på lukket klient: {url}")
            return True
        return False

    # --- httpx wrappers ---
    async def get(self, url: str, **kwargs) -> httpx.Response:
        return await self.httpx_get(url, **kwargs)

    async def get_response(self, url: str, **kwargs) -> httpx.Response:
        return await self.httpx_get(url, **kwargs)

    async def httpx_get(self, url: str, **kwargs) -> httpx.Response:
        async with self._sem:
            headers = _get_headers(kwargs.pop("headers", None), user_agent=self.user_agent)
            return await self._httpx_client.get(url, headers=headers, **kwargs)

    async def head(self, url: str, **kwargs) -> Optional[httpx.Response]:
        if await self._check_if_closed(url):
            return None
        headers = _get_headers(kwargs.pop("headers", None), user_agent=self.user_agent)

        for attempt in range(3):
            async with self._sem:
                try:
                    resp = await self._httpx_client.request("HEAD", url, headers=headers, **kwargs)
                    resp.elapsed_time_ms = int(resp.elapsed.total_seconds() * 1000)

                    # Fallback til GET hvis serveren ikke understøtter HEAD
                    if resp.status_code in (405, 501) or resp.status_code is None:
                        g_headers = {**headers, "Range": "bytes=0-0"}
                        get_resp = await self._httpx_client.request("GET", url, headers=g_headers, **kwargs)
                        get_resp.elapsed_time_ms = int(get_resp.elapsed.total_seconds() * 1000)
                        return get_resp

                    return resp

                except (httpx.RemoteProtocolError, httpx.ProtocolError) as e:
                    if attempt < 2:
                        await asyncio.sleep(0.5 * (2 ** attempt))
                        continue
                    log.warning(f"HEAD request til {url} fejlede permanent: {e}")
                    return None

                except httpx.RequestError as e:
                    if attempt < 2 and "ConnectionTerminated" in str(e):
                        await asyncio.sleep(0.5 * (2 ** attempt))
                        continue
                    log.warning(f"HEAD request til {url} fejlede: {e}")
                    return None

    # --- P0.2: Domain-level robots/sitemap cache ---
    def _domain_key(self, base_url: str) -> str:
        try:
            return urlparse(base_url).netloc.lower()
        except Exception:
            return ""

    async def get_robots_txt(self, base_url: str) -> Optional[str]:
        domain = self._domain_key(base_url)
        if not domain:
            return None
        cache = self._domain_cache.setdefault(domain, {})
        if "robots" in cache:
            return cache["robots"]
        robots_url = urljoin(f"https://{domain}", "/robots.txt")
        try:
            r = await self.httpx_get(robots_url, timeout=10)
            content = r.text if r.status_code == 200 else None
        except Exception:
            content = None
        cache["robots"] = content
        return content

    async def get_sitemap_url(self, base_url: str) -> Optional[str]:
        domain = self._domain_key(base_url)
        if not domain:
            return None
        cache = self._domain_cache.setdefault(domain, {})
        if "sitemap" in cache:
            return cache["sitemap"]

        robots = await self.get_robots_txt(base_url)
        if robots:
            for line in robots.splitlines():
                if line.lower().startswith("sitemap:"):
                    sm = line.split(":", 1)[1].strip()
                    cache["sitemap"] = sm
                    return sm

        for path in ("/sitemap.xml", "/sitemap/sitemap.xml", "/sitemap_index.xml", "/wp-sitemap.xml"):
            candidate = urljoin(f"https://{domain}", path)
            try:
                h = await self.head(candidate, timeout=8, follow_redirects=True)
                if h and h.status_code < 400:
                    cache["sitemap"] = candidate
                    return candidate
            except Exception:
                continue

        cache["sitemap"] = None
        return None

    # --- Metrics helpers ---
    async def check_sitemap_freshness(self, url: str) -> bool:
        try:
            response = await self.httpx_get(url)
            if response.status_code != 200:
                log.warning(f"Sitemap {url} returnerede status {response.status_code} (status: missing)")
                return False
            html = response.text
            if not html:
                log.warning(f"No sitemap content for {url} (status: missing)")
                return False
            import xml.etree.ElementTree as ET
            try:
                root = ET.fromstring(html)
                if root.tag.endswith('sitemapindex'):
                    for sitemap in root.findall(".//{http://www.sitemaps.org/schemas/sitemap/0.9}sitemap"):
                        loc = sitemap.find("{http://www.sitemaps.org/schemas/sitemap/0.9}loc")
                        if loc is not None and loc.text:
                            if await self.check_sitemap_freshness(loc.text):
                                return True
                    log.info(f"Sitemap index {url} has no fresh subsitemaps (status: missing)")
                    return False

                for url_elem in root.findall(".//{http://www.sitemaps.org/schemas/sitemap/0.9}url"):
                    lastmod = url_elem.find("{http://www.sitemaps.org/schemas/sitemap/0.9}lastmod")
                    if lastmod is not None and lastmod.text:
                        txt = lastmod.text.strip()
                        # simple ISO parsing
                        txt = txt.replace("Z", "+00:00")
                        try:
                            dt = datetime.fromisoformat(txt)
                        except Exception:
                            m = re.match(r"(\d{4}-\d{2}-\d{2})", txt)
                            dt = datetime.fromisoformat(m.group(1) + "T00:00:00+00:00") if m else None
                        if dt and (datetime.now(timezone.utc) - dt.astimezone(timezone.utc)).days < 30:
                            log.info(f"Sitemap {url} is fresh (lastmod: {txt}) (status: ok)")
                            return True
                log.info(f"Sitemap {url} has no recent or valid lastmod dates (status: missing)")
                return False
            except Exception as e:
                log.warning(f"Could not parse sitemap {url}: {e} (status: error)")
                return False
        except httpx.HTTPError as e:
            log.warning(f"Failed to fetch sitemap {url}: {e} (status: error)")
            return False

    async def check_analytics(self, url: str) -> Dict[str, bool]:
        
        html_result = await self.get_raw_html(url)
        html = None
        if isinstance(html_result, str):
            html = html_result
        elif isinstance(html_result, tuple) and len(html_result) >= 1:
            html = html_result[0]
        elif isinstance(html_result, dict) and html_result.get('html'):
            html = html_result['html']

        if not html:
            log.info(f"No HTML for analytics check on {url} (status: missing)")
            return {"has_ga4": False, "has_meta_pixel": False, "has_gtm": False}

        text = html or ""

        patterns = {
            "gtm": [
                r"googletagmanager\.com/gtm\.js",
                r"\bGTM-[A-Z0-9]{6,8}\b",
                r"\b(?:window\.)?dataLayer\s*=\s*(?:window\.)?dataLayer\s*\|\|\s*\[",  # init-variant
                r"dataLayer\.push\s*\(",
            ],
            "ga4": [
                r"(?:https?:\/\/)?(?:www\.)?googletagmanager\.com/gtag/js\?id=G-[A-Z0-9]{4,16}",
                r"gtag\s*\(\s*['\"]js['\"]\s*,",
                r"gtag\s*\(\s*['\"]config['\"]\s*,\s*['\"]G-[A-Z0-9]{4,16}['\"]",
            ],
            "ga4_soft": [
                r"\bG-[A-Z0-9]{4,16}\b",
                r"(?:measurement_id|measurementId)['\"]?\s*:\s*['\"]G-[A-Z0-9]{4,16}['\"]",
            ],
            "meta": [
                r"\bf(bq|_fbq)\s*\(",
                r"connect\.facebook\.net/.*/fbevents\.js",
                r"facebook\.com/tr\?",
                r"(?:pixelId|pixel_id)",
            ],
        }

        def any_match(pats: list[str]) -> bool:
            return any(re.search(p, text, re.IGNORECASE) for p in pats)

        has_gtm = any_match(patterns["gtm"])
        has_ga4 = any_match(patterns["ga4"]) or any_match(patterns["ga4_soft"])
        # Hvis GTM er til stede, så prøv en ekstra GA4-config match (via GTM-template)
        if has_gtm and not has_ga4:
            if re.search(r"gtag\s*\(\s*['\"]config['\"]\s*,\s*['\"]G-[A-Z0-9]{6,12}['\"]", text, re.IGNORECASE):
                has_ga4 = True

        has_meta_pixel = any_match(patterns["meta"])

        log.info(f"Analytics check for {url}: GA4={has_ga4}, Meta Pixel={has_meta_pixel}, GTM={has_gtm} (status: ok)")
        return {"has_ga4": bool(has_ga4), "has_meta_pixel": bool(has_meta_pixel), "has_gtm": bool(has_gtm)}


    async def test_response_time(self, url: str) -> float:
        start = time.time()
        response = await self.httpx_get(url)
        elapsed = (time.time() - start) * 1000
        log.info(f"Raw response time for {url}: {elapsed:.2f} ms")
        return elapsed

    async def get_raw_html(
        self,
        url: str,
        force: bool = False,
        return_soup: bool = False,
        force_playwright: bool = False,
        depth: int = 0
    ) -> Union[str, tuple[BeautifulSoup, Dict[str, Any]], Dict[str, Any], None]:
        """
        Returnerer:
         - return_soup=True: (BeautifulSoup, canonical_data)
         - Playwright-case: {"html": str, "canonical_data": dict}
         - Ellers: str (HTML) eller None
        """
        if self._is_closed:
            log.warning(f"Forsøg på at kalde get_raw_html på lukket klient: {url}")
            return None
        if depth > 1:
            log.warning(f"Max depth reached for {url}; stopping.")
            return None

        if urlparse(url).netloc in ALWAYS_HTTPX:
            try:
                html = (await self.httpx_get(url)).text
                if not is_bot_detected(html, url):
                    html_cache.set(url, html, expire=3600)
                if return_soup:
                    return (BeautifulSoup(html, "lxml"), {})
                return html
            except Exception as e:
                log.warning(f"HTTPX fetch failed for {url}: {e}")
                return None

        if not force and not force_playwright:
            cached = html_cache.get(url)
            if cached is not None:
                log.debug(f"Cache hit for {url}")
                if return_soup:
                    return (BeautifulSoup(cached, "lxml"), {})
                return cached

        lock = self._url_locks.setdefault(url, asyncio.Lock())
        async with lock:
            if not force and not force_playwright:
                cached = html_cache.get(url)
                if cached is not None:
                    if return_soup:
                        return (BeautifulSoup(cached, "lxml"), {})
                    return cached

            html_content: Optional[str] = None
            canonical_data: Dict[str, Any] = {}

            try:
                request = httpx.Request("GET", url)
                if force_playwright:
                     raise httpx.RequestError(
                        "Playwright blev tvunget manuelt for at få renderet indhold.",
                        request=request,
                    )

                response = await self.httpx_get(url)
                html_content = response.text
                self.last_fetch_method = "httpx"

                if _looks_like_placeholder(html_content, url):
                    raise httpx.RequestError(
                        "Placeholder content; escalate to Playwright",
                        request=request,
                    )
                if needs_rendering(html_content) and not (url.lower().endswith(('.xml', '.txt')) or 'sitemap' in url.lower() or 'robots' in url.lower()):
                     raise httpx.RequestError(
                        "Dynamic content; escalate to Playwright",
                        request=request,
                    )

            except (httpx.RequestError, RetryError) as e:
                log.info(f"Escalating to Playwright for {url}: {e}")
                # Beslut stealth ved call-tid: override > env
                stealth_flag = self._stealth_override if self._stealth_override is not None else _stealth_env_enabled()
                pw_result = await self._pw_thread.fetch(url, self.user_agent, stealth_flag)
                self.last_fetch_method = "playwright"
                try:
                    self.last_stealth_resolved = bool(pw_result.get("stealth_applied"))
                except Exception:
                    pass

                if not pw_result or not pw_result.get('html'):
                    why = (pw_result or {}).get('reason', 'unknown')
                    log.warning(f"Playwright returned no usable HTML for {url}. Reason={why}. Falling back to HTTPX.")
                    try:
                        r_fallback = await self.httpx_get(url)
                        html_fallback = r_fallback.text
                        if not _looks_like_placeholder(html_fallback, url):
                            self.last_fetch_method = "httpx_fallback"
                            html_cache.set(url, html_fallback, expire=3600)
                            if return_soup:
                                return (BeautifulSoup(html_fallback, "lxml"), {})
                            return html_fallback
                    except Exception as e:
                        log.error(f"HTTPX fallback failed for {url}: {e}", exc_info=True)
                    return None

                html_content = pw_result['html']
                canonical_data = pw_result.get('canonical_data', {})

            # Soft bot-check: bevar Playwright-HTML hvis der er stærke signaler
            if is_bot_detected(html_content, url):
                try:
                    sig = _extract_core_signals_from_html(html_content)
                    strong = (
                        len(html_content or "") >= max(20000, MIN_CONTENT_LEN // 2)
                        and (sig.get("canonical") or (sig.get("h1_count", 0) > 0))
                    )
                except Exception:
                    strong = False

                if strong:
                    log.warning(f"Bot heuristics tripped in get_raw_html for {url}, but core signals present → keeping Playwright HTML.")
                else:
                    log.error(f"❌ Bot detection triggered after max retries for {url}. Aborting.")
                    return None

            if html_content and not is_bot_detected(html_content, url):
                html_cache.set(url, html_content, expire=3600)
            elif html_content:
                log.warning(f"Not caching HTML for {url} due to bot detection.")

            if return_soup and html_content:
                try:
                    return (BeautifulSoup(html_content, "lxml"), canonical_data)
                except Exception as e:
                    log.warning(f"Kunne ikke parse HTML til BeautifulSoup for {url}: {e}")
                    return None

            if self.last_fetch_method == "playwright":
                return {
                    "html": html_content,
                    "canonical_data": canonical_data,
                    "stealth_applied": self.last_stealth_resolved
                }

            return html_content

    async def close(self):
        if self._is_closed:
            return
        self._is_closed = True

        try:
            await self._pw_thread.close()
        except Exception as e:
            log.error(f"Fejl ved lukning af Playwright: {e}", exc_info=True)

        try:
            await asyncio.shield(self._httpx_client.aclose())
        except Exception as e:
            log.error(f"Fejl ved lukning af httpx: {e}", exc_info=True)
