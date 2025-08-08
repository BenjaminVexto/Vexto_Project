# src/vexto/scoring/http_client.py

from __future__ import annotations
import sys
import asyncio
import threading
import time
import logging
import ssl
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, Optional, Set, Tuple, Union, Any
from urllib.parse import urlparse, urljoin
import certifi
import httpx
import random
from bs4 import BeautifulSoup
from diskcache import Cache
from fake_useragent import FakeUserAgentError, UserAgent
from playwright.sync_api import sync_playwright, Browser as PlaywrightBrowser, Error as PlaywrightError, TimeoutError as PlaywrightTimeoutError
from tenacity import AsyncRetrying, RetryError, retry_if_exception, stop_after_attempt, wait_exponential_jitter
import os
import json
import tempfile
import shutil
import re
from datetime import datetime

log = logging.getLogger(__name__)

try:
    from playwright_stealth import stealth_sync
    STEALTH_AVAILABLE = True
except ImportError:
    STEALTH_AVAILABLE = False
    log.warning("playwright-stealth ikke tilgængelig - bruger manual stealth")

CACHE_DIR = Path(".http_diskcache")
CACHE_DIR.mkdir(exist_ok=True)
html_cache = Cache(str(CACHE_DIR / "html"), size_limit=1 * 1024**3)

try:
    ua_generator = UserAgent()
except FakeUserAgentError:
    ua_generator = None

FALLBACK_USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/16.0 Safari/605.1.15",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:128.0) Gecko/20100101 Firefox/128.0",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Edge/126.0.0.0"
]

def get_random_user_agent() -> str:
    return ua_generator.random if ua_generator else random.choice(FALLBACK_USER_AGENTS)

def _get_headers(base_headers: Optional[Dict] = None, user_agent: Optional[str] = None) -> Dict[str, str]:
    headers = {
        "User-Agent": user_agent or get_random_user_agent(),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
        "Accept-Encoding": "gzip, deflate, br",
        "Connection": "keep-alive",
        "Sec-CH-UA": '"Not/A)Brand";v="8", "Chromium";v="126"',
        "Sec-CH-UA-Mobile": "?0",
        "Sec-CH-UA-Platform": '"Windows"'
    }
    if base_headers:
        headers.update(base_headers)
    return headers

def needs_rendering(html: str) -> bool:
    """Enhanced PWA detection including state hints."""
    if not html:
        return True
    lower = html.lower()
    # Immediate PWA indicators
    pwa_hints = [
        "react-root", "ng-version", "next.js", "webpack", "vite", "astro-island", "svelte", "data-rh=", "blazor",
        "vue-app", "data-cmp-src", "__nuxt__", "nuxt-loading", "data-server-rendered", "vue-meta", "magento",
        "vue-storefront", "__initial_state__"
    ]
    if any(hint in lower for hint in pwa_hints):
        log.info("PWA/SPA indicators detected - rendering required.")
        return True
    # High script count = SPA
    if lower.count('<script') > 20:
        log.info("High script density - likely SPA.")
        return True
    # Short HTML = CSR
    if len(html) < 5000:
        log.info("Short HTML - likely client-side rendered.")
        return True
    # State reference check
    if '__initial_state__' in lower or 'window.__' in lower:
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
    if url.lower().endswith(('.xml', '.txt')) or 'sitemap' in url.lower() or 'robots' in url.lower():
        log.debug(f"Skipping placeholder check for static file: {url}")
        return False
    if not html or len(html) < 200:
        log.debug("Placeholder-tjek [Regel 1]: HTML var None eller meget kort.")
        return True
    lowered = html.lower()
    challenge_patterns = [
        "cdn-cgi/challenge-platform", "just a moment...", "attention required!",
        "verifying your browser", "access denied",
    ]
    if any(p in lowered for p in challenge_patterns):
        log.warning("Placeholder-tjek [Regel 2]: Matchede kendt challenge-mønster.")
        return True
    return False

def is_bot_detected(html: str, url: str = "") -> bool:
    if url.lower().endswith('.xml') or 'sitemap' in url.lower():
        log.debug(f"Skipping bot detection for XML or sitemap URL: {url}")
        return False
    lower = html.lower()
    patterns = ["access denied", "unusual traffic", "bot detection", "challenge", "cloudflare", "datadome", "sorry, something went wrong"]
    return any(p in lower for p in patterns)

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
            verify=_ssl_context(True)
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
                    "--enable-features=NetworkService,NetworkServiceInProcess"
                ]
                self._browser = self._playwright_instance.chromium.launch(
                    headless=False,
                    args=launch_args,
                    ignore_default_args=['--enable-automation']
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
        """Enhanced PWA/SPA detection."""
        pwa_indicators = [
            'inventarland.dk', 'magento', 'vue-storefront', 'nuxt', 'react', 'pwa', 'spa', 'ssr', 'csr', 'vue', 'angular', 'svelte'
        ]
        return any(indicator in url.lower() for indicator in pwa_indicators)

    def _handle_cookies_advanced(self, page):
        """Enhanced cookie handling with more selectors."""
        try:
            cookie_selectors = [
                # Danish specific
                "button:has-text('Accepter')", "button:has-text('Godkend')",
                "button:has-text('Tillad')", "button:has-text('OK')",
                # English
                "button:has-text('Accept')", "button:has-text('Allow')",
                "button:has-text('Agree')", "button:has-text('Continue')",
                # ARIA labels
                "button[aria-label*='accept' i]", "button[aria-label*='consent' i]",
                "button[aria-label*='cookie' i]",
                # Common IDs and classes
                "#onetrust-accept-btn-handler", ".cc-btn.cc-accept",
                ".cookie-accept", ".btn-consent", "button.accept-cookies",
                "[id*='cookie'] button", "[class*='cookie'] button",
                "[data-testid*='cookie']", "[data-cy*='cookie']",
                # Generic button patterns
                "button[class*='accept']", "button[id*='accept']"
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
            # Fallback: Hide cookie overlays
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

    def _force_lazy_load_images(self, page):
        """Force all lazy-loaded images to load."""
        try:
            # Method 1: Scroll to trigger intersection observers
            page.evaluate("""
                // Scroll through entire page to trigger lazy loading
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
            
            # Method 2: Force load all images with data-src or data-lazy attributes
            images_loaded = page.evaluate("""
                (() => {
                    let count = 0;
                    const images = document.querySelectorAll('img[data-src], img[data-lazy], img[loading="lazy"], img[data-original]');
                    
                    images.forEach(img => {
                        // Get the lazy source
                        const lazySrc = img.getAttribute('data-src') || 
                                       img.getAttribute('data-lazy') || 
                                       img.getAttribute('data-original');
                        
                        if (lazySrc && !img.src.includes(lazySrc)) {
                            img.src = lazySrc;
                            img.removeAttribute('loading');
                            count++;
                        }
                        
                        // Trigger load event
                        img.dispatchEvent(new Event('load'));
                    });
                    
                    // Also handle background images
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
                page.wait_for_timeout(2000)  # Wait for images to actually load
                
            # Method 3: Trigger all intersection observers manually
            page.evaluate("""
                // Force trigger all IntersectionObservers
                if (window.IntersectionObserver) {
                    const observers = [];
                    const originalObserver = window.IntersectionObserver;
                    
                    // Get all observed elements
                    document.querySelectorAll('*').forEach(el => {
                        if (el._intersectionObserver) {
                            el._intersectionObserver.observe(el);
                        }
                    });
                    
                    // Simulate all elements being visible
                    window.dispatchEvent(new Event('scroll'));
                    window.dispatchEvent(new Event('resize'));
                }
            """)
            
            return images_loaded
            
        except Exception as e:
            log.warning(f"Lazy loading handler failed: {e}")
            return 0

    def _simulate_user_interaction_advanced(self, page):
        """Advanced user simulation with realistic patterns INCLUDING lazy load."""
        try:
            # Random entry point
            entry_x, entry_y = random.randint(100, 500), random.randint(100, 400)
            page.mouse.move(entry_x, entry_y)
            page.wait_for_timeout(random.randint(300, 800))
            # Dummy click for engagement
            page.mouse.click(entry_x + random.randint(-50, 50), entry_y + random.randint(-50, 50))
            page.wait_for_timeout(random.randint(500, 1000))
            # Progressive scrolling with realistic timing
            scroll_positions = [200, 400, 600, 900, 1200]
            for pos in scroll_positions:
                page.mouse.wheel(0, pos)
                page.wait_for_timeout(random.randint(400, 900))
            # Occasional hover simulation
            if random.random() < 0.3:
                hover_x, hover_y = random.randint(200, 800), random.randint(200, 600)
                page.mouse.move(hover_x, hover_y)
                page.wait_for_timeout(random.randint(200, 500))
            # Full scroll to trigger lazy loading
            page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
            page.wait_for_timeout(1500)
            # Return to top with realistic timing
            page.evaluate("window.scrollTo({top: 0, behavior: 'smooth'})")
            page.wait_for_timeout(1000)
            
            # NOW FORCE LAZY LOAD
            loaded_count = self._force_lazy_load_images(page)
            
            # Final verification
            final_check = page.evaluate("""
                () => {
                    const images = document.querySelectorAll('img');
                    let loaded = 0;
                    let total = images.length;
                    
                    images.forEach(img => {
                        if (img.complete && img.naturalHeight !== 0) {
                            loaded++;
                        }
                    });
                    
                    return { loaded, total, percentage: (loaded/total * 100).toFixed(1) };
                }
            """)
            
            log.info(f"📷 Image loading status: {final_check['loaded']}/{final_check['total']} ({final_check['percentage']}%)")
            
        except Exception as e:
            log.warning(f"Advanced user simulation failed: {e}")

    def _force_vue_refresh(self, page):
        """Force Vue/Nuxt state refresh and wait."""
        try:
            page.evaluate("""
                // Multiple refresh strategies
                if (window.$nuxt && window.$nuxt.$router) {
                    const currentRoute = window.$nuxt.$router.currentRoute;
                    window.$nuxt.$router.replace(currentRoute.fullPath);
                }
                // Force re-render via events
                window.dispatchEvent(new Event('resize'));
                window.dispatchEvent(new Event('scroll'));
                // Force Vue reactivity update
                if (window.Vue && window.Vue.nextTick) {
                    window.Vue.nextTick(() => {
                        console.log('Vue nextTick executed');
                    });
                }
                // Trigger intersection observers
                window.scrollTo(0, 1);
                window.scrollTo(0, 0);
            """)
            # Wait for refresh to take effect
            page.wait_for_timeout(3000)
            # Check if refresh worked
            return page.wait_for_function("""
                () => window.__INITIAL_STATE__ && Object.keys(window.__INITIAL_STATE__).length > 0
            """, timeout=5000)
        except Exception as e:
            log.warning(f"Vue refresh failed: {e}")
            raise e

    def _handle_pwa_hydration_hybrid(self, page, url: str, hydration_requests: list) -> Optional[str]:
        """Kombineret PWA hydration med alle tricks PLUS lazy loading."""
        log.info("🔄 Starting hybrid PWA hydration...")
        # Step 1: Wait for framework
        try:
            page.wait_for_function("""
                () => window.Vue || window.$nuxt || window.__NUXT__ || document.querySelector('[data-server-rendered]') || document.querySelector('[id*="app"]') || document.querySelector('[class*="vue"]')
            """, timeout=15000)
            log.info("✅ Vue/Nuxt framework detected")
        except Exception:
            log.warning("⚠️ Framework not detected")
        
        # Step 2: Enhanced user simulation (includes lazy loading now)
        self._simulate_user_interaction_advanced(page)
        
        # Step 3: Wait for API requests
        if hydration_requests:
            log.info(f"🔍 Waiting for {len(hydration_requests)} API requests to complete...")
            try:
                page.wait_for_function("() => document.readyState === 'complete'", timeout=10000)
            except Exception:
                pass
        
        # Step 4: Multi-strategy __INITIAL_STATE__ detection
        strategies = [
            # Direct object check (runtime)
            lambda: page.wait_for_function("""
                () => window.__INITIAL_STATE__ && typeof window.__INITIAL_STATE__ === 'object' && Object.keys(window.__INITIAL_STATE__).length > 0
            """, timeout=8000),
            # Script tag detection
            lambda: page.wait_for_selector("script:has-text('__INITIAL_STATE__')", timeout=8000),
            # Canonical-specific check
            lambda: page.wait_for_function("""
                () => window.__INITIAL_STATE__ && JSON.stringify(window.__INITIAL_STATE__).includes('canonical')
            """, timeout=8000),
            # Force refresh strategy
            lambda: self._force_vue_refresh(page)
        ]
        
        # Løkken afbrydes, så snart en strategi finder 'canonical'
        for i, strategy in enumerate(strategies, 1):
            try:
                log.info(f"🎯 Hydration strategy {i}/4...")
                strategy()
                log.info(f"✅ Strategy {i} succeeded!")
                # Check actual state content
                try:
                    state_check = page.evaluate("""
                        () => {
                            if (window.__INITIAL_STATE__) {
                                const state = window.__INITIAL_STATE__;
                                const hasCanonical = JSON.stringify(state).includes('canonical');
                                return {
                                    hasState: true,
                                    size: JSON.stringify(state).length,
                                    hasCanonical: hasCanonical,
                                    keys: Object.keys(state).slice(0, 5)
                                };
                            }
                            return { hasState: false };
                        }
                    """)
                    if state_check.get('hasState'):
                        log.info(f"🎉 State verified: {state_check['size']} chars, canonical: {state_check.get('hasCanonical', False)}")
                        if state_check.get('hasCanonical'):
                            return page.content() # RETUR HER, HVIS KANONISK ER FUNDET
                except Exception as e:
                    log.warning(f"State verification failed: {e}")
                
            except Exception as e:
                log.warning(f"❌ Strategy {i} failed: {e}")
                continue

        # Ultimate iframe fallback (køres kun hvis ingen af de andre virkede)
        log.info("🏗️ Trying iframe isolation fallback...")
        try:
            page.evaluate(f'''
                const iframe = document.createElement('iframe');
                iframe.src = '{url}';
                iframe.style.display = 'none';
                document.body.appendChild(iframe);
                iframe.onload = () => {{
                    if (iframe.contentWindow && iframe.contentWindow.__INITIAL_STATE__) {{
                        window._IFRAME_STATE = iframe.contentWindow.__INITIAL_STATE__;
                        console.log('Iframe state captured');
                    }}
                }};
            ''')
            page.wait_for_function('() => window._IFRAME_STATE !== undefined', timeout=15000)
            iframe_state = page.evaluate('() => window._IFRAME_STATE')
            if iframe_state:
                log.info("🎉 Iframe fallback succeeded!")
                # Inject state into main window
                page.evaluate('(state) => window.__INITIAL_STATE__ = state', iframe_state)
        except Exception as e:
            log.warning(f"Iframe fallback failed: {e}")
            
        # Final content retrieval
        try:
            page.wait_for_selector("footer", timeout=8000)
            log.info("✅ Footer detected - page fully loaded")
        except Exception:
            log.warning("⚠️ Footer not found")
        return page.content()

    def _extract_canonical_hybrid(self, page, html_content: str, url: str) -> dict:
        """Extract canonical data using both runtime and HTML parsing."""
        canonical_data = {}
        
        # Method 1: Runtime extraction (preferred)
        try:
            runtime_state = page.evaluate("() => window.__INITIAL_STATE__")
            if runtime_state:
                canonical_data['runtime_state'] = runtime_state
                canonical_data['runtime_success'] = True
                # Recursive search for canonical fields
                def find_canonical_keys(obj, path=""):
                    results = {}
                    if isinstance(obj, dict):
                        for key, value in obj.items():
                            new_path = f"{path}.{key}" if path else key
                            if 'canonical' in key.lower():
                                results[new_path] = value
                                log.info(f"🎯 Runtime canonical found: {new_path} = {value}")
                            results.update(find_canonical_keys(value, new_path))
                    elif isinstance(obj, list):
                        for i, item in enumerate(obj):
                            results.update(find_canonical_keys(item, f"{path}[{i}]"))
                    return results
                canonical_fields = find_canonical_keys(runtime_state)
                canonical_data['canonical_fields'] = canonical_fields
            else:
                canonical_data['runtime_success'] = False
        except Exception as e:
            log.warning(f"Runtime extraction failed: {e}")
            canonical_data['runtime_success'] = False
        
        # Method 2: HTML regex parsing (backup)
        try:
            state_pattern = r'window\.__INITIAL_STATE__\s*=\s*({.+?});'
            matches = re.findall(state_pattern, html_content, re.DOTALL)
            if matches:
                canonical_data['html_matches'] = len(matches)
                for i, match in enumerate(matches):
                    try:
                        parsed_state = json.loads(match)
                        if 'canonical' in str(parsed_state).lower():
                            canonical_data[f'html_state_{i}'] = parsed_state
                            log.info(f"📄 HTML canonical found in match {i}")
                    except json.JSONDecodeError:
                        continue
            else:
                canonical_data['html_matches'] = 0
        except Exception as e:
            log.warning(f"HTML extraction failed: {e}")
            canonical_data['html_success'] = False
        
        # Method 3: Traditional <link rel="canonical"> check
        try:
            canonical_link = page.get_attribute('link[rel="canonical"]', 'href', timeout=2000)
            if canonical_link:
                canonical_data['link_canonical'] = canonical_link
                log.info(f"🔗 Traditional canonical found: {canonical_link}")
        except Exception:
            pass
        
        return canonical_data
    
    def _sync_fetch(self, url: str, ua: str, retry_count: int = 0) -> Optional[Dict[str, Any]]:
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
            storage_state = {}
            if os.path.exists(state_path):
                try:
                    with open(state_path, 'r') as f:
                        storage_state = json.load(f)
                except json.JSONDecodeError as e:
                    log.warning(f"Invalid storage state file for {url}: {e}")

            # PWA detection
            is_pwa_site = self._detect_pwa_site(url)

            # Pre-flight HTTPX check for baseline
            try:
                headers = _get_headers({"User-Agent": ua})
                response = self._http_client.get(url, headers=headers)
                html = response.text
                java_script_enabled = needs_rendering(html) or is_pwa_site
            except Exception:
                java_script_enabled = True

            # Enhanced viewport for PWA sites
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
                geolocation={'latitude': 55.6761, 'longitude': 12.5683},  # Copenhagen
                timezone_id='Europe/Copenhagen'
            ) as context:
                page = context.new_page()
                
                # Stealth package integration
                if STEALTH_AVAILABLE:
                    try:
                        stealth_sync(page)
                        log.info("✅ playwright-stealth applied")
                    except Exception as e:
                        log.warning(f"Stealth package failed: {e}")
                
                # Manual stealth as backup
                page.add_init_script("""
                    // Advanced stealth for PWA sites
                    Object.defineProperty(navigator, 'webdriver', { get: () => undefined });
                    // Override automation detection
                    delete window.cdc_adoQpoasnfa76pfcZLmcfl_Array;
                    delete window.cdc_adoQpoasnfa76pfcZLmcfl_Promise;
                    delete window.cdc_adoQpoasnfa76pfcZLmcfl_Symbol;
                    // Chrome object mocking
                    window.chrome = { runtime: {}, loadTimes: function() {}, csi: function() {}, app: {} };
                    // Permissions override
                    const originalQuery = window.navigator.permissions.query;
                    if (originalQuery) {
                        window.navigator.permissions.query = (parameters) => (
                            parameters.name === 'notifications' ?
                            Promise.resolve({ state: 'default' }) :
                            originalQuery(parameters)
                        );
                    }
                    // Mock plugins + languages for Danish context
                    Object.defineProperty(navigator, 'plugins', { get: () => [1, 2, 3, 4, 5], });
                    Object.defineProperty(navigator, 'languages', { get: () => ['da-DK', 'da', 'en-US', 'en'], });
                    // Window dimensions consistency
                    window.outerHeight = screen.height;
                    window.outerWidth = screen.width;
                """)

                # Request interception for API detection
                hydration_requests = []
                if is_pwa_site:
                    def handle_request(request):
                        request_url = request.url.lower()
                        if any(keyword in request_url for keyword in ['graphql', '/api/', 'ajax', 'json']):
                            hydration_requests.append(request.url)
                            log.debug(f"🔍 Detected API call: {request.url}")
                    page.on('request', handle_request)

                log.info(f"🚀 Playwright fetching {url} (PWA: {is_pwa_site}, retry: {retry_count})...")
                
                try:
                    # Navigate with longer timeout for PWA
                    timeout = 90000 if is_pwa_site else 60000
                    page.goto(url, wait_until="domcontentloaded", timeout=timeout)
                    # Basic network settle
                    try:
                        page.wait_for_load_state("networkidle", timeout=15000)
                    except Exception:
                        log.debug("Network idle timeout - continuing")
                except Exception as e:
                    log.warning(f"Navigation failed: {e}")
                    return {"html": None, "canonical_data": {}}

                # Handle cookies first
                self._handle_cookies_advanced(page)

                # PWA-specific hydration or standard rendering
                if is_pwa_site:
                    html_content = self._handle_pwa_hydration_hybrid(page, url, hydration_requests)
                else:
                    if java_script_enabled:
                        self._simulate_user_interaction_advanced(page)
                    try:
                        page.wait_for_selector("footer", timeout=10000)
                        log.info("✅ Footer loaded")
                    except Exception:
                        log.warning("⚠️ Footer not found")
                    html_content = page.content()

                if not html_content:
                    return {"html": None, "canonical_data": {}}

                # Dual extraction (runtime + regex)
                canonical_data = self._extract_canonical_hybrid(page, html_content, url)

                # Debug save with timestamp
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                domain = urlparse(url).netloc.replace('.', '_')
                path = urlparse(url).path.replace('/', '_') or 'index'
                html_filename = f"debug_{domain}_{path}_{timestamp}.html"
                with open(html_filename, "w", encoding="utf-8") as f:
                    f.write(html_content)
                log.info(f"💾 Saved debug HTML: {html_filename}")
                
                # Final bot detection check
                if is_bot_detected(html_content, url):
                    log.error(f"❌ Bot detection in final HTML for {url}")
                    return {"html": None, "canonical_data": {}}

                # Save storage state for next visit
                try:
                    storage_state = context.storage_state()
                    with open(state_path, 'w') as f:
                        json.dump(storage_state, f)
                except Exception as e:
                    log.warning(f"Storage state save failed: {e}")

                return {"html": html_content, "canonical_data": canonical_data}
                
        except Exception as e:
            log.error(f"Fatal error for {url}: {e}", exc_info=True)
            return {"html": None, "canonical_data": {}}
        finally:
            with self._fetch_lock:
                self._active_fetches -= 1

    async def fetch(self, url: str, ua: str) -> Optional[Dict[str, Any]]:
        if not self._is_ready:
            return {"html": None, "canonical_data": {}}
        return await asyncio.get_running_loop().run_in_executor(self._executor, self._sync_fetch, url, ua)

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

        if self._executor._shutdown:
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

class AsyncHtmlClient:
    def __init__(self, *, max_connections: int = 10, total_timeout: float = 45.0, verify_ssl: bool = True, ca_bundle: Optional[Path] = None, proxy: Optional[str] = None):
        self._sem = asyncio.Semaphore(max_connections)
        proxy = proxy or os.getenv("HTTP_PROXY")
        self._httpx_client = httpx.AsyncClient(
            headers={"Accept-Language": "en-US,en;q=0.9,da;q=0.8"},
            timeout=httpx.Timeout(total_timeout, connect=10.0),
            follow_redirects=True,
            http2=True,
            verify=_ssl_context(verify_ssl, ca_bundle),
            proxy=proxy
        )
        self._pw_thread = _PlaywrightThreadFetcher()
        self._url_locks: Dict[str, asyncio.Lock] = {}
        self.last_fetch_method: Optional[str] = None
        self._is_closed: bool = False
        self.user_agent = get_random_user_agent()

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

    async def get(self, url: str, **kwargs) -> httpx.Response:
        """Compatibility method to match expected interface"""
        return await self.httpx_get(url, **kwargs)

    async def get_response(self, url: str, **kwargs) -> httpx.Response:
        """Get response for compatibility with technical SEO"""
        return await self.httpx_get(url, **kwargs)

    async def httpx_get(self, url: str, **kwargs) -> httpx.Response:
        async with self._sem:
            headers = _get_headers(kwargs.pop("headers", None), user_agent=self.user_agent)
            return await self._httpx_client.get(url, headers=headers, **kwargs)

    async def head(self, url: str, **kwargs) -> Optional[httpx.Response]:
        if await self._check_if_closed(url):
            return None
        headers = _get_headers(kwargs.pop("headers", None), user_agent=self.user_agent)
        async with self._sem:
            try:
                response = await self._httpx_client.request("HEAD", url, headers=headers, **kwargs)
                response.elapsed_time_ms = int(response.elapsed.total_seconds() * 1000)
                return response
            except httpx.RequestError as e:
                log.warning(f"HEAD request til {url} fejlede: {e}")
                return None

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
            try:
                from datetime import datetime
                import xml.etree.ElementTree as ET
                root = ET.fromstring(html)
                log.debug(f"Sitemap XML: {html[:200]}")
                if root.tag.endswith('sitemapindex'):
                    for sitemap in root.findall(".//{http://www.sitemaps.org/schemas/sitemap/0.9}sitemap"):
                        loc = sitemap.find("{http://www.sitemaps.org/schemas/sitemap/0.9}loc")
                        if loc is not None and loc.text:
                            log.debug(f"Checking subsitemap: {loc.text}")
                            if await self.check_sitemap_freshness(loc.text):
                                return True
                    log.info(f"Sitemap index {url} has no fresh subsitemaps (status: missing)")
                    return False
                for url_elem in root.findall(".//{http://www.sitemaps.org/schemas/sitemap/0.9}url"):
                    lastmod = url_elem.find("{http://www.sitemaps.org/schemas/sitemap/0.9}lastmod")
                    if lastmod is not None and lastmod.text:
                        try:
                            for fmt in ["%Y-%m-%d", "%Y-%m-%dT%H:%M:%S%z", "%Y-%m-%dT%H:%M:%S.%f%z"]:
                                try:
                                    lastmod_date = datetime.strptime(lastmod.text, fmt)
                                    if (datetime.now() - lastmod_date.replace(tzinfo=None)).days < 30:
                                        log.info(f"Sitemap {url} is fresh (lastmod: {lastmod.text}) (status: ok)")
                                        return True
                                    break
                                except ValueError:
                                    continue
                            log.warning(f"Invalid lastmod format in sitemap {url}: {lastmod.text} (status: error)")
                        except Exception as e:
                            log.warning(f"Could not parse lastmod {lastmod.text} in sitemap {url}: {e} (status: error)")
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
        # Handle different return formats
        html = None
        if isinstance(html_result, str):
            html = html_result
        elif isinstance(html_result, tuple) and len(html_result) >= 1:
            html = html_result[0]
        elif isinstance(html_result, dict) and html_result.get('html'):
            html = html_result['html']
        
        if not html:
            log.info(f"No HTML for analytics check on {url} (status: missing)")
            return {"has_ga4": False, "has_meta_pixel": False}
        
        has_ga4 = any(x in html for x in ["gtag.js", "googletagmanager.com", "gtm.js", "ga.js"])
        has_meta_pixel = any(x in html for x in ["fbq(", "connect.facebook.net", "pixel/facebook.com"])
        log.info(f"Analytics check for {url}: GA4={has_ga4}, Meta Pixel={has_meta_pixel} (status: ok)")
        return {"has_ga4": has_ga4, "has_meta_pixel": has_meta_pixel}

    async def test_response_time(self, url: str) -> float:
        start = time.time()
        response = await self.httpx_get(url)
        elapsed = (time.time() - start) * 1000
        log.info(f"Raw response time for {url}: {elapsed:.2f} ms")
        return elapsed

    async def get_raw_html(self, url: str, force: bool = False, return_soup: bool = False, force_playwright: bool = False, depth: int = 0) -> Union[str, Tuple[str, Dict], Dict[str, Any], None]:
        """
        Get raw HTML with proper return format handling.
        
        Returns:
            - If return_soup=True: (BeautifulSoup, canonical_data) or None
            - If force_playwright=True or escalated to Playwright: {"html": str, "canonical_data": dict}
            - Otherwise: str (HTML content) or None
        """
        if self._is_closed:
            log.warning(f"Forsøg på at kalde get_raw_html på lukket klient: {url}")
            return None
        if depth > 1:
            log.warning(f"Max depth reached for {url}; stopping.")
            return None

        # Special handling for always-HTTPX domains
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

        # Check cache first
        if not force and not force_playwright and (html := html_cache.get(url)) is not None:
            log.debug(f"Cache hit for {url}")
            if return_soup:
                return (BeautifulSoup(html, "lxml"), {})
            return html

        lock = self._url_locks.setdefault(url, asyncio.Lock())
        async with lock:
            # Double-check cache after acquiring lock
            if not force and not force_playwright and (html := html_cache.get(url)) is not None:
                if return_soup:
                    return (BeautifulSoup(html, "lxml"), {})
                return html

            html_content: Optional[str] = None
            canonical_data: Dict = {}
            
            try:
                if force_playwright:
                    raise httpx.RequestError("Playwright blev tvunget manuelt for at få renderet indhold.")

                response = await self.httpx_get(url)
                html_content = response.text
                self.last_fetch_method = "httpx"
                
                if _looks_like_placeholder(html_content, url):
                    raise httpx.RequestError("Placeholder content; escalate to Playwright")
                if needs_rendering(html_content) and not (url.lower().endswith(('.xml', '.txt')) or 'sitemap' in url.lower() or 'robots' in url.lower()):
                    raise httpx.RequestError("Dynamic content; escalate to Playwright")
                    
            except (httpx.RequestError, RetryError) as e:
                log.info(f"Escalating to Playwright for {url}: {e}")
                pw_result = await self._pw_thread.fetch(url, self.user_agent)
                self.last_fetch_method = "playwright"
                
                if not pw_result or not pw_result.get('html'):
                    log.error(f"Playwright failed to fetch HTML for {url}. Aborting.")
                    return None
                    
                html_content = pw_result['html']
                canonical_data = pw_result.get('canonical_data', {})
                
                if is_bot_detected(html_content, url):
                    log.error(f"❌ Bot detection triggered after max retries for {url}. Aborting.")
                    return None

            # Cache valid HTML
            if html_content and not is_bot_detected(html_content, url):
                html_cache.set(url, html_content, expire=3600)
            elif html_content:
                log.warning(f"Not caching HTML for {url} due to bot detection.")

            # Return appropriate format
            if return_soup and html_content:
                try:
                    return (BeautifulSoup(html_content, "lxml"), canonical_data)
                except Exception as e:
                    log.warning(f"Kunne ikke parse HTML til BeautifulSoup for {url}: {e}")
                    return None
            
            # If we used Playwright, return the dict format with canonical data
            if self.last_fetch_method == "playwright":
                return {"html": html_content, "canonical_data": canonical_data}
            
            # Otherwise return just the HTML string
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

__all__ = ["AsyncHtmlClient", "get_random_user_agent", "html_cache", "needs_rendering"]