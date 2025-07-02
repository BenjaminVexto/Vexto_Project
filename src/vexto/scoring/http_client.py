# src/vexto/scoring/http_client.py
from __future__ import annotations
import asyncio, ssl, logging
from pathlib import Path
from typing import Optional, Dict, Set
from urllib.parse import urlparse
from playwright.async_api import (
    async_playwright,
    Browser as PlaywrightBrowser,
    Error as PlaywrightError,
)

import certifi, httpx
from bs4 import BeautifulSoup
from diskcache import Cache
from tenacity import (
    AsyncRetrying,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential_jitter,
    RetryError,
)
from fake_useragent import UserAgent, FakeUserAgentError

log = logging.getLogger(__name__)

# --- Cache --------------------------------------------------------------
CACHE_DIR  = Path(".http_diskcache"); CACHE_DIR.mkdir(exist_ok=True)
html_cache = Cache(str(CACHE_DIR / "html"), size_limit=1 * 1024 ** 3)

# --- UA --------------------------------------------------------------
try:    ua_generator = UserAgent()
except FakeUserAgentError: ua_generator = None
FALLBACK_USER_AGENT = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                       "AppleWebKit/537.36 (KHTML, like Gecko) "
                       "Chrome/126.0.0.0 Safari/537.36")

def get_random_user_agent() -> str:
    return ua_generator.random if ua_generator else FALLBACK_USER_AGENT

# --- Retry -----------------------------------------------------------
RETRY_POLICY = AsyncRetrying(
    retry=retry_if_exception(
        lambda e: isinstance(e, (httpx.RequestError, asyncio.TimeoutError))
        or (isinstance(e, httpx.HTTPStatusError) and e.response.status_code >= 500)
    ),
    wait=wait_exponential_jitter(initial=0.5, max=8),
    stop=stop_after_attempt(3),
    reraise=True,
)

def _ssl_context(verify: bool, ca_bundle: Optional[Path] = None) -> ssl.SSLContext | bool:
    if not verify:
        return False
    try:
        ctx = ssl.create_default_context(cafile=certifi.where())
        if ca_bundle:
            ctx.load_verify_locations(cafile=str(ca_bundle))
        return ctx
    except Exception:
        log.warning("Kunne ikke loade 'certifi' bundle – bruger systemets trust store.")
        return True

# --- Domæner der **aldrig** skal via Playwright ----------------------
ALWAYS_HTTPX = {"www.googleapis.com", "pagespeedonline.googleapis.com"}

# --- Heuristik for at opdage placeholder-HTML ----------------------
MIN_BODY_BYTES = 8_000 # Empirisk værdi, juster ved behov
def _looks_like_placeholder(html: Optional[str]) -> bool:
    if html is None:
        return False

    low = html.lstrip().lower()
    if not low.startswith("<!doctype") and "<html" not in low:
        return False      # robots.txt, XML, JSON … -> ingen placeholder-check

    if '<form' in low and 'action="/search"' in low:
        return False # It's Google.com, not a placeholder

    return (
        len(html) < MIN_BODY_BYTES
        or "<h1" not in low
        or "<title>" not in low
        or "choose your region" in low
        or "cookie consent"   in low
        or "age verification" in low
    )

# --------------------------------------------------------------------
class AsyncHtmlClient:
    """Hybrid-klient: httpx først, Playwright kun til ‘rigtige’ websites."""
    def __init__(self, *, max_connections: int = 10, total_timeout: float = 45.0,
                 verify_ssl: bool = True, proxy: Optional[str] = None, ca_bundle: Optional[Path] = None):
        self._sem = asyncio.Semaphore(max_connections)
        self._httpx_client = httpx.AsyncClient(
            headers={"Accept-Language": "en-US,en;q=0.9,da;q=0.8"},
            timeout=httpx.Timeout(total_timeout, connect=10.0),
            follow_redirects=True, trust_env=True, http2=False,
            verify=_ssl_context(verify_ssl, ca_bundle), proxy=proxy
        )
        self._playwright_browser: PlaywrightBrowser | None = None
        self._playwright_lock  = asyncio.Lock()
        self._playwright_only_domains: Set[str] = set()
        self._url_locks: Dict[str, asyncio.Lock] = {}

    # --- NYT: Gør klassen til en async context manager ---
    async def __aenter__(self):
        """ Gør det muligt at bruge 'async with' ved at returnere sig selv. """
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """ Garanterer at .close() bliver kaldt, når 'async with' forlades. """
        await self.close()

    # --- compat & helpers -------------------------------------------
    @property
    def client(self) -> httpx.AsyncClient:      # eksisterende tests bruger .client
        return self._httpx_client

    async def httpx_get(self, url: str, **kw) -> httpx.Response:
        async with self._sem:
            return await self._httpx_client.get(url, **kw)

    # --- Playwright --------------------------------------------------
    async def _ensure_playwright(self):
        async with self._playwright_lock:
            if not (self._playwright_browser and self._playwright_browser.is_connected()):
                log.info("Playwright-browser starter …")
                try:
                    p = await async_playwright().start()
                    self._playwright_browser = await p.chromium.launch()
                    log.info("Playwright-browser startet succesfuldt.")
                except Exception as e:
                    log.error(
                        "FATAL: Kunne ikke starte Playwright-browser: %s "
                        "(kør 'playwright install')", e, exc_info=True
                    )
                    self._playwright_browser = None
                    raise

    async def _pw_get(self, url: str) -> Optional[str]:
        await self._ensure_playwright()
        if not self._playwright_browser:
            log.error("Playwright browser is not available, cannot fetch %s", url)
            return None
        try:
            page = await self._playwright_browser.new_page(user_agent=get_random_user_agent())
            log.info("Henter %s med Playwright...", url)
            await page.goto(
                url,
                timeout=60_000,
                wait_until="networkidle"
            )
            content = await page.content()
            await page.close()
            log.info("Succes med Playwright for %s.", url)
            return content
        except PlaywrightError as e:
            log.error(f"Playwright fejlede for {url}: {e.message.splitlines()[0]}")
            return None
        except Exception as e:
            log.error(f"Uventet fejl under Playwright fetch for {url}: {e}", exc_info=True)
            return None

    # --- Core fetch --------------------------------------------------
    async def _httpx_fetch(self, url: str) -> str:
        async with self._sem:
            async for attempt in RETRY_POLICY:
                with attempt:
                    resp = await self._httpx_client.get(url, headers={"User-Agent": get_random_user_agent()})
                    resp.raise_for_status()
                    return resp.text

    async def get_raw_html(self, url: str) -> Optional[str]:
        domain = urlparse(url).netloc

        if domain in ALWAYS_HTTPX:
            try:
                return (await self.httpx_get(url)).text
            except Exception as e:
                log.warning("API-request failed %s: %s", url, e)
                return None

        if domain in self._playwright_only_domains:
            html = await self._pw_get(url)
            if html:
                html_cache.set(url, html, expire=3600)
            return html

        if (html := html_cache.get(url)) is not None:
            return html

        lock = self._url_locks.setdefault(url, asyncio.Lock())
        async with lock:
            if (html := html_cache.get(url)) is not None:
                return html
            try:
                html = await self._httpx_fetch(url)
                if _looks_like_placeholder(html):
                    log.info("HTTPX for %s leverede placeholder-lignende HTML. Eskalerer til Playwright.", url)
                    raise httpx.RequestError("Placeholder HTML detected, forcing Playwright fallback.")
                
                html_cache.set(url, html, expire=3600)
                return html
            except (httpx.RequestError, RetryError) as e:
                log.warning("httpx fejlede for %s (%s) – eskalerer til Playwright",
                            url, e.__class__.__name__)
                self._playwright_only_domains.add(domain)

                html = await self._pw_get(url)
                if html:
                    html_cache.set(url, html, expire=3600)
                return html

                log.warning("Playwright returnerede None for %s – laver ét ekstra forsøg", url)
                html = await self._pw_get(url)
                if html:
                    html_cache.set(url, html, expire=3600)
                return html

    # ----------------------------------------------------------------
    async def get_soup(self, url: str):
        if (html := await self.get_raw_html(url)):
            try:
                return BeautifulSoup(html, "lxml")
            except Exception as e:
                log.warning("BS4-parse-fejl %s: %s", url, e)
        return None

    async def head(self, url: str) -> Optional[httpx.Response]:
        try:
            async for attempt in RETRY_POLICY:
                with attempt:
                    r = await self._httpx_client.head(url, headers={"User-Agent": get_random_user_agent()})
                    return r
        except httpx.RequestError as e:
            log.warning(f"HEAD request network error for {url}: {e.__class__.__name__}")
            return None
        except Exception as e:
            log.error(f"Unexpected error in HEAD request for {url}: {e}", exc_info=True)
            return None

    async def resolve_final_url(self, url: str) -> str:
        if response := await self.head(url): return str(response.url)
        return url

    async def close(self):
        await self._httpx_client.aclose()
        if self._playwright_browser and self._playwright_browser.is_connected():
            await self._playwright_browser.close()
            log.info("Playwright lukket.")

# Eksporter cache så performance_fetcher kan importere uden cirkel-problem
__all__ = ["AsyncHtmlClient", "html_cache"]