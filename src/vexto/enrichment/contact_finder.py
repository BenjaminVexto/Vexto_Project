# src/vexto/enrichment/contact_finder.py
# =============================================================================
# Vexto Contact Finder – præcis og hurtig:
# - Kandidater fra JSON-LD, RDFa, microformats (h-card), mailto og DOM
# - Robust parsing: selectolax → BeautifulSoup → simpel fallback
# - Telefon-normalisering (DK + intl), e-mail↔navn-score
# - DOM-afstand og rolle-styrke i scoringen
# - Identity-gate: acceptér kun (navn + email-match) ELLER (navn + reel rolle)
# - Dedup, vægtet scoring, top-1
# - Disk-cache + LRU for hastighed
#
# Offentligt API (stabilt):
#   ContactFinder.find(url: str) -> dict | None          # top-1
#   ContactFinder.find_all(url: str, limit_pages=4) -> list[dict]
#   find_best_contact(url: str) -> dict | None
#
# Kompatibilitets-aliasser (for gamle kald i projektet):
#   find_contacts(url)            == ContactFinder().find_all(url)
#   extract_contacts(url)         == ContactFinder().find_all(url)
#   extract_best_contact(url)     == ContactFinder().find(url)
#   find_top_contact(url)         == ContactFinder().find(url)
# =============================================================================

from __future__ import annotations
import asyncio
import concurrent.futures
import contextlib
import hashlib
import json
import logging
import os
import re
import sys
import time
import threading
import logging.handlers
import random  # ANKER: RATE_LIMIT_BACKOFF
import httpx
import importlib
import dns.resolver
from collections import defaultdict  # ANKER: RATE_LIMIT_BACKOFF
from dataclasses import dataclass
from datetime import datetime, timedelta
from functools import lru_cache
from pathlib import Path
from urllib.parse import urljoin, urlsplit
from typing import Any, Iterable, Optional
from vexto.scoring.http_client import _accept_encoding
from bs4 import BeautifulSoup as _BS4
from urllib.parse import urljoin as _uj, urlsplit as _us
from typing import TYPE_CHECKING

urllib3 = None
if importlib.util.find_spec("urllib3") is not None:
    urllib3 = importlib.import_module("urllib3")

# --- http_client-ankre (robuste med fallback-navne) ---
try:
    from vexto.scoring.http_client import should_check_link_status as _hc_should_check
except Exception:
    try:
        from vexto.scoring.http_client import _http_should_check_link_status as _hc_should_check
    except Exception:
        _hc_should_check = None

try:
    from vexto.scoring.http_client import head_and_resolve as _hc_head_and_resolve
except Exception:
    try:
        from vexto.scoring.http_client import _head_and_resolve as _hc_head_and_resolve
    except Exception:
        _hc_head_and_resolve = None

# Setup log directory
log_dir = Path(__file__).resolve().parents[2] / "logs"
log_dir.mkdir(parents=True, exist_ok=True)

log_file = log_dir / "contact_finder.log"

# RotatingFileHandler: 5 MB pr. fil, behold 5 filer
handler = logging.handlers.RotatingFileHandler(
    log_file, maxBytes=5_000_000, backupCount=5, encoding="utf-8"
)
handler.set_name("cf_rotating")
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
handler.setFormatter(formatter)

log = logging.getLogger("vexto.contact_finder")
log.setLevel(logging.DEBUG)
if not any(getattr(h, "name", "") == "cf_rotating" for h in log.handlers):
    log.addHandler(handler)

# valgfrit: behold konsol kun på WARNING+
console = logging.StreamHandler()
console.set_name("cf_console")
console.setLevel(logging.WARNING)
console.setFormatter(formatter)
if not any(getattr(h, "name", "") == "cf_console" for h in log.handlers):
    log.addHandler(console)

# Ingen top-level runtime-import; vi importerer ved brug (runtime) i de relevante funktioner.
# Dette fjerner Pylance-støj og undgår problemer når filen køres direkte.
if TYPE_CHECKING:
    # Kun for type hints i editoren, påvirker ikke runtime.
    from vexto.scoring.http_client import AsyncHtmlClient as _AsyncHtmlClient  # pragma: no cover

AsyncHtmlClient = None  # type: ignore

_SHARED_PW_CLIENT = None

_HOST_CRAWL_DELAYS = {}

TELEMETRY_CF = os.environ.get("VEXTO_CF_TELEMETRY") == "1"

def _fp_from_text(s: str, prefix: str = "", n: int = 12) -> str:
    """Stabilt kort-fingerprint fra lille tekst-blob (sha1, 12 hexdigits)."""
    try:
        h = hashlib.sha1((prefix + (s or "")[:512]).encode("utf-8", "ignore")).hexdigest()
        return h[:n]
    except Exception:
        return ""



# -------------------- Per-host caches + sticky host helpers ----------------
_PW_FAILS_BY_HOST: dict[str, int] = {}
_PW_FAILS_MAX = 3  # efter 3 fejl i træk for en host stopper vi midlertidigt PW for DEN host
_PW_DISABLED_HOSTS: set[str] = set()

# In-memory (billigt) pr. run; suppler gerne med diskcache hvis ønsket
_ROBOTS_CACHE: dict[str, tuple[int, str]] = {}
_GET_CACHE: dict[str, tuple[int, str]] = {}  # URL→(status, html), kortlevet

# --- ANKER: RATE_LIMIT_BACKOFF ---
# Per-host backoff hvis vi møder 429/503 samt budget på antal requests
_RATE_LIMIT_HOSTS: dict[str, float] = {}      # host -> earliest_epoch_ok
_RL_429_COUNTS: defaultdict[str, int] = defaultdict(int)  # host -> count pr. run
_HOST_REQUESTS: defaultdict[str, int] = defaultdict(int)  # host -> requests pr. run
_HOST_REQUEST_BUDGET = 25                      # hårdt cap pr. host pr. run

# [PATCH] Per-host fejloptælling (DNS/SSL) → tidlig skip, fx ecovis.dk …
_HOST_DNS_FAILS: defaultdict[str, int] = defaultdict(int)
_HOST_SSL_FAILS: defaultdict[str, int] = defaultdict(int)
_HOST_FAIL_LIMIT = 5  # efter 5 DNS/SSL-fejl for en host skipper vi resten af run'et for den host


def _now() -> float:
    return time.time()

def _respect_host_budget(u: str) -> bool:
    if TELEMETRY_CF:
            log.debug(f"Loaded from: {__file__}")    
    host = _host_of(u)
    if not host:
        return True
    if _HOST_REQUESTS[host] >= _HOST_REQUEST_BUDGET:
        log.info(f"Budget reached for host={host} ({_HOST_REQUESTS[host]}) – skipping {u}")
        return False
    return True

class CachingHttpClient:
    def __init__(self, timeout: float = 10.0, retries: int = 3, cache_size: int = 500):
        self.timeout = timeout
        self.retries = retries
        self._cache = {}  # URL -> (status, text)
        self.hits = 0
        self.misses = 0

    @lru_cache(maxsize=500)  # Cache pr. URL
    def get(self, u: str) -> tuple[int, str]:
        """Delegér GET/HTML til fælles http_client.AsyncHtmlClient (SOT).
        Bevarer lokal cache og input-guarding. Ingen lokal retry/backoff/budget.
        """

        # --- Input-validering og normalisering (BEHOLDT) ---
        if not u or not isinstance(u, str):
            return 0, ""
        u = u.strip()
        if u.lower() in ("nan", "none", "null", ""):
            return 0, ""
        if not u.startswith(("http://", "https://")):
            if re.match(r"^(www\.)?([a-z0-9\-]+\.)+[a-z]{2,}", u, re.I):
                u = f"https://{u}"
            else:
                log.debug(f"Invalid URL format: {u}")
                return 0, ""

        u = _norm_url_for_fetch(u)

        # --- Lokal cache (BEHOLDT) ---
        if u in self._cache:
            self.hits += 1
            return self._cache[u]
        self.misses += 1

        # --- Kald fælles netværksklient (AsyncHtmlClient) ---
        # Vi kører et kortlivet event loop her, så kaldet også virker fra sync-kontekst.
        def _run_async_get(url: str) -> tuple[int, str]:
            import asyncio
            # Tolerer eksisterende loop (fx når kaldt inde fra async kode)
            try:
                loop = asyncio.get_running_loop()
                has_running = True
            except RuntimeError:
                loop = None
                has_running = False

            async def go(target: str) -> tuple[int, str]:
                # Import med fleksible stier (afhænger af modul-struktur)
                try:
                    from vexto.scoring.http_client import AsyncHtmlClient  # korrekt ift. dit repo
                except Exception:
                    # fallback når filen køres direkte som script og relative pakker driller
                    from ..scoring.http_client import AsyncHtmlClient  # type: ignore

                client = AsyncHtmlClient(stealth=True)
                await client.startup()
                try:
                    # Lad klienten selv afgøre om der skal renderes (heuristik/autodetect)
                    res = await client.get_raw_html(target, return_soup=False)
                    # Nogle versioner returnerer dict, andre ren str – håndter begge. (udvidet)
                    if isinstance(res, dict):
                        html = (
                            res.get("html")
                            or res.get("rendered_html")
                            or res.get("page_html")
                            or res.get("content")
                            or res.get("text")
                            or ""
                        )
                        status = int(res.get("status_code") or res.get("status") or (200 if html else 0))
                    else:
                        html = res or ""
                        status = 200 if html else 0

                    # Kontakt-fallback: hvis en kontakt-lignende URL ikke indeholder mailto, så prøv Playwright med målrettet vent
                    if (
                        status == 200
                        and "mailto:" not in (html or "").lower()
                        and any(kw in target.lower() for kw in ("/kontakt", "/contact", "/kontakt-os", "/om"))
                    ):
                        try:
                            # Vent eksplicit på email-links eller typiske person-containere
                            wait_selectors = [
                                "a[href^='mailto:'][class='email']",  # Præcis fra crawl: class="email"
                                "div.employee-card, .staff-member, .contact-person",#  Container fra crawl
                                "h3.name, .person-name",  # Name i h3.class="name"
                                "p.title, .role",  # Title i p.class="title"
                                "p.phone, .phone-number"   # Phone i p.class="phone"
                            ]
                            try:
                                res2 = await client.get_raw_html(
                                    target,
                                    force_playwright=True,
                                    wait_for_selectors=wait_selectors,
                                    return_soup=False,
                                    timeout=25.0,
                                )
                            except TypeError:
                                # Fald tilbage hvis klienten ikke kender ovenstående kwargs
                                try:
                                    res2 = await client.get_raw_html(
                                        target,
                                        force_playwright=True,
                                        return_soup=False,
                                    )
                                except TypeError:
                                    res2 = await client.get_raw_html(target, return_soup=False)

                            html2 = (
                                (res2.get("html")
                                or res2.get("rendered_html")
                                or res2.get("page_html")
                                or res2.get("content")
                                or res2.get("text"))
                                if isinstance(res2, dict) else (res2 or "")
                            )
                            if html2 and "mailto:" in html2.lower():
                                html = html2
                                log.info(f"PW fallback success for {target}: mailto found")
                        except Exception as e:
                            log.debug(f"PW fallback ignored for {target}: {e}")

                    return status, html

                finally:
                    # Kompatibel lukning: prøv .close() → .aclose() → .shutdown()
                    import inspect
                    for name in ("close", "aclose", "shutdown"):
                        fn = getattr(client, name, None)
                        if fn:
                            try:
                                res = fn()
                                if inspect.isawaitable(res):
                                    await res
                            except Exception:
                                pass
                            break
                                

            if has_running:
                # Kør i eksisterende loop (blokkerende indtil færdig)
                return loop.run_until_complete(go(url))  # type: ignore[arg-type]
            else:
                new_loop = asyncio.new_event_loop()
                try:
                    asyncio.set_event_loop(new_loop)
                    return new_loop.run_until_complete(go(url))
                finally:
                    new_loop.close()
                    asyncio.set_event_loop(None)

        # Special-case: API/JSON-URLs (render forvrænger svar) → hent råt uden PW
        try:
            from urllib.parse import urlsplit as _split
            _p = _split(u).path.lower()
        except Exception:
            _p = ""
        if _p.startswith("/wp-json"):
            try:
                import httpx
                with httpx.Client(timeout=12.0, follow_redirects=True) as _c:
                    r = _c.get(u, headers={"Accept": "application/json,*/*"})
                    text = r.text
                    self._cache[u] = (r.status_code, text or "")
                    log.info(f"DEBUG(API): Fetched {u} - status={r.status_code}, bytes={len(text or '')}")
                    return self._cache[u]
            except Exception:
                try:
                    import requests
                    r = requests.get(u, timeout=12.0, allow_redirects=True,
                                     headers={"Accept": "application/json,*/*"})
                    text = r.text if getattr(r, "ok", False) else ""
                    self._cache[u] = (getattr(r, "status_code", 0), text or "")
                    log.info(f"DEBUG(API): Fetched {u} (requests) - status={getattr(r, 'status_code', 0)}, bytes={len(text or '')}")
                    return self._cache[u]
                except Exception:
                    pass  # falder igennem til normal sti hvis alt fejler

        # NYT: Kald funktionen og debug (med logging + fallback)
        try:
            status, html = _run_async_get(u)
        except Exception as e:
            log.error(f"_run_async_get crashed for {u}: {e!r}", exc_info=True)
            # Simpel HTTP-fallback så vi undgår "stum" fejl
            try:
                import requests
                r = requests.get(u, timeout=12.0, allow_redirects=True,
                                headers={"User-Agent": "Mozilla/5.0"})
                status, html = r.status_code, (r.text if r.ok else "")
                log.info(f"Requests fallback: {u} -> status={status}, len={len(html)}")
            except Exception as fb_e:
                log.error(f"Requests fallback failed for {u}: {fb_e!r}")
                status, html = 0, ""

        # Fallback #1: prøv med 'www.'-host hvis vi ikke fik HTML
        host = _host_of(u)
        if (status == 0 or not html) and host and not host.startswith("www."):
            alt = _force_host(u, "www." + host)
            s2, h2 = _run_async_get(alt)
            if h2:
                log.debug(f"WWW fallback success: {alt} (len={len(h2)})")
                u, status, html = alt, s2, h2

        # Fallback #2: prøv http:// hvis https:// ikke gav HTML
        if (status == 0 or not html) and u.startswith("https://"):
            try:
                parts = urlsplit(u)
                alt = f"http://{parts.netloc}{parts.path or '/'}" + (f"?{parts.query}" if parts.query else "")
                s3, h3 = _run_async_get(alt)
                if h3:
                    log.debug(f"HTTP fallback success: {alt} (len={len(h3)})")
                    u, status, html = alt, s3, h3
            except Exception:
                pass

        log.info(f"DEBUG: Fetched {u} - status={status}, html_len={len(html) if html else 0}, contains_mailto={'mailto:' in (html or '').lower()}")

        # --- Gem i lokal cache og returnér ---
        self._cache[u] = (status, html or "")
        return status, html or ""

def _host_of(u: str) -> str:
    try:
        return urlsplit(u).hostname.lower()
    except Exception:
        return ""

def _force_host(u: str, host: str) -> str:
    try:
        parts = urlsplit(u)
        return f"{parts.scheme}://{host}{parts.path or '/'}" + (f"?{parts.query}" if parts.query else "")
    except Exception:
        return u

def _canonical_host_from(html: str, fallback: str) -> str:
    """Prøv at læse rel=canonical host; ellers behold fallback."""
    try:
        m = re.search(r'rel=["\']canonical["\']\s+href=["\']([^"\']+)["\']', html, flags=re.I)
        if m:
            return urlsplit(m.group(1)).netloc.lower() or fallback
    except Exception:
        pass
    return fallback

# [PATCH] ccTLD helpers (blokér cross-country redirects)
def _cc_tld_from_host(host: str):
    """Returner landekode-TLD (fx 'dk', 'se') hvis sidste label er 2 bogstaver, ellers None."""
    try:
        last = host.rsplit(".", 1)[1].lower()
    except Exception:
        return None
    if len(last) == 2 and last.isalpha():
        return last
    return None

def _same_country_tld(host_a: str, host_b: str) -> bool:
    """True hvis begge hosts har samme 2-bogstavs ccTLD (dk==dk, se==se)."""
    a = _cc_tld_from_host(host_a or "")
    b = _cc_tld_from_host(host_b or "")
    return bool(a and b and a == b)

def _norm_url_for_fetch(u: str) -> str:
    """Normalisér URL for dedup: fjern trailing slash (undtagen '/'), drop fragment."""
    try:
        s = urlsplit(u)
        path = re.sub(r"/{2,}", "/", s.path or "/")
        if path != "/" and path.endswith("/"):
            path = path[:-1]
        base = f"{s.scheme}://{s.netloc}{path}"
        return base + (f"?{s.query}" if s.query else "")
    except Exception:
        return u

def _fetch_with_playwright_sync(url: str, html0: str | None = None) -> str:
    """
    Render-side via AsyncHtmlClient/Playwright med:
    - Tvungen scroll (trigger lazy-load)
    - Udvidet venteliste af kontakt-/medarbejder-selectors
    - Fallback til oprindelige generiske selectors
    - UDEN afhængighed af modul-globals (_pw_fails_by_host/_pw_fails_max)
    """
    import asyncio
    import inspect
    from urllib.parse import urlsplit

    # Brug globale data hvis de findes – ellers sikre defaults
    _g = globals()
    _pw_fails_by_host = _g.setdefault("_pw_fails_by_host", {})
    _pw_fails_max = _g.get("_pw_fails_max", 2)
    _pw_disabled_hosts = _g.setdefault("_pw_disabled_hosts", set())

    _host = (urlsplit(url).netloc or "").lower()
    try:
        if _host in _pw_disabled_hosts:
            return html0 or ""
    except Exception:
        pass

    try:
        fails = _pw_fails_by_host.get(_host, 0)
        if fails >= _pw_fails_max:
            return html0 or ""
    except Exception:
        pass

    def _run() -> str:
        loop = asyncio.new_event_loop()
        try:
            asyncio.set_event_loop(loop)

            async def go() -> str:
                # Robust import mod forskellige projektstrukturer
                try:
                    from vexto.scoring.http_client import AsyncHtmlClient  # type: ignore
                except Exception:
                    try:
                        from src.vexto.scoring.http_client import AsyncHtmlClient  # type: ignore
                    except Exception:
                        from .scoring.http_client import AsyncHtmlClient  # type: ignore

                client = AsyncHtmlClient(stealth=True)
                await client.startup()
                try:
                    # A) Kontakt-/medarbejder-specifik venteliste
                    wait_contact = [
                        # primære signaler
                        "a[href^='mailto:']",
                        "a[href^='tel:']",
                        ".__cf_email__",
                        "[data-cfemail]",

                        # person-/team-kort
                        ".employee-card", ".person-card", ".contact-card",
                        ".contact-person", ".staff-member", ".team-member",
                        ".team__member", ".member", ".medarbejder", ".medarbejdere",

                        # headings/navn
                        "h1", "h2", "h3", "h4", "h5", "h6",
                        "[itemprop='name']", ".employee-name", ".person-name",

                        # titler
                        "[itemprop='jobTitle']", ".job-title", ".employee-title",
                        ".title", ".role", ".position", ".position-title",

                        # generiske layoutelementer som ofte indeholder kort
                        ".team", ".staff", ".employee", ".elementor-team-member",
                        ".et_pb_team_member", ".vc_row", ".vc_column", ".vc_team",
                    ]

                    # B) Oprindelige generiske (fallback)
                    wait_generic = [
                        "a[href^='mailto:']",
                        "a[href^='tel:']",
                        ".__cf_email__",
                        "[data-cfemail]",
                        ".team", ".staff", ".employee", ".elementor-team-member",
                        ".et_pb_team_member", ".vc_row", ".vc_column", ".vc_team",
                    ]

                    # 1) Første forsøg: scroll + brede kontakt-selectors
                    try:
                        res = await asyncio.wait_for(
                            client.get_raw_html(
                                url,
                                force_playwright=True,
                                # Tvungen scroll – ignoreres hvis klient ikke kender kwargs
                                scroll_to_bottom=True,
                                scroll_pause_ms=900,
                                scroll_repeat=2,
                                wait_after_scroll_ms=900,
                                # Vent på kontakt/medarbejder-selectors
                                wait_for_selectors=wait_contact,
                                return_soup=False,
                            ),
                            timeout=18.0,
                        )
                    except TypeError:
                        # Klienten kender ikke nye kwargs → prøv “simpelt” kald
                        res = await asyncio.wait_for(
                            client.get_raw_html(
                                url,
                                force_playwright=True,
                                return_soup=False,
                            ),
                            timeout=12.0,
                        )
                    except Exception:
                        # 2) Fallback: generiske selectors
                        try:
                            res = await asyncio.wait_for(
                                client.get_raw_html(
                                    url,
                                    force_playwright=True,
                                    wait_for_selectors=wait_generic,
                                    return_soup=False,
                                ),
                                timeout=14.0,
                            )
                        except TypeError:
                            res = await asyncio.wait_for(
                                client.get_raw_html(
                                    url,
                                    force_playwright=True,
                                    return_soup=False,
                                ),
                                timeout=12.0,
                            )

                    # Normalisér svar → html
                    if isinstance(res, dict):
                        html = (
                            res.get("html")
                            or res.get("rendered_html")
                            or res.get("page_html")
                            or res.get("content")
                            or res.get("text")
                            or ""
                        )
                        return html
                    return res or ""
                finally:
                    # Venlig lukning: .close() → .aclose() → .shutdown()
                    for name in ("close", "aclose", "shutdown"):
                        fn = getattr(client, name, None)
                        if fn:
                            try:
                                ret = fn()
                                if inspect.isawaitable(ret):
                                    await ret
                            except Exception:
                                pass
                            break

            return loop.run_until_complete(go())
        finally:
            try:
                loop.close()
            except Exception:
                pass
            try:
                asyncio.set_event_loop(None)
            except Exception:
                pass

    try:
        rendered_html = _run()
        # Reset fail-counter ved succes
        try:
            _pw_fails_by_host[_host] = 0
        except Exception:
            pass
        return rendered_html or (html0 or "")
    except Exception as e:
        # Opdater fail-counter og evt. disable host midlertidigt
        try:
            _pw_fails_by_host[_host] = _pw_fails_by_host.get(_host, 0) + 1
            if _pw_fails_by_host[_host] >= _pw_fails_max:
                _pw_disabled_hosts.add(_host)
        except Exception:
            pass
        try:
            log.debug(f"Playwright fejl for host={_host}, url={url}: {e!r}")
        except Exception:
            pass
        return html0 or ""

# -------------------------- Valgfri eksterne libs -----------------------------

try:
    import httpx  # hurtig HTTP, HTTP/2, gzip
except Exception:  # pragma: no cover
    httpx = None

try:
    from selectolax.parser import HTMLParser  # hurtig DOM
except Exception:  # pragma: no cover
    HTMLParser = None

try:
    from bs4 import BeautifulSoup  # robust fallback-parser
except Exception:  # pragma: no cover
    BeautifulSoup = None

if httpx is None:
    try:
        import requests  # pragma: no cover
    except Exception:
        requests = None  # type: ignore

# Brug den allerede konfigurerede modul-logger; ingen overskrivning her!
# Sikr ingen dobbelt-propagation op til root.
log.propagate = False
log.debug("ContactFinder module bootet og log-handler er aktiv.")

# ------------------------------- Konstanter -----------------------------------

# UI-ord (branche-agnostisk) der IKKE er titler
# UI-ord (branche-agnostisk) der IKKE er titler
UI_TITLE_BLACKLIST = {
    # eksisterende
    "kontakt", "kontakt os", "contact", "contact us",
    "team", "om", "about",
    "support", "help", "helpdesk",
    "mail", "email", "e-mail", "phone", "telefon",
    "info", "adresse", "address",
    "åbningstider", "opening hours",
    "faq", "privacy", "cookies", "cookie", "policy",
    # udvidede DK/EN fraser der ofte lander som “titel”-støj
    "vi værner gennemsigtighed", "har du spørgsmål", "ring til os", "skriv til os",
    "kundeservice", "kundecenter", "find medarbejder", "læs mere",
    "vi hjælper", "mød os", "vores team", "find os", "se mere", "book møde",
    "kontaktformular", "send besked", "send os en mail", "chat med os",
}

CONTACTISH_SLUGS = (
    # DA/EN
    "kontakt", "contact", "kontakt-os", "contact-us",
    "team", "teams", "medarbejder", "medarbejdere",
    "people", "staff", "ledelse", "about", "om-os",
    # SV/NO/DE
    "kontakt-oss", "kontakta", "om-oss", "om-oss/", "om-oss",
    "medarbetare", "personal", "mitarbeiter", "ueber-uns", "uber-uns"
)

# >>> ANKER START: DEDUP_AND_NORM_HELPERS
_EMAIL_RE = re.compile(r"[A-Z0-9._%+\-]{1,64}@[A-Z0-9.\-]{1,255}\.[A-Z]{2,}", re.I)
_PHONE_RE = re.compile(r"(?:\+?\d[\s\-\(\)\.]{0,3}){8,15}", re.I)

def _dedup_list(items: list[str]) -> list[str]:
    """Dedupliker en liste og bevarer rækkefølgen."""
    seen = set()
    return [x for x in items if not (x in seen or seen.add(x))]

def _strip_www(host: str | None) -> str | None:
    """Fjern 'www.' præfiks fra et hostnavn, hvis det findes."""
    if not host:
        return host
    return host[4:] if host.lower().startswith("www.") else host

def _with_www(host: str | None) -> str | None:
    """Tilføj 'www.' hvis ikke til stede. Returnerer None uændret."""
    if not host:
        return host
    return host if host.lower().startswith("www.") else f"www.{host}"
# <<< ANKER SLUT: DEDUP_AND_NORM_HELPERS

# === ANKER: STRUCTURED_CONTACT_EXTRACTOR ===
def extract_structured_contact(html: str) -> list[dict]:
    """Find kontaktpunkter i JSON-LD (Organization/ContactPoint) eller Microdata."""
    results: list[dict] = []
    # JSON-LD
    for m in re.finditer(
        r'<script[^>]+type=["\']application/ld\+json["\'][^>]*>(.*?)</script>',
        html, flags=re.I | re.S
    ):
        raw = (m.group(1) or "").strip()
        for blob in _maybe_json_objects(raw):
            results.extend(_extract_contacts_from_ldjson(blob))
    # Microdata (meget simpel – low/no-cost)
    for m in re.finditer(r'itemscope[^>]*itemtype=["\']https?://schema.org/Organization["\']', html, flags=re.I):
        # Greb efter telephone/email i nærheden
        snippet = html[max(0, m.start()-1500):m.end()+1500]
        emails = _EMAIL_RE.findall(snippet)
        tels = _PHONE_RE.findall(snippet)
        if emails or tels:
            results.append({
                "source": "microdata",
                "emails": _dedup_list(emails),
                "phones": _dedup_list([_normalize_phone(t) for t in tels if _normalize_phone(t)]),
            })
    return results

def _maybe_json_objects(raw: str):
    """Robust JSON-LD parser: håndter arrays/objekter + 'graph'."""
    try:
        data = json.loads(raw)
    except Exception:
        return []
    if isinstance(data, list):
        return data
    return [data]

def _extract_contacts_from_ldjson(obj) -> list[dict]:
    out: list[dict] = []
    try:
        if isinstance(obj, dict):
            # Organization på top-niveau
            if obj.get("@type") in ("Organization", ["Organization"]):
                cps = obj.get("contactPoint") or []
                if isinstance(cps, dict):
                    cps = [cps]
                for cp in cps:
                    if not isinstance(cp, dict):
                        continue
                    if cp.get("@type") not in (None, "ContactPoint", ["ContactPoint"]):
                        continue
                    email = cp.get("email")
                    tel = cp.get("telephone")
                    if email or tel:
                        out.append({
                            "source": "jsonld",
                            "emails": _dedup_list([email] if email else []),
                            "phones": _dedup_list([_normalize_phone(tel)] if tel else []),
                            "contactType": cp.get("contactType") or "",
                        })
    except Exception:
        pass
    return out

# === ANKER: PRECOMPILED REGEX (NEEDS_JS) ===
_CONTACT_CLASS_RE = re.compile(r'class=["\'][^"\']*(team|staff|employee|person|medarbejder|medarbejdere|contact-person|employee-card|staff-member)[^"\']*["\']', re.I)
_CONTACT_H_RE     = re.compile(r'<h[12][^>]*>\s*(kontakt|team|medarbejdere|personale|ansatte|about|om\s+os)\s*</h[12]>', re.I)
_MAILTO_TEL_RE    = re.compile(r'href=["\'](?:mailto:|tel:)', re.I)
_CONTACT_LINK_RE  = re.compile(r'href=["\'][^"\']*(/kontakt|/kontakt-os|/contact|/contact-us|/about|/om-?os|/team|/medarbejdere)[^"\']*["\']', re.I)
_JS_MARKERS_RE    = re.compile(
    r'(id=["\']__next["\']|data-reactroot|ng-app|vue-root|elementor-|wp-json|hydration|astro-island)',
    re.I
)

def __needs_js_primary(html: str) -> bool:
    if not html:
        return True

    if not html.strip():
        return True

    lower = html.lower()
    # --- NYT: Cloudflare e-mail-obfuscation kræver JS-render ---
    if "__cf_email__" in lower or "data-cfemail" in lower:
        return True
    # --- Kontakt-signaler (hurtig short-circuit) ---
    if _MAILTO_TEL_RE.search(lower): return False
    if _CONTACT_CLASS_RE.search(lower): return False
    if _CONTACT_H_RE.search(lower): return False
    if _CONTACT_LINK_RE.search(lower): return False
    # --- JS-markører: kun hvis ingen kontakt-signaler ---
    body_len = len(re.sub(r"\s+", "", html))
    return (body_len < 1800) and bool(_JS_MARKERS_RE.search(html))

def _is_contactish_url(u: str) -> bool:
    p = (urlsplit(u).path or "").strip("/").lower()
    return any(p.endswith(sl) or f"/{sl}/" in f"/{p}/" for sl in CONTACTISH_SLUGS)

def _is_initials_like(local: str) -> bool:
    letters = re.sub(r"[^a-zæøå]", "", local.lower())
    return 2 <= len(letters) <= 4

# [PATCH] alias → brug den primære heuristik
def _needs_js(html: str | None) -> bool:
    return __needs_js_primary(html or "")

_PROFILE_PATTERNS = [
    r'/medarbejder(?:e)?/[\w\-]+',
    r'/team/[\w\-]+',
    r'/people/[\w\-]+',
    r'/staff/[\w\-]+',
    r'/profile/[\w\-]+',
    r'/ansat(?:te)?/[\w\-]+',
    r'/personale/[\w\-]+',
]

def _cf_fetch_text_simple(self, any_url: str, timeout: float = 15.0) -> str | None:
    """Let GET med egen http_client hvis tilgængelig; ellers httpx fallback."""
    try:
        if hasattr(self, "http_client") and hasattr(self.http_client, "get_text_smart"):
            status, html = self.http_client.get_text_smart(any_url)
            if (status or status == 200) and html:
                return html
    except Exception:
        pass
    try:
        import httpx
        with httpx.Client(timeout=timeout, follow_redirects=True) as cli:
            r = cli.get(any_url)
            if r.status_code == 200 and r.text:
                return r.text
    except Exception:
        pass
    return None

def _cf_discover_sitemaps(self, base_url: str) -> list[str]:
    from urllib.parse import urlsplit, urlunsplit
    sp = urlsplit(base_url)
    robots = urlunsplit((sp.scheme, sp.netloc, "/robots.txt", "", ""))
    txt = _cf_fetch_text_simple(self, robots) or ""
    maps: list[str] = []
    for line in txt.splitlines():
        if line.lower().startswith("sitemap:"):
            u = line.split(":", 1)[1].strip()
            if u:
                maps.append(u)
    if not maps:
        guess = urlunsplit((sp.scheme, sp.netloc, "/sitemap.xml", "", ""))
        maps.append(guess)
    # dedup
    seen = set(); out = []
    for m in maps:
        if m not in seen:
            seen.add(m); out.append(m)
    return out

def _cf_extract_urls_from_sitemap_xml(self, xml_text: str) -> list[str]:
    import xml.etree.ElementTree as ET
    urls: list[str] = []
    try:
        root = ET.fromstring(xml_text)
        ns = {"sm": root.tag.split("}")[0].strip("{")} if "}" in root.tag else {}
        # urlset
        url_nodes = root.findall(".//sm:url", ns) if ns else root.findall(".//url")
        for url_el in url_nodes:
            loc_el = url_el.find("sm:loc", ns) if ns else url_el.find("loc")
            if loc_el is not None:
                loc = (loc_el.text or "").strip()
                if loc:
                    urls.append(loc)
        # sitemapindex → følg op til 10 under-sitemaps
        if not urls:
            site_nodes = root.findall(".//sm:sitemap", ns) if ns else root.findall(".//sitemap")
            for s_el in site_nodes[:10]:
                loc_el = s_el.find("sm:loc", ns) if ns else s_el.find("loc")
                if loc_el is None:
                    continue
                sub = (loc_el.text or "").strip()
                if not sub:
                    continue
                sub_xml = _cf_fetch_text_simple(self, sub)
                if sub_xml:
                    urls.extend(_cf_extract_urls_from_sitemap_xml(self, sub_xml))
    except Exception:
        pass
    return urls

def _cf_sitemap_contact_candidates(self, base_url: str) -> list[str]:
    from urllib.parse import urlsplit
    pats = ("kontakt", "contact", "kundeservice", "customer-service",
            "about", "om-os", "team", "staff", "support", "help")
    try:
        host = urlsplit(base_url).netloc
        cands: list[str] = []
        for sm in _cf_discover_sitemaps(self, base_url)[:5]:
            xml = _cf_fetch_text_simple(self, sm)
            if not xml:
                continue
            for u in _cf_extract_urls_from_sitemap_xml(self, xml):
                sp = urlsplit(u)
                if sp.netloc and sp.netloc != host:
                    continue  # kun interne
                if any(p in u.lower() for p in pats):
                    cands.append(_norm_url_for_fetch(u))
        # dedup + cap
        seen = set(); out = []
        for u in cands:
            if u not in seen:
                seen.add(u); out.append(u)
        return out[:5]
    except Exception:
        return []

def _discover_profile_links(base_url: str, html: str, max_links: int = 40) -> list[str]:
    """Find medarbejder-/profil-URLs i HTML – nu med prioritering på anker-tekst og "contactish" hints."""
    
    # Hent disallow rules først
    _, disallows = _fetch_robots_txt(base_url)
    
    scored: list[tuple[int, str]] = []
    seen_raw: set[str] = set()

    # 1) Gå via <a ...> for at kunne score på anker-tekst
    for m in re.finditer(r'<a\s+[^>]*href=["\']([^"\']+)["\'][^>]*>(.*?)</a>', html, flags=re.I | re.S):
        href = (m.group(1) or "").strip()
        text = re.sub(r"<[^>]+>", " ", m.group(2) or "").lower()
        # Kun interne links
        absu = _abs_url(base_url, href)
        absu = _norm_url_for_fetch(absu)
        if not absu or not _same_host(base_url, absu):
            continue
            
        path = (urlsplit(absu).path or "").lower()
        
        # Check om stien er disallowed
        if _is_disallowed(path, disallows):
            log.debug(f"Skipping disallowed profile link: {absu}")
            continue

        # Matcher et profil-mønster?
        if not any(re.search(pat, path, flags=re.I) for pat in _PROFILE_PATTERNS):
            continue

        # Basisscore
        score = 1
        # Boost hvis anker-tekst ligner kontakt/person
        if any(kw in text for kw in ("kontakt", "contact", "email", "mail", "telefon", "phone",
                                     "team", "staff", "medarbejder", "medarbejdere", "personale", "profil", "ansat", "people")):
            score += 2
        # Boost hvis stien selv er "contactish"
        if _is_contactish_url(absu):
            score += 1
        # Favoriser kortere stier
        score += max(0, 2 - len(path.split("/")))

        if absu not in seen_raw:
            seen_raw.add(absu)
            scored.append((score, absu))

    # 2) Fald tilbage: simple href-scan hvis ovenstående ikke fandt noget
    if not scored:
        for pat in _PROFILE_PATTERNS:
            for m in re.finditer(rf'href=["\']({pat})["\']', html, flags=re.I):
                try:
                    full = urljoin(base_url, m.group(1))
                    if _same_host(base_url, full) and full not in seen_raw:
                        seen_raw.add(full)
                        scored.append((1, full))
                except Exception:
                    continue

    # 3) Sortér på score (desc) + kort sti
    scored.sort(key=lambda t: (-t[0], len(urlsplit(t[1]).path)))
    out, seen = [], set()
    for _, u in scored:
        if u not in seen:
            out.append(u); seen.add(u)
        if len(out) >= max_links:
            break
    return out

# >>> ANKER START: DISCOVERY_HELPERS
def _detect_lang_from_html(html: str) -> str:
    """Detekter sprog fra HTML (lang-attribut eller heuristik)."""
    m = re.search(r'<html[^>]+lang=["\']([a-zA-Z-]{2,})["\']', html, flags=re.I)
    if m:
        return m.group(1).lower()
    txt = (html or "").lower()
    if "medarbejder" in txt or "kontakt os" in txt:
        return "da"
    if "medarbetare" in txt or "om oss" in txt:
        return "sv"
    if "contact" in txt or "about us" in txt:
        return "en"
    return "da"

_LANG_TERMS = {
    "da": ["kontakt", "kontakt os", "om", "om os", "team", "medarbejder", "medarbejdere", "ansatte", "personale", "ledelse"],
    "sv": ["kontakt", "om oss", "team", "medarbetare", "personal", "ledning"],
    "en": ["contact", "contact us", "about", "about us", "team", "staff", "management"],
}

def _contactish_terms(lang: str) -> list[str]:
    """Returner sprog-specifikke kontakt-relaterede termer."""
    return _LANG_TERMS.get(lang.split("-")[0], _LANG_TERMS["da"])

def discover_candidates(
    base_url: str,
    base_html: str,
    max_urls: int = 5,
    http_client: Optional[CachingHttpClient] = None,
) -> list[str]:
    """Returner interne kandidatsider i prioriteret rækkefølge (Home → Sitemap → WP)."""

    import ipaddress
    from urllib.parse import urlsplit

    # --- helpers ---
    def _is_private_or_ip(u: str) -> bool:
        try:
            host = (urlsplit(u).hostname or "").strip().lower()
            if not host:
                return True
            if host in {"localhost"}:
                return True
            # IP literal?
            try:
                ip = ipaddress.ip_address(host)
                return (ip.is_private or ip.is_loopback or ip.is_link_local)
            except ValueError:
                # not an IP literal
                pass
            return False
        except Exception:
            return True

    def _has_negatives(s: str) -> bool:
        NEG = (
            "privacy", "privatliv", "persondata", "cookie", "cookies",
            "gdpr", "terms", "betingelser", "vilkår", "legal", "politik",
            "recruit", "rekruttering", "karriere", "job", "stillinger",
            "klage", "patientinformation", "sitemap"
        )
        s = s.lower()
        return any(n in s for n in NEG)

    # --- valider base_url ---
    if not base_url or not isinstance(base_url, str):
        return []
    base_url = base_url.strip()
    if base_url.lower() in ("nan", "none", "null", ""):
        return []
    if not base_url.startswith(('http://', 'https://')):
        if re.match(r'^(www\.)?([a-z0-9\-]+\.)+[a-z]{2,}', base_url, re.I):
            base_url = f"https://{base_url}"
        else:
            return []

    seen: set[str] = set()
    ranked: list[tuple[int, str]] = []

    client = http_client or CachingHttpClient(timeout=DEFAULT_TIMEOUT)

    # 1) Home-link scoring
    lang = _detect_lang_from_html(base_html or "")
    terms = _contactish_terms(lang)
    for m in re.finditer(r'<a\s+[^>]*href=["\']([^"\']+)["\'][^>]*>(.*?)</a>', base_html, flags=re.I | re.S):
        href = (m.group(1) or "").strip()
        text = re.sub(r"<[^>]+>", " ", (m.group(2) or "")).lower().strip()
        abs_u = urljoin(base_url, href)
        abs_u = _norm_url_for_fetch(abs_u)  # ← NYT: dedup '/kontakt' vs '/kontakt/'
        if not _same_host(base_url, abs_u):
            continue
        if _is_private_or_ip(abs_u):
            log.debug(f"Skip private/IP candidate from home: {abs_u}")
            continue
        score = 0
        u_low = abs_u.lower()
        for t in terms:
            if t in text:
                score += 4
            if t.replace(" ", "-") in u_low:
                score += 3
        if score and not _has_negatives(u_low):
            ranked.append((score, abs_u))
            seen.add(abs_u)

    # 2) robots.txt → sitemaps
    sitemaps, disallows = _fetch_robots_txt(base_url, http_client=client)

    # Check sitemaps for contact pages
    pos_slugs = tuple(t.replace(" ", "-") for t in terms)
    for sm_url in sitemaps[:3]:  # Max 3 sitemaps
        try:
            sitemap_urls = _fetch_sitemap_urls(sm_url, http_client=client)
            for loc in sitemap_urls:
                if not _same_host(base_url, loc):
                    continue

                # Respekter Disallow
                loc_path = urlsplit(loc).path or "/"
                if _is_disallowed(loc_path, disallows):
                    log.debug(f"Skipping disallowed URL: {loc}")
                    continue

                # Filtrér private/IP/localhost og security-check domæner
                host_low = (urlsplit(loc).hostname or "").lower()
                if _is_private_or_ip(loc) or "security-check." in host_low:
                    log.debug(f"Skipping non-public/blocked host: {loc}")
                    continue

                loc_low = loc.lower()
                if any(p in loc_low for p in pos_slugs) and not _has_negatives(loc_low):
                    if loc not in seen:
                        ranked.append((3, loc))
                        seen.add(loc)

        except Exception as e:
            log.debug(f"Error fetching sitemap {sm_url}: {e}")
            continue

    # 3) WordPress REST (hvis tilgængelig og ikke disallowed)
    #    Ensret til trailing slash og undgå dobbelte kald.
    if not _is_disallowed("/wp-json/", disallows):
        wp_root = urljoin(base_url, "/wp-json/")
        st_wp, wp_probe = client.get(wp_root)
        if st_wp != 200:
            log.debug(f"WP REST not available (status {st_wp}) for {base_url}")
        else:
            # Kun fortsæt hvis svaret LIGNER JSON – ellers er det ikke et WP REST-endpoint.
            _probe = (wp_probe or "").lstrip()
            if not (_probe.startswith("{") or _probe.startswith("[")):
                log.debug("WP REST probe returned non-JSON (likely SPA HTML) – skipping WP REST path")
            else:
                # >>> ANKER til WP REST logik (pages/search osv.)
                KEYS = ("kontakt", "om", "about", "team", "medarbejdere", "staff", "management")
                wp_blocked = False

                # 1) Liste alle pages én gang og filtrér klient-side
                api_list = urljoin(base_url, "/wp-json/wp/v2/pages/?_fields=link,title&per_page=100")
                st_wp, body = client.get(api_list)
                if st_wp in (401, 403, 404):
                    wp_blocked = True
                elif st_wp == 200 and body:
                    try:
                        for item in json.loads(body):
                            link = (item.get("link") or "").strip()
                            title = ((item.get("title") or {}).get("rendered") or "")
                            if not link:
                                continue
                            lk = link.rstrip("/")
                            s = f"{lk} {title}".lower()

                            if _is_private_or_ip(lk):
                                continue
                            if _has_negatives(s):
                                continue
                            if any(k in s for k in KEYS):
                                if lk not in seen:
                                    seen.add(lk)
                                    ranked.append((4, lk))
                    except Exception:
                        pass

                # 2) Minimal fallback: 1–2 søgninger, kun hvis ikke blokeret
                if not wp_blocked and not any(score == 4 for score, _ in ranked[-8:]):
                    for t in ("kontakt", "om"):
                        api = urljoin(base_url, f"/wp-json/wp/v2/pages/?search={t}&_fields=link,title&per_page=10")
                        st_wp, body = client.get(api)
                        if st_wp in (401, 403, 404):
                            break
                        if st_wp == 200 and body:
                            try:
                                for item in json.loads(body):
                                    link = (item.get("link") or "").strip()
                                    if not link:
                                        continue
                                    lk = link.rstrip("/")
                                    if _is_private_or_ip(lk):
                                        continue
                                    if _has_negatives(lk):
                                        continue
                                    if lk not in seen:
                                        seen.add(lk)
                                        ranked.append((4, lk))
                            except Exception:
                                pass
                # <<< ANKER: CF/WP_REST_SHALLOW

    ranked.sort(key=lambda x: x[0], reverse=True)
    urls = [u for _, u in ranked[:max_urls]]

    # Bind kandidater til canonical host for at undgå 302-pingpong
    try:
        urls = _sticky_host_urls(base_url, base_html, urls)
    except Exception:
        pass

    # Final dedup + sidste private/security-check filter
    out, seen2 = [], set()
    for u in urls:
        if u in seen2:
            continue
        if _is_private_or_ip(u):
            log.debug(f"Drop private/IP at finalize: {u}")
            continue
        host_low = (urlsplit(u).hostname or "").lower()
        if "security-check." in host_low:
            continue
        seen2.add(u)
        out.append(u)
    return out


# Rolle-ord (generiske)
EXEC_ROLES = {
    "ceo","cto","cfo","coo","cmo","cpo","cio","chair","founder","co-founder",
    "direktør","formand",
    # DK-varianter som ofte står på kontakt-/team-sider
    "medindehaver","medejer","ejer"
}
MANAGER_ROLES = {
    "manager","head","lead","chef","leder","afdelingsleder","projektleder","produktchef"
}
SPECIALIST_ROLES = {
    "konsulent","specialist","coordinator","koordinator","analyst","analytiker",
    "engineer","developer","udvikler","designer","rådgiver","architect"
}

ROLE_LIKE = EXEC_ROLES | MANAGER_ROLES | SPECIALIST_ROLES

# --- ANKER: DIRECTOR_HINT_HOOK ---
# Valgfri integration: vexto.enrichment.cvr_bridge.director_names_for_host(host) -> iterable[str]
try:
    from vexto.enrichment.cvr_bridge import director_names_for_host as _director_names_for_host  # type: ignore
except Exception:
    _director_names_for_host = None  # type: ignore

def _norm_person_name(s: Optional[str]) -> str:
    if not s:
        return ""
    t = re.sub(r"[\s\.,;:’'\"-]+", " ", s.strip().lower())
    return re.sub(r"\s+", " ", t).strip()

def _director_match_for_site(name: Optional[str], site_url: Optional[str]) -> bool:
    if not name or not site_url or _director_names_for_host is None:
        return False
    try:
        host = (urlsplit(site_url).hostname or "").lower()
        candidates = list(_director_names_for_host(host) or [])
        n0 = _norm_person_name(name)
        return any(_norm_person_name(c) == n0 for c in candidates)
    except Exception:
        return False


# Generiske e-mail-brugere
GENERIC_EMAIL_USERS = {
    "info","kontakt","mail","sales","support","hello","office",
    "billing","invoice","noreply","no-reply","donotreply","admin",
}

# Simpel navnetoken (kapitaliseret, med danske bogstaver)
NAME_TOKEN = r"[A-ZÆØÅ][a-zA-ZÀ-ÖØ-öø-ÿ'’\.-]{1,30}"

# Sider vi prøver udover root
FALLBACK_PATHS = (
    # EN
    "/contact", "/contact/", "/contact-us", "/contact.aspx", "/contact-us.aspx",
    "/about", "/about/", "/about-us", "/about-us.aspx",
    "/people", "/staff", "/management",
    # DA
    "/kontakt", "/kontakt/", "/kontakt-os", "/kontakt-os.aspx",
    "/om", "/om/", "/om-os", "/om-os/", "/om-os.aspx",
    "/team", "/team/", "/teams", "/vores-team", "/team.aspx",
    "/medarbejder", "/medarbejder/", "/medarbejdere", "/medarbejdere/",
    "/ansatte", "/personale", "/ledelse", "/board", "/organisation",
    # SV/NO/DE
    "/kontakt-oss", "/kontakta-oss", "/kontakta",
    "/om-oss", "/om-oss/", "/medarbetare", "/personal",
    "/mitarbeiter", "/ueber-uns", "/uber-uns", "/unternehmen"
)

# --- Fast HEAD status cache + URL-normalisering ---
from functools import lru_cache

def _norm(u: str) -> str:
    return _norm_url_for_fetch(u)

def _looks_danish(html: str) -> bool:
    return bool(re.search(r"\b(kontakt|om os|åbningstider|find os|vi hjælper)\b", (html or "").lower()))

def _localized_fallback_paths(base_url: str, home_html: str) -> tuple[str, ...]:
    host = urlsplit(base_url).netloc.lower()
    dk_like = host.endswith(".dk")
    da = dk_like or _looks_danish(home_html)
    if da:
        return (
            "/kontakt", "/kontakt/", "/kontakt-os", "/kontakt-os.aspx",
            "/om", "/om/", "/om-os", "/om-os/", "/om-os.aspx",
            "/team", "/team/", "/teams", "/team.aspx",
            "/medarbejder", "/medarbejder/", "/medarbejdere", "/medarbejdere/",
            "/ansatte", "/personale", "/ledelse"
        )
    else:
        return (
            "/contact", "/contact/", "/contact-us", "/contact.aspx", "/contact-us.aspx",
            "/about", "/about/", "/about-us", "/about-us.aspx",
            "/team", "/team/", "/people", "/staff", "/management"
        )

# Link-discovery (ankertekst/URL indeholder disse)
DISCOVERY_KEYWORDS = {
    "kontakt", "contact", "kontakt os", "contact us",
    "om os", "about", "about us",   # fjernet enkeltordet "om"
    "team", "teams", "vores team", "vores-hold", "holdet",
    "medarbejder", "medarbejdere", "ansatte", "personale",
    "people", "staff",
    "ledelse", "management", "board",
    "organisation", "company"
}

DEFAULT_TIMEOUT = 12.0  # sekunder

def _sticky_host_urls(base_url: str, home_html: str, urls: list[str]) -> list[str]:
    """P1: Ensret alle kandidat-URLs til canonical host efter første GET."""
    base_host = _host_of(base_url)
    canon_host = _canonical_host_from(home_html, base_host) or base_host
    if canon_host == base_host:
        return urls
    return [_force_host(u, canon_host) if _host_of(u) and _host_of(u) != canon_host else u for u in urls]

__all__ = [
    "ContactFinder",
    "find_best_contact",
    # aliaser
    "find_contacts",
    "extract_contacts",
    "extract_best_contact",
    "find_top_contact",
    # test/utility shims
    "run_enrichment_on_dataframe",
    "check_api_status",
]

# ------------------------------- Datamodeller --------------------------------

@dataclass
class ContactCandidate:
    name: Optional[str] = None
    title: Optional[str] = None
    emails: list[str] = None  # type: ignore
    phones: list[str] = None  # type: ignore
    source: str = ""          # 'json-ld' | 'rdfa' | 'microformats' | 'mailto' | 'dom'
    url: str = ""
    dom_distance: Optional[int] = None
    hints: dict[str, Any] = None  # fx selector/neartext

    def __post_init__(self):
        if self.emails is None:
            self.emails = []
        if self.phones is None:
            self.phones = []
        if self.hints is None:
            self.hints = {}

@dataclass
class ScoredContact:
    candidate: ContactCandidate
    score: float
    reasons: list[str]

def _confidence_from(sc: ScoredContact) -> float:
    """Heuristisk 0–1 confidence ud fra score + nøglesignaler."""
    s = max(0.0, min(10.0, float(sc.score or 0.0)))
    conf = s / 8.0  # base normalisering
    rs = sc.reasons or []
    if any(str(r).startswith("EMAIL_MATCH") for r in rs):
        conf += 0.10
    if "DOM_NEAR" in rs:
        conf += 0.05
    if "ROLE_3" in rs:
        conf += 0.10
    if "GENERIC_EMAIL" in rs:
        conf -= 0.10
    # clamp 0..1 og afrund
    if conf < 0.0: conf = 0.0
    if conf > 1.0: conf = 1.0
    return round(conf, 3)
# === ANKER: CONFIDENCE_HELPER ===

# === ANKER: TITLE_MATCH_VALIDATION ===
# [NYT] Debug-flag + tællere
import os, logging
log = logging.getLogger(__name__)
_TITLES_DEBUG = os.getenv("VEXTO_TITLES_DEBUG", "0") == "1"
_TITLE_FACIT_OK = 0
_TITLE_FACIT_REJECT = 0
_TITLE_FACIT_SKIPPED = 0

_TITLE_MATCHER = None  # lazy-loaded singleton

def _load_title_matcher():
    global _TITLE_MATCHER
    if _TITLE_MATCHER is None:
        try:
            from vexto.enrichment.title_matcher import TitleMatcher
        except Exception:
            # fallback til relativ import når filen køres direkte
            from .title_matcher import TitleMatcher  # type: ignore
        _TITLE_MATCHER = TitleMatcher.load()
    return _TITLE_MATCHER

def _canonicalize_title_or_none(raw_title: str | None):
    """
    Primært validering – ikke overskrivning.
    Acceptér KUN canonical ved exact/override eller høj fuzzy (≥0.95).
    Ellers behold rå titel (hvis den ligner en rolle).
    """
    if not raw_title:
        return None

    try:
        from vexto.enrichment.title_matcher import TitleMatcher  # type: ignore
        matcher = TitleMatcher.load()
        hit = matcher.match(raw_title)
        if hit:
            title_id, canonical, match_type, score = hit
            score = float(score)
            if (match_type in {"exact", "override"}) or (match_type == "fuzzy" and score >= 0.98):
                return canonical, {
                    "title_id": title_id,
                    "match_type": match_type,
                    "score": score,
                }
            else:
                if _TITLES_DEBUG and match_type == "fuzzy":
                    log.debug(f"[TitleFacit] Rejected fuzzy morph: raw={raw_title!r} -> {canonical!r} (score={score} < 0.98)")
            # ellers: vi falder tilbage til rå titel nedenfor
    except Exception as e:
        if _TITLES_DEBUG:
            log.debug(f"[TitleFacit] ERROR raw={raw_title!r} err={e}")

    # Behold rå titel når den ligner en rolle (konservativ visning)
    try:
        if _looks_like_role(raw_title):
            return raw_title, {"match_type": "raw", "score": 0.5}
    except Exception:
        pass

    return None


# ------------------------------- Utils ----------------------------------------

def _collapse_ws(s: str | None) -> str | None:
    if not s:
        return None
    return re.sub(r"\s+", " ", str(s)).strip()

def _is_plausible_name(s: str | None) -> bool:
    """Vurder om tekst ligner et personnavn.
    - 2–4 tokens hvor mindst 2 starter med stort
    - ingen cifre/@
    - frasortér UI-tekst, adresseord og kendte ikke-navne
    # ANKER: NAME_VALIDATION_RELAXED
    """
    if not s:
        return False
    s = _collapse_ws(s) or ""
    if not s:
        return False

    low = s.lower()

    UI_BLACKLIST = {
        # UI/CTA/sektioner
        "kontakt","kontakt os","kontakt os på","contact","contact us","om","om os","about","about us",
        "ring til os","skriv til os","book","bestil","vores team","team","har du spørgsmål",
        "find medarbejder","læs mere","kundeservice","personale","medarbejdere",
        "menu","trustpilot","cookie"
    }
    # Tillad både eksakt match og indeholdt phrases (fx "Kontakt Os På Trafikcenter...")
    if low in UI_BLACKLIST or any(p in low for p in ("kontakt os", "kontakt os på", "trafikcenter", "dk-4200", "slagelse")):
        return False

    # hurtige hard-stops
    if re.search(r"[@\d]", s):
        return False
    if s.upper() == s:
        return False

    # NYT: kassér typiske rolle/titel-strenge forklædt som navn
    #  - “Kundeservice / Salg” (slash) er næsten aldrig et personnavn
    if "/" in s:
        return False

    #  - Hvis facit kanoniserer strengen til en kendt titel, er det ikke et navn
    try:
        facit = _canonicalize_title_or_none(s)
        if facit:
            return False
    except Exception:
        pass

    #  - Ekstra sikkerhed: indeholder strengen generiske rolle-ord (da/en)?
    low2 = s.lower()
    for role in _ROLE_ALL:
        if role in low2:
            return False

    # adressemønstre (DK/EN)
    ADDR_TOKENS = {
        "vej","gade","allé","alle","boulevard","plads","torv","center","centret",
        "trafikcenter","road","street","ave","avenue","blvd","square","plaza","centre"
    }
    # split til ord, bevar danske bogstaver
    tokens = [t for t in re.split(r"[^\wÆØÅæøå\-]+", s) if t]
    if not (2 <= len(tokens) <= 4):
        return False
    if any(t.lower() in ADDR_TOKENS for t in tokens):
        return False

    # mindst to kapitaliserede “navneord” – tillad småord som af/de/van/von
    SMALL_CONNECTORS = {"af","de","del","der","van","von","da","di"}
    cap_tokens = [t for t in tokens if t[:1].isupper() and t.lower() not in SMALL_CONNECTORS]
    if len(cap_tokens) < 2:
        return False

    return True

# --- Smartere titel-udtræk fra tekstvindue omkring navn (DK/EN) ---
_ROLE_LEXICON_DA = {
    "adm. direktør","administrerende direktør","direktør","salgsdirektør","salgschef","kundeservice",
    "kundechef","marketingchef","marketing","økonomi","bogholder","regnskab","indkøb","indkøber",
    "produktchef","lagerchef","driftschef","logistik","support","salg","key account manager",
    "konsulent","projektleder","butikschef","webshop","hr","personalechef","service"
}
_ROLE_LEXICON_EN = {
    "chief executive officer","ceo","managing director","director","sales director","sales manager",
    "customer service","account manager","marketing manager","marketing","finance","bookkeeper",
    "purchasing","procurement","product manager","operations manager","logistics","support","sales",
    "consultant","project manager","store manager","hr","human resources","service"
}
# prioriter længere matches
_ROLE_ALL = sorted(_ROLE_LEXICON_DA | _ROLE_LEXICON_EN, key=len, reverse=True)

def _title_from_text_window(text: str, name: str | None = None) -> str | None:
    text = _collapse_ws(text or "") or ""
    if not text:
        return None
    low = text.lower()

    # hvis vi kender navnet, kig i et vindue ±120 tegn
    if name:
        nlow = name.lower()
        i = low.find(nlow)
        if i != -1:
            start = max(0, i - 120); end = min(len(low), i + len(nlow) + 160)
            low = low[start:end]

    # prøv “Navn – Titel” / “Navn, Titel”
    if name:
        safe = re.escape(name)
        m = re.search(rf"{safe}\s*[:,\-–—]\s*([A-Za-zÆØÅæøå/ &\-]{{3,80}})", text, flags=re.IGNORECASE)
        if m:
            cand = _sanitize_title(m.group(1))
            if cand:
                # [NYT] Facit-validering
                canon = _canonicalize_title_or_none(cand)
                if canon:
                    return canon[0]
                return None  # falder igennem hvis ikke i katalog

    # prøv rolle-leksikon
    for role in _ROLE_ALL:
        if role in low:
            m = re.search(re.escape(role), text, flags=re.IGNORECASE)
            if m:
                cand = _sanitize_title(m.group(0))
                if cand:
                    # [NYT] Facit-validering
                    canon = _canonicalize_title_or_none(cand)
                    if canon:
                        return canon[0]
                    return None

    return None


def _maybe_promote_name_case(s: str | None) -> Optional[str]:
    """
    Hvis s er helt lowercase men ligner 2–4 navnetokens, returnér kapitaliseret udgave.
    Fx 'brian pedersen' -> 'Brian Pedersen'.
    """
    if not s:
        return None
    t = _collapse_ws(s) or ""
    if not t:
        return None
    # 2-4 ord, kun bogstaver, min. længde 2
    if re.fullmatch(r"[a-zæøå]{2,}(?:\s+[a-zæøå]{2,}){1,3}", t, flags=re.I):
        parts = t.split()
        cap = " ".join(p[:1].upper() + p[1:] for p in parts)
        return cap
    return None

def _sanitize_title(title: str | None) -> str | None:
    if not title:
        return None

    # Bevar original case i output, men normalisér whitespace
    t = _collapse_ws(title) or ""
    # Rens ender for typiske separatorer
    t = t.strip().strip(",;:–—-·|/\\ ")

    # 1) Fjern UI-ord (din blacklist)
    pat = r"\b(" + "|".join(map(re.escape, UI_TITLE_BLACKLIST)) + r")\b"
    t = re.sub(pat, "", t, flags=re.I)
    t = re.sub(r"\s{2,}", " ", t).strip(",;:–—-·|/\\ ").strip()

    # 2) Drop åbenlys støj: emails/URL'er
    if "@" in t or re.search(r"https?://", t, flags=re.I):
        return None

    # 3) Telefonnumre: fjern trailing/leading phone-snippets eller kasser hele strengen
    if re.search(r"\b\d{2}\s?\d{2}\s?\d{2}\s?\d{2}\b", t):  # dansk 8-cifret mønster
        # hvis det ligner "… 58 55 07 15" i enden eller starten -> fjern
        t = re.sub(r"(^|\s)[+\d][\d\s\-\(\)]{6,}($|\s)", " ", t).strip()
    # Internationalt format (+45 …)
    t = re.sub(r"(^|\s)\+\d[\d\s\-\(\)]{6,}($|\s)", " ", t).strip()

    # 4) Hvis for mange cifre i resten -> drop
    digits = sum(ch.isdigit() for ch in t)
    if t and digits / max(1, len(t)) > 0.20:
        return None

    # 5) Drop typiske CTA-/kampagne-linjer (meget sikre signaler)
    if re.search(r"%|\b(spar|tilbud|udsalg|klik|shop|lager|rab[aæ]t)\b", t, flags=re.I):
        return None

    # 6) Ryd resterende støj-tegn
    t = re.sub(r"[|•·]+", " ", t)
    t = re.sub(r"\s{2,}", " ", t).strip(" -–—·|")

    # 7) Længde- og sundhedstjek
    if len(t) < 3 or len(t) > 80:
        return None

    return t or None

def _role_strength(title: str | None) -> int:
    if not title:
        return 0
    t = title.lower()
    if any(k in t for k in EXEC_ROLES): return 3
    if any(k in t for k in MANAGER_ROLES): return 2
    if any(k in t for k in SPECIALIST_ROLES): return 1
    return 0

def _looks_like_role(title: str | None) -> bool:
    return _role_strength(title) > 0

def _normalize_email(e: str | None) -> Optional[str]:
    if not e:
        return None
    e = e.strip().strip("<>").lower()
    e = re.sub(r"^mailto:\s*", "", e)
    if not re.match(r"^[^@\s]+@[^@\s]+\.[^@\s]+$", e):
        return None
    return e

def _is_generic_local(local: str) -> bool:
    return local in GENERIC_EMAIL_USERS

def _email_name_score(email: str, full_name: str) -> int:
    """1 point pr. navne-del i email-brugernavn (robust simpelt match)."""
    try:
        user = email.split("@", 1)[0].lower()
    except Exception:
        return 0
    user = re.sub(r"[^a-z0-9æøå]+", "", user)
    parts = [p for p in re.split(r"[^a-zæøå]+", (full_name or "").lower()) if len(p) >= 2]
    return sum(1 for p in parts if p in user)

def _best_email_score(emails: Iterable[str], full_name: str | None) -> int:
    if not emails or not full_name:
        return 0
    best = 0
    for em in emails:
        sc = _email_name_score(em, full_name)
        if sc > best:
            best = sc
    return best

def _normalize_phone(raw: str | None) -> Optional[str]:
    """Bevar + for intl; autoudfyld +45 for 8-cifrede DK-numre; valider 8–15 cifre."""
    if not raw:
        return None
    s = re.sub(r'[\s\-\(\)\.]+', '', raw)          # fjern visuelle formateringer
    s = re.sub(r'[^\d+]', '', s)                   # behold + og cifre
    if s.startswith('00'):                         # intl 00 → +
        s = '+' + s[2:]
    if re.fullmatch(r'[2-9]\d{7}', s):             # DK national (8 cifre, 2–9 start)
        s = '+45' + s
    digits = re.sub(r'\D', '', s)
    if len(digits) < 8 or len(digits) > 15:
        return None
    if not s.startswith('+'):
        s = '+' + digits
    return s

def _cf_decode(hexstr: str | None) -> Optional[str]:
    # Cloudflare __cf_email__ data-cfemail decoding
    if not hexstr:
        return None
    try:
        b = bytes.fromhex(hexstr)
        key = b[0]
        dec = bytes([x ^ key for x in b[1:]]).decode("utf-8")
        return dec
    except Exception:
        return None

def _same_host(u1: str, u2: str) -> bool:
    try:
        a1 = urlsplit(u1); a2 = urlsplit(u2)
        return a1.scheme in {"http", "https"} and a1.netloc == a2.netloc
    except Exception:
        return False

def _abs_url(base: str, href: str) -> Optional[str]:
    if not href:
        return None
    href = href.strip()
    # skip anchors, javascript, mailto, tel
    if href.startswith("#") or href.startswith("javascript:") or href.startswith("mailto:") or href.startswith("tel:"):
        return None
    try:
        return urljoin(base, href)
    except Exception:
        return None

def _fetch_robots_txt(
    base_url: str,
    timeout: float = DEFAULT_TIMEOUT,
    http_client: Optional[CachingHttpClient] = None,
) -> tuple[list[str], list[str]]:
    """
    Returnér (sitemaps, disallows) fra robots.txt.
    Kun simpel parsing: 'Sitemap:' og 'Disallow:' linjer.
    """
    try:
        # Validér og normaliser base_url
        if not base_url or not isinstance(base_url, str):
            return [], []
        
        base_url = base_url.strip()
        if base_url.lower() in ("nan", "none", "null", ""):
            return [], []
        
        # Tilføj protokol hvis manglende
        if not base_url.startswith(('http://', 'https://')):
            if re.match(r'^(www\.)?([a-z0-9\-]+\.)+[a-z]{2,}', base_url, re.I):
                base_url = f"https://{base_url}"
            else:
                log.debug(f"Invalid base_url for robots.txt: {base_url}")
                return [], []
        
        host = _host_of(base_url)
        if not host:
            return [], []
            
        # P1: genbrug _ROBOTS_CACHE pr. host
        if host in _ROBOTS_CACHE:
            status, text = _ROBOTS_CACHE[host]
        else:
            sp = urlsplit(base_url)
            robots_url = f"{sp.scheme}://{sp.netloc}/robots.txt"
            client = http_client or CachingHttpClient(timeout=timeout)
            status, text = client.get(robots_url)
            _ROBOTS_CACHE[host] = (status, text or "")
        
        # Ingen/ubrugelig robots → returnér tomt og cache, så vi ikke hamrer igen
        if status == 0 or not text or status in (301, 302, 401, 403):
            _ROBOTS_CACHE[host] = (status, text or "")
            if not text and status == 200:
                log.debug(f"Empty text in robots.txt for {base_url} despite status 200 - likely decompression issue, treating as allow all")
            return [], []
        
        sitemaps, disallows = [], []
        
        crawl_delay = None  # TILFØJELSE: Parse crawl-delay (etik: Respektér for at undgå blokering)
        for line in text.splitlines():
            l = line.strip()
            if not l or l.startswith("#"):
                continue
            if l.lower().startswith("sitemap:"):
                sitemaps.append(l.split(":", 1)[1].strip())
            elif l.lower().startswith("disallow:"):
                dis = l.split(":", 1)[1].strip()
                if dis and dis != "/":
                    disallows.append(dis)
            elif l.lower().startswith("crawl-delay:"):  # TILFØJELSE
                try:
                    crawl_delay = float(l.split(":", 1)[1].strip())
                except ValueError:
                    log.warning(f"Invalid crawl-delay: {l}")
                    pass

        # TILFØJELSE: Gem crawl-delay globalt (bruges af CachingHttpClient)
        if crawl_delay:
            _HOST_CRAWL_DELAYS[host] = crawl_delay

        return sitemaps, disallows
    
        # Unit Test Eksempel (nu wrapped og med fixet indentation):
        if __name__ == "__main__":
            def test_robots_parse():
                text = "Crawl-delay: 5.0\nDisallow: /private"
                # Simuler parse (kopiér funktion)
                crawl_delay = None
                disallows = []
                sitemaps = []  # Tilføjet for fuldstændighed
                for line in text.splitlines():
                    l = line.strip()
                    if not l or l.startswith("#"):
                        continue
                    if l.lower().startswith("sitemap:"):
                        sitemaps.append(l.split(":", 1)[1].strip())
                    elif l.lower().startswith("disallow:"):
                        dis = l.split(":", 1)[1].strip()
                        if dis and dis != "/":
                            disallows.append(dis)
                    elif l.lower().startswith("crawl-delay:"):
                        try:
                            crawl_delay = float(l.split(":", 1)[1].strip())
                        except ValueError:
                            pass
                assert crawl_delay == 5.0
                assert "/private" in disallows
            test_robots_parse()
    except Exception as e:
        log.debug(f"Error fetching robots.txt for {base_url}: {e}")
        return [], []

def _is_disallowed(path: str, disallows: list[str]) -> bool:
    """Simpel robots Disallow: prefix-match på sti."""
    if not disallows:
        return False
    p = path or ""
    for d in disallows:
        if p.startswith(d):
            return True
    return False

@lru_cache(maxsize=512)
def _fetch_sitemap_urls(
    sitemap_url: str,
    timeout: float = DEFAULT_TIMEOUT,
    http_client: Optional[CachingHttpClient] = None,
) -> list[str]:
    """Hent sitemap (xml eller .gz) og returnér liste af <loc>-URLs."""
    urls: list[str] = []
    try:
        import httpx, io, gzip as _gz
        client = http_client or CachingHttpClient(timeout=timeout)
        status, text = client.get(sitemap_url)
        if status != 200:
            return []
        data = text.encode("utf-8") if text else b""
        if sitemap_url.lower().endswith(".gz"):
            with _gz.GzipFile(fileobj=io.BytesIO(data)) as gz:
                data = gz.read()
        text = data.decode("utf-8", errors="ignore")
        for m in re.finditer(r"<loc>\s*([^<\s]+)\s*</loc>", text, flags=re.I):
            urls.append(m.group(1).strip())
    except Exception:
        return urls
    return urls

def _discover_internal_links(base_url: str, html: str, max_links: int = 20) -> list[str]:
    """
    Scan <a href> på forsiden og returnér interne kandidatsider (kontakt/om/team/…)
    sorteret efter relevans.
    """
    out: list[tuple[int, str]] = []
    seen: set[str] = set()

    tree = _parse_html(html)

    def consider(href: str, anchor_text: str):
        absu = _abs_url(base_url, href)
        if not absu or not _same_host(base_url, absu):
            return
        # score på baggrund af keywords i path og ankertekst
        path = urlsplit(absu).path.lower()
        text = (anchor_text or "").lower()
        score = 0
        for kw in DISCOVERY_KEYWORDS:
            if kw in path:
                score += 3
            if kw in text:
                score += 2
        # favoriser korte “kontakt/om” paths
        if path.rstrip("/").endswith(("/kontakt", "/contact", "/om", "/about")):
            score += 2
        if score <= 0:
            return
        if absu not in seen:
            seen.add(absu)
            out.append((score, absu))

    # selectolax
    if HTMLParser is not None and isinstance(tree, HTMLParser):
        for a in tree.css("a[href]"):
            consider(a.attributes.get("href", ""), (a.text() or ""))
    # bs4
    elif BeautifulSoup is not None and hasattr(tree, "select"):
        for a in tree.select("a[href]"):
            consider(a.get("href", ""), a.get_text(" ", strip=True))
    # fallback regex
    else:
        for m in re.finditer(r'<a\s+[^>]*href=["\']([^"\']+)["\'][^>]*>(.*?)</a>', html, flags=re.I|re.S):
            href = m.group(1)
            text = re.sub(r"<[^>]+>", " ", m.group(2) or "")
            consider(href, text)

    # sortér efter score (desc) og længde (asc)
    out.sort(key=lambda t: (-t[0], len(urlsplit(t[1]).path)))
    return [u for _, u in out[:max_links]]

def _apex(host: str) -> str:
    if not host:
        return ""
    parts = host.lower().split(".")
    return ".".join(parts[-2:]) if len(parts) >= 2 else host.lower()

def _same_apex(u1: str, u2: str) -> bool:
    try:
        h1 = (urlsplit(u1).hostname or "").lower()
        h2 = (urlsplit(u2).hostname or "").lower()
        return _apex(h1) == _apex(h2)
    except Exception:
        return True

def _email_domain_matches_site(email: str, site_url: str) -> bool:
    try:
        ed = email.split("@", 1)[1].lower()
        host = (urlsplit(site_url).hostname or "").lower()
        if host.startswith("www."):
            host = host[4:]
        return _apex(ed) == _apex(host)
    except Exception:
        return False

# P5: Domain-mismatch whitelist (kan udvides fra andre moduler)
DOMAIN_MISMATCH_WHITELIST: set[str] = set()

# [PATCH] Mulighed for at angive whitelist via env 'VEXTO_DOMAIN_MISMATCH_ALLOW' (komma-separerede apexdomæner)
# [PATCH] Styring af ccTLD-redirects via env 'VEXTO_ALLOW_CROSS_CC_REDIRECTS' (default: OFF)
import os  # lokal import er bevidst for at undgå rækkefølgeproblemer
try:
    _env_whitelist = os.getenv("VEXTO_DOMAIN_MISMATCH_ALLOW", "")
    if _env_whitelist:
        DOMAIN_MISMATCH_WHITELIST |= {
            ap.strip().lower()
            for ap in _env_whitelist.split(",")
            if ap.strip()
        }
except Exception:
    pass

try:
    _ALLOW_CROSS_CC_REDIRECTS = os.getenv("VEXTO_ALLOW_CROSS_CC_REDIRECTS", "").strip().lower() in {"1","true","yes","y"}
except Exception:
    _ALLOW_CROSS_CC_REDIRECTS = False

def _host_in_whitelist(host: str) -> bool:
    """Tjekker om 'host' svarer til en whitelisted apex (direkte eller som subdomæne)."""
    h = (host or "").lower()
    for ap in DOMAIN_MISMATCH_WHITELIST:
        if h == ap or h.endswith("." + ap):
            return True
    return False

def _is_whitelisted_mismatch(target_url: str) -> bool:
    """Returner True hvis target_url's apex er tilladt i whitelist."""
    try:
        ap = _apex((urlsplit(target_url).hostname or "").lower())
        return ap in DOMAIN_MISMATCH_WHITELIST
    except Exception:
        return False
# === ANKER: DOMAIN_MISMATCH_WHITELIST ===

# --- Nye hjælpere til ccTLD-regler og env-toggle ---
def _hostname(u: str) -> str:
    try:
        return (urlsplit(u).hostname or "").lower()
    except Exception:
        return ""

_CC_TLD_RE = re.compile(r"\.([a-z]{2})$")

def _cc_tld(host: str) -> str:
    m = _CC_TLD_RE.search(host or "")
    return m.group(1) if m else ""

def _same_country_tld(host_a: str, host_b: str) -> bool:
    """True hvis begge værter deler samme to-bogstavs ccTLD (fx .dk == .dk)."""
    a = _cc_tld(host_a)
    b = _cc_tld(host_b)
    return bool(a and a == b)

def _allow_cross_cc_redirects() -> bool:
    """Læs env-toggle til at tillade cross-cc omdirigeringer."""
    v = os.environ.get("VEXTO_ALLOW_CROSS_CC_REDIRECTS", "").strip().lower()
    return v in ("1", "true", "yes", "y")

def _passes_identity_gate(
    name: str | None,
    emails: list[str] | None,
    title: str | None,
    url: str | None = None,
    dom_distance: Optional[int] = None,
    *,
    source: str | None = None,
    phones: list[str] | None = None,
) -> bool:
    """
    Hybrid v3:
    1) Baseline: plausibelt navn + (email eller rolle-lignende titel)
    2) Staff-grid: plausibelt navn + telefon (kortet er stærkt kontekst-signal)
    3) Kontakt/om/team + tæt DOM (≤1): plausibelt navn + same-domain, ikke-generisk email
       (Konservativ – vi tillader ikke 'kun email' uden navn som generel regel.)
    """
    # 0) Kræv altid plausibelt navn som udgangspunkt
    if not _is_plausible_name(name):
        # KONTAKT-FALLBACK (meget konservativ): tillad kun hvis kontaktish + tæt DOM + stærk email
        if url and _is_contactish_url(url) and dom_distance is not None and dom_distance <= 1:
            for em in (emails or []):
                try:
                    local = em.split("@", 1)[0]
                except Exception:
                    continue
                if _email_domain_matches_site(em, url) and not _is_generic_local(local):
                    # ekstra sikkerhed: kræv at local har nok bogstaver (personlig)
                    letters = re.sub(r"[^a-zæøå]", "", local.lower())
                    if len(letters) >= 3:
                        return True
        return False

    # 1) Baseline: navn + (email eller rolle-lignende titel)
    has_email = bool(emails and any(emails))
    t = _sanitize_title(title)
    has_role_like = bool(t and _looks_like_role(t))
    if has_email or has_role_like:
        # hvis dom_distance er kendt og stor, kan du (valgfrit) kræve begge signaler:
        # if dom_distance is not None and dom_distance > 1:
        #     return has_email and has_role_like
        return True

    # 2) Staff-grid kort uden mailto, men med telefon → OK
    if source in {"dom-staff-grid", "staff-grid"} and phones and len(phones) > 0:
        return True

    # 3) Kontaktish + tæt DOM + same-domain email med personlig local → OK
    if url and _is_contactish_url(url) and dom_distance is not None and dom_distance <= 1:
        for em in (emails or []):
            try:
                local = em.split("@", 1)[0]
            except Exception:
                continue
            if _email_domain_matches_site(em, url) and not _is_generic_local(local):
                letters = re.sub(r"[^a-zæøå]", "", local.lower())
                if len(letters) >= 3:
                    return True

    return False


def _dedup_key(c: ContactCandidate) -> tuple:
    """Sammenflet-nøgle: (navn, stærk email) → (navn, fingerprint) → (navn, url, source)."""
    name = (c.name or "").strip().lower()

    # 1) stærk email (uændret)
    for em in c.emails:
        try:
            local, dom = em.split("@", 1)
        except Exception:
            continue
        if not _is_generic_local(local):
            return (name, f"{local}@{dom}")

    # 2) fingerprint fra hints (anti-blanding på tværs af kort)
    fp = (c.hints or {}).get("fingerprint") or ""
    if fp:
        return (name, f"fp:{fp}")

    # 3) fallback
    return (name, (c.url or "").strip().lower(), c.source)

def _merge(a: ContactCandidate, b: ContactCandidate) -> ContactCandidate:
    """Flet b ind i a (bevar bedste/rigeste værdier)."""
    if not a.name and b.name:
        a.name = b.name
    if not a.title and b.title:
        a.title = b.title
    a.emails = sorted({e for e in (a.emails + b.emails) if e})
    a.phones = sorted({p for p in (a.phones + b.phones) if p})
    if a.dom_distance is None or (b.dom_distance is not None and b.dom_distance < a.dom_distance):
        a.dom_distance = b.dom_distance
    a.hints.update(b.hints or {})
    return a

def _looks_404(html: str | None) -> bool:
    if not html:
        return True
    h = html.lower()
    # typiske CMS 404-markører
    needles = [
        "404", "not found", "page not found", "siden blev ikke fundet",
        "error-404", "error 404", "wp-block-404", "nothing was found", "ingenting blev fundet",
    ]
    # Kræv både '404' og mindst én “not found”-frase, for at undgå false positives
    if "404" in h and any(n in h for n in ("not found", "siden blev ikke fundet", "page not found", "error-404")):
        return True
    # meget kort HTML er ofte 404/fejl
    if len(h) < 400:
        return True
    return False


# ------------------------------- Cache & HTTP ---------------------------------

def _cache_path_for(url: str, cache_dir: Optional[Path]) -> Optional[Path]:
    if not cache_dir:
        return None
    h = hashlib.md5(url.encode("utf-8")).hexdigest()
    return cache_dir / f"{h}.html"

def _read_cache(cp: Optional[Path], max_age_hours: int) -> Optional[str]:
    if not cp or not cp.exists():
        return None
    age = datetime.now() - datetime.fromtimestamp(cp.stat().st_mtime)
    if age > timedelta(hours=max_age_hours):
        return None
    try:
        return cp.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return None

def _write_cache(cp: Optional[Path], text: str) -> None:
    if not cp:
        return
    try:
        cp.parent.mkdir(parents=True, exist_ok=True)
        cp.write_text(text, encoding="utf-8", errors="ignore")
    except Exception:
        pass

@lru_cache(maxsize=256)
def _fetch_text(url: str, timeout: float = DEFAULT_TIMEOUT, cache_dir: Optional[Path] = None, max_age_hours: int = 24) -> str:
    """Fetch text med caching og validering."""
    
    # Validering
    if not url or not isinstance(url, str):
        return ""
    
    url = url.strip()
    if url.lower() in ("nan", "none", "null", ""):
        return ""
    
    # Sørg for protokol
    if not url.startswith(('http://', 'https://')):
        if re.match(r'^(www\.)?([a-z0-9\-]+\.)+[a-z]{2,}', url, re.I):
            url = f"https://{url}"
        else:
            return ""
    
    cp = _cache_path_for(url, cache_dir)
    cached = _read_cache(cp, max_age_hours)
    if cached:
        return cached

    ae = _accept_encoding() if callable(_accept_encoding) else _accept_encoding
    if not isinstance(ae, (str, bytes)) or (isinstance(ae, str) and not ae.strip()):
        ae = "gzip, deflate, br"
    
    headers = {
        "User-Agent": "VextoContactFinder/1.0 (+https://vexto.io)",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "da-DK,da;q=0.9,en;q=0.8",
        "Accept-Encoding": ae,
    }
    
    if httpx is not None:
        try:
            with httpx.Client(http2=True, timeout=timeout, headers=headers, follow_redirects=True) as cli:
                r = cli.get(url)
                r.raise_for_status()
                text = r.text
                _write_cache(cp, text)
                return text
        except Exception as e:
            log.debug(f"Failed to fetch {url}: {e}")
            return ""
    
    if 'requests' in sys.modules:
        try:
            r = requests.get(url, headers=headers, timeout=timeout)  # type: ignore
            r.raise_for_status()
            text = r.text
            _write_cache(cp, text)
            return text
        except Exception as e:
            log.debug(f"Failed to fetch {url}: {e}")
            return ""
    
    raise RuntimeError("No HTTP client available (install httpx or requests)")

# ------------------------------- Parsing --------------------------------------

def _parse_html(html: str):
    """Parser-prioritet: selectolax → BeautifulSoup → None."""
    if HTMLParser is not None:
        return HTMLParser(html)
    if BeautifulSoup is not None:
        return BeautifulSoup(html, "html.parser")
    return None

def _iter_jsonld(html: str) -> Iterable[dict]:
    # Hurtig regex til at finde JSON-LD blocks
    for m in re.finditer(
        r'<script[^>]+type=["\']application/ld\+json["\'][^>]*>(.*?)</script>',
        html, flags=re.I | re.S,
    ):
        raw = m.group(1).strip()
        with contextlib.suppress(Exception):
            block = json.loads(raw)
            if isinstance(block, dict):
                yield block
            elif isinstance(block, list):
                for item in block:
                    if isinstance(item, dict):
                        yield item

def _near_text(node, max_chars: int = 200) -> str:
    """Saml tekst i node + nærområde (duck typing: virker for selectolax og BS4)."""
    try:
        # selectolax-lignende node
        if hasattr(node, "itertext") and hasattr(node, "parent"):
            texts = []
            texts.extend(t.strip() for t in node.itertext() if t and t.strip())
            parent = getattr(node, "parent", None)
            if parent and hasattr(parent, "iter"):
                sibs = list(parent.iter())
                for sib in sibs[:8]:
                    if sib is node:
                        continue
                    with contextlib.suppress(Exception):
                        if hasattr(sib, "itertext"):
                            txt = " ".join(t.strip() for t in sib.itertext() if t and t.strip())
                            if txt:
                                texts.append(txt)
            out = " ".join(texts)
            return (out[:max_chars] + "…") if len(out) > max_chars else out

        # bs4-lignende node
        if hasattr(node, "stripped_strings"):
            parent = getattr(node, "parent", None)
            texts = []
            if parent and hasattr(parent, "find_all"):
                for sib in parent.find_all(recursive=False)[:8]:
                    with contextlib.suppress(Exception):
                        txt = " ".join(list(sib.stripped_strings))
                        if txt:
                            texts.append(txt)
            out = " ".join(texts)
            return (out[:max_chars] + "…") if len(out) > max_chars else out
    except Exception:
        pass
    return ""

def _dom_distance(node1, node2, max_depth: int = 6) -> Optional[int]:
    """Beregn 'kant-afstand' via fælles forfader; virker for selectolax og BS4."""
    if not node1 or not node2:
        return None
    try:
        a = []
        n = node1
        for _ in range(max_depth):
            p = getattr(n, "parent", None)
            if not p:
                break
            a.append(p)
            n = p
        dist = 0
        n = node2
        for _ in range(max_depth):
            if n in a:
                return dist + a.index(n)
            p = getattr(n, "parent", None)
            if not p:
                break
            dist += 1
            n = p
    except Exception:
        return None
    return max_depth

# ------------------------------- Extractors -----------------------------------

def _extract_from_jsonld(url: str, html: str) -> list[ContactCandidate]:
    out: list[ContactCandidate] = []
    for obj in _iter_jsonld(html):
        t = obj.get("@type") or obj.get("@type".lower())
        if isinstance(t, list):
            types = [str(x).lower() for x in t]
        else:
            types = [str(t).lower()] if t else []
        # Person
        if any(x.endswith("person") for x in types):
            name = _collapse_ws(obj.get("name"))
            email = obj.get("email")
            tel = obj.get("telephone") or obj.get("phone")
            job = _collapse_ws(obj.get("jobTitle") or obj.get("jobtitle"))
            cand = ContactCandidate(
                name=name,
                title=job,
                emails=[_normalize_email(email)] if email else [],
                phones=[_normalize_phone(tel)] if tel else [],
                source="json-ld",
                url=url,
                dom_distance=0,
                hints={"type": "Person"},
            )
            out.append(cand)
        # Organization → contactPoint/personer
        if any(x.endswith("organization") for x in types):
            cps = obj.get("contactPoint") or obj.get("contactPoints")
            if isinstance(cps, dict):
                cps = [cps]
            for cp in cps or []:
                if not isinstance(cp, dict):
                    continue
                name = _collapse_ws(cp.get("name"))
                email = cp.get("email")
                tel = cp.get("telephone") or cp.get("phone")
                job = _collapse_ws(cp.get("contactType") or cp.get("jobTitle") or cp.get("role"))
                cand = ContactCandidate(
                    name=name,
                    title=job,
                    emails=[_normalize_email(email)] if email else [],
                    phones=[_normalize_phone(tel)] if tel else [],
                    source="json-ld",
                    url=url,
                    dom_distance=1,
                    hints={"type": "Organization.contactPoint"},
                )
                out.append(cand)
    return out

def _extract_from_rdfa(url: str, tree) -> list[ContactCandidate]:
    """Minimal RDFa (schema.org Person: name, jobTitle, email, telephone)."""
    out: list[ContactCandidate] = []
    if not tree:
        return out
    
    def _get_node_text(node) -> str:
        if not node: return ""
        try:
            if hasattr(node, "text"): # selectolax
                return (node.text() or "").strip()
            if hasattr(node, "get_text"): # bs4
                return (node.get_text(" ", strip=True) or "").strip()
        except Exception:
            pass
        return ""

    try:
        if hasattr(tree, "css"): # selectolax
            for el in tree.css('[typeof*="Person" i]'):
                name_node = el.css_first('[property="name"]')
                title_node = el.css_first('[property="jobTitle"]')
                email_node = el.css_first('[property="email"]')
                phone_node = el.css_first('[property="telephone"]')
                
                out.append(ContactCandidate(
                    name=_get_node_text(name_node),
                    title=_get_node_text(title_node),
                    emails=[_normalize_email(_get_node_text(email_node))] if email_node else [],
                    phones=[_normalize_phone(_get_node_text(phone_node))] if phone_node else [],
                    source="rdfa", url=url, dom_distance=0,
                    hints={"selector": '[typeof*="Person"]'}
                ))
        elif hasattr(tree, "select"): # bs4
            for el in tree.select('[typeof*="Person" i]'):
                name_node = el.select_one('[property="name"]')
                title_node = el.select_one('[property="jobTitle"]')
                email_node = el.select_one('[property="email"]')
                phone_node = el.select_one('[property="telephone"]')

                out.append(ContactCandidate(
                    name=_get_node_text(name_node),
                    title=_get_node_text(title_node),
                    emails=[_normalize_email(_get_node_text(email_node))] if email_node else [],
                    phones=[_normalize_phone(_get_node_text(phone_node))] if phone_node else [],
                    source="rdfa", url=url, dom_distance=0,
                    hints={"selector": '[typeof*="Person"]'}
                ))
    except Exception as e:
        log.debug(f"Error in _extract_from_rdfa: {e}")
        
    return out


def _NAME_WINDOW_REGEX() -> str:
    # Vindue af 2–4 kapitaliserede tokens – bruges i nærtekst
    return rf"{NAME_TOKEN}(?:\s+{NAME_TOKEN}){{1,3}}"

def _maybe_name_from_text(txt: str | None) -> Optional[str]:
    if not txt:
        return None
    # find 2–4 kapitaliserede tokens (tæt på dit eksisterende NAME_TOKEN)
    m = re.search(rf"\b({NAME_TOKEN}(?:\s+{NAME_TOKEN}){{1,3}})\b", txt)
    if not m:
        return None
    cand = _collapse_ws(m.group(1))
    return cand if _is_plausible_name(cand) else None

def _find_heading_name(node) -> Optional[str]:
    """
    Gå op til 5 forældre og kig efter nærliggende navn i <h1-6>, <strong>, <b>,
    elementer med [itemprop=name], [role=heading], aria-label samt søskende-tekst.
    Virker for både selectolax og BS4 (duck typing).
    """
    try:
        # selectolax-lignende node
        if hasattr(node, "parent") and hasattr(node, "iter"):
            cur = node
            for _ in range(5):
                parent = getattr(cur, "parent", None)
                if not parent:
                    break
                # direkte headings/strong + semantiske navneindikatorer
                if hasattr(parent, "css"):
                    for sel in ["h1","h2","h3","h4","h5","h6","strong","b","[itemprop='name']","[role='heading']"]:
                        for h in parent.css(sel):
                            nm = _maybe_name_from_text(h.text())
                            if nm:
                                return nm
                    # aria-label
                    try:
                        al = parent.attributes.get("aria-label")
                        nm = _maybe_name_from_text(al)
                        if nm:
                            return nm
                    except Exception:
                        pass
                    # søskende
                    sibs = list(parent.iter())
                    for sib in sibs[:8]:
                        with contextlib.suppress(Exception):
                            if hasattr(sib, "itertext"):
                                nm = _maybe_name_from_text(" ".join(t.strip() for t in sib.itertext()))
                                if nm:
                                    return nm
                cur = parent

        # bs4-lignende node
        if hasattr(node, "find_parent"):
            cur = node
            for _ in range(5):
                parent = getattr(cur, "parent", None)
                if not parent:
                    break
                if hasattr(parent, "find_all"):
                    for tag in parent.find_all(["h1","h2","h3","h4","h5","h6","strong","b"], limit=8, recursive=True):
                        nm = _maybe_name_from_text(tag.get_text(" ", strip=True))
                        if nm:
                            return nm
                    # semantiske
                    for tag in parent.find_all(attrs={"itemprop": "name"}, limit=4):
                        nm = _maybe_name_from_text(tag.get_text(" ", strip=True))
                        if nm:
                            return nm
                    for tag in parent.find_all(attrs={"role": "heading"}, limit=4):
                        nm = _maybe_name_from_text(tag.get_text(" ", strip=True))
                        if nm:
                            return nm
                    al = parent.attrs.get("aria-label")
                    nm = _maybe_name_from_text(al)
                    if nm:
                        return nm
                    # søskende
                    for sib in parent.find_all(recursive=False):
                        txt = sib.get_text(" ", strip=True)
                        nm = _maybe_name_from_text(txt)
                        if nm:
                            return nm
                cur = parent
    except Exception:
        return None
    return None


def _harvest_identity_from_container(node, max_depth: int = 8) -> tuple[Optional[str], Optional[str], list[str]]:
    """
    Find nærmeste 'person-kort' omkring node og ekstrahér (name, title, phones).

    Design:
    - Trin A: Gå opad og vælg den mindste fornuftige container (employee/person/team/contact mm.)
      uden at "hæfte" på mega-sektioner (row/section/container/elementor-section).
    - Trin B: Find NAME fra rene headings (h1–h6) og udbredte name-klasser; skip hvis teksten ligner en titel.
              Brug case-promotion hvis overskriften er HELT UPPERCASE.
    - Trin C: Find TITLE: brug facit via _canonicalize_title_or_none først; ellers role-heuristik.
    - Trin D: Saml PHONES fra tel: + fri tekst (ignorér CVR/reg-kontekst).
    """

    if not node:
        return None, None, []

    # ---------- helpers ----------
    def _cls_id(n) -> str:
        # class + id som én lavere-case streng
        try:
            if hasattr(n, "attributes"):  # selectolax
                return ((n.attributes.get("class", "") or "") + " " + (n.attributes.get("id", "") or "")).lower()
            elif hasattr(n, "get"):       # bs4
                cls = n.get("class") or []
                cls = " ".join(cls) if isinstance(cls, (list, tuple)) else str(cls or "")
                return (cls + " " + (n.get("id") or "")).lower()
        except Exception:
            pass
        return ""

    TIGHT_KEYS = (
        "employee", "employee-card", "employee_item", "employee__item",
        "staff", "staff-member", "staff__item",
        "person", "person-card", "person__item",
        "member", "team-member", "team__member", "team-card", "team__card",
        "contact", "kontakt", "contact-card", "contact-person", "contact__item",
        "medarbejder", "profile", "card", "tile", "item", "box", "column-item", "grid-item"
    )
    # grove layout-sektioner vi helst ikke ender på
    COARSE_KEYS = (
        "section", "container", "row", "grid", "columns", "column", "wrapper",
        "elementor-section", "elementor-container", "elementor-row", "elementor-column",
        "wp-block", "blocks"
    )

    def _looks_coarse(n) -> bool:
        s = _cls_id(n)
        return any(k in s for k in COARSE_KEYS)

    def _looks_tight(n) -> bool:
        s = _cls_id(n)
        return any(k in s for k in TIGHT_KEYS)

    def _iter_css(n, sel: str):
        if hasattr(n, "css"):
            return n.css(sel)  # selectolax
        if hasattr(n, "select"):
            return n.select(sel)  # bs4
        return []

    def _el_text(e) -> str:
        try:
            if hasattr(e, "text"):  # selectolax node
                return _collapse_ws(e.text())
        except Exception:
            pass
        try:
            if hasattr(e, "get_text"):  # bs4 tag
                return _collapse_ws(e.get_text(" ", strip=True))
        except Exception:
            pass
        return ""

    def _get_attr(e, key: str) -> str:
        try:
            if hasattr(e, "attributes"):  # selectolax
                return e.attributes.get(key, "") or ""
        except Exception:
            pass
        try:
            if hasattr(e, "get"):  # bs4
                return e.get(key, "") or ""
        except Exception:
            pass
        return ""

    def _node_text(n) -> str:
        try:  # selectolax
            if hasattr(n, "itertext"):
                return " ".join(t.strip() for t in n.itertext() if t and t.strip())
        except Exception:
            pass
        try:  # bs4
            if hasattr(n, "stripped_strings"):
                return " ".join(list(n.stripped_strings))
        except Exception:
            pass
        return ""

    # En lille UI/sektion-blacklist for *navne* (typiske støjoverskrifter)
    UI_NAME_BLACK = {
        "kontakt", "kontakt os", "kontakt os på", "kundeservice", "kundeservice / salg",
        "små og store projekter", "projekter", "storkunder", "trafikcenter", "slagelse",
        "om", "om os", "about", "book", "læs mere", "find medarbejder"
    }

    def _ui_noise_name(s: str) -> bool:
        low = s.lower()
        if low in UI_NAME_BLACK:
            return True
        # også hvis frasen indeholder disse substrings
        for p in ("trafikcenter allé", "dk-4200", "kontakt os"):
            if p in low:
                return True
        return False

    # rolige heuristikker hvis facit ikke siger ja/nej
    ROLE_HINTS = (
        "chef", "leder", "manager", "director", "konsulent", "rådgiver",
        "account", "aftersales", "salg", "marketing", "logistik", "drift",
        "projekter", "kundeservice", "økonomi", "controller", "specialist",
        "koordinator", "assistent", "indkøb", "support", "tekniker"
    )

    def _looks_role_like(txt: str) -> bool:
        t = (_sanitize_title(txt) or "").lower()
        if not t:
            return False
        # facit?
        try:
            facit = _canonicalize_title_or_none(t)
            if facit:
                return True
        except Exception:
            pass
        # ellers simple hints + form
        if any(h in t for h in ROLE_HINTS):
            return True
        if "/" in t or " - " in t:
            return True
        # undgå adresser
        if any(ch.isdigit() for ch in t):
            return False
        return False

    # ---------- A) find nærmeste "tætte" container ----------
    container = node
    best = None
    depth = 0
    while depth < max_depth and getattr(container, "parent", None):
        if _looks_tight(container):
            best = container
            break
        parent = getattr(container, "parent", None)
        if not parent:
            break
        # gem "mindste ikke-grove" som fallback
        if best is None and not _looks_coarse(parent):
            best = parent
        container = parent
        depth += 1

    if best is None:
        best = getattr(node, "parent", None) or node
    container = best

    # ---------- B) NAME: headings først (h1–h6), skip værdier der ligner titler ----------
    name: Optional[str] = None

    heading_sel = "h1,h2,h3,h4,h5,h6," \
                  ".name,.person-name,.staff-name,.member-name,.employee-name," \
                  "[itemprop='name'],[role='heading']," \
                  ".elementor-heading-title,.elementor-widget-heading"

    try:
        for el in _iter_css(container, heading_sel):
            raw = _el_text(el)
            if not raw:
                continue
            # tydelig UI-støj? → skip
            if _ui_noise_name(raw):
                continue
            # hvis teksten ligner en titel, ikke et navn → skip
            try:
                st = _sanitize_title(raw) or ""
                if st and (_canonicalize_title_or_none(st) or _looks_role_like(st)):
                    continue
            except Exception:
                # hvis facit ikke er tilgængelig, brug kun heuristik:
                if _looks_role_like(raw):
                    continue

            # plausibelt navn?
            if _is_plausible_name(raw):
                name = raw
                break

            # case-promotion hvis HELT UPPERCASE
            promo = _maybe_promote_name_case(raw)
            if not name and promo and _is_plausible_name(promo):
                name = promo
                break

            if not name and raw.isupper():
                cap = " ".join(w[:1].upper() + w[1:].lower() for w in raw.split())
                if _is_plausible_name(cap):
                    name = cap
                    break
    except Exception:
        pass

    # [NYT] Hvis vi stadig ikke har et navn: prøv img[alt] og semibold/strong
    if not name:
        try:
            img = None
            if hasattr(container, "css"):
                img = container.css_first("img[alt]")
            elif hasattr(container, "select"):
                img = container.select_one("img[alt]")
            if img:
                alt = _get_attr(img, "alt").strip()
                if alt and _is_plausible_name(alt):
                    name = alt
        except Exception:
            pass

    if not name:
        try:
            node = None
            if hasattr(container, "css"):
                node = container.css_first("p.font-semibold, .font-semibold, strong, b")
            elif hasattr(container, "select"):
                node = container.select_one("p.font-semibold, .font-semibold, strong, b")
            txt = _el_text(node) if node else ""
            if txt and _is_plausible_name(txt):
                name = txt
        except Exception:
            pass

    # ---------- C) TITLE: facit først; ellers “role-like” ----------
    title: Optional[str] = None

    title_sel = (
        "[itemprop='jobTitle'],"
        ".job-title,.title,.role,.position,.position-title,"
        ".team-member__role,.bio-title,"
        ".employee-title,[class*='title']"
    )

    try:
        # stærke titel-felter
        for el in _iter_css(container, title_sel):
            tt_raw = _el_text(el)
            if not tt_raw:
                continue
            # undgå at tage navne som titel
            if _is_plausible_name(tt_raw):
                continue
            tt = _sanitize_title(tt_raw) or ""
            if not tt:
                continue
            facit = None
            try:
                facit = _canonicalize_title_or_none(tt)
            except Exception:
                facit = None
            if facit:
                title = facit[0]
                break
            # fallback: acceptér "role-like"
            if _looks_role_like(tt):
                title = tt
                break

        # svag fallback: scann p/span-linjer i kortet
        if not title:
            for el in _iter_css(container, "p,span,div"):
                txt = _el_text(el)
                if not txt:
                    continue
                # hvis vi mangler name og linjen ligner et navn, så tag navnet men lad titlen være tom endnu
                if not name and _is_plausible_name(txt):
                    name = txt
                    continue
                tt = _sanitize_title(txt) or ""
                if not tt:
                    continue
                try:
                    facit = _canonicalize_title_or_none(tt)
                except Exception:
                    facit = None
                if facit:
                    title = facit[0]
                    break
                if _looks_role_like(tt):
                    title = tt
                    break
    except Exception:
        pass

    # ---------- D) PHONES: tel: + fri tekst (anti-CVR) ----------
    phones: list[str] = []
    try:
        for t in _iter_css(container, "a[href^='tel:']"):
            href = _get_attr(t, "href")
            pn = _normalize_phone(href)
            if pn:
                phones.append(pn)
    except Exception:
        pass

    blob = _node_text(container)
    for m in re.finditer(r"(?:\+?\d[\s\-\(\)\.]{0,3}){8,}", blob):
        sidx, eidx = m.start(), m.end()
        ctx = blob[max(0, sidx - 24):min(len(blob), eidx + 24)].lower()
        if "cvr" in ctx or "reg" in ctx:
            continue
        pn = _normalize_phone(m.group(0))
        if pn:
            phones.append(pn)

    phones = sorted(set(p for p in phones if p))

    return name, title, phones


def _extract_from_mailtos(url: str, html_or_tree) -> list[ContactCandidate]:
    out: list[ContactCandidate] = []

    # ---------------- selectolax ----------------
    if HTMLParser is not None and isinstance(html_or_tree, HTMLParser):
        # mailto:
        for a in html_or_tree.css("a[href^='mailto:']"):
            href = a.attributes.get("href", "")
            em = _normalize_email(href)
            if not em:
                continue

            # NYT: Høst navn/titel/phones fra nærmeste container
            name_h, title_h, phones_h = _harvest_identity_from_container(a)

            near = _near_text(a, 260)
            # fallback hvis harvest ikke gav navn/titel
            if not _is_plausible_name(name_h):
                m_name = re.search(rf"({_NAME_WINDOW_REGEX()})", near)
                if m_name:
                    name_h = _collapse_ws(m_name.group(1))
            if not _sanitize_title(title_h):
                m_title = re.search(r"(?i)\b([A-Za-zÆØÅæøå/\- ]{3,80})\b", near)
                if m_title:
                    title_h = _sanitize_title(m_title.group(1))

            try:
                dist = _dom_distance(a, a.parent or a) or (0 if name_h else 1)
            except Exception:
                dist = 0 if name_h else 1

            # Gate-bypass for generiske same-domain mails uden navn → org-kontakt
            hints = {"near": near[:180], "harvested": True}
            try:
                local = em.split("@", 1)[0]
            except Exception:
                local = ""
            if url and _email_domain_matches_site(em, url) and _is_generic_local(local) and not _is_plausible_name(name_h):
                hints["identity_gate_bypassed"] = True
                hints["org_footer"] = True

            out.append(ContactCandidate(
                name=name_h, title=title_h, emails=[em], phones=phones_h,
                source="mailto", url=url, dom_distance=dist, hints=hints
            ))


        # Cloudflare-obfuskerede e-mails
        for cf in html_or_tree.css(".__cf_email__"):
            em = _normalize_email(_cf_decode(cf.attributes.get("data-cfemail")))
            if not em:
                continue
            text_blob = _near_text(container, 260)
            fp = _fp_from_text(text_blob, prefix="near:")

            name = None
            m = re.search(rf"({_NAME_WINDOW_REGEX()})", text_blob)
            if m:
                name = _collapse_ws(m.group(1))
            if not _is_plausible_name(name):
                name = _find_heading_name(cf) or name

            title = None
            m2 = re.search(r"(?i)\b([A-Za-zÆØÅæøå/\- ]{3,80})\b", text_blob)
            if m2:
                title = _sanitize_title(m2.group(1))

            dist = _dom_distance(cf, container) or (0 if name else 1)
            out.append(ContactCandidate(
                name=name, title=title, emails=[em], phones=[],
                source="dom", url=url, dom_distance=dist,
                hints={"near": text_blob[:180], "fingerprint": fp}
            ))
        return out

    # ---------------- BeautifulSoup ----------------
    if BeautifulSoup is not None and hasattr(html_or_tree, "select"):

        def _bs4_card_fingerprint(node) -> str:
            """Find nærmeste staff-card ancestor og lav stabilt fingerprint (data-v + img[alt])."""
            try:
                target_classes = {
                    "employee", "employee-card", "staff-member", "contact-person",
                    "kontaktperson", "team-member", "person-card", "person",
                    "staff", "staff-card"
                }
                p = node
                card = None
                for _ in range(8):  # gå et par niveauer op
                    if not getattr(p, "parent", None):
                        break
                    classes = set(p.get("class", [])) if hasattr(p, "get") else set()
                    if classes & target_classes:
                        card = p
                        break
                    p = p.parent

                data_v = (card.get("data-v", "") or "").strip() if (card and hasattr(card, "get")) else ""
                img = card.select_one("img[alt]") if (card and hasattr(card, "select_one")) else None
                alt = (img.get("alt", "") or "").strip() if img else ""
                if data_v or alt:
                    return _fp_from_text(f"{data_v}::{alt}", prefix="card:")
                return ""
            except Exception:
                return ""

        for a in html_or_tree.select("a[href^='mailto:']"):
            href = a.get("href", "")
            em = _normalize_email(href)
            if not em:
                continue

            name_h, title_h, phones_h = _harvest_identity_from_container(a)

            # near-fingerprint som fallback
            near = " ".join(list(a.parent.stripped_strings))[:260] if a.parent else ""
            fp_near = _fp_from_text(near, prefix="near:")

            # kort-fingerprint har prioritet, ellers brug near
            fp_card = _bs4_card_fingerprint(a)
            fp = fp_card or fp_near

            if not _is_plausible_name(name_h):
                m_name = re.search(rf"({_NAME_WINDOW_REGEX()})", near)
                if m_name:
                    nm = _collapse_ws(m_name.group(1))
                    promo = _maybe_promote_name_case(nm) or nm
                    if _is_plausible_name(promo):
                        name_h = promo

            if not _sanitize_title(title_h):
                m_title = re.search(r"(?i)\b([A-Za-zÆØÅæøå/\- ]{3,80})\b", near)
                if m_title:
                    title_h = _sanitize_title(m_title.group(1))

            try:
                dist = _dom_distance(a, a.parent or a) or (0 if name_h else 1)
            except Exception:
                dist = 0 if name_h else 1

            hints = {"near": near[:180], "harvested": True, "fingerprint": fp}
            try:
                local = em.split("@", 1)[0]
            except Exception:
                local = ""
            if url and _email_domain_matches_site(em, url) and _is_generic_local(local) and not _is_plausible_name(name_h):
                hints["identity_gate_bypassed"] = True
                hints["org_footer"] = True

            out.append(ContactCandidate(
                name=name_h,
                title=title_h,
                emails=[em],
                phones=phones_h or [],   # sikre liste
                source="mailto",
                url=url,
                dom_distance=dist,
                hints=hints
            ))


        # ---------------- Cloudflare __cf_email__ (BS4) ----------------
        for cf in html_or_tree.select(".__cf_email__"):
            em = _normalize_email(_cf_decode(cf.get("data-cfemail")))
            if not em:
                continue

            container = cf.parent or cf
            near = " ".join(list(container.stripped_strings))[:260] if container else ""
            fp_near = _fp_from_text(near, prefix="near:")
            fp_card = _bs4_card_fingerprint(container)
            fp = fp_card or fp_near

            name = None
            m_name = re.search(rf"({_NAME_WINDOW_REGEX()})", near)
            if m_name:
                name = _collapse_ws(m_name.group(1))
            if not _is_plausible_name(name):
                name = _find_heading_name(cf) or name

            title = None
            m_title = re.search(r"(?i)\b([A-Za-zÆØÅæøå\- ]{3,80})\b", near)
            if m_title:
                title = _sanitize_title(m_title.group(1))

            out.append(ContactCandidate(
                name=name,
                title=title,
                emails=[em],
                phones=[],
                source="dom",
                url=url,
                dom_distance=0,
                hints={"near": near[:180], "cfemail": True, "fingerprint": fp}
            ))

        # ---------------- tel: links (BS4) ----------------
        for t in html_or_tree.select("a[href^='tel:']"):
            href = t.get("href", "")
            pn = _normalize_phone(href)
            if not pn:
                continue

            container = t.parent or t
            near = " ".join(list(container.stripped_strings))[:260] if container else ""
            fp_near = _fp_from_text(near, prefix="near:")
            fp_card = _bs4_card_fingerprint(container)
            fp = fp_card or fp_near

            name = None
            m_name = re.search(rf"({_NAME_WINDOW_REGEX()})", near)
            if m_name:
                name = _collapse_ws(m_name.group(1))
            if not _is_plausible_name(name):
                name = _find_heading_name(t) or name

            title = None
            m_title = re.search(r"(?i)\b([A-Za-zÆØÅæøå/\- ]{3,80})\b", near)
            if m_title:
                title = _sanitize_title(m_title.group(1))

            out.append(ContactCandidate(
                name=name,
                title=title,
                emails=[],
                phones=[pn],
                source="dom",
                url=url,
                dom_distance=0,
                hints={"near": near[:180], "tel": True, "fingerprint": fp}
            ))
        return out

    # ---------------- ren regex fallback ----------------
    for m in re.finditer(r'href=["\']mailto:([^"\']+)["\']', str(html_or_tree), flags=re.I):
        em = _normalize_email(m.group(1))
        if em:
            out.append(ContactCandidate(
                name=None, title=None, emails=[em], phones=[],
                source="mailto", url=url, dom_distance=None, hints={}
            ))
    return out

def _extract_from_dom(url: str, html_or_tree) -> list[ContactCandidate]:
    out: list[ContactCandidate] = []
    if HTMLParser is None or not isinstance(html_or_tree, HTMLParser):
        return out

    # Lokal helper: promover helt-lowercase navne til "Title Case"
    def _promote_lower_name(s: Optional[str]) -> Optional[str]:
        if not s:
            return None
        t = _collapse_ws(s) or ""
        if not t:
            return None
        # 2–4 ord, kun bogstaver (DK inkl.), min. længde 2
        if re.fullmatch(r"[a-zæøå]{2,}(?:\s+[a-zæøå]{2,}){1,3}", t, flags=re.I):
            parts = t.split()
            return " ".join(p[:1].upper() + p[1:] for p in parts)
        return None

    # --- Microdata Person ---
    for el in html_or_tree.css('[itemtype*="Person" i], [itemscope][itemtype*="Person" i]'):
        name = None; title = None; email = None; tel = None
        for sub in el.css('[itemprop]'):
            prop = (sub.attributes.get("itemprop") or "").lower()
            txt = _collapse_ws(sub.text())
            if not prop:
                continue
            if prop == "name" and txt:
                name = txt
            elif prop == "jobtitle" and txt:
                title = txt
            elif prop == "email":
                email = _normalize_email(sub.text() or "")
            elif prop in {"telephone","phone"}:
                tel = _normalize_phone(sub.text() or "")

        # promotér evt. helt-lowercase navn
        if name and not _is_plausible_name(name):
            promo = _promote_lower_name(name)
            if promo and _is_plausible_name(promo):
                name = promo

        out.append(ContactCandidate(
            name=name,
            title=_sanitize_title(title),
            emails=[e for e in [email] if e],
            phones=[p for p in [tel] if p],
            source="microdata",
            url=url,
            dom_distance=0,
            hints={"selector": "Person microdata"},
        ))

    # --- Cards: anchor mailto med nær-navn/titel ---
    for a in html_or_tree.css("a[href^='mailto:']"):
        em = _normalize_email(a.attributes.get("href", ""))
        if not em:
            continue
        container = a.parent or a
        text_blob = _near_text(container, 260)

        name_h, title_h, phones_h = _harvest_identity_from_container(a)

        # Fallback: nærtekst → navn (med case-promotion)
        if not _is_plausible_name(name_h):
            m = re.search(rf"({_NAME_WINDOW_REGEX()})", text_blob)
            if m:
                nm = _collapse_ws(m.group(1))
                promo = _promote_lower_name(nm) or nm
                if _is_plausible_name(promo):
                    name_h = promo

        # Fallback: nærtekst → titel
        if not _sanitize_title(title_h):
            title_h = _title_from_text_window(text_blob, name_h)

        dist = _dom_distance(a, container) or (0 if name_h else 1)
        out.append(ContactCandidate(
            name=name_h,
            title=title_h,
            emails=[em],
            phones=phones_h,
            source="mailto",
            url=url,
            dom_distance=dist,
            hints={"near": text_blob[:180], "harvested": True},
        ))

    # --- tel: ---
    for t in html_or_tree.css("a[href^='tel:']"):
        pn = _normalize_phone(t.attributes.get("href", ""))
        if not pn:
            continue
        container = t.parent or t
        text_blob = _near_text(container, 260)

        name_h, title_h, phones_h = _harvest_identity_from_container(t)
        if pn and pn not in phones_h:
            phones_h.append(pn)

        # Fallback: nærtekst → navn (med case-promotion)
        if not _is_plausible_name(name_h):
            m = re.search(rf"({_NAME_WINDOW_REGEX()})", text_blob)
            if m:
                nm = _collapse_ws(m.group(1))
                promo = _promote_lower_name(nm) or nm
                if _is_plausible_name(promo):
                    name_h = promo

        # Fallback: nærtekst → titel
        if not _sanitize_title(title_h):
            m2 = re.search(r"(?i)\b([A-Za-zÆØÅæøå/\- ]{3,80})\b", text_blob)
            if m2:
                title_h = _sanitize_title(m2.group(1))

        dist = _dom_distance(t, container) or (0 if name_h else 1)
        out.append(ContactCandidate(
            name=name_h,
            title=title_h,
            emails=[],
            phones=sorted(set(phones_h)),
            source="dom",
            url=url,
            dom_distance=dist,
            hints={"near": text_blob[:180], "harvested": True, "tel": True},
        ))

    return out

def _extract_from_text_emails(url: str, html: str) -> list[ContactCandidate]:
    """
    Fald tilbage: find e-mails i ren tekst (inkl. 'navn [at] domæne.dk')
    og forsøg at parre med navn/rolle i nærheden.
    """
    out: list[ContactCandidate] = []

    # DOM-forsøg: hvis vi kan finde mailto med samme email, høst containeren
    tree = _parse_html(html)
    if tree is not None:
        # find alle mailto og map til node
        mailto_map = {}
        try:
            if hasattr(tree, "css"):
                for a in tree.css("a[href^='mailto:']"):
                    em = _normalize_email(a.attributes.get("href",""))
                    if em and em not in mailto_map:
                        mailto_map[em.lower()] = a
            elif hasattr(tree, "select"):
                for a in tree.select("a[href^='mailto:']"):
                    em = _normalize_email(a.get("href",""))
                    if em and em not in mailto_map:
                        mailto_map[em.lower()] = a
        except Exception:
            pass

    # Flad tekst uden tags
    plain = re.sub(r"<[^>]+>", " ", html)
    plain = re.sub(r"\s+", " ", plain)

    re_email = r"[A-Z0-9._%+\-]{1,64}@[A-Z0-9.\-]{1,255}\.[A-Z]{2,}"
    re_obf = r"([A-Z0-9._%+\-]{1,64})\s*(?:\(|\[|\{)?\s*(?:@|at|snabela)\s*(?:\)|\]|\})?\s*([A-Z0-9.\-]{1,255}\.[A-Z]{2,})"

    emails: set[str] = set()
    for m in re.finditer(re_email, plain, flags=re.I):
        emails.add(m.group(0))
    for m in re.finditer(re_obf, plain, flags=re.I):
        emails.add(f"{m.group(1)}@{m.group(2)}")

    for em_raw in emails:
        em = _normalize_email(em_raw)
        if not em:
            continue

        # DOM-match → harvest container
        node = None
        if tree is not None:
            node = (mailto_map.get(em.lower()) if 'mailto_map' in locals() else None)

        if node is not None:
            try:
                name_h, title_h, phones_h = _harvest_identity_from_container(node)
                out.append(ContactCandidate(
                    name=name_h, title=title_h, emails=[em], phones=phones_h,
                    source="text-email", url=url, dom_distance=0,
                    hints={"harvested": True}
                ))
                continue
            except Exception:
                pass

        # Fallback: nærtekst-vindue i plain
        em_idx = plain.lower().find(em.lower())
        if em_idx < 0:
            continue
        start = max(0, em_idx - 220)
        end = min(len(plain), em_idx + 220)
        near = plain[start:end]

        name = None
        m_name = re.search(rf"({_NAME_WINDOW_REGEX()})", near)
        if m_name:
            name = _collapse_ws(m_name.group(1))

        title = None
        m_title = re.search(r"(?i)\b([A-Za-zÆØÅæøå/\- ]{3,80})\b", near)
        if m_title:
            title = _sanitize_title(m_title.group(1))

        out.append(ContactCandidate(
            name=name, title=title, emails=[em],
            phones=[], source="text-email", url=url, dom_distance=None,
            hints={"near": near[:180]}
        ))

    return out

def _extract_generic_org_contacts(base_url: str, pages_html: list[tuple[str, str]]) -> list[dict]:
    """
    Balanceret 'org bucket':
    - KUN generiske org-emails (med domæne-match hvis muligt)
    - Håndterer obfuskering i rå tekst ([at]/(at), [dot]/(dot))
    - Læser JSON-LD (Organization/LocalBusiness) for email/telephone
    - Telefon medtages konservativt: JSON-LD først; ellers tydeligt DK '70 xx xx xx'
    - Returnerer maks 1 kandidat, maks 3 emails og maks 1 phone
    """
    if not base_url or not pages_html:
        return []

    import re, json
    from urllib.parse import urlparse
    try:
        from bs4 import BeautifulSoup as _BS4
    except Exception:
        _BS4 = None

    # --- Hjælpere ---
    def _site_domain(url: str) -> str:
        try:
            return urlparse(url).netloc.lower()
        except Exception:
            return ""

    def _email_site_match(em: str, site: str) -> bool:
        # Hvis vi ikke kan afgøre domænet, tillad (hellere inkluder end ekskluder)
        if not em or not site:
            return True
        try:
            dom = em.split("@", 1)[1].lower()
            # eks. 'www.example.dk' vs 'example.dk' → endswith
            return dom == site or dom.endswith("." + site) or site.endswith("." + dom)
        except Exception:
            return True

    def _deobfuscate(text: str) -> str:
        repl = {
            "[at]": "@", "(at)": "@", " (at) ": "@", " [at] ": "@", " at ": "@",
            " snabel-a ": "@", " snabela ": "@",
            "[dot]": ".", "(dot)": ".", " dot ": ".",
        }
        # lav-case + exact
        out = text
        for k, v in repl.items():
            out = out.replace(k, v).replace(k.upper(), v).replace(k.capitalize(), v)
        return out

    def _looks_like_dk_70(phone: str) -> bool:
        # +45 70 xx xx xx eller 70xxxxxx (med/uden separatorer)
        digits = re.sub(r"\D", "", phone or "")
        return bool(re.match(r"^(45)?70\d{6}$", digits))

    site = _site_domain(base_url)

    # Saml al HTML og deobfusker
    full_html = " ".join((html or "") for _, html in pages_html)
    full_html_deob = _deobfuscate(full_html)

    # --- 1) Emails fra regex (kun generiske + domæne-match) ---
    all_emails = set(re.findall(r"[A-Z0-9._%+\-]+@[A-Z0-9.\-]+\.[A-Z]{2,}", full_html_deob, flags=re.I))
    generic_emails = []
    for raw in all_emails:
        em = _normalize_email(raw)
        if not em:
            continue
        try:
            local = em.split("@", 1)[0].lower()
        except Exception:
            continue
        if _is_generic_local(local) and _email_site_match(em, site):
            generic_emails.append(em)

    # --- 2) JSON-LD (Organization/LocalBusiness) for email/telephone ---
    jsonld_emails, jsonld_phones = [], []
    if _BS4 is not None:
        for _, html in pages_html:
            if not html:
                continue
            try:
                soup = _BS4(html, "lxml")
            except Exception:
                continue
            for sc in soup.select('script[type="application/ld+json"]'):
                try:
                    data = json.loads(sc.string or "")
                    nodes = data if isinstance(data, list) else [data]
                    for node in nodes:
                        t = node.get("@type")
                        if isinstance(t, list):
                            t = " ".join(t)
                        t_l = (t or "").lower()
                        if not any(k in t_l for k in ("organization", "localbusiness")):
                            continue
                        # Email (kun generiske + domæne-match)
                        em = _normalize_email(node.get("email") or "")
                        if em:
                            loc = em.split("@", 1)[0].lower()
                            if _is_generic_local(loc) and _email_site_match(em, site):
                                jsonld_emails.append(em)
                        # Telefon (gem – vurderes konservativt nedenfor)
                        tel = _normalize_phone(str(node.get("telephone") or ""))
                        if tel:
                            jsonld_phones.append(tel)
                except Exception:
                    continue

    # --- 3) Konservativ phone-udvælgelse ---
    phone_final = None
    # (a) JSON-LD telefon prioriteres hvis tilgængelig
    if jsonld_phones:
        phone_final = jsonld_phones[0]
    # (b) ellers forsøg at spotte et klart DK '70'-hovednummer i rå HTML
    if not phone_final:
        m = re.search(r"(?:\+45\s*)?70\s*\d{2}\s*\d{2}\s*\d{2}", full_html_deob)
        if m:
            phone_final = _normalize_phone(m.group(0))

    emails_final = list(dict.fromkeys(generic_emails + jsonld_emails))[:3]
    phones_final = [phone_final] if phone_final else []

    if not emails_final and not phones_final:
        return []

    return [{
        "name": None,
        "title": None,
        "emails": emails_final,
        "phones": phones_final,   # maks 1 org-nummer
        "score": 0.15,            # lav prioritet ift. personlige kontakter
        "reasons": ["GENERIC_ORG_CONTACT"],
        "source": "page-generic",
        "url": base_url,
        "dom_distance": None,
        "hints": {
            "origin": "regex+jsonld+deobfus",
            "identity_gate_bypassed": True
        },
    }]


def _extract_from_microformats(url: str, tree) -> list[ContactCandidate]:
    """Minimal h-card (p-name, u-email, p-tel, p-job-title)."""
    out: list[ContactCandidate] = []
    if not tree:
        return out
    # selectolax
    if hasattr(tree, "css"):
        for el in tree.css('.h-card, .vcard'):
            name = None; email = None; tel = None; title = None
            n = el.css_first('.p-name') or el.css_first('.fn')
            if n: name = _collapse_ws(n.text())
            e = el.css_first('.u-email') or el.css_first('a[href^="mailto:"]')
            if e:
                href = e.attributes.get('href') if hasattr(e, "attributes") else None
                email = _normalize_email(href or e.text())
            t = el.css_first('.p-tel, .tel')
            if t: tel = _normalize_phone(t.text())
            j = el.css_first('.p-job-title, .role')
            if j: title = _collapse_ws(j.text())
            out.append(ContactCandidate(
                name=name, title=title,
                emails=[x for x in [email] if x],
                phones=[p for p in [tel] if p],
                source="microformats", url=url, dom_distance=0, hints={"selector": "h-card"}
            ))
        return out
    # bs4
    if BeautifulSoup is not None and hasattr(tree, "select"):
        for el in tree.select('.h-card, .vcard'):
            def sel_one(css):
                r = el.select_one(css)
                return r.get_text(strip=True) if r else None
            name = sel_one('.p-name') or sel_one('.fn')
            email = None
            e = el.select_one('.u-email') or el.select_one('a[href^="mailto:"]')
            if e:
                href = e.get('href')
                email = _normalize_email(href or e.get_text(strip=True))
            tel = sel_one('.p-tel, .tel')
            title = sel_one('.p-job-title, .role')
            out.append(ContactCandidate(
                name=_collapse_ws(name), title=_collapse_ws(title),
                emails=[x for x in [email] if x],
                phones=[p for p in [_normalize_phone(tel) if tel else None] if p],
                source="microformats", url=url, dom_distance=0, hints={"selector": "h-card"}
            ))
    return out

# >>> STAFF GRID + WP-JSON HELPERS BEGIN

def _extract_lines_from_block_txt(txt: str) -> tuple[Optional[str], Optional[str]]:
    """Giv (name,title) fra et korts tekst: første linje der ligner navn -> name; næste linje -> title."""
    lines = [l.strip() for l in re.split(r"\r?\n", txt or "") if l.strip()]
    if not lines:
        return None, None

    # Find første plausible navnelinje (med ALL CAPS → Title Case fallback)
    name_idx = None
    name = None
    for i, l in enumerate(lines):
        if _is_plausible_name(l):
            name_idx = i
            name = l
            break
        if l.isupper():
            cap = " ".join(w[:1].upper() + w[1:].lower() for w in l.split())
            if _is_plausible_name(cap):
                name_idx = i
                name = cap
                break
    if name_idx is None:
        return None, None
    if not name:
        name = lines[name_idx]

    # Fjern typisk UI-støj i titellinjer
    UI_NOISE = {"kontakt", "kontakt os", "contact", "contact us", "læs mere", "find medarbejder"}
    title = None
    for j in range(name_idx + 1, min(name_idx + 4, len(lines))):
        lj = lines[j].lower()
        if lj in UI_NOISE or _is_plausible_name(lines[j]):
            continue
        t = _sanitize_title(lines[j])
        if t:
            title = t
            break
    return name, title

def _extract_from_staff_grid(url: str, tree) -> list[ContactCandidate]:
    """
    Finder team/medarbejder-kort (særligt WP: wp-block-*, columns/grids).
    Virker både med selectolax (css) og BeautifulSoup (select).
    """
    out: list[ContactCandidate] = []
    if not tree:
        return out

    def _get_text(node) -> str:
        try:
            if hasattr(node, "text"):
                return _collapse_ws(node.text(deep=True)) or ""
        except Exception:
            pass
        try:
            return _collapse_ws(node.get_text(separator="\n")) or ""
        except Exception:
            return ""

    def _node_text_shallow(n) -> str:
        try:
            if hasattr(n, "text") and callable(getattr(n, "text")):
                return n.text()
        except Exception:
            pass
        try:
            if hasattr(n, "get_text"):
                return n.get_text(" ", strip=True)
        except Exception:
            pass
        try:
            t = getattr(n, "text", "")
            if isinstance(t, str):
                return t
        except Exception:
            pass
        return ""

    containers = []
    css_roots = [
        ".wp-block-columns", ".wp-block-group", ".team", ".team-grid",
        ".is-layout-grid", ".is-layout-flow", ".is-layout-flex",
        ".elementor-section", ".elementor-container", ".elementor-row",
        ".elementor-column", ".elementor-widget", ".elementor-widget-container",
        ".elementor-image-box", ".elementor-team-member",
        ".team-container", ".staff-list", ".employee-grid", ".profile-card", ".bio-card",
        ".team-members", ".our-team", ".people", ".staff", ".person-list", ".members", ".employees",
        ".team-member__list", "section", "main", "article"
    ]
    try:
        if hasattr(tree, "css"):
            for sel in css_roots:
                containers.extend(tree.css(sel))
        elif hasattr(tree, "select"):
            for sel in css_roots:
                containers.extend(tree.select(sel))
    except Exception:
        pass
    if not containers:
        containers = [tree]
    if tree and tree not in containers:
        containers = [tree] + containers

    seen: set[tuple[str, str, str]] = set()
    for cont in containers:
        cards = []
        try:
            card_selectors = (
                "figure, article, .wp-block-column, .wp-block-media-text, .team-member, "
                ".wp-block-group > div, .is-layout-flow > div, .elementor-column, .elementor-widget, "
                ".elementor-widget-container, .elementor-image-box, .elementor-team-member, "
                ".staff-card, .employee, .employee-card, .profile-card, .bio-card, "
                ".member, .person, .staff-item, .team-item, .member-card, .people__item, "
                ".team-member__card, .card, .card-body, .contact-card, .contactperson, "
                ".kontaktperson, .kontakt-card, .contact__person, [class*='kontakt'] [class*='person'], "
                "[class*='contact'] [class*='person'], [class*='staff'] [class*='card']"
            )
            if hasattr(cont, "css"):
                cards = cont.css(card_selectors)
            elif hasattr(cont, "select"):
                cards = cont.select(card_selectors)
        except Exception:
            cards = []

        for card in cards or []:
            raw = _get_text(card)
            if not raw or len(raw) < 5:
                continue

            name, title = None, None
            
            # ---------------- START PÅ MINIMAL PATCH ----------------
            # Vi tilføjer den manglende logik her, FØR den gamle logik og FØR "if not name: continue"
            # Dette er en sikker, kirurgisk tilføjelse i stedet for en fuld erstatning.
            
            # Prioritet 1: Prøv at finde navn fra img[alt]
            try:
                img_tag = card.select_one("img[alt]") if hasattr(card, "select_one") else (card.css_first("img[alt]") if hasattr(card, "css_first") else None)
                if img_tag:
                    alt_text = (img_tag.get("alt", "") if hasattr(img_tag, "get") else img_tag.attributes.get("alt", "")).strip()
                    if _is_plausible_name(alt_text):
                        name = alt_text
            except Exception:
                pass
            
            # Prioritet 2: Prøv at finde navn fra p.font-semibold (specifikt for inventarland.dk, men generisk nok)
            if not name:
                try:
                    p_tag = card.select_one("p.font-semibold") if hasattr(card, "select_one") else (card.css_first("p.font-semibold") if hasattr(card, "css_first") else None)
                    if p_tag:
                        p_text = _node_text_shallow(p_tag)
                        if _is_plausible_name(p_text):
                            name = p_text
                except Exception:
                    pass
            
            # ---------------- SLUT PÅ MINIMAL PATCH ----------------

            if hasattr(card, "get_text"):
                text_multiline = card.get_text(separator="\n")
            else:
                try:
                    text_multiline = "\n".join([n.text().strip() for n in card.iter() if n.text() and n.text().strip()])
                except Exception:
                    text_multiline = raw
            
            # Den eksisterende logik fungerer nu som fallback
            if not name:
                direct_name = None
                try:
                    if hasattr(card, "css"):
                        dn = card.css_first(".person-name, .staff-name, .name, [class*='name']")
                        if dn: direct_name = _collapse_ws(dn.text())
                    elif hasattr(card, "select"):
                        dn = card.select_one(".person-name, .staff-name, .name, [class*='name']")
                        if dn: direct_name = _collapse_ws(dn.get_text(" ", strip=True))
                except Exception:
                    pass
                if _is_plausible_name(direct_name):
                    name = direct_name
            
            if name and text_multiline and not title:
                try:
                    lines = [l.strip() for l in text_multiline.split("\n")]
                    # find første linje der indeholder navnet (casefold for robust match)
                    name_cf = name.casefold()
                    idxs = [i for i, l in enumerate(lines) if name_cf in l.casefold()]
                    if idxs:
                        i_name = idxs[0]
                        # gå baglæns for at finde nærmeste ikke-tomme linje
                        j = i_name - 1
                        while j >= 0 and not lines[j]:
                            j -= 1
                        if j >= 0:
                            cand = _sanitize_title(lines[j])
                            if cand and _looks_like_role(cand):
                                title = cand
                except Exception:
                    pass

            if not title:
                direct_title = None
                try:
                    if hasattr(card, "css"):
                        dr = card.css_first(".job-title, .role, .title, [class*='rolle'], [class*='stilling']")
                        if dr: direct_title = _sanitize_title(_collapse_ws(dr.text()))
                    elif hasattr(card, "select"):
                        dr = card.select_one(".job-title, .role, .title, [class*='rolle'], [class*='stilling']")
                        if dr: direct_title = _sanitize_title(_collapse_ws(dr.get_text(" ", strip=True)))
                except Exception:
                    pass
                title = direct_title or None

            if not name:
                n2, t2 = _extract_lines_from_block_txt(text_multiline)
                if n2 and _is_plausible_name(n2):
                    name = n2
                if not title and t2:
                    title = t2

            emails, phones = [], []
            try:
                if hasattr(card, "css"):
                    for a in card.css("a[href^='mailto:']"):
                        em = _normalize_email(a.attributes.get("href", ""))
                        if em and "undefined" not in em.lower(): emails.append(em)
                elif hasattr(card, "select"):
                    for a in card.select("a[href^='mailto:']"):
                        em = _normalize_email(a.get("href", ""))
                        if em and "undefined" not in em.lower(): emails.append(em)
                
                text_all = _get_text(card)
                dk_phones = re.findall(r'\b(?:\+45\s*)?(\d{2}\s?\d{2}\s?\d{2}\s?\d{2})\b', text_all)
                for p in dk_phones:
                    norm_p = _normalize_phone(p)
                    if norm_p: phones.append(norm_p)
            except Exception:
                pass

            if not name:
                continue

            fingerprint = _fp_from_text(raw or "", prefix="card:")
            key = (name.lower(), (title or "").lower(), fingerprint)
            if key in seen:
                continue
            seen.add(key)

            rs = _role_strength(title)
            out.append(ContactCandidate(
                name=name, title=title, emails=sorted(list(set(emails))), phones=sorted(list(set(phones))),
                source="dom-staff-grid", url=url, dom_distance=0,
                hints={"card": (raw or "")[:160], "role_strength": rs, "fingerprint": fingerprint}
            ))
    return out

def _wp_json_enrich(
    page_url: str,
    html: str,
    timeout: float,
    cache_dir: Optional[Path],
    http_client: Optional[CachingHttpClient] = None,
) -> list[ContactCandidate]:
    """
    WP-JSON fallback: find <link ... href=".../wp-json/wp/v2/pages/<id>"> or ?p=<id>,
    fetch content.rendered, parse again for staff-grid/mailto/microformats.
    """
    out: list[ContactCandidate] = []
    # 1) Find API-URLs in HTML
    # === ANKER: WP_JSON_BUILD_URLS BEGIN ===
    # Konsolideret WP-REST strategi: få men effektive kald
    api_urls = [
        urljoin(page_url, "/wp-json/wp/v2/pages/?_fields=link,title&per_page=100"),
    ]
    # Fallback-søgninger (maks 2) – kun hvis liste ikke giver noget brugbart
    for t in ("kontakt","om"):
        api_urls.append(urljoin(page_url, f"/wp-json/wp/v2/pages/?search={t}&_fields=link,title&per_page=10"))
    # Dedupér rækkefølgebevarende
    api_urls = list(dict.fromkeys(api_urls))
    # === ANKER: WP_JSON_BUILD_URLS END ===
    
    # === ANKER: WP_JSON_PROCESS_LOOP BEGIN ===
    client = http_client or CachingHttpClient(timeout=timeout)

    for idx, api in enumerate(api_urls[:10]):  # Cap at 10 calls per page
        try:
            jtxt = _fetch_text(api, timeout=timeout, cache_dir=cache_dir, max_age_hours=6)
            data = json.loads(jtxt)
            # A) WP types → add relevant custom post types
            if isinstance(data, dict) and "/wp/v2/types" in api:
                for k, v in data.items():
                    slug_t = (k or "").lower()
                    rest_base = (v.get("rest_base") or "").lower() if isinstance(v, dict) else ""
                    label = (v.get("name") or "").lower() if isinstance(v, dict) else ""
                    joined = " ".join([slug_t, rest_base, label])
                    if any(x in joined for x in ("team", "staff", "people", "personale", "medarbejder", "medarbetare", "ansat", "employee", "member")):
                        # Byg origin ud fra selve 'api' i stedet for at bruge 'bases'
                        try:
                            _p = urlsplit(api)
                            _origin = f"{_p.scheme}://{_p.netloc}"
                            api_urls.append(f"{_origin}/wp-json/wp/v2/{rest_base or slug_t}?per_page=100&_embed=1")
                        except Exception:
                            pass
                api_urls = list(dict.fromkeys(api_urls))
                continue
            # B) WP search → follow 'url'/'link' and parse HTML
            if isinstance(data, list) and data and isinstance(data[0], dict) and ("url" in data[0] or "link" in data[0]):
                for it in data[:20]:
                    pgurl = it.get("url") or it.get("link")
                    if not pgurl:
                        continue
                    st2, html2 = client.get(pgurl)
                    if st2 == 200 and html2:
                        t2 = _parse_html(html2)
                        out.extend(_extract_from_staff_grid(pgurl, t2))
                        out.extend(_extract_from_mailtos(pgurl, t2 if t2 is not None else html2))
                        out.extend(_extract_from_microformats(pgurl, t2))
                continue
            # C) WP users   
            if isinstance(data, list) and data and isinstance(data[0], dict) and "name" in data[0] and "content" not in data[0]:
                for u in data[:20]:
                    nm = _collapse_ws(u.get("name"))
                    desc = u.get("description") or ""
                    if not desc:
                        acf = u.get("acf") or {}
                        if isinstance(acf, dict):
                            desc = json.dumps(acf, ensure_ascii=False)
                    emails = []
                    for m in re.finditer(r"[A-Z0-9._%+\-]{1,64}@[A-Z0-9.\-]{1,255}\.[A-Z]{2,}", str(desc), flags=re.I):
                        em = _normalize_email(m.group(0))
                        if em:
                            emails.append(em)
                    if nm or emails:
                        out.append(ContactCandidate(
                            name=nm if _is_plausible_name(nm) else None,
                            title=None,
                            emails=sorted(set(emails))[:3],
                            phones=[],
                            source="wp-json-users",
                            url=page_url,
                            dom_distance=0,
                            hints={"api": "wp-json/users"}
                        ))
                continue
            # D) Pages/Posts with content.rendered
            if isinstance(data, list) and data:
                for item in data[:1]:  # Process only first item
                    if not isinstance(item, dict):
                        continue
                    rendered = (item.get("content") or {}).get("rendered", "") or ""
                    if not rendered:
                        continue
                    t2 = _parse_html(rendered)
                    out.extend(_extract_from_staff_grid(page_url, t2))
                    out.extend(_extract_from_mailtos(page_url, t2 if t2 is not None else rendered))
                    out.extend(_extract_from_microformats(page_url, t2))
            elif isinstance(data, dict):
                rendered = (data.get("content") or {}).get("rendered", "") or ""
                if rendered:
                    t2 = _parse_html(rendered)
                    out.extend(_extract_from_staff_grid(page_url, t2))
                    out.extend(_extract_from_mailtos(page_url, t2 if t2 is not None else rendered))
                    out.extend(_extract_from_microformats(page_url, t2))
        except Exception as e:
            log.debug(f"WP-JSON error for {api}: {e}")
            continue
    # === ANKER: WP_JSON_PROCESS_LOOP END ===
    return out
# <<< STAFF GRID + WP-JSON HELPERS END

# ------------------------------- Scoring --------------------------------------

# P6: tunables til scoring
INITIALS_PENALTY = 1.0
EARLY_EXIT_THRESHOLD = 5.0
# === ANKER: SCORING_CONSTANTS ===

def _score_candidate(c: ContactCandidate, directors: Optional[list[str]] = None) -> ScoredContact:

    s = 0.0
    why: list[str] = []

    # Navn
    if _is_plausible_name(c.name):
        s += 2.0
        why.append("NAME")

    # Titel (rolle-styrke)
    t = _sanitize_title(c.title)
    rs = _role_strength(t)
    if rs > 0:
        s += {1: 1.0, 2: 2.0, 3: 3.0}[rs]
        why.append(f"ROLE_{rs}")
    elif c.title and not t:
        why.append("TITLE_UI_DROPPED")

    # Emails
    emails = [_normalize_email(e) for e in (c.emails or [])]
    emails = [e for e in emails if e]
    c.emails = emails
    if emails:
        locals_ = [e.split("@", 1)[0] for e in emails]
        if any(_is_generic_local(l) for l in locals_):
            s -= 2.0
            why.append("GENERIC_EMAIL")

    # Email↔navn match
    em_sc = _best_email_score(emails, c.name)
    if em_sc >= 2:
        s += 3.0
        why.append(f"EMAIL_MATCH({em_sc})")

    # E-mail domæne matcher sitet?
    if emails and any(_email_domain_matches_site(e, c.url) for e in emails):
        s += 0.7
        why.append("DOMAIN_MATCH")

    # Telefon? (stærkere signal når phone er fundet i samme container/card)
    if any(_normalize_phone(p) for p in (c.phones or [])):
        s += 1.0
        why.append("PHONE_IN_CONTAINER")

    # DOM nærhed
    if c.dom_distance is not None:
        if c.dom_distance <= 1:
            s += 1.0
            why.append("DOM_NEAR")
        elif c.dom_distance <= 2:
            s += 0.3
            why.append("DOM_OK")

    # "Identity-gate"
    gate_bypass = (
        c.source == "page-generic"
        or (isinstance(c.hints, dict) and c.hints.get("identity_gate_bypassed"))
    )
    if gate_bypass:
        # lille positiv bundscore så den overlever rensningen men lander nederst
        s = max(s, 0.2)
        if "GENERIC_ORG_CONTACT" not in why:
            why.append("GENERIC_ORG_CONTACT")
    else:
        # normal gate
        if not _passes_identity_gate(c.name, c.emails, _sanitize_title(c.title), c.url, c.dom_distance):
            s = -999.0
            why.append("GATE_FAIL")

    # URL-kontekst: kontakt/om/team/medarbejder-sider
    try:
        pth = (urlsplit(c.url).path or "").lower()
    except Exception:
        pth = ""
    if any(sl in pth for sl in ("/kontakt", "/contact", "/team", "/medarbejder", "/medarbejdere", "/people", "/staff", "/om", "/about")):
        s += 1.0
        why.append("SECTION_MATCH")

    # Initialer-penalty hvis ingen navn og 2-4 bogstaver i local
    if c.emails and not _is_plausible_name(c.name):
        try:
            local = c.emails[0].split("@", 1)[0]
            letters = re.sub(r"[^a-zæøå]", "", local.lower())
            if 2 <= len(letters) <= 4:
                s -= INITIALS_PENALTY
                why.append("INITIALS_DOWNWEIGHT")
        except Exception:
            pass
    # === ANKER: INITIALS_PENALTY_TUNED ===

    # Root-only penalty (forside) — men dæmp hvis vi fandt stærke signaler
    if pth.strip("/") == "":
        if "DOM_NEAR" in why or "EMAIL_MATCH" in " ".join(why):
            s -= 0.3
        else:
            s -= 1.0

    # Direktør-match fra CVR/hook → bonus
    try:
        if _director_match_for_site(c.name, c.url):
            s += 2.0
            why.append("DIRECTOR_MATCH")
    except Exception:
        pass

    # DF-hintede direktørnavne (valgfrit)
    try:
        if directors:
            n0 = _norm_person_name(c.name)
            if n0 and any(_norm_person_name(d) == n0 for d in directors):
                s += 1.5
                why.append("DIRECTOR_HINT")
    except Exception:
        pass

    # [NYT] Promotion: hvis ingen titel, men name ligner en rolle → brug facit på name
    try:
        if not getattr(c, "title", None):
            raw_name = getattr(c, "name", None)
            role_like = _sanitize_title(raw_name)  # fjerner CTA/telefon/URL-støj
            if role_like:
                canon2 = _canonicalize_title_or_none(role_like)
                if canon2:
                    canonical_title2, meta2 = canon2
                    setattr(c, "title", canonical_title2)
                    # name var en rolle/afdeling → ryd, så tabellen ikke viser rollens tekst som navn
                    setattr(c, "name", None)
                    hints = getattr(c, "hints", {}) or {}
                    hints["title_from_name"] = True
                    hints["title_validation"] = meta2
                    setattr(c, "hints", hints)
                    why.append("TITLE_FROM_NAME")
                    globals()["_TITLE_FACIT_OK"] += 1  # tæl som “ok”, da vi fik en valid titel
                    if _TITLES_DEBUG:
                        log.info(f"[TitleFacit] FROM_NAME raw={raw_name!r} -> canonical={canonical_title2!r} meta={meta2}")
    except Exception as e:
        if _TITLES_DEBUG:
            log.debug(f"[TitleFacit] FROM_NAME error err={e}")

    # [NYT] Drop “kontakter” hvor name er CTA/adresse (typisk “Kontakt os på … Trafikcenter Allé …”)
    try:
        nm = (getattr(c, "name", "") or "")
        nm_l = nm.strip().lower()
        CTA_NAME_BLACKLIST = {
            "kontakt", "kontakt os", "kontakt os på",
            "kundeservice", "customer service",
            "kontakt os på trafikcenter allé 12, dk-4200 slagelse"
        }
        looks_like_address = bool(
            re.search(r"\b\d{4}\b", nm_l) or
            re.search(r"\b(all[eé]|vej|gade|boulevard|plads)\b", nm_l, flags=re.I)
        )

        if ("TITLE_FACIT_OK" not in why and "TITLE_FROM_NAME" not in why) and (nm_l in CTA_NAME_BLACKLIST or looks_like_address):
            why.append("DROP_UI_OR_ADDRESS")
            s = 0.0  # gør kandidaten inaktiv i output (støjer ikke i top-N)
            if _TITLES_DEBUG:
                log.info(f"[TitleFacit] DROP nm={nm!r}")
    except Exception:
        pass

    return ScoredContact(candidate=c, score=s, reasons=why)
    # === ANKER: DIRECTOR_SCORE_BOOST ===

def _head_and_resolve(u: str, timeout: float = DEFAULT_TIMEOUT) -> tuple[str, int | None]:
    """Forenklet: Brug GET til at følge redirects og tjek status."""
    try:
        if callable(_hc_head_and_resolve):
            return _hc_head_and_resolve(u, timeout=timeout) # type: ignore[misc]
    except Exception:
        pass
    host = _host_of(u)
    tries = 0
    last_final = u
    last_status: Optional[int] = None
    while tries < 3:
        tries += 1
        # Respekter backoff
        until = _RATE_LIMIT_HOSTS.get(host, 0.0)
        now = time.time()
        if until and now < until:
            time.sleep(min(10.0, until - now) + random.uniform(0, 0.5))
        try:
            import httpx
            with httpx.Client(follow_redirects=True, timeout=timeout,
                              headers={"User-Agent": "VextoContactFinder/1.0"}) as c:
                r = c.get(u)
                final_url = str(r.url) if getattr(r, "url", None) else u
                st = r.status_code
                last_final, last_status = final_url, st
                if st == 429:
                    _RL_429_COUNTS[host] += 1
                    _RATE_LIMIT_HOSTS[host] = time.time() + 60.0 + random.uniform(0, 5.0)
                    time.sleep(5.0 + random.uniform(0, 3.0))
                    continue
                break
        except Exception:
            last_final, last_status = u, None
            break
    return last_final, last_status


# Cache til (final_url->status) pr. kørsel
class _HeadCache:
    def __init__(self):
        self.cache: dict[str, int | None] = {}

    def get(self, u: str) -> int | None:
        return self.cache.get(u)

    def set(self, u: str, status: int | None):
        self.cache[u] = status

# Match navneløse emails til staff-grid navne via initialer (fx jb@ → Jens B…)
def _match_emails_to_names(cands: list["ContactCandidate"]) -> list["ContactCandidate"]:
    grid_by_key: dict[str, ContactCandidate] = {}

    def _keys_from_name(full: str) -> list[str]:
        parts = re.findall(r"[A-Za-zÆØÅæøå]{2,}", full or "")
        if len(parts) < 2:
            return []
        k2 = (parts[0][0] + parts[1][0]).lower()
        keys = [k2]
        if len(parts) >= 3:
            k3 = (parts[0][0] + parts[1][0] + parts[2][0]).lower()
            keys.append(k3)
        return keys

    # Indeksér staff-grid kandidater uden emails
    for c in cands:
        if c.source == "dom-staff-grid" and c.name and not c.emails:
            for k in _keys_from_name(c.name):
                grid_by_key.setdefault(k, c)

    # Prøv at matche navneløse emails til grid-navne via 2-3 initialer
    for c in cands:
        if c.name or not c.emails:
            continue
        for em in c.emails:
            local = em.split("@", 1)[0].lower()
            if local.isalpha() and 2 <= len(local) <= 3 and local in grid_by_key:
                g = grid_by_key[local]
                c.name = g.name
                c.title = c.title or g.title
                if isinstance(c.hints, dict):
                    c.hints["matched_from_grid"] = True
                else:
                    c.hints = {"matched_from_grid": True}
                break

    # ANKER: NAME_FROM_EMAIL_FALLBACK
    # Hvis vi stadig ingen navn har, afled navn fra email-local (pascal@ -> Pascal)
    for c in cands:
        if c.name or not c.emails:
            continue
        try:
            local = c.emails[0].split("@", 1)[0]
        except Exception:
            continue
        # split på . _ - og komprimer
        tokens = [t for t in re.split(r"[._\-]+", local) if re.fullmatch(r"[a-zA-ZæøåÆØÅ]{2,}", t)]
        if not tokens:
            continue
        cand = " ".join(t[:1].upper() + t[1:].lower() for t in tokens[:3])
        if _is_plausible_name(cand):
            c.name = cand
            if isinstance(c.hints, dict):
                c.hints["name_from_email"] = True
            else:
                c.hints = {"name_from_email": True}
    return cands



# ------------------------------- Orkestrering ---------------------------------

class ContactFinder:
    """Henter 1–N sider, udtrækker kandidater, dedupliker, scorer og returnerer top-1."""

    def __init__(
        self,
        timeout: float = DEFAULT_TIMEOUT,
        pw_budget: int = 5,
        cache_dir: Optional[Path] = ".http_diskcache/html",
        use_browser: str = "auto", # {"auto","always","never"}
        gdpr_minimize: bool = False,
        http_client: Optional[CachingHttpClient] = None,
    ):
        self.timeout = timeout
        self.cache_dir = Path(cache_dir) if cache_dir else None
        # ÉN kilde til sandhed for browser-politik
        self.use_browser = (use_browser or "auto").strip().lower()  
        if self.use_browser not in {"auto", "always", "never"}:
            self.use_browser = "auto"
        # GDPR/output-minimering
        self.gdpr_minimize = bool(gdpr_minimize)
        # Per-kørsel HTML-cache (kun for denne instans)
        self._html_cache_local: dict[str, str] = {}
        # Normaliserede URLs som vi allerede har Playwright-renderet i denne kørsel
        self._rendered_once: set[str] = set()
        self._pw_attempted: set[str] = set() # URLs forsøgt til PW (debounce)
        self._pw_budget: int = int(pw_budget)  # maks. render pr. kørsel (konfigurerbart)
        # NY: HTTP-client – brug injiceret eller opret ny
        self.http_client = http_client or CachingHttpClient(timeout=timeout)

    def _diag(self, url: str, code: str, extra: dict | None = None) -> None:
        try:
            payload = {"url": url, "code": code, **(extra or {})}
            log.info("CF DIAG %s", payload)
        except Exception:
            # fail-silent – diag må aldrig vælte flowet
            pass

    # ------------------------- Browser / rendering -------------------------

    def _should_render(self, url: str, html: Optional[str]) -> bool:
        """
        Politisk + heuristisk beslutning om Playwright-rendering.

        - 'never'  -> aldrig render
        - 'always' -> render altid
        - 'auto'   -> render hvis:
            * URL ligner kontakt-/om-/team-side, ELLER
            * HTML er tom/tynd eller indeholder JS-hints (fx Elementor) eller mangler kontaktlinks
        """
        policy = self.use_browser
        if policy == "never":
            return False
        if policy == "always":
            return True

        # AUTO
        u = url or ""
        if _is_contactish_url(u):  # brug den definerede slug-heuristik
            return True

        txt = (html or "")
        if not txt:
            return True
        if _needs_js(txt):
            return True

        return False

    def _render_with_async_client(self, url: str) -> Optional[str]:
        """Kør AsyncHtmlClient i separat tråd/loop for sync-kald."""
        #if AsyncHtmlClient is None:
        #    return None

        result_holder = {"html": None, "err": None}

        def runner():
            try:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)

                async def go():
                    client: AsyncHtmlClient = AsyncHtmlClient(stealth=True)  # type: ignore
                    await client.startup()
                    try:
                        # Brug stærk render når vi beder om den
                        res = await client.get_raw_html(url, force_playwright=True)  # type: ignore
                        if isinstance(res, dict):
                            return res.get("html", "")
                        return res or ""
                    finally:
                        # Kompatibel lukning
                        import inspect
                        for name in ("close", "aclose", "shutdown"):
                            fn = getattr(client, name, None)
                            if fn:
                                try:
                                    res = fn()
                                    if inspect.isawaitable(res):
                                        await res
                                except Exception:
                                    pass
                                break

                result_holder["html"] = loop.run_until_complete(go())
                loop.close()
            except Exception as e:
                result_holder["err"] = e

        t = threading.Thread(target=runner, daemon=True)
        t.start()
        t.join()
        return result_holder["html"]  # kan være None

    def _fetch_text_smart(self, url: str) -> str:
        """
        Billig HTTP først; render én gang for kontakt/om/team-sider
        eller hvis HTML er tom/tynd/JS-afhængig.
        """
        # Validér URL først
        if not url or not isinstance(url, str):
            return ""
        
        url = url.strip()
        if url.lower() in ("nan", "none", "null", ""):
            return ""
        
        # Tilføj protokol hvis manglende
        if not url.startswith(('http://', 'https://')):
            if re.match(r'^(www\.)?([a-z0-9\-]+\.)+[a-z]{2,}', url, re.I):
                url = f"https://{url}"
            else:
                log.debug(f"Invalid URL format: {url}")
                return ""
        
        key = _norm(url)
        # Lokal per-run cache
        cached = self._html_cache_local.get(key)
        if cached is not None:
            return cached
        
        html = ""
        # 1) Prøv hurtig HTTP
        try:
            status, html = self.http_client.get(url)  # NY: Brug self.http_client i stedet for _fetch_text
            if status < 200 or status >= 400:
                log.warning(f"HTTP fetch bad status {status} for {url}")
                html = ""
        except Exception as e:
            log.error(f"http_client.get failed for {url}: {e!r}", exc_info=True)
            html = ""

        html = html or ""
        stripped_html = html.strip()

        # 1b) Hvis vi kan se 404 i HTML, så drop Playwright og returnér
        if stripped_html and _looks_404(html):
            self._html_cache_local[key] = html or ""
            return html or ""

        # 2) Vurdér om vi bør render'e
        should_render = False
        if self.use_browser == "always":
            should_render = True
        elif self.use_browser == "auto":
            lh = html.lower()
            stripped = stripped_html
            # NYT: Tjek for tynd/placeholder HTML (kort og ingen kontakt-signaler)
            placeholderish = bool(stripped) and len(html) < 1500 and ("mailto:" not in lh) and ("tel:" not in lh)
            should_render = (
                _is_contactish_url(url)
                or not stripped
                or placeholderish
                or ("elementor" in lh)
                or _needs_js(html)
            )
        
        # 2b) Render aldrig hvis URL svarer 404
        if should_render:
            with contextlib.suppress(Exception):
                status, _ = self.http_client.get(url)  # NY: Brug self.http_client
                if status and 400 <= status < 500:
                    should_render = False
        
        if should_render and key not in self._rendered_once:
            # Debounce + budget
            if key in self._pw_attempted or self._pw_budget <= 0:
                self._html_cache_local[key] = html or ""
                return html or ""
            self._pw_attempted.add(key)
            self._pw_budget -= 1
            pre_status = 200 if stripped_html and not _looks_404(html) else None
            rendered = _fetch_with_playwright_sync(
                url,
                pre_html=html or None,
                pre_status=pre_status,
                http_client=self.http_client,
            )
            if rendered:
                html = rendered
            self._rendered_once.add(key)
        
        # 3) Gem i lokal cache og retur
        self._html_cache_local[key] = html or ""
        return html or ""

    @staticmethod
    def _abs(base: str, path: str) -> str:
        if not path:
            return base
        if path.startswith("http://") or path.startswith("https://"):
            return path
        if path.startswith("/"):
            m = re.match(r"^(https?://[^/]+)", base)
            if m:
                return m.group(1) + path
        return base.rstrip("/") + "/" + path.lstrip("/")

    def _pages_to_try(self, url: str, limit_pages: int = 4, home_html: str | None = None) -> list[str]:
        """
        Byg fallback-URL'er baseret på sprogdetektion/TLD via _localized_fallback_paths.
        Falder tilbage til globale FALLBACK_PATHS hvis intet kan detekteres.
        """
        try:
            localized = _localized_fallback_paths(url, home_html=home_html)  # relative paths
        except Exception:
            localized = []

        paths = localized or list(FALLBACK_PATHS)  # fallback til global liste hvis nødvendigt
        pages = [url] + [self._abs(url, p) for p in paths]

        # dedup – bevar rækkefølge
        seen: set[str] = set()
        ordered: list[str] = []
        for u in pages:
            if u not in seen:
                ordered.append(u)
                seen.add(u)

        return ordered[:max(1, limit_pages)]

    # ---------------------------- Extract & score ----------------------------

    def _extract_all(self, url: str, html: str) -> list[ContactCandidate]:
        tree = _parse_html(html)
        
        out: list[ContactCandidate] = []
        # Primære extractors
        out.extend(_extract_from_jsonld(url, html))
        out.extend(_extract_from_mailtos(url, tree if tree is not None else html))
        out.extend(_extract_from_dom(url, tree))
        out.extend(_extract_from_text_emails(url, html))
        out.extend(_extract_from_rdfa(url, tree))
        out.extend(_extract_from_microformats(url, tree))
        # NEW: Staff-grid (WP blokke, team/medarbejdere)
        out.extend(_extract_from_staff_grid(url, tree))

        # NEW: WP-JSON fallback hvis vi stadig ikke fandt noget og siden ligner WP
        if not out and ("wp-content" in html or "wp-json" in html or "wp-block" in html or "wp-" in html):
            with contextlib.suppress(Exception):
                out.extend(
                    _wp_json_enrich(
                        url,
                        html,
                        timeout=self.timeout,
                        cache_dir=self.cache_dir,
                        http_client=self.http_client,
                    )
                )

        # --- NYT (P4): Fallback til contact_fetchers på allerede hentet HTML ---
        # Bruges kun hvis ingen kandidater pt. har emails/phones (undgår dobbelt-crawl).
        try:
            has_any_contact = any((c.emails or c.phones) for c in out)
            if not has_any_contact:
                # Robust import af fælles finder (flere mulige pakkestier)
                try:
                    from vexto.scoring.contact_fetchers import find_contact_info  # type: ignore
                except Exception:
                    try:
                        from src.vexto.scoring.contact_fetchers import find_contact_info  # type: ignore
                    except Exception:
                        from ..scoring.contact_fetchers import find_contact_info  # type: ignore

                # Brug eksisterende parse-tree hvis muligt; ellers parse her
                try:
                    from bs4 import BeautifulSoup
                except Exception:
                    BeautifulSoup = None  # type: ignore

                soup = tree
                if soup is None and BeautifulSoup is not None:
                    soup = BeautifulSoup(html or "", "html.parser")  # type: ignore

                # Kald uden deep_contact for ikke at starte ny crawl her
                import asyncio

                def _run_fetchers():
                    loop = asyncio.new_event_loop()
                    try:
                        asyncio.set_event_loop(loop)

                        async def go():
                            return await find_contact_info(soup, url, deep_contact=False)  # type: ignore[arg-type]
                        res = loop.run_until_complete(go())
                        return res or {}
                    finally:
                        try:
                            loop.close()
                        except Exception:
                            pass
                        try:
                            asyncio.set_event_loop(None)
                        except Exception:
                            pass

                data = _run_fetchers()
                emails = list(dict.fromkeys((data.get("emails_found") or [])))[:3]
                phones = list(dict.fromkeys((data.get("phone_numbers_found") or [])))[:3]

                if emails or phones:
                    c = ContactCandidate(
                        name=None,
                        title=None,
                        emails=emails,
                        phones=phones,
                        source="contact_fetchers",
                        url=url,
                        dom_distance=None,
                        hints={"origin": "contact_fetchers_fallback"},
                    )
                    out.append(c)
        except Exception:
            # Bevidst stilhed: primære extractors er stadig authoritative
            pass
        # --- SLUT NYT (P4) ---

        # Normaliser let
        norm: list[ContactCandidate] = []
        for c in out:
            c.title = _sanitize_title(c.title)
            c.emails = [e for e in (_normalize_email(e) for e in c.emails) if e]
            c.phones = [p for p in (_normalize_phone(p) for p in c.phones) if p]
            norm.append(c)
        return norm


    def _merge_dedup(self, cands: Iterable[ContactCandidate]) -> list[ContactCandidate]:
        # 1) Primær dedup: (navn + stærk email) ellers (navn + url + source)
        bykey: dict[tuple, ContactCandidate] = {}
        for c in cands:
            k = _dedup_key(c)
            if k in bykey:
                bykey[k] = _merge(bykey[k], c)
            else:
                bykey[k] = c

        # 2) Sekundær sammenfletning: samme navn + samme apex-domæne på tværs af kilder
        #    – kun hvis mindst én side mangler emails, for at undgå at flette to forskellige personer med samme navn.
        from urllib.parse import urlsplit as _us

        def _host_apex(u: str) -> str:
            try:
                h = (_us(u).hostname or "").lower()
            except Exception:
                h = ""
            return _apex(h)

        final_by_name_host: dict[tuple[str, str], ContactCandidate] = {}
        orphans: list[ContactCandidate] = []

        for c in bykey.values():
            name = (c.name or "").strip().lower()
            host = _host_apex(c.url or "")
            if not name or not host:
                orphans.append(c)
                continue

            nh = (name, host)
            if nh not in final_by_name_host:
                final_by_name_host[nh] = c
                continue

            existing = final_by_name_host[nh]
            # Flet kun hvis mindst én af dem mangler emails – eller hvis de deler en email.
            c_em = set(c.emails or [])
            e_em = set(existing.emails or [])
            if (not c_em) or (not e_em) or (c_em & e_em):
                final_by_name_host[nh] = _merge(existing, c)
            else:
                # To forskellige personer med samme navn + forskellige emails → behold som særskilte
                orphans.append(c)

        return list(final_by_name_host.values()) + orphans


    def _score_sort(self, cands: Iterable[ContactCandidate], directors: Optional[list[str]] = None) -> list[ScoredContact]:
        scored = [_score_candidate(c, directors=directors) for c in cands]
        scored.sort(key=lambda sc: sc.score, reverse=True)
        return scored

    # ------------------------------- Sync API ---------------------------------

        # >>> ANKER START: FIND_ALL_ORCHESTRATION
    def find_all(self, url: str, limit_pages: int = 4, directors: list[str] | None = None) -> list[dict]:
        """Returnér alle (scored) kandidater som dicts (reasons inkluderet)."""
        t0 = time.time()

        # Nulstil kun PW-debounce pr. domæne/run (behold konfigureret budget)
        self._pw_attempted = set()
        self._rendered_once = set()

        # 1) Hent forsiden
        htmls: dict[str, str] = {}
        try:
            root_html = self._fetch_text_smart(url)
            htmls[url] = root_html
        except Exception as e:
            root_html = ""
            log.exception(f"find_all: root fetch failed for {url}")
            self._diag(url, "NO_HTML_FETCHED", {"status": 0, "exc": repr(e)})
            return []

        # >>> ANKER: CF/DOMAIN_MISMATCH_GUARD
        # 2) Domain-mismatch guard – tillad cross-TLD hvis core label (brand) er ens
        try:
            resolved_root, _st0 = _head_and_resolve(url, timeout=self.timeout)
        except Exception:
            resolved_root = url

        from urllib.parse import urlsplit
        base_host = (urlsplit(url).hostname or "").lower()
        final_host = (urlsplit(resolved_root).hostname or "").lower()

        # Faldbaggrund hvis env-flag ikke er defineret endnu
        _allow_cross_cc = globals().get("_ALLOW_CROSS_CC_REDIRECTS", False)

        def _core_label(host: str) -> str:
            """Returnér 'brandet' (næstsidste label) – enkel og robust:
               www.activemotion.dk -> activemotion
               activemotion.se     -> activemotion
               shop.company.com    -> company
            """
            h = (host or "").lstrip(".")
            if h.startswith("www."):
                h = h[4:]
            parts = h.split(".")
            return parts[-2] if len(parts) >= 2 else (parts[0] if parts else "")

        same_core = bool(base_host and final_host and _core_label(base_host) == _core_label(final_host))

        if not _same_apex(url, resolved_root):
            # 1) eksplicit whitelist (tillader også subdomæner af whitelisted apex)
            if _host_in_whitelist(final_host) or _is_whitelisted_mismatch(resolved_root):
                log.info("Domain-mismatch tilladt via whitelist: base=%s -> final=%s", url, resolved_root)
                url = resolved_root
            # 2) samme land (samme 2-bogstavs ccTLD, fx dk==dk)
            elif _same_country_tld(base_host, final_host):
                log.debug("Domain-mismatch men samme ccTLD (%s) – tillades: base=%s -> final=%s",
                          _cc_tld_from_host(final_host), url, resolved_root)
                url = resolved_root
            # 3) samme core label (brand) – fx activemotion.dk -> activemotion.se
            elif same_core:
                log.info("Domain-mismatch allowed (same core label '%s'): base=%s -> final=%s",
                         _core_label(final_host), url, resolved_root)
                url = resolved_root
            # 4) global override via env (midlertidigt tillad cross-cc)
            elif _allow_cross_cc:
                log.info("Domain-mismatch tilladt af env VEXTO_ALLOW_CROSS_CC_REDIRECTS: base=%s -> final=%s",
                         url, resolved_root)
                url = resolved_root
            # 5) ellers bloker (fremmed TLD)
            else:
                log.warning("Domain-mismatch guard: base=%s -> final=%s (fremmed TLD blokeret)", url, resolved_root)
                return [{
                    "name": None,
                    "title": None,
                    "emails": [],
                    "phones": [],
                    "score": 0.2,
                    "reasons": ["GENERIC_ORG_CONTACT", "DOMAIN_MISMATCH", "FOREIGN_TLD_BLOCKED"],
                    "source": "page-generic",
                    "url": url,
                }]

        final_site_url = resolved_root or url
        # <<< ANKER: CF/DOMAIN_MISMATCH_GUARD

        if _needs_js(root_html) and AsyncHtmlClient is not None and self._pw_budget > 0:
            try:
                rendered = _fetch_with_playwright_sync(
                    url,
                    pre_html=root_html,
                    pre_status=200 if root_html else None,
                    http_client=self.http_client,
                )
                if rendered and not _looks_404(rendered):
                    htmls[url] = rendered
                    root_html = rendered
            except Exception:
                pass

        # 3) Structured data (JSON-LD/Microdata) først
        cands: list[ContactCandidate] = []
        sd = _extract_from_jsonld(url, root_html) + _extract_from_microformats(url, _parse_html(root_html))
        for c in sd:
            c.url = url
            cands.append(c)
        if sd:
            scored = self._score_sort(cands, directors=directors)
            if scored and scored[0].score >= EARLY_EXIT_THRESHOLD:
                cleaned = [
                    {
                        "name": sc.candidate.name,
                        "title": sc.candidate.title,
                        "emails": sc.candidate.emails,
                        "phones": sc.candidate.phones,
                        "url": sc.candidate.url,
                        "person_source_url": sc.candidate.url,
                        "final_website": final_site_url,
                        "domain_mismatch": not _same_apex(url, final_site_url),
                        "score": sc.score,
                        "confidence": _confidence_from(sc),
                        "reasons": sc.reasons,
                        "source": sc.candidate.source,
                        "scraped_contact_text": sc.candidate.hints.get("near") or sc.candidate.hints.get("card"),
                    }
                    for sc in scored
                    if sc.score >= 0 and "GATE_FAIL" not in sc.reasons
                ]
                if cleaned:
                    log.debug("EARLY_EXIT: JSON-LD/Microdata hit (score=%.1f)", scored[0].score)
                    return cleaned

        # 4) Kandidat-sider (Home → DOM → Sitemap → WP)
        try:
            from bs4 import BeautifulSoup as _BS4
            soup_home = _BS4((htmls.get(url) or root_html) or "", "html.parser")
        except Exception:
            soup_home = None

        # NYT: Tving fallback kontakt-paths hvis seeding er tom
        candidates = []  # Initialiser hvis nødvendigt (men tjek konteksten)

        if not candidates:
            base = url.rstrip("/")
            for p in FALLBACK_PATHS:
                cand = base + p
                candidates.append(cand)

        def _is_probable_contact(href: str, text: str = "") -> bool:
            """Generel heuristik: match kontakt-/support-/team-sider via path eller linktekst."""
            h = (href or "").lower()
            t = re.sub(r"\s+", " ", (text or "").lower()).strip()

            # URL-path signaler (generelle – IKKE sitespecifikke)
            url_keys = (
                "/kontakt", "/kontaktformular",
                "/contact", "/contacts",
                "/kundeservice", "/customer-service",
                "/support", "/help", "/help-center",
                "/team", "/staff", "/people", "/management", "/board",
                "/medarbejder", "/medarbejdere", "/ansatte", "/personale",
                "/impressum"  # udbredt "kontakt/juridisk" i DACH
            )

            # Link-tekst signaler (flere sprog, men korte og generiske)
            text_keys = (
                "kontakt", "kontakt os", "kontaktformular",
                "kundeservice",
                "contact", "contact us", "customer service",
                "support", "help", "help center",
                "team", "our team", "staff", "people", "management", "board",
                "medarbejder", "medarbejdere", "ansatte", "personale",
                "impressum"
            )

            return any(k in h for k in url_keys) or any(k in t for k in text_keys)

        # --- DOM-seed fra forside ---
        dom_links: list[str] = []
        if soup_home is not None:
            base_for_apex = final_site_url or url  # brug evt. kanonisk/redirectet base
            for a in soup_home.select("a[href]"):
                href = a.get("href", "") or ""
                if not href or href.startswith(("#", "mailto:", "tel:")):
                    continue
                absu = _uj(base_for_apex, href)
                # kun interne links (samme apex; ignorer www.-forskelle)
                if not _same_apex(absu, base_for_apex):
                    continue
                if _is_probable_contact(_us(absu).path or absu, a.get_text(" ", strip=True)):
                    dom_links.append(absu)

            # dedup – bevar rækkefølge
            dom_links = list(dict.fromkeys(dom_links))

        # --- sitemap-seeds (via eksisterende helpers) ---
        site_links: list[str] = []
        if os.getenv("CF_USE_SITEMAP", "true").lower() in {"1","true","yes","y"}:
            try:
                # bruger dine helpers definéret ovenfor i filen
                site_links = _cf_sitemap_contact_candidates(self, url) or []
            except Exception:
                site_links = []

        # --- WP/andre heuristikker via eksisterende discover_candidates ---
        wp_links = discover_candidates(
            url,
            (htmls.get(url) or root_html),  
            max_urls=min(5, limit_pages),
            http_client=self.http_client,
        )

        # Sammensæt: DOM → sitemap → WP; ensret host → dedup → cap
        seeded = dom_links + site_links + wp_links
        seeded = _sticky_host_urls(url, (htmls.get(url) or root_html), seeded)

        # Debug: se hvor mange kandidatlinks vi faktisk ender med (øverste få vist)
        if log.isEnabledFor(logging.DEBUG):
            try:
                _dbg_list = ", ".join(seeded[:6])
            except Exception:
                _dbg_list = str(len(seeded))
            log.debug("Candidates discovered: %d → %s", len(seeded), _dbg_list)

        _seen = set()
        candidates: list[str] = []
        for cu in seeded:
            if cu not in _seen:
                _seen.add(cu)
                candidates.append(cu)

        # Thin-coverage fallback: suppler med kendte kontakt-stier ved lav dækning
        if len(candidates) < 5:  # hævet fra 3 → 5
            extra = self._pages_to_try(url, limit_pages=limit_pages,
                                    home_html=(htmls.get(url) or root_html))
            added = 0
            for c in extra:
                if c not in _seen and c not in candidates:
                    candidates.append(c)
                    _seen.add(c)
                    added += 1
            if log.isEnabledFor(logging.DEBUG):
                log.debug("Added fallback paths: %d → %s", added, extra[:6])

        # Uanset antal kandidater: sikr at de vigtigste lokaliserede kontaktstier er med
        core_rel = _localized_fallback_paths(url, home_html=(htmls.get(url) or root_html))
        core_abs = [self._abs(url, r) for r in core_rel]
        added_core = 0
        for c in core_abs:
            if c not in _seen and c not in candidates:
                candidates.append(c)
                _seen.add(c)
                added_core += 1
        if log.isEnabledFor(logging.DEBUG) and added_core:
            log.debug("Ensure core fallbacks: +%d → %s", added_core, core_rel[:6])

        # Debug: vis endelige candidates før cap
        if log.isEnabledFor(logging.DEBUG):
            _cap = min(len(candidates), (limit_pages or 50))
            log.debug("Final candidates to crawl (cap before slice): %d → %s",
                    len(candidates), ", ".join(candidates[:_cap]))


        # NYT: fallback hvis vi stadig ingen kandidater har (fx JS-tung forside uden tydelige links)
        if not candidates:
            candidates = self._pages_to_try(url, limit_pages=limit_pages, home_html=(htmls.get(url) or root_html))
            if log.isEnabledFor(logging.DEBUG):
                log.debug("SEED (fallback): %d candidate urls -> %s", len(candidates), candidates[:5])


        # hold det stramt – vi henter alligevel struktureret data per side nedenfor
        candidates = candidates[:min(8, limit_pages)]

        # 5) Hent kandidat-sider
        for cu in candidates:
            if len(cands) >= 10:
                break
            if log.isEnabledFor(logging.DEBUG):
                log.debug("Fetching candidate page: %s", cu)
            try:
                html = self._fetch_text_smart(cu)
                htmls[cu] = html
                if not html or _looks_404(html):
                    continue
                # Structured data på kandidatsider
                sd2 = _extract_from_jsonld(cu, html) + _extract_from_microformats(cu, _parse_html(html))
                for c in sd2:
                    c.url = cu
                    cands.append(c)
                if sd2:
                    scored = self._score_sort(cands, directors=directors)
                    if scored and scored[0].score >= EARLY_EXIT_THRESHOLD:
                        cleaned = [
                            {
                                "name": sc.candidate.name,
                                "title": sc.candidate.title,
                                "emails": sc.candidate.emails,
                                "phones": sc.candidate.phones,
                                "url": sc.candidate.url,
                                "person_source_url": sc.candidate.url,
                                "final_website": final_site_url,
                                "domain_mismatch": not _same_apex(url, final_site_url),
                                "score": sc.score,
                                "confidence": _confidence_from(sc),
                                "reasons": sc.reasons,
                                "source": sc.candidate.source,
                                "scraped_contact_text": sc.candidate.hints.get("near") or sc.candidate.hints.get("card"),
                            }
                            for sc in scored
                            if sc.score >= 0 and "GATE_FAIL" not in sc.reasons
                        ]
                        if cleaned:
                            log.debug("EARLY_EXIT: Candidate page hit (score=%.1f, url=%s)", scored[0].score, cu)
                            return cleaned
                # Eksisterende extractors
                cands.extend(self._extract_all(cu, html))
            except Exception:
                log.debug(f"Fejl ved crawl af kandidat: {cu}")
                continue

        # 6) Playwright kun hvis nødvendig
        root_blank = not (root_html or "").strip()
        if (root_blank or _needs_js(root_html)) and AsyncHtmlClient is not None and self._pw_budget > 0:
            try:
                rendered = _fetch_with_playwright_sync(
                    url,
                    pre_html=root_html,
                    pre_status=200 if root_html else None,
                    http_client=self.http_client,
                )
                if rendered and not _looks_404(rendered):
                    htmls[url] = rendered
                    cands.extend(self._extract_all(url, rendered))
            except Exception:
                log.debug(f"Playwright fejl for {url}")

        # 7) Merge, score og rens
        cands = self._merge_dedup(cands)
        cands = _match_emails_to_names(cands)
        scored = self._score_sort(cands, directors=directors)

        # 8) Org-bucket som fallback
        generic_bucket = _extract_generic_org_contacts(url, list(htmls.items()))
        for gb in generic_bucket:
            cands.append(ContactCandidate(
                name=gb.get("name"),
                title=gb.get("title"),
                emails=gb.get("emails") or [],
                phones=gb.get("phones") or [],
                source="page-generic",
                url=url,
                dom_distance=None,
                hints=gb.get("hints") or {"identity_gate_bypassed": True},
            ))

        # 8b) Re-score inkl. generic bucket
        cands = self._merge_dedup(cands)
        scored = self._score_sort(cands, directors=directors)

        # 9) Final output
        cleaned = []
        seen_keys: set[tuple[str, str]] = set()
        for sc in scored:
            reasons = set(sc.reasons or [])
            is_generic = "GENERIC_ORG_CONTACT" in reasons

            # Drop negative kun hvis det IKKE er org-bucket
            if sc.score < 0 and not is_generic:
                continue
            if "GATE_FAIL" in reasons and not is_generic:
                continue
            # Kræv score/navn for person-kandidater – men skip kravet for org-bucket
            if sc.score < 2.0 and not sc.candidate.name and not is_generic:
                continue

            emails = sc.candidate.emails or []
            url_out = sc.candidate.url or ""
            if self.gdpr_minimize:
                sc.candidate.phones = []
                emails = [e for e in emails if _email_domain_matches_site(e, url_out)]
                sc.candidate.emails = emails

            keys = [(e.lower(), url_out) for e in emails] if emails else [("", url_out)]
            if any(k in seen_keys for k in keys):
                continue
            for k in keys:
                seen_keys.add(k)

            cleaned.append({
                "name": sc.candidate.name,
                "title": sc.candidate.title,
                "emails": sc.candidate.emails,
                "phones": sc.candidate.phones,
                "url": sc.candidate.url,
                "person_source_url": sc.candidate.url,
                "final_website": final_site_url,
                "domain_mismatch": not _same_apex(url, final_site_url),
                "score": sc.score,
                "confidence": _confidence_from(sc),
                "reasons": sc.reasons,
                "source": sc.candidate.source,
                "scraped_contact_text": sc.candidate.hints.get("near") or sc.candidate.hints.get("card"),
            })

        # 10) Log summary
        names_found = sum(1 for r in cleaned if r.get("name"))
        titles_found = sum(1 for r in cleaned if r.get("title"))
        phones_found = sum(1 for r in cleaned if r.get("phones"))
        initials_downweighted = sum(1 for r in cleaned if "INITIALS_DOWNWEIGHT" in r.get("reasons", []))
        section_bonus_hits = sum(1 for r in cleaned if "SECTION_MATCH" in r.get("reasons", []))
        containers_detected = sum(1 for _, h in htmls.items() if h and any(k in h.lower() for k in ('team','staff','section','administration','tilbud','entreprise','elementor-team-member')))
        cf_emails_decoded = sum(1 for c in cands if isinstance(c.hints, dict) and c.hints.get("cfemail"))

        log.debug(
            ("ContactFinder.find_all(%s) -> %d raw, %d cleaned in %.3fs | "
             "names=%d, titles=%d, phones=%d, initials_downweighted=%d, section_hits=%d, "
             "containers_detected=%d, cf_emails_decoded=%d"),
            url, len(scored), len(cleaned), time.time() - t0,
            names_found, titles_found, phones_found, initials_downweighted, section_bonus_hits,
            containers_detected, cf_emails_decoded
        )

        try:
            host = (urlsplit(url).hostname or "").lower()
            final_host = (urlsplit(final_site_url).hostname or "").lower()
            pw_renders = len(self._rendered_once)
            emails_any = sum(1 for r in cleaned if r.get("emails"))
            persons_any = sum(1 for r in cleaned if r.get("name"))
            if _TITLES_DEBUG:
                log.info("[TitleFacit] summary ok=%d reject=%d skipped=%d",
                        _TITLE_FACIT_OK, _TITLE_FACIT_REJECT, _TITLE_FACIT_SKIPPED)

            log.info("CF SUMMARY host=%s final=%s items=%d emails=%d persons=%d pw=%d time=%.3fs | titles=%d ok, %d reject, %d skipped",
                    host, final_host, len(cleaned), emails_any, persons_any, pw_renders, time.time()-t0,
                    _TITLE_FACIT_OK, _TITLE_FACIT_REJECT, _TITLE_FACIT_SKIPPED)
        except Exception:
            pass

        return cleaned
    # <<< ANKER SLUT: FIND_ALL_ORCHESTRATION



    def find(self, url: str, limit_pages: int = 10) -> Optional[dict]:
        """Returnér bedste kandidat som dict – eller None."""
        all_ = self.find_all(url, limit_pages=limit_pages)
        return all_[0] if all_ else None


# ------------------------------ Offentligt API --------------------------------

def find_best_contact(url: str, limit_pages: int = 10) -> Optional[dict]:
    return ContactFinder().find(url, limit_pages=limit_pages)

# Kompat-aliaser (gamle kald i projektet)
def find_contacts(url: str, limit_pages: int = 10) -> list[dict]:
    return ContactFinder().find_all(url, limit_pages=limit_pages)

def extract_contacts(url: str, limit_pages: int = 10) -> list[dict]:
    return ContactFinder().find_all(url, limit_pages=limit_pages)

def extract_best_contact(url: str, limit_pages: int = 10) -> Optional[dict]:
    return ContactFinder().find(url, limit_pages=limit_pages)

def find_top_contact(url: str, limit_pages: int = 10) -> Optional[dict]:
    return ContactFinder().find(url, limit_pages=limit_pages)


# -------------------------- Test/utility shims --------------------------------

def run_enrichment_on_dataframe(
    df,
    url_col: str | None = None,
    *,
    limit_pages: int = 10,
    top_n: int = 1,
    gdpr_minimize: bool = False,
    use_browser: str = "auto",
    max_workers: int = 4,
):
    """
    DataFrame-helper (robust):
    - Finder URL-kolonnen case-insensitivt + heuristik (også "Hjemmeside", "Website URL", "WWW" m.fl.)
    - Normaliserer URL-værdier (tilføjer https:// ved domæner uden schema)
    - Finder top-1 eller top-N kontakter pr. række
    - Lægger kolonner: cf_best_* og cf_all (JSON)
    """
    import json, re
    import os
    import pandas as _pd
    from concurrent.futures import ThreadPoolExecutor, as_completed

    if not hasattr(df, "assign"):
        raise TypeError("run_enrichment_on_dataframe forventer en pandas.DataFrame")

    # --- Hjælpere: kolonne-guess + URL-normalisering -------------------
    def _guess_url_col(_df, explicit: str | None):
        if explicit and explicit in _df.columns:
            return explicit
        cols_lower = {c.lower(): c for c in _df.columns}
        # primære aliaser
        aliases = [
            "url","website","web","domain","hjemmeside","site","company_url",
            "homepage","home_page","website_url","website address","websiteaddress",
            "www","company website","firmaside","hjemme side",
        ]
        for a in aliases:
            if a in cols_lower:
                return cols_lower[a]
        # fuzzy: vælg en kolonne der "ligner" web/url/domæne
        candidates = [c for c in _df.columns if any(k in c.lower() for k in ("web","site","url","domain","home","hjem"))]
        url_like = re.compile(r"^(https?://)?([a-z0-9\-]+\.)+[a-z]{2,}(/.*)?$", re.I)
        best, best_ratio = None, 0.0
        for c in candidates or _df.columns:
            ser = _df[c].astype(str)
            vals = [s.strip() for s in ser if s and s.strip() and s.strip().lower() != "nan"]
            if not vals:
                continue
            hits = sum(1 for v in vals if url_like.match(v))
            ratio = hits / max(1, len(vals))
            if ratio > best_ratio:
                best_ratio, best = ratio, c
        # kræv bare nogle få hits for at acceptere (20%)
        if best and best_ratio >= 0.2:
            return best
        return None

    def _norm_url_value(v: str | None) -> str:
        """Normaliser URL-værdi fra DataFrame."""
        s = (str(v or "")).strip()
        
        # Bedre NaN/None check
        if not s or s.lower() in ("nan", "none", "null", "n/a", "na", ""):
            return ""
        
        # Fjern whitespace og split på komma/semicolon - tag første del
        parts = re.split(r"[,;\s]+", s)
        s = parts[0].strip() if parts else ""
        
        # Hvis tom efter split
        if not s:
            return ""
        
        # Har allerede protokol?
        if re.match(r"^https?://", s, flags=re.I):
            return s
        
        # Tilføj https:// hvis det starter med www.
        if s.lower().startswith("www."):
            return f"https://{s}"
        
        # Tilføj https:// hvis det ligner et domæne (mindst et punktum og 2+ bogstaver til sidst)
        if re.match(r"^([a-z0-9\-]+\.)+[a-z]{2,}(/.*)?$", s, flags=re.I):
            return f"https://{s}"
        
        # Hvis det ikke ligner en valid URL/domæne, returner tom streng
        log.debug(f"Could not normalize URL value: {v}")
        return ""

    # --- Find URL-kolonne (case-insensitiv + heuristik) -----------------
    real_url_col = _guess_url_col(df, url_col)
    if not real_url_col:
        candidates = ["url", "website", "web", "domain", "hjemmeside", "site", "company_url",
                      "homepage","home_page","website_url","website address","www"]
        raise ValueError(f"Kunne ikke finde URL-kolonne. Prøv en af (case-insensitivt): {candidates}")

    cf = ContactFinder(gdpr_minimize=gdpr_minimize, use_browser=use_browser)

    best_names, best_titles, best_emails, best_phones, best_scores, best_urls, all_json = \
        [], [], [], [], [], [], []

    # Byg arbejdslisten (beholder rækkefølge via indeks)
    tasks = []
    for i, row in df.reset_index(drop=True).iterrows():
        raw = row.get(real_url_col)
        url = _norm_url_value(raw)
        tasks.append((i, url))

    def _enrich_one(i_url_tuple):
        i, url = i_url_tuple
        if not url:
            return i, {
                "cf_best_name": None, "cf_best_title": None,
                "cf_best_emails": None, "cf_best_phones": None,
                "cf_best_score": None, "cf_best_url": None,
                "cf_best_source": None, "cf_best_confidence": None,
                "cf_all": json.dumps([]),
            }
        time.sleep(random.uniform(0.5, 2.0))  # Jitter to avoid bursts
        try:
            # Egen instans for tråd-sikkerhed; respekter evt. PW-budget via env
            _pw_budget_env = int(os.environ.get("VEXTO_PW_BUDGET", "5"))
            cf = ContactFinder(
                timeout=DEFAULT_TIMEOUT,
                pw_budget=_pw_budget_env,
                use_browser=use_browser,
                gdpr_minimize=gdpr_minimize
            )
            results = cf.find_all(url, limit_pages=limit_pages)
        except Exception as e:
            log.warning("Fejl ved enrichment for %s: %r", url, e)
            results = []

        if not results:
            return i, {
                "cf_best_name": None, "cf_best_title": None,
                "cf_best_emails": None, "cf_best_phones": None,
                "cf_best_score": None, "cf_best_url": None,
                "cf_best_source": None, "cf_best_confidence": None,
                "cf_all": json.dumps([]),
            }

        best = results[0] if isinstance(results, list) and results else None
        if not best:
            return i, {
                "cf_best_name": None, "cf_best_title": None,
                "cf_best_emails": None, "cf_best_phones": None,
                "cf_best_score": None, "cf_best_url": None,
                "cf_best_source": None, "cf_best_confidence": None,
                "cf_all": json.dumps(results if isinstance(results, list) else []),
            }

        return i, {
            "cf_best_name": best.get("name"),
            "cf_best_title": best.get("title"),
            "cf_best_emails": ", ".join(best.get("emails", []) or []),
            "cf_best_phones": ", ".join(best.get("phones", []) or []),
            "cf_best_score": best.get("score"),
            "cf_best_url": best.get("url"),
            "cf_best_source": best.get("source"),
            "cf_best_confidence": best.get("confidence"),
            "cf_all": json.dumps(results),
        }

    # Kør sekventielt hvis max_workers <= 1 (bevar gamle semantics)
    out_rows: list[dict] = []  # <-- initér output-listen, så den findes i begge grene

    if (max_workers or 0) <= 1:
        for t in tasks:
            _, row_out = _enrich_one(t)
            out_rows.append(row_out)
    else:
        # Parallelisering med ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futures = [ex.submit(_enrich_one, t) for t in tasks]
            # Saml resultater og sorter efter indeks for stabil rækkefølge
            tmp = [f.result() for f in as_completed(futures)]
            for _, row_out in sorted(tmp, key=lambda x: x[0]):
                out_rows.append(row_out)

    out = df.copy()
    out["cf_best_name"] = best_names
    out["cf_best_title"] = best_titles
    out["cf_best_emails"] = best_emails
    out["cf_best_phones"] = best_phones
    out["cf_best_score"] = best_scores
    out["cf_best_url"] = best_urls
    out["cf_all"] = all_json
    return out


# ------------------------------- CLI / test -----------------------------------

def _print_table(items: list[dict]) -> None:
    """Print resultater i en kompakt tabel uden eksterne afhængigheder."""
    if not items:
        print("(ingen resultater)")
        return

    def _trunc(s: object | None, w: int) -> str:
        t = "" if s is None else str(s)
        return t if len(t) <= w else t[: max(0, w - 1)] + "…"

    # Dynamisk: vis 'Source' hvis feltet findes
    show_source = any("source" in r for r in items)

    caps = {
        "idx": max(2, len(str(len(items)))),
        "name": 26,
        "title": 36,
        "emails": 30,
        "phones": 18,
        "score": 5,
        "url": 40,
        "source": 10,
    }

    headers = ["#", "Name", "Title", "Emails", "Phones", "Score", "URL"]
    if show_source:
        headers.insert(6, "Source")

    # Header
    line_fmt = (
        f"{{idx:>{caps['idx']}}}  "
        f"{{name:<{caps['name']}}}  "
        f"{{title:<{caps['title']}}}  "
        f"{{emails:<{caps['emails']}}}  "
        f"{{phones:<{caps['phones']}}}  "
        f"{{score:>{caps['score']}}}  "
    )
    if show_source:
        line_fmt += f"{{source:<{caps['source']}}}  "
    line_fmt += f"{{url:<{caps['url']}}}"

    cols = ["idx", "name", "title", "emails", "phones", "score"] + (["source"] if show_source else []) + ["url"]
    sep = "-" * (sum(caps[c] for c in cols) + 2 * (len(cols) - 1))  # 2 mellem hver kolonne

    print("  ".join(headers))
    print(sep)

    for i, r in enumerate(items, start=1):
        name = _trunc(r.get("name"), caps["name"])
        title = _trunc(r.get("title"), caps["title"])
        emails = _trunc(", ".join(r.get("emails", []) or []), caps["emails"])
        phones = _trunc(", ".join(r.get("phones", []) or []), caps["phones"])
        score_val = r.get("score")
        score = "" if score_val is None else f"{float(score_val):.1f}"
        url = _trunc(r.get("url", ""), caps["url"])
        row = {
            "idx": i,
            "name": name,
            "title": title,
            "emails": emails,
            "phones": phones,
            "score": score,
            "url": url,
        }
        if show_source:
            row["source"] = _trunc(r.get("source", ""), caps["source"])
        print(line_fmt.format(**row))
    return

if __name__ == "__main__":
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    ap = argparse.ArgumentParser(description="Vexto Contact Finder")
    ap.add_argument("url", help="Website URL (fx https://example.com)")
    ap.add_argument("--all", action="store_true", help="Udskriv alle kandidater (scored)")
    ap.add_argument("--limit-pages", type=int, default=10, help="Max antal sider at prøve")
    ap.add_argument("--debug", action="store_true", help="Verbose log")
    ap.add_argument("--gdpr-minimize", action="store_true", help="Minimér persondata i output (drop phones, kun arbejdsmails)")
    ap.add_argument(
        "--format", "-f", choices=("json", "table"), default="json",
        help="Vælg output-format: 'json' (default) eller 'table' (kolonner)."
    )

    args = ap.parse_args()

    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
        # Hæv niveauet på vores egen konsol-handler så DEBUG faktisk vises
        for h in logging.getLogger("vexto.contact_finder").handlers:
            if getattr(h, "name", "") == "cf_console":
                h.setLevel(logging.DEBUG)

    cf = ContactFinder(gdpr_minimize=bool(args.gdpr_minimize))
    if args.all:
        results = cf.find_all(args.url, limit_pages=args.limit_pages)
        if args.format == "table":
            _print_table(results)
        else:
            print(json.dumps(results, ensure_ascii=False, indent=2))
    else:
        best = cf.find(args.url, limit_pages=args.limit_pages)
        if args.format == "table":
            _print_table([best] if best else [])
        else:
            print(json.dumps(best, ensure_ascii=False, indent=2))

