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
import re
import sys
import time
import threading
import logging.handlers
import random  # ANKER: RATE_LIMIT_BACKOFF
from collections import defaultdict  # ANKER: RATE_LIMIT_BACKOFF
from dataclasses import dataclass
from datetime import datetime, timedelta
from functools import lru_cache
from pathlib import Path
from urllib.parse import urljoin, urlsplit
from typing import Any, Iterable, Optional
from vexto.scoring.http_client import _accept_encoding

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

try:
    # Kræver at AsyncHtmlClient findes i jeres http_client
    from vexto.scoring.http_client import AsyncHtmlClient
except Exception:
    AsyncHtmlClient = None  # type: ignore

_SHARED_PW_CLIENT = None

# -------------------- P1: Per-host caches + sticky host helpers ----------------
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

def _now() -> float:
    return time.time()

def _respect_host_budget(u: str) -> bool:
    host = _host_of(u)
    if not host:
        return True
    if _HOST_REQUESTS[host] >= _HOST_REQUEST_BUDGET:
        log.info(f"Budget reached for host={host} ({_HOST_REQUESTS[host]}) – skipping {u}")
        return False
    return True

def http_get(u: str, timeout: float = 10.0, retries: int = 3) -> tuple[int, str]:
    """Eneste vej til HTML: GET med budget + exponential backoff + jitter + Retry-After."""
    if not _respect_host_budget(u):
        return 0, ""
    host = _host_of(u)
    # Respektér evt. aktiv backoff-vindue
    earliest = _RATE_LIMIT_HOSTS.get(host, 0.0)
    if _now() < earliest:
        wait = earliest - _now()
        time.sleep(min(wait, 3.0))
    last_status, last_text = 0, ""
    for attempt in range(retries):
        try:
            _HOST_REQUESTS[host] += 1
            ae = _accept_encoding() if callable(_accept_encoding) else _accept_encoding
            if not isinstance(ae, (str, bytes)) or (isinstance(ae, str) and not ae.strip()):
                ae = "gzip, deflate, br"
            headers = {
                "User-Agent": "VextoContactFinder/1.0 (+https://vexto.io)",
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                "Accept-Encoding": ae,
            }
            with httpx.Client(http2=True, timeout=timeout, headers=headers, follow_redirects=True) as cli:
                r = cli.get(u)
                last_status = r.status_code
                last_text = r.text or ""
                if last_status in (429, 503):
                    # Respect Retry-After hvis sat
                    ra = r.headers.get("Retry-After")
                    sleep_for = (2 ** attempt) + random.uniform(0, 1.25)
                    if ra:
                        try:
                            sleep_for = max(sleep_for, float(ra))
                        except Exception:
                            pass
                    _RL_429_COUNTS[host] += 1
                    # Sæt fremtidigt earliest-ok (circuit)
                    _RATE_LIMIT_HOSTS[host] = _now() + min(60.0, sleep_for * 2.5)
                    log.warning(f"Rate-limit {last_status} on {host} attempt={attempt+1}, backoff={sleep_for:.1f}s")
                    time.sleep(sleep_for)
                    continue
                return last_status, last_text
        except Exception as e:
            # TypeError mht. headers + netværksfejl → hurtig retry med jitter
            sleep_for = (1.2 ** attempt) + random.uniform(0, 0.5)
            log.debug(f"http_get error {e!r} on {u} – retry in {sleep_for:.1f}s")
            time.sleep(sleep_for)
    return last_status, last_text

def _host_of(u: str) -> str:
    try:
        return urlsplit(u).netloc.lower()
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

def _fetch_with_playwright_sync(url: str, pre_html: Optional[str] = None, pre_status: Optional[int] = None) -> str:
    """Renderer side med delt Playwright-klient (singleton).
    Indeholder status/_needs_js logging, timeout og circuit-breaker.
    Accepterer evt. pre_html/pre_status for at undgå ekstra GET.
    """
    if AsyncHtmlClient is None:
        return ""
    import asyncio
    async def _go(u: str, given_html: Optional[str], given_status: Optional[int]) -> str:
        global _SHARED_PW_CLIENT, _PW_FAILS_BY_HOST, _PW_FAILS_MAX, _PW_DISABLED_HOSTS
        # Brug pre_html/pre_status hvis givet; ellers billig GET
        status = given_status
        pre_html = given_html
        if status is None or pre_html is None:
            s2, h2 = http_get(u, timeout=10.0)
            if status is None:
                status = s2
            if pre_html is None:
                pre_html = h2
        try:
            needs = _needs_js(pre_html or "")
        except Exception:
            needs = False
        from urllib.parse import urlsplit as _us
        _host = _us(u).netloc.lower()
        log.debug(f"PW gate for {u} -> status={status}, needs_js={needs}, pw_disabled_host={_host in _PW_DISABLED_HOSTS}")
        # 1) Drop PW på ≠200
        if status != 200:
            log.debug(f"Springer Playwright over (status {status}) for {u}")
            return pre_html or ""
        # 2) Drop PW når siden ikke kræver JS
        if not needs:
            log.debug(f"Playwright ikke nødvendig (statisk HTML) for {u}")
            return pre_html
        # 3) Circuit-breaker (per host)
        if _host in _PW_DISABLED_HOSTS:
            log.debug(f"Playwright er midlertidigt deaktiveret for host={_host} (circuit-breaker). Bruger pre_html.")
            return pre_html
        # 4) Kør PW med hård timeout
        if _SHARED_PW_CLIENT is None:
            _SHARED_PW_CLIENT = AsyncHtmlClient(stealth=True)
            await _SHARED_PW_CLIENT.startup()
        try:
            # Forsøg med målrettede selector-waits for kontakt/teamsider
            _wait_selectors = [
                "a[href^='mailto:']",
                "a[href^='tel:']",
                ".__cf_email__",
                "[data-cfemail]",
                ".team", ".staff", ".employee", ".elementor-team-member"
            ]
            try:
                html = await asyncio.wait_for(
                    _SHARED_PW_CLIENT.get_raw_html(
                        u,
                        force_playwright=True,
                        wait_for_selectors=_wait_selectors # nogle klienter understøtter dette
                    ),
                    timeout=12.0
                )
            except TypeError:
                # Fallback hvis klienten ikke kender argumentet
                html = await asyncio.wait_for(
                    _SHARED_PW_CLIENT.get_raw_html(u, force_playwright=True),
                    timeout=12.0
                )
            if isinstance(html, dict):
                html = html.get("html") or ""
            # reset kun for denne host
            _PW_FAILS_BY_HOST[_host] = 0
            return html or (pre_html or "")
        except Exception as e:
            _PW_FAILS_BY_HOST[_host] = _PW_FAILS_BY_HOST.get(_host, 0) + 1
            log.debug(f"Playwright fejl ({_PW_FAILS_BY_HOST[_host]}/{_PW_FAILS_MAX}) for host={_host}, url={u}: {e!r}")
            if _PW_FAILS_BY_HOST[_host] >= _PW_FAILS_MAX:
                _PW_DISABLED_HOSTS.add(_host)
                log.warning(f"Deaktiverer midlertidigt Playwright-eskalering for host={_host} (for mange fejl).")
            return pre_html
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    return loop.run_until_complete(_go(url, pre_html, pre_status))



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
    "people", "staff", "ledelse", "about", "om", "om-os",
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
_CONTACT_CLASS_RE = re.compile(r'class=["\'][^"\']*(team|staff|employee|person|medarbejder|medarbejdere)[^"\']*["\']', re.I)
_CONTACT_H_RE     = re.compile(r'<h[12][^>]*>\s*(kontakt|team|medarbejdere|personale|ansatte|about|om\s+os)\s*</h[12]>', re.I)
_MAILTO_TEL_RE    = re.compile(r'href=["\'](?:mailto:|tel:)', re.I)
_CONTACT_LINK_RE  = re.compile(r'href=["\'][^"\']*(/kontakt|/kontakt-os|/contact|/contact-us|/about|/om-?os|/team|/medarbejdere)[^"\']*["\']', re.I)
_JS_MARKERS_RE    = re.compile(
    r'(id=["\']__next["\']|data-reactroot|ng-app|vue-root|elementor-|wp-json|hydration|astro-island)',
    re.I
)

def _needs_js(html: str) -> bool:
    """Kræv JS kun ved tynd DOM (<1800 tegn) og tydelige JS-markører, medmindre kontakt-signaler findes."""
    if not html:
        return False
    lower = html.lower()
    # --- Kontakt-signaler (hurtig short-circuit) ---
    # 1) mailto/tel
    if _MAILTO_TEL_RE.search(lower):
        return False
    # 2) team/medarbejder-klasser
    if _CONTACT_CLASS_RE.search(lower):
        return False
    # 3) H1/H2 med kontakt-/team-ord
    if _CONTACT_H_RE.search(lower):
        return False
    # 4) Links mod kontakt/om/team/medarbejdere
    if _CONTACT_LINK_RE.search(lower):
        return False
    # --- JS-markører: kun hvis ingen kontakt-signaler ---
    body_len = len(re.sub(r"\s+", "", html))
    return (body_len < 1800) and bool(_JS_MARKERS_RE.search(html))

def _is_contactish_url(u: str) -> bool:
    p = (urlsplit(u).path or "").strip("/").lower()
    return any(p.endswith(sl) or f"/{sl}/" in f"/{p}/" for sl in CONTACTISH_SLUGS)

def _is_initials_like(local: str) -> bool:
    letters = re.sub(r"[^a-zæøå]", "", local.lower())
    return 2 <= len(letters) <= 4

def _needs_js(html: str) -> bool:
    """Kræv JS kun ved tynd DOM (<1800 tegn) og tydelige JS-markører."""
    if not html:
        return False
    body_len = len(re.sub(r"\s+", "", html))
    return (body_len < 1800) and bool(_JS_MARKERS_RE.search(html))

_PROFILE_PATTERNS = [
    r'/medarbejder(?:e)?/[\w\-]+',
    r'/team/[\w\-]+',
    r'/people/[\w\-]+',
    r'/staff/[\w\-]+',
    r'/profile/[\w\-]+',
    r'/ansat(?:te)?/[\w\-]+',
    r'/personale/[\w\-]+',
]

def _discover_profile_links(base_url: str, html: str, max_links: int = 40) -> list[str]:
    """Find medarbejder-/profil-URLs i HTML – nu med prioritering på anker-tekst og “contactish” hints."""
    # === ANKER: PROFILE_DISCOVERY_PRIORITIZE ===
    scored: list[tuple[int, str]] = []
    seen_raw: set[str] = set()

    # 1) Gå via <a ...> for at kunne score på anker-tekst
    for m in re.finditer(r'<a\s+[^>]*href=["\']([^"\']+)["\'][^>]*>(.*?)</a>', html, flags=re.I | re.S):
        href = (m.group(1) or "").strip()
        text = re.sub(r"<[^>]+>", " ", m.group(2) or "").lower()
        # Kun interne links
        absu = _abs_url(base_url, href)
        if not absu or not _same_host(base_url, absu):
            continue
        path = (urlsplit(absu).path or "").lower()

        # Matcher et profil-mønster?
        if not any(re.search(pat, path, flags=re.I) for pat in _PROFILE_PATTERNS):
            continue

        # Basisscore
        score = 1
        # Boost hvis anker-tekst ligner kontakt/person
        if any(kw in text for kw in ("kontakt", "contact", "email", "mail", "telefon", "phone",
                                     "team", "staff", "medarbejder", "medarbejdere", "personale", "profil", "ansat", "people")):
            score += 2
        # Boost hvis stien selv er “contactish”
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

def discover_candidates(base_url: str, base_html: str, max_urls: int = 5) -> list[str]:
    """Returner interne kandidatsider i prioriteret rækkefølge (Home → Sitemap → WP)."""
    seen: set[str] = set()
    ranked: list[tuple[int, str]] = []

    # 1) Home-link scoring
    lang = _detect_lang_from_html(base_html or "")
    terms = _contactish_terms(lang)
    for m in re.finditer(r'<a\s+[^>]*href=["\']([^"\']+)["\'][^>]*>(.*?)</a>', base_html, flags=re.I | re.S):
        href = (m.group(1) or "").strip()
        text = re.sub(r"<[^>]+>", " ", (m.group(2) or "")).lower().strip()
        abs_u = urljoin(base_url, href)
        if not _same_host(base_url, abs_u):
            continue
        score = 0
        u_low = abs_u.lower()
        for t in terms:
            if t in text:
                score += 4
            if t.replace(" ", "-") in u_low:
                score += 3
        if score:
            ranked.append((score, abs_u))
            seen.add(abs_u)

    # 2) robots.txt → sitemap
    st, robots = http_get(urljoin(base_url, "/robots.txt"))
    if st == 200:
        sitemaps = [ln.split(":", 1)[1].strip() for ln in robots.splitlines() if ln.lower().startswith("sitemap:")]
        for sm in sitemaps[:2]:
            st2, xml = http_get(sm, timeout=8.0, retries=2)
            if st2 != 200:
                continue
            for loc in re.findall(r"<loc>(.*?)</loc>", xml, flags=re.I):
                if not _same_host(base_url, loc):
                    continue
                if any(t in loc.lower() for t in (t.replace(" ", "-") for t in terms)):
                    if loc not in seen:
                        ranked.append((3, loc))
                        seen.add(loc)

    # 3) WordPress REST (hvis tilgængelig)
    st_wp, _ = http_get(urljoin(base_url, "/wp-json/"), timeout=5.0, retries=1)
    if st_wp == 200:
        for t in ["kontakt", "om", "about", "team", "medarbejdere", "staff", "management"]:
            api = urljoin(base_url, f"/wp-json/wp/v2/pages?search={t}&_fields=link,title&per_page=5")
            stp, body = http_get(api, timeout=7.0, retries=1)
            if stp == 200:
                try:
                    for item in json.loads(body):
                        link = item.get("link")
                        if link and _same_host(base_url, link) and link not in seen:
                            ranked.append((4, link))
                            seen.add(link)
                except Exception:
                    pass

    ranked.sort(key=lambda x: x[0], reverse=True)
    return [u for _, u in ranked[:max_urls]]
# <<< ANKER SLUT: DISCOVERY_HELPERS

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
    "om", "om os", "about", "about us",
    "team", "teams", "vores team", "vores-hold", "holdet",
    "medarbejder", "medarbejdere", "ansatte", "personale",
    "people", "staff",
    "ledelse", "management", "board",
    "organisation", "company"
}

DEFAULT_TIMEOUT = 12.0  # sekunder

#def _safe_get(url: str, timeout: float = DEFAULT_TIMEOUT) -> tuple[int, str]:
#    return http_get(u)

#    if url in _GET_CACHE:
#        return _GET_CACHE[url]
#    status, html = 0, ""
#    host = _host_of(url)
#    # ANKER: RATE_LIMIT_BACKOFF — respekter backoff pr. host
#    try:
#        until = _RATE_LIMIT_HOSTS.get(host, 0.0)
#        now = time.time()
#        if until and now < until:
#            time.sleep(min(10.0, until - now) + random.uniform(0, 0.5))
#    except Exception:
#        pass
#
#    try:
#        if httpx:
#            ae = _accept_encoding() if callable(_accept_encoding) else _accept_encoding
#            if not isinstance(ae, (str, bytes)) or (isinstance(ae, str) and not ae.strip()):
#                ae = "gzip, deflate, br"
#            with httpx.Client(follow_redirects=True, timeout=timeout, headers={
#                "User-Agent": "VextoContactFinder/1.1",
#                "Accept-Encoding": ae,
#            }) as c:
#                r = c.get(url)
#                status = r.status_code
#                html = r.text or ""
#        elif requests:
#            r = requests.get(url, allow_redirects=True, timeout=timeout)  # type: ignore
#            status, html = r.status_code, r.text or ""
#        # 429 → backoff for host
#        if status == 429:
#            _RL_429_COUNTS[host] += 1
#            _RATE_LIMIT_HOSTS[host] = time.time() + 60.0 + random.uniform(0, 5.0)
#    except Exception:
#        status, html = 0, ""
#    _GET_CACHE[url] = (status, html)
#    return status, html

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

# ------------------------------- Utils ----------------------------------------

def _collapse_ws(s: str | None) -> str | None:
    if not s:
        return None
    return re.sub(r"\s+", " ", str(s)).strip()

def _is_plausible_name(s: str | None) -> bool:
    """1–4 tokens, kapitaliserede, ingen tal/@/breadcrumb-tegn, ikke ALL CAPS.
    Afviser CTA/sektionstitler (kontakt, om os, osv.).
    # ANKER: NAME_VALIDATION_RELAXED
    """
    if not s:
        return False
    s = _collapse_ws(s) or ""
    if not s:
        return False
    low = s.lower()
    # CTA/sektion-blacklist for at undgå falske navne
    _NAME_BLACKLIST = {
        "kontakt","kontakt os","contact","contact us","om","om os","about","about us",
        "ring til os","skriv til os","book","bestil","vores team","team","har du spørgsmål",
        "find medarbejder","læs mere","kundeservice","personale","medarbejdere"
    }
    if low in _NAME_BLACKLIST:
        return False
    if re.search(r"[,/|>@]|\d", s):
        return False
    if s.upper() == s:
        return False
    parts = s.split(" ")
    if not (1 <= len(parts) <= 4):
        return False
    return all(re.fullmatch(NAME_TOKEN, p) for p in parts)

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
    # Bevar original case i output
    t = _collapse_ws(title) or ""
    t = t.strip(",;:-—")
    # Fjern UI-ord (case-insensitive)
    pat = r"\b(" + "|".join(map(re.escape, UI_TITLE_BLACKLIST)) + r")\b"
    t2 = re.sub(pat, "", t, flags=re.I)
    t2 = re.sub(r"\s{2,}", " ", t2).strip(",;:-— ").strip()
    return t2 or None

def _resolve_redirect(u: str, timeout: float = DEFAULT_TIMEOUT) -> str:
    """Returnér endelig URL efter redirects (HEAD med follow_redirects).
    Fallback: returnér u uændret.
    """
    try:
        import httpx
        with httpx.Client(follow_redirects=True, timeout=timeout, headers={"User-Agent": "VextoContactFinder/1.0"}) as c:
            r = c.head(u)
            return str(r.url) if r is not None and r.url is not None else u
    except Exception:
        return u

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

@lru_cache(maxsize=256)
def _detect_site_lang(base_url: str, timeout: float = DEFAULT_TIMEOUT) -> str:
    """Returnér 'da'/'en'/'' baseret på <html lang> eller simple ord-heuristikker."""
    status, html = _safe_get(base_url, timeout=timeout)
    if status == 0 or not html:
        return ""
    m = re.search(r"<html[^>]*lang=['\"]([a-zA-Z-]+)['\"]", html, flags=re.I)
    if m:
        return m.group(1).lower().split("-")[0]
    da_hits = sum(1 for w in ("kontakt", "om os", "tilbage", "forside") if w in html.lower())
    en_hits = sum(1 for w in ("contact", "about us", "back", "home") if w in html.lower())
    if da_hits > en_hits:
        return "da"
    if en_hits > da_hits:
        return "en"
    return ""

def _fetch_robots_txt(base_url: str, timeout: float = DEFAULT_TIMEOUT) -> tuple[list[str], list[str]]:
    """
    Returnér (sitemaps, disallows) fra robots.txt.
    Kun simpel parsing: 'Sitemap:' og 'Disallow:' linjer.
    """
    try:
        host = _host_of(base_url)
        # P1: genbrug _ROBOTS_CACHE pr. host
        if host in _ROBOTS_CACHE:
            status, text = _ROBOTS_CACHE[host]
        else:
            sp = urlsplit(base_url)
            robots_url = f"{sp.scheme}://{sp.netloc}/robots.txt"
            status, text = http_get(robots_url, timeout=timeout)
            _ROBOTS_CACHE[host] = (status, text or "")
        if status == 0 or not text:
            return [], []
        sitemaps, disallows = [], []
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
        return sitemaps, disallows
    except Exception:
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
def _fetch_sitemap_urls(sitemap_url: str, timeout: float = DEFAULT_TIMEOUT) -> list[str]:
    """Hent sitemap (xml eller .gz) og returnér liste af <loc>-URLs."""
    urls: list[str] = []
    try:
        import httpx, io, gzip as _gz
        status, text = http_get(sitemap_url, timeout=timeout)
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
def _is_whitelisted_mismatch(target_url: str) -> bool:
    try:
        ap = _apex((urlsplit(target_url).hostname or "").lower())
        return ap in DOMAIN_MISMATCH_WHITELIST
    except Exception:
        return False
# === ANKER: DOMAIN_MISMATCH_WHITELIST ===

def _passes_identity_gate(
    name: str | None,
    emails: list[str] | None,
    title: str | None,
    url: str | None = None,
    dom_distance: Optional[int] = None,
) -> bool:
    """A: (plausibelt navn + email-match>=2) ELLER
       B: (plausibelt navn + rolle-titel) ELLER
       C: (plausibelt navn + e-mail med samme domæne som sitet + ikke-generisk local + tæt DOM)
       D: (kontakt/om/team-side + e-mail på eget domæne + ikke-generisk local + DOM-afstand ≤ 1) — navn/titel kan mangle"""
    has_plausible_name = _is_plausible_name(name)

    # A
    if has_plausible_name and _best_email_score(emails or [], name) >= 2:
        return True
    # B
    t = _sanitize_title(title)
    if has_plausible_name and t and _looks_like_role(t):
        return True
    # C
    if has_plausible_name and url and dom_distance is not None and dom_distance <= 1:
        for em in (emails or []):
            try:
                local = em.split("@", 1)[0]
            except Exception:
                continue
            if not _is_generic_local(local) and _email_domain_matches_site(em, url):
                return True

    # D (kontakt-/om-/team-sider – kræv 2 stærke signaler når domæne matcher)
    if url and _is_contactish_url(url) and dom_distance is not None and dom_distance <= 1:
        for em in (emails or []):
            try:
                local = em.split("@", 1)[0]
            except Exception:
                continue
            if not _email_domain_matches_site(em, url):
                continue
            letters = re.sub(r"[^a-zæøå]", "", local.lower())
            has_personal_local = (len(letters) >= 3) and (not _is_generic_local(local))
            has_role = _looks_like_role(title)
            has_name_email_match = _is_plausible_name(name) and (_email_name_score(em, name) >= 1)
            # Kræv (personlig local) + (rolle ELLER navn↔email match)
            if has_personal_local and (has_role or has_name_email_match):
                return True
    # === ANKER: GATE_CONTACT_TWO_SIGNALS ===

    return False

def _dedup_key(c: ContactCandidate) -> tuple:
    """Sammenflet-nøgle: (navn, stærk email) ellers (navn, url, source)."""
    name = (c.name or "").strip().lower()
    for em in c.emails:
        try:
            local, dom = em.split("@", 1)
        except Exception:
            continue
        if not _is_generic_local(local):
            return (name, f"{local}@{dom}")
    return (name, c.url.strip().lower(), c.source)

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
    cp = _cache_path_for(url, cache_dir)
    cached = _read_cache(cp, max_age_hours)
    if cached:
        return cached

    ae = _accept_encoding() if callable(_accept_encoding) else _accept_encoding  # kan være str/callable
    if not isinstance(ae, (str, bytes)) or (isinstance(ae, str) and not ae.strip()):
        ae = "gzip, deflate, br"
    headers = {
        "User-Agent": "VextoContactFinder/1.0 (+https://vexto.io)",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Encoding": ae,
}
    if httpx is not None:
        with httpx.Client(http2=True, timeout=timeout, headers=headers, follow_redirects=True) as cli:
            r = cli.get(url)
            r.raise_for_status()
            text = r.text
            _write_cache(cp, text)
            return text
    if 'requests' in sys.modules:
        r = requests.get(url, headers=headers, timeout=timeout)  # type: ignore
        r.raise_for_status()
        text = r.text
        _write_cache(cp, text)
        return text
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


def _harvest_identity_from_container(node, max_depth: int = 7) -> tuple[Optional[str], Optional[str], list[str]]:
    """
    Find nærmeste 'kort/sektion' omkring node, og ekstrahér (name, title, phones).
    - Navn findes før titel (så navne ikke ender som titler).
    - Understøtter Elementor-kort/overskrifter.
    - Case-promotion for helt lowercase navne i kort.
    """
    if not node:
        return None, None, []

    def _is_container(n) -> bool:
        try:
            if hasattr(n, "attributes"):
                cls = (n.attributes.get("class", "") + " " + n.attributes.get("id", "")).lower()
            elif hasattr(n, "get"):
                cls = (" ".join(n.get("class", []) or []) + " " + (n.get("id") or "")).lower()
            else:
                return False
            keys = ("team","staff","employee","person","card","profile","grid","column","section",
                    "administration","tilbud","entreprise",
                    "elementor-team-member","elementor-widget","elementor-column","elementor-section","elementor-container")
            return any(k in cls for k in keys)
        except Exception:
            return False

    container = node
    depth = 0
    while depth < max_depth and getattr(container, "parent", None):
        if _is_container(container):
            break
        container = getattr(container, "parent", None)
        depth += 1
    if not container:
        container = getattr(node, "parent", None) or node

    name = None
    title = None
    phones: list[str] = []

    def _node_text(n) -> str:
        try:
            if hasattr(n, "itertext"):
                return " ".join(t.strip() for t in n.itertext() if t and t.strip())
        except Exception:
            pass
        try:
            if hasattr(n, "stripped_strings"):
                return " ".join(list(n.stripped_strings))
        except Exception:
            pass
        return ""

    # 1) NAVN (headings + semantiske + Elementor + udbredte klasser)
    try:
        selectors = (
            "h1,h2,h3,h4,"
            ".name,.person-name,.staff-name,.member-name,.employee-name,.bio-name,.team-member__name,"
            "[itemprop='name'],[role='heading'],.elementor-heading-title,.elementor-widget-heading"
        )
        if hasattr(container, "css"):
            for el in container.css(selectors):
                raw = _collapse_ws(el.text())
                if _is_plausible_name(raw):
                    name = raw
                    break
                if not name:
                    promo = _maybe_promote_name_case(raw)
                    if promo and _is_plausible_name(promo):
                        name = promo
                        break
                    if raw and raw.isupper():
                        cap = " ".join(w[:1].upper() + w[1:].lower() for w in raw.split())
                        if _is_plausible_name(cap):
                            name = cap
                            break
        elif hasattr(container, "select"):
            for el in container.select(selectors):
                raw = _collapse_ws(el.get_text(" ", strip=True))
                if _is_plausible_name(raw):
                    name = raw
                    break
                if not name:
                    promo = _maybe_promote_name_case(raw)
                    if promo and _is_plausible_name(promo):
                        name = promo
                        break
                    if raw and raw.isupper():
                        cap = " ".join(w[:1].upper() + w[1:].lower() for w in raw.split())
                        if _is_plausible_name(cap):
                            name = cap
                            break
    except Exception:
        pass
    # === ANKER: NAME_SELECTORS_EXTENDED ===

    # 2) TITEL (først stærke selectorer; undgå at tage navne som titler)
    try:
        title_selectors = "[itemprop='jobTitle'], .job-title, .title, .role, .position, .position-title, .team-member__role, .bio-title"
        if hasattr(container, "css"):
            for el in container.css(title_selectors):
                tt_raw = _collapse_ws(el.text())
                if _is_plausible_name(tt_raw):
                    if not name:
                        name = tt_raw
                    continue
                if not name:
                    promo = _maybe_promote_name_case(tt_raw)
                    if promo and _is_plausible_name(promo):
                        name = promo
                        continue
                tt = _sanitize_title(tt_raw)
                if tt and not _is_plausible_name(tt):
                    title = tt
                    break
            if not title:
                for el in container.css("p,span"):
                    txt_raw = _collapse_ws(el.text())
                    if _is_plausible_name(txt_raw):
                        if not name:
                            name = txt_raw
                        continue
                    if not name:
                        promo = _maybe_promote_name_case(txt_raw)
                        if promo and _is_plausible_name(promo):
                            name = promo
                            continue
                    tt = _sanitize_title(txt_raw)
                    if tt and not _is_plausible_name(tt):
                        title = tt
                        break
        elif hasattr(container, "select"):
            for el in container.select(title_selectors):
                tt_raw = _collapse_ws(el.get_text(" ", strip=True))
                if _is_plausible_name(tt_raw):
                    if not name:
                        name = tt_raw
                    continue
                if not name:
                    promo = _maybe_promote_name_case(tt_raw)
                    if promo and _is_plausible_name(promo):
                        name = promo
                        continue
                tt = _sanitize_title(tt_raw)
                if tt and not _is_plausible_name(tt):
                    title = tt
                    break
            if not title:
                for el in container.select("p,span"):
                    txt_raw = _collapse_ws(el.get_text(" ", strip=True))
                    if _is_plausible_name(txt_raw):
                        if not name:
                            name = txt_raw
                        continue
                    if not name:
                        promo = _maybe_promote_name_case(txt_raw)
                        if promo and _is_plausible_name(promo):
                            name = promo
                            continue
                    tt = _sanitize_title(txt_raw)
                    if tt and not _is_plausible_name(tt):
                        title = tt
                        break
    except Exception:
        pass
    # === ANKER: TITLE_SELECTORS_EXTENDED ===


    # 3) PHONES (tel: + tekstnumre m. anti-CVR)
    try:
        if hasattr(container, "css"):
            for t in container.css("a[href^='tel:']"):
                pn = _normalize_phone(t.attributes.get("href",""))
                if pn: phones.append(pn)
        elif hasattr(container, "select"):
            for t in container.select("a[href^='tel:']"):
                pn = _normalize_phone(t.get("href",""))
                if pn: phones.append(pn)
    except Exception:
        pass

    blob = _node_text(container)
    for m in re.finditer(r"(?:\+?\d[\s\-\(\)\.]{0,3}){8,}", blob):
        sidx,eidx = m.start(), m.end()
        ctx = blob[max(0, sidx-20):min(len(blob), eidx+20)].lower()
        if "cvr" in ctx:
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

            out.append(ContactCandidate(
                name=name_h, title=title_h, emails=[em], phones=phones_h,
                source="mailto", url=url, dom_distance=dist,
                hints={"near": near[:180], "harvested": True}
            ))


        # Cloudflare-obfuskerede e-mails
        for cf in html_or_tree.css(".__cf_email__"):
            em = _normalize_email(_cf_decode(cf.attributes.get("data-cfemail")))
            if not em:
                continue
            container = cf.parent or cf
            text_blob = _near_text(container, 260)

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
                hints={"near": text_blob[:180], "cfemail": True}
            ))
        return out

    # ---------------- BeautifulSoup ----------------
    if BeautifulSoup is not None and hasattr(html_or_tree, "select"):
        for a in html_or_tree.select("a[href^='mailto:']"):
            href = a.get("href", "")
            em = _normalize_email(href)
            if not em:
                continue

            name_h, title_h, phones_h = _harvest_identity_from_container(a)
            near = " ".join(list(a.parent.stripped_strings))[:260] if a.parent else ""

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

            out.append(ContactCandidate(
                name=name_h, title=title_h, emails=[em], phones=phones_h or [],
                source="mailto", url=url, dom_distance=dist,
                hints={"near": near[:180], "harvested": True}
            ))

        # Cloudflare __cf_email__
        for cf in html_or_tree.select(".__cf_email__"):
            em = _normalize_email(_cf_decode(cf.get("data-cfemail")))
            if not em:
                continue
            container = cf.parent or cf
            near = " ".join(list(container.stripped_strings))[:260] if container else ""

            name = None
            m_name = re.search(rf"({_NAME_WINDOW_REGEX()})", near)
            if m_name:
                name = _collapse_ws(m_name.group(1))
            if not _is_plausible_name(name):
                name = _find_heading_name(cf) or name

            title = None
            m_title = re.search(r"(?i)\b([A-Za-zÆØÅæøå/\- ]{3,80})\b", near)
            if m_title:
                title = _sanitize_title(m_title.group(1))

            out.append(ContactCandidate(
                name=name, title=title, emails=[em], phones=[],
                source="dom", url=url, dom_distance=0,
                hints={"near": near[:180], "cfemail": True}
            ))

        # tel: links (BS4)
        for t in html_or_tree.select("a[href^='tel:']"):
            href = t.get("href", "")
            pn = _normalize_phone(href)
            if not pn:
                continue
            container = t.parent or t
            near = " ".join(list(container.stripped_strings))[:260] if container else ""

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
                name=name, title=title, emails=[], phones=[pn],
                source="dom", url=url, dom_distance=0,
                hints={"near": near[:180], "tel": True}
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
            m2 = re.search(r"(?i)\b([A-Za-zÆØÅæøå/\- ]{3,80})\b", text_blob)
            if m2:
                title_h = _sanitize_title(m2.group(1))

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
    emails_on_domain: list[str] = []
    phones: list[str] = []

    seen_emails: set[str] = set()
    seen_phones: set[str] = set()

    re_email = r"[A-Z0-9._%+\-]{1,64}@[A-Z0-9.\-]{1,255}\.[A-Z]{2,}"

    def _site_tld(url: str) -> str:
        try:
            host = (urlsplit(url).hostname or "").lower()
            return host.rsplit(".", 1)[-1]
        except Exception:
            return ""

    def _is_dk_phone(pn: str) -> bool:
        return bool(re.fullmatch(r"\+45\d{8}", pn))

    def _last8(pn: str) -> str:
        return re.sub(r"\D", "", pn)[-8:]

    for u, html in pages_html:
        plain = re.sub(r"<[^>]+>", " ", html)

        # E-mails i synlig tekst
        for m in re.finditer(re_email, plain, flags=re.I):
            em = _normalize_email(m.group(0))
            if not em:
                continue
            local = em.split("@", 1)[0]
            if local in {"noreply", "no-reply", "donotreply"}:
                continue
            if _email_domain_matches_site(em, base_url) and em not in seen_emails:
                seen_emails.add(em)
                emails_on_domain.append(em)

        # Cloudflare (data-cfemail)
        for m in re.finditer(r'data-cfemail=["\']([0-9a-fA-F]+)["\']', html):
            em = _normalize_email(_cf_decode(m.group(1)))
            if not em:
                continue
            local = em.split("@", 1)[0]
            if local in {"noreply", "no-reply", "donotreply"}:
                continue
            if _email_domain_matches_site(em, base_url) and em not in seen_emails:
                seen_emails.add(em)
                emails_on_domain.append(em)

        # Telefonnumre i synlig tekst (anti-CVR i nærkontekst)
        for m in re.finditer(r"(?:\+?\d[\s\-\(\)\.]{0,3}){8,}", plain):
            sidx, eidx = m.start(), m.end()
            ctx = plain[max(0, sidx-15):min(len(plain), eidx+15)].lower()
            if "cvr" in ctx:
                continue
            pn = _normalize_phone(m.group(0))
            if pn and pn not in seen_phones:
                seen_phones.add(pn)
                phones.append(pn)

        # Telefonnumre i tel:-links
        for m in re.finditer(r'href=["\']tel:([^"\']+)["\']', html, flags=re.I):
            pn = _normalize_phone(m.group(1))
            if pn and pn not in seen_phones:
                seen_phones.add(pn)
                phones.append(pn)

    # Dedupliker telefoner på “sidste 8 cifre” (samme nr. m. forskellig formattering)
    uniq_by_last8 = []
    seen_last8 = set()
    for pn in phones:
        k = _last8(pn)
        if k and k not in seen_last8:
            seen_last8.add(k)
            uniq_by_last8.append(pn)
    phones = uniq_by_last8

    # Ved .dk-domæner – foretræk danske telefonnumre (+45XXXXXXXX)
    if _site_tld(base_url) == "dk":
        dk_only = [p for p in phones if _is_dk_phone(p)]
        if dk_only:
            phones = dk_only

    if not emails_on_domain and not phones:
        return []

    return [{
        "name": None,
        "title": None,
        "emails": emails_on_domain[:3],
        "phones": phones[:3],
        "score": -1.0,
        "reasons": ["GENERIC_ORG_CONTACT"],
        "source": "page-generic",
        "url": base_url,
        "dom_distance": None,
        "hints": {"identity_gate_bypassed": True},
    }]


def _extract_from_rdfa(url: str, tree) -> list[ContactCandidate]:
    out: list[ContactCandidate] = []
    if not tree or not hasattr(tree, "css"):
        return out
    for el in tree.css('[typeof*="Person" i], [typeof*="schema:Person" i]'):
        name = None; email = None; tel = None; title = None
        for prop in el.css('[property]'):
            p = (prop.attributes.get('property') or '').lower()
            txt = _collapse_ws(prop.text()) if hasattr(prop, "text") else None
            if not p:
                continue
            if 'name' in p and txt:
                name = txt
            elif 'email' in p and txt:
                email = _normalize_email(txt)
            elif 'tel' in p or 'phone' in p or 'telephone' in p:
                tel = _normalize_phone(txt)
            elif 'jobtitle' in p or 'role' in p:
                title = txt
        out.append(ContactCandidate(
            name=name, title=title,
            emails=[e for e in [email] if e],
            phones=[p for p in [tel] if p],
            source="rdfa", url=url, dom_distance=0, hints={"selector": "RDFa"}
        ))
    return out

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
            # selectolax
            if hasattr(node, "text"):
                return _collapse_ws(node.text(deep=True)) or ""
        except Exception:
            pass
        # bs4
        try:
            return _collapse_ws(node.get_text(separator="\n")) or ""
        except Exception:
            return ""

    # Vælg containere der ofte rummer kort
    containers = []
    css_roots = [
        # WordPress core blocks
        ".wp-block-columns", ".wp-block-group", ".team", ".team-grid",
        ".is-layout-grid", ".is-layout-flow", ".is-layout-flex",
        # Elementor layouts
        ".elementor-section", ".elementor-container", ".elementor-row",
        ".elementor-column", ".elementor-widget", ".elementor-widget-container",
        ".elementor-image-box", ".elementor-team-member",
        # Generic/andre CMS
        ".team-container", ".staff-list", ".employee-grid", ".profile-card", ".bio-card",
        ".team-members", ".our-team", ".people", ".staff", ".person-list", ".members", ".employees",
        ".team-member__list",
        # Fald tilbage
        "section", "main", "article"
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

    # Udtræk "kort" pr. container
    seen: set[tuple[str, str]] = set()
    for cont in containers:
        cards = []
        try:
            if hasattr(cont, "css"):
                cards = cont.css(
                    "figure, article, "
                    ".wp-block-column, .wp-block-media-text, .team-member, .wp-block-group > div, .is-layout-flow > div, "
                    ".elementor-column, .elementor-widget, .elementor-widget-container, .elementor-image-box, .elementor-team-member, "
                    ".staff-card, .employee, .employee-card, .profile-card, .bio-card, "
                    ".member, .person, .staff-item, .team-item, .member-card, .people__item, .team-member__card, .card, .card-body"
                )
    # === ANKER: STAFF_GRID_CARD_SELECTORS_EXTENDED ===
            elif hasattr(cont, "select"):
                cards = cont.select(
                    "figure, article, "
                    ".wp-block-column, .wp-block-media-text, .team-member, .wp-block-group > div, .is-layout-flow > div, "
                    ".elementor-column, .elementor-widget, .elementor-widget-container, .elementor-image-box, .elementor-team-member, "
                    ".staff-card, .employee, .employee-card, .profile-card, .bio-card"
                ) or cont.select("div, figure, article")
        except Exception:
            cards = []

        for card in cards or []:
            raw = _get_text(card)
            if not raw or len(raw) < 5:
                continue

            # Lav flere linjer til at afgøre navn/titel
            if hasattr(card, "get_text"):
                text_multiline = card.get_text(separator="\n")
            else:
                try:
                    text_multiline = "\n".join([n.text().strip() for n in card.iter() if n.text() and n.text().strip()])
                except Exception:
                    text_multiline = raw

            # Prøv direkte name/role klasser før heuristik
            direct_name = None
            direct_title = None
            try:
                if hasattr(card, "css"):
                    dn = card.css_first(".person-name,.staff-name,.member-name,.employee-name,.bio-name,.team-member__name,.name")
                    dr = card.css_first(".job-title,.role,.position,.position-title,.team-member__role,.bio-title,.title")
                    if dn: direct_name = _collapse_ws(dn.text())
                    if dr: direct_title = _sanitize_title(_collapse_ws(dr.text()))
                elif hasattr(card, "select"):
                    dn = card.select_one(".person-name,.staff-name,.member-name,.employee-name,.bio-name,.team-member__name,.name")
                    dr = card.select_one(".job-title,.role,.position,.position-title,.team-member__role,.bio-title,.title")
                    if dn: direct_name = _collapse_ws(dn.get_text(" ", strip=True))
                    if dr: direct_title = _sanitize_title(_collapse_ws(dr.get_text(" ", strip=True)))
            except Exception:
                pass

            name, title = None, None
            if _is_plausible_name(direct_name):
                name = direct_name
                title = direct_title or None
            else:
                name, title = _extract_lines_from_block_txt(text_multiline)
            if not name:
                continue

            key = (name.lower(), (title or "").lower())
            if key in seen:
                continue
            seen.add(key)

            rs = _role_strength(title)
            # fjernet ubrugt 'base'-variabel; gem rolle-styrke i hints i stedet
            out.append(ContactCandidate(
                name=name,
                title=title,
                emails=[],
                phones=[],
                source="dom-staff-grid",
                url=url,
                dom_distance=None,
                hints={"card": raw[:160], "role_strength": rs}
            ))
    return out


def _wp_json_enrich(page_url: str, html: str, timeout: float, cache_dir: Optional[Path]) -> list[ContactCandidate]:
    """
    WP-JSON fallback: find <link ... href=".../wp-json/wp/v2/pages/<id>"> or ?p=<id>,
    fetch content.rendered, parse again for staff-grid/mailto/microformats.
    """
    out: list[ContactCandidate] = []
    # 1) Find API-URLs in HTML
    # === ANKER: WP_JSON_BUILD_URLS BEGIN ===
    api_urls: list[str] = []
    # Direct page-API URLs from HTML
    for m in re.finditer(r'href=["\']([^"\']+/wp-json/wp/v2/pages/\d+[^"\']*)["\']', html, flags=re.I):
        api_urls.append(m.group(1))
    # Origin + www-variant
    try:
        s = urlsplit(page_url)
        origin = f"{s.scheme}://{s.netloc}"
        bases = [origin]
        if not s.netloc.startswith("www."):
            bases.append(f"{s.scheme}://www.{s.netloc}")
    except Exception:
        bases = []
    # Slug from URL path
    try:
        parts = [p for p in urlsplit(page_url).path.split("/") if p]
        slug = parts[-1] if parts else None
    except Exception:
        slug = None
    if slug:
        for b in bases:
            api_urls.extend([
                f"{b}/wp-json/wp/v2/pages?slug={slug}&_embed=1",
                f"{b}/wp-json/wp/v2/posts?slug={slug}&_embed=1",
            ])
    # Users + types + search
    for b in bases:
        api_urls.append(f"{b}/wp-json/wp/v2/users?per_page=100&context=view&_embed=1")
        api_urls.append(f"{b}/wp-json/wp/v2/types")
        for q in ["team", "medarbejder", "personale", "ansatte", "staff", "people", "medarbetare", "personal"]:
            api_urls.append(f"{b}/wp-json/wp/v2/search?search={q}&per_page=50&_embed=1")
    api_urls = list(dict.fromkeys(api_urls))
    # === ANKER: WP_JSON_BUILD_URLS END ===
    # === ANKER: WP_JSON_PROCESS_LOOP BEGIN ===
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
                        for b in bases:
                            api_urls.append(f"{b}/wp-json/wp/v2/{rest_base or slug_t}?per_page=100&_embed=1")
                api_urls = list(dict.fromkeys(api_urls))
                continue
            # B) WP search → follow 'url'/'link' and parse HTML
            if isinstance(data, list) and data and isinstance(data[0], dict) and ("url" in data[0] or "link" in data[0]):
                for it in data[:20]:
                    pgurl = it.get("url") or it.get("link")
                    if not pgurl:
                        continue
                    st2, html2 = http_get(pgurl, timeout=timeout)
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
        cache_dir: Optional[Path] = ".http_diskcache/html",
        use_browser: str = "auto", # {"auto","always","never"}
        gdpr_minimize: bool = False,
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
        self._pw_budget: int = 3 # maks. render pr. kørsel (kan tunes)

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
        if AsyncHtmlClient is None:
            return None

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
                        await client.shutdown()

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
        key = _norm(url)
        # Lokal per-run cache
        cached = self._html_cache_local.get(key)
        if cached is not None:
            return cached
        html = ""
        # 1) Prøv hurtig HTTP (httpx/requests via eksisterende _fetch_text)
        with contextlib.suppress(Exception):
            html = _fetch_text(url, self.timeout, self.cache_dir)
        # 1b) Hvis vi kan se 404 i HTML, så drop Playwright og returnér
        if html and _looks_404(html):
            self._html_cache_local[key] = html or ""
            return html or ""
        # 2) Vurdér om vi bør render’e (politik: use_browser = "never" | "auto" | "always")
        # === ANKER: PW_POLICY_DECISION_BEGIN ===
        should_render = False
        if self.use_browser == "always":
            should_render = True
        elif self.use_browser == "auto":
            lh = (html.lower() if html else "")
            # NYT: brug _is_contactish_url + _needs_js (fanger fx "JavaScript er nødvendig...")
            should_render = _is_contactish_url(url) or (not html) or ("elementor" in lh) or _needs_js(html or "")
        # "never" -> False
        # === ANKER: PW_POLICY_DECISION_END ===
        # 2b) Render aldrig hvis URL svarer 404 (brug http_get for tom HTML)
        if should_render and not html:
            with contextlib.suppress(Exception):
                status, _ = http_get(url, timeout=self.timeout)
                if status and 400 <= status < 500:
                    should_render = False # skip 4xx
        if should_render and key not in self._rendered_once:
            # Debounce + budget
            if key in self._pw_attempted or self._pw_budget <= 0:
                self._html_cache_local[key] = html or ""
                return html or ""
            self._pw_attempted.add(key)
            self._pw_budget -= 1
            pre_status = 200 if html and not _looks_404(html) else None
            rendered = _fetch_with_playwright_sync(url, pre_html=html or None, pre_status=pre_status)
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

    def _pages_to_try(self, url: str, limit_pages: int = 4) -> list[str]:
        pages = [url] + [self._abs(url, p) for p in FALLBACK_PATHS]
        # dedup bevar rækkefølge
        seen = set()
        ordered = []
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
                out.extend(_wp_json_enrich(url, html, timeout=self.timeout, cache_dir=self.cache_dir))

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

        # Nulstil PW-budget og -debounce pr. domæne/run
        self._pw_budget = 3
        self._pw_attempted = set()
        self._rendered_once = set()

        # 1) Hent forsiden
        htmls: dict[str, str] = {}
        try:
            root_html = self._fetch_text_smart(url)
            htmls[url] = root_html
        except Exception:
            root_html = ""
            self._diag(url, "NO_HTML_FETCHED", {"status": 0})
            return []

        # 2) Domain-mismatch guard
        try:
            resolved_root, _st0 = _head_and_resolve(url, timeout=self.timeout)
        except Exception:
            resolved_root = url
        if not _same_apex(url, resolved_root) and not _is_whitelisted_mismatch(resolved_root):
            log.warning("Domain-mismatch guard: base=%s -> final=%s (stopper person-scrape)", url, resolved_root)
            return [{
                "name": None,
                "title": None,
                "emails": [],
                "phones": [],
                "score": 0.2,
                "reasons": ["GENERIC_ORG_CONTACT", "DOMAIN_MISMATCH"],
                "source": "page-generic",
                "url": url,
            }]

        final_site_url = resolved_root or url

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

        # 4) Kandidat-sider (Home → Sitemap → WP)
        candidates = discover_candidates(url, root_html, max_urls=min(5, limit_pages))
        candidates = _sticky_host_urls(url, root_html, candidates)

        # 5) Hent kandidat-sider
        for cu in candidates:
            if len(cands) >= 6:  # Cap på kontakter
                break
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
        if _needs_js(root_html) and AsyncHtmlClient is not None and self._pw_budget > 0:
            try:
                rendered = _fetch_with_playwright_sync(url, pre_html=root_html, pre_status=200 if root_html else None)
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

        # 9) Final output
        cleaned = []
        seen_keys: set[tuple[str, str]] = set()
        for sc in scored:
            if sc.score < 0:
                continue
            reasons = set(sc.reasons or [])
            is_generic = "GENERIC_ORG_CONTACT" in reasons
            if "GATE_FAIL" in reasons and not is_generic:
                continue
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
            log.info("CF SUMMARY host=%s final=%s items=%d emails=%d persons=%d pw=%d time=%.3fs",
                     host, final_host, len(cleaned), emails_any, persons_any, pw_renders, time.time()-t0)
        except Exception:
            pass

        return cleaned
    # <<< ANKER SLUT: FIND_ALL_ORCHESTRATION



    def find(self, url: str, limit_pages: int = 4) -> Optional[dict]:
        """Returnér bedste kandidat som dict – eller None."""
        all_ = self.find_all(url, limit_pages=limit_pages)
        return all_[0] if all_ else None


# ------------------------------ Offentligt API --------------------------------

def find_best_contact(url: str, limit_pages: int = 4) -> Optional[dict]:
    return ContactFinder().find(url, limit_pages=limit_pages)

# Kompat-aliaser (gamle kald i projektet)
def find_contacts(url: str, limit_pages: int = 4) -> list[dict]:
    return ContactFinder().find_all(url, limit_pages=limit_pages)

def extract_contacts(url: str, limit_pages: int = 4) -> list[dict]:
    return ContactFinder().find_all(url, limit_pages=limit_pages)

def extract_best_contact(url: str, limit_pages: int = 4) -> Optional[dict]:
    return ContactFinder().find(url, limit_pages=limit_pages)

def find_top_contact(url: str, limit_pages: int = 4) -> Optional[dict]:
    return ContactFinder().find(url, limit_pages=limit_pages)


# -------------------------- Test/utility shims --------------------------------

def run_enrichment_on_dataframe(
    df,
    url_col: str | None = None,
    *,
    limit_pages: int = 4,
    top_n: int = 1,
    gdpr_minimize: bool = False,
    use_browser: str = "auto",
):
    """
    DataFrame-helper (robust):
    - Finder URL-kolonnen case-insensitivt + heuristik (også "Hjemmeside", "Website URL", "WWW" m.fl.)
    - Normaliserer URL-værdier (tilføjer https:// ved domæner uden schema)
    - Finder top-1 eller top-N kontakter pr. række
    - Lægger kolonner: cf_best_* og cf_all (JSON)
    """
    import json, re
    import pandas as _pd

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
        s = (str(v or "")).strip()
        if not s or s.lower() == "nan":
            return ""
        # hvis der er mellemrum/kommaer – tag første token der ligner en URL/domæne
        s = re.split(r"[,\s;]+", s)[0]
        if re.match(r"^https?://", s, flags=re.I):
            return s
        # hvis 'www.' i starten → tilføj https://
        if s.lower().startswith("www."):
            return f"https://{s}"
        # hvis det ligner et domæne → tilføj https://
        if re.match(r"^([a-z0-9\-]+\.)+[a-z]{2,}(/.*)?$", s, flags=re.I):
            return f"https://{s}"
        return s  # returnér som er (giver evt. 0-resultat senere)

    # --- Find URL-kolonne (case-insensitiv + heuristik) -----------------
    real_url_col = _guess_url_col(df, url_col)
    if not real_url_col:
        candidates = ["url", "website", "web", "domain", "hjemmeside", "site", "company_url",
                      "homepage","home_page","website_url","website address","www"]
        raise ValueError(f"Kunne ikke finde URL-kolonne. Prøv en af (case-insensitivt): {candidates}")

    cf = ContactFinder(gdpr_minimize=gdpr_minimize, use_browser=use_browser)

    best_names, best_titles, best_emails, best_phones, best_scores, best_urls, all_json = \
        [], [], [], [], [], [], []

    for _, row in df.iterrows():
        raw = row.get(real_url_col)
        url = _norm_url_value(raw)
        if not url:
            best_names.append(None); best_titles.append(None); best_emails.append([])
            best_phones.append([]); best_scores.append(None); best_urls.append(None); all_json.append("[]")
            continue

        # Prøv at hente direktørnavne fra DF-rækken, hvis kolonne findes
        directors_list: list[str] = []
        try:
            for col in df.columns:
                lc = col.lower()
                if "direkt" in lc or "director" in lc:
                    raw_dn = row.get(col)
                    if raw_dn and str(raw_dn).strip() and str(raw_dn).strip().lower() != "nan":
                        # split på ; , / |
                        parts = re.split(r"[;,/|]+", str(raw_dn))
                        directors_list = [p.strip() for p in parts if p and p.strip()]
                        break
        except Exception:
            directors_list = []

        if top_n and top_n > 1:
            results = cf.find_all(url, limit_pages=limit_pages, directors=directors_list or None)[:top_n]
        else:
            # Brug find_all (top-1) når vi har direktør-hints; ellers find()
            if directors_list:
                results = cf.find_all(url, limit_pages=limit_pages, directors=directors_list)[:1]
            else:
                best = cf.find(url, limit_pages=limit_pages)
                results = [best] if best else []

        if results:
            b = results[0]
            best_names.append(b.get("name"))
            best_titles.append(b.get("title"))
            best_emails.append(b.get("emails") or [])
            best_phones.append(b.get("phones") or [])
            best_scores.append(b.get("score"))
            best_urls.append(b.get("url") or url)
            all_json.append(json.dumps(results, ensure_ascii=False))
        else:
            best_names.append(None); best_titles.append(None); best_emails.append([])
            best_phones.append([]); best_scores.append(None); best_urls.append(url); all_json.append("[]")

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

