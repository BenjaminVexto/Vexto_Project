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
from dataclasses import dataclass
from datetime import datetime, timedelta
from functools import lru_cache
from pathlib import Path
from urllib.parse import urljoin, urlsplit
from typing import Any, Iterable, Optional
from vexto.scoring.http_client import _accept_encoding, should_check_link_status as _http_should_check_link_status

# Setup log directory
log_dir = Path(__file__).resolve().parents[2] / "logs"
log_dir.mkdir(parents=True, exist_ok=True)

log_file = log_dir / "contact_finder.log"

# RotatingFileHandler: 5 MB pr. fil, behold 5 filer
handler = logging.handlers.RotatingFileHandler(
    log_file, maxBytes=5_000_000, backupCount=5, encoding="utf-8"
)
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
handler.setFormatter(formatter)

log = logging.getLogger("vexto.contact_finder")
log.setLevel(logging.DEBUG)
log.addHandler(handler)
# valgfrit: behold konsol kun på WARNING+
console = logging.StreamHandler()
console.setLevel(logging.WARNING)
console.setFormatter(formatter)
log.addHandler(console)

try:
    # Kræver at AsyncHtmlClient findes i jeres http_client
    from vexto.scoring.http_client import AsyncHtmlClient
except Exception:
    AsyncHtmlClient = None  # type: ignore

_SHARED_PW_CLIENT = None

# --- Playwright circuit-breaker ---
_PW_FAILS = 0
_PW_FAILS_MAX = 3  # efter 3 fejl i træk stopper vi midlertidigt PW-eskalering
_PW_DISABLED = False

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

def _is_contact_like(u: str) -> bool:
    p = (u or "").lower()
    return any(k in p for k in (
        "/kontakt", "/contact", "/kontakt-os", "/contact-us",
        "/om", "/about", "/om-os", "/about-us",
        "/team", "/medarbejder", "/people", "/staff"
    ))

def _fetch_with_playwright_sync(url: str) -> str:
    """Renderer side med delt Playwright-klient (singleton).
    Indeholder status/_needs_js logging, timeout og circuit-breaker.
    """
    if AsyncHtmlClient is None:
        return ""

    import asyncio

    async def _go(u: str) -> str:
        global _SHARED_PW_CLIENT, _PW_FAILS, _PW_FAILS_MAX, _PW_DISABLED

        # Billig GET før PW: find status + pre_html til beslutning
        status, pre_html = _safe_get(u)
        needs = False
        try:
            needs = _needs_js(pre_html)
        except Exception:  # defensivt
            needs = False

        log.debug(f"PW gate for {u} -> status={status}, needs_js={needs}, pw_disabled={_PW_DISABLED}")

        # 1) Drop PW på ≠200
        if status != 200:
            log.debug(f"Springer Playwright over (status {status}) for {u}")
            return pre_html

        # 2) Drop PW når siden ikke kræver JS
        if not needs:
            log.debug(f"Playwright ikke nødvendig (statisk HTML) for {u}")
            return pre_html

        # 3) Circuit-breaker
        if _PW_DISABLED:
            log.debug("Playwright er midlertidigt deaktiveret (circuit-breaker). Bruger pre_html.")
            return pre_html

        # 4) Kør PW med hård timeout
        if _SHARED_PW_CLIENT is None:
            _SHARED_PW_CLIENT = AsyncHtmlClient(stealth=True)
            await _SHARED_PW_CLIENT.startup()

        try:
            html = await asyncio.wait_for(
                _SHARED_PW_CLIENT.get_raw_html(u, force_playwright=True),
                timeout=12.0
            )
            if isinstance(html, dict):
                html = html.get("html") or ""
            _PW_FAILS = 0
            return html or pre_html
        except Exception as e:
            _PW_FAILS += 1
            log.debug(f"Playwright fejl ({_PW_FAILS}/{_PW_FAILS_MAX}) for {u}: {e!r}")
            if _PW_FAILS >= _PW_FAILS_MAX:
                _PW_DISABLED = True
                log.warning("Deaktiverer midlertidigt Playwright-eskalering (for mange fejl).")
            return pre_html

    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    return loop.run_until_complete(_go(url))



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
UI_TITLE_BLACKLIST = {
    "kontakt", "kontakt os", "contact", "contact us",
    "team", "om", "about",
    "support", "help", "helpdesk",
    "mail", "email", "e-mail", "phone", "telefon",
    "info", "adresse", "address",
    "åbningstider", "opening hours",
    "faq", "privacy", "cookies", "cookie", "policy",
}

CONTACTISH_SLUGS = (
    "kontakt", "contact", "kontakt-os", "contact-us",
    "team", "teams", "medarbejder", "medarbejdere",
    "people", "staff", "ledelse", "about", "om", "om-os"
)

def _is_contactish_url(u: str) -> bool:
    p = (urlsplit(u).path or "").strip("/").lower()
    return any(p.endswith(sl) or f"/{sl}/" in f"/{p}/" for sl in CONTACTISH_SLUGS)

def _is_initials_like(local: str) -> bool:
    letters = re.sub(r"[^a-zæøå]", "", local.lower())
    return 2 <= len(letters) <= 4

def _needs_js(html: str) -> bool:
    """Grov indikator for JS-renderet indhold."""
    if not html:
        return False
    # typisk Elementor placeholder + ingen kontaktlinks
    if "JavaScript er nødvendig" in html or "elementor" in html:
        return True
    if not re.search(r'href=["\']mailto:|href=["\']tel:', html, flags=re.I):
        return True
    return False

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
    """Find medarbejder-/profil-URLs i HTML (regex-baseret, hurtig)."""
    urls: list[str] = []
    for pat in _PROFILE_PATTERNS:
        for m in re.finditer(rf'href=["\']({pat})["\']', html, flags=re.I):
            try:
                full = urljoin(base_url, m.group(1))
                if _same_host(base_url, full):
                    urls.append(full)
            except Exception:
                continue
    # dedup + cap
    out, seen = [], set()
    for u in urls:
        if u not in seen:
            out.append(u); seen.add(u)
        if len(out) >= max_links:
            break
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

# Generiske e-mail-brugere
GENERIC_EMAIL_USERS = {
    "info","kontakt","mail","sales","support","hello","office",
    "billing","invoice","noreply","no-reply","donotreply","admin",
}

# Simpel navnetoken (kapitaliseret, med danske bogstaver)
NAME_TOKEN = r"[A-ZÆØÅ][a-zA-ZÀ-ÖØ-öø-ÿ'’-]{1,30}"

# Sider vi prøver udover root
FALLBACK_PATHS = (
    "/contact", "/contact/", "/contact-us",
    "/kontakt", "/kontakt/", "/kontakt-os",
    "/om", "/om/", "/om-os", "/om-os/",
    "/about", "/about/", "/about-us",
    "/team", "/team/", "/teams", "/vores-team",
    "/medarbejder", "/medarbejder/", "/medarbejdere", "/medarbejdere/",
    "/ansatte", "/personale", "/people", "/staff",
    "/ledelse", "/management", "/board", "/organisation"
)

# --- Fast HEAD status cache + URL-normalisering ---
from functools import lru_cache

def _same_host(a: str, b: str) -> bool:
    try:
        sa, sb = urlsplit(a), urlsplit(b)
        return sa.scheme == sb.scheme and sa.netloc == sb.netloc
    except Exception:
        return False

@lru_cache(maxsize=256)
def _cheap_head_status(u: str, timeout: float = 8.0) -> int:
    """HEAD med follow_redirects og kort timeout. Cached pr. normaliseret URL."""
    try:
        import httpx
        ae = _accept_encoding() if callable(_accept_encoding) else _accept_encoding  # type: ignore[misc]
        if not isinstance(ae, (str, bytes)) or (isinstance(ae, str) and not ae.strip()):
            ae = "gzip, deflate, br"
        with httpx.Client(follow_redirects=True, timeout=timeout,
                          headers={"User-Agent": "VextoContactFinder/1.0", "Accept-Encoding": ae}) as c:
            r = c.head(u)
            return r.status_code
    except Exception:
        return 0

def _norm(u: str) -> str:
    return _norm_url_for_fetch(u)

def _looks_danish(html: str) -> bool:
    return bool(re.search(r"\b(kontakt|om os|åbningstider|find os|vi hjælper)\b", (html or "").lower()))

def _localized_fallback_paths(base_url: str, home_html: str) -> tuple[str, ...]:
    """Returnér sprog-tilpassede fallback-stier for kandidatplanen."""
    host = urlsplit(base_url).netloc.lower()
    dk_like = host.endswith(".dk")
    da = dk_like or _looks_danish(home_html)
    if da:
        return (
            "/kontakt", "/kontakt/", "/kontakt-os", "/om", "/om/", "/om-os", "/om-os/",
            "/team", "/team/", "/teams", "/medarbejder", "/medarbejder/", "/medarbejdere",
            "/medarbejdere/", "/ansatte", "/personale", "/ledelse"
        )
    else:
        return (
            "/contact", "/contact/", "/contact-us", "/about", "/about/", "/about-us",
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

def _safe_get(url: str, timeout: float = DEFAULT_TIMEOUT) -> tuple[int, str]:
    """Billig GET m. httpx; returnér (status, text)."""
    try:
        import httpx
        # Accept-Encoding kan være str eller callable; sikre en streng
        ae = _accept_encoding() if callable(_accept_encoding) else _accept_encoding  # type: ignore[misc]
        if not isinstance(ae, (str, bytes)) or (isinstance(ae, str) and not ae.strip()):
            ae = "gzip, deflate, br"

        with httpx.Client(
            follow_redirects=True,
            timeout=timeout,
            headers={"User-Agent": "VextoContactFinder/1.0", "Accept-Encoding": ae},
        ) as c:
            r = c.get(url)
            return r.status_code, (r.text or "")
    except Exception as e:
        log.debug(f"_safe_get fejl for {url}: {e!r}")
        return 0, ""

__all__ = [
    "ContactFinder",
    "find_best_contact",
    # aliaser
    "find_contacts",
    "extract_contacts",
    "extract_best_contact",
    "find_top_contact",
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

# ------------------------------- Utils ----------------------------------------

def _collapse_ws(s: str | None) -> str | None:
    if not s:
        return None
    return re.sub(r"\s+", " ", str(s)).strip()

def _is_plausible_name(s: str | None) -> bool:
    """2–4 tokens, kapitaliserede, ingen tal/@/breadcrumb-tegn, ikke ALL CAPS."""
    if not s:
        return False
    s = _collapse_ws(s) or ""
    if not s:
        return False
    if re.search(r"[,/|>@]|\d", s):
        return False
    if s.upper() == s:
        return False
    parts = s.split(" ")
    if not (2 <= len(parts) <= 4):
        return False
    return all(re.fullmatch(NAME_TOKEN, p) for p in parts)

def _sanitize_title(title: str | None) -> str | None:
    if not title:
        return None
    t = _collapse_ws(title) or ""
    t = t.strip(",;:-—").lower()
    if t in UI_TITLE_BLACKLIST:
        return None
    # fjern UI-ord hvis de blot er støj
    pat = r"\b(" + "|".join(map(re.escape, UI_TITLE_BLACKLIST)) + r")\b"
    t2 = re.sub(pat, "", t)
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

def _http_head_status(url: str, timeout: float = 10.0) -> Optional[int]:
    """Let HEAD til at afgøre om ressourcen findes. Følger redirects."""
    try:
        import httpx
        with httpx.Client(follow_redirects=True, timeout=timeout) as c:
            r = c.head(url, headers={"User-Agent": "VextoContactFinder/1.0"})
            return r.status_code
    except Exception:
        return None

def _resolve_redirect(u: str, timeout: float = DEFAULT_TIMEOUT) -> str:
    """Returnér endelig URL efter redirects (HEAD). Fallback: u uændret."""
    try:
        import httpx
        with httpx.Client(follow_redirects=True, timeout=timeout) as c:
            r = c.head(u, headers={"User-Agent": "VextoContactFinder/1.0"})
            return str(r.url) if getattr(r, "url", None) else u
    except Exception:
        return u

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
        sp = urlsplit(base_url)
        robots_url = f"{sp.scheme}://{sp.netloc}/robots.txt"
        status, text = _safe_get(robots_url, timeout=timeout)
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

def _fetch_sitemap_urls(sitemap_url: str, timeout: float = DEFAULT_TIMEOUT) -> list[str]:
    """Hent sitemap (xml eller .gz) og returnér liste af <loc>-URLs."""
    urls: list[str] = []
    try:
        import httpx, io, gzip as _gz
        with httpx.Client(follow_redirects=True, timeout=timeout) as c:
            r = c.get(sitemap_url)
            if r.status_code != 200:
                return []
            data = r.content or b""
            if sitemap_url.lower().endswith(".gz"):
                with _gz.GzipFile(fileobj=io.BytesIO(data)) as gz:
                    data = gz.read()
            text = data.decode("utf-8", errors="ignore")
            for m in re.finditer(r"<loc>\s*([^<\s]+)\s*</loc>", text, flags=re.I):
                urls.append(m.group(1).strip())
    except Exception:
        return urls
    return urls

def _sitemap_discover_urls(base_url: str, timeout: float = DEFAULT_TIMEOUT) -> list[str]:
    """Find sitemap-URLs via robots.txt og returnér flad liste (samme host)."""
    sitemaps, _ = _fetch_robots_txt(base_url, timeout=timeout)
    found: list[str] = []
    for sm in sitemaps:
        found.extend(_fetch_sitemap_urls(sm, timeout=timeout) or [])
    return [u for u in found if _same_host(base_url, u)]



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

def _email_domain_matches_site(email: str, site_url: str) -> bool:
    try:
        ed = email.split("@", 1)[1].lower()
        host = (urlsplit(site_url).hostname or "").lower()
        if host.startswith("www."):
            host = host[4:]
        return _apex(ed) == _apex(host)
    except Exception:
        return False

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

    # D (lempelse på kontakt-/om-/team-sider):
    if url and _is_contactish_url(url) and dom_distance is not None and dom_distance <= 1:
        for em in (emails or []):
            try:
                local = em.split("@", 1)[0]
            except Exception:
                continue
            if not _email_domain_matches_site(em, url):
                continue
            # NYT: kræv mindst EN af disse på kontakt/om/team:
            #  - ikke-generisk local med >=3 bogstaver (ikke bare 2-4 initialer), ELLER
            #  - der findes en titel der ligner en rolle
            letters = re.sub(r"[^a-zæøå]", "", local.lower())
            if (len(letters) >= 3 and not _is_generic_local(local)) or _looks_like_role(title):
                return True

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

    headers = {
        "User-Agent": "VextoContactFinder/1.0 (+https://vexto.io)",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Encoding": _accept_encoding(),   # ⬅️ samme anker
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
    """Saml tekst i node + nærområde (fungerer for selectolax og BS4)."""
    try:
        # selectolax
        if HTMLParser is not None and isinstance(node, HTMLParser.Node):  # type: ignore[attr-defined]
            texts = []
            texts.extend(t.strip() for t in node.itertext() if t and t.strip())
            parent = node.parent
            if parent:
                sibs = list(parent.iter())
                for sib in sibs[:8]:
                    if sib is node:
                        continue
                    with contextlib.suppress(Exception):
                        txt = " ".join(t.strip() for t in sib.itertext() if t and t.strip())
                        if txt:
                            texts.append(txt)
            out = " ".join(texts)
            return (out[:max_chars] + "…") if len(out) > max_chars else out

        # bs4
        if BeautifulSoup is not None and hasattr(node, "stripped_strings"):
            parent = node.parent
            texts = []
            if parent:
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

def _extract_from_mailtos(url: str, html_or_tree) -> list[ContactCandidate]:
    out: list[ContactCandidate] = []
    # selectolax
    if HTMLParser is not None and isinstance(html_or_tree, HTMLParser):
        for a in html_or_tree.css("a[href^='mailto:']"):
            href = a.attributes.get("href", "")
            em = _normalize_email(href)
            if not em:
                continue
            near = _near_text(a, 260)
            name = None
            m_name = re.search(rf"({_NAME_WINDOW_REGEX()})", near)
            if m_name:
                name = _collapse_ws(m_name.group(1))
            title = None
            m_title = re.search(r"(?i)\b([A-Za-zÆØÅæøå/\- ]{3,80})\b", near)
            if m_title:
                title = _sanitize_title(m_title.group(1))
            dist = 0
            try:
                dist = _dom_distance(a, a.parent or a) or (0 if name else 1)
            except Exception:
                dist = 0 if name else 1
            out.append(ContactCandidate(
                name=name, title=title, emails=[em], phones=[],
                source="mailto", url=url, dom_distance=dist,
                hints={"near": near[:180]}
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
            title = None
            m2 = re.search(r"(?i)\b([A-Za-zÆØÅæøå/\- ]{3,80})\b", text_blob)
            if m2:
                title = _sanitize_title(m2.group(1))
            dist = _dom_distance(cf, container) or (0 if name else 1)
            out.append(ContactCandidate(
                name=name, title=title, emails=[em], phones=[],
                source="dom", url=url, dom_distance=dist, hints={"near": text_blob[:180], "cfemail": True}
            ))
        return out
    
    # bs4
    if BeautifulSoup is not None and hasattr(html_or_tree, "select"):
        
        # mailto:
        for a in html_or_tree.select("a[href^='mailto:']"):
            href = a.get("href", "")
            em = _normalize_email(href)
            if not em:
                continue
            near = " ".join(list(a.parent.stripped_strings))[:260] if a.parent else ""
            name = None
            m_name = re.search(rf"({_NAME_WINDOW_REGEX()})", near)
            if m_name:
                name = _collapse_ws(m_name.group(1))
            title = None
            m_title = re.search(r"(?i)\b([A-Za-zÆØÅæøå/\- ]{3,80})\b", near)
            if m_title:
                title = _sanitize_title(m_title.group(1))
            dist = 0
            try:
                dist = _dom_distance(a, a.parent or a) or (0 if name else 1)
            except Exception:
                dist = 0 if name else 1
            out.append(ContactCandidate(
                name=name, title=title, emails=[em], phones=[],
                source="mailto", url=url, dom_distance=dist,
                hints={"near": near[:180]}
            ))
        
        # Cloudflare __cf_email__ (BS4)
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
            title = None
            m_title = re.search(r"(?i)\b([A-Za-zÆØÅæøå/\- ]{3,80})\b", near)
            if m_title:
                title = _sanitize_title(m_title.group(1))
            out.append(ContactCandidate(
                name=name, title=title, emails=[em], phones=[],
                source="dom", url=url, dom_distance=0, hints={"near": near[:180], "cfemail": True}
            ))
        
        # tel: links med nær-navn/titel (BS4 fallback)
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
            title = None
            m_title = re.search(r"(?i)\b([A-Za-zÆØÅæøå/\- ]{3,80})\b", near)
            if m_title:
                title = _sanitize_title(m_title.group(1))
            out.append(ContactCandidate(
                name=name, title=title, emails=[], phones=[pn],
                source="dom", url=url, dom_distance=0, hints={"near": near[:180], "tel": True}
            ))

        return out

    # ren regex fallback
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

    # Microdata Person
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
        out.append(ContactCandidate(
            name=name, title=title,
            emails=[e for e in [email] if e], phones=[p for p in [tel] if p],
            source="dom", url=url, dom_distance=0, hints={"selector": "Person microdata"}
        ))

    # Cards: anchor mailto med nær-navn/titel
    for a in html_or_tree.css("a[href^='mailto:']"):
        em = _normalize_email(a.attributes.get("href", ""))
        if not em:
            continue
        container = a.parent or a
        text_blob = _near_text(container, 260)
        name = None
        m = re.search(rf"({_NAME_WINDOW_REGEX()})", text_blob)
        if m:
            name = _collapse_ws(m.group(1))
        title = None
        m2 = re.search(r"(?i)\b([A-Za-zÆØÅæøå/\- ]{3,80})\b", text_blob)
        if m2:
            title = _sanitize_title(m2.group(1))
        dist = _dom_distance(a, container) or (0 if name else 1)
        out.append(ContactCandidate(
            name=name, title=title, emails=[em], phones=[],
            source="dom", url=url, dom_distance=dist, hints={"near": text_blob[:180]}
        ))
    
    for t in html_or_tree.css("a[href^='tel:']"):
        pn = _normalize_phone(t.attributes.get("href", ""))
        if not pn:
            continue
        container = t.parent or t
        text_blob = _near_text(container, 260)
        name = None
        m = re.search(rf"({_NAME_WINDOW_REGEX()})", text_blob)
        if m:
            name = _collapse_ws(m.group(1))
        title = None
        m2 = re.search(r"(?i)\b([A-Za-zÆØÅæøå/\- ]{3,80})\b", text_blob)
        if m2:
            title = _sanitize_title(m2.group(1))
        dist = _dom_distance(t, container) or (0 if name else 1)
        out.append(ContactCandidate(
            name=name, title=title, emails=[], phones=[pn],
            source="dom", url=url, dom_distance=dist, hints={"near": text_blob[:180], "tel": True}
        ))
    return out

def _extract_from_text_emails(url: str, html: str) -> list[ContactCandidate]:
    """
    Fald tilbage: find e-mails i ren tekst (inkl. 'navn [at] domæne.dk')
    og forsøg at parre med navn/rolle i nærheden.
    """
    out: list[ContactCandidate] = []

    # Lav en "flad" tekst uden tags for nærheds-søgning
    plain = re.sub(r"<[^>]+>", " ", html)
    plain = re.sub(r"\s+", " ", plain)

    # 1) Normal e-mail
    re_email = r"[A-Z0-9._%+\-]{1,64}@[A-Z0-9.\-]{1,255}\.[A-Z]{2,}"
    # 2) Obfuskeret 'at'/'snabela'
    re_obf = r"([A-Z0-9._%+\-]{1,64})\s*(?:\(|\[|\{)?\s*(?:@|at|snabela)\s*(?:\)|\]|\})?\s*([A-Z0-9.\-]{1,255}\.[A-Z]{2,})"

    emails: set[str] = set()

    for m in re.finditer(re_email, plain, flags=re.I):
        emails.add(m.group(0))

    for m in re.finditer(re_obf, plain, flags=re.I):
        emails.add(f"{m.group(1)}@{m.group(2)}")

    # For hver email, prøv at finde et navn/titel i et vindue på ±200 tegn
    for em in emails:
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
            name=name, title=title, emails=[_normalize_email(em)] if _normalize_email(em) else [],
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

        # Telefonnumre i synlig tekst
        for m in re.finditer(r"(?:\+?\d[\s\-\(\)\.]{0,3}){8,}", plain):
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

    # Find første plausible navnelinje
    name_idx = None
    for i, l in enumerate(lines):
        if _is_plausible_name(l):
            name_idx = i
            break
    if name_idx is None:
        return None, None

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
        # Generic containers
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
                    ".elementor-column, .elementor-widget, .elementor-widget-container, .elementor-image-box, .elementor-team-member"
                )
            elif hasattr(cont, "select"):
                cards = cont.select(
                    "figure, article, "
                    ".wp-block-column, .wp-block-media-text, .team-member, .wp-block-group > div, .is-layout-flow > div, "
                    ".elementor-column, .elementor-widget, .elementor-widget-container, .elementor-image-box, .elementor-team-member"
                ) or cont.select("div, figure, article")
        except Exception:
            cards = []

        for card in cards or []:
            raw = _get_text(card)
            if not raw or len(raw) < 5:
                continue

            # Lav flere linjer til at afgøre navn/titel
            # (bevarer tydeligere opdeling end én lang linje)
            if hasattr(card, "get_text"):
                text_multiline = card.get_text(separator="\n")
            else:
                # selectolax: simuler linjeskift mellem blokke
                try:
                    text_multiline = "\n".join([n.text().strip() for n in card.iter() if n.text() and n.text().strip()])
                except Exception:
                    text_multiline = raw

            name, title = _extract_lines_from_block_txt(text_multiline)
            if not name:
                continue

            key = (name.lower(), (title or "").lower())
            if key in seen:
                continue
            seen.add(key)

            rs = _role_strength(title)
            base = 40 + rs * 15  # “kort” uden email giver en høj baselinescore

            out.append(ContactCandidate(
                name=name,
                title=title,
                emails=[],
                phones=[],
                source="dom-staff-grid",
                url=url,
                dom_distance=None,
                hints={"card": raw[:160]}
            ))
    return out


def _wp_json_enrich(page_url: str, html: str, timeout: float, cache_dir: Optional[Path]) -> list[ContactCandidate]:
    """
    WP-JSON fallback: find <link ... href=".../wp-json/wp/v2/pages/<id>"> ELLER ?p=<id>,
    hent content.rendered, parse igen for staff-grid/mailto/microformats.
    """
    out: list[ContactCandidate] = []

    # 1) Find API-URL’er i HTML (ikke kun i headers)
    api_urls: list[str] = []
    for m in re.finditer(r'href=["\']([^"\']+/wp-json/wp/v2/pages/\d+[^"\']*)["\']', html, flags=re.I):
        api_urls.append(m.group(1))

    # 2) Alternativt: shortlink ?p=<id> → byg API-URL
    origin = None
    with contextlib.suppress(Exception):
        from urllib.parse import urlsplit
        s = urlsplit(page_url)
        origin = f"{s.scheme}://{s.netloc}"
    if not api_urls and origin:
        m = re.search(r'href=["\'][^"\']*\?p=(\d+)["\']', html, flags=re.I)
        if m:
            api_urls.append(f"{origin}/wp-json/wp/v2/pages/{m.group(1)}?_embed=1")

    if not api_urls:
        return out

    for api in api_urls[:2]:
        with contextlib.suppress(Exception):
            jtxt = _fetch_text(api, timeout=timeout, cache_dir=cache_dir, max_age_hours=6)
            data = json.loads(jtxt)
            if isinstance(data, dict):
                rendered = (data.get("content") or {}).get("rendered", "")
            elif isinstance(data, list) and data:
                rendered = (data[0].get("content") or {}).get("rendered", "")
            else:
                rendered = ""

            if not rendered:
                continue

            t2 = _parse_html(rendered)
            out.extend(_extract_from_staff_grid(page_url, t2))
            out.extend(_extract_from_mailtos(page_url, t2 if t2 is not None else rendered))
            out.extend(_extract_from_microformats(page_url, t2))

    # Slug-baseret fallback: /wp-json/wp/v2/pages?slug=<slug>
    origin = None
    slug = None
    with contextlib.suppress(Exception):
        from urllib.parse import urlsplit
        s = urlsplit(page_url)
        origin = f"{s.scheme}://{s.netloc}"
        parts = [p for p in s.path.split("/") if p]
        if parts:
            slug = parts[-1]
    if origin and slug:
        api_urls.append(f"{origin}/wp-json/wp/v2/pages?slug={slug}&_embed=1")
        api_urls.append(f"{origin}/wp-json/wp/v2/posts?slug={slug}&_embed=1")  # bare in case

    # dedup
    api_urls = list(dict.fromkeys(api_urls))

    for api in api_urls[:3]:
        with contextlib.suppress(Exception):
            jtxt = _fetch_text(api, timeout=timeout, cache_dir=cache_dir, max_age_hours=6)
            data = json.loads(jtxt)
            if isinstance(data, list) and data:
                data = data[0]
            if isinstance(data, dict):
                rendered = (data.get("content") or {}).get("rendered", "") or ""
            else:
                rendered = ""

            if not rendered:
                continue

            t2 = _parse_html(rendered)
            out.extend(_extract_from_staff_grid(page_url, t2))
            out.extend(_extract_from_mailtos(page_url, t2 if t2 is not None else rendered))
            out.extend(_extract_from_microformats(page_url, t2))
    
    return out
# <<< STAFF GRID + WP-JSON HELPERS END

# ------------------------------- Scoring --------------------------------------

def _score_candidate(c: ContactCandidate) -> ScoredContact:
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

    # Telefon?
    if any(_normalize_phone(p) for p in (c.phones or [])):
        s += 0.5
        why.append("PHONE")

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

    return ScoredContact(candidate=c, score=s, reasons=why)

# ------------------------------- Orkestrering ---------------------------------

class ContactFinder:
    """Henter 1–N sider, udtrækker kandidater, dedupliker, scorer og returnerer top-1."""

    def __init__(
        self,
        timeout: float = DEFAULT_TIMEOUT,
        parallel: int = 4,
        cache_dir: Optional[Path] = ".http_diskcache/html",
        use_browser: str = "auto",  # {"auto","always","never"}
    ):
        self.timeout = timeout
        self.parallel = max(1, int(parallel))
        self.cache_dir = Path(cache_dir) if cache_dir else None

        # ÉN kilde til sandhed for browser-politik
        self.use_browser = (use_browser or "auto").strip().lower()
        if self.use_browser not in {"auto", "always", "never"}:
            self.use_browser = "auto"

        # Per-kørsel HTML-cache (kun for denne instans)
        self._html_cache_local: dict[str, str] = {}
        # Normaliserede URLs som vi allerede har Playwright-renderet i denne kørsel
        self._rendered_once: set[str] = set()
        self._pw_attempted: set[str] = set()   # URLs forsøgt til PW (debounce)
        self._pw_budget: int = 3               # maks. render pr. kørsel (kan tunes)

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
        key = _norm_url_for_fetch(url)

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
        should_render = False
        if self.use_browser == "always":
            should_render = _is_contact_like(url)
        elif self.use_browser == "auto":
            lh = (html.lower() if html else "")
            should_render = _is_contact_like(url) or (not html) or ("elementor" in lh)
        # "never" -> False

        # 2b) Render aldrig hvis URL svarer 404 (billig HEAD for tom HTML)
        if should_render and not html:
            with contextlib.suppress(Exception):
                status = _http_head_status(url, timeout=self.timeout)
                if status and 400 <= status < 500:
                    should_render = False  # skip 4xx

        if should_render and key not in self._rendered_once:
            # Debounce + budget
            if key in self._pw_attempted or self._pw_budget <= 0:
                self._html_cache_local[key] = html or ""
                return html or ""

            self._pw_attempted.add(key)
            self._pw_budget -= 1

            rendered = _fetch_with_playwright_sync(url)
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
        bykey: dict[tuple, ContactCandidate] = {}
        for c in cands:
            k = _dedup_key(c)
            if k in bykey:
                bykey[k] = _merge(bykey[k], c)
            else:
                bykey[k] = c
        return list(bykey.values())

    def _score_sort(self, cands: Iterable[ContactCandidate]) -> list[ScoredContact]:
        scored = [_score_candidate(c) for c in cands]
        scored.sort(key=lambda sc: sc.score, reverse=True)
        return scored

    # ------------------------------- Sync API ---------------------------------

    def find_all(self, url: str, limit_pages: int = 4) -> list[dict]:
        """Returnér alle (scored) kandidater som dicts (reasons inkluderet)."""
        t0 = time.time()

        # 1) Hent forsiden først (så vi kan lave discovery)
        htmls: dict[str, str] = {}
        try:
            root_html = self._fetch_text_smart(url)
            htmls[url] = root_html
        except Exception:
            root_html = ""

        # 2) Byg liste over sider at prøve:
        # - discovery fra forsiden (kontakt/om/team mv.) -> PRIORITET
        # - + statiske fallback-stier
        discovered = _discover_internal_links(url, root_html, max_links=20) if root_html else []
        static_pages = [self._abs(url, p) for p in _localized_fallback_paths(url, root_html or "")]

        # ANKER: sikr at /kontakt og /om prøves tidligt, uanset discovery og limit
        priority_pages = [self._abs(url, p) for p in ("/kontakt", "/kontakt/", "/om", "/om/")]

        # Ny rækkefølge: root, PRIORITY, discovery, statics
        # 0) Robots + sitemap discovery (kun samme host) — én robots GET
        try:
            sitemaps, disallows = _fetch_robots_txt(url, timeout=self.timeout)
        except Exception:
            sitemaps, disallows = [], []
        sm_urls = []
        if sitemaps:
            try:
                for sm in sitemaps:
                    sm_urls.extend(_fetch_sitemap_urls(sm, timeout=self.timeout) or [])
                # behold kun samme host
                sm_urls = [u for u in sm_urls if _same_host(url, u)]
            except Exception as e:
                log.debug(f"Sitemap discovery fejl: {e!r}")
                sm_urls = []

        # 1) Byg rækkefølge med sitemap først (hvis fundet)
        pages_ordered = [u for u in pages_ordered if _http_should_check_link_status(u)]

        # 2) Resolve redirects + få status i ÉT kald (spar en HEAD-runde)
        _head_cache: dict[str, int] = {}
        resolved_unique: list[str] = []
        seen_resolved: set[str] = set()
        _head_and_resolve_fn = globals().get("_head_and_resolve", None)

        # HÅRD CAP mod HEAD-storm
        MAX_HEADS = 25
        heads_done = 0

        for u in pages_ordered:
            if heads_done >= MAX_HEADS:
                # ingen flere HEAD; stadig resolve redirects billigt
                ru = _resolve_redirect(u, timeout=self.timeout) or u
                st = None
            else:
                if callable(_head_and_resolve_fn):
                    ru, st = _head_and_resolve_fn(u, timeout=self.timeout)  # type: ignore[misc]
                else:
                    # Fallback i to trin
                    ru = _resolve_redirect(u, timeout=self.timeout) or u
                    st = _http_head_status(ru, timeout=self.timeout)
                heads_done += 1

            k = _norm_url_for_fetch(ru)
            if k not in seen_resolved:
                resolved_unique.append(ru)
                seen_resolved.add(k)
            if st is not None:
                _head_cache[ru] = st

        # 3) Sprogfilter FØR HEAD (undgå HEAD på stier, som alligevel fjernes)
        site_lang = _detect_site_lang(url, timeout=self.timeout)
        def _is_en_path(p: str) -> bool:
            pl = (urlsplit(p).path or "").lower()
            return any(x in pl for x in ("/contact", "/contact-us", "/about", "/about-us", "/people", "/staff", "/management", "/board"))
        def _is_da_path(p: str) -> bool:
            pl = (urlsplit(p).path or "").lower()
            return any(x in pl for x in ("/kontakt", "/kontakt-os", "/om", "/om-os", "/medarbejder", "/medarbejdere", "/ansatte", "/ledelse"))

        lang_pruned: list[str]
        if site_lang == "da":
            lang_pruned = [p for p in resolved_unique if not _is_en_path(p)]
        elif site_lang == "en":
            lang_pruned = [p for p in resolved_unique if not _is_da_path(p)]
        else:
            lang_pruned = resolved_unique

        # 4a) Robots Disallow + HEAD-cache på den prunede mængde
        def _head_status_cached(u: str) -> int | None:
            if u in _head_cache:
                v = _head_cache[u]; return v if v != -1 else None
            s = _http_head_status(u, timeout=self.timeout)
            _head_cache[u] = s if s is not None else -1
            return s

        pages_filtered: list[str] = []
        for u in lang_pruned:
            path = (urlsplit(u).path or "")
            if _is_disallowed(path, disallows):
                log.debug("Robots Disallow skip: %s", u); continue
            status = _head_status_cached(u)
            if status and 400 <= status < 500:
                log.debug("HEAD drop (%s): %s", status, u); continue
            pages_filtered.append(u)

        # 4b) Sprogfilter: drop engelske fallback-stier på 'da' sites (og omvendt)
        if site_lang == "da":
            def _is_en(p: str) -> bool:
                pl = (urlsplit(p).path or "").lower()
                return any(k in pl for k in ("/contact", "/contact-us", "/about", "/about-us", "/team", "/people", "/staff", "/management", "/board"))
            pages_filtered = [p for p in pages_filtered if not _is_en(p)]
        elif site_lang == "en":
            def _is_da(p: str) -> bool:
                pl = (urlsplit(p).path or "").lower()
                return any(k in pl for k in ("/kontakt", "/kontakt-os", "/om", "/om-os", "/medarbejder", "/medarbejdere", "/ansatte", "/ledelse"))
            pages_filtered = [p for p in pages_filtered if not _is_da(p)]

        # 5) Prioritér kontaktish + kort sti, og cap til limit
        pages_sorted = sorted(
            pages_filtered,
            key=lambda x: (0 if _is_contactish_url(x) else 1, len((urlsplit(x).path or "")))
        )
        # Final dedup + cap
        _seen = set()
        pages_dedup: list[str] = []
        for u in pages_sorted:
            k = _norm_url_for_fetch(u)
            if k not in _seen:
                pages_dedup.append(u)
                _seen.add(k)
        pages = pages_dedup[:max(1, limit_pages)]
        log.debug("Pages planned (%d): %s", len(pages), pages)



        # 3) Hent resten parallelt (root er evt. allerede hentet)
        def fetch(u: str) -> tuple[str, str]:
            try:
                if u in htmls:
                    return (u, htmls[u])
                h = self._fetch_text_smart(u)
                return (u, h)
            except Exception:
                return (u, "")

        if len(pages) > 1 and self.parallel > 1:
            with concurrent.futures.ThreadPoolExecutor(max_workers=self.parallel) as ex:
                for u, h in ex.map(fetch, pages):
                    htmls[u] = h
        else:
            for u in pages:
                if u not in htmls:
                    _, h = fetch(u)
                    htmls[u] = h

        # Saml navnløse initialer@domæne.dk fra kontakt/om/team til én organisationspost
        org_bucket_emails = []
        pruned = []
        for c in all_cands:
            if c.name or c.title:
                pruned.append(c); continue
            if _is_contactish_url(c.url) and c.emails:
                kept = []
                for em in c.emails:
                    local = em.split("@",1)[0]
                    if _email_domain_matches_site(em, c.url) and (_is_initials_like(local) or _is_generic_local(local)):
                        org_bucket_emails.append(em)
                    else:
                        kept.append(em)
                if kept:
                    c.emails = kept
                    pruned.append(c)
            else:
                pruned.append(c)

        if org_bucket_emails:
            pruned.append(ContactCandidate(
                name=None, title=None,
                emails=sorted(set(org_bucket_emails))[:5],
                phones=[], source="page-generic", url=url,
                dom_distance=0, hints={"bucketed": True}
            ))

        all_cands = pruned

        # 4) Udpak kandidater fra alle sider
        all_cands: list[ContactCandidate] = []
        for u, h in htmls.items():
            if not h or _looks_404(h):
                continue
            cands = self._extract_all(u, h)
            all_cands.extend(cands)

        # 5) Merge + dedup + score
        merged = self._merge_dedup(all_cands)
        scored = self._score_sort(merged)

        # 6) Returnér som dicts inkl. reasons (rå liste)
        out = [
            {
                "name": sc.candidate.name,
                "title": sc.candidate.title,
                "emails": sc.candidate.emails,
                "phones": sc.candidate.phones,
                "url": sc.candidate.url,
                "score": sc.score,
                "reasons": sc.reasons,
            }
            for sc in scored
        ]

        # 7) Output-clean: drop negative/GATE_FAIL og dedup pr. (email, url)
        cleaned: list[dict] = []
        seen_keys: set[tuple[str, str]] = set()

        for r in out:
            # Fjern støj
            if r.get("score", 0) < 0:
                continue
            reasons = set(r.get("reasons") or [])
            if "GATE_FAIL" in reasons:
                continue

            emails = r.get("emails") or []
            url_out = r.get("url") or ""

            # Dedup-logik: hvis der er emails, dedup pr. (email, url). Ellers pr. ("" , url)
            keys = [(e.lower(), url_out) for e in emails] if emails else [("", url_out)]
            if any(k in seen_keys for k in keys):
                continue
            for k in keys:
                seen_keys.add(k)

            cleaned.append(r)

        log.debug(
            "ContactFinder.find_all(%s) -> %d raw, %d cleaned in %.3fs",
            url, len(out), len(cleaned), time.time() - t0
        )
        return cleaned

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


# ------------------------------- CLI / test -----------------------------------

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
    args = ap.parse_args()

    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    cf = ContactFinder()
    if args.all:
        results = cf.find_all(args.url, limit_pages=args.limit_pages)
        print(json.dumps(results, ensure_ascii=False, indent=2))
    else:
        best = cf.find(args.url, limit_pages=args.limit_pages)
        print(json.dumps(best, ensure_ascii=False, indent=2))
