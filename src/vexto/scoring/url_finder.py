"""
url_finder.py  –  VERSION 3.0 (Arkitektonisk Korrekt)
--------------------------------------------------------
Dette er den endelige, professionelle version.
Ændringer:
  - Ansvaret for at oprette og lukke httpx-klienten er flyttet ud af modulet.
  - Modulet udstiller nu get_client() og close_client() funktioner.
  - Al "magisk" atexit-logik er fjernet for at gøre modulet 100% genbrugeligt.
"""
import re
import sqlite3
import httpx
import asyncio
import os
import logging
from pathlib import Path
from dotenv import load_dotenv
from typing import Any, Tuple, AsyncGenerator, Optional
from urllib.parse import urljoin
from difflib import SequenceMatcher
import idna # idna import var i tidligere versioner, men ikke brugt, nu er den det for fuld korrekthed.

# --- Opsætning ---------------------------------------------------------------
log = logging.getLogger(__name__)
try:
    PROJECT_DIR = Path(__file__).resolve().parents[3]
except NameError:
    PROJECT_DIR = Path(".").resolve()

DB_PATH = PROJECT_DIR / "output" / "url_lookup.sqlite"
os.makedirs(PROJECT_DIR / "output", exist_ok=True)

with sqlite3.connect(DB_PATH) as conn:
    conn.execute(
        "CREATE TABLE IF NOT EXISTS url_cache "
        "(normalized_key TEXT PRIMARY KEY, url TEXT, method TEXT, reason TEXT, title TEXT)"
    )
    conn.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_key ON url_cache(normalized_key)")

load_dotenv()

# --- Konstanter & Klient Håndtering -------------------------------------------
# Klienten initialiseres nu som None. Den bliver først oprettet, når den er nødvendig.
HTTP_CLIENT: Optional[httpx.AsyncClient] = None

async def get_client() -> httpx.AsyncClient:
    """
    Returnerer den globale klient. Opretter den, hvis den ikke allerede eksisterer.
    Dette kaldes "lazy initialization".
    """
    global HTTP_CLIENT
    if HTTP_CLIENT is None or HTTP_CLIENT.is_closed:
        log.info("Opretter ny genbrugelig httpx.AsyncClient...")
        HTTP_CLIENT = httpx.AsyncClient(
            limits=httpx.Limits(max_keepalive_connections=20, max_connections=40),
            timeout=15,
            follow_redirects=False
        )
    return HTTP_CLIENT

async def close_client() -> None:
    """
    Lukker den globale klient, hvis den eksisterer. Skal kaldes af det
    script, der bruger dette modul, når det er færdigt.
    """
    global HTTP_CLIENT
    if HTTP_CLIENT and not HTTP_CLIENT.is_closed:
        log.info("Lukker httpx.AsyncClient...")
        await HTTP_CLIENT.aclose()
        HTTP_CLIENT = None


PUBLIC_MAIL_PROVIDERS = {
    "gmail.com", "googlemail.com", "hotmail.com", "outlook.com", "outlook.dk",
    "live.com", "live.dk", "yahoo.com", "icloud.com", "protonmail.com",
}

BANNED_DOMAINS = {
    "webspeed.dk", "mail.tele.dk", "live.dk", "bolig.dk", "outlook.dk",
}

KNOWN_BAD_HASHES = {
    "0e6926b1e4e9bf8f26cf03f7d560b87c6edb743736df27bc19d976329a4d4d25",
    "9f1a41985bf2a2e64ce0dc4f44f6d90f692b29239c8804b1ca4b98fca6be7e2a",
    "77e80d7c4f19e4b0c6eabf4ef98d5ee05bf0879c2f6f392a6bfab4e0c79d0a3f",
}

PARKED_DOMAIN_KEYWORDS = {
    "parked domain", "domain for sale", "buy this domain",
    "password-protected", "adgangskode", "enter password",
}

COMPANY_STOP_WORDS = {
    "aps", "as", "a/s", "is", "ivs", "holding", "selskab", "firma"
}

EMPTY_SITE_KEYWORDS = re.compile(
    r"(index of /|coming soon|under construction|no website is currently present|wordpress.*install)",
    re.I,
)

CLOUDFLARE_RE = re.compile(r"cloudflare", re.IGNORECASE)
UA_SIMPLE  = "VextoBot/1.0"
UA_DESKTOP = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
)

# --- Hjælpefunktioner ---------------------------------------------------------
def _normalize_name(name: Any) -> str:
    if not isinstance(name, str):
        return ""
    name = name.lower()
    for suffix in [" a/s", " aps", " i/s", " ivs", " & søn", " ApS", " A/S"]:
        name = name.replace(suffix, "")
    return re.sub(r"[^a-zæøå0-9]", "", name)

def _has_dns(host: str) -> bool:
    try:
        socket.getaddrinfo(idna.encode(host).decode(), None)
        return True
    except (socket.gaierror, UnicodeError):
        try:
            socket.getaddrinfo(host.encode("idna").decode("ascii"), None)
            return True
        except Exception:
            return False

def _token_match(token: str, word: str) -> bool:
    return token == word or token in word or SequenceMatcher(None, token, word).ratio() > 0.85

async def _safe_get_with_redirects(
    client: httpx.AsyncClient, url: str
) -> httpx.Response:
    headers = {"User-Agent": UA_DESKTOP}
    for hop in range(3):
        try:
            r = await client.get(url, headers=headers)
            if r.status_code in {403, 503} and CLOUDFLARE_RE.search(r.text) and hop == 0:
                log.info("Cloudflare-lignende side ved %s – retry med simpel UA", url)
                await asyncio.sleep(1)
                headers = {"User-Agent": UA_SIMPLE}
                r = await client.get(url, headers=headers)
            if r.status_code in {403, 500} and url.startswith("https://"):
                r = await client.get("http://" + url.split("://", 1)[1], headers=headers)
            if 300 <= r.status_code < 400 and "location" in r.headers:
                url = urljoin(str(r.url), r.headers["location"])
                log.debug("Redirect → %s", url)
                continue
            return r
        except httpx.TimeoutException:
            log.warning("Timeout for %s", url)
            raise
    raise httpx.TooManyRedirects(f"For mange redirects for {url}")

async def _fetch_and_validate_title(
    url: str, firm_name: str, *, relaxed: bool = False, slow: bool = False
) -> Tuple[bool, str, str | None]:
    try:
        client = await get_client()
        r = await _safe_get_with_redirects(client, url)

        if r.status_code >= 400 and not (r.status_code in {401, 403} and "<title" in r.text.lower()):
            return False, f"HTTP fejl {r.status_code}", None

        html = r.text
        html_lower = html.lower()

        sha = hashlib.sha256(html[:4096].encode("utf-8", "ignore")).hexdigest()
        if sha in KNOWN_BAD_HASHES:
            return False, "Parked fingerprint", None

        if EMPTY_SITE_KEYWORDS.search(html):
            return False, "Tom/placeholder-side (keyword match)", None

        if any(k in html_lower for k in PARKED_DOMAIN_KEYWORDS):
            return False, "Side indikerer parkering", None
        
        temp_text = re.sub(r'<style.*?</style>', ' ', html_lower, flags=re.DOTALL)
        temp_text = re.sub(r'<script.*?</script>', ' ', temp_text, flags=re.DOTALL)
        body_text = re.sub(r'<[^>]+>', ' ', temp_text)
        word_count = len(body_text.split())
        
        link_count = len(re.findall(r"<a\s+[^>]*href=", html_lower))
        
        if not relaxed and link_count < 3 and word_count < 20:
            return False, f"For få links ({link_count}) og for få ord ({word_count})", None

        m = re.search(r"<title[^>]*>(.*?)</title>", html_lower, re.S)
        title = m.group(1).strip() if m else None
        if not title:
            og = re.search(r'property=["\']og:title["\'][^>]*content=["\'](.+?)["\']', html_lower)
            title = og.group(1).strip() if og else None

        if not title:
            return (True, "OK (Relaxed: ingen titel)", "") if relaxed else (False, "Ingen titel", None)

        if any(k in title for k in PARKED_DOMAIN_KEYWORDS):
            return False, "Titel indikerer parkering", title

        if relaxed:
            return True, "OK (Relaxed mode)", title

        title_words = set(re.findall(r"[a-zæøå0-9]+", title))
        firm_words = set(re.findall(r"[a-zæøå0-9]+", firm_name.lower()))
        
        core_tokens = {w for w in firm_words if len(w) > 2 and w not in COMPANY_STOP_WORDS}

        if not core_tokens:
            return True, "OK (Ingen unikke firmaord)", title

        matches = sum(1 for t in core_tokens if any(_token_match(t, w) for w in title_words))
        
        if len(core_tokens) <= 1 and matches >= 1:
            return True, "OK (Enkelt kerneord match)", title
        if len(core_tokens) > 1 and matches >= 2:
            return True, "OK (Flere kerneord match)", title

        return False, "Titel matcher ikke kerneord", title

    except Exception as exc:
        log.debug("Validering fejlede for %s: %s", url, exc)
        return False, f"FEJL {type(exc).__name__}", None

async def _validate_candidate(
    url: str, firm_name: str, *, relaxed: bool = False, slow: bool = False
) -> Tuple[bool, str, str | None]:
    if not isinstance(url, str) or not url.strip():
        return False, "Ugyldig URL-streng", None

    domain = url.split("//")[-1].split("/")[0].replace("www.", "")
    if domain in BANNED_DOMAINS:
        return False, "Domæne på blacklist", None
    if not _has_dns(domain):
        return False, "DNS-opslag fejlede", None

    https_url = url if url.startswith("http") else f"https://{url}"
    ok, reason, title = await _fetch_and_validate_title(
        https_url, firm_name, relaxed=relaxed, slow=slow
    )
    if ok:
        return True, reason, title

    if not url.startswith("http"):
        http_url = f"http://{url}"
        ok, reason, title = await _fetch_and_validate_title(
            http_url, firm_name, relaxed=relaxed, slow=slow
        )
        if ok:
            return True, reason, title
    return False, reason, title

def _cache_url(key: str, url: str, method: str, reason: str, title: str | None) -> None:
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            "INSERT OR REPLACE INTO url_cache VALUES (?,?,?,?,?)",
            (key, url, method, reason, title or ""),
        )
        conn.commit()

class BaseResolver:
    async def get_candidates(self, name: str, **kwargs) -> AsyncGenerator[str, None]:
        raise NotImplementedError
        yield

class CvrUrlResolver(BaseResolver):
    async def get_candidates(self, name: str, **kwargs) -> AsyncGenerator[str, None]:
        if (cvr_url := kwargs.get("cvr_url")):
            yield cvr_url

class EmailDomainResolver(BaseResolver):
    async def get_candidates(self, name: str, **kwargs) -> AsyncGenerator[str, None]:
        email = kwargs.get("email")
        if not (isinstance(email, str) and "@" in email):
            return
        domain = re.split(r"[>\s;,?]", email.split("@", 1)[1].lower().strip())[0]
        if any(domain.endswith(pub) for pub in PUBLIC_MAIL_PROVIDERS):
            return
        yield domain

class BruteForceResolver(BaseResolver):
    async def get_candidates(self, name: str, **kwargs) -> AsyncGenerator[str, None]:
        key = _normalize_name(name)
        if not key:
            return
        for c in (
            f"{key}.dk", f"www.{key}.dk", f"{key}.com", f"www.{key}.com",
            f"{key}.nu", f"{key}.eu", f"{key}.biz", f"{key}.build",
        ):
            yield c

class GoogleResolver(BaseResolver):
    async def get_candidates(self, *_, **__) -> AsyncGenerator[str, None]:
        if False:
            yield

async def find_company_url(
    name: str, **kwargs
) -> Tuple[str, str, str, str] | None:
    key = _normalize_name(name)
    if not key:
        return None

    with sqlite3.connect(DB_PATH) as conn:
        row = conn.execute(
            "SELECT url, method, reason, title FROM url_cache WHERE normalized_key=?", (key,)
        ).fetchone()
    if row:
        return row

    resolvers = [CvrUrlResolver(), EmailDomainResolver(), BruteForceResolver(), GoogleResolver()]

    for rslv in resolvers:
        try:
            slow = isinstance(rslv, BruteForceResolver)
            async for cand in rslv.get_candidates(name, **kwargs):
                
                relaxed = False

                if isinstance(rslv, CvrUrlResolver):
                    relaxed = True
                
                if isinstance(rslv, EmailDomainResolver):
                    domain_parts = cand.lower().split('.')
                    domain_stub = domain_parts[-2] if len(domain_parts) > 1 else domain_parts[0]
                    company_tokens = set(re.findall(r"[a-z0-9]+", _normalize_name(name)))
                    
                    if domain_stub in company_tokens:
                        log.info(f"E-mail-domæne '{cand}' matcher firmanavn-token '{domain_stub}'. Bruger relaxed mode.")
                        relaxed = True
                
                ok, reason, title = await _validate_candidate(cand, name, relaxed=relaxed, slow=slow)
                if ok:
                    meth = rslv.__class__.__name__
                    log.info("Fandt URL for '%s' via %s: %s", name, meth, cand)
                    _cache_url(key, cand, meth, reason, title)
                    return cand, meth, reason, title
        except Exception as exc:
            log.warning("Resolver %s fejlede for '%s': %s", rslv.__class__.__name__, name, exc)
    return None