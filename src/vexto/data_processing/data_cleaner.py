# data_cleaner.py
# -----------------------------------------------------------------------------
# Rens & transformér rå CVR-hits til et analyseklart DataFrame – inkl.:
# - Udtræk/parse af kontaktinfo fra 'nyesteKontaktoplysninger'
# - Infer 'Hjemmeside' fra Email (med filtrering af ISP/SoMe/aggregator-domæner)
# - Google Places + web-søgning (SerpAPI/Google CSE/Brave) til officiel hjemmeside
# - Dynamiske "brand-stopord" pr. kørselsdatasæt
# - Liveness-check af hjemmeside (placeholder/login = False), støjdæmpet logging
# -----------------------------------------------------------------------------

from __future__ import annotations

import asyncio
import logging
import os
import re
from logging.handlers import RotatingFileHandler
from typing import Any, Dict, List, Tuple

import aiohttp
import pandas as pd
from aiohttp import ClientError, ClientSession, ClientTimeout
from dotenv import load_dotenv
from urllib.parse import urlparse

# --- robust import til ContentEncodingError (varierer ml. aiohttp-versioner) ---
try:
    from aiohttp.http_exceptions import ContentEncodingError  # aiohttp >= 3.8/3.12
except Exception:  # pragma: no cover
    try:
        from aiohttp.client_exceptions import ContentEncodingError  # ældre aiohttp
    except Exception:  # pragma: no cover
        ContentEncodingError = Exception  # defensiv fallback

# -----------------------------------------------------------------------------
# ENV & LOGGING
# -----------------------------------------------------------------------------
load_dotenv()  # indlæs .env fra projektrod

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
LOG_FILE = os.getenv("LOG_FILE", "website_liveness.log")
LOG_MAX_BYTES = int(os.getenv("LOG_MAX_BYTES", "5242880"))         # 5 MB
LOG_BACKUP_COUNT = int(os.getenv("LOG_BACKUP_COUNT", "3"))
LOG_TO_CONSOLE = os.getenv("LOG_TO_CONSOLE", "1") == "1"

handlers = [RotatingFileHandler(LOG_FILE, maxBytes=LOG_MAX_BYTES,
                                backupCount=LOG_BACKUP_COUNT, encoding="utf-8")]
if LOG_TO_CONSOLE:
    handlers.append(logging.StreamHandler())

logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=handlers,
)
log = logging.getLogger(__name__)
logging.getLogger("aiohttp.access").setLevel(logging.WARNING)  # dæmp aiohttp access logs

# -----------------------------------------------------------------------------
# KONSTANTER & REGEX
# -----------------------------------------------------------------------------
PHONE_DIGIT_REGEX = re.compile(r"^\d{8}$")
URL_PREFIX_REGEX = re.compile(r"^(https?:\/\/|www\.)", re.IGNORECASE)
EMAIL_REGEX = re.compile(r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$")
DOMAIN_REGEX = re.compile(r"^[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$")

FREE_EMAIL_DOMAINS = {
    "gmail.com", "hotmail.com", "outlook.com", "yahoo.com",
    "live.dk", "hotmail.dk", "icloud.com", "aol.com",
    "webspeed.dk", "mail.tele.dk", "eriksminde.dk", "outlook.dk",
    "mail.dk", "mail.com", "*.tele.dk", "worldmail.dk", "yahoo.dk",
    "youmail.dk", "youseepost.dk", "yahoo.*", "*.mail.dk"
}
EXCLUDED_SUFFIXES = [re.compile(r"\.mail\.dk$")]

# ISP-lignende suffixes (typisk ikke firmaets officielle site)
ISP_LIKE_SUFFIXES = (
    ".tele.dk", ".post.tele.dk", ".post10.tele.dk", ".mail.tele.dk",
    ".get2net.dk", ".mail.dk",
)

# Aggregator/SoMe/portaler som aldrig er "officiel hjemmeside"
AGGREGATOR_DOMAINS = {
    "facebook.com", "m.facebook.com", "instagram.com", "linkedin.com",
    "krak.dk", "dk.krak.dk", "proff.dk", "danskevirksomheder.dk",
    "trustpilot.com", "trustpilot.dk", "findsmiley.dk",
    "cvr.dk", "virk.dk", "maps.google.", "goo.gl", "g.page",
    "118.dk", "degulesider.dk", "gulesider.dk",
}
AGGREGATOR_PATTERNS = [
    r"(?:^|\.)anmeld-haandvaerker\.dk$",
    r"(?:^|\.)haandvaerker\.dk$",
    r"(?:^|\.)find-virksomhed\.dk$",
    r"(?:^|\.)3byggetilbud\.dk$",
    r"(?:^|\.)tekniq\.dk$",
    r"(?:^|\.)facebook\.com$",
    r"(?:^|\.)linkedin\.com$",
    r"(?:^|\.)wixsite\.com$",
    r"(?:^|\.)cvrapi\.dk$",
]
AGGREGATOR_RE = re.compile("|".join(AGGREGATOR_PATTERNS), re.IGNORECASE)

PLACEHOLDER_PHRASES = [
    # ENG
    "this domain has been registered by one.com",
    "get your own domain name with one.com",
    ">this domain is parked<",
    ">website under construction<",
    "welcome to your new website",
    "this site is hosted by one.com",

    # DK (One.com/hosting/parkering)
    "dette domæne er registreret hos one.com",
    "dette domæne er registreret hos one",
    "dette domæne er parkeret",
    "dette domæne er reserveret",
    "webhotel hos one.com",
    "denne hjemmeside er oprettet hos one.com",
    "din hjemmeside er opsat med one.com",
    "siden er under opbygning",
    "hjemmesiden er under opbygning",

    # Generiske meta/brand-signaturer
    r'<meta name="description" content="domain is parked"',
    r"<title>.*one\.com.*</title>",
    r"konsoleh\s*::\s*login",
    r"your-server\.de",
    r"created by group online - grouponline\.dk",
]

UNDER_CONSTRUCTION_PHRASES = ["under konstruktion"]

# --- Login detection (eksplicit/konservativ) --------------------------------
LOGIN_HINT_PATTERNS = [
    r'wp-login',                 # /wp-login.php
    r'/wp-admin',                # /wp-admin/ → login redirect
    r'password-protected=login'  # WordPress "Password Protected" plugin
]
LOGIN_EXACT_WORDS = [
    "adgangskode", "password", "log ind", "login", "brugernavn",
    "remember me", "husk mig"
]
LOGIN_RE = re.compile("|".join(LOGIN_HINT_PATTERNS), re.IGNORECASE)

# --- Stedvalidering (by/postnr) til google_places + web_search ----------------
def _norm_city(s: str) -> str:
    if not s:
        return ""
    s = s.strip().lower()
    mapping = {"copenhagen": "københavn"}
    return mapping.get(s, s)

def _places_mismatch(cvr_by: str, cvr_post: str, gp_by: str, gp_post: str) -> bool:
    cvr_post = (str(cvr_post).strip() if cvr_post is not None else "")
    gp_post  = (str(gp_post).strip()  if gp_post  is not None else "")
    if cvr_post and gp_post:
        return cvr_post != gp_post
    cb, gb = _norm_city(cvr_by or ""), _norm_city(gp_by or "")
    return bool(cb and gb and cb != gb)

def _domain_from_url(u: str) -> str:
    try:
        if not u:
            return ""
        if not u.lower().startswith(("http://", "https://")):
            u = "http://" + u
        p = urlparse(u)
        return (p.netloc or p.path.split("/")[0]).lower()
    except Exception:
        return ""

def _strong_tokens(name: str) -> list[str]:
    """
    Stærke tokens (≥5 tegn) renset for brand-stopord.
    Bruges bl.a. i 'svagt match' check uden crawl.
    """
    if not name:
        return []
    raw = re.split(r"[^a-zæøåA-ZÆØÅ]+", name)
    out: list[str] = []
    for w in raw:
        if len(w) < 5:
            continue
        n = _norm_token(w)
        if not n or n in BRAND_STOPWORDS:
            continue
        out.append(n)
    return out

# ENV-styring
LIVE_TREAT_LOGIN_AS_DOWN = os.getenv("LIVE_TREAT_LOGIN_AS_DOWN", "1") == "1"
LIVE_FLAG_THIN_PAGES     = os.getenv("LIVE_FLAG_THIN_PAGES",     "0") == "1"
LIVE_THIN_MIN_WORDS      = int(os.getenv("LIVE_THIN_MIN_WORDS",  "40"))
LIVE_THIN_MIN_LINKS      = int(os.getenv("LIVE_THIN_MIN_LINKS",  "2"))

def _looks_like_login_gate(text_lower: str, final_url: str) -> bool:
    """
    - Returnerer True straks hvis:
        * URL'en indeholder 'password-protected=login' (WP password plugin), eller
        * path ender i '/wp-login.php', eller
        * formen poster til wp-login.php
    - Ellers kræves BÅDE (a) login-agtig URL OG (b) password-form indikator.
    """
    try:
        parsed = urlparse(final_url)
        path = (parsed.path or "").lower()
        query = (parsed.query or "").lower()
        host = (parsed.netloc or "").lower()
    except Exception:
        path, query, host = "", "", ""

    # Early: WP Password Protected-plugin i URL → altid login
    if "password-protected=login" in query:
        return True

    # Hard WP-login page eller "fuldt privat" WP
    if path.endswith("/wp-login.php") \
       or ('action="/wp-login.php"' in text_lower) \
       or (host and f'action="https://{host}/wp-login.php"' in text_lower):
        return True

    url_signal = bool(LOGIN_RE.search(path) or LOGIN_RE.search(query))
    pw_signal = (
        ('type="password"' in text_lower)
        or ('id="loginform"' in text_lower)
        or ('name="log"' in text_lower)
        or ('name="pwd"' in text_lower)
        or ('name="wp-submit"' in text_lower)
        or ('name="redirect_to"' in text_lower)
        or ('class="login"' in text_lower)
    )
    return bool(url_signal and pw_signal)

def _looks_like_thin_page(html: str) -> bool:
    cleaned = re.sub(r"<script.*?</script>|<style.*?</style>", " ", html, flags=re.DOTALL | re.IGNORECASE)
    cleaned = re.sub(r"<[^>]+>", " ", cleaned)
    words = len(re.findall(r"\w+", cleaned))
    links = html.lower().count("<a ")
    return (words < LIVE_THIN_MIN_WORDS) and (links <= LIVE_THIN_MIN_LINKS)

# -----------------------------------------------------------------------------
# BRAND STOPWORDS
# -----------------------------------------------------------------------------
BRAND_CORE_STOPWORDS = {
    "aps", "a/s", "ks", "k/s", "ps", "p/s", "ivs",
    "holding", "invest", "investment", "group", "grupp", "gruppen",
    "service", "services", "company", "co", "firma", "solutions", "consult",
    "entreprise", "enterprise", "consulting", "partner", "partners",
}
BRAND_STOPWORDS: set[str] = set(BRAND_CORE_STOPWORDS)

def _norm_token(t: str) -> str:
    t = t.strip().lower()
    t = (t.replace("æ", "ae").replace("ø", "oe").replace("å", "aa")
           .replace("é", "e").replace("ö", "o").replace("ä", "a"))
    return re.sub(r"[^a-z0-9\-]", "", t)

def _tokenize_name_all(name: str) -> list[str]:
    raw = re.split(r"\W+", (name or ""))
    toks: list[str] = []
    for w in raw:
        n = _norm_token(w)
        if len(n) < 2 or n.isdigit():
            continue
        toks.append(n)
    return toks

def _brand_tokens(name: str) -> list[str]:
    toks = _tokenize_name_all(name)
    return [t for t in toks if t not in BRAND_STOPWORDS and len(t) > 2]

def _install_dynamic_brand_stopwords(df: pd.DataFrame) -> None:
    topk = int(os.getenv("BRAND_DYN_TOPK", "40"))
    min_df = float(os.getenv("BRAND_DYN_MIN_DF", "0.02"))
    extra_file = os.getenv("BRAND_STOPWORDS_FILE")

    if extra_file and os.path.exists(extra_file):
        try:
            with open(extra_file, "r", encoding="utf-8") as f:
                for line in f:
                    w = _norm_token(line.strip())
                    if len(w) >= 2:
                        BRAND_STOPWORDS.add(w)
        except Exception:
            pass

    names = df.get("Vrvirksomhed_virksomhedMetadata_nyesteNavn_navn",
                   pd.Series(dtype=str)).astype(str)
    N = max(len(names), 1)
    df_count: dict[str, int] = {}

    for name in names:
        seen = set()
        for w in re.split(r"\W+", name):
            n = _norm_token(w)
            if len(n) < 3 or n.isdigit():
                continue
            if n in BRAND_CORE_STOPWORDS:
                continue
            seen.add(n)
        for n in seen:
            df_count[n] = df_count.get(n, 0) + 1

    freq_sorted = sorted(df_count.items(), key=lambda x: x[1], reverse=True)
    dyn = set()
    for i, (tok, c) in enumerate(freq_sorted):
        if c / N >= min_df or i < topk:
            dyn.add(tok)
    dyn = {t for t in dyn if df_count.get(t, 0) >= 2}
    BRAND_STOPWORDS.update(dyn)

# -----------------------------------------------------------------------------
# LOG-STØJKONTROL
# -----------------------------------------------------------------------------
LOG_ONLY_PROBLEMS = os.getenv("LOG_ONLY_PROBLEMS", "1") == "1"
LOG_HIDE_CONTENT  = os.getenv("LOG_HIDE_CONTENT",  "1") == "1"
LOG_SUMMARY       = os.getenv("LOG_SUMMARY",       "1") == "1"

_mutes = os.getenv(
    "LOG_MUTE_DOMAINS",
    r"(^|\.)cvrapi\.dk$,(^|\.)haandvaerker\.dk$,(^|\.)anmeld-haandvaerker\.dk$",
)
MUTED_DOMAINS_RE = re.compile(
    "|".join([s.strip() for s in _mutes.split(",") if s.strip()]),
    re.IGNORECASE,
)

def _short(text: str, n: int = 120) -> str:
    if not text:
        return ""
    return re.sub(r"\s+", " ", text)[:n]

def _should_log_http(domain: str, status: int | None, live: bool, remark: str) -> bool:
    d = clean_domain(domain or "")
    if d and MUTED_DOMAINS_RE.search(d):
        return False
    if LOG_ONLY_PROBLEMS and live and (remark.strip() == "") and (status is not None) and (200 <= status < 400):
        return False
    return True

# -----------------------------------------------------------------------------
# DOMÆNE/URL HJÆLPERE
# -----------------------------------------------------------------------------
def clean_domain(domain: str) -> str:
    if not isinstance(domain, str):
        return ""
    domain = domain.strip().lower()
    domain = re.sub(r"^https?://", "", domain)
    domain = re.sub(r"^www\.", "", domain)
    domain = domain.rstrip("/")
    return (domain
            .replace("æ", "ae").replace("Æ", "Ae")
            .replace("ø", "oe").replace("Ø", "Oe")
            .replace("å", "aa").replace("Å", "Aa"))

def _strip_host(url_or_host: str) -> str:
    if not isinstance(url_or_host, str):
        return ""
    s = url_or_host.strip().lower()
    s = re.sub(r"^https?://", "", s)
    s = re.sub(r"^www\.", "", s)
    return s.rstrip("/")

def _to_idna(host: str) -> str:
    if not isinstance(host, str):
        return ""
    h = _strip_host(host)
    try:
        return h.encode("idna").decode("ascii")
    except Exception:
        return h

def _netloc(url: str) -> str:
    try:
        return urlparse(url).netloc.lower()
    except Exception:
        return ""

def _registrable_domain(host: str) -> str:
    parts = host.split(".")
    return ".".join(parts[-2:]) if len(parts) >= 2 else host

def _is_aggregator(domain: str) -> bool:
    d = clean_domain(domain or "")
    return bool(AGGREGATOR_RE.search(d))

def _is_bad_website_domain(domain: str) -> bool:
    d = (domain or "").lower().strip()
    if not d:
        return True
    if any(d.endswith(suf) for suf in ISP_LIKE_SUFFIXES):
        return True
    d2 = d[4:] if d.startswith("www.") else d
    if any(agg in d2 for agg in AGGREGATOR_DOMAINS):
        return True
    return False

def _looks_like_placeholder(text_lower: str, final_host: str) -> bool:
    phrase_hit = any(re.search(p, text_lower, re.IGNORECASE) for p in PLACEHOLDER_PHRASES) \
                 or ("under konstruktion" in text_lower)

    onecom_signals = (
        "one.com" in text_lower or
        "onecom" in text_lower or
        "data-onecom" in text_lower or
        "/one-webstatic/" in text_lower or
        "onewebstatic.com" in text_lower
    )
    is_onecom_like = phrase_hit or onecom_signals

    structure_score = (text_lower.count("<a ") + text_lower.count("<nav") +
                       text_lower.count("<section") + text_lower.count("<article"))
    poor_structure = structure_score < 2

    reg = _registrable_domain(final_host or "")
    internal_links = 0
    if reg:
        try:
            internal_link_re = re.compile(r'href=["\'](?:/|https?://[^"\']*' + re.escape(reg) + r')', re.IGNORECASE)
            internal_links = len(internal_link_re.findall(text_lower))
        except re.error:
            internal_links = 0

    few_internal_links = internal_links <= 1

    if is_onecom_like and (poor_structure or few_internal_links):
        return True
    if phrase_hit and poor_structure:
        return True
    return False

# -----------------------------------------------------------------------------
# LIVENESS-CHECK
# -----------------------------------------------------------------------------
async def _check_website_liveness(urls: List[str], timeout: int = 15, max_retries: int = 3) -> tuple[List[Any], List[str]]:
    results = ["N/A"] * len(urls)
    remarks = [""] * len(urls)

    headers = [
        {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
            "Accept-Language": "da, en;q=0.8",
        },
        {
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
            "Accept-Language": "da, en;q=0.8",
        },
        {
            "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
            "Accept-Language": "da, en;q=0.8",
            "Accept-Encoding": "identity",
        },
    ]

    # cache nøgle skal være VARIANT-specifik (ellers overskriver første svar alt)
    cache: Dict[str, Dict[str, Any]] = {}

    async def check_single_url(url: str, domain: str, session: ClientSession, index: int) -> Tuple[bool, str]:
        # Vigtigt: cache på URL-varianten – ikke kun på domain
        cache_key = url
        if cache_key in cache:
            return cache[cache_key]["live"], cache[cache_key]["remark"]

        remark = ""
        for attempt in range(max_retries):
            try:
                try:
                    hdr = headers[attempt % len(headers)]
                except Exception:
                    hdr = {"User-Agent": "Mozilla/5.0", "Accept-Language": "da, en;q=0.8"}

                async with session.get(
                    url,
                    timeout=ClientTimeout(total=timeout),
                    allow_redirects=True,
                    ssl=False,
                    headers=hdr,
                ) as response:
                    final_url = str(response.url)
                    final_host = urlparse(final_url).netloc.lower()

                    text_bytes = await response.read()
                    try:
                        text = text_bytes.decode("utf-8")
                    except UnicodeDecodeError:
                        try:
                            text = text_bytes.decode("iso-8859-1")
                        except UnicodeDecodeError:
                            text = text_bytes.decode("utf-8", errors="replace")
                    text_lower = text.lower()

                    status = response.status
                    is_ok        = 200 <= status < 300
                    is_redirect  = 300 <= status < 400
                    is_auth      = status in (401, 403, 405)
                    is_ratelimit = status == 429
                    is_no_content= status == 204
                    is_custom    = status == 455

                    # ---- LOGIN: early return (trumfer alt) -----------------
                    is_login_like = _looks_like_login_gate(text_lower, final_url)
                    if LIVE_TREAT_LOGIN_AS_DOWN and is_login_like:
                        live = False
                        remark = "login page"
                        if _should_log_http(domain, status, live, remark):
                            snippet = "" if LOG_HIDE_CONTENT else f", Content start: {_short(text)!r}"
                            log.debug(f"[{index}] {url} (→ {final_url}): Status {status}, Remark: {remark}, Live: {live}{snippet} (Attempt {attempt + 1})")
                        cache[cache_key] = {"live": live, "remark": remark}
                        return live, remark
                    # --------------------------------------------------------

                    # Heuristikker (placeholder/tynd side)
                    is_placeholder_like = _looks_like_placeholder(text_lower, final_host)
                    is_thin_like        = LIVE_FLAG_THIN_PAGES and _looks_like_thin_page(text)

                    # Strammere: vi accepterer kun 2xx/3xx som "bestået" status
                    if is_ok or is_redirect:
                        live = not (is_placeholder_like or is_thin_like)
                        remark_bits = []
                        if is_placeholder_like:
                            remark_bits.append("placeholder" if "under konstruktion" not in text_lower else "under konstruktion")
                        if is_thin_like:
                            remark_bits.append("thin page")
                        remark = ", ".join(remark_bits)
                        if _should_log_http(domain, status, live, remark):
                            snippet = "" if LOG_HIDE_CONTENT else f", Content start: {_short(text)!r}"
                            log.debug(f"[{index}] {url} (→ {final_url}): Status {status}, Remark: {remark or ''}, Live: {live}{snippet} (Attempt {attempt + 1})")
                        cache[cache_key] = {'live': live, 'remark': remark}
                        return live, remark

                    # 4xx/5xx og andre "ikke bestået"
                    remark = f"status {status}"
                    if _should_log_http(domain, status, False, remark):
                        log.debug(f"[{index}] {url} (→ {final_url}): Status {status} - False (Attempt {attempt + 1})")

            except ClientError as e:
                msg = str(e).lower()
                if "getaddrinfo failed" in msg:
                    remark = "dns failure"
                    cache[cache_key] = {"live": False, "remark": remark}
                    if _should_log_http(domain, None, False, remark):
                        log.debug(f"[{index}] {url}: DNS failure - False (Attempt {attempt + 1})")
                    await asyncio.sleep(1.0)
                    if attempt == max_retries - 1:
                        return False, remark
                else:
                    remark = "ssl error"
                    if _should_log_http(domain, None, False, remark):
                        log.debug(f"[{index}] {url}: ClientError {e} (Attempt {attempt + 1})")

            except ContentEncodingError:
                # Behandl som "fail" for strenghed
                remark = "content-encoding error"
                cache[cache_key] = {"live": False, "remark": remark}
                if _should_log_http(domain, None, False, remark):
                    log.debug(f"[{index}] {url}: ContentEncodingError - False (Attempt {attempt + 1})")
                return False, remark

            except asyncio.TimeoutError:
                remark = "timeout"
                if _should_log_http(domain, None, False, remark):
                    log.debug(f"[{index}] {url}: Timeout - False (Attempt {attempt + 1})")

            except ValueError as e:
                remark = "encoding error"
                if _should_log_http(domain, None, False, remark):
                    log.debug(f"[{index}] {url}: ValueError {e} (Attempt {attempt + 1})")

            await asyncio.sleep(0.2)

        cache[cache_key] = {"live": False, "remark": remark or "timeout/error"}
        return False, remark or "timeout/error"

    async def test_both_www(domain: str, session: ClientSession, index: int):
        """
        Tjek både https/http og med/uden www.
        Prioritet:
        1) Hvis én variant returnerer 'login page' -> False (hard fail)
        2) Hvis mindst én variant er live -> True
        3) Hvis alle varianter kun fejler med 'ssl error' og LIVE_TOLERATE_SSL_ERRORS=1 -> True (tolereret)
        4) Ellers -> False
        """
        if _is_aggregator(domain):
            log.debug(f"[{index}] {domain}: skip aggregator")
            return True, "aggregator (skipped)"
        if not domain or not DOMAIN_REGEX.match(domain):
            log.debug(f"[{index}] INVALID DOMAIN: '{domain}'")
            return None, "invalid domain"

        bases = [domain]
        if not domain.startswith("www."):
            bases.insert(0, "www." + domain)

        # Prøv altid https først, derefter http
        variants = [f"https://{b}" for b in bases] + [f"http://{b}" for b in bases]

        # Kør alle varianter færdigt (ingen early return), så vi kan aggregere korrekt
        tasks = [asyncio.create_task(check_single_url(u, domain, session, index)) for u in variants]
        try:
            results_all = await asyncio.gather(*tasks, return_exceptions=True)
        except (TimeoutError, asyncio.TimeoutError):
            results_all = []

        if not results_all:
            log.debug(f"[{index}] {domain}: no response within timeout")
            return False, "no response"

        any_login = False
        any_true  = False
        only_ssl  = True   # bliver False hvis vi ser en anden fejl end 'ssl error'
        last_remark = "no response"

        for res in results_all:
            if isinstance(res, Exception):
                only_ssl = False
                last_remark = f"exception:{type(res).__name__}"
                continue

            live, remark = res
            rmk = (remark or "").lower()
            last_remark = remark or last_remark

            # 1) Login page er hard fail uanset andre varianter
            if rmk == "login page":
                any_login = True

            # Track en succes-variant
            if live is True:
                any_true = True

            # 'only_ssl' må kun være sandt hvis ALLE fejl er præcis 'ssl error'
            if live is not True and rmk != "ssl error":
                only_ssl = False

        if any_login:
            return False, "login page"

        # 2) Mindst én variant leverer -> acceptér
        if any_true:
            # En variant svarede OK → remark skal ikke stå som "no response"
            return True, ""

        # 3) Tolerér rene SSL-fejl (fx gammel side uden korrekt certifikat)
        if os.getenv("LIVE_TOLERATE_SSL_ERRORS", "1") == "1" and only_ssl:
            return True, "ssl error (tolerated)"

        # 4) Alt andet = samlet False
        return False, last_remark or "no response"


    connector = aiohttp.TCPConnector(
        limit=int(os.getenv("HTTP_LIMIT_TOTAL", "120")),
        limit_per_host=int(os.getenv("HTTP_LIMIT_PER_HOST", "6")),
        ttl_dns_cache=600,
    )
    sem = asyncio.Semaphore(int(os.getenv("CHECK_CONCURRENCY", "80")))

    async with aiohttp.ClientSession(connector=connector) as session:
        async def guarded(domain, i):
            async with sem:
                return await test_both_www(domain, session, i)

        tasks = [guarded(url, i) for i, url in enumerate(urls) if url and url != "N/A"]
        task_results = await asyncio.gather(*tasks, return_exceptions=True)

        async_pos = 0
        for i, url in enumerate(urls):
            if url and url != "N/A":
                task_result = task_results[async_pos]
                async_pos += 1
                if isinstance(task_result, tuple):
                    results[i], remarks[i] = task_result
                else:
                    results[i] = False
                    remarks[i] = "exception"
            else:
                results[i] = "N/A"
                remarks[i] = "no domain"

    if LOG_SUMMARY:
        total = len([u for u in urls if u and u != "N/A"])
        live_count = sum(1 for r in results if r is True)
        fail_count = sum(1 for r in results if r is False)
        reason_counts: Dict[str, int] = {}
        for r, rem in zip(results, remarks):
            if r is False:
                reason_counts[rem or "unknown"] = reason_counts.get(rem or "unknown", 0) + 1
        top = ", ".join(
            [f"{k}: {v}" for k, v in sorted(reason_counts.items(), key=lambda x: x[1], reverse=True)[:5]]
        ) or "n/a"
        log.info(f"Liveness summary — checked={total}, live={live_count}, failed={fail_count}. Top reasons: {top}")

    return results, remarks

# -----------------------------------------------------------------------------
# PARSE & FELTHJÆLPERE
# -----------------------------------------------------------------------------
def _parse_contact_value(raw_contact_list: Any) -> Dict[str, str]:
    if not isinstance(raw_contact_list, list):
        return {"Telefon": "N/A", "Email": "N/A", "Hjemmeside": "N/A"}

    phones, emails, websites = set(), set(), set()
    for item in raw_contact_list:
        val = ""
        if isinstance(item, dict) and "kontaktoplysning" in item:
            val = str(item["kontaktoplysning"]).strip()
        elif isinstance(item, str):
            val = item.strip()
        if not val:
            continue

        digits_only = re.sub(r"\D", "", val)
        if "@" in val:
            emails.add(val)
        elif PHONE_DIGIT_REGEX.fullmatch(digits_only):
            phones.add(val)
        elif "." in val:
            websites.add(val if URL_PREFIX_REGEX.match(val) else f"https://{val}")

    return {
        "Telefon": "; ".join(sorted(phones)) or "N/A",
        "Email": "; ".join(sorted(emails)) or "N/A",
        "Hjemmeside": "; ".join(sorted(websites)) or "N/A",
    }

def _get_all_directors(rel_list: Any) -> str:
    if not isinstance(rel_list, list):
        return "N/A"

    names = set()
    for rel in rel_list:
        if not isinstance(rel, dict):
            continue
        deltager = rel.get("deltager") or {}
        if not isinstance(deltager, dict):
            continue
        if (deltager.get("enhedstype") or "").upper() != "PERSON":
            continue

        organisationer = rel.get("organisationer") or []
        if not isinstance(organisationer, list):
            organisationer = []
        for org in organisationer:
            if not isinstance(org, dict):
                continue
            role = (org.get("hovedtype") or "").upper()
            if role not in {
                "DIREKTION", "LEDELSESORGAN", "INDEHAVER",
                "FULDT_ANSVARLIG_DELTAGERE", "FULDT_ANSVARLIG_DELTAGER",
            }:
                continue
            for n in (deltager.get("navne") or []):
                if isinstance(n, dict):
                    name = n.get("navn")
                    if name:
                        names.add(name)

    return "; ".join(sorted(names)) or "N/A"

# -----------------------------------------------------------------------------
# WEB-SØGNING & KONFIRMATION
# -----------------------------------------------------------------------------
async def _search_web(query: str, session: ClientSession, max_results: int = 8, engine_out: list[str] | None = None) -> list[dict]:
    serpapi_key = os.getenv("SERPAPI_KEY")
    gkey = os.getenv("GOOGLE_API_KEY")
    gcx = os.getenv("GOOGLE_CSE_ID")
    brave_key = os.getenv("BRAVE_API_KEY")

    headers = {"User-Agent": "Mozilla/5.0"}

    # SerpAPI
    if serpapi_key:
        url = "https://serpapi.com/search.json"
        params = {"engine": "google", "q": query, "num": max_results, "hl": "da", "gl": "dk", "api_key": serpapi_key}
        try:
            async with session.get(url, params=params, headers=headers, timeout=ClientTimeout(total=20)) as r:
                j = await r.json()
                items = j.get("organic_results", []) or []
                out = []
                for it in items:
                    link, title = it.get("link"), it.get("title")
                    snippet = (it.get("snippet") or it.get("about_this_result", {}).get("source", "")) or ""
                    if link and title:
                        out.append({"url": link, "title": title, "snippet": snippet})
                if out and engine_out is not None:
                    engine_out.append("serpapi")
                return out
        except Exception:
            pass

    # Google CSE
    if gkey and gcx:
        url = "https://www.googleapis.com/customsearch/v1"
        params = {"key": gkey, "cx": gcx, "q": query, "num": min(max_results, 10), "hl": "da"}
        try:
            async with session.get(url, params=params, headers=headers, timeout=ClientTimeout(total=20)) as r:
                j = await r.json()
                items = j.get("items", []) or []
                out = []
                for it in items:
                    link, title = it.get("link"), it.get("title")
                    snippet = it.get("snippet", "")
                    if link and title:
                        out.append({"url": link, "title": title, "snippet": snippet})
                if out and engine_out is not None:
                    engine_out.append("google_cse")
                return out
        except Exception:
            pass

    # Brave
    if brave_key:
        url = "https://api.search.brave.com/res/v1/web/search"
        params = {"q": query, "count": max_results, "country": "dk", "safesearch": "off"}
        try:
            async with session.get(url, params=params, headers={"X-Subscription-Token": brave_key, **headers},
                                   timeout=ClientTimeout(total=20)) as r:
                j = await r.json()
                items = j.get("web", {}).get("results", []) or []
                out = []
                for it in items:
                    link, title = it.get("url"), it.get("title")
                    snippet = it.get("description", "")
                    if link and title:
                        out.append({"url": link, "title": title, "snippet": snippet})
                if out and engine_out is not None:
                    engine_out.append("brave")
                return out
        except Exception:
            pass

    return []

async def _quick_confirm_company(session: ClientSession, url: str, company_tokens: list[str], cvr: str | None) -> int:
    try:
        async with session.get(url, timeout=ClientTimeout(total=10), allow_redirects=True, ssl=False) as r:
            if r.status >= 400:
                return 0
            text = (await r.text(errors="ignore")).lower()
            hits = sum(1 for t in company_tokens if t and t in text)
            bonus = 3 if hits >= 2 else 0
            if cvr:
                digits = re.sub(r"\D", "", text)
                if cvr in digits:
                    bonus += 3
            return bonus
    except Exception:
        return 0

# -----------------------------------------------------------------------------
# GOOGLE PLACES → OFFICIEL WEBSITE
# -----------------------------------------------------------------------------
def _extract_from_address_components(components: list[dict]) -> Tuple[str, str]:
    if not isinstance(components, list):
        return "", ""
    city, post = "", ""
    for comp in components:
        types = comp.get("types", [])
        val = comp.get("long_name") or comp.get("short_name") or ""
        if "postal_town" in types or "locality" in types:
            city = val
        if "postal_code" in types:
            post = val
    return city, post

async def _find_official_site_via_places(company: str, by: str, postnr: str, cvr: str | None, session: ClientSession) -> tuple[str, str, str, str, str] | None:
    key = os.getenv("GOOGLE_PLACES_API_KEY")
    if not key or not company:
        return None

    q = " ".join(t for t in [company, postnr or "", by or ""] if t).strip()

    # 1) Text Search
    ts_url = "https://maps.googleapis.com/maps/api/place/textsearch/json"
    ts_params = {"query": q, "region": "dk", "language": "da", "key": key}
    try:
        async with session.get(ts_url, params=ts_params, timeout=ClientTimeout(total=12)) as r:
            ts = await r.json()
            candidates = (ts.get("results") or [])[:3]
            if not candidates:
                return None
            place_id = candidates[0].get("place_id")
            if not place_id:
                return None
    except Exception:
        return None

    # 2) Details
    det_url = "https://maps.googleapis.com/maps/api/place/details/json"
    det_params = {"place_id": place_id, "fields": "website,url,address_components", "language": "da", "key": key}
    try:
        async with session.get(det_url, params=det_params, timeout=ClientTimeout(total=12)) as r:
            det = await r.json()
            res = det.get("result") or {}
            website = (res.get("website") or "").strip()
            if not website:
                return None
            host = urlparse(website).netloc.lower()
            if not host or _is_bad_website_domain(host):
                return None
            reg = _registrable_domain(host)

            g_city, g_post = _extract_from_address_components(res.get("address_components", []) or [])

            company_tokens = _tokenize_name_all(company)
            try:
                bonus = await _quick_confirm_company(session, website, company_tokens, cvr)
            except Exception:
                bonus = 0
            if bonus < 2:
                return None

            return (reg or None), place_id, website, g_city, g_post
    except Exception:
        return None

# -----------------------------------------------------------------------------
# FIND OFFICIEL SIDE FOR ÉN RÆKKE
# -----------------------------------------------------------------------------
async def _find_official_site_for_row(row: pd.Series, session: ClientSession) -> dict | None:
    company = (row.get("Vrvirksomhed_virksomhedMetadata_nyesteNavn_navn") or "").strip()
    by = (row.get("Vrvirksomhed_virksomhedMetadata_nyesteBeliggenhedsadresse_postdistrikt") or "").strip()
    postnr = str(row.get("Vrvirksomhed_virksomhedMetadata_nyesteBeliggenhedsadresse_postnummer") or "").strip()
    cvr = re.sub(r"\D", "", str(row.get("Vrvirksomhed_cvrNummer") or "")) or None
    if not company:
        return None

    brand = _brand_tokens(company)

    # 0) Places først
    try:
        via_places = await _find_official_site_via_places(company, by, postnr, cvr, session)
        if via_places:
            reg, place_id, website_raw, g_city, g_post = via_places
            if reg and not _is_bad_website_domain(reg):
                host_left = reg.split(".")[0]
                if brand and not any(t in host_left for t in brand):
                    pass
                else:
                    return {
                        "domain": reg,
                        "source": "google_places",
                        "query": f"{company} {postnr} {by}".strip(),
                        "engine": "places",
                        "candidates": [reg],
                        "score": 0,
                        "confirm_bonus": 0,
                        "place_id": place_id,
                        "website_raw": website_raw,
                        "gplace_by": g_city or "",
                        "gplace_postnr": g_post or "",
                    }
    except Exception:
        pass

    # 1) Web-søgning
    company_tokens = _tokenize_name_all(company)
    q_bits = [company, by or "", postnr or "", "site:.dk"]
    if cvr:
        q_bits.append(cvr)
    q = " ".join([b for b in q_bits if b]).strip()

    engine_used: list[str] = []
    results = await _search_web(q, session=session, max_results=8, engine_out=engine_used)
    if not results:
        return None

    scored: list[tuple[int, str, int]] = []
    candidates_list: list[str] = []

    for item in results:
        url = item["url"]
        host = _netloc(url)
        if not host or _is_bad_website_domain(host):
            continue

        reg = _registrable_domain(host)
        title = (item.get("title") or "").lower()
        snippet = (item.get("snippet") or "").lower()
        host_left = reg.split(".")[0]

        has_brand = (not brand) or any((t in host_left) or (t in title) or (t in snippet) for t in brand)
        if not has_brand:
            continue

        score = 0
        if reg.endswith(".dk"):
            score += 2
        name_hits = sum(1 for t in company_tokens if t in title or t in snippet)
        score += min(name_hits, 3)
        if brand and any(t in host_left for t in brand):
            score += 3
        if "vvs" in host_left:
            score += 1
        if "el" in host_left:
            score += 1

        confirm_bonus = await _quick_confirm_company(session, f"https://{host}", company_tokens, cvr)
        score += confirm_bonus

        brand_hit_in_domain = bool(brand and any(t in host_left for t in brand))
        if (not brand_hit_in_domain) and (confirm_bonus < 2):
            continue  # skip svag kandidat

        scored.append((score, reg, confirm_bonus))
        if reg not in candidates_list:
            candidates_list.append(reg)

    if not scored:
        return None

    scored.sort(reverse=True, key=lambda x: x[0])
    best_score, best_reg, best_confirm = scored[0]
    return {
        "domain": best_reg,
        "source": f"web_search_{(engine_used[0] if engine_used else 'unknown')}",
        "query": q,
        "engine": (engine_used[0] if engine_used else "unknown"),
        "candidates": candidates_list[:3],
        "score": int(best_score),
        "confirm_bonus": int(best_confirm),
        "place_id": "",
        "website_raw": f"https://{best_reg}",
        "gplace_by": "",
        "gplace_postnr": "",
    }

# -----------------------------------------------------------------------------
# UDFYLD/RET HJEMMESIDER VIA SØGNING
# -----------------------------------------------------------------------------
async def _fill_websites_via_search(df: pd.DataFrame) -> pd.DataFrame:
    candidates_idx: list[int] = []
    for i, row in df.iterrows():
        current = (row.get("Hjemmeside") or "").strip().lower()
        current = re.sub(r"^https?://", "", current).rstrip("/")
        needs = (not current) or current == "n/a" or _is_bad_website_domain(current)
        if needs:
            candidates_idx.append(i)

    if not candidates_idx:
        return df

    for col in ["GPlaceBy", "GPlacePostnr"]:
        if col not in df.columns:
            df[col] = "N/A"

    async with aiohttp.ClientSession() as session:
        tasks = [_find_official_site_for_row(df.loc[i], session) for i in candidates_idx]
        found = await asyncio.gather(*tasks, return_exceptions=True)

    for i, meta in zip(candidates_idx, found):
        if isinstance(meta, dict) and meta.get("domain"):
            dom = meta["domain"]
            if not _is_bad_website_domain(dom):
                df.at[i, "Hjemmeside"] = dom
                src_raw = meta.get("source", "N/A")
                if isinstance(src_raw, str) and src_raw.startswith("web_search_"):
                    src_pretty = f"web_search ({src_raw.replace('web_search_', '', 1) or 'unknown'})"
                else:
                    src_pretty = src_raw
                df.at[i, "HjemmesideKilde"] = src_pretty
                df.at[i, "HjemmesideSøgeord"] = meta.get("query", "N/A")
                df.at[i, "HjemmesideKandidater"] = "; ".join(meta.get("candidates", [])[:3]) or "N/A"
                df.at[i, "HjemmesideScore"] = meta.get("score", "N/A")
                df.at[i, "HjemmesideConfirm"] = meta.get("confirm_bonus", "N/A")
                df.at[i, "HjemmesidePlaceId"] = meta.get("place_id", "N/A") or "N/A"
                df.at[i, "HjemmesideRå"] = meta.get("website_raw", "N/A")
                gp_by = meta.get("gplace_by", "") or ""
                gp_pc = meta.get("gplace_postnr", "") or ""
                if gp_by or gp_pc:
                    df.at[i, "GPlaceBy"] = gp_by or "N/A"
                    df.at[i, "GPlacePostnr"] = gp_pc or "N/A"
        elif isinstance(meta, Exception):
            df.at[i, "HjemmesideKilde"] = "error"
            df.at[i, "HjemmesideNotat"] = f"{type(meta).__name__}"
        else:
            df.at[i, "HjemmesideKilde"] = "none"
    return df

# -----------------------------------------------------------------------------
# POST-VALIDERING (uden crawl)
# -----------------------------------------------------------------------------
def _post_validate_web_candidates(df: pd.DataFrame) -> pd.DataFrame:
    for col in ["HjemmesideKilde","Hjemmeside","HjemmesideSøgeord",
                "HjemmesideBemærkning","HjemmesidePreLiveBool",
                "Vrvirksomhed_virksomhedMetadata_nyesteNavn_navn",
                "Vrvirksomhed_virksomhedMetadata_nyesteBeliggenhedsadresse_bynavn",
                "Vrvirksomhed_virksomhedMetadata_nyesteBeliggenhedsadresse_postnummer",
                "GPlaceBy","GPlacePostnr"]:
        if col not in df.columns:
            df[col] = pd.NA

    def _weak_web(row) -> bool:
        src = str(row.get("HjemmesideKilde","") or "")
        if not src.startswith("web_search"):
            return False
        dom = _domain_from_url(str(row.get("Hjemmeside","") or ""))
        name = str(row.get("Vrvirksomhed_virksomhedMetadata_nyesteNavn_navn","") or "")

        strong = _strong_tokens(name)  # ≥5, uden stopord
        if strong:
            return not any(t in dom for t in strong)

        # fallback: kerne-tokens (≥3) uden stopord, hvis ingen strong tokens
        core = [t for t in _tokenize_name_all(name) if len(t) >= 3 and t not in BRAND_STOPWORDS]
        return not any(t in dom for t in core)

    weak_mask = df.apply(_weak_web, axis=1)
    df.loc[weak_mask, "HjemmesidePreLiveBool"] = False
    df.loc[weak_mask, "HjemmesideBemærkning"]  = "svagt match (web_search)"

    def _gp_mismatch(row) -> bool:
        src = str(row.get("HjemmesideKilde","") or "")
        if src != "google_places":
            return False
        return _places_mismatch(
            row.get("Vrvirksomhed_virksomhedMetadata_nyesteBeliggenhedsadresse_bynavn",""),
            row.get("Vrvirksomhed_virksomhedMetadata_nyesteBeliggenhedsadresse_postnummer",""),
            row.get("GPlaceBy",""),
            row.get("GPlacePostnr",""),
        )

    gp_mask = df.apply(_gp_mismatch, axis=1)
    df.loc[gp_mask, "HjemmesidePreLiveBool"] = False
    df.loc[gp_mask, "HjemmesideBemærkning"]  = "Ikke samme virksomhed"

    return df

# -----------------------------------------------------------------------------
# HOVEDFUNKTION
# -----------------------------------------------------------------------------
def clean_and_prepare_cvr_data(raw_hits: List[Dict[str, Any]]) -> pd.DataFrame:
    df = pd.json_normalize(raw_hits, sep="_")

    # DEBUG: Print alle kolonner der indeholder ansat/beskaeftigelse
    print("=== DEBUG: CVR kolonner med ansat/beskæftigelse ===")
    for c in df.columns:
        if "beskaeft" in c.lower() or "ansat" in c.lower():
            print(" -", c)
    print("=== I ALT", len(df.columns), "kolonner ===")

    # Branche
    df["Branchekode"] = df.get("Vrvirksomhed_virksomhedMetadata_nyesteHovedbranche_branchekode", "N/A").fillna("N/A")
    df["Branchetekst"] = df.get("Vrvirksomhed_virksomhedMetadata_nyesteHovedbranche_branchetekst", "N/A").fillna("N/A")

    # Ansatte
    _col_ansatte          = "Vrvirksomhed_virksomhedMetadata_nyesteAarsbeskaeftigelse_antalAnsatte"
    if _col_ansatte in df.columns:
        df["AntalAnsatte"] = df[_col_ansatte].fillna("N/A")
    else:
        df["AntalAnsatte"] = "N/A"

    # Kontaktinfo
    contact_df = (
        df.get("Vrvirksomhed_virksomhedMetadata_nyesteKontaktoplysninger", pd.Series(dtype=object))
        .apply(_parse_contact_value)
        .apply(pd.Series)
    )
    df[["Telefon", "Email", "Hjemmeside"]] = contact_df

    # Direktørnavne
    df["Direktørnavn"] = (
        df.get("Vrvirksomhed_deltagerRelation", pd.Series(dtype=object))
        .apply(lambda x: _get_all_directors(x) if isinstance(x, list) else "N/A")
    )

    # Kun aktive
    df = df[
        df["Vrvirksomhed_virksomhedMetadata_sammensatStatus"].str.upper().isin({"AKTIV", "NORMAL"})
    ].copy()

    # Dynamiske brand-stopord
    _install_dynamic_brand_stopwords(df)

    # Infer websites fra emails
    df = _infer_website_from_email(df)

    # Supplér/ret via Places/web-søgning
    try:
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        df = loop.run_until_complete(_fill_websites_via_search(df))
    except Exception as e:
        log.warning(f"Website search augmentation skipped due to error: {e}")

    # --- Post-validering (login/svagt match/mismatch) før liveness -----------
    df = _post_validate_web_candidates(df)
    pre_live = list(df.get("HjemmesidePreLiveBool", pd.Series([pd.NA] * len(df))))
    pre_rem  = list(df.get("HjemmesideBemærkning",  pd.Series([""]     * len(df))))
    # --------------------------------------------------------------------------

    # Liveness-check
    raw_domains = df["Hjemmeside"].fillna("").astype(str)
    display_domains = [_strip_host(s) if s and s.lower() != "n/a" else "" for s in raw_domains]
    http_domains = [_to_idna(h) if h else "" for h in display_domains]

    valid_idx = [i for i, host in enumerate(http_domains) if host]
    urls_for_async = [http_domains[i] for i in valid_idx]

    if urls_for_async:
        try:
            try:
                loop = asyncio.get_event_loop()
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
            liveness, remarks = loop.run_until_complete(_check_website_liveness(urls_for_async))
        except Exception as e:
            log.warning(f"Liveness check failed: {e}")
            liveness, remarks = [], []
    else:
        liveness, remarks = [], []

    total_rows = len(df)
    liveness_full = ["N/A"] * total_rows
    remarks_full = [""] * total_rows
    async_pos = 0
    for i in valid_idx:
        liveness_full[i] = liveness[async_pos] if async_pos < len(liveness) else False
        remarks_full[i] = remarks[async_pos] if async_pos < len(remarks) else "error"
        async_pos += 1

    # --- FUSION: bevar pre-filtering (DON'T OVERRIDE FALSE) -------------------
    def _is_false(v) -> bool:
        return v is False or (isinstance(v, str) and v.strip().lower() == "false")

    final_live = ["N/A"] * total_rows
    final_rem  = [""] * total_rows
    for i in range(total_rows):
        if _is_false(pre_live[i]):
            final_live[i] = False
            final_rem[i]  = pre_rem[i] or remarks_full[i] or ""
        else:
            final_live[i] = liveness_full[i]
            final_rem[i]  = remarks_full[i]

    # Ryd "no response" for rækker der endte med live=True
    for i in range(len(final_live)):
        if final_live[i] is True and (not final_rem[i] or str(final_rem[i]).strip().lower() == "no response"):
            final_rem[i] = ""

    df["HjemmesideLiveBool"]   = [(True if v is True else (False if v is False else pd.NA)) for v in final_live]
    df["HjemmesideBemærkning"] = [str(x or "") for x in final_rem]

    # Ingen hjemmeside => bool=<NA>, remark=""
    no_site_mask = df["Hjemmeside"].astype(str).str.strip().str.lower().isin(["", "n/a"])
    df.loc[no_site_mask, "HjemmesideLiveBool"]   = pd.NA
    df.loc[no_site_mask, "HjemmesideBemærkning"] = ""

    # Normalisering af tekst i remark (valgfrit – beholdt)
    df["HjemmesideBemærkning"] = df["HjemmesideBemærkning"].replace(
        {"ssl error (tolerated)": "SSL-fejl — brugte HTTP"}
    )

    # 3) Human-venlig kilde-etiket
    def _pretty_source(s: str) -> str:
        s = (s or "").strip()
        if s == "existing":
            return "cvr"
        if s.startswith("web_search_"):
            eng = s.replace("web_search_", "", 1) or "unknown"
            return f"web_search ({eng})"
        return s or "N/A"
    if "HjemmesideKilde" in df.columns:
        df["HjemmesideKilde"] = df["HjemmesideKilde"].astype(str).apply(_pretty_source)

    # 4) Score/Confirm kun ved web_search – ellers N/A
    if "HjemmesideKilde" in df.columns:
        is_web_search = df["HjemmesideKilde"].str.startswith("web_search")
        for col in ["HjemmesideScore", "HjemmesideConfirm"]:
            if col in df.columns:
                df.loc[~is_web_search, col] = "N/A"

    # 5) Sikre string-typer for CSV-stabilitet
    for col in ["HjemmesideScore", "HjemmesideConfirm", "HjemmesidePlaceId",
                "HjemmesideRå", "HjemmesideNotat"]:
        if col in df.columns:
            df[col] = df[col].astype(object).fillna("N/A")
    
    # ---------------------------------------------------------------------------

    # Endelig opsummering
    if LOG_SUMMARY:
        final_true  = int((pd.Series(final_live) == True).sum())
        final_false = int((pd.Series(final_live) == False).sum())
        final_na    = len(final_live) - final_true - final_false
        log.info(f"Final website status — true={final_true}, false={final_false}, na={final_na}, total={len(df)}")

    # Kolonnerækkefølge (inkl. HjemmesideLiveBool)
    cols = [
        "Vrvirksomhed_cvrNummer",
        "Vrvirksomhed_virksomhedMetadata_nyesteNavn_navn",
        "Direktørnavn",
        "Telefon",
        "Email",
        "Hjemmeside",
        "HjemmesideLiveBool",   
        "HjemmesideBemærkning",
        "HjemmesideKilde",
        "HjemmesideSøgeord",
        "HjemmesideKandidater",
        "HjemmesideScore",
        "HjemmesideConfirm",
        "HjemmesidePlaceId",
        "HjemmesideRå",
        "HjemmesideNotat",
        "GPlaceBy",
        "GPlacePostnr",
        "Vrvirksomhed_virksomhedMetadata_nyesteBeliggenhedsadresse_vejnavn",
        "Vrvirksomhed_virksomhedMetadata_nyesteBeliggenhedsadresse_husnummerFra",
        "Vrvirksomhed_virksomhedMetadata_nyesteBeliggenhedsadresse_postnummer",
        "Vrvirksomhed_virksomhedMetadata_nyesteBeliggenhedsadresse_postdistrikt",
        "Branchekode",
        "Branchetekst",
        "AntalAnsatte",
        "Vrvirksomhed_virksomhedMetadata_nyesteVirksomhedsform_virksomhedsformkode",
        "Vrvirksomhed_virksomhedMetadata_nyesteVirksomhedsform_langBeskrivelse",
        "Vrvirksomhed_virksomhedMetadata_sammensatStatus",
        "Vrvirksomhed_virksomhedMetadata_ophørsDato",
    ]
    for c in cols:
        if c not in df.columns:
            df[c] = "N/A"

    log.debug("Rensede DataFrame med %s aktive rækker", len(df))
    return df.reindex(columns=cols).fillna("N/A")

# -----------------------------------------------------------------------------
# Infer 'Hjemmeside' fra Email
# -----------------------------------------------------------------------------
def _infer_website_from_email(df: pd.DataFrame) -> pd.DataFrame:
    for col in (
        "HjemmesideKilde", "HjemmesideNotat", "HjemmesideSøgeord",
        "HjemmesideKandidater", "HjemmesideScore", "HjemmesideConfirm",
        "HjemmesidePlaceId", "HjemmesideRå",
    ):
        if col not in df.columns:
            df[col] = "N/A"

    def guess_website(row):
        if row["Hjemmeside"] and row["Hjemmeside"] != "N/A":
            return row["Hjemmeside"], "cvr", ""
        if row["Email"] == "N/A" or not row["Email"]:
            return "N/A", "N/A", "no email"

        emails = str(row["Email"]).split(";")
        for email in emails:
            email = email.strip()
            if not email or not EMAIL_REGEX.match(email):
                continue
            try:
                domain = email.split("@")[1].lower()
                if domain in FREE_EMAIL_DOMAINS:
                    continue
                if any(suffix.match(domain) for suffix in EXCLUDED_SUFFIXES):
                    continue
                cleaned_domain = clean_domain(domain)
                if not DOMAIN_REGEX.match(cleaned_domain):
                    continue
                if _is_aggregator(cleaned_domain):
                    continue
                return cleaned_domain, "email", f"inferred from {domain}"
            except IndexError:
                continue
        return "N/A", "N/A", "no valid domain in email"

    tmp = df.apply(guess_website, axis=1, result_type="expand")
    tmp.columns = ["_w", "_src", "_note"]

    df["Hjemmeside"] = tmp["_w"].where(df["Hjemmeside"].isin(["N/A", None, ""]), df["Hjemmeside"])
    fill_mask = (tmp["_w"].notna()) & (tmp["_w"] != "N/A")
    df.loc[fill_mask, "HjemmesideKilde"] = tmp.loc[fill_mask, "_src"]
    df.loc[fill_mask, "HjemmesideNotat"] = tmp.loc[fill_mask, "_note"]

    log.debug("Inferred websites (email/existing) for %s rows", (df["Hjemmeside"] != "N/A").sum())
    return df
