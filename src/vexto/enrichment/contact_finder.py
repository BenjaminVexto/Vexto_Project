#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import logging
import math
import json
import time
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple, Set
import pandas as pd

# Third-party (optional) imports
try:
    import dns.resolver  # MX/A lookups
except Exception:  # pragma: no cover
    dns = None  # type: ignore

import smtplib
import requests
from urllib.parse import urlparse

# --- BeautifulSoup import + type-safe aliasing for Pylance ---
# Kørbar alias til runtime
try:
    from bs4 import BeautifulSoup as _BeautifulSoup  # runtime symbol
    try:
        from bs4 import XMLParsedAsHTMLWarning as _XMLWarn  # til log-støj
    except Exception:
        _XMLWarn = None  # type: ignore
    BS4_AVAILABLE = True
except Exception:
    _BeautifulSoup = None  # type: ignore[assignment]
    _XMLWarn = None  # type: ignore
    BS4_AVAILABLE = False

# Pylance-friendly typing alias for BeautifulSoup: try real type, fallback to Any at runtime.
try:
    from bs4 import BeautifulSoup as BeautifulSoupType  # for type hints (Pylance ser en rigtig type)
except Exception:  # pragma: no cover
    from typing import Any as BeautifulSoupType  # fallback, hvis bs4 ikke er installeret

# Slå BS4's XMLParsedAsHTMLWarning fra (fx sitemaps) uden at irritere Pylance
import warnings as _bs4_warnings
if _XMLWarn is not None:
    _bs4_warnings.filterwarnings("ignore", category=_XMLWarn)

# Contact fetchers (synkrone dele) – valgfri
_CF_MOD = None
try:
    for _name in [
        "vexto.scoring.contact_fetchers",
        "src.vexto.scoring.contact_fetchers",
        "contact_fetchers",
        "src_vexto_scoring_contact_fetchers",
    ]:
        try:
            _CF_MOD = __import__(_name, fromlist=["*"])
            break
        except Exception:
            continue
except Exception:
    _CF_MOD = None

try:
    from apify_client import ApifyClient
    APIFY_AVAILABLE = True
except Exception:
    APIFY_AVAILABLE = False

try:
    from googleapiclient.discovery import build
    from googleapiclient.errors import HttpError
    GOOGLE_API_AVAILABLE = True
except Exception:
    GOOGLE_API_AVAILABLE = False

try:
    import tldextract
    TLD_AVAILABLE = True
except Exception:
    TLD_AVAILABLE = False

from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Konfiguration & globals
# ---------------------------------------------------------------------------
load_dotenv()
log = logging.getLogger(__name__)

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GOOGLE_CSE_ID = os.getenv("GOOGLE_CSE_ID")
USE_LINKEDIN = os.getenv("USE_LINKEDIN", "True").lower() == "true"
USE_SMTP = os.getenv("USE_SMTP", "True").lower() == "true"
APIFY_ACTOR_ID = os.getenv("APIFY_ACTOR_ID", "cIdqlEvw6afc1do1p")
APIFY_API_KEY = os.getenv("APIFY_API_KEY")
ZERBOUNCE_API_KEY = os.getenv("ZERBOUNCE_API_KEY")
HTTP_TIMEOUT = int(os.getenv("HTTP_TIMEOUT", "12"))

USE_PLAYWRIGHT_CONTACT_FALLBACK = os.getenv("USE_PLAYWRIGHT_CONTACT_FALLBACK", "True").lower() == "true"
PLAYWRIGHT_CONTACT_TIMEOUT = int(os.getenv("PLAYWRIGHT_CONTACT_TIMEOUT", "20"))

ALLOW_BROAD_FALLBACK = os.getenv("ALLOW_BROAD_FALLBACK", "True").lower() == "true"

# Early-stop: spring Google over hvis Trin 1 fandt en førsteparts-mail (runtime-optimering)
EARLY_STOP_ON_WEBSITE_HIT = os.getenv("EARLY_STOP_ON_WEBSITE_HIT", "True").lower() == "true"

LOG_LIST_LIMIT = int(os.getenv("LOG_LIST_LIMIT", "10"))
MAX_PHONES_PER_SOURCE = int(os.getenv("MAX_PHONES_PER_SOURCE", "5"))

# Caches
_GOOGLE_SERVICE = None  # type: ignore
_CACHE_GOOGLE_LINKEDIN: Dict[str, Optional[str]] = {}
_CACHE_APIFY_EMPLOYEES: Dict[str, List[Dict[str, Any]]] = {}
_CACHE_FETCHED_URL: Dict[str, str] = {}
_CACHE_RENDERED_URL: Dict[str, str] = {}
# Domæne → (contact, emails, phones) (kun når cachebar)
_CACHE_DOMAIN_RESULT: Dict[str, Tuple[Optional[Dict[str, Any]], List[str], List[str]]] = {}
# Firmanavn → (domain, score)
_CACHE_COMPANY_INFERRED_DOMAIN: Dict[str, Tuple[Optional[str], int]] = {}
_HTTP_CLIENT_MOD = None
_HTTP_CLIENT_MOD_LOGGED = False

# Titler (prioriteret – marketing/digital -> sales -> C-level)
PRIORITY_TITLES = [
    # Lag 1
    'marketing director', 'marketingdirektør', 'head of marketing', 'marketingchef',
    'head of digital', 'digital chef', 'digital manager',
    'head of ecommerce', 'e-handelschef', 'ecommerce manager',
    'seo manager', 'seo specialist', 'seo ansvarlig',
    'sem manager', 'ppc manager',
    'web manager', 'webmaster', 'webansvarlig',
    'chief marketing officer', 'cmo',
    # Lag 2
    'head of sales', 'salgschef', 'sales director', 'salgsdirektør',
    'kommunikationschef', 'head of communications',
    # Lag 3
    'indehaver', 'ejer', 'medejer',
    'chief executive officer', 'ceo',
    'administrerende direktør', 'direktør',
    'økonomidirektør', 'cfo'
    # (bevidst fjernet 'partner' for at undgå støj)
]

EMAIL_REGEX = re.compile(r"[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}", re.IGNORECASE)

# Navnemønster (1–3 led)
NAME_TOKEN = r"[A-ZÆØÅ][a-zæøåA-ZÆØÅ\-']+"
NAME_PATTERN = re.compile(rf"\b({NAME_TOKEN})(\s+{NAME_TOKEN}){{0,2}}\b")

# “Firma-hints” som ikke bør tælle som person-efternavne
NAME_COMPANY_HINTS = {
    'aps','a/s','ivs','i/s','k/s','holding','group','entreprise','service','byg','byggeri',
    'rengøring','co','a.m.b.a','firma','solutions','partner','partners','anpartsselskab',
    'a/s.','aps.'
}

# Blacklist tokens der ofte er schema.org-typer eller junk (udvidet)
NAME_BLACKLIST = {s.lower() for s in [
    # eksisterende
    "creativework", "organization", "website", "breadcrumblist", "article",
    "person", "false", "true", "date", "her", "sammenligne",
    "ydelser", "om", "fra", "år", "aars", "erfaring", "hvem", "vi", "kontakt",
    "managing", "director",
    # navigation/UI
    "information", "about", "mobile", "desktop", "menu", "navigation", "header", "footer",
    "home", "hjem", "forside", "blog", "nyheder", "news", "cookie", "gdpr",
    # virksomhedstermer
    "service", "services", "løsninger", "solutions", "produkter", "company", "group",
    # erhverv UDEN efternavn
    "tømrer", "murer", "maler", "elektriker", "vvs", "snedker",
    # geo/adresse
    "greve", "københavn", "aarhus", "odense", "vej", "gade", "allé", "alle", "plads", "torv", "boulevard"
]}

# Fallback DK-telefon (bruges kun som sidste udvej – DOM-first anvendes nedenfor)
PHONE_REGEX_FALLBACK = re.compile(r"\b(?:\+?45[\s\-]?)?(?:\d{2}[\s\-]?){4}\b")

# Tredjepart (ikke-førsteparts) domæner
THIRDPARTY_DOMAINS = {
    "linkedin.com", "facebook.com", "instagram.com", "x.com", "twitter.com",
    "proff.dk", "cvr.dk", "virk.dk", "krak.dk", "degulesider.dk",
    "findsmiley.dk", "trustpilot.dk", "dk.trustpilot.com", "maps.google.com",
    "google.com", "youtube.com", "bing.com",
    # Kataloger/SEO-sider
    "118.dk", "anmeld-haandvaerker.dk", "haandvaerker.dk", "hvad-koster-et-nyt-tag.dk",
    "minby.dk", "byggetilbud.dk", "byggetilbud-co.dk", "byggetilbud.net",
}

SEO_JUNK_TOKENS = {
    "byggetilbud", "hulmursisolering", "tagdækker", "minby", "anmeld-håndværker",
    "billig", "tilbud", "rabat", "nyt-tag"
}

# Mistænkelige telefonmønstre
SUSPECT_PHONE_SUBSTRINGS = {"214748", "142857", "918367"}
SUSPECT_PHONE_EXACT = {"91458878", "85714285", "85714286", "21474836", "91836734", "42857142"}

# Generiske aliaser (nedprioriteres ved emailvalg)
GENERIC_EMAIL_USERS = {"info", "kontakt", "mail", "hello", "hej", "support", "admin",
                       "postmaster", "office", "sales", "marketing"}

# ---------------------------------------------------------------------------
# Utils
# ---------------------------------------------------------------------------

def _normalize_danish(s: str) -> str:
    if not isinstance(s, str):
        return s
    mapping = {"æ": "ae", "ø": "oe", "å": "aa", "Æ": "Ae", "Ø": "Oe", "Å": "Aa"}
    for k, v in mapping.items():
        s = s.replace(k, v)
    return s

def _to_bool(x: Any) -> bool:
    if isinstance(x, bool):
        return x
    if x is None:
        return False
    if isinstance(x, (int, float)):
        return x != 0
    s = str(x).strip().lower()
    true_set  = {"1","true","t","yes","y","ja","sand","on"}
    false_set = {"0","false","f","no","n","nej","falsk","off",""}
    if s in true_set:
        return True
    if s in false_set:
        return False
    return False

def _split_names_field(val: Any) -> List[str]:
    """Split et felt med potentielt flere navne (fx 'A; B; C') og rens."""
    if val is None:
        return []
    s = str(val).strip()
    if not s:
        return []
    parts = re.split(r"[;|,/•]+", s)
    out = []
    for p in parts:
        p = p.strip()
        # fjern dobbeltmellemrum
        p = re.sub(r"\s+", " ", p)
        if len(p) >= 3:
            out.append(p)
    return out

def _choose_director_name(val: Any) -> Optional[str]:
    """Vælg første brugbare direktørnavn, hvis flere er listet."""
    for cand in _split_names_field(val):
        if NAME_PATTERN.search(cand) and not _looks_like_company_name(cand):
            # kræv mindst 2 tokens for at undgå enkeltnavne
            toks = cand.split()
            if len(toks) >= 2:
                return cand
    # fallback til første stribe tekst
    parts = _split_names_field(val)
    return parts[0] if parts else None

def _get_employee_count(row: pd.Series) -> int:
    # Kendte nøgler først
    preferred_keys = [
        "antal_ansatte", "AntalAnsatte", "employee_count", "employees", "ansatte",
        "Vrvirksomhed_antalAnsatte",
        "Vrvirksomhed_virksomhedMetadata_antalAnsatte",
        "virksomhedMetadata_antalAnsatte",
        "antalAarsvaerk", "Vrvirksomhed_antalAarsvaerk",
    ]
    for key in preferred_keys:
        if key in row and pd.notna(row[key]):
            v = _parse_emp_value(row[key])
            if v is not None:
                return v

    # Heuristik: scan alle kolonnenavne for 'ansat' eller 'employee'
    best: Optional[int] = None
    for col in row.index:
        cl = str(col).lower()
        if "ansat" in cl or "employee" in cl:
            v = _parse_emp_value(row[col])
            if v is not None:
                if best is None or v > best:
                    best = v

    return best if best is not None else -1

def _get_registered_domain(url: Optional[str]) -> Optional[str]:
    if not url or not isinstance(url, str):
        return None
    try:
        if not (url.startswith("http://") or url.startswith("https://")):
            url = "http://" + url
        host = urlparse(url).netloc
        if not host:
            return None
        host = host.lower()
        if host.startswith("www."):
            host = host[4:]
        if TLD_AVAILABLE:
            ext = tldextract.extract(host)
            if ext.registered_domain:
                return ext.registered_domain
        return host
    except Exception:
        return None

def _registered_brand_base(host: Optional[str]) -> Optional[str]:
    """Returner 'brand-basen' for et host/domæne – fx 'mts-byg.dk' -> 'mtsbyg'."""
    if not host:
        return None
    host = host.strip().lower()
    # Fjern evt. schema og/eller path
    host = re.sub(r"^https?://", "", host).split("/", 1)[0]
    # Brug tldextract hvis muligt
    try:
        if TLD_AVAILABLE:
            ext = tldextract.extract(host)
            core = ext.domain  # fx 'mts-byg'
        else:
            core = host.split(".")[-2] if "." in host else host
    except Exception:
        core = host.split(".")[-2] if "." in host else host
    core = re.sub(r"[\W_]+", "", core)  # fjern bindestreger, underscores m.m.
    return core or None

def _is_brand_similar_domain(email_domain: Optional[str], site_domain: Optional[str]) -> bool:
    """Er email-domænet brand-lignende ift. sitets domæne? (ignorer bindestreger/WWW)."""
    if not email_domain or not site_domain:
        return False
    eb = _registered_brand_base(email_domain)
    sb = _registered_brand_base(site_domain)
    if not eb or not sb:
        return False
    return eb == sb or eb in sb or sb in eb

def _extract_emails(text: str, domain: Optional[str] = None) -> List[str]:
    if not text:
        return []
    found = set(m.group(0).lower() for m in EMAIL_REGEX.finditer(text))
    if domain:
        found = {e for e in found if domain in e.split('@')[-1]}
    return sorted(found)

def _is_suspect_phone(digits: str) -> bool:
    if digits in SUSPECT_PHONE_EXACT:
        return True
    if any(sub in digits for sub in SUSPECT_PHONE_SUBSTRINGS):
        return True
    # 142857 rotationer (6-cifret cyklus)
    rotations = ["142857", "428571", "285714", "857142", "571428", "714285"]
    if any(rot in digits for rot in rotations):
        return True
    return False

def _extract_phones_from_dom(soup: BeautifulSoupType) -> List[str]:
    phones: List[str] = []
    # a[href^="tel:"]
    for a in soup.find_all("a", href=True):
        href = (a.get("href") or "").strip()
        if href.lower().startswith("tel:"):
            phones.append(href.split(":", 1)[1])
    # itemprop=telephone
    for t in soup.find_all(attrs={"itemprop": "telephone"}):
        phones.append(t.get_text(" ", strip=True))
    # klasse/id hints
    for el in soup.find_all(True, attrs={"class": True}):
        cls = " ".join(el.get("class") or []).lower()
        if any(k in cls for k in ["telefon", "tlf", "mobile", "phone"]):
            phones.append(el.get_text(" ", strip=True))
    return _normalize_many_phones(phones)

def _extract_emails_from_dom(soup: BeautifulSoupType, domain: Optional[str]) -> List[str]:
    emails: Set[str] = set()
    # a[href^="mailto:"]
    for a in soup.find_all("a", href=True):
        href = (a.get("href") or "").strip()
        if href.lower().startswith("mailto:"):
            addr = href.split(":", 1)[1].split("?")[0]
            if EMAIL_REGEX.fullmatch(addr):
                emails.add(addr.lower())
    # Fald tilbage: tekst
    body_text = soup.get_text(" ", strip=True)
    for em in _extract_emails(body_text, domain):
        emails.add(em)
    return sorted(emails)

def _extract_phones(text: str) -> List[str]:
    # Kun fallback-tekst – dom-first håndteres andetsteds
    if not text:
        return []
    raw = [m.group(0) for m in PHONE_REGEX_FALLBACK.finditer(text)]
    return _normalize_many_phones(raw)

def _email_username(email: str) -> str:
    return email.split("@", 1)[0].lower()

def _normalize_name_tokens(full_name: str) -> Tuple[str, str]:
    parts = [p for p in _normalize_danish(full_name.lower()).split() if p]
    first = re.sub(r"\W+", "", parts[0]) if parts else ""
    last = re.sub(r"\W+", "", parts[-1]) if len(parts) > 1 else ""
    return first, last

def _score_email_for_name(email: str, full_name: str, company_name: Optional[str] = None) -> int:
    """Højere score = bedre match mellem email og navn. (nu med initial+initial og brand-bonus)"""
    user = re.sub(r"[.\-_]", "", _email_username(email))
    first, last = _normalize_name_tokens(full_name)
    if not first:
        return 0
    patterns = {
        f"{first}{last}",
        f"{first}",
        f"{first[0]}{last}" if last else "",
        f"{first}{last[:1]}" if last else "",
        # nyt: initial+initial (fx tk for Torben Kempel)
        f"{first[:1]}{last[:1]}" if last else "",
    }
    score = 0
    if user in patterns:
        score += 4
    if user.startswith(first[0]) and last and last in user:
        score += 3
    if first in user and (last and last in user):
        score += 3

    # brand-bonus: hvis brugernavn indeholder brandelementer
    if company_name:
        brand_tokens = [t for t in re.split(r"\W+", _normalize_danish(company_name.lower())) if len(t) >= 4]
        if any(t in user for t in brand_tokens):
            score += 2

    if user in GENERIC_EMAIL_USERS:
        score -= 3
    return score

def _best_email_for_name(candidates: List[str], full_name: str, domain: Optional[str]) -> Optional[str]:
    if not candidates:
        return None
    # filter: tillad exact first-party ELLER brand-lignende domæner
    if domain:
        kept = []
        for e in candidates:
            dom_e = e.split("@")[-1].lower()
            if dom_e.endswith(domain.lower()) or _is_brand_similar_domain(dom_e, domain):
                kept.append(e)
        candidates = kept
    if not candidates:
        return None
    ranked = sorted(
        candidates,
        key=lambda e: (_score_email_for_name(e, full_name), 0 if _email_username(e) in GENERIC_EMAIL_USERS else 1),
        reverse=True,
    )
    return ranked[0]

def _normalize_dk_phone_num(s: str) -> Optional[str]:
    digits = re.sub(r"\D+", "", s)
    if digits.startswith("0045"):
        digits = digits[4:]
    elif digits.startswith("45") and len(digits) >= 10:
        digits = digits[2:]
    # Kun 8 cifre
    if len(digits) != 8:
        return None
    # Mistænkelige mønstre
    if _is_suspect_phone(digits):
        return None
    # Frasortér nummerserier med samme ciffer >= 6 gange (fx 83333333, 66666666)
    c = Counter(digits)
    if max(c.values()) >= 6:
        return None
    # Frasortér helt ens cifre (00000000, 11111111, ...)
    if len(set(digits)) == 1:
        return None
    return digits

def _normalize_many_phones(raws: List[str]) -> List[str]:
    out: List[str] = []
    seen: set = set()
    for r in raws:
        n = _normalize_dk_phone_num(r)
        if n and n not in seen:
            seen.add(n)
            out.append(n)
    return out

def _filter_suspect_phones(phones: List[str]) -> List[str]:
    good = []
    seen = set()
    for p in phones:
        p2 = _normalize_dk_phone_num(p)
        if p2 and p2 not in seen:
            seen.add(p2)
            good.append(p2)
    return good

def _parse_emp_value(val: Any) -> Optional[int]:
    if val is None:
        return None
    # floats/ints
    if isinstance(val, (int, float)) and not (isinstance(val, float) and (math.isnan(val))):
        try:
            iv = int(val)
            return iv if iv >= 0 else None
        except Exception:
            return None
    s = str(val).strip()
    if not s or s.lower() in {"nan", "none", "null"}:
        return None
    # Intervaller "10-19", "10 – 19", "10 til 19"
    m = re.search(r"(\d+)\s*(?:-|–|til)\s*(\d+)", s)
    if m:
        lower = int(m.group(1))
        # konservativt: brug nedre grænse
        return lower
    # Enkelt tal indlejret i tekst
    m2 = re.search(r"\d+", s)
    if m2:
        return int(m2.group(0))
    return None

def _fetch_url_text(url: str) -> str:
    if url in _CACHE_FETCHED_URL:
        return _CACHE_FETCHED_URL[url]
    try:
        headers = {"User-Agent": "Mozilla/5.0 (compatible; VextoContactFinder/1.0)"}
        r = requests.get(url, headers=headers, timeout=HTTP_TIMEOUT)
        r.raise_for_status()
        text = r.text
        _CACHE_FETCHED_URL[url] = text
        return text
    except Exception as e:
        log.debug(f"HTTP fetch failure for {url}: {e}")
        return ""

def _unique_preserve(seq: List[str]) -> List[str]:
    seen: Set[str] = set()
    out: List[str] = []
    for s in seq:
        if s not in seen:
            seen.add(s)
            out.append(s)
    return out

def _log_compact_list(label: str, items: List[str], limit: int = None) -> None:
    if limit is None:
        limit = LOG_LIST_LIMIT
    if not items:
        return
    shown = items[:limit]
    more = len(items) - len(shown)
    if more > 0:
        log.info(f"{label}: {shown} (+{more} flere)")
    else:
        log.info(f"{label}: {shown}")

# ---------------------------------------------------------------------------
# Google helpers
# ---------------------------------------------------------------------------

def _get_google_service():
    global _GOOGLE_SERVICE
    if _GOOGLE_SERVICE is not None:
        return _GOOGLE_SERVICE
    if not (GOOGLE_API_AVAILABLE and GOOGLE_API_KEY and GOOGLE_CSE_ID):
        log.info("Google CSE ikke tilgængelig (mangler lib eller nøgler)")
        _GOOGLE_SERVICE = None
    else:
        try:
            _GOOGLE_SERVICE = build("customsearch", "v1", developerKey=GOOGLE_API_KEY, cache_discovery=False)
        except Exception as e:
            log.warning(f"Kunne ikke initialisere Google CSE: {e}")
            _GOOGLE_SERVICE = None
    return _GOOGLE_SERVICE

def _google_search(query: str, num: int = 3) -> List[Dict[str, Any]]:
    service = _get_google_service()
    if not service:
        return []
    try:
        resp = service.cse().list(q=query, cx=GOOGLE_CSE_ID, num=num).execute()
        return resp.get('items', [])
    except HttpError as e:
        log.debug(f"Google CSE HttpError for '{query}': {e}")
    except Exception as e:
        log.debug(f"Google CSE error for '{query}': {e}")
    return []

def _same_registered_domain(a: Optional[str], b: Optional[str]) -> bool:
    if not a or not b:
        return False
    return _get_registered_domain(a) == _get_registered_domain(b)

def _is_thirdparty_domain(domain: Optional[str]) -> bool:
    if not domain:
        return True
    return domain in THIRDPARTY_DOMAINS

# ---------------------------------------------------------------------------
# LinkedIn via Apify helpers + Google person-fallback
# ---------------------------------------------------------------------------

def _find_company_linkedin_url(company_name: str) -> Optional[str]:
    if company_name in _CACHE_GOOGLE_LINKEDIN:
        return _CACHE_GOOGLE_LINKEDIN[company_name]
    if not company_name:
        return None
    if not _get_google_service():
        _CACHE_GOOGLE_LINKEDIN[company_name] = None
        return None
    query = f'"{company_name}" site:linkedin.com/company'
    items = _google_search(query, num=3)
    for it in items:
        url = it.get('link')
        if url and 'linkedin.com/company' in url:
            _CACHE_GOOGLE_LINKEDIN[company_name] = url
            log.info(f"[Google/LinkedIn] Fandt firmasiden: {url}")
            return url
    log.info(f"[Google/LinkedIn] Ingen firmasiden fundet for: {company_name}")
    _CACHE_GOOGLE_LINKEDIN[company_name] = None
    return None

def _query_linkedin_api(company_linkedin_url: str) -> List[Dict[str, Any]]:
    if company_linkedin_url in _CACHE_APIFY_EMPLOYEES:
        return _CACHE_APIFY_EMPLOYEES[company_linkedin_url]
    if not (APIFY_AVAILABLE and APIFY_API_KEY and company_linkedin_url):
        return []
    try:
        client = ApifyClient(APIFY_API_KEY)
        run_input = {"identifier": company_linkedin_url, "max_employees": 50}
        log.info(f"[Apify] Starter actor for: {company_linkedin_url}")
        run = client.actor(APIFY_ACTOR_ID).call(run_input=run_input)
        employees = list(client.dataset(run["defaultDatasetId"]).iterate_items())
        _CACHE_APIFY_EMPLOYEES[company_linkedin_url] = employees
        log.info(f"[Apify] {len(employees)} medarbejdere hentet")
        return employees
    except Exception as e:
        log.warning(f"[Apify] Fejl: {e}")
        return []

def _filter_employees_by_title(employees: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    for title_keyword in PRIORITY_TITLES:
        for emp in employees:
            headline = (emp.get('headline') or '').lower()
            if title_keyword in headline:
                log.info(f"[Apify] Match: {emp.get('fullname')} – {headline}")
                return emp
    return None

def _google_linkedin_person_lookup(company_name: Optional[str], domain: Optional[str]) -> Optional[Dict[str, Any]]:
    """Let efter personprofiler på LinkedIn via Google (til mikrovirksomheder)."""
    if not company_name and not domain:
        return None
    if not _get_google_service():
        return None
    terms = ['Owner', 'Founder', 'Indehaver', 'Ejer', 'Direktør', 'CEO', 'Partner']
    base = ' OR '.join(terms)
    pieces = []
    if company_name:
        pieces.append(f'"{company_name}"')
    if domain:
        pieces.append(f'"{domain}"')
    must = " OR ".join(pieces)
    q = f'site:linkedin.com/in ({must}) ({base})'
    items = _google_search(q, num=4)
    for it in items:
        url = it.get('link', '')
        title_txt = it.get('title', '') or ''
        snippet = it.get('snippet', '') or ''
        if 'linkedin.com/in' not in url:
            continue
        # Navn før første " - " eller " | "
        name_match = re.split(r"\s[-|–]\s", title_txt)
        fullname = name_match[0].strip() if name_match else None
        # Titel heuristik
        head = (title_txt + " " + snippet).lower()
        found_title = None
        for t in PRIORITY_TITLES:
            if t in head:
                found_title = t
                break
        if fullname and len(fullname.split()) <= 4:
            log.info(f"[Google/LinkedIn-person] Fandt: {fullname} – {found_title or 'ukendt titel'}")
            return {'fullname': fullname, 'headline': found_title or 'owner', 'confidence': 0.7, 'source': 'google'}
    return None

# ---------------------------------------------------------------------------
# Name/title extraction (tekst + DOM)
# ---------------------------------------------------------------------------

def _looks_like_company_name(fullname: str) -> bool:
    toks = [t.lower().strip(".,:;()") for t in fullname.split()]
    if any(t in NAME_COMPANY_HINTS for t in toks):
        return True
    if len(toks) >= 2 and toks[-1].endswith("ing"):
        return True
    return False

def _is_valid_person_name(fullname: Optional[str]) -> bool:
    """Nuanceret validering af personnavne (DK-venlig, tolerant for bindestreger/partikler)."""
    if not fullname or not isinstance(fullname, str):
        return False
    tokens = [t for t in re.split(r"\s+", fullname.strip()) if t]
    if len(tokens) < 2 or len(tokens) > 4:
        return False

    # disallow åbenlyse støj/brancher/geo
    low = [t.lower().strip(".,:;()") for t in tokens]
    if any(t in NAME_BLACKLIST for t in low):
        return False

    # afvis mønstre som "Tømrer Jensen" (erhverv + efternavn)
    job_prefixes = {'tømrer', 'elektriker', 'maler', 'vvs', 'snedker', 'murer'}
    if low[0] in job_prefixes and len(tokens) == 2:
        return False

    # tillad partikler
    allowed_particles = {'von', 'de', 'van', 'af', 'la'}
    name_tokens = [t for t in tokens if t.lower() not in allowed_particles]

    # kræv mindst to "rigtige" navne-ord (ikke i blacklist)
    name_tokens_clean = [t for t in name_tokens if t.lower() not in NAME_BLACKLIST and len(t) >= 2]
    if len(name_tokens_clean) < 2:
        return False

    # vær ikke rigid omkring stort begyndelsesbogstav på ALLE tokens (tillad fx "von")
    # men kræv dog at de fleste 'kerne'-tokens starter med stor bokstav eller er hyphenated navne
    major = [t for t in name_tokens_clean if not re.match(r"^[A-ZÆØÅ]", t)]
    if len(major) > 1:  # højst én undtagelse (fx 'von')
        return False

    if _looks_like_company_name(fullname):
        return False
    return True

def _extract_name_title_pairs(text: str) -> List[Dict[str, Any]]:
    if not isinstance(text, str) or not text:
        return []
    results: List[Dict[str, Any]] = []
    lower = text.lower()

    for title in PRIORITY_TITLES:
        if re.search(rf"\b{re.escape(title)}\b", lower):
            for m in re.finditer(re.escape(title), lower):
                start = max(0, m.start() - 120)
                end = min(len(text), m.end() + 120)
                window = text[start:end]
                for nm in re.finditer(NAME_PATTERN, window):
                    fullname = nm.group(0).strip()
                    tokens = [t for t in fullname.split() if t]
                    if (len(tokens) >= 2 and
                        fullname.lower() not in NAME_BLACKLIST and
                        not all(tok.lower() in NAME_BLACKLIST for tok in tokens) and
                        not _looks_like_company_name(fullname)):
                        results.append({
                            'fullname': fullname,
                            'headline': title,
                            'confidence': 0.8 if 'director' in title or 'direkt' in title else 0.7
                        })
                        break
            if not any(r['headline'] == title for r in results):
                results.append({'fullname': None, 'headline': title, 'confidence': 0.4})

    def prio_index(title: str) -> int:
        try:
            return PRIORITY_TITLES.index(title)
        except ValueError:
            return 999
    results.sort(key=lambda r: prio_index(r['headline']))
    return results

def _extract_contacts_from_jsonld(soup: BeautifulSoupType, domain: Optional[str]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    try:
        for s in soup.find_all("script", attrs={"type": "application/ld+json"}):
            raw = s.get_text(strip=True)
            if not raw:
                continue
            try:
                data = json.loads(raw)
            except Exception:
                continue
            items = data if isinstance(data, list) else [data]
            for it in items:
                t = (it.get("@type") or it.get("type") or "").lower()
                if t == "person":
                    nm = it.get("name")
                    jt = it.get("jobTitle") or it.get("jobtitle") or None
                    em = it.get("email") or it.get("mail") or None
                    tel = it.get("telephone") or it.get("phone") or None
                    emails = [em] if isinstance(em, str) else []
                    phones = [tel] if isinstance(tel, str) else []
                    out.append({
                        "fullname": nm if _is_valid_person_name(nm) else None,
                        "headline": jt or "kontakt",
                        "emails": _unique_preserve(emails),
                        "phones": _unique_preserve(phones),
                        "score": 7,  # strukturerede data → høj basis-score
                        "source": "jsonld",
                    })
                if t == "organization":
                    cps = it.get("contactPoint") or it.get("contactpoint")
                    if isinstance(cps, list):
                        for cp in cps:
                            em = cp.get("email") if isinstance(cp, dict) else None
                            tel = cp.get("telephone") if isinstance(cp, dict) else None
                            emails = [em] if isinstance(em, str) else []
                            phones = [tel] if isinstance(tel, str) else []
                            if emails or phones:
                                out.append({
                                    "fullname": None,
                                    "headline": cp.get("contactType") if isinstance(cp, dict) else "kontakt",
                                    "emails": _unique_preserve(emails),
                                    "phones": _unique_preserve(phones),
                                    "score": 5,
                                    "source": "jsonld",
                                })
    except Exception:
        pass
    return out

def _extract_from_footer(soup: BeautifulSoupType, domain: Optional[str]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    try:
        footer = soup.find("footer")
        if not footer:
            return out
        # emails og phones i footer
        emails: List[str] = []
        phones: List[str] = []
        for a in footer.find_all("a", href=True):
            href = (a.get("href") or "").strip().lower()
            if href.startswith("mailto:"):
                addr = href.split(":", 1)[1].split("?", 1)[0]
                if EMAIL_REGEX.fullmatch(addr):
                    emails.append(addr)
            if href.startswith("tel:"):
                phones.append(href.split(":", 1)[1])
        # supplement via tekst
        txt = footer.get_text(" ", strip=True)
        emails += _extract_emails(txt, domain=None)  # behold off-domain, prioriteres senere
        phones += _extract_phones(txt)
        emails = _unique_preserve(emails)
        phones = _normalize_many_phones(phones)
        if emails or phones:
            out.append({
                "fullname": None,
                "headline": "kontakt",
                "emails": emails,
                "phones": phones,
                "score": 4,
                "source": "website",
            })
    except Exception:
        pass
    return out


def _extract_contacts_from_dom(soup: BeautifulSoupType, domain: Optional[str]) -> List[Dict[str, Any]]:
    """Find 'kort'/blokke med navn, titel, email og telefon i samme DOM-område, med støj-fjernelse og footer/jsonld-prioritet."""
    contacts: List[Dict[str, Any]] = []
    if not soup:
        return contacts

    # 0) Strukturerede data (schema.org / JSON-LD) – høj prioritet
    contacts.extend(_extract_contacts_from_jsonld(soup, domain))

    # 1) Footer ALTID – ofte ligger kontaktinfo her
    contacts.extend(_extract_from_footer(soup, domain))

    # 2) Fjern åbenlys støj (nav/header/aside og navigation/cookie-klasser)
    try:
        for bad in soup.find_all(['nav', 'header', 'aside']):
            bad.decompose()
        for el in soup.find_all(True, attrs={"class": True, "id": True}):
            cls = " ".join(el.get("class") or []).lower()
            idv = (el.get("id") or "").lower()
            if any(k in cls for k in ["menu", "navbar", "navigation", "cookie", "gdpr", "breadcrumbs", "banner", "hero"]) \
               or any(k in idv  for k in ["menu", "navbar", "navigation", "cookie", "gdpr", "breadcrumbs", "banner", "hero"]):
                try:
                    el.decompose()
                except Exception:
                    pass
    except Exception:
        pass

    # 3) Kontakt-blokke kun hvis "gating" er opfyldt
    block_tags = ["section", "article", "div", "li", "p"]
    for el in soup.find_all(block_tags):
        text = el.get_text(" ", strip=True)
        if not text or len(text) < 15:
            continue

        # gate: skal ligne en kontaktsektion
        idv  = (el.get("id") or "").lower()
        cls  = " ".join(el.get("class") or []).lower()
        has_mailto = any((a.get("href") or "").strip().lower().startswith("mailto:") for a in el.find_all("a", href=True))
        has_tel    = any((a.get("href") or "").strip().lower().startswith("tel:")    for a in el.find_all("a", href=True))
        looks_contact = any(k in idv for k in ["kontakt", "contact", "team", "medarbejd", "ledelse", "about"]) or \
                        any(k in cls for k in ["kontakt", "contact", "team", "medarbejd", "ledelse", "about"])
        if not (has_mailto or has_tel or looks_contact):
            continue

        lower = text.lower()
        titles_here = [t for t in PRIORITY_TITLES if t in lower]

        # navne i blok – valideret
        raw_names = [m.group(0).strip() for m in re.finditer(NAME_PATTERN, text)]
        names_here = [n for n in raw_names if _is_valid_person_name(n)]

        # emails/phones i blok
        emails_here: List[str] = []
        phones_here: List[str] = []
        for a in el.find_all("a", href=True):
            href = (a.get("href") or "").strip().lower()
            if href.startswith("mailto:"):
                addr = href.split(":", 1)[1].split("?", 1)[0]
                if EMAIL_REGEX.fullmatch(addr):
                    emails_here.append(addr.lower())
            if href.startswith("tel:"):
                phones_here.append(href.split(":", 1)[1])

        if not emails_here:
            emails_here += _extract_emails(text, domain=None)  # behold off-domain, prioriteres senere
        if not phones_here:
            phones_here += _extract_phones(text)

        phones_here = _normalize_many_phones(phones_here)
        emails_here = _unique_preserve(emails_here)

        # hvis hverken valid navn eller titel → spring
        if not names_here and not titles_here and not (emails_here or phones_here):
            continue

        # score blokke
        for nm in names_here or [None]:
            title = titles_here[0] if titles_here else None
            score = 0
            if title:
                try:
                    score += max(0, 10 - PRIORITY_TITLES.index(title))
                except ValueError:
                    pass
            if nm and emails_here:
                # navnematch +5 hvis bedste email matcher navnet
                be = _best_email_for_name(emails_here, nm, domain)
                if be:
                    score += 5
            if emails_here:
                score += 1
            if phones_here:
                score += 1
            if nm is None:
                score -= 2  # uden navn er det stadig nyttigt, men nedvægtes

            contacts.append({
                "fullname": nm,
                "headline": title or "kontakt",
                "emails": emails_here,
                "phones": phones_here,
                "score": score,
                "source": "website"
            })

    # sortér på score (desc) + titel-prioritet
    def prio_index(title: str) -> int:
        try:
            return PRIORITY_TITLES.index(title)
        except Exception:
            return 999

    contacts.sort(key=lambda c: (c.get("score", 0), - (999 - prio_index(c.get("headline", "") ))), reverse=True)
    return contacts


# ---------------------------------------------------------------------------
# Email verification
# ---------------------------------------------------------------------------

def _verify_email_address(email: str) -> Dict[str, Any]:
    result = {'ok': False, 'source': 'none', 'status': 'init', 'confidence': 'low'}
    if not email or not isinstance(email, str):
        result['status'] = 'no_email'
        return result
    if not EMAIL_REGEX.fullmatch(email):
        result['status'] = 'regex_fail'
        return result
    result.update({'source': 'regex', 'status': 'regex_ok', 'confidence': 'low'})
    domain = email.split('@')[-1]

    # IDNA (punycode) så MX/A-opslag virker for æ/ø/å-domæner
    try:
        domain_idna = domain.encode('idna').decode('ascii')
    except Exception:
        domain_idna = domain

    mx_host = None
    try:
        if dns and hasattr(dns, 'resolver'):
            answers = dns.resolver.resolve(domain_idna, 'MX')
            mx_records = sorted([(r.preference, str(r.exchange).rstrip('.')) for r in answers])
            if mx_records:
                mx_host = mx_records[0][1]
                result.update({'source': 'mx', 'status': f'mx:{mx_host}', 'confidence': 'medium'})
    except Exception:
        try:
            if dns and hasattr(dns, 'resolver'):
                a_answers = dns.resolver.resolve(domain_idna, 'A')
                if a_answers:
                    mx_host = domain_idna
                    result.update({'source': 'mx_a_fallback', 'status': 'a_fallback', 'confidence': 'medium'})
        except Exception:
            pass

    if USE_SMTP and mx_host:
        try:
            with smtplib.SMTP(mx_host, 25, timeout=10) as server:
                server.ehlo()
                try:
                    server.starttls(); server.ehlo()
                except smtplib.SMTPNotSupportedError:
                    pass
                server.mail('verify@vexto.dk')
                code, _ = server.rcpt(email)
                if code == 250:
                    result.update({'ok': True, 'source': 'smtp', 'status': f'smtp:{code}', 'confidence': 'high'})
                    return result
                else:
                    result.update({'source': 'smtp', 'status': f'smtp:{code}', 'confidence': 'medium'})
        except Exception as e:
            result.update({'source': 'smtp', 'status': f'smtp_error:{e.__class__.__name__}', 'confidence': 'medium'})

    if ZERBOUNCE_API_KEY:
        try:
            url = f"https://api.zerobounce.net/v2/validate?api_key={ZERBOUNCE_API_KEY}&email={email}"
            resp = requests.get(url, timeout=10).json()
            status = resp.get('status')
            if status == 'valid':
                result.update({'ok': True, 'source': 'zerobounce', 'status': 'valid', 'confidence': 'very_high'})
            elif status in {'catch-all', 'unknown'}:
                result.update({'ok': False, 'source': 'zerobounce', 'status': status, 'confidence': 'medium'})
            else:
                result.update({'ok': False, 'source': 'zerobounce', 'status': status, 'confidence': 'low'})
            return result
        except Exception as e:
            result.update({'source': 'zerobounce', 'status': f'error:{e.__class__.__name__}', 'confidence': 'medium'})

    return result

def _generate_email_candidates(full_name: str, domain: str) -> List[str]:
    if not full_name or not domain:
        return []
    first, last = _normalize_name_tokens(full_name)
    initials = first[0] if first else ''
    variants = []
    if first and last:
        variants += [
            f"{first}.{last}@{domain}",
            f"{initials}{last}@{domain}",
            f"{first}{last}@{domain}",
            f"{first}-{last}@{domain}",
            f"{first}{last[:1]}@{domain}",
            f"{first[:1]}{last[:1]}@{domain}",  # nyt: initial+initial (fx tk@)
        ]
    if first:
        variants += [f"{first}@{domain}"]
    variants += [f"info@{domain}", f"kontakt@{domain}", f"sales@{domain}", f"marketing@{domain}"]
    return _unique_preserve(variants)

def _generate_and_verify_email(full_name: str, domain: str, context_emails: Optional[List[str]] = None) -> Tuple[Optional[str], Dict[str, Any]]:
    # Prio: eksisterende mails (samme domæne) – sortér efter navnematch
    ctx = []
    for e in (context_emails or []):
        ed = e.split("@")[-1].lower()
        if ed.endswith(domain.lower()) or _is_brand_similar_domain(ed, domain):
            ctx.append(e)
    if full_name and ctx:
        ctx = sorted(ctx, key=lambda e: _score_email_for_name(e, full_name), reverse=True)
    for e in ctx:
        meta = _verify_email_address(e)
        if meta.get('ok'):
            return e, meta
    # Generer mønstre
    for cand in _generate_email_candidates(full_name, domain):
        meta = _verify_email_address(cand)
        if meta.get('ok'):
            return cand, meta
    # Fald tilbage til bedst matchende kontekstmail, hvis nogen
    if ctx:
        best = ctx[0]
        meta = _verify_email_address(best)
        return best, meta
    return None, {'ok': False, 'source': 'none', 'status': 'no_candidate', 'confidence': 'low'}

# ---------------------------------------------------------------------------
# Playwright on-demand helpers
# ---------------------------------------------------------------------------

def _import_http_client_module():
    global _HTTP_CLIENT_MOD, _HTTP_CLIENT_MOD_LOGGED
    if _HTTP_CLIENT_MOD is not None:
        return _HTTP_CLIENT_MOD
    candidates = [
        "vexto.scoring.http_client",
        "src.vexto.scoring.http_client",
        "http_client",
        "src_vexto_scoring_http_client",
    ]
    for name in candidates:
        try:
            _HTTP_CLIENT_MOD = __import__(name, fromlist=["*"])
            if not _HTTP_CLIENT_MOD_LOGGED:
                log.info(f"[Playwright-fallback] http_client import OK: {name}")
                _HTTP_CLIENT_MOD_LOGGED = True
            return _HTTP_CLIENT_MOD
        except Exception:
            continue
    if not _HTTP_CLIENT_MOD_LOGGED:
        log.info("[Playwright-fallback] http_client not found; skipping render")
        _HTTP_CLIENT_MOD_LOGGED = True
    return None

def _coerce_html(ret: Any) -> str:
    if isinstance(ret, str):
        return ret
    if isinstance(ret, dict):
        for k in ("html", "content", "text", "body", "data"):
            v = ret.get(k)
            if isinstance(v, str) and len(v) > 0:
                return v
    if isinstance(ret, (list, tuple)) and ret:
        if isinstance(ret[0], str):
            return ret[0]
    return ""

def _try_render_with_http_client(url: str, timeout: int = 20, retry_count: int = 2, min_len: int = 500) -> str:
    # cache kun hvis vi har en "rigtig" side
    if url in _CACHE_RENDERED_URL:
        cached = _CACHE_RENDERED_URL[url]
        if isinstance(cached, str) and len(cached) >= min_len:
            return cached

    mod = _import_http_client_module()
    if not mod:
        return ""

    func_names = [
        "get_rendered_html","render_html","fetch_rendered_html",
        "get_html","fetch_html","fetch_page","fetch",
    ]
    for attempt in range(retry_count):
        for fname in func_names:
            fn = getattr(mod, fname, None)
            if not callable(fn):
                continue
            try:
                try:
                    raw = fn(url, timeout=timeout)
                except TypeError:
                    raw = fn(url)
                html = _coerce_html(raw)
                if isinstance(html, str) and len(html) >= min_len:
                    _CACHE_RENDERED_URL[url] = html
                    return html
            except Exception:
                continue
        if attempt < retry_count - 1:
            time.sleep(1.2)  # lille backoff
    return ""

def _playwright_fetch_contact_htmls(website_url: Optional[str]) -> List[str]:
    if not USE_PLAYWRIGHT_CONTACT_FALLBACK or not website_url:
        return []
    domain = _get_registered_domain(website_url)
    if not domain:
        return []

    htmls: List[str] = []
    seen_urls: Set[str] = set()

    def _maybe_add(url: str):
        url = url.rstrip("/")
        if url in seen_urls:
            return
        seen_urls.add(url)
        html = _try_render_with_http_client(url, timeout=PLAYWRIGHT_CONTACT_TIMEOUT)
        if not html or len(html) < 500:
            # HTTP fallback hvis render giver for lidt
            html = _fetch_url_text(url)
        if isinstance(html, str) and len(html) >= 500:
            htmls.append(html)

    # prioriterede stier
    paths = ["", "/kontakt", "/contact", "/kontakt-os", "/om-os", "/about", "/medarbejdere", "/team", "/ledelse", "/management"]
    bases = [f"https://{domain}", f"http://{domain}", f"https://www.{domain}", f"http://www.{domain}"]

    # 1) hent prioriterede stier
    for base in bases:
        for p in paths:
            _maybe_add(base + p)

    # 2) link discovery fra første OK html
    if htmls:
        try:
            soup = _BeautifulSoup(htmls[0], "html.parser") if BS4_AVAILABLE else None  # type: ignore
            if soup:
                for a in soup.find_all("a", href=True):
                    href = a.get("href") or ""
                    txt = (a.get_text(" ", strip=True) or "").lower()
                    if any(k in (href.lower()) for k in ["kontakt", "contact", "om", "about", "team", "medarbejd", "ledelse"]) or \
                       any(k in txt for k in ["kontakt", "contact", "om", "about", "team", "medarbejd", "ledelse"]):
                        # kun interne links
                        if href.startswith("http"):
                            target = href
                        elif href.startswith("/"):
                            target = bases[0] + href
                        else:
                            target = bases[0] + "/" + href
                        _maybe_add(target)
                        if len(htmls) >= 6:
                            break
        except Exception:
            pass

    return htmls

def _get_contact_section_text_from_html(html: str) -> str:
    if _CF_MOD and hasattr(_CF_MOD, "_collect_contact_section_text") and BS4_AVAILABLE:
        try:
            soup = _BeautifulSoup(html, "html.parser")  # type: ignore[call-arg]
            return _CF_MOD._collect_contact_section_text(soup) or ""
        except Exception:
            pass
    if not BS4_AVAILABLE:
        return ""
    try:
        soup = _BeautifulSoup(html, "html.parser")  # type: ignore[call-arg]
        chunks: List[str] = []
        for tag in soup.find_all(["address"]):
            chunks.append(tag.get_text(separator=" ", strip=True))
        for el in soup.find_all(True, attrs={"id": True}):
            if any(k in (el.get("id") or "").lower() for k in ["kontakt", "contact", "support"]):
                chunks.append(el.get_text(separator=" ", strip=True))
        for el in soup.find_all(True, attrs={"class": True}):
            cls = " ".join(el.get("class") or []).lower()
            if any(k in cls for k in ["kontakt", "contact", "support"]):
                chunks.append(el.get_text(separator=" ", strip=True))
        return "\n".join([c for c in chunks if c])
    except Exception:
        return ""

def _extract_emails_phones_from_html(html: str, domain: Optional[str]) -> Tuple[List[str], List[str]]:
    emails: Set[str] = set()
    phones: List[str] = []

    if not html:
        return [], []

    if BS4_AVAILABLE:
        try:
            soup = _BeautifulSoup(html, "html.parser")  # type: ignore[call-arg]
            # 1) DOM-first
            emails_dom = _extract_emails_from_dom(soup, domain=None)
            phones_dom = _extract_phones_from_dom(soup)
            emails.update(emails_dom)
            phones.extend(phones_dom)

            # 2) Kontaktsektion (suppler)
            sec_text = _get_contact_section_text_from_html(html)
            if sec_text:
                emails.update(_extract_emails(sec_text, domain))
                if not phones:
                    phones.extend(_extract_phones(sec_text))
        except Exception:
            # Fallback til ren tekst
            emails.update(_extract_emails(html, domain))
            if not phones:
                phones.extend(_extract_phones(html))
    else:
        emails.update(_extract_emails(html, domain))
        phones.extend(_extract_phones(html))

    # Filtrér telefoner for mistænkelige mønstre og begræns antal
    phones_list = _filter_suspect_phones(_unique_preserve(phones))[:MAX_PHONES_PER_SOURCE]
    emails_list = sorted(emails)
    return emails_list, phones_list

# ---------------------------------------------------------------------------
# Førstepartsdomæne-inferens (Google) – strammet
# ---------------------------------------------------------------------------

def _domain_has_company_mention(domain: str, company_name: str) -> bool:
    for scheme in ("https://", "http://"):
        html = _fetch_url_text(f"{scheme}{domain}")
        if html and company_name and company_name.lower() in html.lower():
            return True
    return False

def _score_candidate_domain(company_name: str, domain: str, freq: int) -> int:
    score = 0
    cname_norm = re.sub(r"[^a-z0-9 ]+", " ", (company_name or "").lower())
    tokens = [t for t in cname_norm.split() if t and t not in ("aps","a/s","as","iv s","i/s","k/s","aps.", "aps,")]
    host = domain.lower()
    # frekvens fra søgeresultater
    score += min(freq, 3)
    # brand i domæne?
    if any(t in host for t in tokens if len(t) >= 4):
        score += 3
    # punycode / SEO-ord straffes
    if host.startswith("xn--"):
        score -= 4
    if any(tok in host for tok in SEO_JUNK_TOKENS):
        score -= 3
    # nævner siden virksomhedsnavnet?
    try:
        if _domain_has_company_mention(domain, company_name or ""):
            score += 4
    except Exception:
        pass
    return score

def _infer_firstparty_domain(company_name: Optional[str]) -> Tuple[Optional[str], int]:
    """Returnerer (domain, score). Score < 4 betyder 'svag/ikke robust'."""
    if not company_name or not _get_google_service():
        return None, 0
    if company_name in _CACHE_COMPANY_INFERRED_DOMAIN:
        return _CACHE_COMPANY_INFERRED_DOMAIN[company_name]

    queries = [
        f'"{company_name}"',
        f'"{company_name}" kontakt',
        f'"{company_name}" "om os"'
    ]
    candidates: List[str] = []
    for q in queries:
        items = _google_search(q, num=5)
        log.info(f"[InferDomain] Google søgning: {q} → {len(items)} resultater")
        for it in items:
            url = it.get("link")
            dom = _get_registered_domain(url)
            if not dom or _is_thirdparty_domain(dom):
                continue
            candidates.append(dom)

    if not candidates:
        _CACHE_COMPANY_INFERRED_DOMAIN[company_name] = (None, 0)
        return None, 0

    # frekvens
    freq: Dict[str, int] = {}
    for d in candidates:
        freq[d] = freq.get(d, 0) + 1

    # scorér og vælg bedst over en tærskel
    scored = [(d, _score_candidate_domain(company_name, d, freq[d])) for d in set(candidates)]
    scored.sort(key=lambda x: x[1], reverse=True)
    best_dom, best_score = scored[0]
    if best_score < 4:
        log.info(f"[InferDomain] Ingen robust kandidat (best={best_dom}, score={best_score})")
        _CACHE_COMPANY_INFERRED_DOMAIN[company_name] = (None, best_score)
        return None, best_score

    log.info(f"[InferDomain] Valgt førstepartsdomæne: {best_dom}")
    _CACHE_COMPANY_INFERRED_DOMAIN[company_name] = (best_dom, best_score)
    return best_dom, best_score

# ---------------------------------------------------------------------------
# Trin-implementering
# ---------------------------------------------------------------------------

def _step1_website(scraped_text: Optional[str], domain: Optional[str], fallback_htmls: Optional[List[str]] = None
                   ) -> Tuple[Optional[Dict[str, Any]], List[str], List[str]]:
    emails_found: List[str] = []
    phones_found: List[str] = []

    # Først: hvis vi har fallback_htmls, brug DOM-first udtræk + blok-parring
    if fallback_htmls:
        accumulated_emails: List[str] = []
        accumulated_phones: List[str] = []
        best_contact: Optional[Dict[str, Any]] = None
        best_score = -1

        for html in fallback_htmls:
            if not html:
                continue
            if BS4_AVAILABLE:
                try:
                    soup = _BeautifulSoup(html, "html.parser")  # type: ignore[call-arg]
                    contacts = _extract_contacts_from_dom(soup, domain)
                    for c in contacts:
                        sc = int(c.get("score", 0))
                        if sc > best_score:
                            best_contact = c
                            best_score = sc
                except Exception:
                    pass

            # Saml generelle emails/phones fra hele HTML'en
            e_html, p_html = _extract_emails_phones_from_html(html, domain)
            accumulated_emails.extend(e_html)
            accumulated_phones.extend(p_html)

        # Hvis vi fandt en “kort”-kontakt, brug kontaktens egne email/telefon først
        if best_contact:
            log.info(f"[Trin 1] (playwright) Navn/Titel fundet: "
                     f"{{'fullname': {best_contact.get('fullname')}, 'headline': {best_contact.get('headline')}, 'confidence': 0.8}}")
            contact_emails = best_contact.get("emails", [])
            contact_phones = best_contact.get("phones", [])
            emails_out = _unique_preserve(contact_emails + accumulated_emails)
            phones_out = _unique_preserve(contact_phones + accumulated_phones)
            _log_compact_list("[Trin 1] (playwright) Emails", emails_out)
            _log_compact_list("[Trin 1] (playwright) Phones", phones_out)
            best_contact['source'] = 'website'
            return best_contact, emails_out, phones_out

        # Ellers: ingen navn, men måske emails/phones
        accumulated_emails = _unique_preserve(accumulated_emails)
        accumulated_phones = _unique_preserve(accumulated_phones)
        if accumulated_emails or accumulated_phones:
            log.info(f"[Trin 1] (playwright) Ingen navn, men emails/phones fundet")
            _log_compact_list("[Trin 1] (playwright) Emails", accumulated_emails)
            _log_compact_list("[Trin 1] (playwright) Phones", accumulated_phones)
        else:
            log.info("[Trin 1] Ingen website-tekst tilgængelig")
        return None, accumulated_emails, accumulated_phones

    # Dernæst: scraped_text fallback (ren tekst)
    if scraped_text and isinstance(scraped_text, str) and scraped_text.strip():
        pairs = _extract_name_title_pairs(scraped_text)
        emails_found = _extract_emails(scraped_text, domain)
        phones_found = _extract_phones(scraped_text)
        if pairs:
            best = pairs[0]
            log.info(f"[Trin 1] (scraped_text) Navn/Titel fundet: {best}")
            _log_compact_list("[Trin 1] (scraped_text) Emails", _unique_preserve(emails_found))
            _log_compact_list("[Trin 1] (scraped_text) Phones", _unique_preserve(phones_found))
            best.update({'source': 'website'})
            return best, _unique_preserve(emails_found), _unique_preserve(phones_found)
        if emails_found or phones_found:
            log.info(f"[Trin 1] (scraped_text) Ingen navn, men emails/phones fundet")
            _log_compact_list("[Trin 1] (scraped_text) Emails", _unique_preserve(emails_found))
            _log_compact_list("[Trin 1] (scraped_text) Phones", _unique_preserve(phones_found))
        else:
            log.info("[Trin 1] (scraped_text) Ingen relevante fund")
        return None, _unique_preserve(emails_found), _unique_preserve(phones_found)

    log.info("[Trin 1] Ingen website-tekst tilgængelig")
    return None, [], []

def _step2_google(company_name: Optional[str], domain: Optional[str]) -> Tuple[Optional[Dict[str, Any]], List[str], List[str]]:
    if not (company_name or domain):
        return None, [], []
    if not _get_google_service():
        log.info("[Trin 2] Google ikke tilgængelig – skipper")
        return None, [], []

    emails_found: List[str] = []
    phones_found: List[str] = []

    queries = []
    if domain:
        queries.extend([
            f'site:{domain} (kontakt OR team OR ledelse OR management OR "about us")',
            f'site:{domain} (email OR e-mail OR kontakt) {company_name or ""}'.strip(),
        ])
    if company_name:
        queries.append(f'"{company_name}" (marketingchef OR "head of marketing" OR direktør)')

    for q in queries:
        items = _google_search(q, num=3)
        log.info(f"[Trin 2] Google søgning: {q} → {len(items)} resultater")
        for it in items:
            url = it.get('link')
            if not url:
                continue
            html = _fetch_url_text(url)
            if not html:
                continue
            page_dom = _get_registered_domain(url)
            same_dom = (domain is not None and page_dom == domain)
            if same_dom:
                e_html, p_html = _extract_emails_phones_from_html(html, domain)
                emails_found += e_html
                phones_found += p_html
                # forsøg at finde navn/titel i samme dokument (tekst-baseret)
                pairs = _extract_name_title_pairs(html)
                if pairs:
                    best = pairs[0]
                    best.update({'source': 'google'})
                    log.info(f"[Trin 2] Fundet via Google på {url}: {best}")
                    _log_compact_list("[Trin 2] Emails via Google-sider", _unique_preserve(emails_found))
                    _log_compact_list("[Trin 2] Phones via Google-sider", _unique_preserve(phones_found))
                    return best, _unique_preserve(emails_found), _unique_preserve(phones_found)

    if emails_found or phones_found:
        log.info(f"[Trin 2] Ingen navne, men emails/phones fundet via Google")
        _log_compact_list("[Trin 2] Emails via Google-sider", _unique_preserve(emails_found))
        _log_compact_list("[Trin 2] Phones via Google-sider", _unique_preserve(phones_found))
    else:
        log.info("[Trin 2] Ingen relevante navne/emails/phones via Google")
    return None, _unique_preserve(emails_found), _unique_preserve(phones_found)

def _step3_linkedin(company_name: Optional[str], employee_count: int, domain: Optional[str]) -> Optional[Dict[str, Any]]:
    if not USE_LINKEDIN:
        log.info("[Trin 3] Skip LinkedIn (USE_LINKEDIN=False)")
        return None

    # Store virksomheder → Apify/company-side
    if employee_count >= 20 and company_name:
        linkedin_url = _find_company_linkedin_url(company_name)
        if linkedin_url:
            employees = _query_linkedin_api(linkedin_url)
            if employees:
                emp = _filter_employees_by_title(employees)
                if emp:
                    emp.update({'source': 'linkedin'})
                    return emp

    # Små virksomheder → Google person-fallback
    person = _google_linkedin_person_lookup(company_name, domain)
    if person:
        return person

    log.info(f"[Trin 3] Ingen LinkedIn-match")
    return None

# ---------------------------------------------------------------------------
# Orchestrator pr. række
# ---------------------------------------------------------------------------

def _find_best_contact(row: pd.Series) -> pd.Series:
    company_name = row.get('Vrvirksomhed_virksomhedMetadata_nyesteNavn_navn') or row.get('company_name')
    website_url = row.get('Hjemmeside') or row.get('website')
    website_live = _to_bool(row.get('HjemmesideLiveBool'))
    director_name_raw = row.get('direktørnavn') or row.get('director_name')
    director_name = _choose_director_name(director_name_raw)
    scraped_text = row.get('scraped_contact_text') or row.get('scraped_text')
    employee_count = _get_employee_count(row)

    # Domænevalg + confidence
    domain_csv = _get_registered_domain(website_url)
    domain_to_use: Optional[str] = domain_csv if website_live and domain_csv else None
    domain_source = "csv" if domain_to_use else "inferred"
    domain_score = 10 if domain_to_use else 0

    log.info(f"=== Start kontakt-berigelse: {company_name} | domain={domain_to_use} | emp={employee_count} ===")

    if not domain_to_use and ALLOW_BROAD_FALLBACK:
        inferred, score = _infer_firstparty_domain(company_name)
        if inferred and not _is_thirdparty_domain(inferred):
            domain_to_use = inferred
            domain_score = score
            log.info(f"[Fallback] Bruger infereret domæne: {domain_to_use} (score={domain_score})")

    # Brug cache KUN hvis domæne er sikkert (csv) eller robust inferens (score>=6)
    cache_allowed = bool(domain_to_use) and (domain_source == "csv" or domain_score >= 6)
    if domain_to_use and cache_allowed and domain_to_use in _CACHE_DOMAIN_RESULT:
        cached_contact, cached_emails, cached_phones = _CACHE_DOMAIN_RESULT[domain_to_use]
        if cached_contact or cached_emails or cached_phones:
            log.info(f"[Cache] Genbruger resultat for domæne: {domain_to_use}")
            contact = cached_contact
            context_emails = list(cached_emails)
            context_phones = list(cached_phones)
            # Hvis kontakt uden navn men vi HAR direktør → brug direktørnavn
            if (not contact or not contact.get('fullname')) and director_name:
                if contact is None:
                    contact = {'fullname': director_name, 'headline': 'Direktør', 'source': 'cvr', 'confidence': 0.6}
                else:
                    contact['fullname'] = director_name
                    contact['source'] = f"{contact.get('source') or 'cache'}+director_fallback"
            return _finalize_row(company_name, contact, context_emails, context_phones, domain_to_use, director_name, employee_count)

    contact: Optional[Dict[str, Any]] = None
    context_emails: List[str] = []
    context_phones: List[str] = []

    fallback_htmls: List[str] = []
    if domain_to_use:
        site_for_render = f"https://{domain_to_use}"
        if (not scraped_text or not str(scraped_text).strip()):
            fallback_htmls = _playwright_fetch_contact_htmls(site_for_render)

        c1, emails1, phones1 = _step1_website(scraped_text, domain_to_use, fallback_htmls)
        if c1:
            contact = c1
            context_emails = emails1
            context_phones = phones1
        else:
            context_emails.extend(emails1)
            context_phones.extend(phones1)

    # Trin 2
    if not contact:
        c2, emails2, phones2 = _step2_google(company_name, domain_to_use)
        if c2:
            contact = c2
            context_emails = _unique_preserve(context_emails + emails2)
            context_phones = _unique_preserve(context_phones + phones2)
        else:
            context_emails = _unique_preserve(context_emails + emails2)
            context_phones = _unique_preserve(context_phones + phones2)

    # Trin 3 – (typisk off i jeres miljø, men lader fallback blive)
    if not contact:
        c3 = _step3_linkedin(company_name, employee_count, domain_to_use)
        if c3:
            contact = c3

    # VIGTIGT: Hvis vi har en kontakt uden navn, men en kendt direktør → brug direktørnavnet
    if contact and not contact.get('fullname') and director_name:
        contact['fullname'] = director_name
        contact['source'] = f"{contact.get('source') or 'unknown'}+director_fallback"
        contact['confidence'] = max(0.6, contact.get('confidence') or 0.0)
        log.info(f"[Navne-fallback] Brug direktørnavn: {director_name}")

    # Trin 4 – CVR/direktør fallback hvis stadig ingen kontakt
    if not contact and pd.notna(director_name):
        contact = {'fullname': director_name, 'headline': 'Direktør', 'source': 'cvr', 'confidence': 0.6}
        log.info(f"[Trin 4] Fallback til direktør: {director_name}")

    # Cache kun når domænet er sikkert/robust
    if domain_to_use and cache_allowed:
        _CACHE_DOMAIN_RESULT[domain_to_use] = (contact, list(context_emails), list(context_phones))

    return _finalize_row(company_name, contact, context_emails, context_phones, domain_to_use, director_name, employee_count)

def _finalize_row(company_name: Optional[str],
                  contact: Optional[Dict[str, Any]],
                  context_emails: List[str],
                  context_phones: List[str],
                  domain_to_use: Optional[str],
                  director_name: Optional[str],
                  employee_count: int) -> pd.Series:
    contact_name = contact.get('fullname') if contact else None
    contact_title = contact.get('headline') if contact else 'Ingen kontaktperson fundet'
    contact_source = contact.get('source') if contact else 'none'

    # Rens emails (behold både first-party og off-domain; vi prioriterer senere)
    context_emails = _unique_preserve([e for e in context_emails if EMAIL_REGEX.fullmatch(e or "")])

    # Vælg email — prioriter *kontaktens egne blok-emails* (fra DOM-kort) før alt andet
    contact_email = None
    email_meta: Dict[str, Any] = {'ok': False, 'source': 'none', 'status': 'no_contact', 'confidence': 'low'}
    if domain_to_use:
        # --- Navn-korrektion: hvis vi har et åbenlyst ikke-personnavn og direktør findes, så brug direktøren ---
        if contact and contact_name and not _is_valid_person_name(contact_name) and director_name:
            contact_name = director_name
            contact_title = 'Direktør'
            contact_source = f"{(contact.get('source') if contact else 'unknown')}+director_fallback"
            log.info(f"[Navne-korrektion] Erstatter tvivlsomt navn med direktør: {director_name}")

        # --- First-party prioritet + blok-først udvælgelse ---
        contact_email = None
        email_meta: Dict[str, Any] = {'ok': False, 'source': 'none', 'status': 'no_contact', 'confidence': 'low'}

        # evt. emails/phones der fulgte med kontakt-blokken (fra website)
        contact_block_emails: List[str] = []
        contact_block_phones: List[str] = []
        if isinstance(contact, dict):
            contact_block_emails = _unique_preserve(contact.get('emails', []) or [])
            contact_block_phones = _unique_preserve(contact.get('phones', []) or [])

        # prioriter first-party fra kontaktens egen blok
        firstparty_block = []
        offdomain_block  = []
        for e in contact_block_emails:
            dom_e = (e.split("@")[-1] or "").lower()
            if domain_to_use and (dom_e.endswith(domain_to_use.lower()) or _is_brand_similar_domain(dom_e, domain_to_use)):
                firstparty_block.append(e)
            else:
                offdomain_block.append(e)

        firstparty_ctx = []
        offdomain_ctx  = []
        for e in context_emails:
            dom_e = (e.split("@")[-1] or "").lower()
            if domain_to_use and (dom_e.endswith(domain_to_use.lower()) or _is_brand_similar_domain(dom_e, domain_to_use)):
                firstparty_ctx.append(e)
            else:
                offdomain_ctx.append(e)

        def _pick_best(cands: List[str], name: Optional[str]) -> Optional[str]:
            if not cands:
                return None
            if name:
                # hvis vi kender navnet, brug navne-match og firma-kontekst i scoren
                ranked = sorted(cands, key=lambda em: _score_email_for_name(em, name, company_name), reverse=True)
                return ranked[0]
            # uden navn: vælg ikke-generisk først
            non_generic = [e for e in cands if _email_username(e) not in GENERIC_EMAIL_USERS]
            return (non_generic or cands)[0]

        # 1) blok-first, first-party
        chosen = _pick_best(firstparty_block, contact_name)
        chosen_source_suffix = "+dom_block" if chosen else ""

        # 2) fald tilbage: kontekst, first-party
        if not chosen:
            chosen = _pick_best(firstparty_ctx, contact_name)
            chosen_source_suffix = "" if not chosen else ""

        # 3) off-domain mikro-politik (kun hvis virkelig nødvendigt)
        if not chosen:
            is_micro = (employee_count >= 0 and employee_count <= 3)
            if is_micro and contact_name:
                # prøv først blok-offdomain, derefter kontekst-offdomain
                off_cands = offdomain_block or offdomain_ctx
                if off_cands:
                    ranked = sorted(off_cands, key=lambda em: _score_email_for_name(em, contact_name, company_name), reverse=True)
                    # kræv en rimelig score for at acceptere off-domain
                    best_off = ranked[0]
                    if _score_email_for_name(best_off, contact_name, company_name) >= 3:
                        chosen = best_off
                        chosen_source_suffix = "+offdomain_firstparty"

        # 4) hvis stadig intet og vi har domæne → generer mønstre
        if not chosen and domain_to_use and contact_name:
            gen, meta = _generate_and_verify_email(contact_name, domain_to_use, firstparty_ctx or None)
            if gen:
                chosen = gen

        # 5) verificér valgt email (hvis nogen)
        if chosen:
            meta = _verify_email_address(chosen)
            if meta:
                contact_email = chosen
                # suffix til kilde hvis vi ved den kom fra DOM-blok eller off-domain policy
                if isinstance(meta.get('source'), str):
                    meta['source'] = f"{meta['source']}{chosen_source_suffix}"
                else:
                    meta['source'] = f"regex{chosen_source_suffix}"
                email_meta = meta

        # telefon: blok-telefoner først, ellers kontekst
        phones_all = _unique_preserve(list(contact_block_phones) + list(context_phones))
        phones_all = _filter_suspect_phones(phones_all)
        contact_phone = phones_all[0] if phones_all else None


    # Rens telefoner for mistænkelige mønstre; hvis kontakt havde egne phones, er de allerede først
    context_phones = _filter_suspect_phones(_unique_preserve(context_phones))
    contact_phone = context_phones[0] if context_phones else None

    log.info(f"[Resultat] source={contact_source} | name={contact_name} | title={contact_title} | "
             f"email={contact_email} | phone={contact_phone} | email_meta={email_meta}")

    return pd.Series({
        'contact_name': contact_name,
        'contact_title': contact_title,
        'contact_source': contact_source,
        'contact_email': contact_email,
        'contact_phone': contact_phone,
        'email_validation_source': email_meta.get('source'),
        'email_validation_status': email_meta.get('status'),
        'email_confidence': email_meta.get('confidence'),
        'firstparty_domain_used': domain_to_use
    })


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_enrichment_on_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Beriger DF med kontaktdata.

    Input-kolonner (valgfri):
      - Vrvirksomhed_virksomhedMetadata_nyesteNavn_navn | company_name
      - Hjemmeside | website
      - HjemmesideLiveBool
      - direktørnavn | director_name
      - scraped_contact_text | scraped_text
      - antal_ansatte | employee_count | employees | ansatte

    Output-kolonner:
      - contact_name, contact_title, contact_source
      - contact_email, contact_phone
      - email_validation_source, email_validation_status, email_confidence
      - firstparty_domain_used
    """
    if df is None or df.empty:
        log.warning("Input DataFrame er tom – returnerer uændret")
        return df

    emp_cols = [c for c in df.columns if re.search(r'ansat|employee', str(c), re.I)]
    log.info(f"Employee-related columns detected: {emp_cols}")

    log.info(f"Starter berigelse på {len(df)} virksomheder")
    enriched = df.apply(_find_best_contact, axis=1)
    out = pd.concat([df.reset_index(drop=True), enriched], axis=1)
    log.info("Berigelse færdig")
    return out
