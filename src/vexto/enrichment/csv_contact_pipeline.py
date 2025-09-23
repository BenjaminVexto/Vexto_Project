# src/vexto/enrichment/csv_contact_pipeline.py
from __future__ import annotations

import os, re, json, math
from typing import Optional, Tuple, List, Dict, Any
from urllib.parse import urlsplit
from dataclasses import dataclass
import pandas as pd
import logging
log = logging.getLogger("csv_contact_pipeline")  # >>> ANKER: CSV/LOGGER

# >>> ANKER: PIPELINE/ENV_CONFIG
@dataclass
class PipelineConfig:
    lille_antal: int = int(os.getenv("LILLE_VIRKSOMHED_ANTAL_ANSATTE", "20"))
    medtag_uden_url: bool = os.getenv("MEDTAG_EMNER_UDEN_URL", "true").lower() in {"1","true","yes","y"}
    parallel_workers: int = int(os.getenv("PIPELINE_PARALLEL_WORKERS", "8"))  # NY: Default til 8 for bedre performance
    cache_size: int = int(os.getenv("PIPELINE_CF_CACHE_SIZE", "500"))
    cache_ttl: int = int(os.getenv("PIPELINE_CF_CACHE_TTL", "3600"))
    use_google: bool = os.getenv("USE_GOOGLE", "true").lower() in {"1","true","yes","y"}
    google_cse_id: str = os.getenv("GOOGLE_CSE_ID") or ""
    google_api_key: str = os.getenv("GOOGLE_API_KEY") or ""

config = PipelineConfig()
# <<< ANKER: PIPELINE/ENV_CONFIG

try:
    import httpx
except Exception:
    httpx = None  # Step C/D kræver httpx

from .contact_finder import ContactFinder  # bruger din eksisterende finder

# ----------------------- ENV & utils -----------------------

def _env_bool(name: str, default: bool=False) -> bool:
    v = os.getenv(name, str(default))
    return str(v).strip().lower() in {"1","true","yes","y"}

LILLE_ANTAL = int(os.getenv("LILLE_VIRKSOMHED_ANTAL_ANSATTE", "20"))
MEDTAG_UDEN_URL = _env_bool("MEDTAG_EMNER_UDEN_URL", True)

# Google CSE
GOOGLE_CSE_ID = os.getenv("GOOGLE_CSE_ID")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
USE_GOOGLE = _env_bool(
    "USE_GOOGLE",
    bool(GOOGLE_CSE_ID and GOOGLE_API_KEY and httpx is not None),
)

# LinkedIn via Apify
USE_LINKEDIN = _env_bool("USE_LINKEDIN", False)
APIFY_TOKEN = os.getenv("APIFY_TOKEN")
APIFY_ACTOR = os.getenv("APIFY_ACTOR_LINKEDIN", "apify/actor-linkedin-scraper")
LINKEDIN_MIN = int(os.getenv("CF_MIN_EMPLOYEES_LINKEDIN", "50"))

ROLE_HINTS = [
    "marketing", "salgs", "kommerc", "ejer", "owner", "partner", "director",
    "head of", "vp", "chief", "leder", "ansvarlig"
]

# >>> ANKER: CSV/NORMALIZE_URL
def _normalize_url(u) -> Optional[str]:
    # Tåler None, NaN, tal/float og andre typer
    if u is None:
        return None
    try:
        import pandas as pd
        if pd.isna(u):
            return None
    except Exception:
        pass

    s = str(u).strip()
    if not s:
        return None
    if s.lower() in {"na", "none", "null", "-", "nan"}:
        return None

    # NY: DNS pre-check for at skippe døde domæner tidligt
    try:
        import dns.resolver  # Kræver dnspython dependency
        domain = urlsplit(s).hostname or s.split('/')[0] if '//' in s else s.split('/')[0]
        dns.resolver.resolve(domain, 'A')  # Prøv A-record (IPv4)
    except (dns.resolver.NXDOMAIN, dns.resolver.NoAnswer, dns.exception.DNSException):
        return None  # Dødt domæne – skip

    if not re.match(r"^https?://", s, re.I):
        s = "https://" + s
    return s
# <<< ANKER: CSV/NORMALIZE_URL

# >>> ANKER: CSV/FIRST_NAME
def _first_name_from_csv(dire_names) -> Optional[str]:
    # Tåler None/NaN/float
    if dire_names is None:
        return None
    try:
        import pandas as pd
        if pd.isna(dire_names):
            return None
    except Exception:
        pass

    s = str(dire_names).strip()
    if not s or s.lower() in {"na", "none", "null", "-", "nan"}:
        return None

    # split på forskellige skilletegn og tag første ikke-tomme
    parts = re.split(r'[;,|/"]+', s)
    for p in parts:
        p = p.strip()
        if p:
            return p
    return None
# <<< ANKER: CSV/FIRST_NAME

def _is_filled(v) -> bool:
    if v is None:
        return False
    try:
        import pandas as pd
        if pd.isna(v):
            return False
    except Exception:
        pass
    s = str(v).strip()
    return bool(s) and s.lower() not in {"na", "none", "null", "-", "nan"}

def _employees_from_row(row) -> Optional[int]:
    try:
        v = row.get("AntalAnsatte")
        if pd.isna(v): return None
        return int(float(v))
    except Exception:
        return None

def _domain_of(u: str | None) -> Optional[str]:
    if not u: return None
    try:
        return (urlsplit(u).hostname or "").lower()
    except Exception:
        return None

def _candidate_satisfactory(c: dict) -> Tuple[bool, float]:
    """Vurder om kandidat er 'tilfredsstillende' og returnér (ok, score_like)."""
    score = float(c.get("score") or 0.0)
    emails = c.get("emails") or []
    title = (c.get("title") or "").lower()
    name  = (c.get("name") or "").strip()
    has_role_hint = any(h in title for h in ROLE_HINTS)
    has_email = any("@" in e for e in emails)
    ok = bool(name) and (has_email or (has_role_hint and score >= 55))
    # score_like favoriserer email + rolle
    score_like = score + (20.0 if has_email else 0.0) + (10.0 if has_role_hint else 0.0)
    return ok, score_like

def _best_from_list(cands: List[dict]) -> Optional[dict]:
    best, best_key = None, -1e9
    for c in (cands or []):
        ok, s = _candidate_satisfactory(c)
        if s > best_key:
            best, best_key = c, s
    return best

# ----------------------- Step implementering -----------------------

def _stepA_small_from_csv(row) -> Optional[dict]:
    """Hvis lille virksomhed og CSV har direktørnavn + Email + Hjemmeside, brug dem."""
    employees = _employees_from_row(row)
    if employees is None or employees > LILLE_ANTAL:
        return None
    dire = _first_name_from_csv(row.get("Direktørnavn"))
    email = row.get("Email")
    site  = _normalize_url(row.get("Hjemmeside"))
    if all(_is_filled(x) for x in (dire, email, site)):
        return {
            "name": dire,
            "title": "Direktør",
            "emails": [str(email).strip()],
            "url": site,
            "person_source_url": site,
            "final_website": site,
            "source": "csv",
            "score": 60.0,
            "reasons": ["CSV_SMALL_COMPANY_DIRECTOR"],
        }
    return None

# >>> ANKER: PIPELINE/STEPB_CACHE
from collections import OrderedDict
import time

class UnifiedCache:
    """
    Robust unified cache med TTL, LRU eviction, og performance metrics.
    Understøtter get/set med auto-eviction og expiration.
    """
    def __init__(self, ttl_seconds: int = 3600, max_size: int = 500):
        self.ttl = ttl_seconds
        self.max_size = max_size
        self.cache = OrderedDict()  # For LRU: move_to_end ved access
        self.timestamps = {}
        self.hits = 0
        self.misses = 0
    
    def get(self, key: str) -> Optional[Any]:
        if key not in self.cache:
            self.misses += 1
            return None
        if time.time() - self.timestamps[key] > self.ttl:
            del self.cache[key]
            del self.timestamps[key]
            self.misses += 1
            return None
        self.cache.move_to_end(key)  # LRU: Mark as recently used
        self.hits += 1
        return self.cache[key]
    
    def set(self, key: str, value: Any):
        if key in self.cache:
            self.cache.move_to_end(key)
        elif len(self.cache) >= self.max_size:
            oldest = self.cache.popitem(last=False)[0]  # Remove least recently used
            del self.timestamps[oldest]
        self.cache[key] = value
        self.timestamps[key] = time.time()
    
    def stats(self) -> dict:
        return {"hits": self.hits, "misses": self.misses, "size": len(self.cache), "hit_rate": self.hits / max(1, self.hits + self.misses)}

# Global instance (tilføj efter klasse-definition)
GLOBAL_CACHE = UnifiedCache(ttl_seconds=3600, max_size=500)

def _stepB_site_crawl(cf: ContactFinder, site_url: Optional[str]) -> Optional[dict]:
    if not site_url:
        return None

    key = str(site_url).strip().lower().rstrip("/")
    cached = GLOBAL_CACHE.get(key)
    if cached is not None:
        return cached

    try:
        best = cf.find(site_url, limit_pages=10)
        if not best:
            GLOBAL_CACHE.set(key, None)
            return None
        ok, _ = _candidate_satisfactory(best)
        result = best if ok else None
        GLOBAL_CACHE.set(key, result)
        return result
    except Exception:
        GLOBAL_CACHE.set(key, None)
        return None
# <<< ANKER: PIPELINE/STEPB_CACHE

def _google_candidates(domain: Optional[str], company_name: Optional[str], limit: int = 5) -> List[str]:
    if not USE_GOOGLE:
        return []
    if httpx is None:
        return []
    q = ""
    if domain:
        q = f'site:{domain} (kontakt OR team OR medarbejder OR "contact us" OR staff OR "marketingchef" OR "salgschef")'
    elif company_name:
        q = f'{company_name} (kontakt OR team OR medarbejder OR "contact us" OR staff)'
    else:
        return []
    try:
        api = "https://www.googleapis.com/customsearch/v1"
        with httpx.Client(timeout=20.0, follow_redirects=True) as cli:
            r = cli.get(api, params={"cx": GOOGLE_CSE_ID, "key": GOOGLE_API_KEY, "q": q, "num": limit})
            items = (r.json() or {}).get("items", []) if r.status_code == 200 else []
            urls = [it.get("link") for it in items if it.get("link")]
            return urls
    except Exception:
        return []

def _stepC_google(cf: ContactFinder, site_url: Optional[str], company_name: Optional[str]) -> Optional[dict]:
    domain = _domain_of(site_url) if site_url else None
    urls = _google_candidates(domain, company_name, limit=5)
    bests: List[dict] = []
    for u in urls:
        try:
            cands = cf.find_all(u, limit_pages=4)
            if not cands: continue
            b = _best_from_list(cands)
            if b: bests.append(b)
        except Exception:
            continue
    return _best_from_list(bests) if bests else None

def _stepD_linkedin(company_name: Optional[str], domain: Optional[str]) -> Optional[dict]:
    """Apify LinkedIn backend. Returnér kandidat med navn/titel/url (emails typisk tom)."""
    if not (USE_LINKEDIN and APIFY_TOKEN and httpx is not None):
        return None
    query = (company_name or domain or "").strip()
    if not query:
        return None
    try:
        api = f"https://api.apify.com/v2/acts/{APIFY_ACTOR}/run-sync?token={APIFY_TOKEN}"
        payload = {"search": query, "maxItems": 10}
        with httpx.Client(timeout=30.0, follow_redirects=True) as cli:
            rr = cli.post(api, json=payload)
            data = rr.json() if rr.status_code == 200 else {}
            items = data.get("items") or data.get("results") or []
        best = None
        best_key = -1e9
        for it in items:
            name = (it.get("name") or it.get("fullName") or "").strip()
            title = (it.get("headline") or it.get("title") or "").strip()
            url   = it.get("url") or it.get("publicUrl")
            if not (name and url): 
                continue
            cand = {
                "name": name,
                "title": title,
                "emails": [],
                "url": url,
                "person_source_url": url,
                "source": "linkedin",
                "score": 50.0 + (10.0 if any(h in title.lower() for h in ROLE_HINTS) else 0.0),
                "reasons": ["LINKEDIN_APIFY"]
            }
            ok, s = _candidate_satisfactory(cand)
            key = (s + (5.0 if ok else 0.0))
            if key > best_key:
                best, best_key = cand, key
        return best
    except Exception:
        return None

# ----------------------- Public pipeline -----------------------

NEW_COLS = [
    "CF_Name","CF_Title","CF_Email","CF_URL","CF_Source","CF_Step","CF_Score","CF_Reasons","KontaktinfoStatus"
]

# TILFØJELSE: Definer helper-funktion for row-logik
def _process_single_row(cf, idx, row, company_name_col):
    result = {}  # Dict til at holde nye kolonneværdier

    employees = _employees_from_row(row)
    company_name = str(row.get(company_name_col) or "").strip() or None
    url_in = _normalize_url(row.get("Hjemmeside"))
    if not url_in:
        result["KontaktinfoStatus"] = "FEJL – DØDT DOMÆNE"
        return result

    # 0) Filter: medtag/frasorter sager uden URL
    if not url_in and not MEDTAG_UDEN_URL:
        result["KontaktinfoStatus"] = "SKIPPET – ingen URL"
        return result

    # Step A
    stepA = _stepA_small_from_csv(row)
    if stepA:
        result["CF_Name"] = stepA["name"]
        result["CF_Title"] = stepA.get("title")
        result["CF_Email"] = (stepA.get("emails") or [None])[0]
        result["CF_URL"] = stepA.get("person_source_url") or stepA.get("url")
        result["CF_Source"] = "csv"
        result["CF_Step"] = "A"
        result["CF_Score"] = stepA.get("score")
        result["CF_Reasons"] = json.dumps(stepA.get("reasons") or [], ensure_ascii=False)
        result["KontaktinfoStatus"] = "OK"
        return result  # næste emne (continue i main)

    # Step B (website crawl) – benytter nu cache via monkey-patch
    stepB = _stepB_site_crawl(cf, url_in)
    if stepB:
        result["CF_Name"] = stepB.get("name")
        result["CF_Title"] = stepB.get("title")
        result["CF_Email"] = ((stepB.get("emails") or [None]) or [None])[0]
        result["CF_URL"] = stepB.get("person_source_url") or stepB.get("url")
        result["CF_Source"] = stepB.get("source") or "website"
        result["CF_Step"] = "B"
        result["CF_Score"] = stepB.get("score")
        result["CF_Reasons"] = json.dumps(stepB.get("reasons") or [], ensure_ascii=False)
        result["KontaktinfoStatus"] = "OK"
        return result

    # Step C (Google)
    stepC = _stepC_google(cf, url_in, company_name) if USE_GOOGLE else None
    if stepC:
        result["CF_Name"] = stepC.get("name")
        result["CF_Title"] = stepC.get("title")
        result["CF_Email"] = ((stepC.get("emails") or [None]) or [None])[0]
        result["CF_URL"] = stepC.get("person_source_url") or stepC.get("url")
        result["CF_Source"] = stepC.get("source") or "google"
        result["CF_Step"] = "C"
        result["CF_Score"] = stepC.get("score")
        result["CF_Reasons"] = json.dumps(stepC.get("reasons") or [], ensure_ascii=False)
        result["KontaktinfoStatus"] = "OK"
        return result

    # Step D (LinkedIn) — kun >50 OG kun hvis A+B+C ikke leverede
    if (employees is not None and employees > LINKEDIN_MIN and USE_LINKEDIN):
        stepD = _stepD_linkedin(company_name, _domain_of(url_in))
    else:
        stepD = None

    if stepD:
        result["CF_Name"] = stepD.get("name")
        result["CF_Title"] = stepD.get("title")
        result["CF_Email"] = None  # LinkedIn har typisk ingen emails
        result["CF_URL"] = stepD.get("person_source_url") or stepD.get("url")
        result["CF_Source"] = "linkedin"
        result["CF_Step"] = "D"
        result["CF_Score"] = stepD.get("score")
        result["CF_Reasons"] = json.dumps(stepD.get("reasons") or [], ensure_ascii=False)
        result["KontaktinfoStatus"] = "OK"
        return result

    # Fallback A (brug CSV felter hvis de trods alt findes)
    dire = _first_name_from_csv(row.get("Direktørnavn"))
    email = row.get("Email")
    site = _normalize_url(row.get("Hjemmeside"))
    if all(_is_filled(x) for x in (dire, email, site)):
        result["CF_Name"] = dire
        result["CF_Title"] = "Direktør"
        result["CF_Email"] = str(email).strip()
        result["CF_URL"] = site
        result["CF_Source"] = "csv-fallback"
        result["CF_Step"] = "FB-A"
        result["CF_Score"] = 55.0
        result["CF_Reasons"] = json.dumps(["CSV_FALLBACK"], ensure_ascii=False)
        result["KontaktinfoStatus"] = "OK"
        return result

    # Fallback B
    result["KontaktinfoStatus"] = "Kontaktinfo ikke OK"
    result["CF_Step"] = "FB-B"
    return result

# >>> ANKER: PIPELINE/enrich_csv_with_contacts REPLACEMENT
def enrich_csv_with_contacts(df: pd.DataFrame, *,
                             company_name_col: str = "Navn",
                             out_cols: List[str] = NEW_COLS,
                             cf: Optional[ContactFinder] = None) -> pd.DataFrame:
    """
    Kører Step A→D + Fallbacks på en DataFrame med kolonner:
      - AntalAnsatte (int), Direktørnavn (str), Email (str), Hjemmeside (str)
      - (valgfrit) Navn (firmanavn) til Google/LinkedIn

    Skriver KUN til nye kolonner (overskriver ikke inputfeltet).
    """
    cf = cf or ContactFinder()

    # === Aktivér LRU+TTL cache for cf.find / cf.find_all ===
    from functools import wraps
    import time

    def ttl_cache(ttl_seconds=3600, maxsize=128):
        def decorator(func):
            cache = {}
            timestamps = {}
            
            @wraps(func)
            def wrapper(*args, **kwargs):
                key = str(args) + str(kwargs)
                now = time.time()
                
                # Check cache
                if key in cache:
                    if now - timestamps[key] < ttl_seconds:
                        return cache[key]
                    else:
                        del cache[key]
                        del timestamps[key]
                
                # Cache miss - call function
                result = func(*args, **kwargs)
                
                # Evict if needed
                if len(cache) >= maxsize:
                    oldest = min(timestamps, key=timestamps.get)
                    del cache[oldest]
                    del timestamps[oldest]
                
                cache[key] = result
                timestamps[key] = now
                return result
            
            wrapper.cache_info = lambda: {"size": len(cache), "maxsize": maxsize}
            return wrapper
        return decorator

    try:
        cfg = PipelineConfig()
        CACHE_SIZE = max(32, int(cfg.cache_size))
        CACHE_TTL  = max(1,  int(cfg.cache_ttl))
    except Exception:
        CACHE_SIZE = int(os.getenv("PIPELINE_CF_CACHE_SIZE", "500"))
        CACHE_TTL  = int(os.getenv("PIPELINE_CF_CACHE_TTL",  "3600"))

    _orig_find = cf.find
    _orig_find_all = cf.find_all

    @ttl_cache(ttl_seconds=CACHE_TTL, maxsize=CACHE_SIZE)
    def _cached_find(url: str, limit_pages: int) -> Optional[dict]:
        return _orig_find(url, limit_pages=limit_pages)

    @ttl_cache(ttl_seconds=CACHE_TTL, maxsize=CACHE_SIZE)
    def _cached_find_all(url: str, limit_pages: int) -> tuple:
        return tuple(_orig_find_all(url, limit_pages=limit_pages))

    def _find(url: str, limit_pages: int = 10) -> Optional[dict]:
        u = (url or "").strip()
        if not u:
            return None
        return _cached_find(u, limit_pages)

    def _find_all(url: str, limit_pages: int = 4) -> list:
        u = (url or "").strip()
        if not u:
            return []
        return list(_cached_find_all(u, limit_pages))

    # Monkey-patch instansen (Pylance-klart; ingen ubrugte locals):
    cf.find = _find
    cf.find_all = _find_all
    # === /cache ===

    # Sørg for at kolonner findes
    out = df.copy()
    for c in out_cols:
        if c not in out.columns:
            out[c] = None

    # NY: Rens input-data tidligt – skip rækker med ugyldig/blank URL
    for idx, row in out.iterrows():
        url_in = _normalize_url(row.get("Hjemmeside"))
        if not url_in:
            out.at[idx, "KontaktinfoStatus"] = "SKIPPET – UGYLDIG URL"
            continue  # Skip videre processing for denne række

    # TILFØJELSE: Brug faktisk parallel workers fra config (etik: Balancér load)
    from concurrent.futures import ThreadPoolExecutor, as_completed

    max_workers = config.parallel_workers
    if max_workers > 1:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {}
            for idx, row in out.iterrows():
                if out.at[idx, "KontaktinfoStatus"] == "SKIPPET – UGYLDIG URL":
                    continue  # Skip allerede filtrerede
                future = executor.submit(_process_single_row, cf, idx, row, company_name_col)
                futures[future] = idx
            
            for future in as_completed(futures):
                idx = futures[future]
                try:
                    result = future.result()
                    if result:
                        for col, val in result.items():
                            out.at[idx, col] = val
                except Exception as e:
                    log.error(f"Row {idx} failed: {e}")
                    out.at[idx, "KontaktinfoStatus"] = f"FEJL: {str(e)[:50]}"
    else:
        # Original seriel kode
        for idx, row in out.iterrows():
            if out.at[idx, "KontaktinfoStatus"] == "SKIPPET – UGYLDIG URL":
                continue  # Skip allerede filtrerede
            result = _process_single_row(cf, idx, row, company_name_col)
            for col, val in result.items():
                out.at[idx, col] = val

    return out