import json, logging, os, asyncio, httpx
from pathlib import Path
from typing import Optional, Dict
from urllib.parse import quote, urlparse, urljoin
from dotenv import load_dotenv
from bs4 import BeautifulSoup

from .http_client import AsyncHtmlClient, html_cache
from .schemas import PerformanceMetrics

log = logging.getLogger(__name__)
load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parents[3]
PSI_STUB_PATH = PROJECT_ROOT / "tests" / "fixtures" / "psi_sample.json"
API_ROOT = "https://www.googleapis.com/pagespeedonline/v5/runPagespeed"

MOBILE, DESKTOP = "mobile", "desktop"
SEMA = asyncio.Semaphore(8)

PSI_REFERER_URL = "https://developers.google.com/speed/pagespeed/insights/"

async def _psi_call(api_url: str, client: AsyncHtmlClient, timeout_s: int = 60, headers: Optional[Dict[str, str]] = None) -> Optional[dict]:
    """Henter PSI-JSON med øget timeout."""
    if (cached := html_cache.get(api_url)) is not None:
        log.info("CACHE HIT: PSI-JSON %s", api_url)
        return cached

    async with SEMA:
        try:
            r = await client.httpx_get(api_url, timeout=timeout_s, headers=headers)
            if r.status_code == 400:
                return {"_fatal": 400}
            r.raise_for_status()
            data = r.json()
            html_cache.set(api_url, data, expire=86_400)
            return data
        except (httpx.RequestError, httpx.TimeoutException) as e:
            log.warning("PSI-kald fejl %s: %s", api_url, e.__class__.__name__)
            return None

def _parse(psi: dict) -> Optional[PerformanceMetrics]:
    try:
        lh = psi.get("lighthouseResult", {})
        aud = lh.get("audits", {})
        cat = lh.get("categories", {})
        
        num = lambda aid: aud.get(aid, {}).get("numericValue")
        lcp = num("largest-contentful-paint")
        cls = num("cumulative-layout-shift")

        # Cap LCP for outliers
        if lcp and lcp > 10000:
            log.warning(f"Høj LCP ({lcp}ms) capped til 10000ms")
            lcp = 10000

        inp = None
        for key in ("interaction-to-next-paint",
                    "experimental-interaction-to-next-paint",
                    "total-blocking-time"):
            inp = num(key)
            if inp is not None:
                log.info("Interaktivitet fundet via '%s'", key)
                break

        perf_score_raw = cat.get("performance", {}).get("score")
        viewport_score = aud.get('viewport', {}).get('score')

        return {
            "lcp_ms": lcp,
            "cls": cls,
            "inp_ms": inp,
            "performance_score": int(perf_score_raw * 100) if perf_score_raw is not None else None,
            "viewport_score": viewport_score,
        }
        
    except Exception:
        log.error("Fejl ved parsing af PSI-JSON: %s", psi, exc_info=True)
        return None

def get_performance_from_stub() -> PerformanceMetrics:
    with PSI_STUB_PATH.open(encoding="utf-8") as f:
        data = _parse(json.load(f)) or {}
    data["psi_status"] = "fallback_stub"
    return data

async def _one_api_call(client: AsyncHtmlClient, url: str, strat: str, timeout_s: int = 60) -> Optional[PerformanceMetrics]:
    key = os.getenv("GOOGLE_API_KEY")
    if not key:
        log.warning("GOOGLE_API_KEY mangler – springer PSI over.")
        return None
    full_api_url = (f"{API_ROOT}?url={quote(url, safe='')}&strategy={strat}"
                    f"&key={key}&category=PERFORMANCE&category=ACCESSIBILITY")

    headers = {"Referer": PSI_REFERER_URL}
    
    j = await _psi_call(full_api_url, client, timeout_s, headers=headers)
    
    if j is None:
        return {"psi_status": f"timeout_{strat}"}
    if j.get("_fatal") == 400:
        return {"psi_status": f"bad_request_{strat}"}

    parsed = _parse(j)
    if parsed:
        parsed["psi_status"] = f"ok_{strat}"
    return parsed

async def get_performance(client: AsyncHtmlClient, url: str,
                          _dummy: str = MOBILE) -> PerformanceMetrics:
    log.info("Henter performance for %s", url)

    candidates = [url]
    parsed_original = urlparse(url)
    host = parsed_original.hostname or ""
    if host.startswith("www."):
        apex = url.replace("://www.", "://", 1)
        if apex not in candidates:
            candidates.append(apex)
    
    for candidate_url in candidates:
        failed_bad = False
        for strategy in (MOBILE, DESKTOP):
            res = await _one_api_call(client, candidate_url, strategy, timeout_s=60)
            
            if res and res["psi_status"].startswith("ok"):
                return res

            if res and res["psi_status"].startswith("bad_request"):
                failed_bad = True
                if strategy == MOBILE:
                    log.warning("PSI bad_request for mobile for %s. Prøver desktop.", candidate_url)
                    continue
                else:
                    break
            
            if res and res["psi_status"].startswith("timeout"):
                if strategy == MOBILE:
                    log.warning("PSI mobile timeout for %s. Prøver desktop med længere timeout.", candidate_url)
                    longer_timeout_res = await _one_api_call(client, candidate_url, DESKTOP, timeout_s=120)
                    if longer_timeout_res and (longer_timeout_res["psi_status"].startswith("ok") or longer_timeout_res["psi_status"].startswith("bad_request")):
                        return longer_timeout_res
                continue

        if failed_bad and candidate_url != candidates[-1]:
            continue
        elif failed_bad:
            return res

    log.warning("Alle PSI-forsøg fejlede for %s – bruger stub.", url)
    return get_performance_from_stub()

async def calculate_js_size(client: AsyncHtmlClient, soup: BeautifulSoup, base_url: str) -> dict:
    if not soup:
        return {'total_js_size_kb': None, 'js_file_count': 0}

    script_tags = soup.find_all('script', src=True)
    
    async def get_size(src: str):
        try:
            absolute_url = urljoin(base_url, src)
            response = await client.head(absolute_url)
            if response and 'content-length' in response.headers:
                return int(response.headers['content-length'])
        except Exception:
            return 0
        return 0

    tasks = [get_size(tag['src']) for tag in script_tags]
    sizes = await asyncio.gather(*tasks)
    
    total_size_bytes = sum(sizes)
    
    return {
        'total_js_size_kb': round(total_size_bytes / 1024) if total_size_bytes > 0 else 0,
        'js_file_count': len(script_tags)
    }

__all__ = ["get_performance_from_stub", "get_performance", "_parse_psi_json", "calculate_js_size"]