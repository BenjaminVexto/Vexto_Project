# src/vexto/scoring/performance_fetcher.py

from __future__ import annotations

import asyncio
import json
import logging
import os
from pathlib import Path
from typing import Optional, Dict, Any
from urllib.parse import quote, urlparse, urljoin
from bs4 import BeautifulSoup
import httpx
from dotenv import load_dotenv

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

# In-memory per-run cache: key = fuld PSI API-URL (inkl. url, strategy, categories)
_PSI_RUN_CACHE: Dict[str, dict] = {}


async def _psi_call(
    api_url: str,
    client: AsyncHtmlClient,
    timeout_s: int = 60,
    headers: Optional[Dict[str, str]] = None,
) -> Optional[dict]:
    """
    Henter PSI-JSON med øget timeout.
    Per-run in-memory cache (_PSI_RUN_CACHE) kortslutter gentagne PSI-kald i samme kørsel.
    Beholder eksisterende html_cache (dags-cache) for tvær-kørsels genbrug.
    """
    # 1) Per-run cache
    if api_url in _PSI_RUN_CACHE:
        log.info("RUNCACHE HIT: PSI-JSON %s", api_url)
        return _PSI_RUN_CACHE[api_url]

    # 2) Dags-cache (eksisterende)
    cached = html_cache.get(api_url)
    if cached is not None:
        log.info("CACHE HIT: PSI-JSON %s", api_url)
        _PSI_RUN_CACHE[api_url] = cached  # hydrer per-run cache for senere opslag i samme run
        return cached

    # 3) Live-kald
    async with SEMA:
        try:
            r = await client.httpx_get(api_url, timeout=timeout_s, headers=headers)
            if r.status_code == 400:
                return {"_fatal": 400}
            r.raise_for_status()
            data = r.json()
            # Skriv til begge caches
            _PSI_RUN_CACHE[api_url] = data
            html_cache.set(api_url, data, expire=86_400)
            return data
        except (httpx.RequestError, httpx.TimeoutException) as e:
            log.warning("PSI-kald fejl %s: %s", api_url, e.__class__.__name__)
            return None

def _has_field_metrics(psi_json: dict) -> bool:
    """
    Returnerer True hvis PSI indeholder CrUX-feltdata (loadingExperience.metrics eller originLoadingExperience.metrics).
    """
    try:
        le = psi_json.get("loadingExperience") or psi_json.get("originLoadingExperience") or {}
        m = le.get("metrics") or {}
        return bool(m)
    except Exception:
        return False

def _parse(psi: dict) -> Optional[PerformanceMetrics]:
    """
    Robust PSI-parser:
      - Feltdata (CrUX) prioriteres for LCP/INP/CLS; fallback til Lighthouse (lab).
      - JS-vægt hentes fra audits.resource-summary → network-requests → diagnostics.
      - performance_score returneres i 0–100.
    """
    try:
        lh = psi.get("lighthouseResult", {}) or {}
        aud = lh.get("audits", {}) or {}
        cat = lh.get("categories", {}) or {}

        def _num(audit_id: str) -> Optional[float]:
            v = (aud.get(audit_id, {}) or {}).get("numericValue")
            try:
                return float(v) if v is not None else None
            except Exception:
                return None

        # --- CrUX (feltdata) først, fallback til lab ---
        # Brug hele PSI-payloaden til at afgøre om vi har field-data (ikke kun LCP).
        crux = psi.get("loadingExperience") or psi.get("originLoadingExperience") or {}
        metrics = crux.get("metrics") or {}
        perf_source = "field" if bool(metrics) else "lab"

        def _crux_percentile(key: str) -> Optional[float]:
            raw = (metrics.get(key) or {}).get("percentile")
            if raw is None:
                return None
            try:
                return float(raw)
            except Exception:
                try:
                    s = str(raw).strip().replace(",", ".")
                    return float(s)
                except Exception:
                    return None

        # Feltmålinger (ms for LCP/INP, "percentile" CLS gives i *hundrede-dele*)
        lcp_field_ms = _crux_percentile("LARGEST_CONTENTFUL_PAINT_MS")
        # INP kan hedde INTERACTION_TO_NEXT_PAINT eller INTERACTION_TO_NEXT_PAINT_MS
        inp_field_ms = _crux_percentile("INTERACTION_TO_NEXT_PAINT")
        if inp_field_ms is None:
            inp_field_ms = _crux_percentile("INTERACTION_TO_NEXT_PAINT_MS")
        # CLS kan hedde CUMULATIVE_LAYOUT_SHIFT eller CUMULATIVE_LAYOUT_SHIFT_SCORE (x100)
        cls_field_pct = _crux_percentile("CUMULATIVE_LAYOUT_SHIFT")
        if cls_field_pct is None:
            cls_field_pct = _crux_percentile("CUMULATIVE_LAYOUT_SHIFT_SCORE")

        # Lab fallback
        lcp_lab_ms = _num("largest-contentful-paint")
        cls_lab = _num("cumulative-layout-shift")

        # LCP: prioriter felt (CrUX), ellers lab. Bevar rå værdi og lav robust capping (0..10000).
        lcp_eff = lcp_field_ms if lcp_field_ms is not None else lcp_lab_ms
        lcp_raw = lcp_eff if lcp_eff is not None else None
        if lcp_raw is not None:
            try:
                lcp_eff = float(lcp_raw)
                if lcp_eff > 10000:
                    log.warning(f"Høj LCP ({lcp_eff}ms) capped til 10000ms")
                lcp_eff = max(0.0, min(lcp_eff, 10000.0))
            except Exception:
                # kunne ikke caste – behold original
                pass

        # INP: felt først; ellers proxy via lab (interaction-to-next-paint/total-blocking-time)
        inp_ms = None
        if inp_field_ms is not None:
            inp_ms = inp_field_ms
            log.info("Interaktivitet fundet via CrUX (INP).")
        else:
            for key in ("interaction-to-next-paint", "experimental-interaction-to-next-paint", "total-blocking-time"):
                v = _num(key)
                if v is not None:
                    inp_ms = v
                    log.info("Interaktivitet fundet via '%s' (lab)", key)
                    break

        # CLS: felt (percentile) er x100; divider til score i 0–1, fallback til lab
        cls = None
        if cls_field_pct is not None:
            cls = round(float(cls_field_pct) / 100.0, 3)
        elif cls_lab is not None:
            cls = round(float(cls_lab), 3)

        # Performance score (0–100)
        perf_score_raw = (cat.get("performance", {}) or {}).get("score")
        performance_score = int(perf_score_raw * 100) if perf_score_raw is not None else None

        # Viewport-score (0/1 fra audit)
        viewport_score = (aud.get("viewport", {}) or {}).get("score") or 0

        # --- JS-vægt + antal scripts fra PSI AUDITS ---
        js_kb: Optional[int] = None
        js_count: Optional[int] = None
        try:
            # 1) Primær: resource-summary (Script)
            rs = (aud.get("resource-summary", {}) or {}).get("details", {}) or {}
            for it in rs.get("items") or []:
                rt = (it.get("resourceType") or it.get("label") or "").strip().lower()
                if rt in ("script", "javascript", "js"):
                    size_bytes = (
                        it.get("transferSize")
                        or it.get("totalBytes")
                        or it.get("size")
                        or 0
                    )
                    if isinstance(size_bytes, (int, float)) and size_bytes > 0:
                        js_kb = int(round(size_bytes / 1024.0))
                    req_cnt = it.get("requestCount") or it.get("count")
                    if isinstance(req_cnt, (int, float)) and req_cnt > 0:
                        js_count = int(req_cnt)
                    break

            # 2) Fallback: network-requests (sum .js)
            if js_kb is None or js_count is None:
                nr = (aud.get("network-requests", {}) or {}).get("details", {}) or {}
                total_bytes = 0
                cnt = 0
                for req in nr.get("items") or []:
                    rtype = (req.get("resourceType") or "").lower()
                    mime = (req.get("mimeType") or "").lower()
                    url = (req.get("url") or "").lower()
                    if "script" in rtype or "javascript" in mime or url.endswith(".js"):
                        tr = req.get("transferSize") or req.get("resourceSize") or 0
                        try:
                            b = int(tr or 0)
                            if b > 0:
                                total_bytes += b
                            cnt += 1
                        except Exception:
                            cnt += 1
                if js_kb is None and total_bytes > 0:
                    js_kb = int(round(total_bytes / 1024.0))
                if js_count is None and cnt > 0:
                    js_count = cnt

            # 3) Ekstra fallback: diagnostics (scriptBytes/numScripts)
            if js_kb is None or (isinstance(js_kb, int) and js_kb <= 0) or js_count is None:
                diag = (aud.get("diagnostics", {}) or {}).get("details", {}) or {}
                ditems = diag.get("items") or []
                if isinstance(ditems, list) and ditems:
                    first = ditems[0] or {}
                    sb = first.get("scriptBytes")
                    ns = first.get("numScripts")
                    if isinstance(sb, (int, float)) and sb > 0 and (js_kb is None or js_kb == 0):
                        js_kb = int(round(sb / 1024.0))
                    if isinstance(ns, (int, float)) and ns > 0 and js_count is None:
                        js_count = int(ns)

        except Exception as e:
            log.debug("Kunne ikke udlede JS-størrelse fra PSI: %s", e)

        # psi_status (undgå 'not_run' ...) og strategi
        form_factor = ((lh.get("configSettings") or {}).get("formFactor") or "").lower()
        if form_factor == "mobile":
            psi_status = "ok_mobile"
            perf_strategy = "mobile"
        elif form_factor == "desktop":
            psi_status = "ok_desktop"
            perf_strategy = "desktop"
        else:
            psi_status = "ok" if lh else "not_run"
            perf_strategy = "unknown"

        # Debug
        try:
            log.info(
                "LCP kilde=%s; field=%s ms, lab=%s ms",
                "field" if lcp_field_ms is not None else "lab",
                f"{lcp_field_ms:.0f}" if isinstance(lcp_field_ms, (int, float)) else "n/a",
                f"{lcp_lab_ms:.0f}" if isinstance(lcp_lab_ms, (int, float)) else "n/a",
            )
        except Exception:
            pass

        return {
            "lcp_ms": lcp_eff,                                # cappet 0..10000
            "lcp_ms_raw": lcp_raw,                            # rå LCP
            "performance_source": perf_source,                # field|lab (hele payloaden)
            "performance_strategy": perf_strategy,            # mobile|desktop|unknown
            "cls": cls,
            "inp_ms": inp_ms,
            "performance_score": performance_score,
            "viewport_score": viewport_score,
            "total_js_size_kb": js_kb if js_kb is not None else 0,
            "js_file_count": js_count if js_count is not None else 0,
            "psi_status": psi_status,
        }
    except Exception:
        log.error("Fejl ved parsing af PSI-JSON: %s", psi, exc_info=True)
        return None


def get_performance_from_stub() -> PerformanceMetrics:
    with PSI_STUB_PATH.open(encoding="utf-8") as f:
        data = _parse(json.load(f)) or {}
        # NY: sikkerhed for nye felter
        if "lcp_ms_raw" not in data and "lcp_ms" in data:
            data["lcp_ms_raw"] = data.get("lcp_ms")
        data.setdefault("performance_source", "unknown")
        data.setdefault("performance_strategy", "unknown")
        data["psi_status"] = "fallback_stub"
        return data


async def _one_api_call(client: AsyncHtmlClient, url: str, strat: str, timeout_s: int = 60) -> Optional[PerformanceMetrics]:
    key = os.getenv("GOOGLE_API_KEY")
    if not key:
        log.warning("GOOGLE_API_KEY mangler – springer PSI over.")
        return None

    full_api_url = (
        f"{API_ROOT}?url={quote(url, safe='')}&strategy={strat}"
        f"&key={key}&category=PERFORMANCE&category=ACCESSIBILITY"
    )
    headers = {"Referer": PSI_REFERER_URL}

    j = await _psi_call(full_api_url, client, timeout_s, headers=headers)

    if j is None:
        return {"psi_status": f"timeout_{strat}"}
    if j.get("_fatal") == 400:
        return {"psi_status": f"bad_request_{strat}"}

    parsed = _parse(j)
    if parsed:
        parsed["psi_status"] = f"ok_{strat}"
        # NY: fallback hvis _parse ikke kunne aflæse formFactor
        if not parsed.get("performance_strategy"):
            parsed["performance_strategy"] = strat
        return parsed


async def get_performance(client: AsyncHtmlClient, url: str, _dummy: str = MOBILE) -> PerformanceMetrics:
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
                    if longer_timeout_res and (
                        longer_timeout_res["psi_status"].startswith("ok")
                        or longer_timeout_res["psi_status"].startswith("bad_request")
                    ):
                        return longer_timeout_res
                continue

        if failed_bad and candidate_url != candidates[-1]:
            continue
        elif failed_bad:
            return res  # type: ignore[return-value]

    log.warning("Alle PSI-forsøg fejlede for %s – bruger stub.", url)
    return get_performance_from_stub()


async def calculate_js_size(client, soup: BeautifulSoup, page_url: str) -> Dict[str, Any]:
    """
    Tæl antal <script>-filer og estimer samlet JS-transfer-size.
    Strategi:
      1) HEAD -> content-length
      2) Fallback: GET med Range: bytes=0-0 -> parse Content-Range
      3) Ignorér enkeltfejl; fortsæt på næste URL
    """
    try:
        script_tags = soup.find_all("script") if soup else []
        js_urls = []
        for s in script_tags:
            src = (s.get("src") or "").strip()
            if src and not src.lower().startswith(("data:", "javascript:")):
                absu = src if src.startswith("http") else urljoin(page_url, src)
                js_urls.append(absu)

        total_bytes = 0
        for u in js_urls:
            try:
                # 1) HEAD
                r = await client.head(u, follow_redirects=True)
                if r and getattr(r, "status_code", 0) < 400:
                    cl = r.headers.get("content-length") or r.headers.get("Content-Length")
                    if cl and cl.isdigit():
                        total_bytes += int(cl)
                        continue
                # 2) Range fallback
                g = await client.httpx_get(u, headers={"Range": "bytes=0-0"}, follow_redirects=True, timeout=15)
                if g and getattr(g, "status_code", 0) < 400:
                    cr = g.headers.get("content-range") or g.headers.get("Content-Range")
                    if cr and "/" in cr:
                        # fx: "bytes 0-0/12345"
                        maybe_total = cr.split("/")[-1].strip()
                        if maybe_total.isdigit():
                            total_bytes += int(maybe_total)
                            continue
                    cl2 = g.headers.get("content-length")
                    if cl2 and cl2.isdigit() and int(cl2) > 1:
                        total_bytes += int(cl2)
            except Exception:
                # fejl på enkelt asset må ikke vælte totalen
                continue

        return {
            "total_js_size_kb": round(total_bytes / 1024, 1) if total_bytes else 0,
            # Tæl kun eksterne scripts (src), deduplikér på URL
            "js_file_count": len({u for u in js_urls}),
        }
    except Exception:
        return {"total_js_size_kb": 0, "js_file_count": 0}


__all__ = ["get_performance_from_stub", "get_performance", "calculate_js_size"]
