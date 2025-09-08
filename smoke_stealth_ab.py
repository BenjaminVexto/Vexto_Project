# smoke_stealth_ab.py
# Hurtig A/B-smoke: sammenligner Stealth ON vs OFF på én URL uden tunge fetchers.
# Toggle sker via ENV-flagget VEXTO_STEALTH (som http_client læser ved call-tid).
# Vi tracker rigtigt om stealth blev anvendt ved at monkeypatche stealth_sync.

import os
import sys
import asyncio
import json
from pathlib import Path
from typing import Any, Dict

# --- Sørg for at 'src' er på sys.path ---
ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

# Importeres først nu, så sys.path er sat
from vexto.scoring import http_client as hc
from vexto.scoring.analyzer import analyze_single_url


# --- Minimal deep_get (uafhængig af resten) ---
def deep_get(d: Dict[str, Any], dotted: str, default=None):
    cur: Any = d
    for key in dotted.split("."):
        if isinstance(cur, dict):
            cur = cur.get(key)
        else:
            return default
        if cur is None:
            return default
    return cur


# --- Monkeypatch tunge fetchers for speed (ingen netværkstunge kald) ---
def patch_fast_fetchers():
    from vexto.scoring import (
        performance_fetcher, image_fetchers, security_fetchers,
        gmb_fetcher, authority_fetcher, tracking_fetchers, crawler
    )

    async def _fast_performance(*args, **kwargs):
        return {
            "psi_status": "skipped", "performance_score": 0,
            "lcp_ms": None, "cls": None, "inp_ms": None,
            "total_js_size_kb": 0, "js_file_count": 0,
        }
    performance_fetcher.get_performance = _fast_performance  # type: ignore

    async def _fast_js_size(*args, **kwargs):
        return {"total_js_size_kb": 0, "js_file_count": 0}
    performance_fetcher.calculate_js_size = _fast_js_size  # type: ignore

    async def _fast_images(*args, **kwargs):
        return {
            "image_count": 0, "avg_image_size_kb": 0,
            "image_alt_count": 0, "image_alt_pct": 0,
        }
    image_fetchers.fetch_image_stats = _fast_images  # type: ignore

    async def _fast_sec(*args, **kwargs):
        return {
            "hsts_enabled": False, "csp_enabled": False,
            "x_content_type_options_enabled": False, "x_frame_options_enabled": False
        }
    security_fetchers.fetch_security_headers = _fast_sec  # type: ignore

    async def _fast_gmb(*args, **kwargs):
        return {"gmb_review_count": 0, "gmb_average_rating": None, "gmb_profile_complete": None}
    gmb_fetcher.fetch_gmb_data = _fast_gmb  # type: ignore

    async def _fast_auth(*args, **kwargs):
        return {"domain_authority": None, "page_authority": 0, "global_rank": None, "authority_status": "skipped"}
    authority_fetcher.get_authority = _fast_auth  # type: ignore

    async def _fast_tracking(*args, **kwargs):
        return {"has_ga4": None, "has_meta_pixel": None, "has_gtm": None}
    tracking_fetchers.fetch_tracking_data = _fast_tracking  # type: ignore

    async def _fast_crawl(client, url: str, max_pages: int = 1):
        return {
            "total_pages_crawled": 1, "total_links_found": 0,
            "broken_links_count": 0, "broken_links_pct": 0.0,
            "broken_links_list": [], "internal_link_score": 0,
            "visited_urls": {url},
            "page_type_distribution": {"product": 0, "category": 0, "info": 1, "blog": 0, "other": 0},
            "links_checked": 0,
        }
    crawler.crawl_site_for_links = _fast_crawl  # type: ignore


async def run_once(url: str, max_pages: int, stealth_env: str) -> Dict[str, Any]:
    """
    Kør en hurtig A/B-run med VEXTO_STEALTH=stealth_env ('1' eller '0').
    Vi monkeypatcher stealth_sync for at se om den faktisk bliver kaldt.
    """
    # 1) Sæt det rigtige ENV-flag som http_client bruger ved call-tid
    os.environ["VEXTO_STEALTH"] = stealth_env

    # 2) Track om stealth faktisk blev anvendt (kun hvis stealth-pakken er tilgængelig)
    stealth_applied_flag = {"applied": False}
    real_stealth_sync = getattr(hc, "stealth_sync", None)
    if real_stealth_sync and getattr(hc, "STEALTH_AVAILABLE", False):
        def _tracked_stealth_sync(page):
            stealth_applied_flag["applied"] = True
            return real_stealth_sync(page)
        hc.stealth_sync = _tracked_stealth_sync  # type: ignore

    # 3) Patch de tunge fetchers
    patch_fast_fetchers()

    # 4) Init klient og start Playwright-tråd
    client = hc.AsyncHtmlClient()
    if hasattr(client, "startup") and asyncio.iscoroutinefunction(client.startup):
        await client.startup()

    try:
        # 5) Kør analyse (kun ~1 side, pga. vores crawl-stub)
        res = await analyze_single_url(client, url, max_pages=max_pages)
    finally:
        if hasattr(client, "close"):
            maybe_close = client.close()
            if asyncio.iscoroutine(maybe_close):
                await maybe_close

    # 6) Byg kompakt opsummering
    summary = {
        "stealth_env": stealth_env,                                  # hvad vi bad om
        "stealth_gate_now": hc._stealth_env_enabled(),               # hvad gate-funktionen mener NU
        "stealth_available": getattr(hc, "STEALTH_AVAILABLE", None), # om lib er installeret
        "stealth_was_applied": stealth_applied_flag["applied"],      # blev stealth_sync faktisk kaldt?
        "schema_found": bool(deep_get(res, "basic_seo.schema_markup_found")),
        "schema_types": deep_get(res, "basic_seo.schema_types", []) or [],
        "canonical_url": deep_get(res, "basic_seo.canonical_url"),
        "canonical_source": deep_get(res, "basic_seo.canonical_source"),
        "has_gtm": deep_get(res, "conversion.has_gtm"),
        "has_ga4": deep_get(res, "conversion.has_ga4"),
        "form_count_total": sum(deep_get(res, "conversion.form_field_counts", []) or []),
        "broken_links_pct": deep_get(res, "technical_seo.broken_links_pct"),
        "js_files": deep_get(res, "performance.js_file_count"),
        "js_kb": deep_get(res, "performance.total_js_size_kb"),
    }
    return summary


def _fmt_row(tag: str, a: Any, b: Any) -> str:
    return f"{tag:<20} | {str(a):<20} | {str(b):<20}"


def print_comparison(a: Dict[str, Any], b: Dict[str, Any]) -> None:
    print("\n=== Stealth A/B (hurtig smoke) ===")
    print(f"Mode A: env(VEXTO_STEALTH)={a['stealth_env']} | gate_now={a['stealth_gate_now']} | applied={a['stealth_was_applied']}")
    print(f"Mode B: env(VEXTO_STEALTH)={b['stealth_env']} | gate_now={b['stealth_gate_now']} | applied={b['stealth_was_applied']}")
    print("-" * 68)
    print(_fmt_row("schema_found", a["schema_found"], b["schema_found"]))
    print(_fmt_row("schema_types[:3]", a["schema_types"][:3], b["schema_types"][:3]))
    print(_fmt_row("canonical_source", a["canonical_source"], b["canonical_source"]))
    print(_fmt_row("has_gtm", a["has_gtm"], b["has_gtm"]))
    print(_fmt_row("has_ga4", a["has_ga4"], b["has_ga4"]))
    print(_fmt_row("form_count_total", a["form_count_total"], b["form_count_total"]))
    print(_fmt_row("broken_links_pct", a["broken_links_pct"], b["broken_links_pct"]))
    print(_fmt_row("js_files", a["js_files"], b["js_files"]))
    print(_fmt_row("js_kb", a["js_kb"], b["js_kb"]))
    print("-" * 68)
    print("A full JSON dump is printed below for reference.\n")
    print("A:", json.dumps(a, ensure_ascii=False, indent=2))
    print("B:", json.dumps(b, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Hurtig stealth A/B-smoke (1 side, tunge fetchers stubs).")
    p.add_argument("--url", default="https://www.inventarland.dk", help="URL at teste (default: inventarland)")
    p.add_argument("--max-pages", type=int, default=1, help="Antal sider at crawle (1 er hurtigst)")
    args = p.parse_args()

    # Kør A (ON) og B (OFF) i samme proces
    a = asyncio.run(run_once(args.url, args.max_pages, "1"))
    b = asyncio.run(run_once(args.url, args.max_pages, "0"))
    print_comparison(a, b)
