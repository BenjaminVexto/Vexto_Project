# fetcher_diagnose_full.py

import logging
import sys
import io  # For TextIOWrapper (beholdt som reference)
import asyncio
import argparse
import locale
from datetime import datetime
from typing import Tuple, Union
from rich.console import Console
from rich.table import Table
from src.vexto.scoring.log_utils import install_log_masking
from src.vexto.scoring.http_client import AsyncHtmlClient
from src.vexto.scoring.analyzer import analyze_single_url
from src.vexto.scoring.scorer import calculate_score
from bs4 import BeautifulSoup

# =========================
# Console/Logging setup
# =========================

if sys.platform == "win32":
    # Windows UTF-8 fix
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    # Set Windows console code page to UTF-8
    import ctypes
    kernel32 = ctypes.windll.kernel32
    kernel32.SetConsoleCP(65001)
    kernel32.SetConsoleOutputCP(65001)
else:
    # Unix/Linux UTF-8 fix
    locale.setlocale(locale.LC_ALL, 'en_US.UTF-8')

# Generer log-filnavn
log_filename = f"vexto_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"

# Basic config (stream = INFO)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Tilføj FileHandler med UTF-8 encoding
file_handler = logging.FileHandler(log_filename, encoding='utf-8')
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
logging.getLogger().addHandler(file_handler)
try:
    file_handler = logging.FileHandler(log_filename, encoding='utf-8', errors='replace')
except Exception as e:
    # Fallback hvis UTF-8 fejler
    file_handler = logging.FileHandler(log_filename, encoding='latin-1')


# Maskér hemmeligheder i ALLE logs (efter handlers er sat op)
install_log_masking()

log = logging.getLogger(__name__)
console = Console()

# =========================
# Hjælpere
# =========================

def _vexto_fmt(value):
    """Vis 'ukendt' for None og human-readable bools."""
    if value is None:
        return "ukendt"
    if isinstance(value, bool):
        return "True" if value else "False"
    return value

def _load_rule_points_by_description() -> dict:
    """
    Læser config/scoring_rules.yml og returnerer et opslag:
      { "<rule description>": <max points>, ... }
    Matcher på 'description' (unik i dine regler).
    """
    from pathlib import Path
    import yaml

    # Prøv standardplacering; fallback til relativ sti ved behov
    candidates = [
        Path("config") / "scoring_rules.yml",
        Path(__file__).resolve().parent / "config" / "scoring_rules.yml",
    ]
    yml_path = next((p for p in candidates if p.exists()), None)
    if not yml_path:
        return {}

    try:
        with open(yml_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
    except Exception:
        return {}

    mapping = {}
    for section in cfg.values():
        if not isinstance(section, dict):
            continue
        rules = section.get("rules") or {}
        for rule_key, rule_def in (rules.items() if isinstance(rules, dict) else []):
            if isinstance(rule_def, dict):
                desc = rule_def.get("description")
                pts = rule_def.get("points")
                if desc and isinstance(pts, (int, float)):
                    mapping[desc] = pts
    return mapping

def _normalize_schema_flag(analysis_data: dict) -> None:
    """Hold schema-flag i sync mellem basic/technical."""
    basic_flag = bool(analysis_data.get('basic_seo', {}).get('schema_markup_found', False))
    analysis_data.setdefault('basic_seo', {})['schema_markup_found'] = basic_flag
    tech = analysis_data.get('technical_seo') or {}
    if 'schema_markup_found' in tech:
        tech['schema_markup_found'] = basic_flag
        analysis_data['technical_seo'] = tech

def log_metric_status(metric: str, value: object, status: str = "ok"):
    log.info(f"Metric {metric}: {value} (status: {status})")

async def _safe_get_soup(client: AsyncHtmlClient, url: str) -> Union[BeautifulSoup, None]:
    """
    Hent BeautifulSoup robust, uanset om get_raw_html returnerer str, tuple eller dict.
    """
    res = await client.get_raw_html(url, return_soup=True)
    if res is None:
        return None

    # (soup, canonical_data)
    if isinstance(res, tuple) and len(res) >= 1 and isinstance(res[0], BeautifulSoup):
        return res[0]

    # dict fra Playwright {"html": str, "canonical_data": {...}}
    if isinstance(res, dict) and res.get("html"):
        try:
            return BeautifulSoup(res["html"], "lxml")
        except Exception:
            return None

    # ren HTML-streng
    if isinstance(res, str):
        try:
            return BeautifulSoup(res, "lxml")
        except Exception:
            return None

    # Allerede en soup?
    if isinstance(res, BeautifulSoup):
        return res

    return None

async def crawl_site(client: AsyncHtmlClient, start_url: str, max_pages: int = 50) -> dict:
    """
    Simpel crawler (samme domæne). Filtrerer åbenlyse assets, robust soup-håndtering.
    Bruges kun til lokal test/diagnose.
    """
    from urllib.parse import urljoin, urlparse

    parsed_start = urlparse(start_url)
    base_domain = parsed_start.netloc

    urls_to_visit = {start_url}
    visited_urls = set()

    exclude_ext = ('.pdf', '.jpg', '.jpeg', '.png', '.gif', '.svg', '.webp', '.ico',
                   '.xml', '.css', '.js', '.woff', '.woff2', '.ttf', '.eot', '.mp4', '.webm')

    while urls_to_visit and len(visited_urls) < max_pages:
        current_url = urls_to_visit.pop()
        if current_url in visited_urls or any(current_url.lower().split('?', 1)[0].endswith(ext) for ext in exclude_ext):
            continue

        visited_urls.add(current_url)
        log.info(f"Crawling: {current_url} ({len(visited_urls)}/{max_pages})")

        soup = await _safe_get_soup(client, current_url)
        if not soup:
            continue

        for a in soup.find_all('a', href=True):
            href = a.get('href') or ''
            if href.startswith(('#', 'mailto:', 'tel:', 'javascript:', 'data:', 'callto:')):
                continue
            absolute = urljoin(current_url, href)
            if urlparse(absolute).netloc == base_domain and not any(absolute.lower().split('?', 1)[0].endswith(ext) for ext in exclude_ext):
                urls_to_visit.add(absolute)

    return {'total_pages_crawled': len(visited_urls), 'visited_urls': visited_urls}

# =========================
# Hovedkørsel
# =========================

async def run_diagnostic(url: str, do_score: bool = False, max_pages: int = 50):
    log.debug(f"Running diagnostic with max_pages={max_pages}")

    async with AsyncHtmlClient() as client:
        log.info(f"Starter diagnose for: {url} med max_pages={max_pages}...")

        # Kør den egentlige analyse (bruger din analyzer + crawler)
        analysis_data = await analyze_single_url(client, url, max_pages=max_pages)

        if not isinstance(analysis_data, dict):
            log.error("analyze_single_url returned None/invalid — using empty shell")
            analysis_data = {
                "url": url,
                "fetch_method": "unknown",
                "basic_seo": {},
                "technical_seo": {},
                "performance": {},
                "authority": {},
                "security": {},
                "content": {},
                "social_and_reputation": {},
                "conversion": {},
                "privacy": {},
                "benchmark_complete": False,
                "ux_ui_score": 0,
                "niche_score": 0,
            }

        # Sørg for nøgler før setdefault-brug senere
        analysis_data.setdefault("technical_seo", {})
        analysis_data.setdefault("content", {})

        # Hent sitemap URL via klientens domæne-cache (1 pr. domæne)
        sitemap_url = await client.get_sitemap_url(url)
        if not sitemap_url:
            from urllib.parse import urljoin
            sitemap_url = urljoin(url, "/sitemap/sitemap.xml")  # Fallback til typisk sti
            log.debug(f"No sitemap in robots.txt; using fallback: {sitemap_url}")

        # Tjek friskhed
        sitemap_is_fresh = await client.check_sitemap_freshness(sitemap_url)
        analysis_data.setdefault('technical_seo', {})['sitemap_is_fresh'] = sitemap_is_fresh
        log_metric_status("sitemap_is_fresh", sitemap_is_fresh, "ok" if sitemap_is_fresh else "missing")

        # Analytics detection (GA4 / Meta Pixel)
        analysis_data['conversion'].update(await client.check_analytics(url))
        log_metric_status("has_ga4", analysis_data['conversion'].get('has_ga4'), "ok" if analysis_data['conversion'].get('has_ga4') else "missing")
        log_metric_status("has_meta_pixel", analysis_data['conversion'].get('has_meta_pixel'), "ok" if analysis_data['conversion'].get('has_meta_pixel') else "missing")

    # Ensret schema-flag på tværs af sektioner (viser konsekvent status)
    _normalize_schema_flag(analysis_data)

    # Udskriv resultater
    console.rule(f"[bold green]Diagnose Resultater: {url}[/bold green]")

    for section_key, section_data in analysis_data.items():
        if not isinstance(section_data, dict) or not section_data:
            continue
        table = Table(title=f"[bold cyan]{section_key.upper()}[/bold cyan]")
        table.add_column("Målepunkt", style="white", no_wrap=True)
        table.add_column("Værdi", style="magenta")

        for key, value in section_data.items():
            table.add_row(str(key), str(_vexto_fmt(value)))

        console.print(table)

    if 'fetch_error' in analysis_data:
        console.print(f"[red]Fetch-fejl: {analysis_data['fetch_error']}[/red]")

    # Valgfri samlet score
    if do_score:
        console.rule("[bold blue]Samlet Vexto Score[/bold blue]")
        score_result = calculate_score(analysis_data)

        console.print(f"[bold]Samlet Score:[/] [yellow]{score_result['total_score']}[/] / "
                    f"{score_result['max_possible_points']} point ({score_result['score_percentage']}%)")

        # slå max-points op via YAML (description -> max_points)
        desc_to_max = _load_rule_points_by_description()

        # Helper til pct
        def _pct(opnaaet: float, maks: float) -> str:
            if not maks:
                return "0%"
            try:
                return f"{round((opnaaet / maks) * 100)}%"
            except Exception:
                return "0%"

        # ✅ Beståede
        if score_result.get('achieved_rules'):
            achieved_table = Table(title="✅ Beståede Regler")
            achieved_table.add_column("Regel")
            achieved_table.add_column("Opnået")
            achieved_table.add_column("Maks")
            achieved_table.add_column("%")
            for rule in score_result['achieved_rules']:
                desc = rule.get('description', '')
                got = float(rule.get('points', 0) or 0)
                maks = float(desc_to_max.get(desc, got) or 0)
                achieved_table.add_row(
                    desc,
                    str(int(got)) if got.is_integer() else f"{got}",
                    str(int(maks)) if maks.is_integer() else f"{maks}",
                    _pct(got, maks)
                )
            console.print(achieved_table)

        # ❌ Fejlede
        if score_result.get('failed_rules'):
            failed_table = Table(title="❌ Fejlede Regler")
            failed_table.add_column("Regel")
            failed_table.add_column("Opnået")
            failed_table.add_column("Maks")
            failed_table.add_column("%")
            failed_table.add_column("Målt Værdi")
            for rule in score_result['failed_rules']:
                desc = rule.get('description', '')
                maks = float(desc_to_max.get(desc, 0) or 0)
                failed_table.add_row(
                    desc,
                    "0",
                    str(int(maks)) if maks.is_integer() else f"{maks}",
                    _pct(0, maks),
                    str(_vexto_fmt(rule.get('value')))
                )
            console.print(failed_table)

        # ⚠️ Ikke evaluerede (vis kun maks)
        if score_result.get('not_evaluated_rules'):
            not_eval_table = Table(title="⚠️ Ikke Evaluerede Regler (manglende data)")
            not_eval_table.add_column("Regel")
            not_eval_table.add_column("Maks")
            for rule in score_result['not_evaluated_rules']:
                desc = rule.get('description', '')
                maks = desc_to_max.get(desc, 0)
                not_eval_table.add_row(desc, str(maks))
            console.print(not_eval_table)

# =========================
# CLI
# =========================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Diagnose og scoring for en enkelt URL")
    parser.add_argument("--url", required=True, help="URL til hjemmesiden, f.eks. https://www.dr.dk")
    parser.add_argument("--cvr", help="(Valgfrit) CVR-nummer")  # reserveret til senere brug
    parser.add_argument("--company_name", help="(Valgfrit) Virksomhedsnavn")  # reserveret til senere brug
    parser.add_argument("--score", action="store_true", help="Vis den samlede Vexto-score baseret på reglerne")
    parser.add_argument("--max-pages", type=int, default=50, help="Maximum pages to crawl")
    args = parser.parse_args()

    asyncio.run(run_diagnostic(args.url, args.score, args.max_pages))
