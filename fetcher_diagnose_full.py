# fetcher_diagnose_full.py

import logging
import sys
import io  # For TextIOWrapper
import asyncio  
import argparse  
from datetime import datetime
from rich.console import Console
from rich.table import Table

# Fix for Windows console encoding (tving UTF-8)
if sys.platform == "win32":
    # Mulighed 1: Brug reconfigure (Python 3.7+ - din 3.11 er fin)
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')  # 'replace' håndterer ukendte tegn ved at erstatte dem
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    
    # Alternativ (hvis reconfigure ikke virker): Brug TextIOWrapper
    # sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    # sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

# Generer log-filnavn (som tidligere)
log_filename = f"vexto_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"

# Basic config (som du har)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Tilføj FileHandler med UTF-8 encoding
file_handler = logging.FileHandler(log_filename, encoding='utf-8')  # Tilføj encoding her
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
logging.getLogger().addHandler(file_handler)

log = logging.getLogger(__name__)

from src.vexto.scoring.http_client import AsyncHtmlClient
from src.vexto.scoring.analyzer import analyze_single_url
import inspect
from src.vexto.scoring.scorer import calculate_score

console = Console()


async def crawl_site(client: AsyncHtmlClient, start_url: str, max_pages: int = 50) -> dict:
    log.debug(f"Crawling {start_url} with max_pages={max_pages}")
    from urllib.parse import urljoin, urlparse
    parsed_start = urlparse(start_url)
    base_domain = parsed_start.netloc
    urls_to_visit = {start_url}
    visited_urls = set()
    exclude_extensions = {'.pdf', '.jpg', '.jpeg', '.png', '.gif', '.xml'}

    while urls_to_visit and len(visited_urls) < max_pages:
        current_url = urls_to_visit.pop()
        if current_url in visited_urls or any(current_url.lower().endswith(ext) for ext in exclude_extensions):
            continue
        visited_urls.add(current_url)
        log.info(f"Crawling: {current_url} ({len(visited_urls)}/{max_pages})")

        html = await client.get_raw_html(current_url, return_soup=True)
        if not html:
            continue

        for link in html.find_all('a', href=True):
            href = link.get('href')
            if not href or href.startswith(('#', 'mailto:', 'tel:', 'javascript:', 'data:', 'callto:')):
                continue
            absolute_url = urljoin(current_url, href)
            if urlparse(absolute_url).netloc == base_domain and not any(absolute_url.lower().endswith(ext) for ext in exclude_extensions):
                urls_to_visit.add(absolute_url)

    return {'total_pages_crawled': len(visited_urls), 'visited_urls': visited_urls}

def log_metric_status(metric: str, value: any, status: str = "ok"):
    log.info(f"Metric {metric}: {value} (status: {status})")

async def run_diagnostic(url: str, do_score: bool = False, max_pages: int = 50):
    log.debug(f"Running diagnostic with max_pages={max_pages}")
    async with AsyncHtmlClient() as client:
        log.info(f"Starter diagnose for: {url} med max_pages={max_pages}...")
        analysis_data = await analyze_single_url(client, url, max_pages=max_pages)
        from urllib.parse import urljoin
        # Hent robots.txt for at finde sitemap
        robots_url = urljoin(url, "/robots.txt")
        robots_result = await client.get_raw_html(robots_url)
        if isinstance(robots_result, dict):
            robots_content = robots_result.get('html')
        elif isinstance(robots_result, tuple):
            robots_content, _ = robots_result
        else:
            robots_content = robots_result
        sitemap_url = None
        if robots_content:
            for line in robots_content.splitlines():
                if line.lower().startswith("sitemap:"):
                    sitemap_url = line.split(":", 1)[1].strip()
                    log.debug(f"Found sitemap URL in robots.txt: {sitemap_url}")
                    break
        if not sitemap_url:
            sitemap_url = urljoin(url, "/sitemap/sitemap.xml")  # Fallback til korrekt sti
            log.debug(f"No sitemap in robots.txt, using fallback: {sitemap_url}")
        
        sitemap_is_fresh = await client.check_sitemap_freshness(sitemap_url)
        if 'technical_seo' not in analysis_data:
            analysis_data['technical_seo'] = {}
        analysis_data['technical_seo']['sitemap_is_fresh'] = sitemap_is_fresh
        log_metric_status("sitemap_is_fresh", sitemap_is_fresh, "ok" if sitemap_is_fresh else "missing")
        analysis_data['conversion'].update(await client.check_analytics(url))
        log_metric_status("has_ga4", analysis_data['conversion']['has_ga4'], "ok" if analysis_data['conversion']['has_ga4'] else "missing")
        log_metric_status("has_meta_pixel", analysis_data['conversion']['has_meta_pixel'], "ok" if analysis_data['conversion']['has_meta_pixel'] else "missing")

    console.rule(f"[bold green]Diagnose Resultater: {url}[/bold green]")
    
    for section_key, section_data in analysis_data.items():
        if not isinstance(section_data, dict) or not section_data:
            continue
        table = Table(title=f"[bold cyan]{section_key.upper()}[/bold cyan]")
        table.add_column("Målepunkt", style="white", no_wrap=True)
        table.add_column("Værdi", style="magenta")

        for key, value in section_data.items():
            table.add_row(str(key), str(value))
        
        console.print(table)

    if 'fetch_error' in analysis_data:
        console.print(f"[red]Fetch-fejl: {analysis_data['fetch_error']}[/red]")

    if do_score:
        console.rule("[bold blue]Samlet Vexto Score[/bold blue]")
        score_result = calculate_score(analysis_data)
        
        console.print(f"[bold]Samlet Score:[/] [yellow]{score_result['total_score']}[/] / {score_result['max_possible_points']} point ({score_result['score_percentage']}%)")
        
        if score_result['achieved_rules']:
            achieved_table = Table(title="✅ Beståede Regler")
            achieved_table.add_column("Regel")
            achieved_table.add_column("Point")
            for rule in score_result['achieved_rules']:
                achieved_table.add_row(rule['description'], str(rule['points']))
            console.print(achieved_table)

        if score_result['failed_rules']:
            failed_table = Table(title="❌ Fejlede Regler")
            failed_table.add_column("Regel")
            failed_table.add_column("Målt Værdi")
            for rule in score_result['failed_rules']:
                failed_table.add_row(rule['description'], str(rule['value']))
            console.print(failed_table)

        if score_result['not_evaluated_rules']:
            not_eval_table = Table(title="⚠️ Ikke Evaluerede Regler (manglende data)")
            not_eval_table.add_column("Regel")
            not_eval_table.add_column("Beskrivelse")
            for rule in score_result['not_evaluated_rules']:
                not_eval_table.add_row(rule['rule'], rule['description'])
            console.print(not_eval_table)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Diagnose og scoring for en enkelt URL")
    parser.add_argument("--url", required=True, help="URL til hjemmesiden, f.eks. https://www.dr.dk")
    parser.add_argument("--cvr", help="(Valgfrit) CVR-nummer")
    parser.add_argument("--company_name", help="(Valgfrit) Virksomhedsnavn")
    parser.add_argument("--score", action="store_true", help="Vis den samlede Vexto-score baseret på reglerne")
    parser.add_argument("--max-pages", type=int, default=50, help="Maximum pages to crawl")
    args = parser.parse_args()
    asyncio.run(run_diagnostic(args.url, args.score, args.max_pages))