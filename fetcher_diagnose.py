# fetcher_diagnose.py
import asyncio
import argparse
from rich.console import Console
from rich.table import Table
from src.vexto.scoring.http_client import AsyncHtmlClient
from src.vexto.scoring.contact_fetchers import find_contact_info
from src.vexto.scoring.trust_signal_fetcher import find_trust_signals
from src.vexto.scoring.authority_fetcher import get_authority
from src.vexto.scoring.performance_fetcher import get_performance, calculate_js_size
from src.vexto.scoring.gmb_fetcher import fetch_gmb_data
from bs4 import BeautifulSoup

console = Console()

async def diagnose_all(url: str, cvr: str = None, company_name: str = None):
    client = AsyncHtmlClient()
    await client.startup()

    table = Table(title=f"Fetcher-diagnose: {url}", show_lines=True)
    table.add_column("Fetcher", style="bold cyan")
    table.add_column("Resultat", style="bold green")

    try:
        html = await client.get_raw_html(url)
        soup = BeautifulSoup(html, "html.parser")
    except Exception as e:
        console.print(f"[red]Fejl ved hentning af HTML: {e}[/red]")
        await client.close()
        return

    try:
        contact = find_contact_info(soup)
        result = f"{len(contact['emails_found'])} emails, {len(contact['phone_numbers_found'])} tlf"
        table.add_row("Contact", result)
    except Exception as e:
        table.add_row("Contact", f"[red]Fejl: {e}[/red]")

    try:
        trust = find_trust_signals(soup)
        result = f"{len(trust['trust_signals_found'])} signaler"
        table.add_row("Trust Signals", result)
    except Exception as e:
        table.add_row("Trust Signals", f"[red]Fejl: {e}[/red]")

    try:
        authority = await get_authority(client, url)
        if authority:
            result = f"DA: {authority['domain_authority']}, PA: {authority['page_authority']}"
        else:
            result = "Ingen data"
        table.add_row("Authority", result)
    except Exception as e:
        table.add_row("Authority", f"[red]Fejl: {e}[/red]")

    try:
        perf = await get_performance(client, url)
        result = f"Score: {perf.get('performance_score')}, LCP: {perf.get('lcp_ms')}ms"
        table.add_row("Performance", result)
    except Exception as e:
        table.add_row("Performance", f"[red]Fejl: {e}[/red]")

    try:
        js = await calculate_js_size(client, soup, url)
        result = f"{js['js_file_count']} filer, {js['total_js_size_kb']} KB"
        table.add_row("JS Size", result)
    except Exception as e:
        table.add_row("JS Size", f"[red]Fejl: {e}[/red]")

    try:
        gmb = await fetch_gmb_data(client, url, cvr=cvr, company_name=company_name)
        if gmb:
            result = f"Rating: {gmb.get('gmb_average_rating')} ({gmb.get('gmb_review_count')} reviews)"
        else:
            result = "Ingen data"
        table.add_row("GMB", result)
    except Exception as e:
        table.add_row("GMB", f"[red]Fejl: {e}[/red]")

    await client.close()
    console.print(table)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Kør fetcher-diagnose for en URL")
    parser.add_argument("--url", required=True, help="URL til website (f.eks. https://example.dk)")
    parser.add_argument("--cvr", required=False, help="CVR-nummer (valgfrit)")
    parser.add_argument("--company_name", required=False, help="Firmanavn (valgfrit)")
    args = parser.parse_args()

    asyncio.run(diagnose_all(args.url, cvr=args.cvr, company_name=args.company_name))
