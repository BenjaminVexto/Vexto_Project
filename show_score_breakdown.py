# show_score_breakdown.py

import asyncio
import argparse
from rich.console import Console
from rich.table import Table
from rich import box

from src.vexto.scoring.http_client import AsyncHtmlClient
from src.vexto.scoring.analyzer import analyze_single_url
from src.vexto.scoring.scorer import calculate_score

def print_score_breakdown(url: str, score_result: dict):
    console = Console()
    console.rule(f"[bold green]Score Breakdown for {url}[/bold green]")

    # --- Samlet resultattabel ---
    table = Table(title="🧠 Vexto Score – Punktvis Evaluering", box=box.SIMPLE_HEAVY)
    table.add_column("Nr.", style="dim", width=4)
    table.add_column("Punktnavn", style="cyan", no_wrap=True)
    table.add_column("Point", style="green", justify="right")
    table.add_column("Fetcher", style="magenta")
    table.add_column("Status", style="bold")
    table.add_column("Beskrivelse", style="white")
    table.add_column("Råværdi", style="yellow")

    # Samlet liste over alle punkter: beståede, fejlede, ikke evaluerede
    all_rules = []

    for rule in score_result["achieved_rules"]:
        all_rules.append({
            "rule": rule["rule"],
            "points": rule["points"],
            "fetcher": rule.get("fetcher", "-"),
            "status": "[green]✅ Bestået[/green]",
            "description": rule["description"],
            "value": rule.get("value", "-")
        })

    for rule in score_result["failed_rules"]:
        all_rules.append({
            "rule": rule["rule"],
            "points": "0",
            "fetcher": rule.get("fetcher", "-"),
            "status": "[red]❌ Fejlet[/red]",
            "description": rule["description"],
            "value": rule.get("value", "-")
        })

    for rule in score_result["not_evaluated_rules"]:
        all_rules.append({
            "rule": rule["rule"],
            "points": "0",
            "fetcher": rule.get("fetcher", "-"),
            "status": "[yellow]⚠️ Ikke evalueret[/yellow]",
            "description": rule["description"],
            "value": "-"
        })

    # Sortér alfabetisk (eller evt. efter regelnavn)
    all_rules_sorted = sorted(all_rules, key=lambda x: x["rule"])

    for idx, punkt in enumerate(all_rules_sorted, 1):
        table.add_row(
            str(idx),
            punkt["rule"],
            str(punkt["points"]),
            punkt["fetcher"],
            punkt["status"],
            punkt["description"],
            str(punkt["value"])
        )

    console.print(table)

    # --- Samlet oversigt ---
    console.rule("[bold blue]Samlet Score[/bold blue]")
    console.print(f"[bold]Total:[/bold] {score_result['total_score']} / {score_result['max_possible_points']}  ({score_result['score_percentage']}%)")

# --- Async main ---
async def main():
    parser = argparse.ArgumentParser(description="Vis detaljeret Vexto-score med breakdown")
    parser.add_argument("--url", required=True, help="Virksomhedens URL")
    args = parser.parse_args()

    async with AsyncHtmlClient() as client:
        analysis_data = await analyze_single_url(client, args.url)
        score_result = calculate_score(analysis_data)
        print_score_breakdown(args.url, score_result)

if __name__ == "__main__":
    asyncio.run(main())
