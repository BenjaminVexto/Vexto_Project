# src/vexto/cli.py

import argparse
import asyncio
import logging
from pathlib import Path
from tqdm.asyncio import tqdm_asyncio # Vi beholder tqdm for nu

from .scoring.analyzer import analyze_multiple_urls
from .reporter import generate_excel_report

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)

# -- KIRURGISK INDGREB: run_analysis er nu integreret i main for at simplificere --
def main():
    """Hovedfunktionen for kommandolinje-interfacet."""
    parser = argparse.ArgumentParser(description="Kør en Vexto-analyse på en liste af URLs.")
    parser.add_argument(
        "-i", "--input-file",
        type=Path,
        required=True,
        help="Sti til en tekstfil med én URL pr. linje."
    )
    parser.add_argument(
        "-o", "--output-file",
        type=Path,
        default="scoring_report.xlsx",
        help="Navn på output Excel-fil. Standard: scoring_report.xlsx"
    )
    # -- KIRURGISK INDGREB: Tilføjet nye CLI-flags --
    parser.add_argument(
        "-w", "--workers",
        type=int,
        default=10,
        help="Antal samtidige analyser, der skal køres. Standard: 10"
    )
    parser.add_argument(
        "--insecure",
        action="store_false",  # Gemmer 'False' hvis flaget er til stede
        dest="verify_ssl",      # Navnet på variablen bliver 'verify_ssl'
        help="Deaktiverer SSL-certifikatvalidering. Brug med forsigtighed."
    )
    # Sæt default for verify_ssl til True, så det er sikkert som standard
    parser.set_defaults(verify_ssl=True)

    args = parser.parse_args()

    try:
        with open(args.input_file, 'r', encoding='utf-8') as f:
            urls = [line.strip() for line in f if line.strip() and line.startswith('http')]
        
        if not urls:
            print(f"Input-filen '{args.input_file}' er tom eller indeholder ingen gyldige URLs.")
            return

    except FileNotFoundError:
        print(f"Fejl: Input-filen '{args.input_file}' blev ikke fundet.")
        return

    print(f"Fandt {len(urls)} URLs. Starter analyse med {args.workers} workers...")
    
    # Kør den asynkrone analyse med de nye parametre
    analysis_results = asyncio.run(
        analyze_multiple_urls(urls, num_workers=args.workers, verify_ssl=args.verify_ssl)
    )
    
    print("\nAnalyse færdig. Genererer rapport...")
    generate_excel_report(analysis_results, str(args.output_file))


if __name__ == '__main__':
    main()