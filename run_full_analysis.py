# run_full_analysis.py

import asyncio
import logging
import pandas as pd

from src.vexto.scoring.analyzer import analyze_multiple_urls
from src.vexto.reporter import create_excel_report

# Logging er sat op som før
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

async def main():
    """
    Kører en fuld analyse for URL'er indlæst fra urls.txt og genererer en Excel-rapport.
    """
    # --- NYT: Hent URL'er fra urls.txt ---
    logging.info("Henter URL'er fra 'urls.txt'...")
    try:
        with open('urls.txt', 'r', encoding='utf-8') as f:
            # Vi læser hver linje, fjerner unødvendigt whitespace (f.eks. linjeskift)
            # og ignorerer tomme linjer i filen.
            urls_to_analyze = [line.strip() for line in f if line.strip()]

        if not urls_to_analyze:
            logging.error("Filen 'urls.txt' er tom. Tilføj venligst URL'er, der skal analyseres, og kør scriptet igen.")
            return

    except FileNotFoundError:
        logging.error("Filen 'urls.txt' blev ikke fundet i projektmappen.")
        logging.error("Opret venligst filen, indsæt en URL pr. linje, og kør scriptet igen.")
        return
    # --- SLUT PÅ NY KODEBLOK ---
    
    
    logging.info(f"--- Starter fuld analyse for {len(urls_to_analyze)} URL'er fundet i filen ---")
    
    # Kør selve analysen (resten af scriptet er uændret)
    analysis_results = await analyze_multiple_urls(urls_to_analyze)
    
    if not analysis_results:
        logging.error("Kunne ikke hente analyse-data for nogen URL'er. Afbryder.")
        return
        
    logging.info(f"Analyse færdig. Indsamlet data for {len(analysis_results)} URL'er.")
    
    try:
        # 1. Konverter listen af dictionaries til en pandas DataFrame
        master_df = pd.DataFrame(analysis_results)
        logging.info("Resultater er konverteret til en DataFrame.")
        
        
        # 2. Kald rapport-generatoren med vores DataFrame
        create_excel_report(master_df)
        
        logging.info("--- FÆRDIG! Excel-rapporten er blevet genereret i 'output'-mappen. ---")

    except Exception as e:
        logging.error(f"Der opstod en fejl under oprettelse af Excel-rapport: {e}")
        logging.error("Tjek at output fra 'analyze_multiple_urls' indeholder de forventede kolonner.")


if __name__ == "__main__":
    asyncio.run(main())