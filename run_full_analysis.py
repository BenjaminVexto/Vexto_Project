#run_full_analysis.py

import asyncio
import logging
import pandas as pd

from src.vexto.scoring.analyzer import analyze_multiple_urls
from src.vexto.reporter import create_excel_report

# Logging er sat op som før
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

async def main():
    """
    Kører en fuld analyse for URL'er indlæst fra URLs.csv og genererer en Excel-rapport.
    """
    # --- START PÅ OPDATERET KODEBLOK ---
    logging.info("Henter URL'er fra 'URLs.csv'...")
    try:
        # Læs CSV-filen med pandas
        df = pd.read_csv('URLs.csv', sep=';', encoding='utf-8-sig')
        
        # Tjek om den nødvendige kolonne findes
        if 'fundet_url' not in df.columns:
            logging.error("Kolonnen 'fundet_url' blev ikke fundet i 'URLs.csv'.")
            return
            
        # Udtræk URL'er, fjern eventuelle tomme rækker, og konverter til en liste
        urls_to_analyze = df['fundet_url'].dropna().tolist()

        if not urls_to_analyze:
            logging.error("Filen 'URLs.csv' er tom eller indeholder ingen URL'er i 'fundet_url'-kolonnen.")
            return

    except FileNotFoundError:
        logging.error("Filen 'URLs.csv' blev ikke fundet i projektmappen.")
        return
    except Exception as e:
        logging.error(f"Der opstod en uventet fejl under indlæsning af 'URLs.csv': {e}")
        return
    # --- SLUT PÅ OPDATERET KODEBLOK ---
    
    
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