"""
run_url_finding_standalone.py
-----------------------------
Et standalone script til at køre og teste den nye, robuste `url_finder`.
Scriptet læser en liste af virksomheder fra cvr_data.csv, finder deres
URL'er ved hjælp af vandfalds-strategien og gemmer resultatet i en ny fil.

Denne version er opdateret til at håndtere livscyklussen for den nye,
genbrugelige url_finder (v3.0).
"""
import pandas as pd
import asyncio
from pathlib import Path
import sys
import logging
from tqdm.asyncio import tqdm_asyncio
from typing import Any

# --- Opsætning af stier ---
PROJECT_DIR = Path(__file__).resolve().parent
# Sørger for at Python kan finde 'vexto'-pakken inde i 'src'
sys.path.append(str(PROJECT_DIR)) 

# ÆNDRING #1: Importerer nu hele modulet for at kunne kalde både find og close.
from src.vexto.scoring import url_finder

# --- Input/Output filer ---
INPUT_CSV = PROJECT_DIR / "output" / "cvr_data.csv"
OUTPUT_CSV = PROJECT_DIR / "output" / "cvr_data_med_urls_DIAGNOSTIC.csv"

# --- Logging ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)-7s] %(message)s",
    datefmt="%H:%M:%S",
)

def safe_str(value: Any) -> str:
    """Konverterer en værdi sikkert til en streng og håndterer NaN/None."""
    if pd.isna(value) or value is None:
        return ""
    return str(value).strip()

async def process_row_safely(row: pd.Series):
    """
    En sikker "wrapper", der kalder den primære logik.
    Den fanger exceptions for en enkelt række, så hele processen ikke stopper.
    """
    company_name = safe_str(row.get('Vrvirksomhed_virksomhedMetadata_nyesteNavn_navn'))
    if not company_name:
        return None, None, None, None # Returner tuple for at matche succes-casen

    try:
        # Hent de nødvendige data fra rækken med sikker konvertering.
        city = safe_str(row.get('Vrvirksomhed_virksomhedMetadata_nyesteBeliggenhedsadresse_postdistrikt'))
        cvr_url = safe_str(row.get('Hjemmeside'))
        email = safe_str(row.get('Email'))

        # ÆNDRING #2: Kaldet er nu præfikset med modulnavnet
        result_tuple = await url_finder.find_company_url(
            name=company_name,
            city=city,
            cvr_url=cvr_url,
            email=email
        )
        return result_tuple if result_tuple else (None, None, "Ingen URL fundet", None)
    except Exception as e:
        logging.error(f"Uventet fejl under behandling af '{company_name}': {e}", exc_info=False)
        return None, None, f"FEJL: {e}", None # Returner tuple ved fejl

async def main():
    """Hovedfunktionen der læser, processerer og gemmer data."""
    logging.info(f"Læser virksomheder fra: {INPUT_CSV}")
    try:
        df = pd.read_csv(INPUT_CSV, dtype=str).fillna('')
    except FileNotFoundError:
        logging.error(f"FEJL: Inputfilen '{INPUT_CSV}' blev ikke fundet. Kør main.py først for at generere den.")
        return

    tasks = [process_row_safely(row) for _, row in df.iterrows()]
    logging.info(f"Finder URL'er for {len(df)} virksomheder (dette kan tage tid)...")
    
    results = await tqdm_asyncio.gather(*tasks)

    df['fundet_url'] = [res[0] for res in results]
    df['fundet_via_metode'] = [res[1] for res in results]
    df['validerings_status'] = [res[2] for res in results]
    df['fundet_titel'] = [res[3] for res in results]

    df.to_csv(OUTPUT_CSV, index=False, encoding='utf-8-sig')
    
    logging.info(f"\nFærdig! Diagnostik-resultater er gemt i: {OUTPUT_CSV}")
    
    found_count = df['fundet_url'].notna().sum()
    total_count = len(df)
    if total_count > 0:
        logging.info(f"Statistik: Fandt {found_count} ud af {total_count} URL'er ({found_count/total_count:.1%}).")
        method_counts = df[df['fundet_via_metode'].notna()]['fundet_via_metode'].value_counts().to_dict()
        logging.info(f"Fordeling pr. metode: {method_counts}")
    else:
        logging.info("Statistik: Ingen virksomheder at behandle.")


if __name__ == "__main__":
    # ÆNDRING #3: Korrekt livscyklus-håndtering med try...finally
    # Dette sikrer, at ressourcer (som netværks-klienten) altid lukkes korrekt.
    try:
        # Kør hovedlogikken
        asyncio.run(main())
    except KeyboardInterrupt:
        # Håndter hvis brugeren afbryder med Ctrl+C
        logging.info("Processen blev afbrudt af brugeren.")
    finally:
        # Denne blok kører ALTID, uanset om 'main' lykkes eller fejler.
        logging.info("Lukker ressourcer ned...")
        # Kalder den nye oprydnings-funktion i url_finder.
        asyncio.run(url_finder.close_client())
    
    logging.info("Program afsluttet rent.")