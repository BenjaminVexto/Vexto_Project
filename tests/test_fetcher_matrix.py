#!/usr/bin/env python3
# tests/test_fetcher_matrix.py
import sys
import asyncio

# Denne linje er NØDVENDIG for dette standalone script på Windows.
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy()) 

import os
import importlib
import inspect
import logging
import pandas as pd
# --- NØDVENDIG TILFØJELSE ---
import shutil
from pathlib import Path
# --- SLUT TILFØJELSE ---


# Konfigurer logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Projektstier og import
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_PATH = os.path.join(PROJECT_ROOT, "src")
if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)
from vexto.scoring.http_client import AsyncHtmlClient


# Moduler til test
MODULES = [
    "vexto.scoring.contact_fetchers", "vexto.scoring.authority_fetcher", "vexto.scoring.performance_fetcher",
    "vexto.scoring.trust_signal_fetcher", "vexto.scoring.social_fetchers", "vexto.scoring.gmb_fetcher",
    "vexto.scoring.link_fetcher", "vexto.scoring.privacy_fetchers", "vexto.scoring.robots_sitemap_fetcher",
    "vexto.scoring.website_fetchers",
]

async def run_fetchers_for_row(client: AsyncHtmlClient, row: pd.Series) -> dict:
    # Denne funktion er uændret
    url = row["fundet_url"]
    company = row["Vrvirksomhed_virksomhedMetadata_nyesteNavn_navn"]
    cvr = row.get("Vrvirksomhed_cvrNummer")
    entry = {"url": url, "cvr": cvr, "company": company}

    for mod_name in MODULES:
        try:
            mod = importlib.import_module(mod_name)
        except ImportError:
            logging.warning(f"Could not import module: {mod_name}")
            continue

        for func_name in dir(mod):
            if not func_name.startswith("fetch_"):
                continue
            func = getattr(mod, func_name)
            if not inspect.isfunction(func):
                continue

            try:
                sig = inspect.signature(func).parameters
                kwargs_to_pass = {}
                if 'client' in sig: kwargs_to_pass['client'] = client
                if 'url' in sig: kwargs_to_pass['url'] = url
                if 'cvr' in sig: kwargs_to_pass['cvr'] = cvr
                if 'company_name' in sig: kwargs_to_pass['company_name'] = company

                maybe_coro = func(**kwargs_to_pass)
                result = await maybe_coro if inspect.iscoroutine(maybe_coro) else maybe_coro

                if isinstance(result, dict):
                    formatted_result = ";".join(f"{k}={v}" for k, v in result.items() if v is not None)
                    entry[func_name] = formatted_result
                else:
                    entry[func_name] = str(result)
            except Exception as e:
                entry[func_name] = f"ERROR: {type(e).__name__}"
                logging.error(f"Error running {func_name} for {url}: {e}", exc_info=False)
    return entry

async def main():
    logging.info("Starter asynkron fetcher-matrix kørsel...")

    # --- NØDVENDIG ÆNDRING: SLET CACHE VED START ---
    cache_dir_to_delete = Path(PROJECT_ROOT) / ".http_diskcache"
    if cache_dir_to_delete.exists():
        logging.warning(f"Sletter eksisterende cache-mappe: {cache_dir_to_delete}")
        try:
            shutil.rmtree(cache_dir_to_delete)
            logging.info("Cache slettet succesfuldt.")
        except OSError as e:
            logging.error(f"Kunne ikke slette cache-mappen. Fejl: {e}. Tjek om en anden proces bruger den.")
            # Valgfrit: stop scriptet hvis cachen ikke kan slettes
            # return
    # --- SLUT PÅ ÆNDRING ---

    CSV_PATH = os.path.join(PROJECT_ROOT, "urls.csv")
    try:
        df_urls = pd.read_csv(CSV_PATH, sep=";", encoding="utf-8-sig", low_memory=False)
        logging.info(f"Succesfuldt indlæst {len(df_urls)} rækker fra {CSV_PATH}")
        if df_urls.empty:
            logging.error("CSV-filen er tom eller kunne ikke parses korrekt. Afslutter.")
            return
    except FileNotFoundError:
        logging.error(f"FATAL: Kunne ikke finde urls.csv på stien: {CSV_PATH}")
        return

    client = AsyncHtmlClient(max_connections=10)
    await client.startup()

    results = []
    try:
        tasks = [run_fetchers_for_row(client, row) for _, row in df_urls.iterrows()]
        results = await asyncio.gather(*tasks, return_exceptions=True)
    finally:
        await client.close()

    successful_results = [res for res in results if isinstance(res, dict)]
    if not successful_results:
        logging.error("Ingen resultater blev indsamlet.")
        return

    OUT_PATH = os.path.join(PROJECT_ROOT, "tests", "output", "fetcher_matrix_result.csv")
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    pd.DataFrame(successful_results).to_csv(OUT_PATH, index=False, sep=';', encoding='utf-8-sig')
    logging.info(f"✅ Fetcher-matrix med {len(successful_results)} resultater gemt som: {OUT_PATH}")

if __name__ == "__main__":
    asyncio.run(main())