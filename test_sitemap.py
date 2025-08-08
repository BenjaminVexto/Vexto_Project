# test_sitemap.py

import asyncio
import logging
import pprint # Til at printe resultatet pænt

# Juster stien så den passer, hvis du kører fra en anden mappe
from src.vexto.scoring.http_client import AsyncHtmlClient
from src.vexto.scoring.technical_fetchers import fetch_technical_seo_data

# Opsæt simpel logging for at se eventuelle fejl
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

async def main():
    """
    Tester udelukkende fetch_technical_seo_data for en enkelt URL.
    """
    # Vælg en URL du vil teste. dr.dk er et godt eksempel med et opdateret sitemap.
    test_url = "https://www.vexto.dk/"
    
    # --- RETTET ---
    logging.info(f"--- Starter sitemap-test for: {test_url} ---")
    
    async with AsyncHtmlClient() as client:
        # Kald kun den funktion vi vil teste
        result = await fetch_technical_seo_data(client, test_url)
    
    print("\n--- Resultat ---")
    pprint.pprint(result)
    print("----------------")

    # Tjek om testen er bestået
    if result.get("sitemap_is_fresh"):
        # --- RETTET ---
        logging.info("✅ SUCCESS: Sitemap er fundet og er opdateret inden for 90 dage.")
    else:
        # --- RETTET ---
        logging.warning("❌ FAILED: Sitemap blev ikke fundet eller er forældet.")


if __name__ == "__main__":
    asyncio.run(main())