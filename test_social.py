# test_social.py

import asyncio
import logging
import pprint

# Juster stien om nødvendigt
from src.vexto.scoring.http_client import AsyncHtmlClient
from src.vexto.scoring.social_fetchers import find_social_media_links

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

async def main():
    """
    Tester udelukkende find_social_media_links for en enkelt URL
    og gemmer den analyserede HTML.
    """
    test_url = "https://www.dr.dk"
    html_output_file = "dr_dk_fetched_content.html"
    
    logging.info(f"--- Starter social media link test for: {test_url} ---")
    
    async with AsyncHtmlClient() as client:
        # Henter og parser HTML til et BeautifulSoup-objekt
        # Dette vil automatisk bruge Playwright-fallback, hvis nødvendigt
        soup = await client.get_soup(test_url)
        
        if soup:
            # Gemmer den pænt formaterede HTML, som funktionen har arbejdet med
            with open(html_output_file, "w", encoding="utf-8") as f:
                f.write(soup.prettify())
            logging.info(f"HTML-indhold er gemt i filen: {html_output_file}")
            
            # Kører selve funktionen
            result = find_social_media_links(soup)
            
            print("\n--- Resultat ---")
            pprint.pprint(result)
            print("----------------")

            if result.get("social_media_links"):
                logging.info("✅ SUCCESS: Der blev fundet links til sociale medier.")
            else:
                logging.warning("❌ FAILED: Ingen links til sociale medier blev fundet.")
        else:
            logging.error(f"Kunne ikke hente HTML-indhold for {test_url}")


if __name__ == "__main__":
    asyncio.run(main())