# src/vexto/scoring/contact_fetchers.py
import re
import logging
from bs4 import BeautifulSoup
from .schemas import ConversionMetrics

log = logging.getLogger(__name__)

# Simpel regex til at finde e-mailadresser.
EMAIL_PATTERN = re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b')

# Simpel regex til at finde danske telefonnumre (8 cifre, evt. med +45 og mellemrum).
PHONE_PATTERN = re.compile(r'(\+45\s?)?(\d{2}\s?){3}\d{2}')

def find_contact_info(soup: BeautifulSoup) -> ConversionMetrics:
    """
    Analyserer et BeautifulSoup-objekt for at finde e-mails og telefonnumre.
    """
    if not soup:
        return {'emails_found': [], 'phone_numbers_found': []}

    try:
        # Hent al tekst fra siden for at søge i den
        text = soup.get_text(separator=' ')

        # Brug re.findall til at finde alle unikke matches
        emails = sorted(list(set(EMAIL_PATTERN.findall(text))))
        
        # re.findall med grupper returnerer tupler, så vi sammensætter dem
        raw_phone_matches = PHONE_PATTERN.findall(text)
        phones = sorted(list(set(["".join(match).replace(" ", "") for match in raw_phone_matches])))

        contact_data: ConversionMetrics = {
            'emails_found': emails,
            'phone_numbers_found': phones
        }
        return contact_data

    except Exception as e:
        log.error(f"Fejl under detektion af kontaktinformation: {e}", exc_info=True)
        return {'emails_found': [], 'phone_numbers_found': []}