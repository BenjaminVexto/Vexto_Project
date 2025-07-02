# src/vexto/scoring/gmb_fetcher.py
import os
import logging
from urllib.parse import urlparse, quote_plus
from .http_client import AsyncHtmlClient
from .schemas import SocialAndReputationMetrics

log = logging.getLogger(__name__)

def _guess_business_name_from_url(url: str) -> str:
    """ Gætter på et virksomhedsnavn ud fra domænet. """
    hostname = urlparse(url).hostname
    if not hostname:
        return ""
    
    # Fjerner www. og .dk/.com etc.
    name = hostname.replace("www.", "").split('.')[0]
    # Erstatter bindestreger med mellemrum og gør første bogstav stort
    return name.replace("-", " ").title()

async def fetch_gmb_data(client: AsyncHtmlClient, url: str) -> SocialAndReputationMetrics:
    """
    Forsøger at finde en Google Business-profil og hente anmeldelsesdata.
    """
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        log.warning("GOOGLE_API_KEY mangler – springer GMB-tjek over.")
        return {}

    business_name = _guess_business_name_from_url(url)
    if not business_name:
        return {}

    log.info(f"Forsøger at finde GMB-data for: '{business_name}'")
    
    try:
        # TRIN 1: Find Place ID ud fra navn
        find_place_url = (
            "https://maps.googleapis.com/maps/api/place/findplacefromtext/json"
            f"?input={quote_plus(business_name)}&inputtype=textquery"
            f"&fields=place_id&key={api_key}"
        )
        
        find_response = await client.httpx_get(find_place_url)
        find_data = find_response.json()

        if find_data.get("status") != "OK" or not find_data.get("candidates"):
            log.warning(f"Kunne ikke finde et 'Place ID' for '{business_name}'")
            return {}

        place_id = find_data["candidates"][0]["place_id"]

        # TRIN 2: Hent detaljer ud fra Place ID
        details_url = (
            "https://maps.googleapis.com/maps/api/place/details/json"
            f"?place_id={place_id}&fields=rating,user_ratings_total"
            f"&key={api_key}"
        )
        
        details_response = await client.httpx_get(details_url)
        details_data = details_response.json()
        
        if details_data.get("status") != "OK" or not details_data.get("result"):
            log.warning(f"Kunne ikke hente detaljer for Place ID '{place_id}'")
            return {}
        
        result = details_data["result"]
        
        return {
            'gmb_review_count': result.get('user_ratings_total'),
            'gmb_average_rating': result.get('rating'),
            'gmb_profile_complete': True # Vi antager at profilen er komplet, hvis vi finder den
        }

    except Exception as e:
        log.error(f"Fejl under GMB fetch for '{business_name}': {e}", exc_info=True)
        return {}