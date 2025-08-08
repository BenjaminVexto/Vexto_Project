import os
import re
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
    name = hostname.replace("www.", "").split('.')[0]
    return name.replace("-", " ").title()

def _clean_company_name(name: str) -> str:
    """
    Fjerner almindelige juridiske endelser fra et virksomhedsnavn, fx "A/S", "ApS" osv.
    """
    if not isinstance(name, str):
        return ""
    suffixes = [r'\bA/S\b', r'\bApS\b', r'\bI/S\b', r'\bK/S\b', r'\bIVS\b']
    cleaned = name
    for s in suffixes:
        cleaned = re.sub(s, '', cleaned, flags=re.IGNORECASE)
    # Fjern specialtegn, kun bogstaver, tal og mellemrum
    cleaned = re.sub(r'[^0-9A-Za-zæøåÆØÅ ]+', '', cleaned)
    return cleaned.strip()

async def fetch_gmb_data(
    client: AsyncHtmlClient,
    url: str,
    cvr: str = None,
    company_name: str = None
) -> SocialAndReputationMetrics:
    """
    Henter Google Business-data med en flertrins-strategi:
      1) Renset brand-navn (fjern juridiske endelser)
      2) Fuldt juridisk navn
      3) URL-baseret gæt
    Stopper ved første profil, hvor vi rent faktisk får reviews.
    """
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        log.warning("GOOGLE_API_KEY mangler – springer GMB-tjek over.")
        return {}

    # Byg prioriteret søgeliste uden duplikater
    queries = []
    if company_name:
        cleaned = _clean_company_name(company_name)
        if cleaned and cleaned not in queries:
            queries.append(cleaned)
        if company_name not in queries:
            queries.append(company_name)
    url_guess = _guess_business_name_from_url(url)
    if url_guess and url_guess not in queries:
        queries.append(url_guess)

    if not queries:
        log.warning(f"Intet gyldigt virksomhedsnavn for GMB-søgning: url={url}, company_name={company_name}")
        return {}

    log.info(f"Forsøger GMB for CVR {cvr or ''}, søgeliste: {queries}")

    # TRIN 1+2: For hvert navn, find place_id og hent detaljer med det samme.
    for name in queries:
        # Find Place ID
        find_url = (
            "https://maps.googleapis.com/maps/api/place/findplacefromtext/json"
            f"?input={quote_plus(name)}&inputtype=textquery"
            f"&fields=place_id&key={api_key}"
        )
        try:
            r1 = await client.httpx_get(find_url)
            j1 = r1.json()
        except Exception as e:
            log.error(f"Fejl under Place-ID opslag for '{name}': {e}")
            continue

        if j1.get("status") != "OK" or not j1.get("candidates"):
            log.warning(f"Status {j1.get('status')} for søgning '{name}'")
            continue

        place_id = j1["candidates"][0]["place_id"]
        log.info(f"→ Fandt Place ID '{place_id}' for '{name}'")

        # Hent detaljer med det samme
        details_url = (
            "https://maps.googleapis.com/maps/api/place/details/json"
            f"?place_id={place_id}&fields=rating,user_ratings_total"
            f"&key={api_key}&language=da"
        )
        try:
            r2 = await client.httpx_get(details_url)
            j2 = r2.json()
        except Exception as e:
            log.error(f"Fejl ved hentning af detaljer for '{name}': {e}")
            continue

        if j2.get("status") != "OK":
            log.warning(f"Detalje-status {j2.get('status')} for Place ID '{place_id}'")
            continue

        result = j2.get("result", {})
        # Bryd kun, hvis vi faktisk har rating eller antal reviews
        if result.get("rating") is not None or result.get("user_ratings_total") is not None:
            return {
                "gmb_review_count": result.get("user_ratings_total"),
                "gmb_average_rating": result.get("rating"),
                "gmb_profile_complete": True
            }
        else:
            log.warning(f"Ingen anmeldelser for '{name}', prøver næste…")
            continue

    # Hvis vi kommer hertil, fandt vi aldrig en profil med reviews
    log.warning(f"GMB-søgning mislykkedes for {company_name or url}")
    return {}