#gmb_fetcher.py

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
    Henter Google Business-data via Places API:
      1) Søger på (a) renset firmanavn, (b) fuldt firmanavn, (c) domæne-baseret gæt
      2) Finder place_id med Find Place
      3) Henter detaljer (rating, reviews, website, åbningstider, fotos)
    Returnerer første match, der har reelle reviews (count>0 eller rating>0).

    Output har faste nøgler:
      - gmb_review_count: int
      - gmb_average_rating: float|None
      - gmb_profile_complete: bool
      - gmb_has_website: bool
      - gmb_has_hours: bool
      - gmb_photo_count: int
      - gmb_business_name: str|None
      - gmb_address: str|None
      - gmb_place_id: str|None
    """

    def _empty_metrics(status: str = "unknown") -> SocialAndReputationMetrics:
        return {
            "gmb_review_count": 0,
            "gmb_average_rating": None,
            "gmb_profile_complete": False,
            "gmb_has_website": False,
            "gmb_has_hours": False,
            "gmb_photo_count": 0,
            "gmb_business_name": None,
            "gmb_address": None,
            "gmb_place_id": None,
            # NYT: markér neutral/ukendt
            "gmb_status": status,
        }

    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        log.warning("GOOGLE_API_KEY mangler – springer GMB-tjek over.")
        # Ny: eksplicit 'unknown' status
        return _empty_metrics(status="unknown")

    # Byg prioriteret søgeliste uden duplikater
    queries: list[str] = []
    if company_name:
        cleaned = _clean_company_name(company_name)
        for cand in (cleaned, company_name):
            if cand and cand not in queries:
                queries.append(cand)

    url_guess = _guess_business_name_from_url(url)
    if url_guess and url_guess not in queries:
        queries.append(url_guess)

    # Ekstra danske varianter for enkeltords-brands
    base = (queries[0] if queries else url_guess) or ""
    if base and " " not in base:
        extra = [
            f"{base} Danmark",
            f"{base} ApS",
            f"{base}.dk",
            f"{base} marketing",
            f"{base} bureau",
        ]
        for q in extra:
            if q not in queries:
                queries.append(q)

    # Dedup – bevar rækkefølgen
    seen = set()
    queries = [q for q in queries if not (q in seen or seen.add(q))]

    if not queries:
        log.warning(f"Intet gyldigt virksomhedsnavn for GMB-søgning: url={url}, company_name={company_name}")
        return _empty_metrics()

    log.info(f"Forsøger GMB for CVR {cvr or 'N/A'}, søgeliste: {queries}")

    for i, name in enumerate(queries, 1):
        log.info(f"🔎 [{i}/{len(queries)}] Søger place_id for '{name}'")
        find_url = (
            "https://maps.googleapis.com/maps/api/place/findplacefromtext/json"
            f"?input={quote_plus(name)}&inputtype=textquery"
            f"&fields=place_id&locationbias=ipbias&key={api_key}"
        )

        try:
            r1 = await client.httpx_get(find_url)
            j1 = r1.json()
        except Exception as e:
            log.error(f"Fejl under Place-ID opslag for '{name}': {e}")
            continue

        if j1.get("status") != "OK" or not j1.get("candidates"):
            log.warning(f"Status {j1.get('status')} for søgning '{name}' – ingen kandidater")
            continue

        place_id = (j1["candidates"][0] or {}).get("place_id")
        if not place_id:
            log.warning(f"Ingen place_id i kandidat for '{name}'")
            continue

        log.info(f"→ Fandt Place ID '{place_id}' for '{name}'")

        fields = "rating,user_ratings_total,name,formatted_address,opening_hours,website,photos"
        details_url = (
            "https://maps.googleapis.com/maps/api/place/details/json"
            f"?place_id={place_id}&fields={fields}&key={api_key}&language=da"
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

        result = j2.get("result") or {}
        rating = result.get("rating")
        count = result.get("user_ratings_total")
        photos = result.get("photos") or []
        opening_hours = result.get("opening_hours") or {}

        # Kræv reelt signal før vi accepterer profilen
        has_reviews = (isinstance(count, (int, float)) and count > 0) or (
            isinstance(rating, (int, float, str)) and float(rating) > 0
        )

        # Fallback: ingen reviews, men domains match → brug profilen (giver måleligt signal)
        website_ok = False
        try:
            site = (result.get("website") or "").strip().lower()
            host_wanted = (urlparse(url).hostname or "").lower()
            host_found = (urlparse(site).hostname or "").lower()
            website_ok = bool(site) and (host_wanted and host_found and host_wanted.endswith(host_found) or host_found.endswith(host_wanted))
        except Exception:
            website_ok = False

        if not (has_reviews or website_ok):
            log.warning(f"Ingen anmeldelser/bedømmelse og intet domæne-match for '{name}', prøver næste…")
            continue

        has_hours = bool(opening_hours.get("weekday_text") or opening_hours.get("periods"))

        final_result: SocialAndReputationMetrics = {
            "gmb_review_count": int(count or 0),
            "gmb_average_rating": float(rating) if rating is not None else None,
            "gmb_profile_complete": True,
            "gmb_has_website": bool(result.get("website")),
            "gmb_has_hours": has_hours,
            "gmb_photo_count": int(len(photos)),
            "gmb_business_name": result.get("name"),
            "gmb_address": result.get("formatted_address"),
            "gmb_place_id": place_id,
            # NYT
            "gmb_status": "ok",
        }

        log.info(
            f"GMB data fetched once: review_count={final_result['gmb_review_count']}, "
            f"rating={final_result['gmb_average_rating']}"
        )
        return final_result

    # Hvis vi kommer hertil, fandt vi ikke en sikker profil
    log.warning(f"GMB-søgning mislykkedes for {company_name or url}")
    return _empty_metrics(status="unknown")