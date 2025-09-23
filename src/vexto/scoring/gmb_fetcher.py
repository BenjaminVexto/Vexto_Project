# gmb_fetcher.py
import os
import re
import logging
from difflib import SequenceMatcher
from urllib.parse import urlparse, quote_plus
from .http_client import AsyncHtmlClient
from .schemas import SocialAndReputationMetrics

log = logging.getLogger(__name__)

# -----------------------------
# Locale & geobias helpers
# -----------------------------

def _resolve_places_locale(url: str) -> dict:
    """
    Returnér {'language': 'xx', 'region': 'XX'} ud fra TLD (best-effort).
    Brug neutral/global hvis ukendt.
    """
    host = (urlparse(url).hostname or "").lower()
    tld = host.rsplit('.', 1)[-1] if '.' in host else ''
    mapping = {
        'dk': ('da', 'dk'),
        'se': ('sv', 'se'),
        'no': ('nb', 'no'),
        'fi': ('fi', 'fi'),
        'de': ('de', 'de'),
        'nl': ('nl', 'nl'),
        'fr': ('fr', 'fr'),
        'es': ('es', 'es'),
        'it': ('it', 'it'),
        'pl': ('pl', 'pl'),
        'pt': ('pt', 'pt'),
        'cz': ('cs', 'cz'),
        'sk': ('sk', 'sk'),
        'hu': ('hu', 'hu'),
        'ro': ('ro', 'ro'),
        'bg': ('bg', 'bg'),
        'gr': ('el', 'gr'),
        'lt': ('lt', 'lt'),
        'lv': ('lv', 'lv'),
        'ee': ('et', 'ee'),
        'ie': ('en', 'ie'),
        'uk': ('en', 'gb'),
        'co': ('es', 'co'),
        'mx': ('es', 'mx'),
        'ar': ('es', 'ar'),
        'br': ('pt', 'br'),
        'au': ('en', 'au'),
        'nz': ('en', 'nz'),
        'ca': ('en', 'ca'),
        'us': ('en', 'us'),
    }
    if tld in mapping:
        lang, reg = mapping[tld]
        return {'language': lang, 'region': reg}
    return {'language': None, 'region': None}


def _locationbias_from_url(url: str) -> str:
    """
    Returnér en deterministisk locationbias-streng baseret på TLD.
    Fald tilbage til ipbias, hvis ukendt.
    Format: "circle:<radius_m>@<lat>,<lng>" eller "ipbias".
    """
    host = (urlparse(url).hostname or "").lower()
    tld = host.rsplit('.', 1)[-1] if '.' in host else ''

    # center_lat, center_lng, radius_m (bevidst rummelig radius)
    centers = {
        'dk': (56.0, 10.0, 250000),
        'se': (60.0, 15.0, 400000),
        'no': (60.5, 8.5, 350000),
        'fi': (64.0, 26.0, 450000),
        'de': (51.0, 10.0, 400000),
        'nl': (52.2, 5.3, 250000),
        'fr': (46.5, 2.2, 500000),
        'uk': (53.5, -2.0, 350000),
        'ie': (53.4, -8.0, 250000),
        'us': (39.8, -98.6, 1500000),
        'ca': (56.1, -106.3, 1500000),
        'au': (-25.3, 133.8, 1500000),
        'nz': (-41.0, 174.0, 400000),
        'br': (-15.6, -47.9, 1500000),
        'mx': (23.6, -102.5, 1000000),
        'es': (40.2, -3.7, 500000),
        'it': (42.5, 12.5, 500000),
        'pl': (52.1, 19.2, 400000),
        'pt': (39.6, -8.0, 350000),
    }
    if tld in centers:
        lat, lng, radius = centers[tld]
        return f"circle:{radius}@{lat},{lng}"
    return "ipbias"


def _center_from_url(url: str):
    """Bruges til Text Search fallback (lat,lng,radius) eller None."""
    bias = _locationbias_from_url(url)
    if bias.startswith("circle:"):
        try:
            r_part, coord = bias.split("@", 1)
            radius = int(r_part.split(":")[1])
            lat, lng = map(float, coord.split(","))
            return lat, lng, radius
        except Exception:
            return None
    return None


# -----------------------------
# Query helpers
# -----------------------------

def _guess_business_name_from_url(url: str) -> str:
    """Gætter på et virksomhedsnavn ud fra domænet."""
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


def _build_search_queries(url: str, company_name: str | None) -> list[str]:
    """
    Prioriteret søgeliste uden duplikater:
      1) renset firmanavn
      2) fuldt firmanavn
      3) domæne-baseret gæt
      4) extra danske varianter for enkeltords-brands
    """
    queries: list[str] = []
    if company_name:
        cleaned = _clean_company_name(company_name)
        for cand in (cleaned, company_name):
            if cand and cand not in queries:
                queries.append(cand)

    url_guess = _guess_business_name_from_url(url)
    if url_guess and url_guess not in queries:
        queries.append(url_guess)

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
    return [q for q in queries if not (q in seen or seen.add(q))]


# -----------------------------
# Result helpers
# -----------------------------

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
        "gmb_status": status,
    }


def _base_domain(host: str) -> str:
    host = (host or "").lower().strip().strip(".")
    parts = host.split(".")
    if len(parts) >= 2:
        return ".".join(parts[-2:])  # naive eTLD+1 (ok for .dk/.com/.net osv.)
    return host

def _domain_match(website_url: str, target_url: str) -> bool:
    try:
        wanted = (urlparse(target_url).hostname or "").lower()
        found  = (urlparse((website_url or "").strip()).hostname or "").lower()
        if not wanted or not found:
            return False

        # eksakt base-domæne-match (eTLD+1), og tillad subdomæner på kandidat
        wanted_base = _base_domain(wanted)
        found_base  = _base_domain(found)
        if wanted_base != found_base:
            return False

        # hvis base er ens, accepter subdomæner/’www’
        return True
    except Exception:
        return False


def _build_details_url(place_id: str, api_key: str, language: str | None) -> str:
    fields = "rating,user_ratings_total,name,formatted_address,opening_hours,website,photos,geometry"
    lang_param = f"&language={language}" if language else ""
    return (
        "https://maps.googleapis.com/maps/api/place/details/json"
        f"?place_id={place_id}&fields={fields}{lang_param}&key={api_key}"
    )

def _norm_company_name(s: str) -> str:
    """Normaliser firmanavn for robust sammenligning."""
    if not s:
        return ""
    s = s.strip().lower()
    # fjern selskabsendelser og nemme støjord
    s = re.sub(r'\b(aps|a\/s|ivS|aps\.|a\/s\.|ivS\.)\b', '', s, flags=re.I)
    s = re.sub(r'[^a-z0-9æøåéèüö ]+', ' ', s)
    s = re.sub(r'\s+', ' ', s).strip()
    return s

def _name_similarity(a: str, b: str) -> float:
    """Jaro/Levenshtein-lignende ratio via stdlib (SequenceMatcher)."""
    a_n = _norm_company_name(a)
    b_n = _norm_company_name(b)
    if not a_n or not b_n:
        return 0.0
    return SequenceMatcher(None, a_n, b_n).ratio()

def _target_company_from(url: str, company_name: str | None) -> str:
    """
    Best-effort mål-firmanavn: foretræk eksplicit company_name,
    ellers afled fra værtsnavn (fx 'vexto.dk' -> 'vexto').
    """
    if company_name:
        return company_name
    host = (urlparse(url).hostname or "").lower()
    base = host.split('.', 1)[0] if '.' in host else host
    return re.sub(r'[^a-z0-9æøå]', ' ', base).strip()


# -----------------------------
# Main fetcher
# -----------------------------

async def fetch_gmb_data(
    client: AsyncHtmlClient,
    url: str,
    cvr: str | None = None,
    company_name: str | None = None
) -> SocialAndReputationMetrics:
    """
    Henter Google Business-data via Places API:
      1) Søger på (a) renset firmanavn, (b) fuldt firmanavn, (c) domæne-baseret gæt
      2) Finder place_id med Find Place
      3) Henter detaljer (rating, reviews, website, åbningstider, fotos)
      4) Returnerer første match, som passer gates:
        - website-domænematch (foretrukket), eller
        - høj navnesimilaritet (>= 0.85)
        'has_reviews' er kun informativt og udløser IKKE accept alene.
    """
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        log.warning("GOOGLE_API_KEY mangler – springer GMB-tjek over.")
        return _empty_metrics(status="unknown")

    queries = _build_search_queries(url, company_name)
    if not queries:
        log.warning(f"Intet gyldigt virksomhedsnavn for GMB-søgning: url={url}, company_name={company_name}")
        return _empty_metrics(status="invalid_input")

    loc = _resolve_places_locale(url)
    lang_param = f"&language={loc['language']}" if loc['language'] else ""
    locationbias = _locationbias_from_url(url)

    log.info(f"Forsøger GMB for CVR {cvr or 'N/A'}, søgeliste: {queries}")

    # --------- Først: Find Place ----------
    for i, name in enumerate(queries, 1):
        log.info(f"🔎 [{i}/{len(queries)}] Søger place_id for '{name}'")

        find_url = (
            "https://maps.googleapis.com/maps/api/place/findplacefromtext/json"
            f"?input={quote_plus(name)}&inputtype=textquery"
            f"&fields=place_id,name,formatted_address"
            f"{lang_param}"
            f"&locationbias={locationbias}"
            f"&key={api_key}"
        )

        try:
            r1 = await client.httpx_get(find_url)
            j1 = r1.json()
        except Exception as e:
            log.error(f"Fejl under Place-ID opslag for '{name}': {e}")
            continue

        status = (j1 or {}).get("status")
        candidates = (j1 or {}).get("candidates") or []

        if status not in ("OK", "ZERO_RESULTS"):
            log.warning(f"FindPlace status {status} for søgning '{name}'")
        if not candidates:
            log.warning(f"Status {status} for søgning '{name}' – ingen kandidater")
            continue

        host_target = (urlparse(url).hostname or "").lower()

        for cand in candidates:
            place_id = (cand or {}).get("place_id")
            if not place_id:
                continue

            details_url = _build_details_url(place_id, api_key, loc['language'])
            try:
                r2 = await client.httpx_get(details_url)
                j2 = r2.json()
            except Exception as e:
                log.error(f"Fejl ved hentning af detaljer for Place ID '{place_id}': {e}")
                continue

            if (j2 or {}).get("status") != "OK":
                log.warning(f"Detalje-status {j2.get('status')} for Place ID '{place_id}'")
                continue

            result = (j2 or {}).get("result") or {}
            rating = result.get("rating")
            count = result.get("user_ratings_total") or 0
            photos = result.get("photos") or []
            opening_hours = result.get("opening_hours") or {}
            website = (result.get("website") or "").strip().lower()

            has_reviews = (isinstance(count, (int, float)) and count > 0) or (
                isinstance(rating, (int, float, str)) and str(rating).replace(",", ".").replace(" ", "").replace("\u00a0", "").isdigit()
            )

            target_name = _target_company_from(url, company_name)
            name_sim = _name_similarity(result.get("name") or "", target_name)
            name_ok = name_sim >= 0.85

            website_ok = _domain_match(website, url)

            # Valgfrit: simpelt by-tjek (hvis du vil gate mere).
            # Eksempel: kræv at target-by optræder i formatted_address, hvis du kan udlede 'target_city'.
            # target_city = ...  # hent fra site/caller hvis muligt
            # city_ok = (target_city.lower() in (result.get("formatted_address") or "").lower()) if target_city else True

            # HÅRD GATE:
            #  - Accepter KUN hvis website-match ELLER (høj navnesimilaritet)
            #  - 'has_reviews' må ALDRIG alene være accept-kriterie (det gav 'Veo Technologies'-fejlen)
            if website_ok or name_ok:  # og evt. 'and city_ok'
                has_hours = bool(opening_hours.get("weekday_text") or opening_hours.get("periods"))
                final_result: SocialAndReputationMetrics = {
                    "gmb_review_count": int(count or 0),
                    "gmb_average_rating": float(rating) if rating is not None else None,
                    "gmb_profile_complete": True,             # sæt kun True ved PASS af gates
                    "gmb_has_website": bool(website),
                    "gmb_has_hours": has_hours,
                    "gmb_photo_count": int(len(photos)),
                    "gmb_business_name": result.get("name"),
                    "gmb_address": result.get("formatted_address"),
                    "gmb_place_id": place_id,
                    "gmb_status": "ok"
                }
                log.info(
                    f"[GMB/ACCEPT] name='{result.get('name')}' sim={name_sim:.2f} "
                    f"website_ok={website_ok} reviews={count} url='{website}'"
                )
                return final_result

            # Ellers: log præcis hvorfor vi afviser
            log.warning(
                f"[GMB/REJECT] name='{result.get('name')}' sim={name_sim:.2f} "
                f"website_ok={website_ok} reason='gates_failed' website='{website}'"
            )

        log.warning(f"Ingen brugbar kandidat for '{name}', prøver næste søgeterm…")

    # --------- Fallback: Text Search nær TLD-center ----------
    center = _center_from_url(url)
    if center:
        lat, lng, radius = center
        host_target = (urlparse(url).hostname or "").lower()

        for i, name in enumerate(queries, 1):
            log.info(f"[Fallback/TextSearch] Søger '{name}' nær {lat},{lng} (r={radius}m)")
            try:
                ts_url = (
                    "https://maps.googleapis.com/maps/api/place/textsearch/json"
                    f"?query={quote_plus(name)}{lang_param}"
                    f"&location={lat},{lng}&radius={radius}"
                    f"&key={api_key}"
                )
                r_ts = await client.httpx_get(ts_url)
                j_ts = r_ts.json()
            except Exception as e:
                log.error(f"Fejl under Text Search for '{name}': {e}")
                continue

            results = (j_ts or {}).get("results") or []
            if not results:
                log.warning(f"[TextSearch] Ingen resultater for '{name}'")
                continue

            for res in results:
                place_id = (res or {}).get("place_id")
                if not place_id:
                    continue

                try:
                    details_url = _build_details_url(place_id, api_key, loc['language'])
                    r2 = await client.httpx_get(details_url)
                    j2 = r2.json()
                except Exception as e:
                    log.error(f"Fejl ved detaljer (TextSearch) for Place ID '{place_id}': {e}")
                    continue

                if (j2 or {}).get("status") != "OK":
                    log.warning(f"[TextSearch] Detalje-status {j2.get('status')} for Place ID '{place_id}'")
                    continue

                result = (j2 or {}).get("result") or {}
                rating = result.get("rating")
                count = result.get("user_ratings_total") or 0
                photos = result.get("photos") or []
                opening_hours = result.get("opening_hours") or {}
                website = (result.get("website") or "").strip().lower()

                has_reviews = (isinstance(count, (int, float)) and count > 0) or (
                    isinstance(rating, (int, float, str)) and str(rating).replace(",", ".").replace(" ", "").replace("\u00a0", "").isdigit()
                )
                website_ok = _domain_match(website, url)

                target_name = _target_company_from(url, company_name)
                name_sim = _name_similarity(result.get("name") or "", target_name)
                name_ok = name_sim >= 0.85

                if website_ok or name_ok:  # evt. også city gate, hvis du senere tilføjer den
                    has_hours = bool(opening_hours.get("weekday_text") or opening_hours.get("periods"))
                    final_result: SocialAndReputationMetrics = {
                        "gmb_review_count": int(count or 0),
                        "gmb_average_rating": float(rating) if rating is not None else None,
                        "gmb_profile_complete": True,
                        "gmb_has_website": bool(website),
                        "gmb_has_hours": has_hours,
                        "gmb_photo_count": int(len(photos)),
                        "gmb_business_name": result.get("name"),
                        "gmb_address": result.get("formatted_address"),
                        "gmb_place_id": place_id,
                        "gmb_status": "ok",
                    }
                    log.info(
                        f"[GMB/ACCEPT/FALLBACK] name='{result.get('name')}' sim={name_sim:.2f} "
                        f"website_ok={website_ok} reviews={count} url='{website}'"
                    )
                    return final_result
                log.warning(
                    f"[GMB/REJECT/FALLBACK] name='{result.get('name')}' sim={name_sim:.2f} "
                    f"website_ok={website_ok} reason='gates_failed' website='{website}'"
                )
            log.warning(f"[TextSearch] Ingen brugbar kandidat for '{name}' – prøver næste søgeterm…")

    # --------- Intet fundet ----------
    log.warning("GMB: Ingen kandidater fundet på nogen søgetermer.")
    return _empty_metrics(status="zero_results")
