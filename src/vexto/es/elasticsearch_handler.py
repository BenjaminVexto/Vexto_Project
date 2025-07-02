"""
elasticsearch_handler.py

Denne fil indeholder funktioner til at interagere med CVR.dk's
ElasticSearch API for at hente virksomhedsdata.
"""
import gzip
import json
import requests
import time
import os
import logging
from collections import defaultdict
from typing import List, Dict, Any

# Indlæs miljøvariabler fra .env-filen.
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    print("Advarsel: 'python-dotenv' bibliotek ikke fundet. Kan ikke indlæse .env-fil automatisk.")
    print("Sørg for, at miljøvariablerne ELASTICSEARCH_USER, ELASTICSEARCH_PASSWORD og ELASTICSEARCH_URL er sat manuelt.")

# --- Opsætning af Logging ---
log = logging.getLogger("vexto.es")
log.setLevel(logging.INFO)
if not log.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    log.addHandler(handler)


# --- Konfiguration for CVR API-hentning ---
ELASTICSEARCH_USER = os.getenv("ELASTICSEARCH_USER")
ELASTICSEARCH_PASSWORD = os.getenv("ELASTICSEARCH_PASSWORD")
ELASTICSEARCH_URL = os.getenv("ELASTICSEARCH_URL")

if not ELASTICSEARCH_USER or not ELASTICSEARCH_PASSWORD or not ELASTICSEARCH_URL:
    log.error("Fejl: En eller flere krævede miljøvariabler (ELASTICSEARCH_USER, ELASTICSEARCH_PASSWORD, ELASTICSEARCH_URL) er ikke sat.")
    log.error("Sørg for, at din .env-fil er korrekt konfigureret og indlæses.")
    raise RuntimeError("Manglende ElasticSearch login/URL konfiguration. Se log for detaljer.")

S = requests.Session()
S.auth = (ELASTICSEARCH_USER, ELASTICSEARCH_PASSWORD)


# --- Hjælpefunktioner ---

def _clean_branchekode_for_api(branche: str) -> str:
    """
    Fjerner punktummer og mellemrum fra branchekoden og padder til 6 cifre med foranstillede nuller.
    """
    return str(branche).replace('.', '').strip().zfill(6)

def _make_es_query(branche: str, postnumre: List[int], include_active_only: bool = True) -> Dict[str, Any]:
    """
    Bygger ElasticSearch forespørgslen for en given (renset) branchekode og liste af postnumre.
    Inkluderer filtrering for kun aktive virksomheder, hvis angivet.
    """
    postnumre_str = [str(p) for p in postnumre] # Konverter postnumre til strenge for 'terms'

    # Statusser der indikerer en IKKE-aktiv virksomhed (baseret på CVR's URL, i VERSALER)
    BAD_STATUSES_FOR_ES_EXCLUSION = [
        "OPHØRT", "UNDER FRIVILLIG LIKVIDATION", "UNDER KONKURS", "UNDER REASSUMERING",
        "UNDER REKONSTRUKTION", "UNDER TVANGSOPLØSNING", "OPLØST EFTER ERKLAERING",
        "OPLØST EFTER FRIVILLIG LIKVIDATION", "OPLØST EFTER FUSION",
        "OPLØST EFTER GRAENSEOVERSKRIDENDE FUSION", "OPLØST EFTER GRAENSEOVERSKRIDENDE HJEMSTEDSFLYTNING",
        "OPLØST EFTER GRAENSEOVERSKRIDENDE SPALTNING", "OPLØST EFTER KONKURS",
        "OPLØST EFTER SPALTNING", "SLETTET", "TVANGSOPLØST"
    ]

    # Basis filtre for ElasticSearch:
    filter_conditions = [
        {"term": {"Vrvirksomhed.virksomhedMetadata.nyesteHovedbranche.branchekode": branche}},
        # Kombiner postnummer OG kommunenavn (fra CVR's URL fritekst)
        {"bool": {
            "should": [
                {"terms": {"Vrvirksomhed.virksomhedMetadata.nyesteBeliggenhedsadresse.postnummer": postnumre_str}},
                {"match_phrase": {"Vrvirksomhed.virksomhedMetadata.nyesteBeliggenhedsadresse.kommuneNavn": "Greve"}}
            ],
            "minimum_should_match": 1
        }},
        # Ekskluder ALTID virksomheder der har en ophørsDato - dette er en klar markør for ophør
        {"bool": {"must_not": {"exists": {"field": "Vrvirksomhed.virksomhedMetadata.ophørsDato"}}}}
    ]

    # Den samlede bool query struktur
    es_query_bool = {
        "filter": filter_conditions
    }

    if include_active_only:
        # must_not: Ekskluderer alle dokumenter, der har en af de "dårlige" statusser i deres virksomhedsstatus-array.
        must_not_status_conditions = []
        for bad_phrase in BAD_STATUSES_FOR_ES_EXCLUSION:
            must_not_status_conditions.append({
                "nested": {
                    "path": "Vrvirksomhed.virksomhedsstatus",
                    "query": { "match_phrase": { "Vrvirksomhed.virksomhedsstatus.status": bad_phrase } }
                }
            })

        # Hvis der er nogen must_not statusser, tilføj dem til den overordnede query
        if must_not_status_conditions:
            es_query_bool["must_not"] = must_not_status_conditions

    es_query_body = {
        "_source": [
            "Vrvirksomhed.cvrNummer",
            "Vrvirksomhed.virksomhedMetadata.nyesteNavn.navn",
            "Vrvirksomhed.virksomhedMetadata.nyesteBeliggenhedsadresse.vejnavn",
            "Vrvirksomhed.virksomhedMetadata.nyesteBeliggenhedsadresse.husnummerFra",
            "Vrvirksomhed.virksomhedMetadata.nyesteBeliggenhedsadresse.postnummer",
            "Vrvirksomhed.virksomhedMetadata.nyesteBeliggenhedsadresse.postdistrikt",
            "Vrvirksomhed.virksomhedMetadata.nyesteKontaktoplysninger", # Behold denne - nu er det vores primære kilde
            "Vrvirksomhed.virksomhedMetadata.nyesteHovedbranche.branchekode",
            "Vrvirksomhed.virksomhedMetadata.nyesteHovedbranche.branchetekst",
            "Vrvirksomhed.virksomhedMetadata.nyesteVirksomhedsform.virksomhedsformkode",
            "Vrvirksomhed.virksomhedMetadata.nyesteVirksomhedsform.langBeskrivelse",
            "Vrvirksomhed.virksomhedMetadata.sammensatStatus",
            "Vrvirksomhed.virksomhedMetadata.ophørsDato",
            # Fjernet: Vrvirksomhed.telefonNummer og Vrvirksomhed.elektroniskPost
            # da Vrvirksomhed.virksomhedMetadata.nyesteKontaktoplysninger nu er den samlede kilde
            "Vrvirksomhed.deltagerRelation"
        ],
        "query": { "bool": es_query_bool },
        "sort": [{"_doc": "asc"}],
        "size": 3000
    }

    return es_query_body

def _unpack_es_response(response: requests.Response) -> Dict[str, Any]:
    """
    Dekomprimerer og deserialiserer en JSON-respons fra ElasticSearch.
    """
    try:
        return response.json()
    except json.JSONDecodeError as e:
        raw_content = response.content
        if raw_content.startswith(b'\x1f\x8b'):
            try:
                return json.loads(gzip.decompress(raw_content))
            except (gzip.BadGzipFile, json.JSONDecodeError) as gz_e:
                log.error(f"Fallback Gzip/JSON fejl under dekomprimering: {gz_e}")
                log.error(f"Rå svarindhold (første 500 tegn): {raw_content[:500]}...")
                raise
        log.error(f"JSONDecodeError: Kunne ikke parse svar som JSON: {e}")
        log.error(f"Rå svarindhold (første 500 tegn): {response.text[:500]}...")
        raise


def _exponential_backoff_retry(func: Any, *args: Any, **kwargs: Any) -> requests.Response:
    """
    Udfører en funktion (typisk et requests API-kald) med eksponentiel back-off
    ved specifikke HTTP-fejl (429 - Too Many Requests, 503 - Service Unavailable).
    """
    retries = 0
    max_retries = 5
    base_delay = 1

    request_url = args[0] if args else "Ukendt URL"
    request_json_body = kwargs.get('json', {})
    request_params = kwargs.get('params', {})

    while retries < max_retries:
        try:
            if log.isEnabledFor(logging.DEBUG):
                log.debug(f"Sender forespørgsel til {request_url}")
                if request_params:
                    log.debug(f"URL-parametre: {json.dumps(request_params, indent=2, ensure_ascii=False)}")
                if request_json_body:
                    log.debug(f"Body: {json.dumps(request_json_body, indent=2, ensure_ascii=False)}")

            response = func(*args, **kwargs)
            response.raise_for_status()
            return response
        except requests.exceptions.HTTPError as e:
            if e.response.status_code in [429, 503]:
                delay = base_delay * (2 ** retries)
                log.warning(f"Rate limit / Service fejl ({e.response.status_code}). Forsøger igen om {delay:.1f} sekunder (forsøg {retries + 1}/{max_retries}).")
                time.sleep(delay)
                retries += 1
            else:
                log.error(f"HTTP Fejl ({e.response.status_code}) for URL {request_url}: {e}")
                log.error(f"Svarindhold fra API: {e.response.text}")
                raise
        except requests.exceptions.ConnectionError as e:
            log.error(f"Forbindelsesfejl til {request_url}: {e}. Forsøger igen om {base_delay:.1f} sekunder (forsøg {retries + 1}/{max_retries}).")
            time.sleep(base_delay)
            retries += 1
            base_delay *= 1.5
        except Exception as e:
            log.error(f"En uventet fejl opstod under API-kald: {e}", exc_info=True)
            raise

    raise RuntimeError(f"API-kald mislykkedes efter {max_retries} forsøg på grund af gentagne fejl.")


def search_companies(postnumre: List[int], branchekode: str, size: int = 3000, scroll_ttl: str = "1m") -> List[Dict[str, Any]]:
    """
    Henter virksomhedsdata fra CVR API'et for en given branchekode og liste af postnumre.
    """
    all_hits = []
    branchekode_cleaned = _clean_branchekode_for_api(branchekode)

    log.info(f"Forespørger CVR for branche: {branchekode_cleaned} og postnumre: {', '.join(map(str, postnumre[:5]))}{'...' if len(postnumre) > 5 else ''}")

    query_body = _make_es_query(branchekode_cleaned, postnumre, include_active_only=True)

    query_params = {
        "size": size
    }

    try:
        response = _exponential_backoff_retry(S.post, ELASTICSEARCH_URL, json=query_body, params=query_params)
        data = _unpack_es_response(response)

        hits = data.get("hits", {}).get("hits", [])
        all_hits.extend(h['_source'] for h in hits)
        log.info(f"Hentede {len(hits)} hits. Totalt: {len(all_hits)}")

        total_hits_obj = data.get("hits", {}).get("total", 0)
        if isinstance(total_hits_obj, dict):
            total_hits_found = total_hits_obj.get("value", 0)
        else:
            total_hits_found = total_hits_obj

        if total_hits_found > size:
            log.warning(f"CVR API'et tillader kun at hente de første {size} hits. Fandt {total_hits_found} totalt for denne forespørgsel. Overvej 'search_after' for fuld paginering.")

    except requests.exceptions.RequestException as e:
        log.error(f"Netværks-eller HTTP-fejl under CVR API-kald: {e}")
    except RuntimeError as e:
        log.error(f"Runtime fejl under API-kald: {e}")
    except Exception as e:
        log.error(f"En uventet fejl opstod under API-kald: {e}", exc_info=True)

    return all_hits

# --- Eksempel på brug (kun hvis denne fil køres direkte, ikke når importeret af main.py) ---
if __name__ == "__main__":
    log.info("Dette er en modulfil og bør normalt importeres af main.py.")
    log.info("Kører en lille testkørsel...")

    try:
        test_postnumre = [2670]
        test_branchekode = "43.32.00"

        found_companies = search_companies(test_postnumre, test_branchekode, size=100)

        if found_companies:
            log.info(f"Testkørsel fuldført. Fundet {len(found_companies)} virksomheder for {test_branchekode} i {test_postnumre}.")
            for company in found_companies[:3]:
                sammensat_status = company.get('Vrvirksomhed_virksomhedMetadata_sammensatStatus', 'N/A')
                ophors_dato = company.get('Vrvirksomhed_virksomhedMetadata_ophørsDato', 'N/A')
                virksomhedsform_kode = company.get('Vrvirksomhed_virksomhedMetadata_nyesteVirksomhedsform_virksomhedsformkode', 'N/A')
                virksomhedsform_text = company.get('Vrvirksomhed_virksomhedMetadata_nyesteVirksomhedsform_langBeskrivelse', 'N/A')
                nyeste_kontakt = company.get('Vrvirksomhed_virksomhedMetadata_nyesteKontaktoplysninger', 'N/A') # Rå kontakt
                deltager_relation_raw = company.get('Vrvirksomhed_deltagerRelation', 'N/A')

                # Logik for at parse nyeste_kontakt for test-output
                telefon_nummers = []
                emails = []
                websites = []
                if isinstance(nyeste_kontakt, list):
                    for item in nyeste_kontakt:
                        if isinstance(item, dict) and 'kontaktoplysning' in item:
                            val = item.get('kontaktoplysning', '').strip()
                            if '@' in val:
                                emails.append(val)
                            elif 'http' in val or 'www.' in val:
                                websites.append(val)
                            elif re.fullmatch(r"\d{8}", val): # Simple check for 8 digits for phone
                                telefon_nummers.append(val)
                            
                telefon_str = '; '.join(telefon_nummers) if telefon_nummers else 'N/A'
                email_str = '; '.join(emails) if emails else 'N/A'
                website_str = '; '.join(websites) if websites else 'N/A'


                director_name = "N/A"
                if isinstance(deltager_relation_raw, list):
                    for dr in deltager_relation_raw:
                        if dr.get('deltager', {}).get('enhedstype') == 'PERSON':
                            organisations = dr.get('organisationer')
                            if isinstance(organisations, list):
                                for org in organisations:
                                    hovedtype = (org.get("hovedtype") or "").upper()
                                    org_navn_list = org.get('organisationsNavn')
                                    org_role_from_name = (org_navn_list[0].get('navn') if isinstance(org_navn_list, list) and org_navn_list else "").upper()

                                    if hovedtype in {"DIREKTION", "LEDELSESORGAN", "INDEHAVER", "FULDT_ANSVARLIG_DELTAGERE"} or \
                                       org_role_from_name in {"DIREKTION", "INTERESSENTER"}:
                                        person_names = dr.get('deltager', {}).get('navne')
                                        if isinstance(person_names, list) and person_names:
                                            director_name = person_names[-1].get('navn', 'N/A')
                                            break
                                    if director_name != "N/A": break
                            if director_name != "N/A": break # Break outer loop if director found
                
                log.info(f"  - CVR: {company.get('Vrvirksomhed_cvrNummer')}, Navn: {company.get('Vrvirksomhed_virksomhedMetadata_nyesteNavn_navn')}, Form: {virksomhedsform_kode} ({virksomhedsform_text}), Status: {sammensat_status}, OphørsDato: {ophors_dato}, Tlf: {telefon_str}, Mail: {email_str}, Hjemmeside: {website_str}, Direktør: {director_name}")
        else:
            log.info("Testkørsel: Ingen virksomheder fundet eller fejl opstod.")
    except RuntimeError as e:
        log.error(f"Testkørsel fejlede på grund af konfigurationsfejl: {e}")
    except Exception as e:
        log.error(f"En uventet fejl opstod under testkørsel: {e}", exc_info=True)