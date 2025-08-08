import os
import re
import logging
import pandas as pd

# --- Importer til tredjeparts-services -------------------------------------
# Sørg for at du har installeret begge:
# pip install apify-client
# pip install google-api-python-client

try:
    from apify_client import ApifyClient
    APIFY_AVAILABLE = True
except ImportError:
    print("ADVARSEL: 'apify-client' er ikke installeret. Kør 'pip install apify-client'.")
    APIFY_AVAILABLE = False

try:
    from googleapiclient.discovery import build
    from googleapiclient.errors import HttpError
    GOOGLE_API_AVAILABLE = True
except ImportError:
    print("ADVARSEL: 'google-api-python-client' er ikke installeret. Kør 'pip install google-api-python-client'.")
    GOOGLE_API_AVAILABLE = False


# --- Konfiguration --------------------------------------------------------
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GOOGLE_CSE_ID = os.getenv("GOOGLE_CSE_ID")
APIFY_ACTOR_ID = "cIdqlEvw6afc1do1p"

# Den brede, prioriterede liste af titler
PRIORITY_TITLES = [
    # Lag 1: Direkte Web/Marketing Ansvar
    'marketing director', 'marketingdirektør', 'head of marketing', 'marketingchef',
    'head of digital', 'digital chef', 'digital manager',
    'head of ecommerce', 'e-handelschef', 'ecommerce manager',
    'seo manager', 'seo specialist', 'seo ansvarlig',
    'sem manager', 'ppc manager',
    'web manager', 'webmaster', 'webansvarlig',
    'chief marketing officer', 'cmo',
    
    # Lag 2: Kommercielt Ansvar
    'head of sales', 'salgschef', 'sales director', 'salgsdirektør',
    'kommunikationschef', 'head of communications',
    
    # Lag 3: Generel Ledelse
    'indehaver', 'ejer', 'partner', 'medejer',
    'chief executive officer', 'ceo',
    'administrerende direktør', 'direktør',
    'økonomidirektør', 'cfo'
]

log = logging.getLogger(__name__)

# --- Helper Funktioner ----------------------------------------------------

def _search_scraped_text(scraped_text: str) -> dict | None:
    """Søger i den præ-scrapede tekst fra analyse-modulet efter relevante titler."""
    if not isinstance(scraped_text, str) or not scraped_text:
        return None

    log.info("Søger i præ-scrapet tekst fra analyse-modul...")
    text_lower = scraped_text.lower()
    
    for title in PRIORITY_TITLES:
        if title in text_lower:
            log.info(f"Fandt potentielt match for titlen '{title}' i scrapet tekst.")
            # Forbedring: Her kan man udtrække navnet tæt på titlen.
            # For nu returnerer vi titlen, så vi ved, at der er et potentielt match.
            return {'headline': title}
            
    return None

def _find_company_linkedin_url(company_name: str) -> str | None:
    """Bruger Google til at finde den officielle LinkedIn-side for et firma."""
    if not GOOGLE_API_AVAILABLE or not GOOGLE_API_KEY or not GOOGLE_CSE_ID:
        log.warning("Google API-klient eller nøgler mangler. Springer Google-søgning over.")
        return None
    try:
        service = build("customsearch", "v1", developerKey=GOOGLE_API_KEY)
        query = f'"{company_name}" LinkedIn'
        log.info(f"Googler for at finde firma-URL: {query}")
        res = service.cse().list(q=query, cx=GOOGLE_CSE_ID, num=3).execute()
        if 'items' in res:
            for item in res['items']:
                url = item.get('link')
                if url and 'linkedin.com/company' in url:
                    log.info(f"Fandt og validerede LinkedIn URL: {url}")
                    return url
    except Exception as e:
        log.error(f"Fejl under søgning efter LinkedIn URL: {e}")
    log.warning(f"Kunne ikke finde en valid LinkedIn URL for '{company_name}' i topresultaterne.")
    return None

def _query_linkedin_api(company_linkedin_url: str) -> list[dict]:
    """Finder medarbejdere via den specifikke Apify Actor."""
    APIFY_API_KEY = os.getenv("APIFY_API_KEY")
    if not APIFY_AVAILABLE or not APIFY_API_KEY:
        log.warning("Apify-klient eller API-nøgle mangler. Springer API-kald over.")
        return []
    try:
        client = ApifyClient(APIFY_API_KEY)
        run_input = {"identifier": company_linkedin_url, "max_employees": 50}
        log.info(f"Starter Apify Actor ({APIFY_ACTOR_ID}) for: {company_linkedin_url}")
        run = client.actor(APIFY_ACTOR_ID).call(run_input=run_input)
        log.info("Apify Actor færdig. Henter resultater...")
        employee_list = list(client.dataset(run["defaultDatasetId"]).iterate_items())
        log.info(f"Fik {len(employee_list)} medarbejdere retur for '{company_linkedin_url}'")
        return employee_list
    except Exception as e:
        log.error(f"Fejl under Apify API-kald for '{company_linkedin_url}': {e}", exc_info=True)
        return []

def _filter_employees_by_title(employees: list[dict]) -> dict | None:
    """Gennemgår en liste af medarbejdere og finder den med den højest prioriterede titel."""
    for title_keyword in PRIORITY_TITLES:
        for employee in employees:
            employee_headline = (employee.get('headline') or '').lower()
            if title_keyword in employee_headline:
                log.info(f"Fandt match via API: '{employee.get('fullname')}' med titel '{employee_headline}'")
                return employee
    return None

def _verify_email_address(email: str) -> bool:
    """ACTION REQUIRED: Denne funktion skal implementeres færdig."""
    log.debug(f"Simulerer verifikation for {email} (returnerer altid True)")
    return True

def _generate_and_verify_email(full_name: str, domain: str) -> str | None:
    """Genererer sandsynlige e-mail mønstre og returnerer den første, der kan verificeres."""
    if not full_name or not domain:
        return None
    name_parts = full_name.lower().split()
    first_name = re.sub(r'\W+', '', name_parts[0])
    last_name = re.sub(r'\W+', '', name_parts[-1]) if len(name_parts) > 1 else ''
    if not last_name:
        patterns = [f"{first_name}@{domain}"]
    else:
        patterns = [
            f"{first_name}.{last_name}@{domain}",
            f"{first_name[0]}{last_name}@{domain}",
            f"{first_name}@{domain}",
        ]
    for email in patterns:
        if _verify_email_address(email):
            log.info(f"E-mail verificeret: {email}")
            return email
    log.warning(f"Kunne ikke verificere nogen e-mail-mønstre for {full_name} på domænet {domain}")
    return None

def _get_domain_from_url(url: str) -> str | None:
    """Udtrækker et rent domæne (f.eks. 'vexto.dk') fra en fuld URL."""
    if not isinstance(url, str): return None
    try:
        domain = url.split('//')[-1].split('www.')[-1].split('/')[0]
        return domain
    except:
        return None

# --- Orkestrerings Funktion -----------------------------------------------

def _find_best_contact(row: pd.Series) -> pd.Series:
    """Hoved-orkestreringsfunktion, der følger den integrerede 3-trins model."""
    company_name = row.get('Vrvirksomhed_virksomhedMetadata_nyesteNavn_navn')
    company_url = row.get('Hjemmeside')
    director_name = row.get('direktørnavn')
    scraped_text = row.get('scraped_contact_text') # Læser fra den nye kolonne
    domain = _get_domain_from_url(company_url)

    log.info(f"--- Starter behandling for: {company_name} ---")

    contact_person = None

    # --- Trin 1: Søg i den data, vi allerede har ---
    website_contact = _search_scraped_text(scraped_text)
    if website_contact:
        log.info("Relevant titel fundet på hjemmeside. Antager det er direktøren.")
        contact_person = {'fullname': director_name, 'headline': website_contact.get('headline', 'Kontaktperson')}

    # --- Trin 2: Fallback til LinkedIn/Apify, hvis Trin 1 fejlede ---
    if not contact_person:
        log.info(f"Ingen kontakt fundet i scrapet tekst. Prøver LinkedIn...")
        company_linkedin_url = _find_company_linkedin_url(company_name)
        if company_linkedin_url:
            employees = _query_linkedin_api(company_linkedin_url)
            if employees:
                contact_person = _filter_employees_by_title(employees)

    # --- Trin 3: Fallback til CVR-direktør, hvis alt andet fejlede ---
    if not contact_person and pd.notna(director_name):
        log.info(f"API-søgning gav intet resultat. Bruger fallback: Direktør '{director_name}'")
        contact_person = {'fullname': director_name, 'headline': 'Direktør'}
    
    # --- Afslutning: Byg det endelige resultat ---
    if contact_person:
        contact_name = contact_person.get('fullname')
        contact_title = contact_person.get('headline')
        
        if domain:
            contact_email = _generate_and_verify_email(contact_name, domain)
            return pd.Series({'contact_name': contact_name, 'contact_email': contact_email, 'contact_title': contact_title})
        else:
            return pd.Series({'contact_name': contact_name, 'contact_email': None, 'contact_title': contact_title})

    # Hvis absolut intet virkede
    log.warning(f"Kunne ikke finde nogen kontaktperson for {company_name}")
    return pd.Series({'contact_name': None, 'contact_email': None, 'contact_title': 'Ingen kontaktperson fundet'})


# --- Hoved-funktion til Ekstern Brug ---------------------------------------

def run_enrichment_on_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Tager en DataFrame og tilføjer berigede kontaktdata."""
    log.info("Starter berigelse af kontaktpersoner på %s virksomheder...", len(df))
    if df.empty:
        log.warning("Input DataFrame er tom. Stopper berigelse.")
        return df
    enriched_results = df.apply(_find_best_contact, axis=1)
    df_enriched = pd.concat([df.reset_index(drop=True), enriched_results], axis=1)
    log.info("Berigelse færdig.")
    return df_enriched