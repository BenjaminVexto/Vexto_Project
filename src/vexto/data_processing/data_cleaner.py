# data_cleaner.py
# ---------------
# Funktioner til at rense og transformere rå CVR-hits
# til et pænt, analyse-klar DataFrame-output.

from __future__ import annotations
import logging
import re
from typing import List, Dict, Any
import pandas as pd
import aiohttp
import asyncio
from aiohttp import ClientSession, ClientError, ClientTimeout

# Opsæt logging til både fil og konsol
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("website_liveness.log", encoding="utf-8"),
        logging.StreamHandler()
    ]
)
log = logging.getLogger(__name__)

# --- RegEx-konstanter -------------------------------------------------------
PHONE_DIGIT_REGEX = re.compile(r"^\d{8}$")
URL_PREFIX_REGEX = re.compile(r"^(https?:\/\/|www\.)", re.IGNORECASE)
EMAIL_REGEX = re.compile(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$')
DOMAIN_REGEX = re.compile(r'^[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$')

# --- Free email domains for website inference -------------------------------
FREE_EMAIL_DOMAINS = {
    'gmail.com', 'hotmail.com', 'outlook.com', 'yahoo.com',
    'live.dk', 'hotmail.dk', 'icloud.com', 'aol.com',
    'webspeed.dk', 'mail.tele.dk', 'eriksminde.dk', 'outlook.dk'
}

# --- Excluded suffixes for website inference (e.g., subdomains like *.mail.dk) ---
EXCLUDED_SUFFIXES = [re.compile(r'\.mail\.dk$')]

# --- Placeholder phrases to detect hosting placeholders ---------------------
PLACEHOLDER_PHRASES = [
    "this domain has been registered by one.com",
    "get your own domain name with one.com",
    r'<meta name="description" content="Domain is Parked"',
    r'<title>.*one\.com.*</title>'
]

# --- Separate phrases for "under konstruktion" ------------------------------
UNDER_CONSTRUCTION_PHRASES = [
    "under konstruktion"
]

def clean_domain(domain: str) -> str:
    """
    Erstatter danske specialtegn (æ, ø, å) med ae, oe, aa for bedre domæne-kompatibilitet.
    Fjerner også http(s):// og www. præfiks.
    """
    if not isinstance(domain, str):
        return ""
    domain = domain.strip().lower()
    domain = re.sub(r"^https?://", "", domain)
    domain = re.sub(r"^www\.", "", domain)
    domain = domain.rstrip("/")
    return (
        domain
        .replace("æ", "ae").replace("Æ", "Ae")
        .replace("ø", "oe").replace("Ø", "Oe")
        .replace("å", "aa").replace("Å", "Aa")
    )

async def _check_website_liveness(urls: List[str], timeout: int = 15, max_retries: int = 3) -> tuple[List[Any], List[str]]:
    """
    Checks if the provided URLs are live by sending asynchronous HTTP GET requests.
    Returns both liveness and remarks for nuanced evaluation.
    """
    results = ['N/A'] * len(urls)
    remarks = [''] * len(urls)
    
    headers = [
        {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'},
        {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
    ]
    
    # In-memory cache for liveness and remarks
    cache = {}
    
    async def test_both_www(domain: str, session: ClientSession, index: int):
        if not domain or not DOMAIN_REGEX.match(domain):
            log.info(f"[{index}] INVALID DOMAIN: '{domain}'")
            return None, "invalid domain"
        variants = [domain]
        if not domain.startswith("www."):
            variants.append("www." + domain)
        for dom in variants:
            for scheme in ["https://", "http://"]:
                url = f"{scheme}{dom}"
                result, remark = await check_single_url(url, dom, session, index)
                if result is not None:
                    return result, remark
        log.info(f"[{index}] {domain}: No valid response for any variant")
        return False, "no response"
    
    async def check_single_url(url: str, domain: str, session: ClientSession, index: int):
        cache_key = domain
        if cache_key in cache:
            log.debug(f"[{index}] Cache hit for {domain}: {cache[cache_key]}")
            return cache[cache_key]['live'], cache[cache_key]['remark']
        
        remark = ''
        for attempt in range(max_retries):
            try:
                async with session.get(url, timeout=ClientTimeout(total=timeout), allow_redirects=True, ssl=False, headers=headers[attempt % len(headers)]) as response:
                    final_url = str(response.url)
                    text_bytes = await response.read()
                    try:
                        text = text_bytes.decode('utf-8')
                    except UnicodeDecodeError:
                        try:
                            text = text_bytes.decode('iso-8859-1')
                        except UnicodeDecodeError:
                            text = text_bytes.decode('utf-8', errors='replace')
                    text_lower = text.lower()
                    is_placeholder = any(re.search(phrase, text_lower, re.IGNORECASE) for phrase in PLACEHOLDER_PHRASES)
                    is_under_construction = any(phrase in text_lower for phrase in UNDER_CONSTRUCTION_PHRASES)
                    if response.status in [200, 301, 302, 204, 455]:
                        if is_placeholder or ("one.com" in final_url.lower() and is_under_construction):
                            remark = "placeholder" if is_placeholder else "under konstruktion + one.com"
                            live = False
                        else:
                            remark = "under konstruktion" if is_under_construction else ""
                            if response.status == 455:
                                remark = f"custom status {response.status}" if not remark else f"{remark}, custom status {response.status}"
                            live = True
                        log.debug(f"[{index}] {url} (redirected to {final_url}): Status {response.status}, Remark: {remark}, Live: {live}, Content start: {text[:120]!r} (Attempt {attempt + 1})")
                        cache[cache_key] = {'live': live, 'remark': remark}
                        return live, remark
                    else:
                        log.debug(f"[{index}] {url} (redirected to {final_url}): Status {response.status} - False (Attempt {attempt + 1})")
                        remark = f"status {response.status}"
            except ClientError as e:
                if 'getaddrinfo failed' in str(e).lower():
                    remark = "dns failure"
                    live = False
                    log.debug(f"[{index}] {url}: DNS failure - False (Attempt {attempt + 1})")
                    cache[cache_key] = {'live': live, 'remark': remark}
                    await asyncio.sleep(1.0)  # Længere forsinkelse for DNS retry
                    if attempt == max_retries - 1:
                        return live, remark
                remark = "ssl error"
                log.debug(f"[{index}] {url}: ClientError {e} (Attempt {attempt + 1})")
            except asyncio.TimeoutError:
                remark = "timeout"
                log.debug(f"[{index}] {url}: Timeout - False (Attempt {attempt + 1})")
            except ValueError as e:
                remark = "encoding error"
                log.debug(f"[{index}] {url}: ValueError {e} (Attempt {attempt + 1})")
            await asyncio.sleep(0.2)  # Forsinkelse mod rate-limiting
        
        cache[cache_key] = {'live': False, 'remark': remark or "timeout/error"}
        return False, remark or "timeout/error"
    
    async with aiohttp.ClientSession() as session:
        tasks = [test_both_www(url, session, i) for i, url in enumerate(urls) if url != 'N/A' and url]
        task_results = await asyncio.gather(*tasks, return_exceptions=True)
        async_pos = 0
        for i, url in enumerate(urls):
            if url and url != "N/A":
                task_result = task_results[async_pos]
                async_pos += 1
                if isinstance(task_result, tuple):
                    results[i], remarks[i] = task_result
                else:
                    results[i] = False
                    remarks[i] = "exception"
            else:
                results[i] = "N/A"
                remarks[i] = "no domain"
    
    log.debug("Checked liveness for %s URLs, %s live", len(urls), sum(r is True for r in results if r != 'N/A'))
    return results, remarks

def _infer_website_from_email(df: pd.DataFrame) -> pd.DataFrame:
    """
    Infers a website URL from the Email column if Hjemmeside is 'N/A', excluding free email domains and suffixes.
    """
    def guess_website(row):
        if row['Hjemmeside'] != 'N/A' and row['Hjemmeside']:
            return row['Hjemmeside']
        
        if row['Email'] == 'N/A' or not row['Email']:
            return 'N/A'
        
        emails = row['Email'].split(';')
        for email in emails:
            email = email.strip()
            if not email or not EMAIL_REGEX.match(email):
                continue
            try:
                domain = email.split('@')[1].lower()
                if domain in FREE_EMAIL_DOMAINS:
                    continue
                if any(suffix.match(domain) for suffix in EXCLUDED_SUFFIXES):
                    continue
                cleaned_domain = clean_domain(domain)
                if not DOMAIN_REGEX.match(cleaned_domain):
                    continue
                return cleaned_domain
            except IndexError:
                continue
        return 'N/A'

    df['Hjemmeside'] = df.apply(guess_website, axis=1)
    log.debug("Inferred websites for %s rows", (df['Hjemmeside'] != 'N/A').sum())
    return df

def _parse_contact_value(raw_contact_list: Any) -> Dict[str, str]:
    """Splitter en liste af rå kontaktoplysninger i Telefon / Email / Hjemmeside."""
    if not isinstance(raw_contact_list, list):
        return {"Telefon": "N/A", "Email": "N/A", "Hjemmeside": "N/A"}

    phones, emails, websites = set(), set(), set()

    for item in raw_contact_list:
        val = ""
        if isinstance(item, dict) and "kontaktoplysning" in item:
            val = str(item["kontaktoplysning"]).strip()
        elif isinstance(item, str):
            val = item.strip()

        if not val:
            continue

        digits_only = re.sub(r"\D", "", val)

        if "@" in val:
            emails.add(val)
        elif PHONE_DIGIT_REGEX.fullmatch(digits_only):
            phones.add(val)
        elif "." in val:
            websites.add(val if URL_PREFIX_REGEX.match(val) else f"https://{val}")

    return {
        "Telefon": "; ".join(sorted(phones)) or "N/A",
        "Email": "; ".join(sorted(emails)) or "N/A",
        "Hjemmeside": "; ".join(sorted(websites)) or "N/A",
    }

def _get_all_directors(rel_list: Any) -> str:
    """Returnerer semikolon-separeret streng med direktør/indehaver-navne."""
    if not isinstance(rel_list, list):
        return "N/A"

    names = set()
    for rel in rel_list:
        if rel.get("deltager", {}).get("enhedstype") != "PERSON":
            continue
        for org in rel.get("organisationer", []) or []:
            role = (org.get("hovedtype") or "").upper()
            if role not in {
                "DIREKTION",
                "LEDELSESORGAN",
                "INDEHAVER",
                "FULDT_ANSVARLIG_DELTAGERE",
            }:
                continue
            for n in rel.get("deltager", {}).get("navne", []):
                if (name := n.get("navn")):
                    names.add(name)
    return "; ".join(sorted(names)) or "N/A"

def _best_contact(row: pd.Series) -> str:
    """Returnerer den mest værdifulde kontaktinfo i prioriteret rækkefølge."""
    return next(
        (row[col] for col in ("Telefon", "Email", "Hjemmeside") if row[col] != "N/A"),
        "N/A",
    )

def clean_and_prepare_cvr_data(raw_hits: List[Dict[str, Any]]) -> pd.DataFrame:
    df = pd.json_normalize(raw_hits, sep="_")

    # Branche
    df["Branchekode"] = df.get(
        "Vrvirksomhed_virksomhedMetadata_nyesteHovedbranche_branchekode", "N/A"
    ).fillna("N/A")
    df["Branchetekst"] = df.get(
        "Vrvirksomhed_virksomhedMetadata_nyesteHovedbranche_branchetekst", "N/A"
    ).fillna("N/A")

    # Kontaktinfo
    contact_df = (
        df.get("Vrvirksomhed_virksomhedMetadata_nyesteKontaktoplysninger", pd.Series(dtype=object))
        .apply(_parse_contact_value)
        .apply(pd.Series)
    )
    df[["Telefon", "Email", "Hjemmeside"]] = contact_df

    # Direktører
    df["Direktørnavn"] = df.get(
        "Vrvirksomhed_deltagerRelation", pd.Series(dtype=object)
    ).apply(_get_all_directors)

    # Filtrér status
    df = df[
        df["Vrvirksomhed_virksomhedMetadata_sammensatStatus"]
        .str.upper()
        .isin({"AKTIV", "NORMAL"})
    ].copy()

    # Infer websites from emails
    df = _infer_website_from_email(df)

    # Check website liveness with remarks
    loop = asyncio.get_event_loop()
    hjemmesider = df['Hjemmeside'].fillna('').apply(clean_domain).tolist()
    valid_idx = [i for i, url in enumerate(hjemmesider) if url and url != "N/A"]
    urls_to_check = [hjemmesider[i] if i in valid_idx else None for i in range(len(hjemmesider))]
    urls_for_async = [url for url in urls_to_check if url]
    liveness, remarks = loop.run_until_complete(_check_website_liveness(urls_for_async)) if urls_for_async else ([], [])
    liveness_full = ['N/A'] * len(hjemmesider)
    remarks_full = [''] * len(hjemmesider)
    async_pos = 0
    for i in valid_idx:
        liveness_full[i] = liveness[async_pos]
        remarks_full[i] = remarks[async_pos]
        async_pos += 1
    df['HjemmesideLive'] = liveness_full
    df['HjemmesideBemærkning'] = remarks_full

    # Bedste kontaktfelt
    if not df.empty:
        df["Kontaktoplysninger"] = df.apply(_best_contact, axis=1)
    else:
        df["Kontaktoplysninger"] = pd.Series(dtype=str)

    # Kolonnerækkefølge
    cols = [
        "Vrvirksomhed_cvrNummer",
        "Vrvirksomhed_virksomhedMetadata_nyesteNavn_navn",
        "Direktørnavn",
        "Kontaktoplysninger",
        "Telefon",
        "Email",
        "Hjemmeside",
        "HjemmesideLive",
        "HjemmesideBemærkning",
        "Vrvirksomhed_virksomhedMetadata_nyesteBeliggenhedsadresse_vejnavn",
        "Vrvirksomhed_virksomhedMetadata_nyesteBeliggenhedsadresse_husnummerFra",
        "Vrvirksomhed_virksomhedMetadata_nyesteBeliggenhedsadresse_postnummer",
        "Vrvirksomhed_virksomhedMetadata_nyesteBeliggenhedsadresse_postdistrikt",
        "Branchekode",
        "Branchetekst",
        "Vrvirksomhed_virksomhedMetadata_nyesteVirksomhedsform_virksomhedsformkode",
        "Vrvirksomhed_virksomhedMetadata_nyesteVirksomhedsform_langBeskrivelse",
        "Vrvirksomhed_virksomhedMetadata_sammensatStatus",
        "Vrvirksomhed_virksomhedMetadata_ophørsDato",
    ]

    # Sikr at alle kolonner findes
    for c in cols:
        if c not in df.columns:
            df[c] = "N/A"

    log.debug("Rensede DataFrame med %s aktive rækker", len(df))
    return df.reindex(columns=cols).fillna("N/A")