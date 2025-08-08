# src/vexto/scoring/contact_fetchers.py
import re
import logging
from bs4 import BeautifulSoup
from typing import List, Set, Tuple, Optional
from urllib.parse import urljoin
from .schemas import ConversionMetrics
from .http_client import AsyncHtmlClient

log = logging.getLogger(__name__)

EMAIL_PATTERN = re.compile(r'[\w\.-]+@[\w\.-]+\.\w+')
PHONE_PATTERN = re.compile(r'(?:\+?45)?\s?(?:[2-9]\d[ \.-]?){3}[2-9]\d')  # Matcher danske numre: 70101013, +4570101013, 70-10-10-13
CONTACT_LINK_HINTS = [
    "kontakt", "contact", "support", "kundeservice", "om-os", "about",
    "info", "service", "business", "erhverv", "b2b", "presse", "job"
]

def extract_emails_and_phones_from_text(text: str, exclude_phone: Optional[str] = None) -> Tuple[Set[str], Set[str]]:
    """Finder og renser e-mails og telefonnumre kun fra en tekststreng."""
    raw_email_matches = EMAIL_PATTERN.findall(text)
    cleaned_emails = set()
    for email in raw_email_matches:
        clean_email = email.strip(" .>;:,()[]'\"")
        if "@" in clean_email and "." in clean_email.split('@')[1]:
            cleaned_emails.add(clean_email.lower())
    
    raw_phone_matches = PHONE_PATTERN.findall(text)
    cleaned_phones = set()
    for match in raw_phone_matches:
        phone = re.sub(r'\D', '', match)
        # Valider: len=8 start 2-9 (ikke 20 eller 10), eller len=10 start '45' + 8 cifre start 2-9
        if (len(phone) == 8 and phone[0] in '23456789' and not phone.startswith('20') and not phone.startswith('10') and not phone.startswith('3')) or \
           (len(phone) == 10 and phone.startswith('45') and phone[2] in '23456789' and not phone[2:4] == '20'):
            cleaned_phones.add(phone)

    if exclude_phone and exclude_phone in cleaned_phones:
        cleaned_phones.remove(exclude_phone)

    return cleaned_emails, cleaned_phones

async def find_contact_info(
    soup: BeautifulSoup,
    base_url: str,
    deep_contact: bool = False,
    exclude_phone: Optional[str] = None
) -> ConversionMetrics:
    """
    Analyserer et BeautifulSoup-objekt og dets undersider for at finde alle e-mails og telefonnumre.
    """
    if not soup:
        return {'emails_found': [], 'phone_numbers_found': []}

    all_emails: Set[str] = set()
    all_phones: Set[str] = set()

    try:
        # --- Trin 1: Analyser hovedsiden (både tekst og mailto/tel/callto-links) ---
        main_text = soup.get_text(separator=' ')
        text_emails, text_phones = extract_emails_and_phones_from_text(main_text, exclude_phone=exclude_phone)
        all_emails.update(text_emails)
        all_phones.update(text_phones)

        for a in soup.select('a[href^="mailto:"], a[href^="tel:"], a[href^="callto:"]'):
            href = a.get("href", "")
            if href.startswith('mailto:'):
                match = EMAIL_PATTERN.search(href)
                if match:
                    all_emails.add(match.group(0).lower())
            elif href.startswith(('tel:', 'callto:')):
                # Normalize callto: to tel: and clean spaces/%20
                clean_href = re.sub(r'%20|\s+', '', href.replace('callto:', 'tel:'))
                phone = re.sub(r'^tel:', '', clean_href)
                # Valider som ovenfor
                if (len(phone) == 8 and phone[0] in '23456789' and not phone.startswith('20') and not phone.startswith('10') and not phone.startswith('3')) or \
                   (len(phone) == 10 and phone.startswith('45') and phone[2] in '23456789' and not phone[2:4] == '20'):
                    all_phones.add(phone)
                else:
                    log.debug(f"Invalid phone number format in link: {phone}")

        # --- Trin 2: Deep Contact - Analyser relevante undersider ---
        if deep_contact:
            client = AsyncHtmlClient()
            visited_urls: Set[str] = set()

            try:
                for a_tag in soup.find_all("a", href=True):
                    href = a_tag.get("href", "")
                    if not href or href.lower().startswith(("mailto:", "tel:", "javascript:", "callto:")):
                        continue

                    if any(hint in href.lower() for hint in CONTACT_LINK_HINTS):
                        full_url = urljoin(base_url, href)
                        if full_url in visited_urls:
                            continue
                        visited_urls.add(full_url)
                        
                        log.debug(f"Deep contact: Forsøger at hente {full_url}")
                        try:
                            sub_soup = await client.get_raw_html(full_url, return_soup=True)
                            if sub_soup:
                                # A. Find i almindelig tekst
                                sub_text = sub_soup.get_text(separator=' ')
                                sub_text_emails, sub_phones = extract_emails_and_phones_from_text(sub_text, exclude_phone=exclude_phone)
                                
                                # B. Find i mailto/tel/callto-links
                                mailto_emails = set()
                                sub_phones_from_links = set()
                                for sub_a in sub_soup.select('a[href^="mailto:"], a[href^="tel:"], a[href^="callto:"]'):
                                    sub_href = sub_a.get("href", "")
                                    if sub_href.startswith('mailto:'):
                                        match = EMAIL_PATTERN.search(sub_href)
                                        if match:
                                            mailto_emails.add(match.group(0).lower())
                                    elif sub_href.startswith(('tel:', 'callto:')):
                                        clean_href = re.sub(r'%20|\s+', '', sub_href.replace('callto:', 'tel:'))
                                        phone = re.sub(r'^tel:', '', clean_href)
                                        if (len(phone) == 8 and phone[0] in '23456789' and not phone.startswith('20') and not phone.startswith('10') and not phone.startswith('3')) or \
                                           (len(phone) == 10 and phone.startswith('45') and phone[2] in '23456789' and not phone[2:4] == '20'):
                                            sub_phones_from_links.add(phone)
                                        else:
                                            log.debug(f"Invalid phone number format in subpage link: {phone}")

                                # C. Læg det hele sammen
                                found_emails = sub_text_emails.union(mailto_emails)
                                all_emails.update(found_emails)
                                all_phones.update(sub_phones.union(sub_phones_from_links))

                                if found_emails or sub_phones.union(sub_phones_from_links):
                                     log.info(f"Fandt info på {full_url}: E-mails={found_emails}, Tlf={sub_phones.union(sub_phones_from_links)}")

                        except Exception as e:
                            log.warning(f"Fejl under hentning af deep_contact URL {full_url}: {e}")
            finally:
                await client.close()

        return {
            'emails_found': sorted(list(all_emails)),
            'phone_numbers_found': sorted(list(all_phones))
        }

    except Exception as e:
        log.error(f"Uventet fejl i find_contact_info: {e}", exc_info=True)
        return {'emails_found': [], 'phone_numbers_found': []}