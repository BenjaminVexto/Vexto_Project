# src/vexto/scoring/contact_fetchers.py

import re
import logging
from bs4 import BeautifulSoup
from typing import List, Set, Tuple, Optional
from urllib.parse import urljoin
from typing import Dict, Any, List

# src/vexto/scoring/contact_fetchers.py
from .schemas import ConversionMetrics
from .http_client import AsyncHtmlClient

log = logging.getLogger(__name__)

EMAIL_PATTERN = re.compile(r'[\w\.-]+@[\w\.-]+\.\w+')
PHONE_PATTERN = re.compile(r'(?:\+?45)?\s?(?:[2-9]\d[ \.-]?){3}[2-9]\d')  # Matcher DK-numre
CONTACT_LINK_HINTS = [
    "kontakt", "contact", "support", "kundeservice", "om-os", "about",
    "info", "service", "business", "erhverv", "b2b", "presse", "job"
]

try:
    extract_emails_and_phones_from_text  # type: ignore[name-defined]
except NameError:
    from typing import Set, Tuple, Optional

    def extract_emails_and_phones_from_text(
        text: str,
        exclude_phone: Optional[str] = None
    ) -> Tuple[Set[str], Set[str]]:
        raw_email_matches = EMAIL_PATTERN.findall(text or "")
        cleaned_emails: Set[str] = set()
        for email in raw_email_matches:
            clean_email = email.strip(" .>;:,()[]'\"")
            if "@" in clean_email and "." in clean_email.split('@')[1]:
                cleaned_emails.add(clean_email.lower())

        raw_phone_matches = PHONE_PATTERN.findall(text or "")
        cleaned_phones: Set[str] = set()
        for match in raw_phone_matches:
            phone = re.sub(r'\D', '', match)
            # DK-validering (som i din nuværende logik)
            if (len(phone) == 8 and phone[0] in '23456789' and not phone.startswith('20') and not phone.startswith('10') and not phone.startswith('3')) or \
               (len(phone) == 10 and phone.startswith('45') and phone[2] in '23456789' and not phone[2:4] == '20'):
                cleaned_phones.add(phone)

        if exclude_phone and exclude_phone in cleaned_phones:
            cleaned_phones.remove(exclude_phone)

        return cleaned_emails, cleaned_phones


CONTACT_SECTION_RX = re.compile(
    r"(contact|kontakt|kundeservice|support|kontaktinfo|contact-info|contactinfo|kontakt-os|kontaktoplys)",
    re.I,
)

def _collect_contact_section_text(soup: BeautifulSoup) -> str:
    """
    Returnerer samlet tekst fra tydelige kontaktsektioner:
    - <address>
    - elementer med id/class der matcher CONTACT_SECTION_RX
    """
    if not soup:
        return ""
    chunks: List[str] = []
    try:
        # <address> tags
        for addr in soup.find_all("address"):
            chunks.append(addr.get_text(" ", strip=True))

        # id/class match
        candidates = soup.select(
            '[id*="contact" i], [class*="contact" i], '
            '[id*="kontakt" i], [class*="kontakt" i], '
            '[id*="kundeservice" i], [class*="kundeservice" i]'
        )
        for el in candidates:
            # undgå at duplikere <address>-tekster hvis nested
            if el.name != "address":
                chunks.append(el.get_text(" ", strip=True))
    except Exception:
        pass
    return " \n ".join(filter(None, chunks))

async def find_contact_info(
    soup: BeautifulSoup,
    base_url: str,
    deep_contact: bool = False,
    exclude_phone: Optional[str] = None
) -> ConversionMetrics:
    """
    Finder kontaktdata med høj/lav sikkerhed.
    Høj sikkerhed: mailto:/tel:/callto: + tydelige kontaktsektioner.
    Lav sikkerhed: regex i vilkårlig brødtekst uden for kontaktsektioner.
    """
    if not soup:
        return {
            'emails_found': [],
            'phone_numbers_found': [],
            'emails_low_confidence': [],
            'phone_numbers_low_confidence': [],
        }

    high_emails: Set[str] = set()
    high_phones: Set[str] = set()
    low_emails: Set[str] = set()
    low_phones: Set[str] = set()

    try:
        # 1) Klikbare links (HØJ sikkerhed)
        for a in soup.select('a[href^="mailto:"], a[href^="tel:"], a[href^="callto:"]'):
            href = a.get("href", "") or ""
            if href.startswith('mailto:'):
                m = EMAIL_PATTERN.search(href)
                if m:
                    high_emails.add(m.group(0).lower())
            else:
                # tel:/callto:
                clean_href = re.sub(r'%20|\s+', '', href.replace('callto:', 'tel:'))
                phone = re.sub(r'^tel:', '', clean_href)
                if (len(phone) == 8 and phone[0] in '23456789' and not phone.startswith('20') and not phone.startswith('10') and not phone.startswith('3')) or \
                   (len(phone) == 10 and phone.startswith('45') and phone[2] in '23456789' and not phone[2:4] == '20'):
                    high_phones.add(phone)
                else:
                    log.debug(f"[contact] Ignorerer ugyldigt tel-link: {phone}")

        # 2) Tydelige kontaktsektioner (HØJ sikkerhed)
        section_text = _collect_contact_section_text(soup)
        if section_text:
            sec_emails, sec_phones = extract_emails_and_phones_from_text(section_text, exclude_phone=exclude_phone)
            high_emails.update(sec_emails)
            high_phones.update(sec_phones)

        # 3) Brødtekst uden for kontaktsektioner (LAV sikkerhed)
        main_text = soup.get_text(separator=' ') if soup else ""
        if section_text:
            # Fjern kontaktsektionstekst groft, så vi ikke dobbelt-tæller
            try:
                main_text = main_text.replace(section_text, " ")
            except Exception:
                pass
        text_emails, text_phones = extract_emails_and_phones_from_text(main_text, exclude_phone=exclude_phone)
        # Alt der ikke allerede er i HIGH → LOW
        low_emails.update(text_emails - high_emails)
        low_phones.update(text_phones - high_phones)

        # 4) Deep contact (kontakt-/support-undersider) = HØJ sikkerhed
        if deep_contact:
            client = AsyncHtmlClient()
            visited: Set[str] = set()
            try:
                for a_tag in soup.find_all("a", href=True):
                    href = a_tag.get("href", "")
                    if not href or href.lower().startswith(("mailto:", "tel:", "javascript:", "callto:")):
                        continue
                    if any(h in href.lower() for h in CONTACT_LINK_HINTS):
                        full_url = urljoin(base_url, href)
                        if full_url in visited:
                            continue
                        visited.add(full_url)
                        log.debug(f"[contact] Deep contact fetch: {full_url}")
                        try:
                            sub_soup = await client.get_raw_html(full_url, return_soup=True)
                            if sub_soup:
                                # Hele undersiden behandles som HØJ sikkerhed (kontaktkontekst)
                                sub_text = sub_soup.get_text(separator=' ')
                                sub_emails, sub_phones = extract_emails_and_phones_from_text(sub_text, exclude_phone=exclude_phone)
                                # Mailto/tel i undersiden
                                for sub_a in sub_soup.select('a[href^="mailto:"], a[href^="tel:"], a[href^="callto:"]'):
                                    sub_href = sub_a.get("href", "")
                                    if sub_href.startswith('mailto:'):
                                        m = EMAIL_PATTERN.search(sub_href)
                                        if m:
                                            sub_emails.add(m.group(0).lower())
                                    else:
                                        clean_href = re.sub(r'%20|\s+', '', sub_href.replace('callto:', 'tel:'))
                                        phone = re.sub(r'^tel:', '', clean_href)
                                        if (len(phone) == 8 and phone[0] in '23456789' and not phone.startswith('20') and not phone.startswith('10') and not phone.startswith('3')) or \
                                           (len(phone) == 10 and phone.startswith('45') and phone[2] in '23456789' and not phone[2:4] == '20'):
                                            sub_phones.add(phone)
                                high_emails.update(sub_emails)
                                high_phones.update(sub_phones)
                        except Exception as e:
                            log.warning(f"[contact] Fejl i deep_contact {full_url}: {e}")
            finally:
                await client.close()

        # Fjern evt. nummer der eksplicit skal udelukkes
        if exclude_phone:
            high_phones.discard(exclude_phone)
            low_phones.discard(exclude_phone)

        return {
            'emails_found': sorted(high_emails),
            'phone_numbers_found': sorted(high_phones),
            'emails_low_confidence': sorted(low_emails - high_emails),
            'phone_numbers_low_confidence': sorted(low_phones - high_phones),
        }

    except Exception as e:
        log.error(f"Uventet fejl i find_contact_info: {e}", exc_info=True)
        return {
            'emails_found': [],
            'phone_numbers_found': [],
            'emails_low_confidence': [],
            'phone_numbers_low_confidence': [],
        }

def detect_forms(html: str) -> Dict[str, Any]:
    """Konsistent formular-udtræk:
    - Ignorerer standard-nyhedsbrevsskjemaer (footer / 'newsletter' / 'subscribe')
    - Returnerer både detaljeret liste og aggregerede tællinger til scorer/UI
    Output:
    {
      "form_count": int,
      "forms": [ ... { + is_newsletter: bool } ... ],
      "form_field_counts": List[int],        # tællinger *uden* newsletter-forms (fallback: alle)
      "newsletter_forms_ignored": int
    }
    """
    try:
        soup = BeautifulSoup(html or "", "html.parser")
        forms: List[Dict[str, Any]] = []
        field_counts_all: List[int] = []
        field_counts_effective: List[int] = []

        def _is_newsletter(form) -> bool:
            # Heuristik: placeret i <footer> ELLER tydelige ord i id/class/tekst
            in_footer = bool(form.find_parent(["footer"]))
            label = " ".join([
                (form.get("id") or ""),
                (form.get("name") or ""),
                " ".join(form.get("class") or []),
                (form.get_text(" ") or "")
            ]).lower()
            return in_footer or any(k in label for k in ("newsletter", "nyhedsbrev", "subscribe", "tilmeld nyhedsbrev"))

        for f in soup.find_all("form"):
            inputs = f.find_all(["input", "select", "textarea"])
            req = [el for el in inputs if el.has_attr("required") or "required" in (el.get("aria-required", "").lower())]

            names = " ".join([(el.get("name") or el.get("id") or "").lower() for el in inputs])
            placeholders = " ".join([(el.get("placeholder") or "").lower() for el in inputs])
            txt = (names + " " + placeholders)

            has_email = bool(re.search(r"\b(email|e-mail|mail)\b", txt))
            has_phone = bool(re.search(r"\b(phone|telefon|tlf)\b", txt))
            has_textarea = f.find("textarea") is not None

            # simple honeypot: hidden fields med 'website/url/homepage' eller 'fax'
            has_honeypot = any(
                (el.get("type", "").lower() == "hidden") and
                re.search(r"(website|homepage|url|fax)", (el.get("name") or el.get("id") or "").lower())
                for el in inputs
            )

            is_newsletter = _is_newsletter(f)

            forms.append({
                "action": f.get("action"),
                "method": (f.get("method") or "").lower() or None,
                "input_count": len(inputs),
                "required_pct": round((len(req) / max(1, len(inputs))) * 100.0, 1),
                "has_email": has_email,
                "has_phone": has_phone,
                "has_textarea": has_textarea,
                "has_honeypot": has_honeypot,
                "is_newsletter": is_newsletter,
            })

            field_counts_all.append(len(inputs))
            if not is_newsletter:
                field_counts_effective.append(len(inputs))

        return {
            "form_count": len(forms),
            "forms": forms,
            # konsistente tællinger (bruges af config: form_usability)
            "form_field_counts": field_counts_effective or field_counts_all,
            "newsletter_forms_ignored": sum(1 for f in forms if f.get("is_newsletter")),
        }
    except Exception as e:
        log.error("Fejl i detect_forms: %s", e, exc_info=True)
        return {"form_count": 0, "forms": [], "form_field_counts": [], "newsletter_forms_ignored": 0}