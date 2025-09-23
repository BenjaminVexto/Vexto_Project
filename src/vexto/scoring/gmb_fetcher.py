# gmb_fetcher.py
import os
import re
import json
import logging
import unicodedata
from difflib import SequenceMatcher
from urllib.parse import urlparse, quote_plus
from typing import Optional, Tuple, List
from .http_client import AsyncHtmlClient
from .schemas import SocialAndReputationMetrics

# --- Robust imports (abs -> rel -> local) for Pylance friendliness ---
try:
    from src.vexto.http_client import AsyncHtmlClient  # type: ignore[reportMissingImports]
except Exception:  # pragma: no cover
    try:
        from .http_client import AsyncHtmlClient  # type: ignore
    except Exception:  # pragma: no cover
        from http_client import AsyncHtmlClient  # type: ignore

try:
    from src.vexto.schemas import SocialAndReputationMetrics  # type: ignore[reportMissingImports]
except Exception:  # pragma: no cover
    try:
        from .schemas import SocialAndReputationMetrics  # type: ignore
    except Exception:  # pragma: no cover
        from schemas import SocialAndReputationMetrics  # type: ignore
# --------------------------------------------------------------------

log = logging.getLogger(__name__)

# =============================
# Locale & geobias helpers
# =============================

def _resolve_places_locale(url: str) -> dict:
    host = (urlparse(url).hostname or "").lower()
    tld = host.rsplit(".", 1)[-1] if "." in host else ""
    mapping = {
        "dk": ("da", "dk"), "se": ("sv", "se"), "no": ("nb", "no"), "fi": ("fi", "fi"),
        "de": ("de", "de"), "nl": ("nl", "nl"), "fr": ("fr", "fr"), "es": ("es", "es"),
        "it": ("it", "it"), "pl": ("pl", "pl"), "pt": ("pt", "pt"), "cz": ("cs", "cz"),
        "sk": ("sk", "sk"), "hu": ("hu", "hu"), "ro": ("ro", "ro"), "bg": ("bg", "bg"),
        "gr": ("el", "gr"), "lt": ("lt", "lt"), "lv": ("lv", "lv"), "ee": ("et", "ee"),
        "ie": ("en", "ie"), "uk": ("en", "gb"), "co": ("es", "co"), "mx": ("es", "mx"),
        "ar": ("es", "ar"), "br": ("pt", "br"), "au": ("en", "au"), "nz": ("en", "nz"),
        "ca": ("en", "ca"), "us": ("en", "us"),
    }
    if tld in mapping:
        lang, reg = mapping[tld]
        return {"language": lang, "region": reg}
    return {"language": None, "region": None}


def _locationbias_from_url(url: str) -> str:
    host = (urlparse(url).hostname or "").lower()
    tld = host.rsplit(".", 1)[-1] if "." in host else ""
    centers = {
        "dk": (56.0, 10.0, 250000), "se": (60.0, 15.0, 400000), "no": (60.5, 8.5, 350000),
        "fi": (64.0, 26.0, 450000), "de": (51.0, 10.0, 400000), "nl": (52.2, 5.3, 250000),
        "fr": (46.5, 2.2, 500000), "uk": (53.5, -2.0, 350000), "ie": (53.4, -8.0, 250000),
        "us": (39.8, -98.6, 1500000), "ca": (56.1, -106.3, 1500000), "au": (-25.3, 133.8, 1500000),
        "nz": (-41.0, 174.0, 400000), "br": (-15.6, -47.9, 1500000), "mx": (23.6, -102.5, 1000000),
        "es": (40.2, -3.7, 500000), "it": (42.5, 12.5, 500000), "pl": (52.1, 19.2, 400000),
        "pt": (39.6, -8.0, 350000),
    }
    if tld in centers:
        lat, lng, radius = centers[tld]
        return f"circle:{radius}@{lat},{lng}"
    return "ipbias"


def _center_from_url(url: str):
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


# =============================
# Query helpers
# =============================

def _nfc(s: Optional[str]) -> str:
    return unicodedata.normalize("NFC", s) if isinstance(s, str) else ""


def _guess_business_name_from_url(url: str) -> str:
    hostname = urlparse(url).hostname
    if not hostname:
        return ""
    name = hostname.replace("www.", "").split(".")[0]
    return name.replace("-", " ").title()


def _clean_company_name(name: str) -> str:
    if not isinstance(name, str):
        return ""
    suffixes = [r"\bA/S\b", r"\bApS\b", r"\bI/S\b", r"\bK/S\b", r"\bIVS\b"]
    cleaned = name
    for s in suffixes:
        cleaned = re.sub(s, "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"[^0-9A-Za-zæøåÆØÅ ]+", "", cleaned)
    return cleaned.strip()


def _build_search_queries(url: str, company_name: Optional[str]) -> List[str]:
    """
    Byg søgeliste uden web-search operators (quotes/site: virker ikke i Places).
    Brug brand + branchekontekst + domæne-varianter for at undgå autocorrect.
    Rækkefølge = fra mest præcise til bredere.
    """
    queries: List[str] = []
    host = (urlparse(url).hostname or "").lower()
    base = (host.replace("www.", "").split(".")[0] if host else "").strip()

    if company_name:
        queries.append(_nfc(company_name))

    if base:
        queries.extend([
            base,
            f"{base} marketing",
            f"{base} digital marketing",
            f"{base} bureau",
            f"{base} aps",
        ])

    if host:
        queries.extend([host, f"https://{host}/"])

    # dedup – bevar rækkefølgen
    seen = set()
    return [q for q in queries if q and not (q in seen or seen.add(q))]


# =============================
# Result helpers
# =============================

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
        return ".".join(parts[-2:])
    return host


def _domain_match(website_url: str, target_url: str) -> bool:
    try:
        wanted = (urlparse(target_url).hostname or "").lower()
        found = (urlparse((website_url or "").strip()).hostname or "").lower()
        if not wanted or not found:
            return False
        return _base_domain(wanted) == _base_domain(found)
    except Exception:
        return False


def _build_details_url(place_id: str, api_key: str, language: Optional[str]) -> str:
    fields = (
        "rating,user_ratings_total,name,formatted_address,opening_hours,website,photos,"
        "geometry,types,international_phone_number,formatted_phone_number"
    )
    lang_param = f"&language={language}" if language else ""
    return (
        "https://maps.googleapis.com/maps/api/place/details/json"
        f"?place_id={place_id}&fields={fields}{lang_param}&key={api_key}"
    )


def _norm_company_name(s: str) -> str:
    if not s:
        return ""
    s = _nfc(s).strip().lower()
    s = re.sub(r"\b(aps|a\/s|ivs)\b", "", s, flags=re.I)
    s = re.sub(r"[^a-z0-9æøåéèüö ]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _name_similarity(a: str, b: str) -> float:
    a_n = _norm_company_name(a)
    b_n = _norm_company_name(b)
    if not a_n or not b_n:
        return 0.0
    return SequenceMatcher(None, a_n, b_n).ratio()


def _target_company_from(url: str, company_name: Optional[str]) -> str:
    if company_name:
        return company_name
    host = (urlparse(url).hostname or "").lower()
    base = host.split(".", 1)[0] if "." in host else host
    return re.sub(r"[^a-z0-9æøå]", " ", base).strip()


# --------- Generic confusable detector (replaces hardcoded pairs) ---------

_COMMON_QUALIFIERS = {
    "aps","a/s","ivs","k/s","i/s","group","holding","ltd","gmbh",
    "bureau","agency","studio","media","digital","marketing","consult",
    "consulting","solutions","partners","company","co","inc","ab","oy"
}

def _brand_tokens(name: str) -> List[str]:
    toks = re.findall(r"[0-9a-zæøåéèüö]+", (name or "").lower())
    return [t for t in toks if t not in _COMMON_QUALIFIERS]

def _contains_as_token(haystack: str, needle: str) -> bool:
    h = set(_brand_tokens(haystack))
    n = _brand_tokens(needle)
    return bool(n) and n[0] in h

def _damerau_levenshtein(a: str, b: str, max_dist: int = 2) -> int:
    if a == b:
        return 0
    la, lb = len(a), len(b)
    if abs(la - lb) > max_dist:
        return max_dist + 1
    if la > lb:
        a, b, la, lb = b, a, lb, la
    prev = list(range(lb + 1))
    cur = [0] * (lb + 1)
    for i in range(1, la + 1):
        cur[0] = i
        min_row = cur[0]
        ai = a[i - 1]
        for j in range(1, lb + 1):
            c = 0 if ai == b[j - 1] else 1
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + c)
            if i > 1 and j > 1 and a[i - 1] == b[j - 2] and a[i - 2] == b[j - 1]:
                cur[j] = min(cur[j], prev[j - 2] + 1)
            if cur[j] < min_row:
                min_row = cur[j]
        if min_row > max_dist:
            return max_dist + 1
        prev, cur = cur, prev
    return prev[lb]

def _looks_confusable(target: str, candidate: str) -> bool:
    # Token inclusion => not confusable (e.g., "acme" vs "acme marketing")
    if _contains_as_token(candidate, target):
        return False
    t_core = "".join(_brand_tokens(target))
    c_core = "".join(_brand_tokens(candidate))
    if not t_core or not c_core:
        return False
    n = min(len(t_core), len(c_core))
    thr = 1 if n <= 7 else 2
    dist = _damerau_levenshtein(t_core, c_core, max_dist=thr)
    return dist <= thr


def _norm_phone_for_google(phone: Optional[str]) -> Optional[str]:
    """Normaliser til format Google bedst forstår – gerne med landekode."""
    if not phone:
        return None
    p = re.sub(r"[^\d+]", "", phone)
    # Hvis 8 cifre (typisk DK), præfiks +45
    if p and not p.startswith("+") and len(re.sub(r"\D", "", p)) == 8:
        return "+45" + re.sub(r"\D", "", p)
    # Hvis 10 og starter med 45 -> +45xxxxxx
    if len(p) == 10 and p.startswith("45"):
        return "+" + p
    return p if p.startswith("+") else f"+{p}"

def _digits_only(s: Optional[str]) -> str:
    return re.sub(r"\D", "", s or "")

def _normalize_for_compare(p: Optional[str]) -> str:
    """
    Sammenlign lokale/intl. varianter robust:
    - normaliser til E.164 (+45 for DK hvor muligt via _norm_phone_for_google)
    - fjern ikke-cifre
    - hvis '45' + 8 lokale cifre -> brug sidste 8 til robust sammenligning
    """
    d = _digits_only(_norm_phone_for_google(p))
    if len(d) == 10 and d.startswith("45"):
        return d[-8:]
    return d

def _phones_equal(a: Optional[str], b: Optional[str]) -> bool:
    return _normalize_for_compare(a) == _normalize_for_compare(b)


# =============================
# Phone autodiscovery (valgfri)
# =============================

_PHONE_SELECTORS = (r"kontakt", r"contact", r"kontakt-os", r"om-os")
# E.164 maks 15 cifre – strammere match for at undgå meget lange tokens fra scripts/IDs.
_PHONE_RE = re.compile(r"(?<!\w)(\+?\d[\d\s().-]{6,18}\d)(?!\w)")
# Fang og prioriter tel:-links (typisk mest pålidelige).
_TEL_HREF_RE = re.compile(r'href=["\']tel:([+\d][\d\s().-]{6,18}\d)["\']', re.IGNORECASE)
# Labels vi leder efter i nærheden af telefonnumre
_LABEL_WORDS = ("telefon", "tlf", "phone", "ring", "kontakt", "sales")
_LABEL_NEAR_RE = re.compile(r"(?i)(telefon|tlf|phone|ring|kontakt|sales)")
# Simple kendt "fake"
_FAKE_NUMBERS = {"+1234567890"}

def _extract_jsonld_phones(html: str) -> List[str]:
    phones: List[str] = []
    try:
        for m in re.finditer(r'<script[^>]+type=["\']application/ld\+json["\'][^>]*>(.*?)</script>', html or "", re.I | re.S):
            raw = m.group(1)
            # Nogle sites har flere JSON-objekter i samme <script>; prøv begge formater
            candidates = []
            try:
                candidates = [json.loads(raw)]
            except Exception:
                # Prøv at finde {…} blokke inde i scriptet
                for jm in re.finditer(r"\{.*?\}", raw, re.S):
                    try:
                        candidates.append(json.loads(jm.group(0)))
                    except Exception:
                        pass
            def walk(obj):
                if isinstance(obj, dict):
                    for k, v in obj.items():
                        if k.lower() == "telephone" and isinstance(v, str):
                            phones.append(v)
                        elif k.lower() == "telephone" and isinstance(v, list):
                            phones.extend([x for x in v if isinstance(x, str)])
                        else:
                            walk(v)
                elif isinstance(obj, list):
                    for it in obj:
                        walk(it)
            for cand in candidates:
                walk(cand)
    except Exception:
        pass
    # normaliser + dedup
    out: List[str] = []
    seen = set()
    for p in phones:
        norm = _norm_phone_for_google(p)
        if not norm:
            continue
        digits = re.sub(r"\D", "", norm)
        if not (8 <= len(digits) <= 15):
            continue
        if norm in _FAKE_NUMBERS:
            continue
        if norm not in seen:
            seen.add(norm)
            out.append(norm)
    return out


def _phones_with_html_evidence(html: str, candidates: List[str]) -> dict:
    """
    Returnér et dict phone->score baseret på evidens i HTML:
    - tel:-links (stærk)
    - label-nærhed (Telefon/Tlf/Phone/…)
    """
    html = html or ""
    score: dict[str, float] = {p: 0.0 for p in candidates}
    # tel: links
    tel_hits = set(_norm_phone_for_google(p) for p in _TEL_HREF_RE.findall(html))
    for p in tel_hits:
        if p in score:
            score[p] += 50.0
    # label-nærhed: kig i nærheden (± 40 tegn) hvor nummer forekommer i rå HTML
    for p in candidates:
        if not p:
            continue
        # søg efter forskellige varianter (med/uden mellemrum)
        pattern = re.escape(p).replace(r"\+", r"\+").replace(r"\ ", r"[ \s]*")
        for m in re.finditer(pattern, html):
            start = max(0, m.start() - 40)
            end = min(len(html), m.end() + 5)
            snippet = html[start:end]
            if _LABEL_NEAR_RE.search(snippet):
                score[p] += 20.0
    return score

def _extract_candidate_phones(text: str) -> List[str]:
    text = text or ""
    tel_hits = [m.strip() for m in _TEL_HREF_RE.findall(text)]
    txt_hits = [m.strip() for m in _PHONE_RE.findall(text)]

    def to_e164(p: str) -> Optional[str]:
        norm = _norm_phone_for_google(p)
        if not norm:
            return None
        digits = re.sub(r"\D", "", norm)
        # E.164: max 15 cifre (uden +). Min 8 som simpel heuristik.
        if not (8 <= len(digits) <= 15):
            return None
        if norm in _FAKE_NUMBERS:
            return None
        # DK-plausibilitet: +45 og første lokale ciffer må ikke være 0/1
        if norm.startswith("+45"):
            last8 = re.sub(r"\D", "", norm)[-8:]
            if not last8 or last8[0] in ("0", "1"):
                return None
        return norm

    cleaned: List[str] = []
    # 1) tel: links først
    for raw in tel_hits:
        e164 = to_e164(raw)
        if e164:
            cleaned.append(e164)
    # 2) synlig tekst som fallback
    for raw in txt_hits:
        e164 = to_e164(raw)
        if e164:
            cleaned.append(e164)

    # dedup + prioritering: +45 først; derefter længde tæt på 10 (45 + 8 cifre)
    seen = set()
    deduped = [p for p in cleaned if not (p in seen or seen.add(p))]
    deduped.sort(key=lambda p: (0 if p.startswith("+45") else 1,
                                abs(len(re.sub(r"\D", "", p)) - 10)))
    return deduped

async def _discover_site_phone_numbers(client: AsyncHtmlClient, url: str) -> List[str]:
    """
    Hent / og sandsynlige kontakt-URLs, udtræk telefoner fra:
      - JSON-LD (telephone)
      - tel:-links
      - synlig tekst
    og score dem på tværs af sider (kontakt-side bonus, tel-link/label bonus).
    """
    host = (urlparse(url).scheme + "://" + (urlparse(url).hostname or "")).rstrip("/")
    pages = [host] + [f"{host}/{slug.strip('/')}/" for slug in _PHONE_SELECTORS]

    # aggreger
    total_score: dict[str, float] = {}
    seen_on_pages: dict[str, int] = {}
    dk_only = (urlparse(url).hostname or "").lower().endswith(".dk")

    for u in pages:
        try:
            r = await client.httpx_get(u)
            html = r.text or ""

            # 1) JSON-LD
            jsonld_phones = _extract_jsonld_phones(html)

            # 2) Synlig tekst/tel: (genbruger eksisterende extractor til at normalisere)
            text_phones = _extract_candidate_phones(html)

            # union af kandidater på siden
            page_candidates = list({*jsonld_phones, *text_phones})

            # evidens-score i denne HTML
            evid = _phones_with_html_evidence(html, page_candidates)

            # bonus: hvis URL ligner kontakt-side
            is_contact = any(slug in u.lower() for slug in ("kontakt", "contact"))
            for p in page_candidates:
                if dk_only and not (p or "").startswith("+45"):
                    # Hvis .dk-domæne → foretræk danske numre
                    continue
                base = 0.0
                if p in jsonld_phones:
                    base += 100.0  # stærk kilde
                base += evid.get(p, 0.0)
                if is_contact:
                    base += 15.0
                # fordeling på tværs af sider
                total_score[p] = total_score.get(p, 0.0) + base
                seen_on_pages[p] = seen_on_pages.get(p, 0) + 1

            # Kun vis DK i preview for mindre støj
            preview_dk = [p for p in page_candidates if p.startswith("+45")]
            preview = preview_dk[:10] + (["…"] if len(preview_dk) > 10 else [])
            if preview:
                log.info(f"[GMB/PHONE/AUTO] {u} -> {preview}")
        except Exception as e:
            log.debug(f"[GMB/PHONE/AUTO] fejl på {u}: {e}")

    if not total_score:
        return []

    # Endelig rangering:
    #  - høj score først
    #  - flere sider først
    #  - +45 først
    #  - længde tæt på 10 (45 + 8 cifre)
    def rank_key(p: str):
        digits_len = len(re.sub(r"\D", "", p))
        return (-total_score.get(p, 0.0),
                -seen_on_pages.get(p, 0),
                0 if p.startswith("+45") else 1,
                abs(digits_len - 10))

    ranked = sorted(total_score.keys(), key=rank_key)

    # Debug: vis top-10 med score og antal sider
    debug_top = sorted(total_score.keys(), key=lambda p: (-total_score[p], -seen_on_pages[p]))[:10]
    dbg = [f"{p} (score={total_score[p]:.1f}, pages={seen_on_pages[p]})" for p in debug_top]
    if dbg:
        log.info("[GMB/PHONE/RANKED] %s%s", dbg[:10], " …" if len(debug_top) > 10 else "")

    return ranked


# =============================
# Core HTTP helpers
# =============================

async def _find_place_candidates(client: AsyncHtmlClient, api_key: str, query: str, language: Optional[str], locationbias: str):
    url = (
        "https://maps.googleapis.com/maps/api/place/findplacefromtext/json"
        f"?input={quote_plus(query)}&inputtype=textquery"
        f"&fields=place_id,name,formatted_address"
        f"{f'&language={language}' if language else ''}"
        f"&locationbias={locationbias}"
        f"&key={api_key}"
    )
    r = await client.httpx_get(url)
    j = r.json() or {}
    return (j.get("candidates") or []), (j.get("status") or "UNKNOWN")


async def _get_place_details(client: AsyncHtmlClient, api_key: str, place_id: str, language: Optional[str]):
    r = await client.httpx_get(_build_details_url(place_id, api_key, language))
    return r.json() or {}


async def _find_by_phone(client: AsyncHtmlClient, api_key: str, phone: str):
    url = (
        "https://maps.googleapis.com/maps/api/place/findplacefromtext/json"
        f"?input={quote_plus(phone)}&inputtype=phonenumber"
        f"&fields=place_id,name,formatted_address"
        f"&key={api_key}"
    )
    r = await client.httpx_get(url)
    return r.json() or {}


# =============================
# Main fetcher
# =============================

async def fetch_gmb_data(
    client: AsyncHtmlClient,
    url: str,
    cvr: Optional[str] = None,
    company_name: Optional[str] = None,
    phone: Optional[str] = None,
) -> SocialAndReputationMetrics:
    """
    Strategi:
      0) ANKER: log FØR
      1) Telefon-søgning FØRST (inkl. autodetektion hvis phone ikke givet)
      2) Find Place på en række kontekstuelle queries
      3) Fallback: Text Search nær TLD-center
      4) HÅRDE GATES: domæne-match ELLER (navn >=0.90 og ikke 'confusable')
      5) ANKER: log EFTER med beslutning
    """
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        log.warning("GOOGLE_API_KEY mangler – springer GMB-tjek over.")
        return _empty_metrics(status="unknown")

    loc = _resolve_places_locale(url)
    language = loc["language"]
    locationbias = _locationbias_from_url(url)

    # ---- ANKER: BEFORE ----
    log.info(
        "===== [GMB/START] domain=%s cvr=%s company=%s phone=%s =====",
        (urlparse(url).hostname or ""), cvr or "N/A", company_name or "N/A", phone or "N/A"
    )

    # -------- Telefon først --------
    norm_phone = _norm_phone_for_google(phone)
    phones_tried: List[str] = []
    if not norm_phone:
        try:
            phones_tried = await _discover_site_phone_numbers(client, url)
            if phones_tried:
                log.info(
                    "[GMB/PHONE/DISCOVERED] %s%s",
                    phones_tried[:10], " …" if len(phones_tried) > 10 else ""
                )
        except Exception as e:
            log.debug(f"[GMB/PHONE/DISCOVER] fejl: {e}")
    else:
        phones_tried = [norm_phone]

    if phones_tried:
        log.info(
            "[GMB/PHONE/CANDIDATES] top=%s%s",
            phones_tried[:5], " …" if len(phones_tried) > 5 else ""
        )

    # --- PATCH: samle site-telefoner til senere phone-match ---
    site_phones_set = set()
    if norm_phone:
        site_phones_set.add(norm_phone)
    if phones_tried:
        site_phones_set.update(phones_tried)

    # Prøv op til 5 bedste kandidater i prioriteret rækkefølge
    for idx, ph in enumerate(phones_tried[:5], 1):
        log.info("[GMB/PHONE] (%d/%d) søger place_id for phone='%s'",
                 idx, min(5, len(phones_tried)), ph)
        try:
            jph = await _find_by_phone(client, api_key, ph)
            # --- PATCH: tydelig status/candidates log ---
            ph_status = str((jph or {}).get("status"))
            ph_candidates = int(len((jph or {}).get("candidates") or []))
            log.info("[GMB/PHONE] status=%s candidates=%d for %s", ph_status, ph_candidates, ph)

            if ph_status == "OK":
                for cand in (jph.get("candidates") or []):
                    pid = (cand or {}).get("place_id")
                    if not pid:
                        continue
                    j2 = await _get_place_details(client, api_key, pid, language=None)
                    if (j2 or {}).get("status") != "OK":
                        continue
                    result = (j2 or {}).get("result") or {}
                    website = (result.get("website") or "").strip().lower()
                    g_place_phone_int = result.get("international_phone_number") or ""
                    g_place_phone_fmt = result.get("formatted_phone_number") or ""

                    target_name = _target_company_from(url, company_name)
                    name_sim = _name_similarity(result.get("name") or "", target_name)
                    name_ok = (name_sim >= 0.90) and not _looks_confusable(
                        target_name, result.get("name") or ""
                    )
                    website_ok = _domain_match(website, url)

                    # match begge Google-felter mod alle kendte site-telefoner
                    phone_ok = any(
                        _phones_equal(gp, sp)
                        for gp in (g_place_phone_int, g_place_phone_fmt)
                        for sp in site_phones_set
                    )

                    decision = "ACCEPT" if (website_ok or name_ok or phone_ok) else "REJECT"
                    log.info(
                        "[GMB/ANCHOR/PHONE] decision=%s name='%s' sim=%.2f website_ok=%s phone_ok=%s "
                        "place_phone_int='%s' place_phone_fmt='%s' website='%s' place_id=%s",
                        decision, result.get("name"), name_sim, website_ok, phone_ok,
                        g_place_phone_int, g_place_phone_fmt, website, pid
                    )

                    if decision == "ACCEPT":
                        opening_hours = result.get("opening_hours") or {}
                        photos = result.get("photos") or []
                        count = result.get("user_ratings_total") or 0
                        rating = result.get("rating")
                        has_hours = bool(opening_hours.get("weekday_text") or opening_hours.get("periods"))
                        metrics: SocialAndReputationMetrics = {
                            "gmb_review_count": int(count or 0),
                            "gmb_average_rating": float(rating) if rating is not None else None,
                            "gmb_profile_complete": True,
                            "gmb_has_website": bool(website),
                            "gmb_has_hours": has_hours,
                            "gmb_photo_count": int(len(photos)),
                            "gmb_business_name": result.get("name"),
                            "gmb_address": result.get("formatted_address"),
                            "gmb_place_id": pid,
                            "gmb_status": "ok",
                        }
                        log.info("===== [GMB/RESULT] status=ok via PHONE place_id=%s name='%s' =====",
                                 pid, result.get("name"))
                        return metrics
        except Exception as e:
            log.error(f"[GMB/PHONE] fejl: {e}")

    # -------- Find Place på queries --------
    queries = _build_search_queries(url, company_name)
    if cvr:
        queries.insert(0, f"CVR {cvr}")

    if not queries:
        log.warning("Ingen gyldige søgetermer til GMB.")
        return _empty_metrics(status="invalid_input")

    for i, name in enumerate(queries, 1):
        log.info("🔎 [FindPlace %d/%d] '%s'", i, len(queries), name)
        try:
            candidates, status = await _find_place_candidates(
                client, api_key, name, language, locationbias
            )
        except Exception as e:
            log.error("Fejl under FindPlace for '%s': %s", name, e)
            continue

        if status not in ("OK", "ZERO_RESULTS"):
            log.warning("FindPlace status %s for '%s'", status, name)
        if not candidates:
            continue

        for cand in candidates:
            pid = (cand or {}).get("place_id")
            if not pid:
                continue
            try:
                j2 = await _get_place_details(client, api_key, pid, language)
            except Exception as e:
                log.error("Fejl ved detaljer for Place ID '%s': %s", pid, e)
                continue

            if (j2 or {}).get("status") != "OK":
                log.warning("Detalje-status %s for Place ID '%s'", (j2 or {}).get("status"), pid)
                continue

            result = (j2 or {}).get("result") or {}
            rating = result.get("rating")
            count = result.get("user_ratings_total") or 0
            photos = result.get("photos") or []
            opening_hours = result.get("opening_hours") or {}
            website = (result.get("website") or "").strip().lower()

            target_name = _target_company_from(url, company_name)
            name_sim = _name_similarity(result.get("name") or "", target_name)
            contains_token = _contains_as_token(result.get("name") or "", target_name)
            name_ok = (name_sim >= 0.92 or contains_token) and not _looks_confusable(
                target_name, result.get("name") or ""
            )
            website_ok = _domain_match(website, url)

            # --- PATCH: phone-match mod site_phones_set ---
            g_place_phone_int = result.get("international_phone_number") or ""
            g_place_phone_fmt = result.get("formatted_phone_number") or ""
            phone_ok = any(
                _phones_equal(gp, sp)
                for gp in (g_place_phone_int, g_place_phone_fmt)
                for sp in site_phones_set
            )

            decision = "ACCEPT" if (website_ok or name_ok or phone_ok) else "REJECT"
            log.info(
                "[GMB/ANCHOR] decision=%s name='%s' sim=%.2f website_ok=%s phone_ok=%s reviews=%s "
                "place_phone_int='%s' place_phone_fmt='%s' website='%s' place_id=%s",
                decision, result.get("name"), name_sim, website_ok, phone_ok, count,
                g_place_phone_int, g_place_phone_fmt, website, pid
            )

            if decision == "ACCEPT":
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
                        "gmb_place_id": pid,
                        "gmb_status": "ok",
                        "website": website or "",
                    }
                    log.info("===== [GMB/RESULT] status=ok via QUERY place_id=%s name='%s' =====",
                             pid, result.get("name"))
                    return final_result

    # -------- Fallback: Text Search nær center --------
    center = _center_from_url(url)
    if center:
        lat, lng, radius = center
        for i, name in enumerate(queries, 1):
            log.info("[Fallback/TextSearch %d/%d] '%s' near %.4f,%.4f (r=%dm)",
                     i, len(queries), name, lat, lng, radius)
            try:
                ts_url = (
                    "https://maps.googleapis.com/maps/api/place/textsearch/json"
                    f"?query={quote_plus(name)}{f'&language={language}' if language else ''}"
                    f"&location={lat},{lng}&radius={radius}"
                    f"&key={api_key}"
                )
                r_ts = await client.httpx_get(ts_url)
                j_ts = r_ts.json() or {}
            except Exception as e:
                log.error("Fejl under TextSearch for '%s': %s", name, e)
                continue

            results = (j_ts or {}).get("results") or []
            if not results:
                continue

            for res in results:
                pid = (res or {}).get("place_id")
                if not pid:
                    continue
                try:
                    j2 = await _get_place_details(client, api_key, pid, language)
                except Exception as e:
                    log.error("Fejl ved detaljer (TextSearch) for Place ID '%s': %s", pid, e)
                    continue
                if (j2 or {}).get("status") != "OK":
                    continue

                result = (j2 or {}).get("result") or {}
                rating = result.get("rating")
                count = result.get("user_ratings_total") or 0
                photos = result.get("photos") or []
                opening_hours = result.get("opening_hours") or {}
                website = (result.get("website") or "").strip().lower()

                target_name = _target_company_from(url, company_name)
                name_sim = _name_similarity(result.get("name") or "", target_name)
                contains_token = _contains_as_token(result.get("name") or "", target_name)
                name_ok = (name_sim >= 0.92 or contains_token) and not _looks_confusable(
                    target_name, result.get("name") or ""
                )
                website_ok = _domain_match(website, url)

                # --- PATCH: phone-match mod site_phones_set ---
                g_place_phone_int = result.get("international_phone_number") or ""
                g_place_phone_fmt = result.get("formatted_phone_number") or ""

                phone_ok = any(
                    _phones_equal(gp, sp)
                    for gp in (g_place_phone_int, g_place_phone_fmt)
                    for sp in site_phones_set
                )

                decision = "ACCEPT" if (website_ok or name_ok or phone_ok) else "REJECT"
                log.info(
                    "[GMB/ANCHOR/FALLBACK] decision=%s name='%s' sim=%.2f website_ok=%s phone_ok=%s reviews=%s "
                    "place_phone_int='%s' place_phone_fmt='%s' website='%s' place_id=%s",
                    decision, result.get("name"), name_sim, website_ok, phone_ok, count,
                    g_place_phone_int, g_place_phone_fmt, website, pid
                )

                if decision == "ACCEPT":
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
                        "gmb_place_id": pid,
                        "gmb_status": "ok",
                        "website": website or "",
                    }
                    log.info("===== [GMB/RESULT] status=ok via FALLBACK place_id=%s name='%s' =====",
                             pid, result.get("name"))
                    return final_result

    # -------- Intet fundet --------
    log.info("===== [GMB/RESULT] status=zero_results note='no acceptable candidates' =====")
    return _empty_metrics(status="zero_results")
