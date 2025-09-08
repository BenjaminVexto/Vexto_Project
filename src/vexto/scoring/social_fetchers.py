# src/vexto/scoring/social_fetchers.py

from urllib.parse import urlparse, urljoin, parse_qs
import logging
from typing import Dict, List, Set
from bs4 import BeautifulSoup
from typing import Dict, Any, List
import re
from .schemas import SocialAndReputationMetrics

log = logging.getLogger(__name__)

SOCIAL_MEDIA_DOMAINS = [
    "facebook.com", "fb.me", "m.facebook.com",
    "instagram.com",
    "linkedin.com", "dk.linkedin.com",
    "twitter.com", "x.com",
    "tiktok.com",
    "youtube.com", "youtu.be",
    "pinterest.com", "pin.it",
    "threads.net",
    "telegram.me", "t.me",
    "whatsapp.com", "wa.me",
    "snapchat.com", "sc.app",
    "vimeo.com",
    "reddit.com",
    "trustpilot.com",  
]

KNOWN_SOCIAL = [
    "facebook.com", "instagram.com", "linkedin.com", "twitter.com", "x.com",
    "youtube.com", "tiktok.com", "pinterest.com", "threads.net",
    "t.me", "telegram.me", "whatsapp.com",
]

SOCIAL_PAT = re.compile(
    r"(facebook|instagram|linkedin|youtube|twitter|x\.com|tiktok|pinterest|threads|whatsapp)\.(com|net)|t\.me|telegram\.me",
    re.I,
)
SOCIAL_HOST_RE = re.compile(
    r"(facebook\.com|instagram\.com|linkedin\.com|youtube\.com|tiktok\.com|twitter\.com|x\.com|pinterest\.com|threads\.net|t\.me|telegram\.me|whatsapp\.com)",
    re.I,
)

def find_social_media_links(soup, base_url: str) -> Dict[str, Any]:
    """
    Wrapper der bruger den mere præcise klassifikation i find_social_links(...).
    Returnerer altid begge felter: social_media_links (profiler) og social_share_links (dele-URLs).
    Indeholder en defensiv efter-filtrering, der flytter delings-/post-URLs fra profiler -> shares.
    """
    try:
        html = str(soup) if soup else ""
        res = find_social_links(html)  # genbrug avanceret parsing/normalisering
        profiles = res.get("social_media_links", []) or []
        shares = res.get("social_share_links", []) or []

        # Gør eventuelle relative URLs absolutte ift. base_url (defensivt, lav risiko)
        def _abs(u: str) -> str:
            try:
                return urljoin(base_url, u) if u.startswith(("/", "./")) else u
            except Exception:
                return u

        profiles = [_abs(u) for u in profiles]
        shares   = [_abs(u) for u in shares]

        # Defensiv efter-filtrering: flyt tydelige delings-/post-URLs fra profiler til shares
        try:
            from urllib.parse import urlparse, parse_qs  # lokal import for robusthed
            def _is_share_like(u: str) -> bool:
                """
                Klassificér kun som 'share' ved eksplicitte delings-/post-mønstre.
                YouTube-videoer (/watch, youtu.be/…) er IKKE 'share' her – de bliver i profiler.
                """
                try:
                    p = urlparse(u)
                    netloc = (p.netloc or "").lower().replace("www.", "")
                    path = p.path or ""
                    q = parse_qs(p.query or "")

                    # Domænespecifikke mønstre for deling/post
                    if netloc == "instagram.com":
                        # /p/<id> er et post; /reel/ også
                        if path.startswith("/p/") or path.startswith("/reel/"):
                            return True
                    if netloc == "facebook.com":
                        # klassiske delingsmønstre
                        if "/share" in path or "/posts/" in path or "/watch/" in path:
                            return True
                    if netloc in {"x.com", "twitter.com"}:
                        # /status/<id> er et tweet, ikke en profil
                        if "/status/" in path:
                            return True
                    if netloc == "tiktok.com":
                        # /@user/video/<id> er en post
                        if "/video/" in path:
                            return True
                    if netloc == "pinterest.com":
                        # /pin/<id> er en pin (post)
                        if path.startswith("/pin/"):
                            return True

                    # Generisk: eksplicit “share” i query (fx ?share=1)
                    if "share" in q:
                        return True
                except Exception:
                    return False
                return False

            moved = []
            kept_profiles = []
            for u in profiles:
                if _is_share_like(u):
                    moved.append(u)
                else:
                    kept_profiles.append(u)
            # merge og dedup
            profiles = kept_profiles
            shares = list(dict.fromkeys(shares + moved))
        except Exception:
            # Hvis noget går galt i klassifikation, behold oprindelige lister
            pass

        # Dedup (afsluttende)
        profiles = list(dict.fromkeys(profiles))
        shares   = list(dict.fromkeys(shares))
        return {"social_media_links": profiles, "social_share_links": shares}
    except Exception:
        return {"social_media_links": [], "social_share_links": []}


def find_social_links(html: str) -> Dict[str, Any]:
    """
    Finder både profil-links og delings-links til sociale medier.
    - Opdager <a href> direkte, inkl. relative og protocol-relative URL'er
    - Opdager ikon-links (SVG/i/img) via aria-label/title/klasse
    - Opdager onclick-share-handlers (window.open(...))
    - Normaliserer og dedupliker links (fjerner fx utm/fbclid/gclid/locale)
    - Skelner mellem profiler og share-handlers (sharer, intent, share, tweet, etc.)
    """
    import re
    from urllib.parse import urlparse, urlunparse, parse_qsl, urlencode

    soup = BeautifulSoup(html or "", "html.parser")

    # Konstanter
    SOCIAL_DOMAINS = (
        "facebook.com","instagram.com","linkedin.com","twitter.com","x.com",
        "tiktok.com","youtube.com","pinterest.com","threads.net","t.me","telegram.me"
    )
    SHARE_HINTS = ("share", "sharer", "intent", "tweet", "post", "send")
    NOISE_QUERY_KEYS = {
        "utm_source","utm_medium","utm_campaign","utm_term","utm_content",
        "fbclid","gclid","mc_cid","mc_eid","ref","locale","lang"
    }
    # Host-kanonisering til dedup
    HOST_MAP = {
        "www.facebook.com": "facebook.com",
        "m.facebook.com": "facebook.com",
        "mobile.twitter.com": "x.com",
        "twitter.com": "x.com",   # kanoniser til x.com
        "www.twitter.com": "x.com",
        "www.instagram.com": "instagram.com",
        "www.linkedin.com": "linkedin.com",
        "dk.linkedin.com": "linkedin.com",
        "m.youtube.com": "youtube.com",
        "www.youtube.com": "youtube.com",
        "youtu.be": "youtube.com",  # ny: short-links -> normaliser
    }

    # Kendte share-endpoints (mere præcis end blot substring)
    SHARE_PATH_RX = re.compile(
        r"("
        r"/sharer\.php|/share\.php|/dialog/share|/intent/(tweet|post)|"
        r"/shareArticle|/pin/create/button|/send|/share|/post"
        r")",
        re.I,
    )

    def _domain_match(netloc: str) -> str | None:
        host = (netloc or "").lower().lstrip("www.")
        for d in SOCIAL_DOMAINS:
            if host.endswith(d):
                return d
        return None

    def _normalize(u: str) -> str:
        try:
            p = urlparse(u)
            scheme = "https" if p.scheme in ("http", "https", "") else p.scheme
            netloc = HOST_MAP.get(p.netloc.lower(), p.netloc.lower())

            # Fjern trackingparametre
            q = [(k, v) for k, v in parse_qsl(p.query, keep_blank_values=True)
                 if k.lower() not in NOISE_QUERY_KEYS and not k.lower().startswith("utm_")]
            query = urlencode(q, doseq=True)

            # Fjern fragment for profil-URLs (behold for share hvis det indeholder nyttedata)
            fragment = "" if not SHARE_PATH_RX.search(p.path) else p.fragment

            # Trim trailing slash på profiler
            path = p.path.rstrip("/") if not SHARE_PATH_RX.search(p.path) else p.path

            return urlunparse((scheme, netloc, path, p.params, query, fragment))
        except Exception:
            return u

    def _resolve(href: str | None) -> str | None:
        if not href:
            return None
        href = href.strip()
        # Prøv projektets egen resolver hvis tilgængelig
        try:
            return _resolve_href(href)  # type: ignore[name-defined]
        except Exception:
            pass
        if href.startswith("//"):
            return "https:" + href
        return href

    def _is_share_url(u: str) -> bool:
        p = urlparse(u)
        if SHARE_PATH_RX.search(p.path):
            return True
        # Twitter/X særligt
        if p.netloc.lower().endswith(("twitter.com", "x.com")) and "/intent/" in p.path.lower():
            return True
        return any(h in p.path.lower() for h in SHARE_HINTS)

    def _maybe_social_from_label(a_tag) -> str | None:
        bits = " ".join(filter(None, [
            a_tag.get("aria-label"), a_tag.get("title")
        ])).lower()

        i_tag = a_tag.find("i")
        img = a_tag.find("img")
        svg = a_tag.find("svg")
        if i_tag:
            bits += " " + " ".join((i_tag.get("class") or []))
        if img:
            bits += " " + (img.get("alt") or "") + " " + (img.get("title") or "")
        if svg:
            bits += " " + (svg.get("title") or "") + " " + (a_tag.get("aria-label") or "")

        if re.search(r"\bfacebook\b", bits): return "facebook"
        if re.search(r"\binstagram\b", bits): return "instagram"
        if re.search(r"\blinked?in\b", bits): return "linkedin"
        if re.search(r"\b(?:twitter|x)\b", bits): return "twitter"
        if re.search(r"\btiktok\b", bits): return "tiktok"
        if re.search(r"\byoutube\b", bits): return "youtube"
        if re.search(r"\bpinterest\b", bits): return "pinterest"
        if re.search(r"\bthreads\b", bits): return "threads"
        if re.search(r"\b(telegram|t\.me)\b", bits): return "telegram"
        return None

    profiles: List[str] = []
    shares: List[str] = []

    # 1) <link rel="me"> og andre link-tags
    for link in soup.find_all("link", href=True):
        rels = " ".join(link.get("rel") or []).lower()
        if "me" in rels or "author" in rels or "publisher" in rels:
            href = _resolve(link["href"])
            if href and _domain_match(urlparse(href).netloc):
                profiles.append(_normalize(href))

    # 2) Direkte <a href>-links
    for a in soup.find_all("a", href=True):
        href = _resolve(a.get("href"))
        if not href:
            continue
        p = urlparse(href)
        # Kun http(s)
        if p.scheme and p.scheme not in ("http", "https"):
            # Onclick-share senere
            pass
        dmatch = _domain_match(p.netloc)
        if dmatch:
            path_lower = p.path.lower()
            # Udeluk policy/terms etc.
            if any(x in path_lower for x in ("policy", "privacy", "legal", "terms")):
                continue
            if _is_share_url(href):
                shares.append(_normalize(href))
            else:
                profiles.append(_normalize(href))
            continue

        # 3) Ikon/label-hints (hvis href er JS/relativ)
        hinted = _maybe_social_from_label(a)
        if hinted and href:
            if _is_share_url(href):
                shares.append(_normalize(href))
            else:
                profiles.append(_normalize(href))

        # 4) Onclick-share (window.open('https://...'))
        onclick = (a.get("onclick") or "") + " " + (a.get("data-onclick") or "")
        m = re.search(r"window\.open\(\s*['\"](https?://[^'\"\s]+)['\"]", onclick, re.I)
        if m:
            u2 = m.group(1)
            if _domain_match(urlparse(u2).netloc) or SHARE_PATH_RX.search(urlparse(u2).path):
                if _is_share_url(u2):
                    shares.append(_normalize(u2))
                else:
                    profiles.append(_normalize(u2))

    # 5) Ekstra: eksplicitte selektorer for ikon-links
    for name in ("facebook","instagram","linkedin","twitter","x","tiktok","youtube","pinterest","threads","telegram"):
        sel = f'a[aria-label*="{name}" i], a[title*="{name}" i], a i[class*="{name}" i], a svg[title*="{name}" i]'
        for a in soup.select(sel):
            href = _resolve(a.get("href"))
            if not href:
                continue
            if _is_share_url(href):
                shares.append(_normalize(href))
            else:
                profiles.append(_normalize(href))

    # 6) Dedup
    def _dedup(seq: List[str]) -> List[str]:
        out, seen = [], set()
        for u in seq:
            if u not in seen:
                seen.add(u)
                out.append(u)
        return out

    return {
        "social_media_links": _dedup(profiles),
        "social_share_links": _dedup(shares),
    }


def _normalize_href(h, base_url):
    if not h: return ""
    h = h.strip()
    if h.startswith("//"): return "https:" + h
    return urljoin(base_url, h)

def _is_social(h: str) -> bool:
    h = (h or "").lower()
    return any(x in h for x in (
        "facebook.com","instagram.com","linkedin.com","youtube.com","tiktok.com","x.com","twitter.com"
    ))

def extract_social(soup, base_url: str):
    """
    Behold API-kontrakten (liste af profiler), men udnyt nu find_social_media_links(...)
    for konsistent klassifikation på tværs af moduler.
    """
    res = find_social_media_links(soup, base_url)
    return res.get("social_media_links", []) or []
