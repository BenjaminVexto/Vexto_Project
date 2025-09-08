# src/vexto/scoring/image_fetchers.py

import asyncio
import logging
from typing import Dict, Any, List, Set
from urllib.parse import urljoin, urlparse
from bs4 import BeautifulSoup
import httpx
import re

log = logging.getLogger(__name__)

ASSET_IMG_EXT = (".jpg", ".jpeg", ".png", ".gif", ".webp", ".ico")
BLOCKED_IMG_HOST_PREFIXES = ("m2.", "cdn.", "media.", "assets.")
RESIZE_PATTERNS = ("/img/160/160/resize/", "/img/360/360/resize/")

# NY: typiske dekorative/placeholder/ikon-mønstre
DECORATIVE_CLASS_RX = re.compile(r"(lazy\-?placeholder|placeholder|spacer|pixel|tracking|sprite|icon|badge|avatar|logo)", re.I)
DECORATIVE_SRC_RX   = re.compile(r"(placeholder|spacer|pixel|tracking|sprite|icons?|logo|avatar|transparent|blank)\b", re.I)

def _is_meaningful_image(img, src_url: str) -> bool:
    """
    Et billede betragtes som 'meningsfuldt', hvis det ikke er eksplicit dekorativt.
    Vi udelukker:
      - alt="" (eksplicit dekorativt). *NB:* manglende alt-attribut tæller som meningsfuldt (for at kunne give straf).
      - role="presentation" eller aria-hidden="true"
      - skjult via inline style
      - kendte dekorations-/placeholder-klasser eller -filnavne
      - billeder i <footer>/<nav> (typisk ikoner/footers)
    """
    # alt="" → dekorativt; hvis alt ATTRIBUTTEN mangler, regnes billedet som meningsfuldt
    if img.has_attr("alt") and (img.get("alt") or "").strip() == "":
        return False

    role = (img.get("role") or "").strip().lower()
    if role == "presentation":
        return False
    if (img.get("aria-hidden") or "").strip().lower() == "true":
        return False

    style = (img.get("style") or "").replace(" ", "").lower()
    if "display:none" in style or "visibility:hidden" in style:
        return False

    classes = " ".join(img.get("class") or []).lower()
    if DECORATIVE_CLASS_RX.search(classes):
        return False
    if DECORATIVE_SRC_RX.search(src_url or ""):
        return False

    # Tjek for placeringskontekst: <footer> / <nav> → ofte dekorativt
    parent = img
    hops = 0
    while parent is not None and hops < 4:
        parent = getattr(parent, "parent", None)
        hops += 1
        if not parent:
            break
        tag = getattr(parent, "name", "") or ""
        if tag in ("footer", "nav"):
            return False
        if isinstance(getattr(parent, "attrs", {}), dict) and (parent.attrs.get("aria-hidden") == "true"):
            return False

    return True

def _is_blocked_host(url: str) -> bool:
    try:
        host = urlparse(url).netloc.lower()
    except Exception:
        return False
    return any(host.startswith(p) for p in BLOCKED_IMG_HOST_PREFIXES)

def _is_resized_pattern(url: str) -> bool:
    u = url.lower()
    return any(p in u for p in RESIZE_PATTERNS)

def _pick_src_from_tag(img) -> str | None:
    # Prioritet: src > data-src > srcset (første kandidat)
    for attr in ("src", "data-src"):
        v = (img.get(attr) or "").strip()
        if v:
            return v
    srcset = (img.get("srcset") or "").strip()
    if srcset:
        # vælg første URL før evt. “ 2x”
        first = srcset.split(",")[0].strip().split(" ")[0].strip()
        return first or None
    return None

async def _head_content_length_many(urls: List[str], base_headers: Dict[str, str] | None = None) -> Dict[str, int]:
    """
    HEAD et begrænset sæt billeder og returnér content-length i bytes (hvis tilgængelig).
    Fallback: lav en letvægts GET med Range: bytes=0-0 og parse Content-Range.
    """
    if not urls:
        return {}

    timeout = httpx.Timeout(10.0, connect=10.0)
    limits = httpx.Limits(max_keepalive_connections=10, max_connections=20)
    headers = base_headers or {}
    out: Dict[str, int] = {}

    async def _try_range_get(hc: httpx.AsyncClient, u: str) -> None:
        try:
            r = await hc.get(u, headers={**headers, "Range": "bytes=0-0"})
            # Eksempel: Content-Range: bytes 0-0/12345
            cr = r.headers.get("content-range") or r.headers.get("Content-Range")
            if cr and "/" in cr:
                total = cr.split("/")[-1].strip()
                if total.isdigit():
                    out[u] = int(total)
                    return
            # Som sidste lille fallback: hvis server gav content-length på 1 byte, brug den ikke (irrelevant)
            cl = r.headers.get("content-length")
            if cl and cl.isdigit() and int(cl) > 1:
                out[u] = int(cl)
        except Exception as e:
            log.debug(f"Image Range GET failed for {u}: {e}")

    async with httpx.AsyncClient(timeout=timeout, limits=limits, follow_redirects=True, headers=headers, http2=True) as hc:
        async def probe(u: str):
            # Først HEAD
            try:
                r = await hc.head(u)
                cl = r.headers.get("content-length")
                if cl and cl.isdigit():
                    out[u] = int(cl)
                    return
            except Exception as e:
                log.debug(f"Image HEAD failed for {u}: {e}")
            # Fallback hvis HEAD ikke gav længde
            await _try_range_get(hc, u)

        await asyncio.gather(*(probe(u) for u in urls))

    return out

async def fetch_image_stats(client, soup: BeautifulSoup, page_url: str) -> Dict[str, Any]:
    """
    Returnerer:
      image_count, image_alt_count, image_alt_pct, avg_image_size_kb
    HEAD-probe begrænses (MAX_PROBE), skipper kendte CDN/resize-mønstre.
    """
    if not soup:
        return {"image_count": 0, "image_alt_count": 0, "image_alt_pct": 0, "avg_image_size_kb": 0}

    imgs = soup.find_all("img")
    if not imgs:
        return {"image_count": 0, "image_alt_count": 0, "image_alt_pct": 0, "avg_image_size_kb": 0}

    abs_urls: List[str] = []
    alt_count = 0
    seen: Set[str] = set()

    for img in imgs:
        src = _pick_src_from_tag(img)
        if not src:
            continue
        if src.startswith("data:") or src.lower().endswith(".svg"):
            continue
        absu = src if src.startswith("http") else urljoin(page_url, src)
        key = absu.split("?", 1)[0]

        # NY: kun meningsfulde billeder tælles i ALT-KPI
        if not _is_meaningful_image(img, key):
            continue

        # dedup
        if key in seen:
            continue
        seen.add(key)

        # kun typiske billedext (ellers lader vi probe/HEAD cleare)
        if not any(key.lower().endswith(ext) for ext in ASSET_IMG_EXT):
            pass
        abs_urls.append(absu)

        # ALT-count: tæller KUN hvis alt er ikke-tomt (manglende alt tæller som manglende ALT via procenten)
        alt = (img.get("alt") or "").strip()
        if alt:
            alt_count += 1

    image_count = len(abs_urls)
    image_alt_pct = round((alt_count / image_count) * 100, 1) if image_count else 0.0

    # Begræns HEAD-probe: kun et lille udsnit, og skip CDN/resize-patterns
    # Du kan skrue op/ned for MAX_PROBE efter behov.
    MAX_PROBE = 8
    probe_candidates = []
    for u in abs_urls:
        if _is_blocked_host(u):
            continue
        if _is_resized_pattern(u):
            continue
        if u.lower().endswith(".svg") or u.startswith("data:"):
            continue
        probe_candidates.append(u)
        if len(probe_candidates) >= MAX_PROBE:
            break

    # Brug samme headers som din AsyncHtmlClient hvis de findes
    base_headers = getattr(client, "headers", None)
    sizes = await _head_content_length_many(probe_candidates, base_headers=base_headers)
    avg_kb = 0
    if sizes:
        avg_kb = round(sum(sizes.values()) / len(sizes) / 1024, 1)

    return {
        "image_count": image_count,
        "image_alt_count": alt_count,
        "image_alt_pct": image_alt_pct,
        "avg_image_size_kb": avg_kb,
    }
