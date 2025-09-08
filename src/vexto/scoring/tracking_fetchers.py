# file: src/vexto/scoring/tracking_fetchers.py

import logging
import re
from typing import Dict, Any, List, Optional

from bs4 import BeautifulSoup
from .schemas import ConversionMetrics  # kun type hinting; ikke strikt påkrævet
from .http_client import AsyncHtmlClient

log = logging.getLogger(__name__)


def _collect_text_targets(soup: Optional[BeautifulSoup]) -> List[str]:
    """
    Samler relevante tekststykker/URLs fra DOM for robust regex-detektion:
    - <script src> og inline script-tekst
    - <img>/<iframe>/<link> src/href (pixels kalder ofte disse endpoints)
    - <noscript> indhold (indeholder ofte fallback-pixels)
    - data-src attributter
    """
    targets: List[str] = []
    if not soup:
        return targets

    # script src + inline
    for s in soup.find_all("script"):
        src = s.get("src")
        if src:
            targets.append(src)
        txt = s.string or s.get_text()
        if txt:
            targets.append(txt)

    # img / iframe / link src/href
    for tag in soup.find_all(["img", "iframe", "link"]):
        src = tag.get("src") or tag.get("href")
        if src:
            targets.append(src)

    # noscript HTML (kan indeholde <img src="..."> til pixels)
    for ns in soup.find_all("noscript"):
        targets.append(str(ns))

    # diverse data-* der kan indeholde URLs
    for el in soup.find_all(attrs={"data-src": True}):
        targets.append(el.get("data-src"))

    # Rens tomme
    return [t for t in targets if t]


async def fetch_tracking_data(client: AsyncHtmlClient, soup: BeautifulSoup) -> Dict[str, Any]:
    """
    Udvidet tracking-detektor:
    - Google: GA4 (G-XXXX), GTM (GTM-XXXX), Google Ads (AW-ids + labels + viewthroughconversion)
    - Meta/Facebook Pixel
    - TikTok Pixel
    - Pinterest Tag
    - Snapchat Pixel
    - LinkedIn Insight Tag
    - Bing/Microsoft UET
    - Twitter/X Pixel
    Returnerer et dict der kan merges i ConversionMetrics.
    """
    # Standard-felter (bagudkompatibel med din eksisterende output)
    result: Dict[str, Any] = {
        # Google   
        "has_ga4": False,   
        "ga4_measurement_id": None,   
        "has_gtm": False,   
        "gtm_container_id": None,
        "google_tag_id": None,              # <-- NY: Google Tag (GT-XXXXX)
        "google_ads_conversion_ids": [],    # f.eks. ["AW-1008353992"]   
        "google_ads_labels": [],            # f.eks. ["AW-1008353992/abcdEFGH"]
        # Meta (Facebook/Instagram)   
        "has_meta_pixel": False,   
        "meta_pixel_id": None, 

        # Meta (Facebook/Instagram)
        "has_meta_pixel": False,
        "meta_pixel_id": None,

        # TikTok
        "has_tiktok_pixel": False,
        "tiktok_pixel_id": None,

        # Pinterest
        "has_pinterest_tag": False,
        "pinterest_tag_id": None,

        # Snapchat
        "has_snap_pixel": False,
        "snap_pixel_id": None,

        # LinkedIn
        "has_linkedin_insight": False,
        "linkedin_partner_id": None,

        # Bing / Microsoft Ads (UET)
        "has_bing_uet": False,
        "bing_uet_id": None,

        # Twitter / X
        "has_twitter_pixel": False,
        "twitter_pixel_id": None,
    }

    if not soup:
        return result

    try:
        # Kombinér alt "interessant" indhold i én stor blob til regex
        parts = _collect_text_targets(soup)
        html_str = str(soup)
        blobs = "\n".join(parts + [html_str])

        # -------------------------
        # Google Analytics 4 (GA4) + Google Tag (GT-)
        # -------------------------
        ga4_id = None
        ga4_src = None  # 'html' | 'runtime_gtag' | 'runtime_send_to' | 'inferred'
        google_tag_id = None
        send_to_ids: list[str] = []

        # 1) Direkte G- i HTML (stærk)
        m = re.search(r"\bG-[A-Z0-9\-]{6,}\b", blobs, re.I)
        if m:
            ga4_id = m.group(0); ga4_src = "html"
        if not ga4_id:
            m = re.search(r"gtag\(\s*['\"]config['\"]\s*,\s*['\"](G-[A-Z0-9\-]{6,})['\"]", blobs, re.I)
            if m: ga4_id = m.group(1); ga4_src = "html"
        if not ga4_id:
            m = re.search(r"googletagmanager\.com/gtag/js\?id=(G-[A-Z0-9\-]{6,})", blobs, re.I)
            if m: ga4_id = m.group(1); ga4_src = "html"

        # 2) data-attributes (html)
        if not ga4_id:
            m = re.search(r'data-(?:ga4-)?measurement-id=["\'](G-[A-Z0-9\-]{6,})["\']', blobs, re.I)
            if m: ga4_id = m.group(1); ga4_src = "html"

        # 3) Runtime hints (send_to / gtag queue) – hvis Playwright fanger dem, læg dem i blobs før dette
        m = re.findall(r"['\"]send_to['\"].*?['\"](G-[A-Z0-9\-]{6,})['\"]", blobs, re.S | re.I)
        if m:
            send_to_ids.extend([x for x in m if x])
            if not ga4_id:
                ga4_id = m[0]; ga4_src = "runtime_send_to"
        if not ga4_id:
            m = re.search(r"gtag\(\s*['\"]config['\"].*?['\"](G-[A-Z0-9\-]{6,})['\"]", blobs, re.S | re.I)
            if m: ga4_id = m.group(1); ga4_src = "runtime_gtag"

        # 4) Google Tag (GT-*) – label separat
        m = re.search(r"\bGT-[A-Z0-9\-]{6,}\b", blobs, re.I)
        if m:
            google_tag_id = m.group(0)

        # Sæt resultater
        if ga4_id:
            result["has_ga4"] = True
            result["ga4_measurement_id"] = ga4_id
            result["ga4_measurement_id_source"] = ga4_src or "html"
        else:
            # Svage indikatorer – behold flag men uden måle-ID
            if re.search(r"googletagmanager\.com/(gtm|gtag)/", blobs) or \
            re.search(r"window\.dataLayer\s*=", blobs) or \
            re.search(r"google-analytics\.com/(analytics|ga)\.js", blobs):
                result["has_ga4"] = True
                result["ga4_measurement_id_source"] = "inferred"

        if google_tag_id:
            result["google_tag_id"] = google_tag_id
        if send_to_ids:
            # gem som telemetri (bruges også til Ads)
            result["google_ads_send_to"] = list({*result.get("google_ads_send_to", []), *send_to_ids})


        # Valgfri GA4-fallback: data-measurement-id="G-XXXX"   
        if not result["ga4_measurement_id"]:   
            m = re.search(r'data-(?:ga4-)?measurement-id=["\'](G-[A-Z0-9\-]{6,})["\']', blobs, re.IGNORECASE)   
            if m:   
                result["ga4_measurement_id"] = m.group(1)   
                result["has_ga4"] = True   
            else:   
                # Ekstra fallback: GA4 config i dataLayer   
                m2 = re.search(r"dataLayer\s*=\s*\[.*?['\"]config['\"].*?['\"](G-[A-Z0-9\-]{6,})['\"]", blobs, re.S | re.I)   
                if m2:   
                    result["ga4_measurement_id"] = m2.group(1)  


        # -------------------------
        # Google Tag Manager (GTM)
        # -------------------------
        m = re.search(r"\bGTM-[A-Z0-9]{4,}\b", blobs)
        if m:
            result["has_gtm"] = True
            result["gtm_container_id"] = m.group(0)
        # Fallback på script/noscript uden at ID kunne læses
        if not result["has_gtm"]:
            if "googletagmanager.com/gtm.js" in blobs or "googletagmanager.com/ns.html" in blobs:
                result["has_gtm"] = True

        # -----------------------------------------
        # Google Ads (AdWords) conversion IDs/label
        # -----------------------------------------
        # 1) AW-XXXXXX  (send_to, gtag config)
        aw_ids = set(re.findall(r"\bAW-\d{6,}\b", blobs))
        # 2) AW-XXXXXX/label
        aw_labels = set(re.findall(r"\bAW-\d{6,}/[A-Za-z0-9_-]+\b", blobs))
        # 3) viewthroughconversion/123456 (doubleclick pixel) -> AW-123456
        for conv_id in re.findall(r"viewthroughconversion/(\d{6,})", blobs):
            aw_ids.add(f"AW-{conv_id}")

        result["google_ads_conversion_ids"] = sorted(aw_ids)
        result["google_ads_labels"] = sorted(aw_labels)

        # --------------------------
        # Meta / Facebook Pixel
        # --------------------------
        # ID i fbq('init','123...')
        fb_id = None
        m = re.search(r"fbq\(\s*['\"]init['\"]\s*,\s*['\"](\d{5,20})['\"]\s*\)", blobs, re.IGNORECASE)
        if m:
            fb_id = m.group(1)
        if not fb_id:
            # facebook.com/tr?id=123...
            m = re.search(r"facebook\.com/tr\?[^\"'\s]*\bid=(\d{5,20})\b", blobs, re.IGNORECASE)
            if m:
                fb_id = m.group(1)
        if not fb_id:
            # connect.facebook.net/signals/config/123...
            m = re.search(r"signals/config/(\d{5,20})", blobs, re.IGNORECASE)
            if m:
                fb_id = m.group(1)

        # flag hvis script/endpoint findes
        if fb_id or "connect.facebook.net" in blobs or "facebook.com/tr" in blobs or "fbevents.js" in blobs:
            result["has_meta_pixel"] = True
            result["meta_pixel_id"] = fb_id or "ukendt"

        # --------------------------
        # TikTok Pixel
        # --------------------------
        # ttq.load('CXXXXX') eller analytics.tiktok.com/?sdkid=CXXXX
        tt_id = None
        m = re.search(r"ttq\.load\(\s*['\"]([A-Z0-9_-]{5,})['\"]\s*\)", blobs, re.IGNORECASE)
        if not m:
            m = re.search(r"analytics\.tiktok\.com/[^\"'\s]*[?&]sdkid=([A-Z0-9_-]{5,})", blobs, re.IGNORECASE)
        if m:
            tt_id = m.group(1)
        if tt_id or "analytics.tiktok.com" in blobs or "ttq.load(" in blobs:
            result["has_tiktok_pixel"] = True
            result["tiktok_pixel_id"] = tt_id or "ukendt"

        # --------------------------
        # Pinterest Tag
        # --------------------------
        # pintrk('load','1234567890') eller ct.pinterest.com/v3/?tid=123
        pin_id = None
        m = re.search(r"pintrk\(\s*['\"]load['\"]\s*,\s*['\"](\d+)['\"]\s*\)", blobs, re.IGNORECASE)
        if not m:
            m = re.search(r"ct\.pinterest\.com/[^\"'\s]*[?&]tid=(\d+)", blobs, re.IGNORECASE)
        if m:
            pin_id = m.group(1)
        if pin_id or "pinimg.com/ct/" in blobs or "ct.pinterest.com" in blobs or "pintrk(" in blobs:
            result["has_pinterest_tag"] = True
            result["pinterest_tag_id"] = pin_id or "ukendt"

        # --------------------------
        # Snapchat Pixel
        # --------------------------
        # snaptr('init','abc123...')
        snap_id = None
        m = re.search(r"snaptr\(\s*['\"]init['\"]\s*,\s*['\"]([A-Za-z0-9]{8,})['\"]\s*\)", blobs, re.IGNORECASE)
        if m:
            snap_id = m.group(1)
        if snap_id or "sc-static.net/scevent.min.js" in blobs or "snaptr(" in blobs:
            result["has_snap_pixel"] = True
            result["snap_pixel_id"] = snap_id or "ukendt"

        # --------------------------
        # LinkedIn Insight Tag
        # --------------------------
        # window._linkedin_data_partner_ids.push(123456)
        # window._linkedin_partner_id = "123456"
        li_id = None
        m = re.search(r"_linkedin_data_partner_ids\.push\((\d+)\)", blobs)
        if m:
            li_id = m.group(1)
        if not li_id:
            m = re.search(r"_linkedin_partner_id\s*=\s*['\"]?(\d+)['\"]?", blobs)
            if m:
                li_id = m.group(1)

        if (
            li_id
            or "snap.licdn.com/li.lms-analytics/insight" in blobs
            or "px.ads.linkedin.com" in blobs
            or "licdn.com/li/insight.min.js" in blobs
        ):
            result["has_linkedin_insight"] = True
            result["linkedin_partner_id"] = li_id or "ukendt"

        # --------------------------
        # Bing / Microsoft Advertising (UET)
        # --------------------------
        # Tag ID i query-param 'ti='
        uet_id = None
        m = re.search(r"[?&]ti=(\d+)\b", blobs)
        if m:
            uet_id = m.group(1)
        if uet_id or "bat.bing.com/bat.js" in blobs or "bat.bing.com" in blobs:
            result["has_bing_uet"] = True
            result["bing_uet_id"] = uet_id or "ukendt"

        # --------------------------
        # Twitter / X Pixel
        # --------------------------
        # twq('init','o0o0o') + script static.ads-twitter.com/uwt.js
        tw_id = None
        m = re.search(r"twq\(\s*['\"]init['\"]\s*,\s*['\"]([A-Za-z0-9_-]{5,})['\"]\s*\)", blobs)
        if m:
            tw_id = m.group(1)
        if tw_id or "static.ads-twitter.com/uwt.js" in blobs or "twq(" in blobs:
            result["has_twitter_pixel"] = True
            result["twitter_pixel_id"] = tw_id or "ukendt"

        # --------------------------
        # Konsistent logging
        # --------------------------
        log.info(
            "Analytics detection complete - GA4: %s (id=%s), Meta: %s (pixel_id=%s), GTM: %s%s; "
            "TikTok: %s (id=%s), Pinterest: %s (id=%s), Snapchat: %s (id=%s); "
            "LinkedIn: %s (partner_id=%s), Bing UET: %s (id=%s), Twitter/X: %s (id=%s); "
            "Google Ads: ids=%s, labels=%s",
            result["has_ga4"], result["ga4_measurement_id"] or "ukendt",
            result["has_meta_pixel"], result["meta_pixel_id"] or "ukendt",
            result["has_gtm"], f" ({result['gtm_container_id']})" if result.get("gtm_container_id") else "",
            result["has_tiktok_pixel"], result["tiktok_pixel_id"] or "ukendt",
            result["has_pinterest_tag"], result["pinterest_tag_id"] or "ukendt",
            result["has_snap_pixel"], result["snap_pixel_id"] or "ukendt",
            result["has_linkedin_insight"], result["linkedin_partner_id"] or "ukendt",
            result["has_bing_uet"], result["bing_uet_id"] or "ukendt",
            result["has_twitter_pixel"], result["twitter_pixel_id"] or "ukendt",
            result["google_ads_conversion_ids"], result["google_ads_labels"],
        )

        return result

    except Exception as e:
        log.error(f"Error in fetch_tracking_data: {e}", exc_info=True)
        return result
