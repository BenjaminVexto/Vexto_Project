# src/vexto/scoring/security_fetchers.py

import logging
import re
from .schemas import SecurityMetrics
from .http_client import AsyncHtmlClient

log = logging.getLogger(__name__)

async def fetch_security_headers(client: AsyncHtmlClient, url: str) -> SecurityMetrics:
    """
    Checks for the presence of important security headers in the HTTP response.
    Accepterer også moderne beskyttelse via CSP 'frame-ancestors' som ækvivalent til X-Frame-Options.
    Rapporterer desuden:
      - measured_on: 'HEAD' eller 'GET' (hvilken metode, der gav headers)
      - hsts_include_subdomains / hsts_preload (parsede HSTS-detaljer)
    """
    security_data: SecurityMetrics = {
        'hsts_enabled': False,
        'csp_enabled': False,
        'x_content_type_options_enabled': False,
        'x_frame_options_enabled': False,
        # NYT:
        'measured_on': None,                 # 'HEAD' | 'GET'
        'hsts_include_subdomains': False,    # parsed fra HSTS
        'hsts_preload': False,               # parsed fra HSTS
    }

    try:
        response = await client.head(url)
        if not response or getattr(response, "status_code", 0) >= 400:
            # Nogle servere returnerer intet/brudt på HEAD → prøv GET
            get_resp = await client.get(url)
            if not get_resp:
                log.warning(f"Could not get a response for security headers check on {url}")
                return security_data
            base_headers = getattr(get_resp, "headers", {}) or {}
            security_data['measured_on'] = 'GET'   # NYT
        else:
            base_headers = getattr(response, "headers", {}) or {}
            security_data['measured_on'] = 'HEAD'  # NYT

        # Case-insensitive og evt. merge med GET som fallback
        headers = {k.lower(): v for k, v in base_headers.items()}
        if ("content-security-policy" not in headers) or ("x-frame-options" not in headers):
            try:
                get_resp2 = await client.get(url)
                if get_resp2 and getattr(get_resp2, "headers", None):
                    headers.update({k.lower(): v for k, v in get_resp2.headers.items()})
                    # hvis vi oprindeligt målte på HEAD men manglede noget, kan vi notere GET som kilde
                    if not security_data['measured_on']:
                        security_data['measured_on'] = 'GET'
            except Exception:
                pass

        # HSTS: kræv max-age>0
        hsts = headers.get('strict-transport-security', '')
        hsts_ok = False
        if isinstance(hsts, str) and 'max-age=' in hsts.lower():
            try:
                ma = int(hsts.split('max-age=')[1].split(';')[0].strip())
                hsts_ok = ma > 0
            except Exception:
                hsts_ok = False
        security_data['hsts_enabled'] = hsts_ok

        # NYT: parse includeSubDomains / preload fra HSTS
        if isinstance(hsts, str):
            hsts_l = hsts.lower()
            security_data['hsts_include_subdomains'] = 'includesubdomains' in hsts_l
            security_data['hsts_preload'] = 'preload' in hsts_l

        # CSP: accepter også report-only
        csp_present = (
            'content-security-policy' in headers or
            'content-security-policy-report-only' in headers
        )
        security_data['csp_enabled'] = csp_present

        # X-Content-Type-Options: kræv nosniff
        security_data['x_content_type_options_enabled'] = headers.get('x-content-type-options', '').lower() == 'nosniff'

        # X-Frame-Options: validerede værdier + accepter CSP 'frame-ancestors' som ækvivalent
        xfo = headers.get('x-frame-options', '')
        xfo_ok = xfo.lower() in ('deny', 'sameorigin', 'allow-from')

        # Undersøg CSP for frame-ancestors (både aktiv og report-only)
        csp_value = headers.get('content-security-policy', '') or headers.get('content-security-policy-report-only', '')
        has_frame_ancestors = False
        if isinstance(csp_value, str) and csp_value:
            m = re.search(r'frame-ancestors\s+([^;]+)', csp_value, re.IGNORECASE)
            if m:
                val = m.group(1).strip().strip('"').strip("'").lower()
                # Beskyt kun hvis ikke globalt åbent ('*') og der faktisk er en værdi
                if val and val != '*':
                    has_frame_ancestors = True

        security_data['x_frame_options_enabled'] = xfo_ok or has_frame_ancestors

    except Exception as e:
        log.error(f"Error fetching security headers for {url}: {e}", exc_info=True)

    return security_data