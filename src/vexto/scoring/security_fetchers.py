# src/vexto/scoring/security_fetchers.py

import logging
from .schemas import SecurityMetrics
from .http_client import AsyncHtmlClient

log = logging.getLogger(__name__)

async def fetch_security_headers(client: AsyncHtmlClient, url: str) -> SecurityMetrics:
    """
    Checks for the presence of important security headers in the HTTP response.
    """
    security_data: SecurityMetrics = {
        'hsts_enabled': False,
        'csp_enabled': False,
        'x_content_type_options_enabled': False,
        'x_frame_options_enabled': False,
    }

    try:
        # A HEAD request is efficient as we only need the headers, not the body.
        response = await client.head(url)
        if not response:
            log.warning(f"Could not get a response for security headers check on {url}")
            return security_data

        # Headers are case-insensitive, so we work with a lowercased version.
        headers = {k.lower(): v for k, v in response.headers.items()}

        security_data['hsts_enabled'] = 'strict-transport-security' in headers
        security_data['csp_enabled'] = 'content-security-policy' in headers
        security_data['x_content_type_options_enabled'] = headers.get('x-content-type-options', '').lower() == 'nosniff'
        security_data['x_frame_options_enabled'] = 'x-frame-options' in headers

    except Exception as e:
        log.error(f"Error fetching security headers for {url}: {e}", exc_info=True)

    return security_data