# src/vexto/scoring/social_fetchers.py
from urllib.parse import urlparse, urljoin
import logging
from typing import Dict, List, Set
from bs4 import BeautifulSoup
from .schemas import SocialAndReputationMetrics

log = logging.getLogger(__name__)

SOCIAL_MEDIA_DOMAINS = [
    "facebook.com",
    "instagram.com",
    "linkedin.com",
    "twitter.com",
    "x.com",
    "tiktok.com",
    "youtube.com",
]

def find_social_media_links(soup: BeautifulSoup) -> Dict[str, List[str]]:
    if not soup:
        return {'social_media_links': [], 'social_share_links': []}

    profiles, shares = set(), set()

    for a in soup.find_all('a', href=True):
        href = a['href'].strip()
        # Normalize relative or protocol-relative URLs
        if href.startswith('//'):
            href = 'https:' + href
        elif href.startswith('/'):
            href = urljoin('https://example.com', href)  # Replace with base_url in analyzer
        if not href.startswith('http'):
            continue
        try:
            u = urlparse(href)
            host = u.netloc.lower().lstrip('www.')
            match = next((d for d in SOCIAL_MEDIA_DOMAINS if host.endswith(d)), None)
            if not match:
                continue
            # Exclude privacy policy, legal, or terms pages
            if any(term in u.path.lower() for term in ['policy', 'privacy', 'legal', 'terms']):
                log.debug(f"Excluded policy link: {href}")
                continue
            if any(x in u.path.lower() for x in ['share', 'sharer', 'intent']):
                shares.add(href)
                log.info(f"Found social share link: {href}")
            else:
                profiles.add(href)
                log.info(f"Found social profile link: {href}")
        except Exception as e:
            log.debug(f"Error parsing social link {href}: {e}")
            continue

    return {
        'social_media_links': sorted(profiles),
        'social_share_links': sorted(shares)
    }