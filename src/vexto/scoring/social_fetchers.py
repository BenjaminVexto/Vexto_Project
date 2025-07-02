# src/vexto/scoring/social_fetchers.py
import logging
from bs4 import BeautifulSoup
from .schemas import SocialAndReputationMetrics

log = logging.getLogger(__name__)

# Domæner vi leder efter i links
SOCIAL_MEDIA_DOMAINS = [
    'facebook.com',
    'instagram.com',
    'linkedin.com',
    'twitter.com',
    'x.com', # Twitter's nye domæne
    'youtube.com',
    'tiktok.com'
]

def find_social_media_links(soup: BeautifulSoup) -> SocialAndReputationMetrics:
    """
    Analyserer et BeautifulSoup-objekt for at finde udgående links til sociale medier.
    """
    if not soup:
        return {'social_media_links': []}

    found_links = set() # Brug et set for at undgå duplikater
    
    try:
        links = soup.find_all('a', href=True)
        for link in links:
            href = link['href'].lower()
            for domain in SOCIAL_MEDIA_DOMAINS:
                if domain in href:
                    found_links.add(link['href']) # Tilføj den originale URL
                    break # Gå til næste link, når et match er fundet

        social_data: SocialAndReputationMetrics = {
            'social_media_links': sorted(list(found_links)) # Returner en sorteret liste
        }
        return social_data

    except Exception as e:
        log.error(f"Fejl under detektion af SoMe-links: {e}", exc_info=True)
        return {'social_media_links': []}
