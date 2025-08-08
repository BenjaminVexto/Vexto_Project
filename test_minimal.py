from bs4 import BeautifulSoup
from urllib.parse import urlparse
from typing import List, Set, Dict

# ------------------------------------------------------------------
# VI KOPIERER DEN CENTRALE LOGIK DIREKTE IND HER
# ------------------------------------------------------------------

SOCIAL_MEDIA_DOMAINS = [
    "facebook.com", "instagram.com", "linkedin.com", "twitter.com",
    "x.com", "youtube.com", "tiktok.com"
]

def find_social_media_links(soup: BeautifulSoup) -> Dict[str, List[str]]:
    if not soup:
        return {'social_media_links': []}

    found_links: Set[str] = set()
    for a_tag in soup.find_all('a', href=True):
        href = a_tag.get('href', '').strip()
        if not href.startswith("http"):
            continue

        try:
            netloc = urlparse(href).netloc.lower()
            if netloc.startswith("www."):
                netloc = netloc[4:]

            if any(social_domain == netloc for social_domain in SOCIAL_MEDIA_DOMAINS):
                found_links.add(href)
        except Exception:
            continue
    
    return {'social_media_links': sorted(list(found_links))}

# ------------------------------------------------------------------
# VI DEFINERER EN KENDT, PERFEKT HTML-STUMP
# ------------------------------------------------------------------

TEST_HTML = """
<html>
<body>
    <h1>Test Side</h1>
    <p>Noget tekst her.</p>
    <footer>
        <a href="https://www.facebook.com/testprofil">Facebook</a>
        <a href="https://www.instagram.com/testprofil">Instagram</a>
        <a href="https://www.dr.dk">Et link til DR</a>
        <a href="mailto:test@test.dk">Email</a>
        <a href="https://linkedin.com/in/testprofil">LinkedIn</a>
    </footer>
</body>
</html>
"""

# ------------------------------------------------------------------
# SELVE TESTEN: KØR LOGIKKEN PÅ VORES PERFEKTE DATA
# ------------------------------------------------------------------

print("--- Starter fundamental test ---")

# 1. Opret "soup" fra vores test-HTML
test_soup = BeautifulSoup(TEST_HTML, "lxml")

# 2. Kald funktionen, vi vil teste
resultat = find_social_media_links(test_soup)

# 3. Print resultatet
print("Funktionen fandt følgende sociale links:")
print(resultat)

print("--- Testen er slut ---")