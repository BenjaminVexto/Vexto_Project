# tests/test_website_fetchers_unit.py

import pytest
from bs4 import BeautifulSoup
from vexto.scoring.website_fetchers import _parse_basic_seo_from_soup

# --- Fixture ---
HTML_COMPLEX_FIXTURE = """
<!DOCTYPE html>
<html>
<head>
    <title>My Awesome <span>Test</span> Page</title>
    <link rel="canonical" href="https://example.com/canonical-page">
    <script type="application/ld+json">{"@context": "https://schema.org"}</script>
</head>
<body>
    <h1>First H1 Tag</h1>
    <h1>Second H1 Tag</h1>
    <p>Some content here to count words.</p>
</body>
</html>
"""

# --- Tests ---
def test_full_parse_with_complex_fixture():
    soup = BeautifulSoup(HTML_COMPLEX_FIXTURE, 'lxml')
    result = _parse_basic_seo_from_soup(soup)

    # Test felter, der stadig findes i _parse_basic_seo_from_soup
    assert result['title_text'] == "My Awesome Test Page"
    assert result['h1_count'] == 2
    assert result['canonical_url'] == "https://example.com/canonical-page"
    assert result['schema_markup_found'] is True
    # Billed-assertions er fjernet.

def test_parse_with_no_soup_object():
    result = _parse_basic_seo_from_soup(None)
    assert result['title_text'] is None
    assert result['canonical_error'] == "No HTML content to parse"

# --- Alle de gamle 'test_img_alt_pct_...' og schema-tests er fjernet herfra ---
# De hører enten til i en ny test-fil for billeder eller er dækket ovenfor.