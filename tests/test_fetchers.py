# tests/test_fetchers.py

import pytest
from vexto.scoring.website_fetchers import fetch_basic_seo_data
from vexto.scoring.schemas import BasicSEO

# OPDATERET: Billed-relaterede felter er fjernet fra facitlisten.
EXPECTED_EMPTY_BASIC_SEO: BasicSEO = {
    'h1': None,
    'h1_count': None,
    'h1_texts': None,
    'meta_description': None,
    'meta_description_length': None,
    'title_text': None,
    'title_length': None,
    'word_count': None,
    'canonical_url': None,
    'canonical_error': "No HTML content to parse",
    'schema_markup_found': False,
}

@pytest.mark.asyncio
@pytest.mark.network
async def test_fetch_google_com(client):
    """
    End-to-end test against a real, reliable domain, now using the client fixture.
    """
    data = await fetch_basic_seo_data(client, "https://www.google.com")

    assert data is not None
    assert data.get('title_text') == "Google"
    assert data.get('word_count') > 5
    # RETTELSE: Denne assertion er fjernet, da billed-data ikke længere hentes her.
    # assert data.get('image_alt_pct') is not None

@pytest.mark.asyncio
@pytest.mark.network
async def test_fetch_non_existent_domain(client):
    """
    Tests graceful failure for a non-existent domain, using the client fixture.
    """
    data = await fetch_basic_seo_data(client, "https://this-is-not-a-real-domain-xyz.com")
    
    assert data == EXPECTED_EMPTY_BASIC_SEO