import pytest
from vexto.scoring.website_fetchers import fetch_basic_seo_data
from vexto.scoring.schemas import BasicSEO

# Forventet tomt resultat ved fejl
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
    End-to-end test against a real, stabil domæne, nu med korrekt tuple-unpacking.
    """
    basic, _ = await fetch_basic_seo_data(client, "https://www.google.com")

    assert basic is not None
    assert basic.get('title_text') == "Google"
    assert basic.get('word_count') is not None and basic.get('word_count') > 5


@pytest.mark.asyncio
@pytest.mark.network
async def test_fetch_non_existent_domain(client):
    """
    Tester robust fallback ved ugyldigt domæne, med tuple-unpacking.
    """
    basic, _ = await fetch_basic_seo_data(client, "https://this-is-not-a-real-domain-xyz.com")

    assert basic == EXPECTED_EMPTY_BASIC_SEO
