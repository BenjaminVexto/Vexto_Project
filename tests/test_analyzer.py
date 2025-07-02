# tests/test_analyzer.py (NY FIL)

import pytest
from vexto.scoring.analyzer import analyze_multiple_urls

@pytest.mark.asyncio
@pytest.mark.network  # Marker som en test, der bruger netværket
async def test_analyze_google_com():
    # Test med en liste, da det er den primære use case
    results = await analyze_multiple_urls(["https://www.google.com"])
    
    # Valider at vi fik data tilbage
    assert len(results) == 1
    data = results[0]

    # Tjek data fra begge fetchers
    assert data["url"] == "https://www.google.com"
    assert data["basic_seo"]["title_text"] == "Google"
    assert data["technical_seo"]["is_https"] is True
    assert data["technical_seo"]["status_code"] == 200
    assert data["technical_seo"]["robots_txt_found"] is True