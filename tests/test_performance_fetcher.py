# tests/test_performance_fetcher.py

import os
import json
import pytest
from pathlib import Path

# -- KIRURGISK INDGREB: Importerer nu det korrekte funktionsnavn --
from vexto.scoring.performance_fetcher import get_performance_from_stub, get_performance, _parse_psi_json
from vexto.scoring.http_client import AsyncHtmlClient


def test_parser_with_live_google_fixture():
    """
    Tester parseren direkte mod et gemt, realistisk JSON-svar.
    Dette bekræfter, at vores nøgler og fallback-logik er korrekt.
    """
    fixture_path = Path(__file__).parent / "fixtures" / "psi_google_live.json"
    with open(fixture_path, 'r') as f:
        live_data = json.load(f)

    parsed_data = _parse_psi_json(live_data)

    assert parsed_data is not None
    assert parsed_data["performance_score"] == 98
    assert parsed_data["lcp_ms"] == 1850.5
    # Bekræft at INP-fallback til Total Blocking Time virkede
    assert parsed_data["inp_ms"] == 120
    # Bekræft at det nye mobil-tjek virker
    assert parsed_data["viewport_score"] == 1

def test_get_performance_from_stub():
    """Tester at stub-funktionen kan læse og parse den gamle stub-fil korrekt."""
    # -- KIRURGISK INDGREB: Kalder nu det korrekte funktionsnavn --
    data = get_performance_from_stub()
    
    assert data is not None
    assert data["performance_score"] == 92
    assert data["viewport_score"] == 1
    assert "lcp_ms" in data

@pytest.mark.network
@pytest.mark.asyncio
async def test_get_performance_live():
    """Laver et rigtigt API-kald til PSI."""
    if not os.getenv("GOOGLE_API_KEY"):
        pytest.skip("GOOGLE_API_KEY er ikke sat, skipper live-test.")

    client = AsyncHtmlClient()
    try:
        data = await get_performance(client, "https://www.google.com")
        assert data is not None
        assert "performance_score" in data
        assert 0 <= data["performance_score"] <= 100
        assert data["lcp_ms"] is not None
    finally:
        await client.close()