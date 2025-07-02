# tests/test_gmb_fetcher.py
import pytest
import respx
from httpx import Response

from vexto.scoring.gmb_fetcher import fetch_gmb_data

# Definer de falske API-svar fra Google
FIND_PLACE_SUCCESS_RESPONSE = {
    "candidates": [{"place_id": "CH_PLACE_ID_123"}],
    "status": "OK"
}
PLACE_DETAILS_SUCCESS_RESPONSE = {
    "result": {
        "rating": 4.5,
        "user_ratings_total": 150
    },
    "status": "OK"
}
FIND_PLACE_NOT_FOUND_RESPONSE = {"candidates": [], "status": "ZERO_RESULTS"}

@pytest.mark.asyncio
@respx.mock
async def test_fetch_gmb_data_happy_path(client, monkeypatch):
    """Tester det perfekte scenarie, hvor alt findes."""
    # Sæt en falsk API nøgle for testen
    monkeypatch.setenv("GOOGLE_API_KEY", "fake_key_for_testing")
    
    # Mock de to API-kald, så de returnerer succesfulde svar
    respx.get(url__regex=r"https://maps.googleapis.com/maps/api/place/findplacefromtext/.*").mock(
        return_value=Response(200, json=FIND_PLACE_SUCCESS_RESPONSE)
    )
    respx.get(url__regex=r"https://maps.googleapis.com/maps/api/place/details/.*").mock(
        return_value=Response(200, json=PLACE_DETAILS_SUCCESS_RESPONSE)
    )

    result = await fetch_gmb_data(client, "https://www.example.com")
    
    assert result['gmb_average_rating'] == 4.5
    assert result['gmb_review_count'] == 150
    assert result['gmb_profile_complete'] is True

@pytest.mark.asyncio
@respx.mock
async def test_fetch_gmb_data_place_not_found(client, monkeypatch):
    """Tester scenariet, hvor Google ikke kan finde en virksomhed med det navn."""
    monkeypatch.setenv("GOOGLE_API_KEY", "fake_key_for_testing")
    
    # Mock kun det første API-kald til at returnere "ikke fundet"
    respx.get(url__regex=r"https://maps.googleapis.com/maps/api/place/findplacefromtext/.*").mock(
        return_value=Response(200, json=FIND_PLACE_NOT_FOUND_RESPONSE)
    )

    result = await fetch_gmb_data(client, "https://nonexistentbusiness.com")
    
    assert result == {}

@pytest.mark.asyncio
async def test_fetch_gmb_no_api_key(client, monkeypatch):
    """Tester, at funktionen returnerer tomt, hvis API-nøglen mangler."""
    # Slet den midlertidigt fra test-miljøet
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    
    result = await fetch_gmb_data(client, "https://www.example.com")

    assert result == {}