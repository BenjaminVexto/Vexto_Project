# tests/test_simple_cleanup.py
import pytest

@pytest.mark.asyncio
async def test_client_is_created_and_yielded(client):
    """
    Denne test gør intet andet end at modtage 'client' fixturen.
    Formålet er at tjekke, om pytest kan rydde op efter sig selv
    uden at crashe med "Event loop is closed".
    """
    assert client is not None