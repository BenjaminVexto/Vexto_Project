# tests/conftest.py
import pytest
from vexto.scoring.http_client import AsyncHtmlClient

@pytest.fixture
async def client():
    """
    En simpel og robust fixture med function-scope.
    Bruger den opgraderede klient til automatisk at håndtere setup og teardown.
    """
    async with AsyncHtmlClient() as client_instance:
        yield client_instance
    # Oprydning sker nu automatisk og stabilt for hver test.