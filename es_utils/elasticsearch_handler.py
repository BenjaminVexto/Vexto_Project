"""
Elasticsearch handler
---------------------
search_companies(postnumre, branchekode, size=10_000)
    ➜ returnerer liste af dicts (flade felter) – klar til at lægge i en DataFrame.
"""
from typing import List
import os
from pathlib import Path

from elasticsearch import Elasticsearch
from dotenv import load_dotenv

load_dotenv()  # indlæs evt. .env med ES_HOST, ES_APIKEY

ES_HOST   = os.getenv("ES_HOST", "http://localhost:9200")
ES_APIKEY = os.getenv("ES_APIKEY")

# Index-navn kan justeres her, men navnet **må ikke** ændres andre steder,
# da main.py forventer det samme.
ES_INDEX  = os.getenv("ES_INDEX", "vexto-cvr")


def _connect() -> Elasticsearch:
    """Opret ES-klient – uanset om API-key eller basic-auth benyttes."""
    if ES_APIKEY:
        return Elasticsearch(ES_HOST, api_key=ES_APIKEY, verify_certs=False)
    # Fallback til no-auth (localhost)
    return Elasticsearch(ES_HOST, verify_certs=False)


def search_companies(postnumre: List[int], branchekode: str, size: int = 10_000) -> List[dict]:
    """
    Returnerer en liste af hits (dict).
    Felt-strukturen flades ud ved _source-niveau => direkte til DataFrame.
    """
    es = _connect()

    query_body = {
        "size": size,
        "query": {
            "bool": {
                "must": [
                    {"term":   {"branchekode": branchekode}},
                    {"terms":  {"postnummer":  postnumre}}
                ]
            }
        }
    }

    resp = es.search(index=ES_INDEX, body=query_body)
    hits = resp.get("hits", {}).get("hits", [])

    # Fladgør resultater:  {"_id": ...,  ..._source}
    flat = []
    for h in hits:
        row = {"_id": h["_id"]}
        row.update(h.get("_source", {}))
        flat.append(row)

    return flat
