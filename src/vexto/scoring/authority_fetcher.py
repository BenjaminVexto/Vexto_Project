# src/vexto/scoring/authority_fetcher.py
import os, httpx, logging, asyncio
from urllib.parse import urlparse
from typing import Optional
from tenacity import retry, stop_after_attempt, wait_exponential  # Ny import for retry
# Import AsyncHtmlClient and html_cache from http_client
from .http_client import AsyncHtmlClient, html_cache
from .schemas import AuthorityMetrics # Import AuthorityMetrics for type hinting

log = logging.getLogger(__name__)

# OPR_KEY is now retrieved inside the function, not at module level
API_URL = "https://openpagerank.com/api/v1.0/getPageRank"

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=5), reraise=True)  # Ny: Retry decorator med 3 forsøg, exponential wait
async def get_authority(client: AsyncHtmlClient, url: str) -> Optional[AuthorityMetrics]: # Return type Optional[AuthorityMetrics]
    opr_key = os.getenv("OPENPAGERANK_API_KEY") # Get OPR_KEY inside the function
    if not opr_key:
        log.warning("OPENPAGERANK_API_KEY mangler – bruger stub.")
        return None

    domain = urlparse(url).hostname # Extract domain only
    if not domain: # Handle cases where urlparse might fail to get a hostname
        log.warning(f"Could not extract domain from URL: {url} for OpenPageRank.")
        return None

    # --- Caching for Open PageRank ---
    cache_key = f"opr_{domain}"
    if (cached_data := html_cache.get(cache_key)) is not None:
        log.info(f"CACHE HIT: Open PageRank for {domain}")
        return cached_data
    # --- End Caching ---

    params  = {"domains[]": domain}
    headers = {"API-OPR": opr_key}

    try:
        # Use httpx_get from the shared AsyncHtmlClient, which respects global semaphore
        # OPR is generally fast, 15s timeout is fine.
        response = await client.httpx_get(API_URL, params=params, headers=headers, timeout=15)
        
        response.raise_for_status() # Raise for 4xx/5xx if not 404
        json_response = response.json()
        
        if not json_response or not json_response.get("response"):
             log.warning(f"Open PageRank API returned empty/malformed response for {domain}.")
             result: AuthorityMetrics = {"domain_authority": None, "page_authority": None, "global_rank": None, "authority_status": "no_data_openpagerank"}
             html_cache.set(cache_key, result, expire=86_400) # Cache 404s too
             return result

        data_list = json_response.get("response")
        if not data_list: # Check if response list is empty (e.g., if domain not found)
             log.warning(f"Open PageRank API returned no data for {domain}.")
             result: AuthorityMetrics = {"domain_authority": None, "page_authority": None, "global_rank": None, "authority_status": "no_data_openpagerank"}
             html_cache.set(cache_key, result, expire=86_400)
             return result

        data = data_list[0] # Extract the first domain's data
        
        page_rank = data.get("page_rank_integer")
        global_rank = data.get("rank")
        domain_authority_opr = data.get("domain_authority") 

        authority_status = "ok_openpagerank"
        if page_rank is None and global_rank is None and domain_authority_opr is None:
            authority_status = "no_data_openpagerank" # Explicitly no data if all are None

        result: AuthorityMetrics = { # Explicitly type the result dictionary
            "domain_authority": float(domain_authority_opr) if domain_authority_opr is not None else None,
            "page_authority": float(page_rank) if page_rank is not None else None, # Convert to float as per schema
            "global_rank": global_rank,
            "authority_status": authority_status,
        }
        
        html_cache.set(cache_key, result, expire=86_400) # Cache for 24 hours
        return result

    except httpx.RequestError as e: # Catch network errors, timeouts
        log.warning(f"Open PageRank API request failed for {domain}: {e.__class__.__name__}", exc_info=True)
        return {"domain_authority": None, "page_authority": None, "global_rank": None, "authority_status": "network_error"}
    except httpx.HTTPStatusError as e: # Catch non-404 HTTP errors
        log.warning(f"Open PageRank API HTTP error for {domain}: {e.response.status_code}", exc_info=True)
        return {"domain_authority": None, "page_authority": None, "global_rank": None, "authority_status": f"http_error_{e.response.status_code}"}
    except IndexError: # Catches if data_list is empty after all checks (e.g., response was {} or {"response": []})
        log.warning(f"Open PageRank API returned empty response list for {domain}.", exc_info=True)
        return {"domain_authority": None, "page_authority": None, "global_rank": None, "authority_status": "no_data_openpagerank"}
    except Exception as e: # Catch any other parsing or unexpected errors
        log.error(f"Unexpected error parsing Open PageRank response for {domain}: {e}", exc_info=True)
        return {"domain_authority": None, "page_authority": None, "global_rank": None, "authority_status": "parse_error"}