# src/vexto/scoring/authority_fetcher.py

import os, httpx, logging
from urllib.parse import urlparse
from typing import Optional
from tenacity import retry, stop_after_attempt, wait_exponential  # Ny import for retry

from .http_client import AsyncHtmlClient, html_cache
from .schemas import AuthorityMetrics

log = logging.getLogger(__name__)

API_URL = "https://openpagerank.com/api/v1.0/getPageRank"

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=5), reraise=True)
async def get_authority(client: AsyncHtmlClient, url: str) -> Optional[AuthorityMetrics]:
    opr_key = os.getenv("OPENPAGERANK_API_KEY")
    if not opr_key:
        log.warning("OPENPAGERANK_API_KEY mangler – bruger stub.")
        return {
            "domain_authority": None,
            "page_authority": None,
            "global_rank": None,
            "authority_status": "missing_api_key",
        }

    domain = urlparse(url).hostname
    if not domain:
        log.warning(f"Could not extract domain from URL: {url} for OpenPageRank.")
        return {
            "domain_authority": None,
            "page_authority": None,
            "global_rank": None,
            "authority_status": "invalid_url",
        }
    # normalize domain (avoid www.)
    if domain.startswith("www."):
        domain = domain[4:]

    # --- Caching for Open PageRank ---
    cache_key = f"opr_{domain}"
    if (cached_data := html_cache.get(cache_key)) is not None:
        log.info(f"CACHE HIT: Open PageRank for {domain}")
        return cached_data
    # --- End Caching ---

    params  = {"domains[]": domain}
    headers = {"API-OPR": opr_key}

    try:
        response = await client.httpx_get(API_URL, params=params, headers=headers, timeout=15)
        response.raise_for_status()

        try:
            json_response = response.json()
        except ValueError:
            log.warning(f"Open PageRank API returned invalid JSON for {domain}.")
            result: AuthorityMetrics = {
                "domain_authority": None,
                "page_authority": None,
                "global_rank": None,
                "authority_status": "invalid_json_openpagerank",
            }
            html_cache.set(cache_key, result, expire=86_400)
            return result

        if not json_response or not json_response.get("response"):
            log.warning(f"Open PageRank API returned empty/malformed response for {domain}.")
            result: AuthorityMetrics = {
                "domain_authority": None,
                "page_authority": None,
                "global_rank": None,
                "authority_status": "no_data_openpagerank",
            }
            html_cache.set(cache_key, result, expire=86_400)
            return result

        data_list = json_response.get("response")
        if not data_list:
            log.warning(f"Open PageRank API returned no data for {domain}.")
            result: AuthorityMetrics = {
                "domain_authority": None,
                "page_authority": None,
                "global_rank": None,
                "authority_status": "no_data_openpagerank",
            }
            html_cache.set(cache_key, result, expire=86_400)
            return result

        data = data_list[0]

        page_rank = data.get("page_rank_integer")
        global_rank = data.get("rank")
        domain_authority_opr = data.get("domain_authority")

        # Fallback: afled "domain_authority" groft fra page_rank (0-10 → 0-100)
        if domain_authority_opr is None and page_rank is not None:
            try:
                domain_authority_opr = float(page_rank) * 10.0
            except Exception:
                domain_authority_opr = None

        authority_status = "ok_openpagerank"
        if page_rank is None and global_rank is None and domain_authority_opr is None:
            authority_status = "no_data_openpagerank"

        result: AuthorityMetrics = {
            "domain_authority": float(domain_authority_opr) if domain_authority_opr is not None else None,
            "page_authority": float(page_rank) if page_rank is not None else None,
            "global_rank": global_rank,
            "authority_status": authority_status,
        }

        html_cache.set(cache_key, result, expire=86_400)
        return result

    except httpx.RequestError as e:
        log.warning(f"Open PageRank API request failed for {domain}: {e.__class__.__name__}", exc_info=True)
        return {
            "domain_authority": None,
            "page_authority": None,
            "global_rank": None,
            "authority_status": "network_error",
        }
    except httpx.HTTPStatusError as e:
        log.warning(f"Open PageRank API HTTP error for {domain}: {e.response.status_code}", exc_info=True)
        return {
            "domain_authority": None,
            "page_authority": None,
            "global_rank": None,
            "authority_status": f"http_error_{e.response.status_code}",
        }
    except IndexError:
        log.warning(f"Open PageRank API returned empty response list for {domain}.", exc_info=True)
        return {
            "domain_authority": None,
            "page_authority": None,
            "global_rank": None,
            "authority_status": "no_data_openpagerank",
        }
    except Exception as e:
        log.error(f"Unexpected error parsing Open PageRank response for {domain}: {e}", exc_info=True)
        return {
            "domain_authority": None,
            "page_authority": None,
            "global_rank": None,
            "authority_status": "parse_error",
        }
