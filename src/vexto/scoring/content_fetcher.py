import logging
import datetime
import asyncio
from bs4 import BeautifulSoup
from urllib.parse import urljoin
from .schemas import ContentMetrics
from .http_client import AsyncHtmlClient

log = logging.getLogger(__name__)

KEYWORD_LIST = ['kontormøbler', 'kontorstole', 'hæve sænkeborde', 'nyheder', 'blog']  # Tilpasset til inventarland.dk

async def fetch_content_data(client: AsyncHtmlClient, soup: BeautifulSoup, base_url: str) -> dict:
    """
    Scraper for indholds-friskhed, nøgleord og tekstlængde.
    Crawler blog/nyheder eller produkter for datoer, beregner days_since_last_post, average_word_count, og keyword_relevance_score.
    Returnerer dict til ContentMetrics.
    """
    if not soup:
        return {'days_since_last_post': None, 'keyword_relevance_score': 0.0, 'average_word_count': 0}

    try:
        # 1. Find blog/nyheder eller produkt-links (crawl max 5 subpages) – Udvidet søgning
        content_links = []
        for a in soup.find_all('a', href=True):
            href = a['href'].lower()
            link_text = a.get_text(strip=True).lower()
            if any(kw in href or kw in link_text for kw in ['blog', 'news', 'nyheder', 'produkt', 'product', 'tilbud', 'avis']):
                content_links.append(urljoin(base_url, a['href']))
        content_links = list(set(content_links))[:5]  # Max 5 for performance
        if not content_links:
            log.info(f"Ingen blog/nyheder/produkt-links fundet på {base_url}, bruger homepage")
            content_links = [base_url]  # Fallback til homepage

        # 2. Crawl og scrape datoer/tekst – Udvidet date-parsing
        latest_date = None
        total_words = 0
        page_count = 0
        keyword_hits = 0
        total_keywords = len(KEYWORD_LIST) or 1

        async def scrape_page(url: str):
            html = await client.get_raw_html(url)
            if html:
                sub_soup = BeautifulSoup(html, "lxml")
                # Udvidet dato-søgning: Tjek i article/post divs
                date_str = None
                for container in sub_soup.find_all(['article', 'div'], attrs={'class': ['post', 'blog-post', 'news-item', 'product-item', 'update']}):
                    # Separat check for datetime attr
                    time_tag_datetime = container.find(['time', 'span', 'div'], attrs={'datetime': True})
                    if time_tag_datetime:
                        date_str = time_tag_datetime.get('datetime') or time_tag_datetime.text.strip()
                        break
                    
                    # Separat check for class-based date
                    time_tag_class = container.find(['time', 'span', 'div'], attrs={'class': ['date', 'published']})
                    if time_tag_class:
                        date_str = time_tag_class.text.strip()
                        break
                
                # Fallback til meta/og hvis ikke fundet i container
                if not date_str:
                    meta_date = sub_soup.find('meta', attrs={'name': 'article:published_time'})
                    if meta_date and meta_date.get('content'):
                        date_str = meta_date['content']
                if not date_str:
                    og_date = sub_soup.find('meta', attrs={'property': 'og:updated_time'})
                    if og_date and og_date.get('content'):
                        date_str = og_date['content']
                
                words = len(sub_soup.get_text(' ', strip=True).split())
                hits = sum(1 for kw in KEYWORD_LIST if kw in sub_soup.get_text().lower())
                try:
                    page_date = datetime.datetime.fromisoformat(date_str) if date_str else None
                    return page_date, words, hits
                except ValueError:
                    # Forsøg alternative date-formats
                    try:
                        page_date = datetime.datetime.strptime(date_str, '%Y-%m-%d') if date_str else None
                    except ValueError:
                        page_date = None
                    return page_date, words, hits
            return None, 0, 0

        tasks = [scrape_page(link) for link in content_links]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        for result in results:
            if isinstance(result, Exception):
                log.warning(f"Fejl ved crawl af subpage: {result}", exc_info=True)
                continue
            date, words, hits = result
            if date:
                if not latest_date or date > latest_date:
                    latest_date = date
            total_words += words
            keyword_hits += hits
            if words > 0:
                page_count += 1

        # 3. Beregn metrics
        days_since = (datetime.datetime.now() - latest_date).days if latest_date else None
        avg_words = total_words // page_count if page_count > 0 else 0
        relevance = keyword_hits / (total_keywords * page_count) if page_count > 0 else 0.0

        return {
            'days_since_last_post': days_since,
            'keyword_relevance_score': min(relevance, 1.0),
            'average_word_count': avg_words
        }

    except Exception as e:
        log.error(f"Fejl under content-scraping: {e}", exc_info=True)
        return {'days_since_last_post': None, 'keyword_relevance_score': 0.0, 'average_word_count': 0}