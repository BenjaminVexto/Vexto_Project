# src/vexto/scoring/content_fetcher.py

import logging
import datetime
import asyncio
import re
import json
from collections import Counter
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse  # +urlparse for internal link count
from typing import Optional, Dict, Any
from .schemas import ContentMetrics
from .http_client import AsyncHtmlClient

log = logging.getLogger(__name__)

# -------------------- Global konfiguration/consts --------------------

STOP_WORDS = {
    # dansk + engelsk (uddrag)
    "og","i","på","for","der","det","den","til","er","som","med","af","en","et","at",
    "this","that","with","from","about","your","our","you","we","the","and","for"
}

CTA_WORDS = {
    'da': ['køb','bestil','kontakt','pris','tilbud','gratis','book','ring','skriv','få tilbud'],
    'en': ['buy','order','contact','price','offer','free','book','call','email','get quote']
}

UNIVERSAL_CONVERSION_KWS = {
    "køb","bestil","kontakt","pris","tilbud","gratis","levering","garanti",
    "buy","order","contact","price","offer","free","delivery","warranty"
}

# -------------------- Helpers (globale) --------------------

def _extract_cta_score(soup: BeautifulSoup) -> tuple[int, list[str]]:
    if not soup:
        return 0, []
    candidates = []
    candidates += [a.get_text(" ") for a in soup.find_all("a")]
    candidates += [b.get_text(" ") for b in soup.find_all(["button","input"])]
    text = " ".join(candidates).lower()

    hits = {w for w in (CTA_WORDS['da'] + CTA_WORDS['en']) if w in text}
    # simple scoring: 0=ingen, 50=nogle, 100=mange
    score = 0 if not hits else 50 if len(hits) <= 2 else 100
    return score, sorted(hits)

def _extract_weighted_text(soup: BeautifulSoup) -> str:
    if not soup:
        return ""
    out = []
    # Titel (vægt 3)
    t = soup.find("title")
    if t and t.get_text(strip=True):
        out.append((" " + t.get_text(" ", strip=True) + " ") * 3)
    # Meta description (vægt 2)
    md = soup.find("meta", attrs={"name": "description"})
    if md and (md.get("content") or "").strip():
        out.append((" " + md.get("content").strip() + " ") * 2)
    # H1-H3 (vægt 1)
    for h in soup.find_all(["h1", "h2", "h3"]):
        out.append(" " + h.get_text(" ", strip=True) + " ")
    return " ".join(out)

def _top_keywords_from_pages(soups: list[str], top_n: int = 10) -> list[str]:
    text = " ".join(_extract_weighted_text(BeautifulSoup(html, "lxml")) for html in soups if html)
    words = re.findall(r"[a-zA-ZæøåÆØÅ]{4,}", text.lower())
    words = [w for w in words if w not in STOP_WORDS]
    freq = Counter(words)
    return [w for (w, _) in freq.most_common(top_n)]

def _score_keyword_distribution(sample_text: str, keywords: list[str]) -> float:
    if not sample_text or not keywords:
        return 0.0
    sample = sample_text.lower()
    scores = []
    for kw in keywords:
        c = sample.count(kw)
        if c == 0:
            scores.append(0.0)
        elif c <= 3:
            scores.append(0.8)
        elif c <= 6:
            scores.append(1.0)
        elif c <= 10:
            scores.append(0.9)
        else:
            scores.append(0.7)  # mild stuffing-penalty
    return round(sum(scores) / len(keywords), 3)

# -------------------- Hovedfunktion --------------------

async def fetch_content_data(client: AsyncHtmlClient, soup: BeautifulSoup, base_url: str) -> dict:
    """
    Indholds-metrics:
      - days_since_last_post (nyheder/blog/produkt-opdateringer)
      - average_word_count (gennemsnit af besøgte sider)
      - keyword_relevance_score (dynamisk, domæne-agnostisk)
      - keywords_in_content (top-ord -> forekomst på sample-siden)
      - latest_post_date (ISO)
      - cta_score + cta_analysis
    """
    import datetime as dt  # lokal alias
    # Dansk dato-understøttelse
    DA_MONTHS = {
        "jan":1,"feb":2,"mar":3,"apr":4,"maj":5,"jun":6,"jul":7,"aug":8,"sep":9,"okt":10,"nov":11,"dec":12,
        "januar":1,"februar":2,"marts":3,"april":4,"juni":6,"juli":7,"august":8,"september":9,"oktober":10,"november":11,"december":12
    }
    DATE_REGEXES = [
        # Opdateret den/d. 16. feb. 2022 / 16. feb 2022 / 16. februar 2022
        re.compile(r"(?:opdateret\s+(?:den|d\.)\s+)?(\d{1,2})\.\s*([a-zæøå]{3,9})\.?\s*(\d{4})", re.I),
        # ISO fallback YYYY-MM-DD
        re.compile(r"\b(\d{4})-(\d{2})-(\d{2})\b")
    ]

    def _parse_date_string(txt: str) -> dt.datetime | None:
        if not txt:
            return None
        txt = txt.strip()
        # 1) ISO direkte
        try:
            return dt.datetime.fromisoformat(txt.replace("Z", "+00:00"))
        except Exception:
            pass
        # 2) email/HTTP-datoer
        try:
            from email.utils import parsedate_to_datetime
            d = parsedate_to_datetime(txt)
            if d:
                return d
        except Exception:
            pass
        # 3) danske mønstre
        for rx in DATE_REGEXES:
            m = rx.search(txt)
            if not m:
                continue
            if rx is DATE_REGEXES[1]:
                # ISO yyyy-mm-dd
                y, mm, dd = int(m.group(1)), int(m.group(2)), int(m.group(3))
                return dt.datetime(y, mm, dd, tzinfo=dt.timezone.utc)
            else:
                dd = int(m.group(1))
                mon_raw = m.group(2).lower()
                yy = int(m.group(3))
                mm = DA_MONTHS.get(mon_raw)
                if mm:
                    return dt.datetime(yy, mm, dd, tzinfo=dt.timezone.utc)
        return None

    if not soup:
        return {
            'days_since_last_post': None,
            'keyword_relevance_score': 0.0,
            'average_word_count': 0,
            'latest_post_date': None,
            'keywords_in_content': {},
            'cta_score': 0,
            'cta_analysis': [],
        }
    
    def _has_article_signal(s: BeautifulSoup, url: str) -> bool:
        """
        Returnerer True, hvis siden ligner redaktionelt indhold (artikel/nyhed),
        ikke en kampagne/tilbudsavis.
        """
        try:
            p = urlparse(url).path.lower()
        except Exception:
            p = url.lower()

        # Path-baseret inkl./ekskl.
        INCLUDE = ('/nyheder', '/nyheder/artikler', '/blog', '/inspiration')
        EXCLUDE = ('/tilbudsavis', '/tilbud', '/kampagne', '/udsalg')

        if not any(seg in p for seg in INCLUDE):
            return False
        if any(seg in p for seg in EXCLUDE):
            return False

        # 1) <article> tag
        if s.find('article'):
            return True

        # 2) og:type = article
        og_type = s.find('meta', attrs={'property': 'og:type'})
        if og_type and str(og_type.get('content', '')).lower() == 'article':
            return True

        # 3) JSON-LD med Article/BlogPosting
        for tag in s.find_all('script', attrs={'type': 'application/ld+json'}):
            try:
                data = json.loads(tag.string or '')  # kan være dict eller liste
                nodes = data if isinstance(data, list) else [data]
                for n in nodes:
                    t = (n.get('@type') if isinstance(n, dict) else None)
                    if not t:
                        continue
                    if isinstance(t, list):
                        t = [str(x).lower() for x in t]
                        if any(x in ('article', 'blogposting') for x in t):
                            return True
                    else:
                        if str(t).lower() in ('article', 'blogposting'):
                            return True
            except Exception:
                continue

        return False


    # ---------------- 1) Find kandidatsider (KUN redaktionelt indhold) ----------------
    content_links: list[str] = []
    for a in soup.find_all('a', href=True):
        href = str(a.get('href', '')).strip()
        if not href:
            continue
        low = href.lower()
        if low.startswith(('mailto:', 'tel:', 'javascript:', 'callto:', '#')):
            continue
        abs_url = urljoin(base_url, href)
        path = urlparse(abs_url).path.lower()

        # Kun potentielle nyheds-/blog-oversigter eller artikler
        if any(seg in path for seg in ('/nyheder', '/nyheder/artikler', '/blog', '/inspiration')):
            # Udeluk “tilbud/avis/kampagne/udsalg”
            if any(seg in path for seg in ('/tilbudsavis', '/tilbud', '/kampagne', '/udsalg')):
                continue
            content_links.append(abs_url)

    # dedup + fallback
    seen = set()
    deduped = []
    for u in content_links:
        if u not in seen:
            seen.add(u)
            deduped.append(u)
    content_links = deduped[:5] if deduped else [base_url]  # mindst forsiden

    # ---------------- 2) Crawl undersider (async) og udtræk dato/tekst ----------------
    latest_date: dt.datetime | None = None
    total_words = 0
    page_count = 0
    pages_html: list[str] = []   # til keyword-korpus

    async def scrape_page(url: str):
        try:
            raw = await client.get_raw_html(url)
            if isinstance(raw, dict):
                html = raw.get('html')
            elif isinstance(raw, tuple):
                html = raw[0] if raw else None
            elif isinstance(raw, str):
                html = raw
            else:
                html = None
            if not html:
                return None, 0, None

            s = BeautifulSoup(html, "lxml")
            pages_html.append(html)

            # Kun brug dato fra sider med redaktionelt signal
            is_editorial = _has_article_signal(s, url)
            # ordtælling uanset (til gennemsnit)
            text_content = s.get_text(" ", strip=True)
            words = len(text_content.split())

            if not is_editorial:
                return None, words, html

            # --- primære dato-signaler (som før) ---
            date_str = None
            for container in s.find_all(['article', 'div'], attrs={'class': ['post', 'blog-post', 'news-item', 'update', 'article', 'entry']}):
                time_tag = container.find(['time', 'span', 'div'], attrs={'datetime': True})
                if time_tag and time_tag.get('datetime'):
                    date_str = time_tag.get('datetime'); break
                date_el = container.find(['time', 'span', 'div'], attrs={'class': ['date', 'published', 'post-date', 'entry-date']})
                if date_el:
                    date_str = date_el.get_text(strip=True); break

            if not date_str:
                meta_date = (s.find('meta', attrs={'property': 'article:published_time'}) or
                            s.find('meta', attrs={'name': 'article:published_time'}) or
                            s.find('meta', attrs={'property': 'article:modified_time'}) or
                            s.find('meta', attrs={'name': 'article:modified_time'}))
                if meta_date and meta_date.get('content'):
                    date_str = meta_date['content']

            if not date_str:
                og_date = (s.find('meta', attrs={'property': 'og:updated_time'}) or
                        s.find('meta', attrs={'property': 'og:published_time'}))
                if og_date and og_date.get('content'):
                    date_str = og_date.get('content')

            # Tekstlig fallback – KUN for editoriale sider
            if not date_str:
                body_text = s.get_text(" ", strip=True)[:200000]

                # 1) <time datetime="..."> globalt på siden (oversigtssider har ofte mange)
                best_dt = None
                for t in s.find_all("time"):
                    dt_attr = t.get("datetime")
                    if dt_attr:
                        cand = _parse_date_string(dt_attr)
                        if cand and (best_dt is None or cand > best_dt):
                            best_dt = cand

                # 2) Tekstlige mønstre – saml ALLE matches fra alle regex’er
                if best_dt is None:
                    candidates = []
                    for rx in DATE_REGEXES:
                        for m in rx.finditer(body_text):
                            candidates.append(m.group(0))
                    for c in candidates:
                        cand = _parse_date_string(c)
                        if cand and (best_dt is None or cand > best_dt):
                            best_dt = cand

                if best_dt:
                    return best_dt, words, html  # returnér direkte med den nyeste dato

            page_date = _parse_date_string(date_str) if date_str else None
            return page_date, words, html

        except Exception:
            return None, 0, None


    tasks = [scrape_page(u) for u in content_links]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    for (url, res) in zip(content_links, results):
        if isinstance(res, Exception) or not isinstance(res, tuple) or len(res) != 3:
            continue
        page_date, words, html = res
        if page_date and (latest_date is None or page_date > latest_date):
            latest_date = page_date
        if words > 0:
            total_words += words
            page_count += 1

    # medtag forsiden i korpus for dynamiske keywords
    try:
        pages_html.append(str(soup))
    except Exception:
        pass

    # ---------------- 3) Dynamiske keywords + score ----------------
    top_keywords = _top_keywords_from_pages(pages_html, top_n=10)

    # sample = forsiden (den soup vi fik ind)
    sample_text = soup.get_text(" ", strip=True)[:200000].lower()
    keywords_in_content = {kw: sample_text.count(kw) for kw in top_keywords}
    keyword_relevance_score = _score_keyword_distribution(sample_text, top_keywords)

    # lille bonus for universelle konverteringsord
    if any(kw in sample_text for kw in UNIVERSAL_CONVERSION_KWS):
        keyword_relevance_score = min(1.0, keyword_relevance_score + 0.1)

    # 3b) CTA
    cta_score, cta_terms = _extract_cta_score(soup)

    # ---------------- 4) Final metrics ----------------
    days_since = None
    if latest_date:
        try:
            now = dt.datetime.now(dt.timezone.utc)
            d = latest_date if latest_date.tzinfo else latest_date.replace(tzinfo=dt.timezone.utc)
            days_since = (now - d).days
        except Exception:
            days_since = None

    avg_words = total_words // page_count if page_count > 0 else 0

    return {
        'days_since_last_post': days_since,
        'keyword_relevance_score': keyword_relevance_score,
        'keywords_in_content': keywords_in_content,
        'average_word_count': avg_words,
        'latest_post_date': latest_date.isoformat() if latest_date else None,
        'cta_score': cta_score,
        'cta_analysis': cta_terms,
    }
