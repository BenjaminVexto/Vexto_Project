# src/vexto/scoring/form_fetcher.py
import logging
from bs4 import BeautifulSoup
from .schemas import ConversionMetrics

log = logging.getLogger(__name__)

def analyze_forms(soup: BeautifulSoup) -> ConversionMetrics:
    """
    Finder alle formularer på siden og tæller antallet af input-felter i hver.
    """
    if not soup:
        return {'form_field_counts': []}

    try:
        forms = soup.find_all('form')
        field_counts = []

        for form in forms:
            # Tæl de mest almindelige typer af input-felter
            inputs = form.find_all(['input', 'textarea', 'select'])
            # Fjerner skjulte felter, submit-knapper etc. fra tællingen
            visible_fields = [
                field for field in inputs 
                if field.get('type') not in ('hidden', 'submit', 'reset', 'button')
            ]
            field_counts.append(len(visible_fields))

        return {'form_field_counts': field_counts}

    except Exception as e:
        log.error(f"Fejl under analyse af formularer: {e}", exc_info=True)
        return {'form_field_counts': []}