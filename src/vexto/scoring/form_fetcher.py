# src/vexto/scoring/form_fetcher.py
from __future__ import annotations
import re
from typing import Dict, List
from bs4 import BeautifulSoup, Tag

def _is_visible_field(field: Tag) -> bool:
    t = (field.get("type") or "").lower()
    if t in ("hidden", "submit", "reset", "button", "image"):
        return False
    # simple visibility checks
    style = (field.get("style") or "").lower()
    if "display:none" in style or "visibility:hidden" in style:
        return False
    if field.has_attr("hidden") or field.get("aria-hidden") in ("true", True):
        return False
    return True

def _count_visible_fields(container: Tag) -> int:
    if not container:
        return 0
    fields = container.find_all(["input", "textarea", "select"])
    visible = [f for f in fields if _is_visible_field(f)]
    return len(visible)

def analyze_forms_enhanced(soup: BeautifulSoup) -> Dict:
    result = {'form_field_counts': [], 'forms_meta': {'form_count': 0, 'forms': []}}
    if not soup:
        return result

    def _meta_from_container(container) -> dict:
        # simple feature flags
        inputs = container.find_all(['input', 'textarea', 'select'])
        texts  = container.find_all('textarea')
        method = (container.get('method') or '').lower() if container.name == 'form' else None
        has_email = any((i.get('type') or '').lower() == 'email' for i in inputs)
        has_phone = any((i.get('type') or '').lower() in ('tel','phone') for i in inputs)
        has_textarea = len(texts) > 0
        # honeypot heuristik
        has_honeypot = any(
            (i.get('type') or '').lower() == 'text' and
            (i.get('name') or '').lower() in ('website','homepage','url','hp','honeypot')
            for i in inputs
        )
        # newsletter/footer heuristik
        is_newsletter = False
        parent = container
        hop = 0
        while parent is not None and hop < 5:
            if parent.name == 'footer':
                is_newsletter = True
                break
            parent = parent.parent
            hop += 1
        classes = ' '.join((container.get('class') or []))
        if re.search(r'(newsletter|subscribe|tilmeld)', classes, re.I):
            is_newsletter = True

        return {
            'method': method,
            'has_email': has_email,
            'has_phone': has_phone,
            'has_textarea': has_textarea,
            'has_honeypot': has_honeypot,
            'is_newsletter': is_newsletter,
        }

    counts: List[int] = []
    # 1) Ægte <form>
    forms = soup.find_all('form')
    for frm in forms:
        c = _count_visible_fields(frm)
        if not c:
            continue
        meta = _meta_from_container(frm)
        # Ignorér newsletter i footer per default
        if meta['is_newsletter']:
            continue
        counts.append(c)
        result['forms_meta']['forms'].append({
            'type': 'form',
            'input_count': c,
            'id': frm.get('id'),
            'class': frm.get('class'),
            'action': frm.get('action'),
            **meta,
        })

    # 2) Knap-heuristik (uændret, men med meta + newsletter-filter)
    submit_btns = (
        soup.find_all(['button', 'input'], attrs={'type': 'submit'}) +
        soup.find_all('button', string=re.compile(r'(send|submit|køb|tilmeld|book|kontakt)', re.I))
    )
    for btn in submit_btns:
        container = btn.find_parent(['form', 'section', 'article', 'div'])
        hop = 0
        while container and hop < 5:
            c = _count_visible_fields(container)
            if c:
                meta = _meta_from_container(container)
                if not meta['is_newsletter']:
                    counts.append(c)
                    result['forms_meta']['forms'].append({
                        'type': 'btn-heuristic',
                        'input_count': c,
                        'id': container.get('id'),
                        'class': container.get('class'),
                        **meta,
                    })
                break
            container = container.find_parent(['section', 'article', 'div'])
            hop += 1

    # 3) SPA-containere (som nu)
    indicators = [
        {'class_': re.compile(r'(form|kontakt|contact|checkout|subscribe|newsletter)', re.I)},
        {'id': re.compile(r'(form|kontakt|contact|checkout|subscribe|newsletter)', re.I)},
        {'attrs': {'data-form': True}},
        {'attrs': {'role': 'form'}},
        {'class_': re.compile(r'(contact-form|form-container|wpcf7|gravityform|elementor-form|umbraco-forms)', re.I)},
        {'id': re.compile(r'(contact-form|form-container|wpcf7|gravityform|elementor-form|umbraco-forms)', re.I)},
    ]
    for ind in indicators:
        if 'class_' in ind:
            containers = soup.find_all(['div', 'section', 'article'], class_=ind['class_'])
        elif 'id' in ind:
            containers = soup.find_all(['div', 'section', 'article'], id=ind['id'])
        else:
            containers = soup.find_all(['div', 'section', 'article'], **ind['attrs'])
        for ctn in containers:
            c = _count_visible_fields(ctn)
            if c:
                meta = _meta_from_container(ctn)
                if meta['is_newsletter']:
                    continue
                counts.append(c)
                result['forms_meta']['forms'].append({
                    'type': 'spa-container',
                    'input_count': c,
                    'id': ctn.get('id'),
                    'class': ctn.get('class'),
                    **meta,
                })

    # Dedup ±2
    deduped: List[int] = []
    for c in counts:
        if not any(abs(c - d) <= 2 for d in deduped):
            deduped.append(c)

    result['form_field_counts'] = deduped
    result['forms_meta']['form_count'] = len(deduped)
    return result


# Bevar gammel API-navn hvis resten af koden importerer analyze_forms
def analyze_forms(soup: BeautifulSoup) -> Dict:
    return analyze_forms_enhanced(soup)
