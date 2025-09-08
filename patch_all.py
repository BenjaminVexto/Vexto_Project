# patch_all.py
from __future__ import annotations
import re
from pathlib import Path

ROOT = Path(__file__).parent.resolve()
ANALYZER = ROOT / "src" / "vexto" / "scoring" / "analyzer.py"

def read_text(p: Path) -> str:
    return p.read_text(encoding="utf-8")

def write_text(p: Path, s: str) -> None:
    p.write_text(s, encoding="utf-8")

def replace_func_block(text: str, func_name: str, new_src: str) -> tuple[str, bool]:
    """
    Erstat en hel Python-funktionsblok med indholdet i new_src.
    Matcher 'def func_name(...):' frem til næste top-level def/class eller filslut.
    """
    pattern = rf"""
(^[ \t]*def[ \t]+{re.escape(func_name)}[ \t]*\([^)]*\):   # def-linje
[\s\S]*?                                                 # krop (non-greedy)
)(?=^[ \t]*def[ \t]+|^[ \t]*class[ \t]+|\Z)              # til før næste def/class/EOF
"""
    rx = re.compile(pattern, re.MULTILINE | re.VERBOSE)
    if rx.search(text):
        text2 = rx.sub(new_src.rstrip() + "\n\n", text, count=1)
        return text2, True
    return text, False

# --- nye versioner af funktionerne --- #

RESOLVE_CANONICAL_ENHANCED = r'''
def resolve_canonical_enhanced(base_url: str, rendered_html: str, runtime_state: Optional[dict]) -> Tuple[Optional[str], str]:
    """
    Returnerer (canonical_url, source) med sikrere fallback-strategier:
    1) <link rel="canonical"> (DOM)
    2) <meta property="og:url">
    3) Runtime-state SNEVER: kun canonical*-nøgler (canonical, canonical_url, custom_canonical)
       - Hvis base er forsiden ("/"), accepter kun eksakt base eller samme path — ikke menu-links
    4) Self-canonical (sidens egen URL) som sidste udvej
    """
    # 1: DOM canonical
    dom_canonical = _extract_canonical_from_html(rendered_html, base_url)
    if dom_canonical:
        return dom_canonical, "dom_link"

    from bs4 import BeautifulSoup  # sikre import
    from urllib.parse import urljoin, urlparse

    soup = BeautifulSoup(rendered_html or "", "html.parser")

    # 2: OpenGraph url
    og = soup.find("meta", attrs={"property": "og:url"})
    if og and og.get("content"):
        try:
            og_url = og["content"].strip()
            if og_url:
                return og_url, "og_url"
        except Exception:
            pass

    # 3: Runtime – SNEVER søgning i state
    state = runtime_state or _extract_runtime_state_from_html(rendered_html)

    def _is_homepage(u: str) -> bool:
        try:
            pu = urlparse(u)
            return (pu.path or "/").rstrip("/") == ""
        except Exception:
            return False

    if state:
        # Kun canonical-lignende nøgler
        key_patterns = [r"(^|[_\-.])canonical($|[_\-.])", r"(^|[_\-.])canonical_url($|[_\-.])", r"(^|[_\-.])custom_canonical($|[_\-.])"]
        for candidate, path_str in _search_nested_urls(state, key_patterns=key_patterns, include_non_http=False, yield_paths=True):
            if not candidate:
                continue
            href = str(candidate).strip()
            if not href or href.lower() in ("none", "null", "false"):
                continue

            # Normaliser til absolut
            if not href.startswith("http"):
                href = "/" + href.strip("/")
                href = urljoin(base_url, href)

            # Homepage-safeguard: hvis base er forside, accepter ikke en helt anden kategori
            try:
                p_base = urlparse(base_url)
                p_href = urlparse(href)
                same_host = (p_base.netloc == p_href.netloc) or (p_href.netloc == "")

                if same_host and _is_homepage(base_url):
                    # tillad kun samme path som base (dvs. tom eller "/")
                    if (p_href.path or "/").rstrip("/") not in ("", "/"):
                        # skip menu/andre paths
                        continue
            except Exception:
                # ved parsing-fejl: fortsæt, men uden det stramme match
                pass

            # Debug log (ikke kritisk for drift)
            try:
                logging.getLogger(__name__).info("🎯 Runtime canonical (narrow) match via %s = %s", path_str, href)
            except Exception:
                pass

            return href, "runtime_broad"

    # 4: Self canonical fallback
    return base_url, "self_canonical"
'''

SEARCH_NESTED_URLS = r'''
def _search_nested_urls(obj, key_patterns=None, include_non_http=True, yield_paths=False, _path=()):
    """
    Sikker, rekursiv søgning i indlejrede dict/list-strukturer efter URL-værdier.
    - key_patterns: liste af regex-strenge der matche *nøgle-navne* (ikke værdier).
      Hvis None => ingen filtrering på nøgle-navn.
    - include_non_http: medtag relative eller path-only værdier.
    - yield_paths: hvis True, yield'er (value, path_str) i stedet for bare value.
    - Alle nøgler/indekser konverteres til str i path for at undgå 'unhashable type: slice'.
    """
    import re as _re
    from urllib.parse import urlparse

    def _path_to_str(parts):
        safe = []
        for p in parts:
            try:
                safe.append(str(p))
            except Exception:
                safe.append(repr(p))
        return ".".join(safe)

    kp = None
    if key_patterns:
        kp = [_re.compile(k, _re.IGNORECASE) for k in key_patterns]

    # Dict
    if isinstance(obj, dict):
        for k, v in obj.items():
            sk = str(k)
            # Hvis key-patterns er angivet, filtrer på nøgle
            key_ok = True
            if kp:
                key_ok = any(r.search(sk) for r in kp)

            # Besøg altid rekursivt (så vi kan finde match dybere nede)
            # men kun tage værdien her, hvis key_ok
            if key_ok and isinstance(v, (str, int, float)):
                s = str(v).strip()
                if s:
                    if include_non_http or s.startswith("http") or s.startswith("/"):
                        if yield_paths:
                            yield s, _path_to_str(_path + (sk,))
                        else:
                            yield s

            # Recurse
            yield from _search_nested_urls(v, key_patterns=key_patterns, include_non_http=include_non_http, yield_paths=yield_paths, _path=_path + (sk,))

    # List/Tuple
    elif isinstance(obj, (list, tuple)):
        for idx, v in enumerate(obj):
            sidx = str(idx)
            yield from _search_nested_urls(v, key_patterns=key_patterns, include_non_http=include_non_http, yield_paths=yield_paths, _path=_path + (sidx,))

    # Andre typer ignoreres
    else:
        return
'''

def patch_analyzer(p: Path) -> bool:
    text = read_text(p)
    changed = False

    # 1) Erstat resolve_canonical_enhanced
    new_text, ch = replace_func_block(text, "resolve_canonical_enhanced", RESOLVE_CANONICAL_ENHANCED)
    if ch:
        text, changed = new_text, True

    # 2) Erstat/tilføj _search_nested_urls (sikker version)
    if re.search(r"^[ \t]*def[ \t]+_search_nested_urls[ \t]*\(", text, re.MULTILINE):
        new_text, ch = replace_func_block(text, "_search_nested_urls", SEARCH_NESTED_URLS)
        if ch:
            text, changed = new_text, True
    else:
        # Hvis funktionen ikke findes, injicerer vi den én gang i bunden
        text = text.rstrip() + "\n\n" + SEARCH_NESTED_URLS.strip() + "\n"
        changed = True

    if changed:
        write_text(p, text)
    return changed

def main():
    print(f"→ Patcher {ANALYZER}")
    if not ANALYZER.exists():
        print("  ✗ analyzer.py blev ikke fundet")
        return
    if patch_analyzer(ANALYZER):
        print("  ✓ analyzer.py patched")
    else:
        print("  (ingen ændringer)")

if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)
    main()
