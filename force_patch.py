import re
from pathlib import Path
from textwrap import dedent

ANALYZER = Path("src/vexto/scoring/analyzer.py")

NEW_SEARCH_NARROW = r'''
def _search_nested_urls(obj, key_patterns=None, include_non_http=True, yield_paths=False, _path=()):
    """
    DFS i vilkårlige dict/list-strukturer.
    - key_patterns: liste af substrings (lowercase) der SKAL indgå i nøglen for at yield'e (narrow-mode).
    - include_non_http: hvis True, accepteres også "/path" som vi kan urljoine.
    - yield_paths: hvis True, yield'er (value, "a.b[2].c") til debug.
    """
    import collections

    def key_ok(k: str) -> bool:
        if key_patterns is None:
            return True
        k = str(k).lower()
        return any(p in k for p in key_patterns)

    if isinstance(obj, dict):
        for k, v in obj.items():
            p = (*_path, k)
            if isinstance(v, (dict, list, tuple)):
                yield from _search_nested_urls(v, key_patterns, include_non_http, yield_paths, p)
            else:
                if key_ok(k) and isinstance(v, str):
                    sv = v.strip()
                    if not sv or sv.lower() in ("none", "null", "false"):
                        continue
                    if sv.startswith("http://") or sv.startswith("https://"):
                        yield (sv, ".".join(map(str, p))) if yield_paths else sv
                    elif include_non_http and sv.startswith("/"):
                        yield (sv, ".".join(map(str, p))) if yield_paths else sv
    elif isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            p = (*_path, i)
            if isinstance(v, (dict, list, tuple)):
                yield from _search_nested_urls(v, key_patterns, include_non_http, yield_paths, p)
            else:
                # lister har ikke meningsfulde nøgle-navne; kun direkte URL-strenge tæller
                if isinstance(v, str):
                    sv = v.strip()
                    if not sv or sv.lower() in ("none", "null", "false"):
                        continue
                    if sv.startswith("http://") or sv.startswith("https://"):
                        yield (sv, ".".join(map(str, p))) if yield_paths else sv
                    elif include_non_http and sv.startswith("/"):
                        yield (sv, ".".join(map(str, p))) if yield_paths else sv
    else:
        # primitive typer ignoreres
        return
'''.strip("\n")

NEW_RESOLVE = r'''
def resolve_canonical_enhanced(base_url: str, rendered_html: str, runtime_state: Optional[dict]) -> Tuple[Optional[str], str]:
    """
    Returnerer (canonical_url, source) med sikre fallback-strategier:
    1) <link rel="canonical"> (DOM)
    2) <meta property="og:url">
    3) Runtime-state NARROW: kun nøgler der ligner canonical (canonical, canonical_url, custom_canonical_url, og:url)
    4) Self-canonical (sidens egen URL) som sidste udvej
    """
    # 1: DOM canonical
    dom_canonical = _extract_canonical_from_html(rendered_html, base_url)
    if dom_canonical:
        return dom_canonical, "dom_link"

    soup = BeautifulSoup(rendered_html or "", "html.parser")

    # 2: OpenGraph url
    og = soup.find("meta", attrs={"property": "og:url"})
    if og and og.get("content"):
        content = og.get("content").strip()
        if content and content.lower() not in ("none", "null", "false"):
            try:
                return urljoin(base_url, content), "og_url"
            except Exception:
                pass

    # 3: Runtime narrow search (kun canonical-lignende nøgler)
    try:
        state = runtime_state or _extract_runtime_state_from_html(rendered_html)
    except Exception:
        state = None

    if state:
        key_patterns = ["canonical", "canonical_url", "custom_canonical_url", "og:url", "ogurl"]
        candidates = list(_search_nested_urls(state, key_patterns=key_patterns, include_non_http=True, yield_paths=True))
        # Filtrér, normalisér og prioriter:
        norm = []
        for item in candidates:
            val, path = item if isinstance(item, tuple) else (item, "")
            if not isinstance(val, str):
                continue
            s = val.strip()
            if not s or s.lower() in ("none", "null", "false"):
                continue
            if not (s.startswith("http://") or s.startswith("https://")):
                s = urljoin(base_url, s)  # håndter "/path"
            norm.append((s, path))

        # Dedup med orden bevaring
        seen = set()
        uniq = []
        for s, p in norm:
            if s not in seen:
                seen.add(s)
                uniq.append((s, p))

        # Heuristik: vælg første kandidat på samme host som base_url, ellers første
        from urllib.parse import urlparse
        try:
            base_host = urlparse(base_url).netloc
        except Exception:
            base_host = ""

        for s, p in uniq:
            try:
                if urlparse(s).netloc == base_host:
                    return s, "runtime_canonical"
            except Exception:
                continue
        if uniq:
            return uniq[0][0], "runtime_canonical"

    # 4: Self canonical fallback
    return base_url, "self_canonical"
'''.strip("\n")

def force_replace_function(text, func_name, new_code):
    """
    Ersætter *hele* funktionsblokken, fra 'def func_name' til lige før næste 'def ' eller EOF.
    Virker uanset tidligere indhold. Returnerer (ny_tekst, ændret_bool).
    """
    pattern = re.compile(rf'(?ms)^def\s+{re.escape(func_name)}\s*\(.*?\)\s*:(?:.*?\n)(?=(?:def\s+|class\s+)|\Z)')
    if pattern.search(text):
        new_text = pattern.sub(new_code + "\n\n", text, count=1)
        return new_text, True
    else:
        # Hvis ikke fundet: prøv at indsætte lige før EOF
        return text.rstrip() + "\n\n" + new_code + "\n", True

def main():
    if not ANALYZER.exists():
        print(f"✗ Fandt ikke {ANALYZER}")
        return

    src = ANALYZER.read_text(encoding="utf-8", errors="ignore")
    backup = ANALYZER.with_suffix(".py.bak")
    backup.write_text(src, encoding="utf-8")
    print(f"• Backup skrevet til {backup}")

    # Sikr import-helpers
    header_inject = "\n".join([
        "from typing import Optional, Tuple",
        "from bs4 import BeautifulSoup",
        "from urllib.parse import urljoin",
    ])
    if "from bs4 import BeautifulSoup" not in src:
        src = header_inject + "\n" + src

    # Overstyr _search_nested_urls (narrow version) og resolve_canonical_enhanced
    src, _ = force_replace_function(src, "_search_nested_urls", NEW_SEARCH_NARROW)
    src, _ = force_replace_function(src, "resolve_canonical_enhanced", NEW_RESOLVE)

    ANALYZER.write_text(src, encoding="utf-8")
    print(f"✓ Skrev tvangs-patch til {ANALYZER}")

if __name__ == "__main__":
    main()
