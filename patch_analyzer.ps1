# --- patch_analyzer.ps1 ---
$analyzer = "src\vexto\scoring\analyzer.py"
if (!(Test-Path $analyzer)) { Write-Error "Fandt ikke $analyzer"; exit 1 }

# 1) Indsæt/erstat Enhanced canonical + schema + forms + content freshness helpers
$enhanced = @"
# === VEXTO ENHANCED HELPERS (AUTO-INSERT) ===
from urllib.parse import urljoin
import re
from bs4 import BeautifulSoup

def _search_nested_object(obj, keys_like):
    found = []
    def walk(o):
        try:
            if isinstance(o, dict):
                for k,v in o.items():
                    kl = str(k).lower()
                    if any(x in kl for x in keys_like):
                        # pluk åbenlyse URL-strenge
                        if isinstance(v, str) and re.search(r'https?://|^/[^/]', v):
                            found.append(v)
                    walk(v)
            elif isinstance(o, (list, tuple)):
                for it in o: walk(it)
        except Exception:
            pass
    walk(obj)
    return found

def resolve_canonical_enhanced(base_url: str, rendered_html: str, runtime_state: dict) -> tuple[str,str]:
    soup = BeautifulSoup(rendered_html or "", "html.parser")

    # 1) klassisk <link rel="canonical">
    link = soup.find("link", attrs={"rel": re.compile(r"\bcanonical\b", re.I)})
    if link and link.get("href"):
        return (urljoin(base_url, link["href"]), "dom_link")

    # 2) og:url
    og = soup.find("meta", attrs={"property": "og:url"})
    if og and og.get("content"):
        return (og.get("content"), "og_url")

    # 3) bred runtime-søgning
    if runtime_state:
        urls = _search_nested_object(runtime_state, ["canonical", "url", "path", "href", "link"])
        for u in urls:
            if isinstance(u, str):
                if u.startswith("http://") or u.startswith("https://"):
                    return (u, "runtime_enhanced")
                if u.startswith("/"):
                    return (urljoin(base_url, u), "runtime_enhanced")
                # undgå at behandle slices/indices som nøgler -> ingen [x:y]!
                if re.match(r"^[\w\-/]+$", u):
                    try:
                        return (urljoin(base_url, "/"+u.lstrip("/")), "runtime_enhanced")
                    except Exception:
                        pass

    # 4) fallback: selv-kanonisk
    return (base_url, "self_canonical")

def detect_schema_enhanced(rendered_html: str):
    soup = BeautifulSoup(rendered_html or "", "html.parser")
    jsonld_types = []
    jsonlds = soup.find_all("script", attrs={"type": re.compile(r"application/ld\+json", re.I)})
    for tag in jsonlds:
        try:
            import json
            data = json.loads(tag.string or "{}")
            t = data.get("@type") or data.get("@graph", [{}])[0].get("@type")
            if t:
                jsonld_types.append(t if isinstance(t, str) else str(t))
        except Exception:
            pass
    micro = bool(soup.find_all(attrs={"itemscope": True}))
    rdfa = bool(soup.find_all(attrs={"typeof": True}))
    og = bool(soup.find_all("meta", attrs={"property": re.compile(r"^og:", re.I)}))
    any_found = bool(jsonld_types) or micro or rdfa or og
    return any_found, {"jsonld": jsonld_types, "microdata": micro, "rdfa": rdfa, "opengraph": og}

def analyze_forms_enhanced(soup: BeautifulSoup) -> dict:
    if not soup:
        return {"form_field_counts": []}
    try:
        field_counts = []
        # klassiske forms
        for form in soup.find_all("form"):
            inputs = [i for i in form.find_all(["input","textarea","select"])
                      if i.get("type") not in ("hidden","submit","reset","button")]
            field_counts.append(len(inputs))
        # dynamiske containere
        dyn = soup.find_all(attrs={"class": re.compile(r"(form|contact|checkout|subscribe)", re.I)})
        dyn += soup.find_all(attrs={"id": re.compile(r"(form|contact|checkout|subscribe)", re.I)})
        for c in set(dyn):
            ins = c.find_all(["input","textarea","select"])
            if ins: field_counts.append(len(ins))
        return {"form_field_counts": field_counts}
    except Exception:
        return {"form_field_counts": []}

def extract_latest_date_candidates(html: str) -> list[str]:
    # flere datoformater, inkl. dansk
    patterns = [
        r"\b\d{4}-\d{2}-\d{2}\b",             # 2025-08-11
        r"\b\d{2}/\d{2}/\d{4}\b",             # 11/08/2025
        r"\b\d{1,2}\.\s*\w+\s*\d{4}\b",       # 11. august 2025
        r"\b\d{1,2}-\d{1,2}-\d{2,4}\b",       # 11-08-2025
    ]
    found = []
    for p in patterns:
        found += re.findall(p, html or "", flags=re.I)
    return found
# === /VEXTO ENHANCED HELPERS ===
"@

$code = Get-Content $analyzer -Raw

# a) Indsæt helpers hvis de ikke findes
if ($code -notmatch "VEXTO ENHANCED HELPERS") {
  # sæt dem lige efter imports i filen
  $code = $code -replace "(?s)(^from .*?\n(?:import .*\n)+)", "`$1`r`n$enhanced`r`n"
}

# b) Erstat kald til gamle helpers, hvis de findes, med de nye navne
$code = $code -replace "resolve_canonical\(", "resolve_canonical_enhanced("
$code = $code -replace "detect_schema\(", "detect_schema_enhanced("

# c) Sikr at vi bruger analyze_forms_enhanced hvor vi danner form_field_counts
$code = $code -replace "analyze_forms\s*\(\s*soup\s*\)", "analyze_forms_enhanced(soup)"

# d) Robust håndtering af runtime state (fix for 'unhashable type: slice'):
#    erstatter mønstre hvor vi logger/indekserer 'category-next.menuCategories[<tal>].custom_canonical_url'
$code = $code -replace "category-next\.menuCategories\[[^\]]+\]\.custom_canonical_url",
                                  "safe_runtime_value"

# e) Tilføj en lille wrapper-funktion til at hente safe_runtime_value, hvis den ikke findes
if ($code -notmatch "def safe_runtime_value\(") {
  $safeFn = @"
def safe_runtime_value(runtime_state: dict, *path):
    try:
        cur = runtime_state
        for key in path:
            if isinstance(cur, dict) and key in cur:
                cur = cur[key]
            elif isinstance(cur, list) and isinstance(key, int) and 0 <= key < len(cur):
                cur = cur[key]
            else:
                return None
        return cur
    except Exception:
        return None
"@
  # Indsæt før enhanced helpers eller lige efter dem
  $code = $code -replace "(# === /VEXTO ENHANCED HELPERS ===)",
                         "$safeFn`r`n`$1"
}

# f) Gem
Set-Content $analyzer $code -Encoding UTF8
Write-Host "✅ analyzer.py patched."

# g) (valgfrit) kør din upload-task
if (Test-Path ".\upload_vexto.ps1") {
  powershell -ExecutionPolicy Bypass -File ".\upload_vexto.ps1"
}
