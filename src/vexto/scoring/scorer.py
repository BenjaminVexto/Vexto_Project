# src/vexto/scoring/scorer.py

import yaml
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union
from .schemas import UrlAnalysisData
from vexto.utils.paths import deep_get as _deep_get

log = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
RULES_PATH = PROJECT_ROOT / "config" / "scoring_rules.yml"

def _normalize_rule_type(rt: str) -> str:
    rt = (rt or "").lower().strip()
    synonyms = {
        "boolean_true": "bool_true",
        "true": "bool_true",
        "less_than_or_equal": "lte", "<=": "lte", "le": "lte",
        "less_than": "lt", "<": "lt",
        "greater_than_or_equal": "gte", ">=": "gte", "ge": "gte",
        "greater_than": "gt", ">": "gt",
        "eq": "equals",
    }
    return synonyms.get(rt, rt)

def _unwrap_value(v):
    # Nogle regler får vist {'value': X} som "målt værdi" – scor kun på X
    if isinstance(v, dict) and set(v.keys()) == {"value"}:
        return v.get("value")
    return v

def _to_float(x):
    try:
        return float(x)
    except Exception:
        return None

def load_rules() -> Dict:
    """Indlæser point-reglerne fra YAML-filen."""
    try:
        with open(RULES_PATH, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f) or {}
    except FileNotFoundError:
        log.error(f"Regel-fil ikke fundet på: {RULES_PATH}")
        return {}
    except yaml.YAMLError as e:
        log.error(f"Fejl ved parsing af YAML-regel-fil: {e}")
        return {}


def get_nested_value(data: Dict, dotted_key: str) -> Any:
    """Robust opslag via dot-path (understøtter også listeindeks)"""
    if not dotted_key:
        return None
    return _deep_get(data, dotted_key, default=None)


def _smart_get(analysis: Dict, default_section: str, key_path: str):
    """
    Fleksibelt opslag:
    - fuld sti "a.b.c" → direkte
    - ellers prøv "{section}.{key}"
    - ellers scan 1. niveau i alle sektioner
    - ellers forsøg igen som dotted key
    """
    if not key_path:
        return None
    if "." in key_path:
        return get_nested_value(analysis, key_path)
    val = get_nested_value(analysis, f"{default_section}.{key_path}")
    if val is not None:
        return val
    for sect, data in (analysis or {}).items():
        if isinstance(data, dict) and key_path in data:
            return data.get(key_path)
    return get_nested_value(analysis, key_path)


def _filter_eval_kwargs(rule_details: dict) -> dict:
    allowed = {
        "min", "max", "equals", "threshold",
        "regex", "one_of", "any_of", "all_of",
        "contains", "present", "true_if_present",
        "min_len", "max_len", "value_type", "negate"
    }
    out = {k: v for k, v in (rule_details or {}).items() if k in allowed}

    rt = _normalize_rule_type(rule_details.get("type"))
    thr = rule_details.get("threshold", None)
    if thr is not None:
        # Map 'threshold' til den rigtige nøgle ift. operator
        if rt in {"gte", "gt", "min"}:
            out.setdefault("min", thr)
        elif rt in {"lte", "lt", "max"}:
            out.setdefault("max", thr)
        elif rt in {"equals"}:
            out.setdefault("equals", thr)

    return out

def _evaluate_rule(value, rule_type: str, **kw) -> bool:
    rt = _normalize_rule_type(rule_type)
    v = _unwrap_value(value)

    # Bool
    if rt in ("bool_true",):
        if isinstance(v, bool):
            return v is True
        if isinstance(v, (int, float)):
            return v == 1  # fx viewport_audit score=1
        if isinstance(v, str):
            return v.strip().lower() in {"true", "1", "yes"}
        return False

    # EQUALS
    if rt == "equals":
        return v == kw.get("equals")

    # GT / LT
    if rt == "gt":
        m = _to_float(kw.get("min"))
        vf = _to_float(v)
        return (m is not None) and (vf is not None) and (vf > m)
    if rt == "lt":
        M = _to_float(kw.get("max"))
        vf = _to_float(v)
        return (M is not None) and (vf is not None) and (vf < M)

    # GTE / LTE
    if rt == "gte":
        m = _to_float(kw.get("min"))
        vf = _to_float(v)
        return (m is not None) and (vf is not None) and (vf >= m)
    if rt == "lte":
        M = _to_float(kw.get("max"))
        vf = _to_float(v)
        return (M is not None) and (vf is not None) and (vf <= M)

    # RANGE (min/max samtidig)
    if rt in ("range",):
        m = _to_float(kw.get("min"))
        M = _to_float(kw.get("max"))
        vf = _to_float(v)
        ok_min = (m is None) or (vf is not None and vf >= m)
        ok_max = (M is None) or (vf is not None and vf <= M)
        return ok_min and ok_max

    # PRESENT / true_if_present
    if rt == "present":
        return v is not None and (v != "")

    if rt == "true_if_present":
        # hvis feltet eksisterer → bestå
        return v is not None

    # CONTAINS
    if rt == "contains":
        needle = kw.get("contains")
        if isinstance(v, (list, set, tuple)):
            return needle in v
        return isinstance(v, str) and isinstance(needle, str) and (needle.lower() in v.lower())

    # ONE_OF
    if rt == "one_of":
        options = kw.get("one_of") or []
        return v in options

    # REGEX
    if rt == "regex":
        pattern = kw.get("regex")
        try:
            return bool(re.search(pattern, str(v or ""), flags=re.IGNORECASE))
        except Exception:
            return False

    # Fallback: ukendt type → False
    return False



# ---------------------------
# Custom scorings (enkeltstående)
# ---------------------------

def calculate_sitemap_score(data: dict) -> int:
    """Beregner sitemap-point baseret på 'found' og 'fresh' status."""
    is_found = data.get('found', False)
    is_fresh = data.get('fresh', False)
    if is_found and is_fresh:
        return 15
    elif is_found:
        return 10
    return 0

def score_gmb_profile(data: dict) -> int:
    """
    0–65p: 20 (profil) + 15 (website) + 15 (hours) + 15 (≥3 photos)
    """
    profile = bool(data.get("profile_complete"))
    has_site = bool(data.get("has_website"))
    has_hours = bool(data.get("has_hours"))
    photo_count = int(data.get("photo_count") or 0)

    points = 0
    if profile:
        points += 20
    if has_site:
        points += 15
    if has_hours:
        points += 15
    if photo_count >= 3:
        points += 15
    return min(points, 65)

def score_social_and_reputation(payload: Union[UrlAnalysisData, dict]) -> Union[int, Tuple[bool, int]]:
    """
    Kan bruges på to måder:
    - Givet hele analysis_data (har nøgle 'social_and_reputation') → returnerer (passed: bool, points: int)
    - Givet et lille dict med social felter → returnerer kun points (int)
    """
    # Hvis det ligner hele analysis_data
    if isinstance(payload, dict) and 'social_and_reputation' in payload:
        social_data = payload.get('social_and_reputation', {}) or {}
        social_links = social_data.get('social_media_links') or []
        share_links = social_data.get('social_share_links') or []
        gmb_count = social_data.get('gmb_review_count')
        gmb_rating = social_data.get('gmb_average_rating')

        if (len(social_links) + len(share_links)) > 0:
            return True, 5
        if isinstance(gmb_count, (int, float)) and gmb_count > 0:
            return True, 2
        return False, 0

    # Ellers: lille dict direkte
    d = payload or {}
    social_links = d.get('social_media_links') or []
    share_links = d.get('social_share_links') or []
    gmb_count = d.get('gmb_review_count')
    if (len(social_links) + len(share_links)) > 0:
        return 5
    if isinstance(gmb_count, (int, float)) and gmb_count > 0:
        return 2
    return 0


def calculate_canonical_score(data: dict) -> int:
    error = data.get('error')
    canonical_url = data.get('canonical_url')
    if error is None and canonical_url:
        return 15
    elif error or canonical_url is None:
        log.debug(f"Canonical score: error={error}, canonical_url={canonical_url}")
        return 0
    return 0


def evaluate_canonical(data: dict) -> int:
    """Evaluerer canonical URL baseret på tilstedeværelse og kilde."""
    # Under _invoke_custom_function med "data_key" modtager vi payload som {"value": <canonical_url>}
    canonical_url = data.get('canonical_url')
    if not canonical_url:
        canonical_url = data.get('value')  # <- håndter payload {"value": "..."}
    if not canonical_url and isinstance(data, dict):
        canonical_url = (data.get('basic_seo', {}) or {}).get('canonical_url')
    return 10 if canonical_url else 0


def calculate_google_reviews(data: dict) -> int:
    # Neutral ved unknown status (ingen straf/bonus)
    status = (data.get('status') or data.get('gmb_status') or "").lower()
    if status == "unknown":
        return 0

    count = data.get('count') or 0
    rating = data.get('rating') or 0
    if count >= 50 and rating >= 4.5:
        return 65
    elif count >= 20 and rating >= 4.0:
        return 45
    elif count >= 10 and rating >= 3.5:
        return 25
    return 0


def calculate_authority_score(data: dict) -> int:
    """
    Review-fri autoritet (65p) baseret på OPR:
      - PageRank (0–10)  max 30p
      - Domain Authority (0–100)  max 20p
      - Global Rank (trafik)  max 15p
    """
    points = 0
    pagerank = float(data.get('pagerank') or 0)
    domain_auth = float(data.get('domain_authority') or 0)
    global_rank_str = data.get('global_rank') or '999999'
    try:
        global_rank = int(global_rank_str)
    except (TypeError, ValueError):
        global_rank = 999999

    # PageRank → 0..30
    if pagerank >= 6:
        points += 30
    elif pagerank >= 4:
        points += 20
    elif pagerank >= 2:
        points += 10

    # Domain Authority → 0..20 (mere fin kornet)
    if domain_auth >= 60:
        points += 20
    elif domain_auth >= 40:
        points += 12
    elif domain_auth >= 30:
        points += 6

    # Global Rank → 0..15 (giv lidt for 1–5M)
    if global_rank <= 500_000:
        points += 15
    elif global_rank <= 1_000_000:
        points += 8
    elif global_rank <= 5_000_000:
        points += 3

    return min(points, 65)


def calculate_niche_positioning(data: UrlAnalysisData) -> int:
    """
    Niche-positionering baseret på eksisterende metrics:
      - content.keyword_relevance_score (0..1)
      - basic_seo.schema_types (svag proxy for struktur)
      - content.average_word_count (svag proxy for dybde)
    Point: 30/15/0
    """
    try:
        content = (data.get("content") or {})
        basic   = (data.get("basic_seo") or {})
        rel     = float(content.get("keyword_relevance_score") or 0.0)
        words   = float(content.get("average_word_count") or 0.0)
        has_schema = bool((basic.get("schema_types") or []))

        score = 0
        if rel >= 0.70:
            score = 30
        elif rel >= 0.40:
            score = 15
        else:
            score = 0

        # Små bounded justeringer
        if has_schema:
            score = min(30, score + 2)
        if words >= 1000:
            score = min(30, score + 3)

        return int(score)
    except Exception:
        return 0


def calculate_social_presence(data: UrlAnalysisData) -> int:
    """
    Robust overfor:
      - fuldt analysis_data (forventer 'social_and_reputation' dict)
      - payload dicts af formen {"value": [...]} (direkte liste)
      - payload dicts med felter 'social_media_links'/ 'social_share_links'
    """
    # 1) Fuldt analysis_data
    if isinstance(data, dict) and "social_and_reputation" in data:
        s = data.get("social_and_reputation") or {}
        links = (s.get("social_media_links") or []) + (s.get("social_share_links") or [])
        count = len(links)
    # 2) Direkte {"value": [...]}
    elif isinstance(data, dict) and "value" in data:
        v = data.get("value")
        count = len(v or []) if isinstance(v, list) else 0
    # 3) Dict med direkte felter
    else:
        links = []
        if isinstance(data, dict):
            links += data.get("social_media_links") or []
            links += data.get("social_share_links") or []
        count = len(links)

    if count >= 3:
        return 20
    if count >= 1:
        return 10
    return 0


def calculate_tracking(data: Dict) -> int:
    """
    Points: 70
    - GA4 & Pixel → 70
    - Én af (GA4, Pixel) + GTM → 50
    - Kun én af (GA4, Pixel) → 35
    - Kun GTM → 40
    - Ingen → 0
    Begrundelse: GTM alene indikerer professionel tracking-opsætning, men uden verificeret GA4/Pixel gives delvise point.
    """
    ga4 = bool(data.get("ga4"))
    pixel = bool(data.get("pixel"))
    gtm = bool(data.get("gtm"))

    if ga4 and pixel:
        return 70
    if (ga4 ^ pixel) and gtm:
        return 50
    if ga4 or pixel:
        return 35
    if gtm:
        return 40
    return 0


def calculate_contact_score(data: dict) -> int:
    emails = len(data.get('emails') or [])
    phones = len(data.get('phones') or [])
    if emails > 0 and phones > 0:
        return 35
    elif emails > 0 or phones > 0:
        return 15
    return 0


def calculate_form_score(data: list) -> int:
    """
    Accepterer både:
      - rå liste [int, ...]
      - payload dict {"value": [int, ...]}
    """
    if isinstance(data, dict) and "value" in data:
      data = data.get("value")
        
    if not data or not isinstance(data, list):
        return 0
    nums = [x for x in data if isinstance(x, (int, float))]
    if not nums:
        return 0
    
    min_fields = min(nums)
    if min_fields <= 5:
        return 35
    elif min_fields <= 10:
        return 15
    return 0


def calculate_trust_score(data: list) -> int:
    """
    Accepterer:
      - rå liste ["trustpilot", "partner", ...]
      - payload dict {"value": [...]}
    """
    # Tillad payload {"value": [...]}
    if isinstance(data, dict) and "value" in data:
        data = data.get("value")

    if not data or not isinstance(data, list):
        return 0

    # Dedup & normalisering
    uniq = {str(x).strip().lower() for x in data if x}
    n = len(uniq)
    if n >= 2:
        return 25
    if n >= 1:
        return 10
    return 0

def percentage_linear(data: dict) -> int:
    """
    Forventer payload {"value": <pct>} fra _invoke_custom_function.
    Returnerer int-point 0..10.
    """
    try:
        pct = float((data or {}).get("value", 0))
    except Exception:
        return 0
    if pct < 0:
        pct = 0.0
    if pct > 100:
        pct = 100.0
    return int(round(pct / 10.0))

CUSTOM_FUNCTIONS = {
    'calculate_sitemap_score': calculate_sitemap_score,
    'score_social_and_reputation': score_social_and_reputation,  # håndteres fleksibelt
    'calculate_canonical_score': calculate_canonical_score,
    'evaluate_canonical': evaluate_canonical,
    'calculate_google_reviews': calculate_google_reviews,
    'calculate_authority_score': calculate_authority_score,
    'calculate_social_presence': calculate_social_presence,
    'calculate_niche_positioning': calculate_niche_positioning,  # P8
    'calculate_tracking': calculate_tracking,
    'calculate_contact_score': calculate_contact_score,
    'calculate_form_score': calculate_form_score,
    'calculate_trust_score': calculate_trust_score,
    'percentage_linear': percentage_linear,
    'score_gmb_profile': score_gmb_profile,
}


def _invoke_custom_function(
    func_name: str,
    rule_details: dict,
    analysis_data: UrlAnalysisData,
    section_key: str
) -> Tuple[int, Any]:
    """
    Kalder en custom-funktion robust.
    Returnerer (points:int, value_for_report:Any).
    """
    func = CUSTOM_FUNCTIONS.get(func_name)
    if not func:
        log.error(f"Unknown custom function: {func_name}")
        return 0, None

    # Byg payload:
    payload: Any
    if rule_details.get("data_keys"):
        payload = {}
        for alias, key_path in (rule_details.get("data_keys") or {}).items():
            payload[alias] = _smart_get(analysis_data, section_key, key_path)
    elif rule_details.get("data_key"):
        payload = {"value": _smart_get(analysis_data, section_key, rule_details.get("data_key"))}
    else:
        # Som fallback giver vi hele analysis_data (nogle funktioner forventer dette)
        payload = analysis_data

    try:
        result = func(payload)
    except Exception as e:
        log.error(f"Error in custom function {func_name}: {e}", exc_info=True)
        return 0, payload

    # Normalisér resultat til points
    points: int
    if isinstance(result, tuple):
        # fx (passed, points) eller (points, extra)
        if len(result) == 2 and isinstance(result[1], (int, float)):
            points = int(result[1])
        elif len(result) >= 1 and isinstance(result[0], (int, float)):
            points = int(result[0])
        else:
            points = 0
    elif isinstance(result, dict) and 'points' in result:
        points = int(result.get('points') or 0)
    else:
        # antag int/float
        try:
            points = int(result or 0)
        except Exception:
            points = 0

    return points, payload


def calculate_score(analysis_data: UrlAnalysisData) -> Dict[str, Any]:
    """
    Beregner en detaljeret score baseret på indsamlet data og et regelsæt.
    Returnerer total score, max score, procent, samt beståede, fejlede og ikke-evaluerede regler.
    """
    rules = load_rules()
    total_score = 0
    max_possible_points = 0
    achieved_rules: List[Dict[str, Any]] = []
    failed_rules: List[Dict[str, Any]] = []
    not_evaluated_rules: List[Dict[str, Any]] = []

    if not rules:
        return {
            "url": analysis_data.get("url") if isinstance(analysis_data, dict) else None,
            "total_score": 0,
            "achieved_rules": [],
            "failed_rules": [],
            "not_evaluated_rules": [],
            "max_possible_points": 0,
            "score_percentage": 0
        }

    for section_key, section_content in rules.items():
        # Sektionens teoretiske maksimum (fra YAML)
        max_possible_points += section_content.get("max_points", 0)

        for rule_key, rule_details in (section_content.get("rules") or {}).items():
            rule_passed = False
            points_achieved = 0
            description = rule_details.get("description", "")
            fetcher = rule_details.get("fetcher", "ukendt")
            value_for_report: Any = None

            # 1) Custom function
            if rule_details.get("type") == "custom_function":
                func_name = rule_details.get("function_name")
                points_achieved, value_for_report = _invoke_custom_function(
                    func_name, rule_details, analysis_data, section_key
                )
                rule_passed = points_achieved > 0

            # 2) (Bevar bagudkomp.) Sektion-specifik specialcase
            elif section_key == "social_and_reputation" and rule_details.get("type") == "composite":
                # Kompatibilitet med tidligere versioner hvor denne sektion blev special-scoreret
                passed, pts = score_social_and_reputation(analysis_data)
                rule_passed = bool(passed)
                points_achieved = int(pts)
                value_for_report = {
                    "social_media_links": (analysis_data.get('social_and_reputation', {}) or {}).get('social_media_links', []),
                    "social_share_links": (analysis_data.get('social_and_reputation', {}) or {}).get('social_share_links', []),
                    "gmb_review_count": (analysis_data.get('social_and_reputation', {}) or {}).get('gmb_review_count', None),
                    "gmb_average_rating": (analysis_data.get('social_and_reputation', {}) or {}).get('gmb_average_rating', None)
                }

            # 3) Normal regel-evaluering
            else:
                data_key = rule_details.get("data_key")

                # Ny: fleksibelt opslag på tværs af sektioner (matcher Excel/YML-omlægning)
                # - prøv "{section}.{key}"
                # - ellers scan 1. niveau på tværs af sektioner
                # - accepter også fulde dot-paths (fx "content.keyword_relevance_score")
                value_for_report = _smart_get(analysis_data, section_key, data_key)

                # Strategic keys uden fuld sti (bevar special-case)
                if value_for_report is None and rule_key in ['competitor_comparison', 'ux_ui_quality', 'market_fit']:
                    if isinstance(analysis_data, dict):
                        value_for_report = analysis_data.get(data_key, None)

                # Sær-case: content.days_since_last_post → markér som ikke evalueret hvis ukendt
                if (
                    rule_details.get("path") == "content.days_since_last_post"
                    or (section_key == "content" and rule_details.get("data_key") == "days_since_last_post")
                ):
                    if value_for_report is None or value_for_report in ("N/A", "", "ukendt"):
                        not_evaluated_rules.append({
                            "rule": rule_details.get("name") or "content.days_since_last_post",
                            "description": rule_details.get("description", "Ingen dato for seneste indlæg"),
                        })
                        # Skip resten af denne regel
                        continue

                # Manglende data → N/A
                
                if value_for_report is None:
                    # Defensiv: fastsæt fuld nøglesti for logging (støtter både dot-paths og sektion+key)
                    full_key_path = rule_details.get("path")
                    if not full_key_path:
                        if data_key and "." in (data_key or ""):
                            full_key_path = data_key
                        elif data_key:
                            full_key_path = f"{section_key}.{data_key}"
                        else:
                            full_key_path = None

                    value_for_report = "N/A"
                    log.debug(f"Missing data for {full_key_path or data_key} in {rule_key}, defaulting to 0 points")


                # Levels (trinvis point)
                if "levels" in rule_details and isinstance(rule_details["levels"], list):
                    for level in rule_details["levels"]:
                        level_type = level.get("type") or rule_details.get("type")
                        if _evaluate_rule(value_for_report, level_type, **_filter_eval_kwargs(level)):
                            rule_passed = True
                            points_achieved = int(level.get("points", 0))
                            suffix = level.get('description_suffix')
                            if suffix:
                                description += f" - {suffix}"
                            break
                else:
                    if _evaluate_rule(value_for_report, rule_details.get("type"), **_filter_eval_kwargs(rule_details)):
                        rule_passed = True
                        points_achieved = int(rule_details.get("points", 0))

            # Resultat-registrering
            if value_for_report == "N/A" and rule_key not in ['competitor_comparison', 'ux_ui_quality', 'market_fit']:
                not_evaluated_rules.append({"rule": rule_key, "description": description, "fetcher": fetcher})
            elif rule_passed:
                total_score += points_achieved
                achieved_rules.append({
                    "rule": rule_key,
                    "description": description,
                    "points": points_achieved,
                    "fetcher": fetcher,
                    "value": value_for_report
                })
            else:
                failed_rules.append({
                    "rule": rule_key,
                    "description": description,
                    "value": value_for_report,
                    "fetcher": fetcher
                })

    score_percentage = round((total_score / max_possible_points) * 100) if max_possible_points > 0 else 0

    return {
        "url": analysis_data.get("url") if isinstance(analysis_data, dict) else None,
        "total_score": total_score,
        "max_possible_points": max_possible_points,
        "score_percentage": score_percentage,
        "achieved_rules": sorted(achieved_rules, key=lambda x: x.get('points', 0), reverse=True),
        "failed_rules": failed_rules,
        "not_evaluated_rules": not_evaluated_rules
    }


# ---------------------------
# Lokal hurtig-test
# ---------------------------

def test_scoring():
    dummy_data = {
        'technical_seo': {'status_code': 200, 'is_https': True, 'sitemap_xml_found': True, 'sitemap_is_fresh': True, 'robots_txt_found': True, 'broken_links_pct': 0.5, 'canonical_error': None, 'canonical_url': 'https://example.com'},
        'performance': {'lcp_ms': 2000, 'cls': 0.05, 'inp_ms': 150, 'viewport_score': 1},
        'basic_seo': {'title_length': 55, 'meta_description_length': 140, 'avg_image_size_kb': 80, 'image_alt_pct': 95, 'schema_markup_found': True, 'word_count': 400, 'canonical_error': None, 'canonical_url': 'https://example.com'},
        'content': {'days_since_last_post': 60, 'keyword_relevance_score': 0.8, 'internal_link_score': 1},
        'authority': {'page_authority': 5, 'global_rank': '8130', 'gmb_profile_complete': True},
        'social_and_reputation': {'gmb_review_count': 60, 'gmb_average_rating': 4.6, 'social_media_links': ['fb', 'ig', 'tw'], 'gmb_profile_complete': True},
        'conversion': {'has_ga4': True, 'has_meta_pixel': True, 'emails_found': ['a@b.com'], 'phone_numbers_found': ['12345678'], 'form_field_counts': [4], 'trust_signals_found': ['cert1', 'cert2']},
        'benchmark_complete': False,
        'ux_ui_score': 0,
        'niche_score': 0
    }
    score = calculate_score(dummy_data)
    print(score)
