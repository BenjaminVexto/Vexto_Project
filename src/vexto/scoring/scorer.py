# src/vexto/scoring/scorer.py
import yaml
import logging
from pathlib import Path
from typing import Any, Dict, List, Tuple

from .schemas import UrlAnalysisData

log = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
RULES_PATH = PROJECT_ROOT / "config" / "scoring_rules.yml"

def load_rules() -> Dict:
    """Indlæser point-reglerne fra YAML-filen."""
    try:
        with open(RULES_PATH, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        log.error(f"Regel-fil ikke fundet på: {RULES_PATH}")
        return {}
    except yaml.YAMLError as e:
        log.error(f"Fejl ved parsing af YAML-regel-fil: {e}")
        return {}

def get_nested_value(data: Dict, dotted_key: str) -> Any:
    """Slår en værdi op i et nested dict vha. punktum-notation, f.eks. 'technical_seo.is_https'."""
    keys = dotted_key.split(".")
    current_level_data = data.get(keys[0])
    if current_level_data is None:
        return None
    for key in keys[1:]:
        if isinstance(current_level_data, dict):
            current_level_data = current_level_data.get(key)
            if current_level_data is None:
                return None
        else:
            return None
    return current_level_data

def _evaluate_rule(value: Any, rule_type: str, threshold: Any = None, **kwargs) -> bool:
    """Intern funktion til at evaluere en enkelt regelbetingelse."""
    if value is None:
        return False
    # Sikkerhedscheck for numeriske sammenligninger: konverter til float hvis muligt
    numeric_types = ("greater_than_or_equal", "less_than_or_equal", "greater_than", "less_than", "range", "between")
    if rule_type in numeric_types:
        try:
            value = float(value)
            if threshold is not None:
                threshold = float(threshold)
            if 'min' in kwargs:
                kwargs['min'] = float(kwargs['min'])
            if 'max' in kwargs:
                kwargs['max'] = float(kwargs['max'])
        except (ValueError, TypeError):
            log.debug(f"Værdi '{value}' kunne ikke konverteres til tal for regeltype '{rule_type}'. Returnerer False.")
            return False
    if rule_type == "boolean_true" and value is True: return True
    if rule_type == "equals" and value is not None and value == threshold: return True
    if rule_type == "greater_than_or_equal" and value is not None and threshold is not None and value >= threshold: return True
    if rule_type == "less_than_or_equal" and value is not None and threshold is not None and value <= threshold: return True
    if rule_type == "greater_than" and value is not None and threshold is not None and value > threshold: return True
    if rule_type == "less_than" and value is not None and threshold is not None and value < threshold: return True
    if rule_type in ("range", "between"):
        min_val = kwargs.get('min')
        max_val = kwargs.get('max')
        if min_val is not None and max_val is not None:
            return min_val <= value <= max_val
    return False

def calculate_sitemap_score(data: dict) -> int:
    """Beregner sitemap-point baseret på 'found' og 'fresh' status."""
    is_found = data.get('found', False)
    is_fresh = data.get('fresh', False)
    
    if is_found and is_fresh:
        return 15
    elif is_found:
        return 10
    else:
        return 0

def score_social_and_reputation(data: UrlAnalysisData) -> Tuple[bool, int]:
    """Custom scoring for social_and_reputation section."""
    social_data = data.get('social_and_reputation', {})
    # Tjek for social links
    social_links = social_data.get('social_media_links', []) or []
    share_links = social_data.get('social_share_links', []) or []
    if len(social_links) > 0 or len(share_links) > 0:
        return True, 5
    # Tjek for GMB reviews (håndter None eksplicit)
    gmb_count = social_data.get('gmb_review_count')
    if gmb_count is not None and isinstance(gmb_count, (int, float)) and gmb_count > 0:
        return True, 2
    return False, 0

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
    # Data kommer som dict med canonical_url key
    canonical_url = data.get('canonical_url')
    
    # Tjek også i basic_seo hvis det ikke er direkte tilgængeligt
    if not canonical_url and isinstance(data, dict):
        canonical_url = data.get('basic_seo', {}).get('canonical_url')
    
    if canonical_url:
        log.debug(f"Canonical found: {canonical_url}")
        return 10
    else:
        log.debug(f"No canonical found in data: {data}")
        return 0

def calculate_google_reviews(data: dict) -> int:
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
    points = 0
    pagerank = data.get('pagerank') or 0
    global_rank_str = data.get('global_rank') or '999999'
    try:
        global_rank = int(global_rank_str)
    except ValueError:
        global_rank = 999999
    trustpilot_count = data.get('trustpilot_count') or 0
    trustpilot_rating = data.get('trustpilot_rating') or 0
    if pagerank >= 6:
        points += 25
    elif pagerank >= 4:
        points += 15
    if global_rank <= 500:
        points += 10
    if trustpilot_count >= 500:
        points += 10
    if trustpilot_rating >= 4.3:
        points += 10
    return min(points, 65)

def calculate_social_presence(data: UrlAnalysisData) -> int:
    social_data = data.get('social_and_reputation', {})
    count = len(social_data.get('social_media_links', []) or []) + len(social_data.get('social_share_links', []) or [])
    if count >= 3:
        return 20
    elif count >= 1:
        return 10
    return 0

def calculate_tracking(data: dict) -> int:
    ga4 = data.get('ga4') or False
    pixel = data.get('pixel') or False
    if ga4 and pixel:
        return 70
    elif ga4 or pixel:
        return 35
    return 0

def calculate_contact_score(data: dict) -> int:
    emails = len(data.get('emails') or [])
    phones = len(data.get('phones') or [])
    if emails > 0 and phones > 0:
        return 35
    elif emails > 0 or phones > 0:
        return 15
    return 0

def calculate_form_score(data: dict) -> int:
    if not data or not isinstance(data, list):
        return 0
    if not data:  # Ingen formularer
        return 0
    min_fields = min(data)
    if min_fields <= 5:
        return 35
    elif min_fields <= 10:
        return 15
    return 0

def calculate_trust_score(data: dict) -> int:
    if not data or not isinstance(data, list):
        return 0
    count = len(data)
    if count >= 2:
        return 25
    elif count >= 1:
        return 10
    return 0

CUSTOM_FUNCTIONS = {
    'calculate_sitemap_score': calculate_sitemap_score,
    'score_social_and_reputation': score_social_and_reputation,
    'calculate_canonical_score': calculate_canonical_score,
    'evaluate_canonical': evaluate_canonical,
    'calculate_google_reviews': calculate_google_reviews,
    'calculate_authority_score': calculate_authority_score,
    'calculate_social_presence': calculate_social_presence,
    'calculate_tracking': calculate_tracking,
    'calculate_contact_score': calculate_contact_score,
    'calculate_form_score': calculate_form_score,
    'calculate_trust_score': calculate_trust_score,
}

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
            "total_score": 0,
            "achieved_rules": [],
            "failed_rules": [],
            "not_evaluated_rules": [],
            "max_possible_points": 0,
            "score_percentage": 0
        }

    for section_key, section_content in rules.items():
        max_possible_points += section_content.get("max_points", 0)

        for rule_key, rule_details in section_content.get("rules", {}).items():
            rule_passed = False
            points_achieved = 0
            description = rule_details.get("description", "")
            fetcher = rule_details.get("fetcher", "ukendt")
            
            rule_type = rule_details.get("type")
            
            if rule_type == "custom_function":
                custom_func_name = rule_details.get("function_name")
                custom_func = CUSTOM_FUNCTIONS.get(custom_func_name)
                
                if custom_func:
                    required_data, all_data_found = {}, True
                    data_keys_map = rule_details.get("data_keys", {})
                    
                    for name, key_in_section in data_keys_map.items():
                        if '.' in key_in_section:
                            full_key_path = key_in_section
                        else:
                            full_key_path = f"{section_key}.{key_in_section}"
                        val = get_nested_value(analysis_data, full_key_path)
                        if val is None:
                            all_data_found = False
                            log.warning(f"Missing data for {full_key_path} in {rule_key}")
                            break
                        required_data[name] = val
                    
                    if all_data_found:
                        points_achieved = custom_func(analysis_data if custom_func.__name__ == 'calculate_social_presence' else required_data)
                        if points_achieved > 0: rule_passed = True
                        value_for_report = required_data
                    else:
                        value_for_report = None
                else:
                    log.warning(f"Custom function '{custom_func_name}' ikke fundet i scorer.py.")
                    value_for_report = None
            
            elif section_key == "social_and_reputation":
                rule_passed, points_achieved = score_social_and_reputation(analysis_data)
                value_for_report = {
                    "social_media_links": analysis_data.get('social_and_reputation', {}).get('social_media_links', []),
                    "social_share_links": analysis_data.get('social_and_reputation', {}).get('social_share_links', []),
                    "gmb_review_count": analysis_data.get('social_and_reputation', {}).get('gmb_review_count', None),
                    "gmb_average_rating": analysis_data.get('social_and_reputation', {}).get('gmb_average_rating', None)
                }
            
            else:
                data_key = rule_details.get("data_key")
                full_key_path = f"{section_key}.{data_key}" if data_key else None
                value_for_report = get_nested_value(analysis_data, full_key_path) if full_key_path else None
                
                if value_for_report is None and rule_key in ['competitor_comparison', 'ux_ui_quality', 'market_fit']:
                    value_for_report = analysis_data.get(data_key, None)  # Direkte tjek for strategic_assessment keys
                
                if value_for_report is None:
                    value_for_report = "N/A"
                    points_achieved = 0
                    log.debug(f"Missing data for {full_key_path or data_key} in {rule_key}, defaulting to 0 points")
                
                if "levels" in rule_details:
                    for level in rule_details["levels"]:
                        level_type = level.get("type") or rule_details.get("type")
                        if _evaluate_rule(value_for_report, level_type, **level):
                            rule_passed = True
                            points_achieved = level.get("points", 0)
                            description += f" - {level.get('description_suffix', '')}"
                            break
                else:
                    if _evaluate_rule(value_for_report, rule_details.get("type"), **rule_details):
                        rule_passed = True
                        points_achieved = rule_details.get("points", 0)
            
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
        "url": analysis_data.get("url"),
        "total_score": total_score,
        "max_possible_points": max_possible_points,
        "score_percentage": score_percentage,
        "achieved_rules": sorted(achieved_rules, key=lambda x: x.get('points', 0), reverse=True),
        "failed_rules": failed_rules,
        "not_evaluated_rules": not_evaluated_rules
    }

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