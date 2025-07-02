# src/vexto/scoring/scorer.py (SCORER V2)

import yaml
import logging
from pathlib import Path
from typing import Any, Dict, List

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

def _evaluate_rule(value: Any, rule_type: str, threshold: Any) -> bool:
    """Intern funktion til at evaluere en enkelt regelbetingelse."""
    if value is None:
        return False
    if rule_type == "boolean_true" and value is True:
        return True
    if rule_type == "greater_than_or_equal" and value is not None and threshold is not None and value >= threshold:
        return True
    if rule_type == "less_than_or_equal" and value is not None and threshold is not None and value <= threshold:
        return True
    return False

def calculate_score(analysis_data: UrlAnalysisData) -> Dict[str, Any]:
    """
    Beregner en detaljeret score baseret på indsamlet data og et regelsæt.
    Returnerer nu total score, max score, procent, beståede og fejlede regler.
    """
    rules = load_rules()
    total_score = 0
    max_possible_points = 0
    achieved_rules: List[Dict[str, Any]] = []
    failed_rules: List[Dict[str, Any]] = []

    if not rules:
        return {"total_score": 0, "achieved_rules": [], "failed_rules": [], "max_possible_points": 0, "score_percentage": 0}

    for section_key, section_content in rules.items():
        data_section = analysis_data.get(section_key, {})
        max_possible_points += section_content.get("max_points", 0)

        for rule_key, rule_details in section_content.get("rules", {}).items():
            data_key = rule_details.get("data_key")
            value = data_section.get(data_key) if data_key else None
            
            rule_passed = False
            points_achieved = 0
            description = rule_details.get("description", "")

            # Håndter regler med flere niveauer
            if "levels" in rule_details:
                for level in rule_details["levels"]:
                    if _evaluate_rule(value, level.get("type"), level.get("threshold")):
                        rule_passed = True
                        points_achieved = level.get("points", 0)
                        description += f" - {level.get('description_suffix')}"
                        break
            # Håndter simple regler
            else:
                if _evaluate_rule(value, rule_details.get("type"), rule_details.get("threshold")):
                    rule_passed = True
                    points_achieved = rule_details.get("points", 0)
            
            # Tilføj til den korrekte liste
            if rule_passed:
                total_score += points_achieved
                achieved_rules.append({"rule": rule_key, "description": description, "points": points_achieved})
            elif value is not None: # Reglen blev evalueret, men fejlede
                failed_rules.append({"rule": rule_key, "description": description, "value": value})

    # Beregn procent
    score_percentage = round((total_score / max_possible_points) * 100) if max_possible_points > 0 else 0

    return {
        "url": analysis_data.get("url"),
        "total_score": total_score,
        "max_possible_points": max_possible_points,
        "score_percentage": score_percentage,
        "achieved_rules": sorted(achieved_rules, key=lambda x: x['points'], reverse=True),
        "failed_rules": failed_rules,
    }