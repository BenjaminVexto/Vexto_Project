# src/vexto/enrichment/title_matcher.py
from __future__ import annotations
import json, re, unicodedata
from pathlib import Path
from typing import Optional, Tuple, Dict

DATA_DIR = Path(__file__).resolve().parents[3] / "data" / "title_catalog"

def _norm(s: str) -> str:
    if not isinstance(s, str):
        return ""
    t = s.strip().lower()
    t = unicodedata.normalize("NFKD", t)
    t = "".join(c for c in t if not unicodedata.combining(c))
    t = re.sub(r"[^\w\s/+\-&]", " ", t)
    t = re.sub(r"[/\-]+", " ", t)
    t = re.sub(r"\s{2,}", " ", t).strip()
    return t

def _tokens(s: str) -> list[str]:
    return [w for w in re.split(r"\W+", _norm(s)) if w]

class TitleMatcher:
    def __init__(self, exact: Dict[str,str], token_index: Dict[str,Dict[str,int]], catalog_csv: Path):
        self.exact = exact
        self.token_index = token_index
        self.catalog_csv = catalog_csv
        self._id2name: Dict[str, str] = {}
        try:
            import csv
            with catalog_csv.open("r", encoding="utf-8") as f:
                for row in csv.DictReader(f):
                    self._id2name[row["title_id"]] = row["da_name"]
        except Exception:
            pass

    @classmethod
    def load(cls) -> "TitleMatcher":
        def _read_json_any(path: Path):
            try:
                return json.loads(path.read_text(encoding="utf-8"))
            except UnicodeDecodeError:
                # håndter evt. BOM eller tidligere cp1252-skrivninger
                try:
                    return json.loads(path.read_text(encoding="utf-8-sig"))
                except UnicodeDecodeError:
                    return json.loads(path.read_text(encoding="cp1252"))

        exact = _read_json_any(DATA_DIR / "titles_exact_index.json")
        token = _read_json_any(DATA_DIR / "titles_token_index.json")

        # byg objektet først
        obj = cls(exact, token, DATA_DIR / "title_catalog.csv")

        # [NYT] valgfri overrides (data/title_catalog/overrides.json)
        ovr_path = DATA_DIR / "overrides.json"
        obj.overrides = {}
        if ovr_path.exists():
            try:
                obj.overrides = _read_json_any(ovr_path)
            except Exception:
                obj.overrides = {}

        return obj

    def match(self, raw: Optional[str]) -> Optional[Tuple[str, str, str, float]]:
        if not raw:
            return None
        key = _norm(raw)
        if not key:
            return None

        # 0) [NYT] Overrides (hurtig mapping uden at røre Excel)
        ovr = getattr(self, "overrides", {}) or {}
        if key in ovr:
            canonical = ovr[key]               # fx "Kundeservicemedarbejder"
            tid = self.exact.get(_norm(canonical))
            if tid:
                return (tid, self._id2name.get(tid, canonical), "override", 1.0)
            # hvis canonical ikke findes i exact-index, falder vi bare videre til normal logik

        # 1) Exact/alias
        tid = self.exact.get(key)
        if tid:
            return (tid, self._id2name.get(tid, raw), "exact", 1.0)

        # 2) Fuzzy via token-overlap (hurtigt og deterministisk)
        toks = _tokens(raw)
        if not toks:
            return None
        cand_scores: Dict[str, int] = {}
        for tok in toks:
            for tid2, w in self.token_index.get(tok, {}).items():
                cand_scores[tid2] = cand_scores.get(tid2, 0) + w

        if not cand_scores:
            return None

        # Normaliser til [0..1] med max-score for denne kandidatpose
        best_tid, best_score = max(cand_scores.items(), key=lambda kv: kv[1])
        max_possible = sum(sorted(cand_scores.values(), reverse=True)[:3]) or 1
        ratio = min(1.0, best_score / max_possible)

        if ratio >= 0.90:
            return (best_tid, self._id2name.get(best_tid, raw), "fuzzy", round(ratio, 3))
        if ratio >= 0.80:
            return (best_tid, self._id2name.get(best_tid, raw), "fuzzy", round(ratio, 3))
        return None
