# scripts/make_titles_index.py
# ANKER: IMPORTS
import json, re, unicodedata
import pandas as pd
from pathlib import Path

SRC_XLSX = Path("data/title_catalog/Stillingsbetegnelser.xlsx")
OUT_DIR  = Path("data/title_catalog")
OUT_DIR.mkdir(parents=True, exist_ok=True)

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

def main():
    xls = pd.ExcelFile(SRC_XLSX)  # ét ark: Aktive_Stillinger
    df = xls.parse(xls.sheet_names[0]).copy()

    alias_cols = [c for c in df.columns if str(c).lower().startswith("alias")]
    df["title_id"] = df["StillingsBetegnelseKode"].astype(str)
    df["da_name"]  = df["StillingsBetegnelseNavn"].astype(str)
    df["isco_code"] = df.get("DiscoAms08Kode", "")
    df["isco_name"] = df.get("DiscoAms08Navn", "")
    df["hotjob"]    = df.get("Hotjob", False).fillna(False).astype(bool)

    # Saml alias
    aliases = []
    for _, row in df.iterrows():
        al = [str(row.get(c) or "").strip() for c in alias_cols]
        al = [a for a in al if a]
        aliases.append(sorted(set(al), key=lambda x: x.lower()))
    df["aliases_da"] = aliases

    # Skriv katalog CSV
    out_csv = OUT_DIR / "title_catalog.csv"
    cols = ["title_id","da_name","isco_code","isco_name","hotjob","aliases_da"]
    df.to_csv(out_csv, index=False)

    # Byg exact-index: canonical og alle alias → title_id
    exact = {}
    for _, r in df.iterrows():
        tid = r["title_id"]
        names = [r["da_name"]] + list(r["aliases_da"])
        for n in names:
            key = _norm(n)
            if key:
                exact[key] = tid

    # Byg token-index: token → {title_id: weight}
    token_index = {}
    for _, r in df.iterrows():
        tid = r["title_id"]
        bag = set()
        for n in [r["da_name"]] + list(r["aliases_da"]):
            for tok in _tokens(n):
                if not tok:
                    continue
                bag.add(tok)
        for tok in bag:
            token_index.setdefault(tok, {}).setdefault(tid, 0)
            token_index[tok][tid] += 1

    (OUT_DIR / "titles_exact_index.json").write_text(json.dumps(exact, ensure_ascii=False))
    (OUT_DIR / "titles_token_index.json").write_text(json.dumps(token_index, ensure_ascii=False))
    print(f"OK → {out_csv}\nOK → titles_exact_index.json\nOK → titles_token_index.json")

if __name__ == "__main__":
    main()
