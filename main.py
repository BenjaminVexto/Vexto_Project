"""
main.py
-------
Kører GUI’en, henter CVR-data via Elasticsearch-handleren og gemmer resultater
i /output.
"""
from pathlib import Path
from datetime import datetime
import sys
import pandas as pd

#  ► importér nu fra es_utils i stedet for “elasticsearch”
from es_utils.elasticsearch_handler import search_companies
from gui.vexto_selector_gui import run_selector

PROJECT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR  = PROJECT_DIR / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

VALGTE_PARAM_FILE = OUTPUT_DIR / "valgte_parametre.csv"
CVR_DATA_FILE     = OUTPUT_DIR / "cvr_data.csv"


def main() -> None:
    print("▶ Kører Vexto Selector GUI...")
    selection = run_selector()
    if selection is None:
        print("→ Bruger annullerede – stopper.")
        sys.exit(0)

    postnumre, branchekode = selection
    print(f"▶ Valgte postnumre: {postnumre}")
    print(f"▶ Valgt branchekode: {branchekode}")

    # gem valgene som log
    pd.DataFrame(
        {"postnummer": postnumre, "branchekode": [branchekode] * len(postnumre)}
    ).to_csv(VALGTE_PARAM_FILE, index=False)

    print("▶ Henter data fra Elasticsearch ...")
    hits = search_companies(postnumre, branchekode, size=10_000)
    print(f"▶ {len(hits)} virksomheder fundet.")

    if hits:
        pd.DataFrame(hits).to_csv(CVR_DATA_FILE, index=False)
        print(f"▶ Gemte CSV i {CVR_DATA_FILE.relative_to(PROJECT_DIR)}")
    else:
        print("⚠ Ingen virksomheder fundet; CSV oprettes ikke.")

    print("✓ Færdig kl.", datetime.now().strftime("%H:%M:%S"))


if __name__ == "__main__":
    main()
