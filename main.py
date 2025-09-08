"""
main.py
--------
Hoved-dirigent for Vexto-workflowet:
1) Kør GUI og indsamling af brugerinput
2) Hent CVR-data via Elasticsearch-handleren
3) Rens & forbered data
4) Gem resultat i /output
(Fremtidigt: 5) Kør scoring-motoren
"""
from __future__ import annotations
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
import pandas as pd
from dotenv import load_dotenv

# --- Grund-opsætning -------------------------------------------------------
PROJECT_DIR = Path(__file__).resolve().parent
SRC_DIR = PROJECT_DIR / "src"
sys.path.append(str(SRC_DIR))  # gør vexto-pakken importérbar uden pip install
load_dotenv(PROJECT_DIR / ".env")

# --- Egne moduler (efter sys.path-patch) -----------------------------------
from vexto.gui.vexto_selector_gui import run_selector  # type: ignore
from vexto.es.elasticsearch_handler import search_companies  # type: ignore
from vexto.data_processing.data_cleaner import clean_and_prepare_cvr_data

# --- Logging ---------------------------------------------------------------
logging.basicConfig(
    level=os.getenv("VEXTO_LOG_LEVEL", "INFO").upper(),
    format=os.getenv("VEXTO_LOG_FMT", "%(asctime)s  %(levelname)-8s %(name)s: %(message)s"),
    datefmt=os.getenv("VEXTO_LOG_DATEFMT", "%H:%M:%S"),
)

# Lad .env styre ES-loggeren (default INFO). Undgå at tvinge DEBUG.
_es_level = os.getenv("VEXTO_ES_LOG_LEVEL", "INFO").upper()
logging.getLogger("vexto.es").setLevel(getattr(logging, _es_level, logging.INFO))

# --- Faste stier -----------------------------------------------------------
OUTPUT_DIR = PROJECT_DIR / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

VALGTE_PARAM_FILE = OUTPUT_DIR / "valgte_parametre.csv"
CVR_DATA_FILE = OUTPUT_DIR / "cvr_data.csv"

# ---------------------------------------------------------------------------

def _save_param_csv(postnumre: list[str], branchekoder: list[str]) -> None:
    """
    Gemmer ét CSV-ark med alle kombinationer af (postnummer, branchekode),
    så videre scripts kan læse præcis det samme input som brugeren valgte.
    """
    rows = [
        {"postnummer": pn, "branchekode": bk}
        for pn in postnumre
        for bk in branchekoder
    ]
    pd.DataFrame(rows).to_csv(VALGTE_PARAM_FILE, index=False)
    logging.debug("Valgte parametre gemt → %s", VALGTE_PARAM_FILE.name)

def _fetch_cvr_hits(postnumre: list[str], branchekoder: list[str]) -> list[dict]:
    """
    Henter alle hits for de valgte branchekoder.
    Returnerer én samlet liste af rå hits.
    """
    hits: list[dict] = []
    for bk in branchekoder:
        logging.info("Henter CVR-hits for branchekode %s …", bk)
        try:
            hits.extend(search_companies(postnumre, bk, size=3_000))
        except Exception as e:
            logging.error("Fejl ved Elasticsearch-kald (%s): %s", bk, e, exc_info=True)
    return hits

def main() -> None:
    logging.info("🎛️  Starter Vexto-applikationen")
    # 1) GUI
    try:
        selection = run_selector()
    except Exception as e:
        logging.critical("Kritisk fejl under kørsel af GUI: %s", e, exc_info=True)
        sys.exit(1)

    if selection is None:
        logging.info("Bruger annullerede – lukker ned.")
        sys.exit(0)

    postnumre, branchekoder = selection
    logging.info("Valg: %s postnumre · %s branchekoder", len(postnumre), len(branchekoder))
    _save_param_csv(postnumre, branchekoder)

    # 2) CVR-kald
    raw_hits = _fetch_cvr_hits(postnumre, branchekoder)
    logging.info("%s rå virksomheder modtaget", len(raw_hits))
    if not raw_hits:
        logging.warning("Ingen virksomheder fundet – stopper.")
        return

    # 3) Dataclean
    df_ready = clean_and_prepare_cvr_data(raw_hits)

    # 4) Gem resultat
    df_ready.to_csv(CVR_DATA_FILE, index=False, encoding="utf-8-sig")
    logging.info("Gemte %s aktive virksomheder → %s", len(df_ready), CVR_DATA_FILE.name)

    # 5) (TODO) scoring-motor
    # from vexto.scoring.cli import run_scoring_on_file
    # run_scoring_on_file(CVR_DATA_FILE)

    logging.info("✔  Workflow afsluttet kl. %s", datetime.now().strftime("%H:%M:%S"))

if __name__ == "__main__":
    main()