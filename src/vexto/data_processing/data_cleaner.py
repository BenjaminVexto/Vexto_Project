"""
data_cleaner.py
---------------
Funktioner til at rense og transformere rå CVR-hits
til et pænt, analyse-klar DataFrame-output.
"""
from __future__ import annotations
import logging
import re
from typing import List, Dict, Any
import pandas as pd

log = logging.getLogger(__name__)

# --- RegEx-konstanter -------------------------------------------------------
PHONE_DIGIT_REGEX = re.compile(r"^\d{8}$")
URL_PREFIX_REGEX = re.compile(r"^(https?:\/\/|www\.)", re.IGNORECASE)

# ---------------------------------------------------------------------------

def _parse_contact_value(raw_contact_list: Any) -> Dict[str, str]:
    """Splitter en liste af rå kontaktoplysninger i Telefon / Email / Hjemmeside."""
    if not isinstance(raw_contact_list, list):
        return {"Telefon": "N/A", "Email": "N/A", "Hjemmeside": "N/A"}

    phones, emails, websites = set(), set(), set()

    for item in raw_contact_list:
        val = ""
        if isinstance(item, dict) and "kontaktoplysning" in item:
            val = str(item["kontaktoplysning"]).strip()
        elif isinstance(item, str):
            val = item.strip()

        if not val:
            continue

        digits_only = re.sub(r"\D", "", val)

        if "@" in val:
            emails.add(val)
        elif PHONE_DIGIT_REGEX.fullmatch(digits_only):
            phones.add(val)
        elif "." in val:
            websites.add(val if URL_PREFIX_REGEX.match(val) else f"https://{val}")

    return {
        "Telefon": "; ".join(sorted(phones)) or "N/A",
        "Email": "; ".join(sorted(emails)) or "N/A",
        "Hjemmeside": "; ".join(sorted(websites)) or "N/A",
    }

def _get_all_directors(rel_list: Any) -> str:
    """Returnerer semikolon-separeret streng med direktør/indehaver-navne."""
    if not isinstance(rel_list, list):
        return "N/A"

    names = set()
    for rel in rel_list:
        if rel.get("deltager", {}).get("enhedstype") != "PERSON":
            continue
        for org in rel.get("organisationer", []) or []:
            role = (org.get("hovedtype") or "").upper()
            if role not in {
                "DIREKTION",
                "LEDELSESORGAN",
                "INDEHAVER",
                "FULDT_ANSVARLIG_DELTAGERE",
            }:
                continue
            for n in rel.get("deltager", {}).get("navne", []):
                if (name := n.get("navn")):
                    names.add(name)
    return "; ".join(sorted(names)) or "N/A"

def _best_contact(row: pd.Series) -> str:
    """Returnerer den mest værdifulde kontaktinfo i prioriteret rækkefølge."""
    return next(
        (row[col] for col in ("Telefon", "Email", "Hjemmeside") if row[col] != "N/A"),
        "N/A",
    )

# ---------------------------------------------------------------------------

def clean_and_prepare_cvr_data(raw_hits: List[Dict[str, Any]]) -> pd.DataFrame:
    df = pd.json_normalize(raw_hits, sep="_")

    # Branche
    df["Branchekode"] = df.get(
        "Vrvirksomhed_virksomhedMetadata_nyesteHovedbranche_branchekode", "N/A"
    ).fillna("N/A")
    df["Branchetekst"] = df.get(
        "Vrvirksomhed_virksomhedMetadata_nyesteHovedbranche_branchetekst", "N/A"
    ).fillna("N/A")

    # Kontaktinfo
    contact_df = (
        df.get("Vrvirksomhed_virksomhedMetadata_nyesteKontaktoplysninger", pd.Series(dtype=object))
        .apply(_parse_contact_value)
        .apply(pd.Series)
    )
    df[["Telefon", "Email", "Hjemmeside"]] = contact_df

    # Direktører
    df["Direktørnavn"] = df.get(
        "Vrvirksomhed_deltagerRelation", pd.Series(dtype=object)
    ).apply(_get_all_directors)

    # Filtrér status
    df = df[
        df["Vrvirksomhed_virksomhedMetadata_sammensatStatus"]
        .str.upper()
        .isin({"AKTIV", "NORMAL"})
    ].copy()

    # Bedste kontaktfelt
    if not df.empty:
        df["Kontaktoplysninger"] = df.apply(_best_contact, axis=1)
    else:
        df["Kontaktoplysninger"] = pd.Series(dtype=str)


    # Kolonnerækkefølge
    cols = [
        "Vrvirksomhed_cvrNummer",
        "Vrvirksomhed_virksomhedMetadata_nyesteNavn_navn",
        "Direktørnavn",
        "Kontaktoplysninger",
        "Telefon",
        "Email",
        "Hjemmeside",
        "Vrvirksomhed_virksomhedMetadata_nyesteBeliggenhedsadresse_vejnavn",
        "Vrvirksomhed_virksomhedMetadata_nyesteBeliggenhedsadresse_husnummerFra",
        "Vrvirksomhed_virksomhedMetadata_nyesteBeliggenhedsadresse_postnummer",
        "Vrvirksomhed_virksomhedMetadata_nyesteBeliggenhedsadresse_postdistrikt",
        "Branchekode",
        "Branchetekst",
        "Vrvirksomhed_virksomhedMetadata_nyesteVirksomhedsform_virksomhedsformkode",
        "Vrvirksomhed_virksomhedMetadata_nyesteVirksomhedsform_langBeskrivelse",
        "Vrvirksomhed_virksomhedMetadata_sammensatStatus",
        "Vrvirksomhed_virksomhedMetadata_ophørsDato",
    ]

    # Sikr at alle kolonner findes
    for c in cols:
        if c not in df.columns:
            df[c] = "N/A"

    log.debug("Rensede DataFrame med %s aktive rækker", len(df))
    return df.reindex(columns=cols).fillna("N/A")
