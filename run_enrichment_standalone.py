import sys
from pathlib import Path
import pandas as pd
from dotenv import load_dotenv

# --- Opsætning -----------------------------------------------------------
PROJECT_DIR = Path(__file__).resolve().parent
SRC_DIR = PROJECT_DIR / "src"
sys.path.append(str(SRC_DIR))
load_dotenv(PROJECT_DIR / ".env")

# --- Import fra dit eget modul -------------------------------------------
from vexto.enrichment.contact_finder import run_enrichment_on_dataframe

def main():
    """
    Standalone script til at køre kontakt-berigelse på den færdige fil
    fra main.py.
    """
    # Definer input- og output-filer
    # Sørg for at dette er navnet på den fil, der kommer ud af din main.py
    INPUT_FILE = PROJECT_DIR / "output" / "cvr_data.csv"
    OUTPUT_FILE = PROJECT_DIR / "output" / "cvr_data_enriched.csv"

    print(f"Indlæser data fra: {INPUT_FILE.name}")
    try:
        df_input = pd.read_csv(INPUT_FILE)
    except FileNotFoundError:
        print(f"Fejl: Inputfilen '{INPUT_FILE}' blev ikke fundet. Kør main.py først.")
        return

    # --- Filtrering: Beholder kun rækker med en live hjemmeside ---
    # Tjekker for både boolean True og strengen "True" for robusthed
    if 'HjemmesideLive' in df_input.columns:
        df_filtered = df_input[df_input['HjemmesideLive'] == True].copy()
    else:
        print("Advarsel: Kolonnen 'HjemmesideLive' blev ikke fundet. Fortsætter med alle rækker.")
        df_filtered = df_input.copy()

    print(f"Filtreret: {len(df_input)} rækker --> {len(df_filtered)} rækker med en live hjemmeside.")
    
    if df_filtered.empty:
        print("Ingen virksomheder med en live hjemmeside fundet. Stopper.")
        return

    # Kør selve berigelses-funktionen på den filtrerede data
    df_output = run_enrichment_on_dataframe(df_filtered)

    # Gem det færdige resultat
    df_output.to_csv(OUTPUT_FILE, index=False, encoding='utf-8-sig')
    print(f"✔ Resultat gemt i: {OUTPUT_FILE.name}")


if __name__ == "__main__":
    main()