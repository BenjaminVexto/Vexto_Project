# tests/debug_csv.py
import pandas as pd
import os
import sys

# Konfigurer stier præcis som i dit hovedscript
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CSV_PATH = os.path.join(PROJECT_ROOT, "urls.csv")

print(f"--- Starter Debug af CSV-indlæsning ---")
print(f"Beregnet projekt-rod: {PROJECT_ROOT}")
print(f"Fuld sti til CSV-fil: {CSV_PATH}")
print(f"Eksisterer filen på stien? {'✅ Ja' if os.path.exists(CSV_PATH) else '❌ NEJ'}")

if os.path.exists(CSV_PATH):
    try:
        # Første forsøg: Præcis som i dit script
        print("\n--- Forsøg 1: Indlæsning med dine nuværende indstillinger ---")
        df1 = pd.read_csv(
            CSV_PATH,
            sep=";",
            encoding="utf-8",
            low_memory=False
        )
        print(f"Antal rækker fundet: {len(df1)}")
        if not df1.empty:
            print(f"Kolonner fundet: {df1.columns.tolist()}")
            print("Første række:")
            print(df1.head(1))
        else:
            print("--> RESULTAT: DataFrame er TOM.")

        # Andet forsøg: Med 'utf-8-sig' for at håndtere potentiel BOM (Byte Order Mark)
        print("\n--- Forsøg 2: Indlæsning med encoding 'utf-8-sig' ---")
        df2 = pd.read_csv(
            CSV_PATH,
            sep=";",
            encoding="utf-8-sig", # Denne encoding ignorerer usynlige start-tegn
            low_memory=False
        )
        print(f"Antal rækker fundet: {len(df2)}")
        if not df2.empty:
             print(f"Kolonner fundet: {df2.columns.tolist()}")

    except Exception as e:
        print(f"\n❌ FEJL: Der opstod en uventet fejl under indlæsning: {e}")

print("\n--- Debug færdig ---")