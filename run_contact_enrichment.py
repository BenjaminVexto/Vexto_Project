# run_contact_enrichment.py
import pandas as pd
from dotenv import load_dotenv
load_dotenv()  # læser .env fra projektroden

from src.vexto.enrichment.csv_contact_pipeline import enrich_csv_with_contacts

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Enrich CSV med kontaktdata (Steps A-D + Fallbacks).")
    p.add_argument("input_csv", help="Sti til input CSV (kræver kolonner: AntalAnsatte, Direktørnavn, Email, Hjemmeside)")
    p.add_argument("-o", "--output", default="output_enriched.csv", help="Sti til output CSV")
    p.add_argument("--company-name-col", default="Navn", help="Kolonne med firmanavn (bruges i Step C/D)")
    args = p.parse_args()

    df = pd.read_csv(args.input_csv)
    out = enrich_csv_with_contacts(df, company_name_col=args.company_name_col)
    out.to_csv(args.output, index=False)
    print(f"✅ Skrevet: {args.output}")
