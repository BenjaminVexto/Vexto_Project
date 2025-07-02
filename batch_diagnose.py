import asyncio
import logging
# ✱ KORREKT IMPORT-STI BASERET PÅ DIN MAPPESTRUKTUR ✱
from src.vexto.scoring.url_finder import _validate_candidate

# Opsæt simpel logging for at se fejl fra httpx osv.
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)-8s] %(message)s')

# --- Liste over virksomheder der skal diagnosticeres ---
# Hver linje er et tuple: (Fuldt Virksomhedsnavn, Korrekt URL)
COMPANIES_TO_DIAGNOSE = [
    ("Amber-Byg ApS", "https://www.amberbyg.dk/"),
    ("HJEMTEX ApS", "https://www.hjemtex.com/"),
    ("J.L. ENTREPRISE ApS", "https://www.jle.dk/"),
    ("Kan-Byg Tømrer & Snedkerfirma", "www.kan-byg.dk"),
    ("Modukas ApS", "modukas.dk"),
    ("MTS BYG ApS", "https://mts-byg.dk/"),
    ("Ovenlysmanden ApS", "https://ovenlysmanden.dk/"),
    ("SNEDKERMESTER ARNE PEDERSEN A/S", "Arnepedersen.dk"),
]

async def run_batch_diagnosis():
    """
    Itererer gennem en fast liste af virksomheder og udskriver en
    diagnose-rapport for hver enkelt.
    """
    print("=" * 70)
    print("Starter batch-diagnose for manglende virksomheder...")
    print("=" * 70)

    for firm_name, url in COMPANIES_TO_DIAGNOSE:
        print(f"\n\nDiagnostiserer for firma: '{firm_name}'")
        print(f"Tester URL-kandidat:    '{url}'")
        print("-" * 70)

        # Vi sætter relaxed=False for at simulere et tjek fra "BruteForceResolver",
        # som er den mest sandsynlige kilde og bruger de strengeste regler.
        try:
            is_ok, reason, title = await _validate_candidate(
                url, firm_name, relaxed=False, slow=True
            )

            # Udskriv pæn rapport
            if is_ok:
                print("STATUS:  ✅ ACCEPTERET")
            else:
                print("STATUS:  ❌ AFVIST")

            print(f"GRUND:   {reason}")
            print(f"TITEL:   {title if title is not None else 'Ingen titel fundet'}")

        except Exception as e:
            print("STATUS:  🔥 FEJLEDE UNDER KØRSEL")
            print(f"GRUND:   En uventet fejl opstod: {e}")
            log.exception(f"Fejl under diagnose for {firm_name}")

        print("-" * 70)

if __name__ == "__main__":
    # Kør den asynkrone batch-diagnose
    asyncio.run(run_batch_diagnosis())