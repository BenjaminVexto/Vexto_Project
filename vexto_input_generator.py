import os
import requests
import csv
from dotenv import load_dotenv
from time import sleep

# 🔄 Indlæs .env
print("🔄 Indlæser .env...")
load_dotenv()
cvr_token = os.getenv("CVR_API_KEY", "demo")

# 📍 Parametre
branchekoder = ["432200"]  # VVS
postnumre = ["2670"]
resultater = []
brugte_cvr = set()

# 🧠 CVR API kræver korrekt User-Agent
headers = {
    "User-Agent": "Vexto - Dataindsamling - Benjamin Eriksson +45 12345678"
}

def hent_virksomheder_fra_cvr(branchekode):
    url = f"https://cvrapi.dk/api?branche={branchekode}&country=dk&token={cvr_token}"
    print(f"\n📡 Henter CVR-data for branche {branchekode}")
    try:
        r = requests.get(url, headers=headers)
        if r.status_code != 200:
            print(f"⚠️ HTTP-fejl {r.status_code}: {r.text}")
            return []

        # Prøv at dekode som JSON
        data = r.json()

        # Håndter tekst-baserede fejl
        if isinstance(data, dict) and data.get("error"):
            error_msg = data["error"]
            if error_msg == "QUOTA_EXCEEDED":
                print("🚫 Dagskvote brugt – prøv igen i morgen eller skaf en API-nøgle.")
            elif error_msg == "BANNED":
                print("🚫 Adgang blokeret – din IP er blevet bannet.")
            elif error_msg == "INVALID_UA":
                print("🚫 User-Agent ikke angivet korrekt – tilføj virksomheds- og kontaktinfo.")
            else:
                print(f"⚠️ API-fejl: {error_msg}")
            return []

        if not isinstance(data, list):
            print("⚠️ API returnerede intet brugbart.")
            return []

        return data

    except Exception as e:
        print(f"💥 Undtagelse under kald: {e}")
        return []

# 🔁 Loop over branchekoder
for kode in branchekoder:
    firmaer = hent_virksomheder_fra_cvr(kode)
    sleep(1)
    for firma in firmaer:
        cvr = firma.get("vat")
        post = str(firma.get("zipcode", "")).strip()

        if not cvr or cvr in brugte_cvr:
            continue
        if post not in postnumre:
            continue

        brugte_cvr.add(cvr)
        navn = firma.get("name", "").strip()
        domæne = firma.get("website", "")
        direktør = firma.get("person", "")

        print(f"✅ {navn} ({cvr}) – {post}")
        resultater.append({
            "navn": navn,
            "domæne": domæne,
            "cvr": cvr,
            "postnummer": post,
            "direktør": direktør
        })

# 💾 Gem CSV
filnavn = "vexto_input.csv"
print(f"\n💾 Gemmer {len(resultater)} virksomheder i {filnavn}")
with open(filnavn, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=["navn", "domæne", "cvr", "postnummer", "direktør"])
    writer.writeheader()
    for row in resultater:
        writer.writerow(row)

print("✅ Færdig!")
