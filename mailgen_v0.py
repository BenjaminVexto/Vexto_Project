#!/usr/bin/env python3
"""
Vexto MailGen v0.1
------------------
Læser jeres CVR-CSV, afgør outreach-mode (Analyse vs. No-site), kalder OpenAI Responses API
for at generere *målrettede* mails (DA), og skriver en output-CSV klar til udsendelse.

Kørsel:
  export OPENAI_API_KEY=sk-...
  python mailgen_v0.py --in cvr_data.csv --out mails_out.csv --model gpt-5 --temp 0.3 --max 500

Krav:
  pip install openai pandas python-slugify
Docs (Responses API): https://platform.openai.com/docs/api-reference/responses
"""

import re, time, csv, json, math, argparse, sys, random
from dataclasses import dataclass
from typing import Optional, Dict, Any, List
import pandas as pd

import os  # ← .env fallback/ENV
try:
    from dotenv import load_dotenv, find_dotenv
    DOTENV_AVAILABLE = True
except Exception:
    DOTENV_AVAILABLE = False

# --- Optional OpenAI import (script kan køres i --dry-run uden lib) ---
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except Exception:
    OPENAI_AVAILABLE = False


@dataclass
class Config:
    model: str = "gpt-5"
    temperature: float = 0.3
    max_tokens: int = 1200
    concurrency: int = 4
    max_retries: int = 4
    base_url: Optional[str] = None   # custom gateway, hvis I bruger proxy
    timeout: float = 45.0


# --- Feltnavne i jeres CSV (som inspiceret) ---
CSV_FIELDS = dict(
    cvr="Vrvirksomhed_cvrNummer",
    name="Vrvirksomhed_virksomhedMetadata_nyesteNavn_navn",
    phone="Telefon",
    email="Email",
    website="Hjemmeside",
    live="HjemmesideLiveBool",
    remark="HjemmesideBemærkning",
    source="HjemmesideKilde",
    size="AntalAnsatte",
    form_code="Vrvirksomhed_virksomhedMetadata_nyesteVirksomhedsform_virksomhedsformkode",
    form_desc="Vrvirksomhed_virksomhedMetadata_nyesteVirksomhedsform_langBeskrivelse",
    status="Vrvirksomhed_virksomhedMetadata_sammensatStatus",
    street="Vrvirksomhed_virksomhedMetadata_nyesteBeliggenhedsadresse_vejnavn",
    house_no="Vrvirksomhed_virksomhedMetadata_nyesteBeliggenhedsadresse_husnummerFra",
    zip="Vrvirksomhed_virksomhedMetadata_nyesteBeliggenhedsadresse_postnummer",
    city="Vrvirksomhed_virksomhedMetadata_nyesteBeliggenhedsadresse_postdistrikt",
)

# --- Output kolonner ---
OUT_COLUMNS = [
    "cvr", "company_name", "outreach_mode", "target_role", "size_bucket", "form_desc",
    "email_to_hint", "phone_hint", "website", "live", "website_remark",
    "linkedin_query", "template_variant",
    "subject", "preview_text", "body_text", "body_html",
    "model", "temperature", "notes"
]

# --- Path helpers (Windows-sikre) ---
def _resolve_path(p: str) -> str:
    """Normalisér sti: fjern citattegn, udvid ~ og %VARS%, og lav absolut sti."""
    if p is None:
        return ""
    s = str(p).strip().strip('"').strip("'")
    s = os.path.expandvars(os.path.expanduser(s))
    try:
        return os.path.abspath(s)
    except Exception:
        return s

def _check_exists(path: str, label: str) -> None:
    if not os.path.exists(path):
        print(f"[MailGen] {label} findes ikke: {path}\nCWD: {os.getcwd()}", file=sys.stderr)
        sys.exit(2)

# --- .env indlæsning ---
def _load_env(env_path: Optional[str]) -> None:
    """
    Indlæs .env og sæt OPENAI_API_KEY, evt. fra alternative navne.
    """
    if DOTENV_AVAILABLE:
        if env_path:
            load_dotenv(dotenv_path=env_path, override=False)
        else:
            found = find_dotenv(usecwd=True)
            if found:
                load_dotenv(found, override=False)
            else:
                load_dotenv(override=False)  # .env i CWD hvis findes

    # Fallback til alternative navne
    if not os.getenv("OPENAI_API_KEY"):
        alt = os.getenv("OPENAI_KEY") or os.getenv("OPENAI_TOKEN")
        if alt:
            os.environ["OPENAI_API_KEY"] = alt


def read_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, dtype=str)
    # Coerce numerics/bools
    if CSV_FIELDS["live"] in df.columns:
        df[CSV_FIELDS["live"]] = df[CSV_FIELDS["live"]].map(lambda x: str(x).strip().lower() in ("true", "1", "yes"))
    if CSV_FIELDS["size"] in df.columns:
        df[CSV_FIELDS["size"]] = pd.to_numeric(df[CSV_FIELDS["size"]], errors="coerce")
    return df


def canonical_domain(url: str) -> str:
    if not url or str(url).strip() == "" or str(url).lower().startswith("nan"):
        return ""
    u = str(url).strip()
    u = re.sub(r"^https?://", "", u, flags=re.I)
    u = re.sub(r"^www\.", "", u, flags=re.I)
    u = u.split("/")[0].strip()
    return u.lower()


def detect_login_only(remark: str) -> bool:
    if not remark:
        return False
    r = remark.lower()
    hits = 0
    if "wp-login" in r or "wp-admin" in r or "login" in r:
        hits += 1
    if "200" in r or "302" in r:
        hits += 1
    return hits >= 2


def detect_http_error(remark: str) -> bool:
    if not remark:
        return False
    codes = re.findall(r"\b([45]\d{2})\b", remark)
    return any(code.startswith(("4","5")) for code in codes)


def size_bucket(n: Optional[float]) -> str:
    if n is None or (isinstance(n, float) and math.isnan(n)): return "unknown"
    n = int(n)
    if n <= 1: return "0-1"
    if n <= 4: return "2-4"
    if n <= 9: return "5-9"
    if n <= 49: return "10-49"
    if n <= 199: return "50-199"
    return "200+"


def choose_target_role(size_b: str, form_desc: str) -> str:
    form = (form_desc or "").lower()
    if "enkelt" in form or size_b in ("0-1","2-4"):
        return "Ejer/Indehaver/Direktør"
    if "aktie" in form or size_b in ("200+",):
        return "Head of Digital / E-commerce / Marketing Director"
    # default ApS / SMB
    if size_b in ("5-9", "10-49"):
        return "Marketingansvarlig / E-commerce Manager / Head of Sales"
    if size_b == "50-199":
        return "Digital/Marketing Manager (sekundært CMO)"
    return "Direktør / ejernær kontakt"


def build_linkedin_query(company_name: str, size_b: str, form_desc: str) -> str:
    if not company_name:
        return ""
    company = company_name.strip()
    if "enkelt" in (form_desc or "").lower() or size_b in ("0-1","2-4"):
        titles = '("ejer" OR "indehaver" OR "direktør")'
    elif size_b in ("200+",):
        titles = '("Head of Digital" OR "E-commerce Manager" OR "Marketing Director")'
    elif size_b in ("10-49","50-199"):
        titles = '("marketingansvarlig" OR "digital" OR "e-commerce")'
    else:
        titles = '("direktør" OR "marketingansvarlig")'
    return f'site:linkedin.com/in {titles} "{company}"'


def decide_outreach_mode(website: str, live: Optional[bool], remark: str) -> str:
    domain = canonical_domain(website)
    if not domain:
        return "no_site"
    if live is False:
        return "no_site"
    if detect_login_only(remark) or detect_http_error(remark):
        return "no_site"
    return "analyse"


def pick_variant(cvr: str, ab: Optional[str]) -> str:
    if ab in ("A","B"): return ab
    # stable hash: even/odd picks
    try:
        return "A" if (int(cvr) % 2 == 0) else "B"
    except Exception:
        return random.choice(["A","B"])


def make_prompt_payload(row: Dict[str, Any], mode: str, target_role: str, size_b: str) -> Dict[str, Any]:
    return {
        "mode": mode,
        "company": {
            "name": row.get(CSV_FIELDS["name"], ""),
            "cvr": row.get(CSV_FIELDS["cvr"], ""),
            "website": row.get(CSV_FIELDS["website"], ""),
            "website_remark": row.get(CSV_FIELDS["remark"], ""),
            "live": row.get(CSV_FIELDS["live"], None),
            "email": row.get(CSV_FIELDS["email"], ""),
            "phone": row.get(CSV_FIELDS["phone"], ""),
            "employees": row.get(CSV_FIELDS["size"], None),
            "form_desc": row.get(CSV_FIELDS["form_desc"], ""),
            "address": {
                "street": row.get(CSV_FIELDS["street"], ""),
                "house_no": row.get(CSV_FIELDS["house_no"], ""),
                "zip": row.get(CSV_FIELDS["zip"], ""),
                "city": row.get(CSV_FIELDS["city"], ""),
            }
        },
        "audience": {
            "target_role": target_role,
            "size_bucket": size_b
        },
        "brand": {
            "name": "Vexto",
            "tone": "professionel, konkret, handlingsorienteret, dansk",
            "value_props": [
                "Hjemmesider der konverterer (mobil-first, Core Web Vitals)",
                "Google Business + Ads/Shopping opsætning med måling",
                "Hurtig levering, fast pris, dokumenteret effekt"
            ]
        }
    }


SYSTEM_INSTRUCTIONS = (
    "Du er Vextos autoresponder til B2B-outreach på dansk. "
    "Skriv korte, konkrete mails med stærke emnelinjer og en klar CTA. "
    "Tilpas kommunikationen til virksomhedsstørrelse og målrolle."
)

# JSON output-kontrakt sikrer strukturer
RESPONSE_FORMAT = {"type": "json_object"}

def generate_with_openai(cfg: Config, payload: Dict[str, Any], variant: str) -> Dict[str, str]:
    if not OPENAI_AVAILABLE:
        raise RuntimeError("openai-pakken er ikke installeret. Kør pip install openai")

    client = OpenAI(timeout=cfg.timeout) if cfg.base_url is None else OpenAI(base_url=cfg.base_url, timeout=cfg.timeout)
    # (SDK læser OPENAI_API_KEY/OPENAI_ORG/OPENAI_PROJECT fra miljøet automatisk)

    # Variant A/B styrer vinkling i prompten (analyse vs. ROI)
    if payload["mode"] == "analyse":
        brief = (
            "Skriv en e-mail der nævner 3 konkrete forbedringer ud fra en hurtig webanalyse "
            "(fx hastighed, sporing, CTA, indeksering, schema). "
            "Brug talpunkter. Afslut med klar CTA om at sende en 1-sides handlingsplan og prisinterval."
        )
    else:
        brief = (
            "Skriv en e-mail til en virksomhed uden effektiv hjemmeside. "
            "Brug virksomhedens website-bemærkning (hvis nogen) som ærlig observation. "
            "Foreslå hurtig løsning: landingsside + Google Business + tracking. "
            "Hold det positivt og konkret. Klar CTA om demo og fast-pris-setup."
        )

    if variant == "A":
        angle = "Fokusér på hurtige gevinster og lav implementeringsfriktion."
    else:
        angle = "Fokusér på kommercielle effekter (flere leads, lavere CPA) og risiko ved status quo."

    user_input = {
        "instructions": brief + " " + angle,
        "data": payload,
        "format": {
            "required_fields": ["subject", "preview_text", "body_text", "body_html"],
            "language": "da-DK"
        }
    }

    # Responses API (2025)
    for attempt in range(cfg.max_retries):
        try:
            resp = client.responses.create(
                model=cfg.model,
                temperature=cfg.temperature,
                response_format=RESPONSE_FORMAT,
                input={
                    "role": "user",
                    "content": [
                        {"type": "text", "text": SYSTEM_INSTRUCTIONS},
                        {"type": "input_text", "text": json.dumps(user_input, ensure_ascii=False)},
                    ],
                },
            )
            # Library returns helper .output_text (may contain JSON). Safer to parse from top candidate.
            text = getattr(resp, "output_text", None)
            if not text and hasattr(resp, "output"):
                # Fallback: join any text segments
                try:
                    parts = []
                    for item in resp.output:
                        if hasattr(item, "content"):
                            for c in item.content:
                                if getattr(c, "type", "") in ("output_text", "text"):
                                    parts.append(getattr(c, "text", ""))
                    text = "\n".join(parts)
                except Exception:
                    pass

            if not text:
                raise RuntimeError("Tomt svar fra modellen")

            data = json.loads(text)
            # Minimal validering
            for k in ("subject", "preview_text", "body_text", "body_html"):
                data.setdefault(k, "")
            return data
        except Exception as e:
            # 429/backoff
            slp = 1.5 * (2 ** attempt) + random.random()
            if attempt < cfg.max_retries - 1:
                time.sleep(slp)
            else:
                raise

def process(df: pd.DataFrame, cfg: Config, max_rows: Optional[int], ab_choice: Optional[str], dry_run: bool, mode_filter: Optional[str]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    n = len(df) if not max_rows else min(max_rows, len(df))

    for idx in range(n):
        row = df.iloc[idx].fillna("")
        company_name = str(row.get(CSV_FIELDS["name"], "")).strip()
        cvr = str(row.get(CSV_FIELDS["cvr"], "")).strip()
        website = str(row.get(CSV_FIELDS["website"], "")).strip()
        live = bool(row.get(CSV_FIELDS["live"], False))
        remark = str(row.get(CSV_FIELDS["remark"], "")).strip()
        form_desc = str(row.get(CSV_FIELDS["form_desc"], "")).strip()
        size = row.get(CSV_FIELDS["size"], None)
        sb = size_bucket(size if size != "" else None)

        mode = decide_outreach_mode(website, live, remark)
        if mode_filter and mode != mode_filter:
            continue

        target_role = choose_target_role(sb, form_desc)
        linkedin_q = build_linkedin_query(company_name, sb, form_desc)
        variant = pick_variant(cvr, ab_choice)

        payload = make_prompt_payload(row.to_dict(), mode, target_role, sb)

        subject = preview = body_text = body_html = ""
        if dry_run:
            # Generér placeholder-tekst, så CSV stadig kan testes
            subject = f"[DRY-RUN/{mode}/{variant}] Forslag til {company_name or 'jeres virksomhed'}"
            preview = "Kort, konkret værditilbud fra Vexto (demo/plan/fast pris)."
            body_text = f"(Dette er en dry-run)\n\nHej {company_name or 'der'},\n\nKort oplæg...\n\nMvh Vexto"
            body_html = f"<p>(Dette er en dry-run)</p><p>Hej {company_name or 'der'},</p><p>Kort oplæg...</p><p>Mvh Vexto</p>"
        else:
            gen = generate_with_openai(cfg, payload, variant)
            subject, preview, body_text, body_html = gen["subject"], gen["preview_text"], gen["body_text"], gen["body_html"]

        rows.append(dict(
            cvr=cvr,
            company_name=company_name,
            outreach_mode=mode,
            target_role=target_role,
            size_bucket=sb,
            form_desc=form_desc,
            email_to_hint=row.get(CSV_FIELDS["email"], ""),
            phone_hint=row.get(CSV_FIELDS["phone"], ""),
            website=website,
            live=live,
            website_remark=remark,
            linkedin_query=linkedin_q,
            template_variant=variant,
            subject=subject.strip(),
            preview_text=preview.strip(),
            body_text=body_text.strip(),
            body_html=body_html.strip(),
            model=cfg.model,
            temperature=cfg.temperature,
            notes=""
        ))
    return rows


def write_out(path: str, rows: List[Dict[str, Any]]):
    from pathlib import Path
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=OUT_COLUMNS)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in OUT_COLUMNS})


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Vexto MailGen – generér outreach-mails i CSV via OpenAI")
    ap.add_argument("--in", dest="inp", required=True, help="Input CSV fra Vexto (CVR)")
    ap.add_argument("--out", dest="out", required=True, help="Output CSV med mails klar til udsendelse")
    ap.add_argument("--model", default="gpt-5", help="OpenAI modelnavn (fx gpt-5 eller gpt-4.1)")
    ap.add_argument("--temp", type=float, default=0.3, help="Temperature (0.0–1.0)")
    ap.add_argument("--max", dest="max_rows", type=int, default=None, help="Maks antal rækker at generere")
    ap.add_argument("--ab", dest="ab_choice", choices=["A","B"], default=None, help="Tving A eller B (ellers hash pr. CVR)")
    ap.add_argument("--mode", dest="mode_filter", choices=["analyse","no_site"], default=None, help="Begræns til én outreach-mode")
    ap.add_argument("--dry-run", action="store_true", help="Ingen API-kald – lav placeholder-tekster (til pipeline-test)")
    ap.add_argument("--env", dest="env", default=".env", help="Sti til .env (default: .env i CWD)")
    return ap.parse_args()


def main():
    args = parse_args()

    # Indlæs .env (hvis du allerede har denne fra tidligere patch)
    try:
        _load_env(args.env)  # hvis du ikke har --env-flagget, kan denne linje udelades
    except NameError:
        pass

    cfg = Config(
        model=args.model,
        temperature=args.temp,
    )
    cfg.base_url = os.getenv("OPENAI_BASE_URL") or None

    if not args.dry_run and not OPENAI_AVAILABLE:
        print("openai-biblioteket mangler. Kør: pip install openai", file=sys.stderr)
        sys.exit(2)
    if not args.dry_run and not os.getenv("OPENAI_API_KEY"):
        print("Mangler OPENAI_API_KEY (læg i .env eller miljø).", file=sys.stderr)
        sys.exit(2)

    inp_path = _resolve_path(args.inp)
    out_path = _resolve_path(args.out)
    _check_exists(inp_path, "Input-fil (--in)")

    df = read_csv(inp_path)
    rows = process(df, cfg, args.max_rows, args.ab_choice, args.dry_run, args.mode_filter)
    write_out(out_path, rows)
    print(f"Skrev {len(rows)} mails til: {out_path}")


if __name__ == "__main__":
    main()
