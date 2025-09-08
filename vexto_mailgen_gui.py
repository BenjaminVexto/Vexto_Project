#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Vexto MailGen – PySide6 GUI v0.1 (forenklet og guided)
------------------------------------------------------
- 1) Data: vælg CSV, output-sti og .env → indlæs data og se opsummering
- 2) Kriterier: definér målgrupper (segmenter) via formular (eller avanceret JSON/YAML)
- 3) Udsendelse: vælg kontakt-type, A/B, AI-model, regler for kontakt, sendetid, kanalrækkefølge
- 4) Resultater: KPI, log og tabel (sorterbar). Eksporter til CSV.

Afhængigheder:
  pip install PySide6 pandas python-dotenv pyyaml openai

Bemærk:
- Appen forsøger at importere contact_finder.find_contacts hvis tilgængelig
- YAML kræver pyyaml, ellers brug JSON
"""

from __future__ import annotations

import os, sys, re, csv, json, math, random, traceback
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List

import pandas as pd

# Optional: .env
dotenv_ok = False
try:
    from dotenv import load_dotenv, find_dotenv
    dotenv_ok = True
except Exception:
    pass

# Optional: OpenAI
openai_ok = False
try:
    from openai import OpenAI
    openai_ok = True
except Exception:
    pass

# Optional: YAML
yaml_ok = False
try:
    import yaml
    yaml_ok = True
except Exception:
    pass

# Optional: contact_finder
cf_ok = False
try:
    try:
        from src.vexto.contact_finder import find_contacts  # type: ignore
    except Exception:
        from contact_finder import find_contacts  # type: ignore
    cf_ok = True
except Exception:
    find_contacts = None  # type: ignore

# ---------------------------
# Forretningslogik (port af MailGen)
# ---------------------------

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

OUT_COLUMNS = [
    "cvr", "company_name", "outreach_mode", "target_role", "size_bucket", "form_desc",
    "email_to_hint", "phone_hint", "website", "live", "website_remark",
    "linkedin_query", "template_variant",
    # kanal + timing
    "channel", "send_window", "send_after",
    # valgt kontakt (fra contact_finder)
    "contact_name", "contact_email", "contact_role", "contact_source",
    # tekster
    "subject", "preview_text", "body_text", "body_html",
    "model", "temperature", "notes"
]

# Vises kun i GUI (eksport bruger OUT_COLUMNS)
OUT_HEADERS_DK = [
    "CVR","Virksomhed","Kontakt-type","Målrolle","Størrelse","Virksomhedsform",
    "Email (hint)","Tlf (hint)","Website","Live","Website-bemærkning",
    "LinkedIn-søgning","A/B-variant",
    "Kanal","Sendetidsrum","Udskyd til",
    "Kontakt navn","Kontakt email","Kontakt rolle","Kilde",
    "Emne","Forhåndstekst","Brødtekst (txt)","Brødtekst (HTML)",
    "AI-model","Kreativitet","Målgruppe"
]

DEFAULT_PLAYBOOK = {
    "segments": [
        {"name": "micro", "forms": ["Enkeltmandsvirksomhed", "PMV"], "size_max": 4,
         "target_role": "Ejer/Indehaver/Direktør", "channel": "email"},
        {"name": "smb", "forms": ["Anpartsselskab"], "size_min": 5, "size_max": 199,
         "target_role": "Marketingansvarlig / E-commerce Manager / Head of Sales", "channel": "email"},
        {"name": "corp", "forms": ["Aktieselskab"], "size_min": 200,
         "target_role": "Head of Digital / E-commerce / Marketing Director", "channel": "email"}
    ],
    "outreach_rules": {
        "analyse_if_live": True,
        "no_site_if_login_or_4xx_5xx": True,
        "fallback_mode": "analyse"
    },
    "scheduling": {
        "send_window": "workdays_08_16",
        "send_after_hours": 0
    },
    "channel_priority": ["email", "linkedin_dm"]
}

@dataclass
class GenConfig:
    model: str = "gpt-5"
    temperature: float = 0.3
    timeout: float = 45.0
    base_url: Optional[str] = None

# ---------- Hjælpere ----------

def _resolve_path(p: str) -> str:
    if p is None:
        return ""
    s = str(p).strip().strip('"').strip("'")
    s = os.path.expandvars(os.path.expanduser(s))
    try:
        return os.path.abspath(s)
    except Exception:
        return s

def load_env(env_path: Optional[str]):
    if dotenv_ok:
        if env_path:
            load_dotenv(dotenv_path=env_path, override=False)
        else:
            found = find_dotenv(usecwd=True)
            if found:
                load_dotenv(found, override=False)
            else:
                load_dotenv(override=False)
    # fallback til alternative navne
    if not os.getenv("OPENAI_API_KEY"):
        alt = os.getenv("OPENAI_KEY") or os.getenv("OPENAI_TOKEN")
        if alt:
            os.environ["OPENAI_API_KEY"] = alt

def read_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, dtype=str)
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
    if n is None or (isinstance(n, float) and math.isnan(n)):
        return "unknown"
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

def match_segment(playbook: dict, form_desc: str, emp_bucket: str) -> dict:
    form = (form_desc or "").lower()
    for seg in playbook.get("segments", []):
        forms = [f.lower() for f in seg.get("forms", [])]
        size_min = seg.get("size_min", None)
        size_max = seg.get("size_max", None)
        form_ok = (not forms) or any(x in form for x in forms)
        size_ok = True
        if size_min is not None:
            if emp_bucket in ["0-1","2-4","5-9"] and size_min >= 10:
                size_ok = False
        if size_max is not None:
            if emp_bucket in ["200+"] and size_max < 200:
                size_ok = False
        if form_ok and size_ok:
            return seg
    return {}

def choose_channel(seg: dict, playbook: dict) -> str:
    if seg.get("channel"):
        return seg["channel"]
    prio = playbook.get("channel_priority", ["email"])
    return prio[0] if prio else "email"

def compute_timing(playbook: dict) -> tuple[str, str]:
    sw = playbook.get("scheduling", {}).get("send_window", "workdays_08_16")
    delay_h = int(playbook.get("scheduling", {}).get("send_after_hours", 0) or 0)
    send_after = (datetime.utcnow() + timedelta(hours=delay_h)).isoformat(timespec="seconds") + "Z"
    return sw, send_after

SYSTEM_INSTRUCTIONS = (
    "Du er Vextos autoresponder til B2B-outreach på dansk. "
    "Skriv korte, konkrete mails med stærke emnelinjer og en klar CTA. "
    "Tilpas kommunikationen til virksomhedsstørrelse og målrolle."
)

RESPONSE_FORMAT = {"type": "json_object"}

def generate_with_openai(cfg: GenConfig, payload: Dict[str, Any], variant: str) -> Dict[str, str]:
    if not openai_ok:
        raise RuntimeError("openai-biblioteket er ikke installeret. Kør: pip install openai")
    client = OpenAI(timeout=cfg.timeout) if cfg.base_url is None else OpenAI(base_url=cfg.base_url, timeout=cfg.timeout)

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
    angle = "Fokusér på hurtige gevinster og lav implementeringsfriktion." if variant == "A" else \
            "Fokusér på kommercielle effekter (flere leads, lavere CPA) og risiko ved status quo."

    user_input = {
        "instructions": brief + " " + angle,
        "data": payload,
        "format": {
            "required_fields": ["subject", "preview_text", "body_text", "body_html"],
            "language": "da-DK"
        }
    }

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
    text = getattr(resp, "output_text", None)
    if not text and hasattr(resp, "output"):
        parts = []
        for item in resp.output:
            if hasattr(item, "content"):
                for c in item.content:
                    if getattr(c, "type", "") in ("output_text", "text"):
                        parts.append(getattr(c, "text", ""))
        text = "\n".join(parts)
    if not text:
        raise RuntimeError("Tomt svar fra modellen")
    data = json.loads(text)
    for k in ("subject", "preview_text", "body_text", "body_html"):
        data.setdefault(k, "")
    return data

def pick_variant(cvr: str, ab_choice: Optional[str]) -> str:
    if ab_choice in ("A", "B"):
        return ab_choice
    try:
        return "A" if (int(cvr) % 2 == 0) else "B"
    except Exception:
        return random.choice(["A", "B"])

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

def get_contacts_via_cf(cvr: str, domain: str, role_hint: str, size_bucket: str, enabled: bool):
    if not enabled or not cf_ok or not find_contacts:
        return []
    try:
        res = find_contacts(cvr=cvr, domain=domain, role_hint=role_hint, size_bucket=size_bucket)
        out = []
        for r in res or []:
            out.append({
                "name": r.get("name", ""),
                "email": r.get("email", ""),
                "role": r.get("role", ""),
                "source": r.get("source", "contact_finder"),
            })
        return out
    except Exception:
        return []

# ---------------------------
# PySide6 GUI
# ---------------------------
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QFileDialog, QGridLayout, QLabel, QLineEdit,
    QPushButton, QHBoxLayout, QVBoxLayout, QCheckBox, QComboBox, QSpinBox, QTextEdit,
    QTableWidget, QTableWidgetItem, QMessageBox,
    QTabWidget, QFormLayout, QGroupBox, QListWidget, QListWidgetItem, QHeaderView,
    QAbstractItemView
)
from PySide6.QtWidgets import QDoubleSpinBox

class QDoubleSpinBoxFixed(QDoubleSpinBox):
    def __init__(self, value: float = 0.0):
        super().__init__()
        self.setDecimals(2)
        self.setValue(value)

class VextoMailGenGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Vexto MailGen – PySide6 GUI v0.1")
        self.resize(1200, 800)

        # State
        self.df: Optional[pd.DataFrame] = None
        self.playbook: dict = DEFAULT_PLAYBOOK.copy()
        self.model_cfg = GenConfig()

        # Root
        cw = QWidget(self)
        self.setCentralWidget(cw)
        lay = QGridLayout(cw)

        # Banner + Tabs (kun disse i root)
        self.banner = QLabel(" ➜  1) Data  →  2) Kriterier  →  3) Udsendelse  →  4) Resultater")
        self.banner.setStyleSheet("font-weight:600; padding:6px;")
        lay.addWidget(self.banner, 0, 0, 1, 3)

        self.main_tabs = QTabWidget()
        lay.addWidget(self.main_tabs, 1, 0, 1, 3)

        # Byg faner
        self._build_tab_data()
        self._build_tab_criteria()
        self._build_tab_outreach()
        self._build_tab_results()

        # Gate
        self.main_tabs.setTabEnabled(0, True)
        self.main_tabs.setTabEnabled(1, False)
        self.main_tabs.setTabEnabled(2, False)
        self.main_tabs.setTabEnabled(3, False)
        self.main_tabs.setCurrentIndex(0)

        # Tabs-only i root
        self._enforce_tabs_only(lay)

        # Init: kriterier i Formular
        self.playbook = DEFAULT_PLAYBOOK.copy()
        self.builder_load_from_playbook()

        # Builder signaler
        self.seg_list.currentItemChanged.connect(self._seg_on_select)
        self.seg_add_btn.clicked.connect(self._seg_add)
        self.seg_del_btn.clicked.connect(self._seg_del)
        self.seg_dup_btn.clicked.connect(self._seg_dup)
        self.seg_update_btn.clicked.connect(self._seg_update_current)
        self.builder_apply_btn.clicked.connect(self.builder_apply_to_playbook)
        self.builder_load_btn.clicked.connect(self.builder_load_from_playbook)

    # ------- Tabs (1–4) -------
    def _build_tab_data(self):
        tab = QWidget(); v = QVBoxLayout(tab)

        # Paths
        row = QGridLayout()
        self.in_edit = QLineEdit(); self.in_btn = QPushButton("Vælg CSV…"); self.in_btn.clicked.connect(self.pick_in)
        self.out_edit = QLineEdit(); self.out_btn = QPushButton("Vælg output CSV…"); self.out_btn.clicked.connect(self.pick_out)
        self.env_edit = QLineEdit(); self.env_btn = QPushButton("Vælg .env…"); self.env_btn.clicked.connect(self.pick_env)
        row.addWidget(QLabel("Input CSV"), 0, 0); row.addWidget(self.in_edit, 0, 1); row.addWidget(self.in_btn, 0, 2)
        row.addWidget(QLabel("Output CSV"),1, 0); row.addWidget(self.out_edit,1, 1); row.addWidget(self.out_btn,1, 2)
        row.addWidget(QLabel(".env (valgfri)"),2, 0); row.addWidget(self.env_edit,2, 1); row.addWidget(self.env_btn,2, 2)
        v.addLayout(row)

        # Action
        self.btn_load = QPushButton("Indlæs data")
        self.btn_load.clicked.connect(self.load_and_summarize)
        v.addWidget(self.btn_load)

        # Summary
        self.summary_text = QTextEdit(); self.summary_text.setReadOnly(True)
        self.summary_text.setPlaceholderText("Opsummering vises her efter indlæsning…")
        v.addWidget(self.summary_text)

        self.main_tabs.addTab(tab, "1) Data")

    def _build_tab_criteria(self):
        tab = QWidget(); v = QVBoxLayout(tab)

        # Sti + knapper
        row = QHBoxLayout()
        self.play_path_edit = QLineEdit()
        self.play_load_btn = QPushButton("Indlæs kriterier…"); self.play_load_btn.clicked.connect(self.load_playbook)
        self.play_save_btn = QPushButton("Gem kriterier…");   self.play_save_btn.clicked.connect(self.save_playbook)
        row.addWidget(QLabel("Kriterier (kan gemmes/indlæses)")); row.addWidget(self.play_path_edit)
        row.addWidget(self.play_load_btn); row.addWidget(self.play_save_btn)
        v.addLayout(row)

        # Builder/Avanceret tabs
        self.tabs = QTabWidget()

        # --- Builder (Formular) ---
        builder = QWidget(); b_lay = QVBoxLayout(builder)

        # Overskrift over segmenter
        seg_hdr = QLabel("Segmentering af virksomheder:")
        seg_hdr.setStyleSheet("font-weight:600; padding:4px 0;")
        b_lay.addWidget(seg_hdr)

        # Liste af segmenter + knapper
        seg_row = QHBoxLayout()
        self.seg_list = QListWidget()
        self.seg_add_btn = QPushButton("Tilføj segment")
        self.seg_del_btn = QPushButton("Slet segment")
        self.seg_dup_btn = QPushButton("Klon segment")
        seg_btns = QVBoxLayout()
        for _b in (self.seg_add_btn, self.seg_del_btn, self.seg_dup_btn):
            seg_btns.addWidget(_b)
        seg_btns.addStretch(1)
        seg_row.addWidget(self.seg_list, 2)
        seg_btns_w = QWidget(); seg_btns_w.setLayout(seg_btns)
        seg_row.addWidget(seg_btns_w, 0)
        b_lay.addLayout(seg_row)

        # Målgruppe-egenskaber
        seg_box = QGroupBox("Målgruppe – egenskaber")
        form = QFormLayout(seg_box)
        self.seg_name = QLineEdit()

        # Multi-select over virksomhedsformer (fyldes fra CSV; ellers 'ikke mulig')
        self.seg_forms_multi = QListWidget()
        self.seg_forms_multi.setSelectionMode(QAbstractItemView.MultiSelection)
        self.seg_forms_multi.setFixedHeight(120)
        self._ensure_form_options()

        self.seg_size_min = QSpinBox(); self.seg_size_min.setRange(0, 100000)
        self.seg_size_max = QSpinBox(); self.seg_size_max.setRange(0, 100000)
        self.seg_target_role = QLineEdit()
        self.seg_channel = QComboBox(); self.seg_channel.addItems(["", "email", "linkedin_dm", "phone_call"])
        self.seg_update_btn = QPushButton("Opdater valgt segment")

        form.addRow("Navn", self.seg_name)
        form.addRow("Virksomhedsformer (vælg én eller flere)", self.seg_forms_multi)
        form.addRow("Min ansatte (0=ingen)", self.seg_size_min)
        form.addRow("Max ansatte (0=ingen)", self.seg_size_max)
        form.addRow("Target-rolle", self.seg_target_role)
        form.addRow("Kanal", self.seg_channel)
        form.addRow(self.seg_update_btn)
        b_lay.addWidget(seg_box)

        # (Regler & Sendetid er flyttet til fanen 'Udsendelse')

        # Builder actions
        b_actions = QHBoxLayout()
        self.builder_apply_btn = QPushButton("Formular → Kriterier")
        self.builder_load_btn  = QPushButton("Kriterier → Formular")
        b_actions.addWidget(self.builder_apply_btn); b_actions.addWidget(self.builder_load_btn); b_actions.addStretch(1)
        b_actions_w = QWidget(); b_actions_w.setLayout(b_actions)
        b_lay.addWidget(b_actions_w)

        # --- Avanceret (Raw JSON/YAML) ---
        raw = QWidget(); raw_lay = QVBoxLayout(raw)
        self.play_text = QTextEdit()
        self.play_text.setPlaceholderText(json.dumps(DEFAULT_PLAYBOOK, indent=2, ensure_ascii=False))
        self.play_text.setText(json.dumps(DEFAULT_PLAYBOOK, indent=2, ensure_ascii=False))
        raw_lay.addWidget(self.play_text)

        self.tabs.addTab(builder, "Formular")
        self.tabs.addTab(raw, "Avanceret (JSON/YAML)")

        v.addWidget(self.tabs)
        self.main_tabs.addTab(tab, "2) Kriterier")

    def _build_tab_outreach(self):
        tab = QWidget(); v = QVBoxLayout(tab)

        # Options (kontakt-type, A/B, målgruppe, AI, kreat., maks)
        opt = QHBoxLayout()
        self.cb_dry = QCheckBox("Testkørsel (uden AI-forbrug)")
        self.cb_cf  = QCheckBox("Brug Kontaktfinder (hent kontaktperson)")
        self.mode_combo = QComboBox()
        self.mode_combo.addItem("Begge typer", "both")
        self.mode_combo.addItem("Analyse (har hjemmeside)", "analyse")
        self.mode_combo.addItem("Ingen hjemmeside", "no_site")
        self.ab_combo = QComboBox(); self.ab_combo.addItems(["auto","A","B"])
        self.segment_edit = QLineEdit(); self.segment_edit.setPlaceholderText("målgruppe-navn (fx micro/smb/corp) – tom = alle")
        self.model_edit = QLineEdit(self.model_cfg.model)
        self.temp_spin = QDoubleSpinBoxFixed(0.3); self.temp_spin.setRange(0.0, 1.0); self.temp_spin.setSingleStep(0.1)
        self.maxrows_spin = QSpinBox(); self.maxrows_spin.setRange(0, 1000000); self.maxrows_spin.setValue(0)

        for w,lbl in [(self.cb_dry,""),(self.cb_cf,""),(self.mode_combo,"Kontakt-type"),(self.ab_combo,"A/B"),
                      (self.segment_edit,"Målgruppe"),(self.model_edit,"AI-model"),(self.temp_spin,"Kreativitet"),
                      (self.maxrows_spin,"Maks. rækker")]:
            if lbl: opt.addWidget(QLabel(lbl))
            opt.addWidget(w)
        v.addLayout(opt)

        self.mode_combo.setToolTip("Kontakt virksomheder med hjemmeside (Analyse), uden hjemmeside – eller begge.")
        self.segment_edit.setToolTip("Kør kun for en bestemt målgruppe fra Kriterier-fanen. Lad tom for alle.")
        self.cb_dry.setToolTip("Kører uden AI-kald – genererer placeholders så du kan teste flowet gratis.")
        self.cb_cf.setToolTip("Brug Kontaktfinder til at foreslå en kontaktperson (kræver opsætning).")

        # Regler for kontakt
        rule_box = QGroupBox("Regler for kontakt")
        rform = QFormLayout(rule_box)
        self.rule_analyse_if_live = QCheckBox("Analyse hvis live website"); self.rule_analyse_if_live.setChecked(True)
        self.rule_no_site_login_4xx = QCheckBox("No-site hvis login-only eller 4xx/5xx"); self.rule_no_site_login_4xx.setChecked(True)
        self.rule_fallback = QComboBox(); self.rule_fallback.addItems(["analyse","no_site"])
        rform.addRow(self.rule_analyse_if_live)
        rform.addRow(self.rule_no_site_login_4xx)
        rform.addRow("Fallback-mode", self.rule_fallback)
        v.addWidget(rule_box)

        # Sendetid & Kanalrækkefølge
        sched_box = QGroupBox("Sendetid & Kanalrækkefølge")
        sform = QFormLayout(sched_box)
        self.sched_window = QComboBox(); self.sched_window.addItems(
            ["workdays_08_16","early_07_09","lunch_11_13","afternoon_14_17","evening_18_21"]
        )
        self.sched_after = QSpinBox(); self.sched_after.setRange(0, 168)
        self.channel_prio = QLineEdit("email,linkedin_dm")
        sform.addRow("Sendetidsrum", self.sched_window)
        sform.addRow("Udskyd timer", self.sched_after)
        sform.addRow("Kanalrækkefølge (komma)", self.channel_prio)
        v.addWidget(sched_box)

        # Actions
        hb = QHBoxLayout()
        self.btn_preview = QPushButton("Forhåndsvis 10 mails"); self.btn_preview.clicked.connect(lambda: self.generate(sample_n=10, write=False))
        self.btn_generate = QPushButton("Gem mails som CSV");   self.btn_generate.clicked.connect(lambda: self.generate(sample_n=0, write=True))
        hb.addWidget(self.btn_preview); hb.addWidget(self.btn_generate); hb.addStretch(1)
        hbw = QWidget(); hbw.setLayout(hb)
        v.addWidget(hbw)

        self.main_tabs.addTab(tab, "3) Udsendelse")

    def _build_tab_results(self):
        tab = QWidget(); v = QVBoxLayout(tab)

        # KPI
        kpi = QHBoxLayout()
        self.kpi_total = QLabel("Total: –")
        self.kpi_analyse = QLabel("Analyse: –")
        self.kpi_nosite = QLabel("No-site: –")
        self.kpi_with_contact = QLabel("Med kontakt: –")
        for w in (self.kpi_total, self.kpi_analyse, self.kpi_nosite, self.kpi_with_contact):
            w.setStyleSheet("padding:4px;")
            kpi.addWidget(w)
        kpi.addStretch(1)
        v.addLayout(kpi)

        self.kpi_total.setToolTip("Antal genererede rækker i denne kørsel")
        self.kpi_analyse.setToolTip("Antal mails markeret som 'Analyse' (har hjemmeside)")
        self.kpi_nosite.setToolTip("Antal mails til virksomheder uden hjemmeside")
        self.kpi_with_contact.setToolTip("Hvor mange rækker har fundet en kontaktperson")

        # Log
        self.log = QTextEdit(); self.log.setReadOnly(True)
        v.addWidget(self.log)

        # Tabel
        self.table = QTableWidget(0, len(OUT_COLUMNS))
        self.table.setHorizontalHeaderLabels(OUT_HEADERS_DK)
        self.table.setSortingEnabled(True)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        v.addWidget(self.table)

        self.main_tabs.addTab(tab, "4) Resultater")

    # ---------- Tabs-only i root ----------
    def _enforce_tabs_only(self, root_layout):
        keep = {self.banner, self.main_tabs}
        def _clear(layout):
            for i in reversed(range(layout.count())):
                item = layout.itemAt(i)
                if item is None:
                    continue
                w = item.widget()
                if w is not None:
                    if w not in keep:
                        layout.removeWidget(w)
                        w.setVisible(False)
                        w.setParent(None)
                        w.deleteLater()
                    continue
                sub = item.layout()
                if sub is not None:
                    _clear(sub)
                    layout.removeItem(sub)
                    continue
                layout.removeItem(item)
        _clear(root_layout)
        try:
            root_layout.addWidget(self.banner, 0, 0, 1, 3)
            root_layout.addWidget(self.main_tabs, 1, 0, 1, 3)
            root_layout.setRowStretch(0, 0)
            root_layout.setRowStretch(1, 1)
        except Exception:
            pass

    def showEvent(self, event):
        try:
            root = self.centralWidget().layout()
            if root is not None:
                self._enforce_tabs_only(root)
        except Exception:
            pass
        super().showEvent(event)

    # ---------- Kriterier: virksomhedsform-muligheder ----------
    def _ensure_form_options(self):
        """Udfyld liste over virksomhedsformer fra CSV. Hvis intet → 'ikke mulig' (disabled)."""
        if not hasattr(self, "seg_forms_multi"):
            return
        self.seg_forms_multi.clear()
        forms = []
        try:
            if self.df is not None and CSV_FIELDS["form_desc"] in self.df.columns:
                forms = sorted(set(str(x).strip() for x in self.df[CSV_FIELDS["form_desc"]].dropna().tolist() if str(x).strip()))
        except Exception:
            forms = []
        if not forms:
            itm = QListWidgetItem("ikke mulig")
            itm.setFlags(itm.flags() & ~Qt.ItemIsSelectable)
            self.seg_forms_multi.addItem(itm)
            return
        for f in forms:
            self.seg_forms_multi.addItem(QListWidgetItem(f))

    # ------- UI helpers -------
    def pick_in(self):
        path, _ = QFileDialog.getOpenFileName(self, "Vælg CSV", "", "CSV (*.csv)")
        if path:
            self.in_edit.setText(path)

    def pick_out(self):
        path, _ = QFileDialog.getSaveFileName(self, "Vælg output CSV", "mails_out.csv", "CSV (*.csv)")
        if path:
            self.out_edit.setText(path)

    def pick_env(self):
        path, _ = QFileDialog.getOpenFileName(self, "Vælg .env", "", "ENV (*.env);;Alle (*)")
        if path:
            self.env_edit.setText(path)

    def load_playbook(self):
        path, _ = QFileDialog.getOpenFileName(self, "Indlæs kriterier", "", "YAML/JSON (*.yml *.yaml *.json);;Alle (*)")
        if not path:
            return
        try:
            with open(path, "r", encoding="utf-8") as f:
                if path.lower().endswith((".yml", ".yaml")) and yaml_ok:
                    self.playbook = yaml.safe_load(f) or DEFAULT_PLAYBOOK
                else:
                    self.playbook = json.load(f)
                self.play_text.setText(json.dumps(self.playbook, indent=2, ensure_ascii=False))
                self.play_path_edit.setText(path)
                self.builder_load_from_playbook()
                self.log_append(f"Kriterier indlæst: {path}")
        except Exception as e:
            self.err(f"Kunne ikke indlæse kriterier: {e}")

    def save_playbook(self):
        # Indsamling afhænger af aktiv tab
        if self.tabs.currentIndex() == 0:  # Formular
            self.builder_apply_to_playbook()
            pb = self.playbook
        else:  # Avanceret
            text = self.play_text.toPlainText().strip()
            if not text:
                self.err("Kriterie-tekst er tom")
                return
            try:
                pb = json.loads(text)
            except Exception:
                if yaml_ok:
                    try:
                        pb = yaml.safe_load(text)
                    except Exception as e:
                        self.err(f"Ugyldigt JSON/YAML: {e}")
                        return
                else:
                    self.err("Ugyldigt JSON, og YAML er ikke installeret (pyyaml)")
                    return
        # gem
        path, _ = QFileDialog.getSaveFileName(self, "Gem kriterier", self.play_path_edit.text() or "kriterier.json", "JSON/YAML (*.json *.yml *.yaml)")
        if not path:
            return
        try:
            with open(path, "w", encoding="utf-8") as f:
                if path.lower().endswith((".yml", ".yaml")) and yaml_ok:
                    yaml.safe_dump(pb, f, allow_unicode=True, sort_keys=False)
                else:
                    json.dump(pb, f, ensure_ascii=False, indent=2)
            self.playbook = pb
            self.play_path_edit.setText(path)
            self.log_append(f"Kriterier gemt: {path}")
        except Exception as e:
            self.err(f"Kunne ikke gemme kriterier: {e}")

    # ---------- Builder helpers ----------
    def _seg_to_dict(self) -> dict:
        # Læs valgte virksomhedsformer fra multi-select
        forms: List[str] = []
        if hasattr(self, "seg_forms_multi"):
            for i in range(self.seg_forms_multi.count()):
                it = self.seg_forms_multi.item(i)
                if it.isSelected():
                    forms.append(it.text())
        seg = {
            "name": self.seg_name.text().strip() or "segment",
            "forms": forms,
            "target_role": self.seg_target_role.text().strip() or "",
            "channel": self.seg_channel.currentText().strip() or ""
        }
        minv = int(self.seg_size_min.value()); maxv = int(self.seg_size_max.value())
        if minv: seg["size_min"] = minv
        if maxv: seg["size_max"] = maxv
        return seg

    def _seg_fill_form(self, seg: dict):
        self.seg_name.setText(seg.get("name",""))
        want = set(seg.get("forms", []) or [])
        if hasattr(self, "seg_forms_multi"):
            for i in range(self.seg_forms_multi.count()):
                it = self.seg_forms_multi.item(i)
                it.setSelected(it.text() in want)
        self.seg_target_role.setText(seg.get("target_role",""))
        self.seg_channel.setCurrentText(seg.get("channel",""))
        self.seg_size_min.setValue(int(seg.get("size_min", 0) or 0))
        self.seg_size_max.setValue(int(seg.get("size_max", 0) or 0))

    def _seg_on_select(self, cur: QListWidgetItem, prev: QListWidgetItem):
        if cur is None: return
        seg = cur.data(Qt.UserRole) or {}
        self._seg_fill_form(seg)

    def _seg_add(self):
        seg = {"name":"nyt_segment","forms":[],"target_role":"","channel":""}
        item = QListWidgetItem(seg["name"]); item.setData(Qt.UserRole, seg)
        self.seg_list.addItem(item); self.seg_list.setCurrentItem(item)

    def _seg_del(self):
        row = self.seg_list.currentRow()
        if row >= 0: self.seg_list.takeItem(row)

    def _seg_dup(self):
        cur = self.seg_list.currentItem()
        if not cur: return
        seg = dict(cur.data(Qt.UserRole) or {})
        seg["name"] = (seg.get("name","") or "segment") + "_copy"
        item = QListWidgetItem(seg["name"]); item.setData(Qt.UserRole, seg)
        self.seg_list.addItem(item); self.seg_list.setCurrentItem(item)

    def _seg_update_current(self):
        cur = self.seg_list.currentItem()
        if not cur: return
        seg = self._seg_to_dict()
        cur.setText(seg["name"])
        cur.setData(Qt.UserRole, seg)

    def builder_apply_to_playbook(self):
        # indsamle segments
        segments = []
        for i in range(self.seg_list.count()):
            it = self.seg_list.item(i)
            segments.append(it.data(Qt.UserRole) or {})
        # regler og scheduling (fra Udsendelse-fanen)
        pb = {
            "segments": segments,
            "outreach_rules": {
                "analyse_if_live": self.rule_analyse_if_live.isChecked(),
                "no_site_if_login_or_4xx_5xx": self.rule_no_site_login_4xx.isChecked(),
                "fallback_mode": self.rule_fallback.currentText().strip() or "analyse"
            },
            "scheduling": {
                "send_window": self.sched_window.currentText().strip() or "workdays_08_16",
                "send_after_hours": int(self.sched_after.value())
            },
            "channel_priority": [x.strip() for x in self.channel_prio.text().split(",") if x.strip()]
        }
        self.playbook = pb
        # sync Raw
        self.play_text.setText(json.dumps(self.playbook, indent=2, ensure_ascii=False))
        self.log_append("Kriterier opdateret fra Formular.")

    def builder_load_from_playbook(self):
        pb = self.playbook or DEFAULT_PLAYBOOK
        # segments
        self.seg_list.clear()
        for seg in pb.get("segments", []):
            item = QListWidgetItem(seg.get("name","segment"))
            item.setData(Qt.UserRole, seg)
            self.seg_list.addItem(item)
        if self.seg_list.count() == 0:
            self._seg_add()
        self.seg_list.setCurrentRow(0)
        self._seg_fill_form(self.seg_list.item(0).data(Qt.UserRole) or {})
        # rules (Udsendelse)
        r = pb.get("outreach_rules", {})
        self.rule_analyse_if_live.setChecked(bool(r.get("analyse_if_live", True)))
        self.rule_no_site_login_4xx.setChecked(bool(r.get("no_site_if_login_or_4xx_5xx", True)))
        self.rule_fallback.setCurrentText(str(r.get("fallback_mode","analyse")))
        # sched
        s = pb.get("scheduling", {})
        self.sched_window.setCurrentText(str(s.get("send_window","workdays_08_16")))
        self.sched_after.setValue(int(s.get("send_after_hours", 0) or 0))
        # channel prio
        self.channel_prio.setText(", ".join(pb.get("channel_priority", ["email"])))
        # sync Raw
        self.play_text.setText(json.dumps(pb, indent=2, ensure_ascii=False))

    # ---------- Data indlæsning ----------
    def load_and_summarize(self):
        inp = _resolve_path(self.in_edit.text())
        if not inp or not os.path.exists(inp):
            self.err(f"Input findes ikke: {inp}")
            return
        try:
            self.df = read_csv(inp)
        except Exception as e:
            self.err(f"CSV load-fejl: {e}")
            return
        # Opsummering
        total = len(self.df)
        has_site = self.df.get(CSV_FIELDS["website"], pd.Series([], dtype=str)).fillna("").str.strip().ne("").sum()
        live_true = self.df.get(CSV_FIELDS["live"], pd.Series([], dtype=str)).astype(str).str.lower().isin(["true","1","yes"]).sum()
        no_email = self.df.get(CSV_FIELDS["email"], pd.Series([], dtype=str)).fillna("").str.strip().eq("").sum()
        top_forms = self.df.get(CSV_FIELDS["form_desc"], pd.Series([], dtype=str)).fillna("").value_counts().head(5)
        msg = [f"Rækker: {total}", f"Med hjemmeside: {has_site}", f"Live=True: {live_true}", f"Mangler email: {no_email}"]
        msg.append("Top5 virksomhedsformer:")
        for k,v in top_forms.items():
            msg.append(f"  - {k}: {v}")

        # Opdatér Kriterier → virksomhedsformer
        self._ensure_form_options()

        # Gate + skift
        self.summary_text.setPlainText("\n".join(msg))
        self.main_tabs.setTabEnabled(1, True)  # Kriterier
        self.main_tabs.setTabEnabled(2, True)  # Udsendelse
        self.main_tabs.setTabEnabled(3, True)  # Resultater
        self.main_tabs.setCurrentIndex(1)

    # ---------- Generering ----------
    def generate(self, sample_n: int = 0, write: bool = False):
        try:
            start_ts = datetime.now()

            # .env + model
            envp = self.env_edit.text().strip() or None
            load_env(envp)
            self.model_cfg.model = self.model_edit.text().strip() or "gpt-5"
            self.model_cfg.temperature = float(self.temp_spin.value())
            self.model_cfg.base_url = os.getenv("OPENAI_BASE_URL") or None

            # API-krav (kun hvis ikke testkørsel)
            if not self.cb_dry.isChecked():
                if not openai_ok:
                    self.err("OpenAI-SDK er ikke installeret (pip install openai) eller kunne ikke importeres.")
                    return
                if not os.getenv("OPENAI_API_KEY"):
                    self.err("OPENAI_API_KEY mangler. Angiv i .env eller miljøet.")
                    return

            # Kriterier (Formular → playbook)
            if hasattr(self, "builder_apply_btn"):
                self.builder_apply_to_playbook()
            playbook = self.playbook or DEFAULT_PLAYBOOK

            # Data
            if self.df is None:
                self.load_and_summarize()
                if self.df is None:
                    return
            df = self.df.copy()

            # Flags
            dry = self.cb_dry.isChecked()
            use_cf = self.cb_cf.isChecked()
            mode_token = self.mode_combo.currentData() or "both"
            ab_ch = self.ab_combo.currentText()
            if ab_ch == "auto":
                ab_ch = None
            segment_only = self.segment_edit.text().strip() or None
            max_rows = int(self.maxrows_spin.value()) or None
            mode_filter = None if mode_token == "both" else mode_token

            # Generér
            rows = self.process_rows(
                df, self.model_cfg, max_rows, ab_ch, dry, mode_filter, playbook, use_cf, segment_only
            )

            if not rows:
                self.log_append("[INFO] Ingen rækker matchede dine kriterier (målgruppe/kontakt-type/maks). Justér i 2) Kriterier eller 3) Udsendelse.")
                self.main_tabs.setCurrentIndex(1)
                return

            # Preview
            visible_n = sample_n if sample_n > 0 else min(len(rows), 200)
            self.populate_table(rows[:visible_n])

            # Skriv CSV
            if write:
                outp = _resolve_path(self.out_edit.text())
                if not outp:
                    self.err("Angiv output CSV-sti")
                    return
                try:
                    self.write_out(outp, rows)
                    self.log_append(f"Skrev {len(rows)} rækker til: {outp}")
                except Exception as e:
                    self.err(f"Skrive-fejl: {e}")
                    return
            else:
                self.log_append(f"Genereret {len(rows)} rækker (ikke skrevet – preview viser {visible_n}).")

            # KPI + log
            self.main_tabs.setTabEnabled(3, True)
            self.main_tabs.setCurrentIndex(3)
            self._update_results_summary(rows)

            analyse = sum(1 for r in rows if r.get("outreach_mode") == "analyse")
            nosite  = sum(1 for r in rows if r.get("outreach_mode") == "no_site")
            with_contact = sum(1 for r in rows if (r.get("contact_email") or r.get("contact_name")))
            by_seg = {}
            by_channel = {}
            ab_count = {"A": 0, "B": 0}
            for r in rows:
                by_seg[r.get("notes", r.get("size_bucket",""))] = by_seg.get(r.get("notes", r.get("size_bucket","")), 0) + 1
                by_channel[r.get("channel","")] = by_channel.get(r.get("channel",""), 0) + 1
                ab = r.get("template_variant","")
                if ab in ab_count:
                    ab_count[ab] += 1

            dur = (datetime.now() - start_ts).total_seconds()
            lines = [
                "[OVERBLIK]",
                f"- Varighed: {dur:.1f}s | Total: {len(rows)} | Analyse: {analyse} | No-site: {nosite} | Med kontakt: {with_contact}",
                f"- Kanalfordeling: " + ", ".join(f"{k or '—'}={v}" for k,v in sorted(by_channel.items(), key=lambda x: -x[1])),
                f"- Segmenter: " + ", ".join(f"{k or '—'}={v}" for k,v in sorted(by_seg.items(), key=lambda x: -x[1])),
                f"- A/B: A={ab_count['A']} | B={ab_count['B']}",
                f"- Preview viste: {visible_n} rækker"
            ]
            self.log_append("\n".join(lines))

        except Exception as e:
            self.err(f"Fejl i generering: {e}\n{traceback.format_exc()}")

    # ---------- Kernegenerering ----------
    def process_rows(self, df: pd.DataFrame, cfg: GenConfig, max_rows: Optional[int], ab_choice: Optional[str], dry_run: bool,
                     mode_filter: Optional[str], playbook: dict, use_contact_finder: bool, only_segment: Optional[str]) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        n = len(df) if not max_rows else min(max_rows, len(df))
        for i in range(n):
            row = df.iloc[i].fillna("")
            company_name = str(row.get(CSV_FIELDS["name"], "")).strip()
            cvr = str(row.get(CSV_FIELDS["cvr"], "")).strip()
            website = str(row.get(CSV_FIELDS["website"], "")).strip()
            live = bool(row.get(CSV_FIELDS["live"], False))
            remark = str(row.get(CSV_FIELDS["remark"], "")).strip()
            form_desc = str(row.get(CSV_FIELDS["form_desc"], "")).strip()
            size = row.get(CSV_FIELDS["size"], None)
            sb = size_bucket(size if size != "" else None)

            seg = match_segment(playbook, form_desc, sb)
            if only_segment and seg.get("name") != only_segment:
                continue

            mode = decide_outreach_mode(website, live, remark)
            rules = playbook.get("outreach_rules", {})
            if not rules.get("analyse_if_live", True) and mode == "analyse":
                mode = rules.get("fallback_mode", "analyse")
            if mode_filter and mode != mode_filter:
                continue

            target_role = seg.get("target_role") or choose_target_role(sb, form_desc)
            linkedin_q = build_linkedin_query(company_name, sb, form_desc)
            variant = pick_variant(cvr, ab_choice)

            channel = choose_channel(seg, playbook)
            send_window, send_after = compute_timing(playbook)

            payload = make_prompt_payload(row.to_dict(), mode, target_role, sb)

            if dry_run or not openai_ok:
                subject = f"[DRY][{mode}/{variant}] Gevinster til {company_name or 'jeres virksomhed'}"
                preview = "Kort værditilbud fra Vexto (demo/plan/fast pris)."
                body_text = f"(Dry-run) Hej {company_name or 'der'},\n\nKort oplæg...\n\nMvh Vexto"
                body_html = f"<p>(Dry-run)</p><p>Hej {company_name or 'der'},</p><p>Kort oplæg...</p><p>Mvh Vexto</p>"
            else:
                gen = generate_with_openai(cfg, payload, variant)
                subject, preview, body_text, body_html = gen["subject"], gen["preview_text"], gen["body_text"], gen["body_html"]

            contacts = get_contacts_via_cf(cvr=cvr, domain=canonical_domain(website), role_hint=target_role, size_bucket=sb, enabled=use_contact_finder)
            best = contacts[0] if contacts else {"name": "", "email": "", "role": "", "source": ""}

            out.append(dict(
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
                channel=channel,
                send_window=send_window,
                send_after=send_after,
                contact_name=best.get("name", ""),
                contact_email=best.get("email", ""),
                contact_role=best.get("role", ""),
                contact_source=best.get("source", ""),
                subject=subject.strip(),
                preview_text=preview.strip(),
                body_text=body_text.strip(),
                body_html=body_html.strip(),
                model=cfg.model,
                temperature=cfg.temperature,
                notes=seg.get("name", "")
            ))
        return out

    # ---------- Resultat-visning ----------
    def _update_results_summary(self, rows: List[Dict[str, Any]]):
        total = len(rows)
        analyse = sum(1 for r in rows if r.get("outreach_mode") == "analyse")
        nosite  = sum(1 for r in rows if r.get("outreach_mode") == "no_site")
        with_contact = sum(1 for r in rows if (r.get("contact_email") or r.get("contact_name")))
        self.kpi_total.setText(f"Total: {total}")
        self.kpi_analyse.setText(f"Analyse: {analyse}")
        self.kpi_nosite.setText(f"No-site: {nosite}")
        self.kpi_with_contact.setText(f"Med kontakt: {with_contact}")

    def write_out(self, path: str, rows: List[Dict[str, Any]]):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=OUT_COLUMNS)
            w.writeheader()
            for r in rows:
                w.writerow({k: r.get(k, "") for k in OUT_COLUMNS})

    def populate_table(self, rows: List[Dict[str, Any]]):
        self.table.setSortingEnabled(False)
        self.table.setRowCount(0); self.table.setRowCount(len(rows))
        for i, r in enumerate(rows):
            for j, col in enumerate(OUT_COLUMNS):
                val = str(r.get(col, ""))
                item = QTableWidgetItem(val)
                if col in ("company_name","outreach_mode","channel","contact_name","contact_email","subject"):
                    font = item.font(); font.setBold(True); item.setFont(font)
                self.table.setItem(i, j, item)
        self.table.resizeColumnsToContents()
        self.table.setSortingEnabled(True)

    def log_append(self, text: str):
        self.log.append(text)
        self.log.ensureCursorVisible()

    def err(self, text: str):
        self.log_append(f"[FEJL] {text}")
        QMessageBox.critical(self, "Fejl", text)

def main():
    app = QApplication(sys.argv)
    w = VextoMailGenGUI()
    w.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
