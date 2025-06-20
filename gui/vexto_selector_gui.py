"""
Vexto Selector GUI
------------------
En simpel Tk-dreven selector der lader brugeren vælge:
  • Postnumre (multi-select)
  • Én branchekode
Returnerer Tuple[List[int], str]  – eller None hvis man trykker Annullér.
"""
from __future__ import annotations

import tkinter as tk
from tkinter import messagebox, ttk
from pathlib import Path
import pandas as pd


# --- filplaceringer ----------------------------------------------------------
DATA_DIR       = Path(__file__).resolve().parent.parent / "data"
POSTNUMRE_CSV  = DATA_DIR / "postnumre_gui.csv"
BRANCHER_CSV   = DATA_DIR / "brancher_gui.csv"


# --------------------------------------------------------------------------- #
# Helper-funktioner
# --------------------------------------------------------------------------- #
def _load_lists() -> tuple[list[int], list[str]]:
    """
    Indlæser postnumre & branchekoder fra CSV.  Tåler 3+ forskellige kolonnenavne.
    Returnerer (postnumre, branchekoder)
    """
    # ------------ Postnumre --------------------------------------------------
    df_post = pd.read_csv(POSTNUMRE_CSV, sep=";")
    post_col = next(
        (c for c in df_post.columns
         if c.lower().replace(" ", "") in ("postnummer", "postnr", "postnr.")),
        None,
    )
    if post_col is None:
        raise KeyError(
            f"Kunne ikke finde kolonnen 'postnummer' i {POSTNUMRE_CSV.name}"
        )
    postnumre = df_post[post_col].astype(int).tolist()

    # ------------ Branchekoder ----------------------------------------------
    df_branche = pd.read_csv(BRANCHER_CSV, sep=";")
    bran_col = next(
         (c for c in df_branche.columns
          if c.lower().replace(" ", "") in ("branchekode", "branche", "kode")),
         None,
     )
    if bran_col is None:
        raise KeyError(
            f"Kunne ikke finde kolonnen 'branchekode' i {BRANCHER_CSV.name}"
        )
    brancher = df_branche[bran_col].astype(str).tolist()
    return postnumre, brancher


# --------------------------------------------------------------------------- #
# GUI-logik
# --------------------------------------------------------------------------- #
def run_selector() -> tuple[list[int], str] | None:
    postnumre, brancher = _load_lists()

    root = tk.Tk()
    root.title("Vexto Selector")
    root.geometry("700x500")          # større initial­vindue
    root.columnconfigure((0, 1), weight=1)
    root.rowconfigure(0,  weight=1)

    # Rammer / frames
    frm_post = ttk.LabelFrame(root, text="Vælg postnumre")
    frm_post.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")
    frm_bran = ttk.LabelFrame(root, text="Vælg branchekode")
    frm_bran.grid(row=0, column=1, padx=10, pady=10, sticky="nsew")

      
	
    # Postnumre – Listbox med multiselect
    listbox = tk.Listbox(frm_post, selectmode="multiple",
                         height=20, width=8)
    listbox.pack(fill="both", expand=True, padx=5, pady=5)
    for pn in postnumre:
        listbox.insert(tk.END, pn)

    # ------------ Søgefelt til postnumre -----------------------------------
    search_post = tk.StringVar()
    ttk.Entry(frm_post, textvariable=search_post
             ).pack(fill="x", padx=5, pady=(10, 5))
    def _filter_post(*_):
        q = search_post.get().strip()
        listbox.delete(0, tk.END)
        for pn in postnumre:
            if q in str(pn):
                listbox.insert(tk.END, pn)

    search_post.trace_add("write", _filter_post)

    
    # Branchekoder – Combobox (single-select)
    cbo = ttk.Combobox(frm_bran, state="readonly",
                       values=brancher, width=15)
    cbo.pack(fill="x", padx=5, pady=5)
    cbo.set(brancher[0])

    # ------------ Søgefelt til branchekoder -------------------------------
    search_bran = tk.StringVar()
    ttk.Entry(frm_bran, textvariable=search_bran
             ).pack(fill="x", padx=5, pady=(10, 5))

    def _filter_bran(*_):
        q = search_bran.get().strip().lower()
        filt = [b for b in brancher if q in b.lower()]
        cbo["values"] = filt if filt else brancher
        if cbo.get() not in cbo["values"]:
            cbo.set(filt[0] if filt else "")

    search_bran.trace_add("write", _filter_bran)

    # ---- knapper -----------------------------------------------------------
    def _ok():
        sel_idx = listbox.curselection()
        if not sel_idx:
            messagebox.showwarning("Manglende valg", "Vælg mindst ét postnummer.")
            return
        root.quit()

    ttk.Button(root, text="OK", command=_ok, width=10).grid(row=1, column=0, pady=5)
    ttk.Button(root, text="Annullér", command=root.destroy, width=10).grid(row=1, column=1, pady=5)

    root.mainloop()

    # Blev GUI’en lukket via Annullér/hjørne-X?
    if not listbox.curselection():
        return None

    valgte_postnumre = [int(listbox.get(i)) for i in listbox.curselection()]
    valgt_branche    = cbo.get()
    return valgte_postnumre, valgt_branche
