"""
vexto_selector_gui.py
---------------------
En simpel Tkinter GUI til at vælge postnumre og branchekoder,
inklusive søgefunktionalitet, multi-select (Shift/Ctrl) og knapper
for Markér alle / Ryd. Intervalvalg virker på de synlige (filtrerede) rækker.
"""
import tkinter as tk
from tkinter import ttk, messagebox
from pathlib import Path
import pandas as pd
import re

def _parse_code(text: str) -> int | None:
    """
    Robust ekstraktion af kode fra listbox-linje.
    Tåler formater som '423300.0 - ...', '423300 - ...', '2300 København S' mv.
    """
    if not text:
        return None
    head = text.split(" - ")[0].strip()
    head = head.split(" ")[0].strip()
    digits = "".join(re.findall(r"\d+", head))
    if not digits:
        return None
    try:
        return int(digits)
    except Exception:
        return None

# --- Definer stier korrekt fra projektets rod ---
PROJECT_DIR = Path(__file__).resolve().parents[3]
DATA_DIR = PROJECT_DIR / "data"

POSTNUMRE_CSV = DATA_DIR / "postnumre_gui.csv"
BRANCHER_CSV = DATA_DIR / "brancher_gui.csv"

def _load_lists():
    """Henter postnumre og branchekoder fra CSV-filer."""
    try:
        df_post = pd.read_csv(POSTNUMRE_CSV, sep=";")
        df_post.columns = [col.strip().lower() for col in df_post.columns]

        df_brancher = pd.read_csv(BRANCHER_CSV, sep=";")
        df_brancher.columns = [col.strip().lower() for col in df_brancher.columns]

        # Bruger kolonnerne fra dine filer
        postnumre = sorted([f"{row.postnummer} {row.bynavn}" for _, row in df_post.iterrows()])
        brancher = sorted([f"{row.branchekode} - {row.nace_titel_dk}" for _, row in df_brancher.iterrows()])

        return postnumre, brancher
    except FileNotFoundError as e:
        messagebox.showerror("Fejl", f"Kunne ikke finde en nødvendig datafil:\n{e}")
        return None, None
    except Exception as e:
        messagebox.showerror("Fejl", f"En uventet fejl opstod under indlæsning af data:\n{e}")
        return None, None


class SelectorApp(tk.Tk):
    def __init__(self, postnumre, brancher):
        super().__init__()
        self.title("Vexto Parameter Vælger")
        self.geometry("900x620")  # lidt bredere for knapper

        self.postnumre = postnumre  # Den fulde liste
        self.brancher = brancher    # Den fulde liste
        self.result = None

        self.create_widgets()
        self._bind_shortcuts()

    # ---------- UI ----------
    def create_widgets(self):
        main_frame = ttk.Frame(self, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)

        # --- Postnummer sektion ---
        post_frame = ttk.LabelFrame(main_frame, text="Vælg Postnumre", padding="10")
        post_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))

        # Listbox for postnumre (EXTENDED = Shift/Ctrl multi-select)
        self.post_listbox = tk.Listbox(post_frame, selectmode=tk.EXTENDED, exportselection=False)
        self.post_listbox.pack(fill=tk.BOTH, expand=True)

        # Kontrolsektion under listbox: Søg + Interval + Knapper
        post_controls = ttk.Frame(post_frame)
        post_controls.pack(fill=tk.X, pady=(8, 0))

        # Søg (postnumre)
        search_post_row = ttk.Frame(post_controls)
        search_post_row.pack(fill=tk.X, pady=(0, 6))
        ttk.Label(search_post_row, text="Søg:").pack(side=tk.LEFT)
        self.post_search_var = tk.StringVar()
        post_search_entry = ttk.Entry(search_post_row, textvariable=self.post_search_var)
        post_search_entry.pack(side=tk.LEFT, fill=tk.X, expand=True)
        post_search_entry.bind("<KeyRelease>", self.on_post_search)
        # Også ved paste/klip/programmatiske ændringer
        self.post_search_var.trace_add("write", lambda *_: self.on_post_search())
        post_search_entry.bind("<<Paste>>", self.on_post_search)
        post_search_entry.bind("<<Cut>>", self.on_post_search)

        # Interval (postnumre)
        post_range_row = ttk.Frame(post_controls)
        post_range_row.pack(fill=tk.X, pady=(0, 6))
        ttk.Label(post_range_row, text="Fra:").pack(side=tk.LEFT)
        self.post_from_var = tk.StringVar()
        ttk.Entry(post_range_row, width=8, textvariable=self.post_from_var).pack(side=tk.LEFT, padx=(4, 8))
        ttk.Label(post_range_row, text="Til:").pack(side=tk.LEFT)
        self.post_to_var = tk.StringVar()
        ttk.Entry(post_range_row, width=8, textvariable=self.post_to_var).pack(side=tk.LEFT, padx=(4, 8))
        ttk.Button(post_range_row, text="Vælg interval", command=self.select_post_interval).pack(side=tk.LEFT)

        # Knaprække (postnumre)
        post_btns = ttk.Frame(post_controls)
        post_btns.pack(fill=tk.X, pady=(0, 0))
        ttk.Button(post_btns, text="Markér alle", command=self.select_all_post).pack(side=tk.LEFT)
        ttk.Button(post_btns, text="Ryd", command=self.clear_post).pack(side=tk.LEFT, padx=(6, 0))

        # Fyld listen fra start (efter widgets er oprettet)
        self.on_post_search()

        # --- Branchekode sektion ---
        branche_frame = ttk.LabelFrame(main_frame, text="Vælg Brancher", padding="10")
        branche_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(5, 0))

        # Listbox for brancher (EXTENDED)
        self.branche_listbox = tk.Listbox(branche_frame, selectmode=tk.EXTENDED, exportselection=False)
        self.branche_listbox.pack(fill=tk.BOTH, expand=True)

        # Kontrolsektion under listbox: Søg + Interval + Knapper
        branche_controls = ttk.Frame(branche_frame)
        branche_controls.pack(fill=tk.X, pady=(8, 0))

        # Søg (brancher)
        search_branche_row = ttk.Frame(branche_controls)
        search_branche_row.pack(fill=tk.X, pady=(0, 6))
        ttk.Label(search_branche_row, text="Søg:").pack(side=tk.LEFT)
        self.branche_search_var = tk.StringVar()
        branche_search_entry = ttk.Entry(search_branche_row, textvariable=self.branche_search_var)
        branche_search_entry.pack(side=tk.LEFT, fill=tk.X, expand=True)
        branche_search_entry.bind("<KeyRelease>", self.on_branche_search)
        self.branche_search_var.trace_add("write", lambda *_: self.on_branche_search())
        branche_search_entry.bind("<<Paste>>", self.on_branche_search)
        branche_search_entry.bind("<<Cut>>", self.on_branche_search)

        # Interval (branchekoder)
        branche_range_row = ttk.Frame(branche_controls)
        branche_range_row.pack(fill=tk.X, pady=(0, 6))
        ttk.Label(branche_range_row, text="Fra:").pack(side=tk.LEFT)
        self.branche_from_var = tk.StringVar()
        ttk.Entry(branche_range_row, width=10, textvariable=self.branche_from_var).pack(side=tk.LEFT, padx=(4, 8))
        ttk.Label(branche_range_row, text="Til:").pack(side=tk.LEFT)
        self.branche_to_var = tk.StringVar()
        ttk.Entry(branche_range_row, width=10, textvariable=self.branche_to_var).pack(side=tk.LEFT, padx=(4, 8))
        ttk.Button(branche_range_row, text="Vælg interval", command=self.select_branche_interval).pack(side=tk.LEFT)

        # Knaprække (brancher)
        branche_btns = ttk.Frame(branche_controls)
        branche_btns.pack(fill=tk.X, pady=(0, 0))
        ttk.Button(branche_btns, text="Markér alle", command=self.select_all_branche).pack(side=tk.LEFT)
        ttk.Button(branche_btns, text="Ryd", command=self.clear_branche).pack(side=tk.LEFT, padx=(6, 0))

        # Fyld listen fra start
        self.on_branche_search()

        # --- Knapper nederst ---
        button_frame = ttk.Frame(self, padding="10")
        button_frame.pack(fill=tk.X)
        ok_button = ttk.Button(button_frame, text="OK", command=self.on_ok)
        ok_button.pack(side=tk.RIGHT, padx=5)
        cancel_button = ttk.Button(button_frame, text="Annuller", command=self.on_cancel)
        cancel_button.pack(side=tk.RIGHT)

    def _bind_shortcuts(self):
        # Ctrl+A = markér alle i den listbox, der har fokus
        self.bind_all("<Control-a>", self._shortcut_select_all)
        self.bind_all("<Control-A>", self._shortcut_select_all)  # for sikkerheds skyld
        # Ctrl+D = ryd markering i den listbox, der har fokus
        self.bind_all("<Control-d>", self._shortcut_clear)
        self.bind_all("<Control-D>", self._shortcut_clear)

    # ---------- Søgefunktioner ----------
    def on_post_search(self, event=None):
        search_term = (self.post_search_var.get() or "").lower().strip()
        self.post_listbox.delete(0, tk.END)
        for item in self.postnumre:
            if search_term in item.lower():
                self.post_listbox.insert(tk.END, item)

    def on_branche_search(self, event=None):
        search_term = (self.branche_search_var.get() or "").lower().strip()
        self.branche_listbox.delete(0, tk.END)
        for item in self.brancher:
            if search_term in item.lower():
                self.branche_listbox.insert(tk.END, item)

    # ---------- Markér/Ryd (postnumre) ----------
    def select_all_post(self):
        self.post_listbox.select_set(0, tk.END)

    def clear_post(self):
        self.post_listbox.selection_clear(0, tk.END)

    def select_post_interval(self):
        """
        Marker alle viste postnumre i intervallet [fra, til] (inkl.).
        Arbejder på den aktuelt viste/filtrerede liste i listboxen.
        """
        try:
            from_code = int((self.post_from_var.get() or "").strip())
            to_code = int((self.post_to_var.get() or "").strip())
        except Exception:
            messagebox.showwarning("Ugyldigt interval", "Angiv hele tal for 'Fra' og 'Til' (fx 1000 og 3000).")
            return

        if from_code > to_code:
            from_code, to_code = to_code, from_code

        self.clear_post()
        count = 0
        for i in range(self.post_listbox.size()):
            text = self.post_listbox.get(i)
            code = _parse_code(text)
            if code is not None and from_code <= code <= to_code:
                self.post_listbox.select_set(i)
                count += 1

        if count == 0:
            messagebox.showinfo("Ingen fundet", f"Ingen viste postnumre i intervallet {from_code}-{to_code}.")

    # ---------- Markér/Ryd (brancher) ----------
    def select_all_branche(self):
        self.branche_listbox.select_set(0, tk.END)

    def clear_branche(self):
        self.branche_listbox.selection_clear(0, tk.END)

    def select_branche_interval(self):
        """
        Marker alle viste branchekoder i intervallet [fra, til] (inkl.).
        Arbejder på den aktuelt viste/filtrerede liste i listboxen.
        """
        try:
            from_code = int((self.branche_from_var.get() or "").strip())
            to_code = int((self.branche_to_var.get() or "").strip())
        except Exception:
            messagebox.showwarning("Ugyldigt interval", "Angiv hele tal for 'Fra' og 'Til' (fx 423300 og 423399).")
            return

        if from_code > to_code:
            from_code, to_code = to_code, from_code

        self.clear_branche()
        count = 0
        for i in range(self.branche_listbox.size()):
            text = self.branche_listbox.get(i)
            code = _parse_code(text)
            if code is not None and from_code <= code <= to_code:
                self.branche_listbox.select_set(i)
                count += 1

        if count == 0:
            messagebox.showinfo("Ingen fundet", f"Ingen viste branchekoder i intervallet {from_code}-{to_code}.")

    # ---------- Tastaturgenveje ----------
    def _shortcut_select_all(self, event=None):
        widget = self.focus_get()
        if widget is self.post_listbox:
            self.select_all_post()
            return "break"
        if widget is self.branche_listbox:
            self.select_all_branche()
            return "break"

    def _shortcut_clear(self, event=None):
        widget = self.focus_get()
        if widget is self.post_listbox:
            self.clear_post()
            return "break"
        if widget is self.branche_listbox:
            self.clear_branche()
            return "break"

    # ---------- OK/Cancel ----------
    def on_ok(self):
        selected_post_indices = self.post_listbox.curselection()
        selected_branche_indices = self.branche_listbox.curselection()

        if not selected_post_indices or not selected_branche_indices:
            messagebox.showwarning("Mangler valg", "Du skal vælge mindst ét postnummer og én branche.")
            return

        # Få de valgte items direkte fra listboxen for at matche det filtrerede view
        selected_post_items = [self.post_listbox.get(i) for i in selected_post_indices]
        selected_branche_items = [self.branche_listbox.get(i) for i in selected_branche_indices]

        selected_postnumre = [item.split(" ")[0] for item in selected_post_items]
        selected_brancher = [item.split(" - ")[0] for item in selected_branche_items]

        self.result = (selected_postnumre, selected_brancher)
        self.destroy()

    def on_cancel(self):
        self.result = None
        self.destroy()


def run_selector():
    """Kører GUI'en og returnerer brugerens valg."""
    postnumre_data, branchekoder_data = _load_lists()
    if postnumre_data is None:
        return None

    app = SelectorApp(postnumre_data, branchekoder_data)
    app.mainloop()
    return app.result


if __name__ == '__main__':
    valg = run_selector()
    if valg:
        print("Valgte postnumre:", valg[0])
        print("Valgte brancher:", valg[1])
    else:
        print("Brugeren annullerede.")
