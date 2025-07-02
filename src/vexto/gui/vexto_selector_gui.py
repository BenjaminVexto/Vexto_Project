"""
vexto_selector_gui.py
---------------------
En simpel Tkinter GUI til at vælge postnumre og branchekoder,
inklusive søgefunktionalitet.
"""
import tkinter as tk
from tkinter import ttk, messagebox
from pathlib import Path
import pandas as pd

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
        
        # Bruger nu de korrekte kolonnenavne fra dine filer
        postnumre = sorted([f"{row.postnummer} {row.bynavn}" for index, row in df_post.iterrows()])
        brancher = sorted([f"{row.branchekode} - {row.nace_titel_dk}" for index, row in df_brancher.iterrows()])
        
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
        self.geometry("800x600")

        self.postnumre = postnumre  # Den fulde liste
        self.brancher = brancher    # Den fulde liste
        self.result = None

        self.create_widgets()

    def create_widgets(self):
        main_frame = ttk.Frame(self, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)

        # --- Postnummer sektion ---
        post_frame = ttk.LabelFrame(main_frame, text="Vælg Postnumre", padding="10")
        post_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))

        # Søgefelt for postnumre (GENINDSAT)
        search_post_frame = ttk.Frame(post_frame)
        search_post_frame.pack(fill=tk.X, pady=(0, 5))
        ttk.Label(search_post_frame, text="Søg:").pack(side=tk.LEFT)
        self.post_search_var = tk.StringVar()
        post_search_entry = ttk.Entry(search_post_frame, textvariable=self.post_search_var)
        post_search_entry.pack(side=tk.LEFT, fill=tk.X, expand=True)
        post_search_entry.bind("<KeyRelease>", self.on_post_search)

        # Listbox for postnumre
        self.post_listbox = tk.Listbox(post_frame, selectmode=tk.MULTIPLE, exportselection=False)
        self.on_post_search() # Fyld listen fra start
        self.post_listbox.pack(fill=tk.BOTH, expand=True)

        # --- Branchekode sektion ---
        branche_frame = ttk.LabelFrame(main_frame, text="Vælg Brancher", padding="10")
        branche_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(5, 0))

        # Søgefelt for brancher (GENINDSAT)
        search_branche_frame = ttk.Frame(branche_frame)
        search_branche_frame.pack(fill=tk.X, pady=(0, 5))
        ttk.Label(search_branche_frame, text="Søg:").pack(side=tk.LEFT)
        self.branche_search_var = tk.StringVar()
        branche_search_entry = ttk.Entry(search_branche_frame, textvariable=self.branche_search_var)
        branche_search_entry.pack(side=tk.LEFT, fill=tk.X, expand=True)
        branche_search_entry.bind("<KeyRelease>", self.on_branche_search)

        # Listbox for brancher
        self.branche_listbox = tk.Listbox(branche_frame, selectmode=tk.MULTIPLE, exportselection=False)
        self.on_branche_search() # Fyld listen fra start
        self.branche_listbox.pack(fill=tk.BOTH, expand=True)

        # --- Knapper ---
        button_frame = ttk.Frame(self, padding="10")
        button_frame.pack(fill=tk.X)
        ok_button = ttk.Button(button_frame, text="OK", command=self.on_ok)
        ok_button.pack(side=tk.RIGHT, padx=5)
        cancel_button = ttk.Button(button_frame, text="Annuller", command=self.on_cancel)
        cancel_button.pack(side=tk.RIGHT)

    # --- Søgefunktioner (GENINDSAT) ---
    def on_post_search(self, event=None):
        search_term = self.post_search_var.get().lower()
        self.post_listbox.delete(0, tk.END)
        for item in self.postnumre:
            if search_term in item.lower():
                self.post_listbox.insert(tk.END, item)

    def on_branche_search(self, event=None):
        search_term = self.branche_search_var.get().lower()
        self.branche_listbox.delete(0, tk.END)
        for item in self.brancher:
            if search_term in item.lower():
                self.branche_listbox.insert(tk.END, item)

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
