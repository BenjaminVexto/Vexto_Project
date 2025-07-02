"""
test_async_gui.py
-----------------
Et simpelt script til at teste, om en asynkron opgave kan køre
i baggrunden, uden at en Tkinter GUI fryser.

Dette er en "smoke test" for at validere arkitekturen, før vi bygger
den fulde scoring-motor.
"""
import tkinter as tk
from tkinter import ttk
import asyncio
import threading
import time

class AsyncGuiApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Async GUI Test")
        self.geometry("400x150")

        # --- Opsætning af GUI-elementer ---
        main_frame = ttk.Frame(self, padding="20")
        main_frame.pack(fill=tk.BOTH, expand=True)

        self.status_label = ttk.Label(main_frame, text="Klar. Tryk på knappen for at starte.", font=("Helvetica", 10))
        self.status_label.pack(pady=(0, 10))

        self.start_button = ttk.Button(main_frame, text="Start 3-sekunders Asynkron Opgave", command=self.start_async_task)
        self.start_button.pack(pady=10, ipady=5)

        self.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.running = True

    def start_async_task(self):
        """
        Denne metode starter den asynkrone opgave i en separat tråd.
        Det er afgørende for ikke at blokere GUI'ens mainloop.
        """
        self.start_button.config(state=tk.DISABLED)
        self.status_label.config(text="Kører opgave... (GUI er IKKE frosset)")
        print("GUI: Starter baggrundstråd...")

        # Kør den asynkrone funktion i en separat tråd
        thread = threading.Thread(target=self.run_async_in_thread)
        thread.start()

    def run_async_in_thread(self):
        """
        Denne funktion kører i baggrundstråden.
        Den starter asyncio's event loop og kører vores dummy-task.
        """
        asyncio.run(self.dummy_task())

    async def dummy_task(self):
        """
        Dette er vores asynkrone funktion. Den simulerer en langsom
        netværksopgave (f.eks. at hente data fra en hjemmeside).
        """
        print("ASYNC: Opgave startet. Venter i 3 sekunder...")
        start_time = time.time()
        
        await asyncio.sleep(3) # Simulerer I/O-bundet arbejde
        
        end_time = time.time()
        print(f"ASYNC: Opgave færdig efter {end_time - start_time:.2f} sekunder.")

        # Opdater GUI'en sikkert fra baggrundstråden ved at bruge 'after'
        self.after(0, self.task_done)

    def task_done(self):
        """
        Denne metode kaldes på GUI-tråden, når den asynkrone opgave er færdig.
        """
        if self.running:
            self.status_label.config(text="Opgave færdig! Klar igen.")
            self.start_button.config(state=tk.NORMAL)
            print("GUI: Status opdateret.")

    def on_closing(self):
        """Håndterer lukning af vinduet."""
        self.running = False
        self.destroy()


if __name__ == "__main__":
    print("Starter test-applikation...")
    print("MÅL: Knappen skal starte en 3-sekunders opgave, uden at vinduet fryser.")
    app = AsyncGuiApp()
    app.mainloop()
    print("Applikation lukket.")
