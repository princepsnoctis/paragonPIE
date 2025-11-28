import tkinter as tk
from tkinter import messagebox
import csv
import os

FILENAME = 'old_data_backup.csv'

class ReceiptEditorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Edytor Paragonów CSV")
        self.root.geometry("1200x1100")

        self.data = []
        self.current_index = 0
        self.fieldnames = []

        self.load_csv()  # wczytanie pliku CSV

        # --- Elementy GUI ---
        self.lbl_counter = tk.Label(root, text="", font=("Arial", 20, "bold"))
        self.lbl_counter.pack(pady=10)

        lbl_raw_info = tk.Label(root, text="Oryginalna linijka z paragonu (tylko do odczytu):", fg="gray", font=("Arial", 16))
        lbl_raw_info.pack(anchor="w", padx=40)

        self.txt_raw_line = tk.Text(root, height=6, width=140, bg="#f0f0f0", font=("Consolas", 18))
        self.txt_raw_line.pack(padx=40, pady=(0, 20))
        self.txt_raw_line.config(state=tk.DISABLED)  # tylko do odczytu

        # Formularz pól edycyjnych
        self.form_frame = tk.Frame(root)
        self.form_frame.pack(padx=40, pady=10, fill="both", expand=True)

        self.entries = {}
        self.editable_fields = ["nazwa", "ilość", "cena_jedn", "kategoria_podatku", "jednostka", "całkowita_cena"]

        for i, field in enumerate(self.editable_fields):
            lbl = tk.Label(self.form_frame, text=field.capitalize().replace('_', ' ') + ":", anchor="e", width=40, font=("Arial", 18))
            lbl.grid(row=i, column=0, padx=10, pady=10, sticky="e")

            ent = tk.Entry(self.form_frame, width=80, font=("Arial", 18))
            ent.grid(row=i, column=1, padx=10, pady=10, sticky="w")
            self.entries[field] = ent

            # Nawigacja strzałkami między polami
            ent.bind("<Up>", lambda e, idx=i: self.focus_field(idx - 1))
            ent.bind("<Down>", lambda e, idx=i: self.focus_field(idx + 1))

        # Panel przycisków
        btn_frame = tk.Frame(root)
        btn_frame.pack(pady=40, fill="x")

        self.btn_prev = tk.Button(btn_frame, text="< Cofnij", command=self.prev_record, width=30, height=2, font=("Arial", 18))
        self.btn_prev.pack(side="left", padx=80)

        self.btn_save = tk.Button(btn_frame, text="Zapisz (Ctrl+S)", command=self.save_current_to_memory_and_file, width=30, height=2, bg="#dddddd", font=("Arial", 18))
        self.btn_save.pack(side="left", padx=20)

        self.btn_next = tk.Button(btn_frame, text="Dalej >", command=self.next_record, width=30, height=2, bg="#4CAF50", fg="white", font=("Arial", 18))
        self.btn_next.pack(side="right", padx=80)

        # Skróty klawiszowe
        root.bind('<Control-s>', lambda e: self.save_current_to_memory_and_file())
        self.txt_raw_line.bind("<Return>", self.insert_selected_to_name_and_next)

        if self.data:
            self.load_record_to_ui()
        else:
            messagebox.showerror("Błąd", f"Brak danych w pliku CSV: {FILENAME}")

    def focus_field(self, idx):
        if 0 <= idx < len(self.editable_fields):
            self.entries[self.editable_fields[idx]].focus_set()

    def load_csv(self):
        if not os.path.exists(FILENAME):
            messagebox.showerror("Błąd", f"Plik {FILENAME} nie istnieje!")
            self.data = []
            return

        try:
            with open(FILENAME, 'r', newline='', encoding='utf-8') as f:
                reader = csv.DictReader(f, delimiter=';')
                self.fieldnames = reader.fieldnames
                self.data = list(reader)
        except Exception as e:
            messagebox.showerror("Błąd", f"Nie udało się otworzyć pliku: {e}")
            self.data = []

    def save_csv(self):
        try:
            with open(FILENAME, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=self.fieldnames, delimiter=';')
                writer.writeheader()
                writer.writerows(self.data)
        except Exception as e:
            messagebox.showerror("Błąd zapisu", f"Nie udało się zapisać zmian: {e}")

    def load_record_to_ui(self):
        if not self.data or self.current_index >= len(self.data):
            return

        record = self.data[self.current_index]

        self.lbl_counter.config(text=f"Rekord {self.current_index + 1} z {len(self.data)}")

        self.txt_raw_line.config(state=tk.NORMAL)
        self.txt_raw_line.delete("1.0", tk.END)
        self.txt_raw_line.insert(tk.END, record.get("linijka", ""))
        self.txt_raw_line.config(state=tk.DISABLED)

        for field in self.editable_fields:
            self.entries[field].delete(0, tk.END)
            self.entries[field].insert(0, record.get(field, ""))

        if self.current_index == 0:
            self.btn_prev.config(state=tk.DISABLED)
        else:
            self.btn_prev.config(state=tk.NORMAL)

        if self.current_index == len(self.data) - 1:
            self.btn_next.config(text="Zakończ")
        else:
            self.btn_next.config(text="Dalej >")

    def save_current_to_memory_and_file(self):
        if not self.data or self.current_index >= len(self.data):
            return
        current_record = self.data[self.current_index]
        for field in self.editable_fields:
            current_record[field] = self.entries[field].get()
        self.save_csv()

    def insert_selected_to_name_and_next(self, event=None):
        """Przenosi zaznaczony tekst do pola 'nazwa' i przechodzi do następnego rekordu"""
        try:
            selected_text = self.txt_raw_line.get(tk.SEL_FIRST, tk.SEL_LAST)
        except tk.TclError:
            return "break"  # brak zaznaczenia

        self.entries['nazwa'].delete(0, tk.END)
        self.entries['nazwa'].insert(0, selected_text)

        self.save_current_to_memory_and_file()

        # przechodzimy do następnego rekordu, jeśli jest
        if self.current_index < len(self.data) - 1:
            self.current_index += 1
            self.load_record_to_ui()
        else:
            messagebox.showinfo("Koniec", "To był ostatni rekord. Zapisano zmiany.")

        return "break"

    def next_record(self):
        """Przycisk Dalej > – przejście do następnego rekordu ręcznie"""
        if self.current_index < len(self.data) - 1:
            self.current_index += 1
            self.load_record_to_ui()
        else:
            messagebox.showinfo("Koniec", "To był ostatni rekord. Zapisano zmiany.")

    def prev_record(self):
        self.save_current_to_memory_and_file()
        if self.current_index > 0:
            self.current_index -= 1
            self.load_record_to_ui()


if __name__ == "__main__":
    root = tk.Tk()
    app = ReceiptEditorApp(root)
    root.mainloop()
