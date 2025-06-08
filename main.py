#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🧠 Analiza Wzorców Konsumpcji Narkotyków - UCI Dataset
Autor: Karol Dąbrowski
Główny plik startowy aplikacji
"""

import sys
import os
import tkinter as tk
from tkinter import messagebox
import warnings

# Dodaj ścieżkę do modułów
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Wycisz ostrzeżenia
warnings.filterwarnings('ignore')

try:
    from gui.main_window import DrugConsumptionAnalyzer

    def main():
        """Główna funkcja uruchamiająca aplikację"""
        try:
            # Utwórz główne okno
            root = tk.Tk()

            # Utwórz aplikację
            app = DrugConsumptionAnalyzer(root)

            # Uruchom pętlę główną
            root.mainloop()

        except Exception as e:
            messagebox.showerror("Błąd krytyczny",
                                 f"Nie można uruchomić aplikacji:\n{str(e)}")
            sys.exit(1)

    if __name__ == "__main__":
        main()

except ImportError as e:
    print(f"❌ Błąd importu: {e}")
    print("📁 Sprawdź strukturę plików i zainstaluj wymagane biblioteki:")
    print("""
    Wymagane biblioteki:
    - tkinter (wbudowana)
    - pandas
    - numpy
    - matplotlib
    - seaborn
    - scikit-learn
    - scipy
    
    Instalacja: pip install pandas numpy matplotlib seaborn scikit-learn scipy
    """)
    sys.exit(1)