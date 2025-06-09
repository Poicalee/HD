def create_test_data(self):
    """Tworzy syntetyczne dane testowe"""
    try:
        print("Tworzenie danych testowych...")

        # Użyj data_processor do utworzenia danych
        self.df, self.processed_df = self.data_processor.create_synthetic_data()

        # POPRAWKA: Upewnij się że kolumny kategoryczne istnieją
        if self.processed_df is not None:
            # Sprawdź i utwórz brakujące kolumny kategoryczne
            demographic_cols_to_categorize = []

            for col in DEMOGRAPHIC_COLS:
                if col in self.processed_df.columns:
                    cat_col = f"{col}_Category"
                    if cat_col not in self.processed_df.columns:
                        demographic_cols_to_categorize.append(col)

            # Utwórz kategorie demograficzne jeśli ich brak
            if demographic_cols_to_categorize:
                print(f"Tworzenie kategorii dla: {demographic_cols_to_categorize}")

                for col in demographic_cols_to_categorize:
                    if col == 'Age':
                        # Kategorie wieku
                        self.processed_df[f'{col}_Category'] = pd.cut(
                            self.processed_df[col],
                            bins=[0, 25, 35, 45, 55, 100],
                            labels=['18-25', '26-35', '36-45', '46-55', '55+']
                        )
                    elif col == 'Gender':
                        # Kategorie płci
                        self.processed_df[f'{col}_Category'] = self.processed_df[col].map({
                            0.48: 'Female',
                            -0.48: 'Male'
                        }).fillna('Other')
                    elif col == 'Education':
                        # Kategorie wykształcenia
                        self.processed_df[f'{col}_Category'] = pd.cut(
                            self.processed_df[col],
                            bins=[-3, -1, 0, 1, 3],
                            labels=['Primary', 'Secondary', 'University', 'Masters+']
                        )
                    elif col == 'Country':
                        # Kategorie krajów
                        self.processed_df[f'{col}_Category'] = pd.cut(
                            self.processed_df[col],
                            bins=[-3, -1, 0, 1, 3],
                            labels=['USA', 'UK', 'Canada', 'Australia', 'Other']
                        )
                    elif col == 'Ethnicity':
                        # Kategorie etniczne
                        self.processed_df[f'{col}_Category'] = pd.cut(
                            self.processed_df[col],
                            bins=[-3, -1, 0, 1, 3],
                            labels=['White', 'Black', 'Asian', 'Mixed', 'Other']
                        )

            print(f"✅ Dane testowe utworzone: {len(self.processed_df)} wierszy, {len(self.processed_df.columns)} kolumn")

            # Odśwież interfejs
            self.refresh_table()
            self.update_substance_combos()
            self.data_info_label.config(text="✅ Dane testowe wczytane", foreground="green")

            messagebox.showinfo("Sukces", f"✅ Utworzono dane testowe:\n• {len(self.processed_df)} osób\n• {len(self.processed_df.columns)} zmiennych\n• Wszystkie kategorie demograficzne")
        else:
            messagebox.showerror("Błąd", "Nie można utworzyć danych testowych!")

    except Exception as e:
        messagebox.showerror("Błąd", f"Nie można utworzyć danych testowych: {str(e)}")
        print(f"Błąd w create_test_data: {str(e)}")# -*- coding: utf-8 -*-
"""
Główne okno aplikacji UCI Drug Consumption Analyzer
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# Dodaj ścieżki do modułów
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.data_processor import DataProcessor
from core.analyzer import StatisticalAnalyzer
from models.clustering import ClusterAnalyzer
from models.classification import ClassificationManager
from visualization.plots import PlotManager
from utils.constants import *
from utils.helpers import validate_dataframe

class DrugConsumptionAnalyzer:
    """Główna klasa aplikacji"""

    def __init__(self, root):
        self.root = root
        self.setup_window()

        # Komponenty
        self.data_processor = DataProcessor()
        self.analyzer = StatisticalAnalyzer()
        self.cluster_analyzer = ClusterAnalyzer()
        self.classification_manager = ClassificationManager()
        self.plot_manager = PlotManager()

        # Dane
        self.df = None
        self.processed_df = None

        # Zmienne UI
        self.filter_var = tk.StringVar()
        self.substance_var = tk.StringVar()
        self.classification_substance_var = tk.StringVar()

        # Komponenty UI
        self.data_info_label = None
        self.result_frame = None
        self.result_text = None
        self.plot_frame = None
        self.notebook = None
        self.data_table = None
        self.table_scrollbar_v = None
        self.table_scrollbar_h = None
        self.table_info_label = None

        self.setup_ui()

    def setup_window(self):
        """Konfiguruje główne okno"""
        self.root.title(GUI_CONFIG['title'])
        self.root.geometry(GUI_CONFIG['geometry'])
        self.root.configure(bg=GUI_CONFIG['bg_color'])

        # Stylowanie
        style = ttk.Style()
        style.theme_use('clam')

        # Niestandardowe style
        style.configure('Title.TLabel', font=('Segoe UI', 16, 'bold'),
                        foreground=COLORS['primary'])
        style.configure('Header.TLabel', font=('Segoe UI', 12, 'bold'),
                        foreground=COLORS['dark'])
        style.configure('Success.TButton', foreground='white',
                        background=COLORS['success'])
        style.configure('Danger.TButton', foreground='white',
                        background=COLORS['danger'])
        style.configure('TCombobox', padding=5)
        style.configure('TLabelFrame', font=('Segoe UI', 10, 'bold'))

    def setup_ui(self):
        """Konfiguruje interfejs użytkownika"""
        # Główna ramka
        main_frame = ttk.Frame(self.root, padding="15")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # Grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(1, weight=1)

        # Lewy panel kontrolny (przewijalny)
        self.setup_control_panel(main_frame)

        # Prawy panel wyników
        self.setup_result_panel(main_frame)

    def setup_control_panel(self, parent):
        """Konfiguruje panel kontrolny"""
        # Kontener z scrollbarem
        control_container = ttk.LabelFrame(parent, text="🎛️ Panel Kontrolny", padding="0")
        control_container.grid(row=1, column=0, sticky=(tk.N, tk.S, tk.W), padx=(0, 15))

        # Canvas i scrollbar
        canvas = tk.Canvas(control_container, borderwidth=0, highlightthickness=0)
        scrollbar = ttk.Scrollbar(control_container, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)

        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )

        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.grid(row=0, column=0, sticky=(tk.N, tk.S, tk.W, tk.E))
        scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))

        control_container.columnconfigure(0, weight=1)
        control_container.rowconfigure(0, weight=1)

        # Sekcje panelu
        self.create_control_sections(scrollable_frame)

    def create_control_sections(self, parent):
        """Tworzy sekcje panelu kontrolnego"""
        row = 0

        # 1. Wczytywanie danych
        row = self.create_data_section(parent, row)

        # 2. Statystyki opisowe
        row = self.create_stats_section(parent, row)

        # 3. Analiza korelacji
        row = self.create_correlation_section(parent, row)

        # 4. Filtrowanie danych
        row = self.create_filter_section(parent, row)

        # 5. Przetwarzanie danych
        row = self.create_processing_section(parent, row)

        # 6. Wizualizacje
        row = self.create_visualization_section(parent, row)

        # 7. Modelowanie zaawansowane
        row = self.create_modeling_section(parent, row)

        # 8. Analizy specyficzne UCI
        row = self.create_uci_section(parent, row)

    def create_data_section(self, parent, row):
        """Tworzy sekcję wczytywania danych"""
        data_frame = ttk.LabelFrame(parent, text="📁 1. Wczytywanie Danych", padding="10")
        data_frame.grid(row=row, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))

        ttk.Button(data_frame, text="📂 Wczytaj dane UCI (.data/.csv)",
                   command=self.load_data).grid(row=0, column=0, columnspan=2,
                                                sticky=(tk.W, tk.E), pady=(0, 5))

        ttk.Button(data_frame, text="🧪 Utwórz dane testowe",
                   command=self.create_test_data).grid(row=1, column=0, columnspan=2,
                                                       sticky=(tk.W, tk.E), pady=(0, 5))

        # Przyciski pomocy i resetowania
        help_frame = ttk.Frame(data_frame)
        help_frame.grid(row=2, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(5, 0))

        ttk.Button(help_frame, text="📚 Pomoc",
                   command=self.show_help).grid(row=0, column=0, padx=(0, 5), sticky=tk.W)
        ttk.Button(help_frame, text="ℹ️ O aplikacji",
                   command=self.show_about).grid(row=0, column=1, padx=(0, 5), sticky=tk.W)
        ttk.Button(help_frame, text="🔄 Reset",
                   command=self.reset_analysis).grid(row=0, column=2, padx=(0, 5), sticky=tk.W)
        ttk.Button(help_frame, text="🔍 Debug",
                   command=self.debug_demographic_data).grid(row=0, column=3, sticky=tk.W)

        self.data_info_label = ttk.Label(data_frame, text="❌ Brak danych", foreground="red")
        self.data_info_label.grid(row=3, column=0, columnspan=2, sticky=tk.W, pady=(5, 0))

        return row + 1

    def create_stats_section(self, parent, row):
        """Tworzy sekcję statystyk"""
        stats_frame = ttk.LabelFrame(parent, text="📈 2. Statystyki Opisowe", padding="10")
        stats_frame.grid(row=row, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))

        ttk.Button(stats_frame, text="📊 Oblicz statystyki podstawowe",
                   command=self.calculate_basic_stats).grid(row=0, column=0, columnspan=2,
                                                            sticky=(tk.W, tk.E), pady=(0, 5))

        return row + 1

    def create_correlation_section(self, parent, row):
        """Tworzy sekcję analizy korelacji"""
        corr_frame = ttk.LabelFrame(parent, text="🔗 3. Analiza Korelacji", padding="10")
        corr_frame.grid(row=row, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))

        ttk.Button(corr_frame, text="🧠 Korelacje cech osobowości",
                   command=self.analyze_personality_correlations).grid(row=0, column=0, columnspan=2,
                                                                       sticky=(tk.W, tk.E), pady=(0, 5))

        ttk.Button(corr_frame, text="💊 Korelacje substancji",
                   command=self.analyze_substance_correlations).grid(row=1, column=0, columnspan=2,
                                                                     sticky=(tk.W, tk.E), pady=(0, 5))

        return row + 1

    def create_filter_section(self, parent, row):
        """Tworzy sekcję filtrowania"""
        filter_frame = ttk.LabelFrame(parent, text="🎯 4. Filtrowanie Danych", padding="10")
        filter_frame.grid(row=row, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))

        filter_combo = ttk.Combobox(filter_frame, textvariable=self.filter_var, width=25)
        filter_combo['values'] = FILTER_OPTIONS
        filter_combo.set('Wszystkie dane')
        filter_combo.grid(row=0, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 5))

        ttk.Button(filter_frame, text="✅ Zastosuj filtr",
                   command=self.apply_filter).grid(row=1, column=0, columnspan=2,
                                                   sticky=(tk.W, tk.E), pady=(0, 5))

        return row + 1

    def create_processing_section(self, parent, row):
        """Tworzy sekcję przetwarzania danych"""
        process_frame = ttk.LabelFrame(parent, text="⚙️ 5. Przetwarzanie Danych", padding="10")
        process_frame.grid(row=row, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))

        ttk.Button(process_frame, text="📏 Standaryzacja cech",
                   command=self.standardize_features).grid(row=0, column=0, columnspan=2,
                                                           sticky=(tk.W, tk.E), pady=(0, 5))

        ttk.Button(process_frame, text="🔧 Obsługa brakujących wartości",
                   command=self.handle_missing_values).grid(row=1, column=0, columnspan=2,
                                                            sticky=(tk.W, tk.E), pady=(0, 5))

        ttk.Button(process_frame, text="🔀 Kodowanie binarne",
                   command=self.binary_encode).grid(row=2, column=0, columnspan=2,
                                                    sticky=(tk.W, tk.E), pady=(0, 5))

        return row + 1

    def create_visualization_section(self, parent, row):
        """Tworzy sekcję wizualizacji"""
        viz_frame = ttk.LabelFrame(parent, text="📈 6. Wizualizacje", padding="10")
        viz_frame.grid(row=row, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))

        ttk.Button(viz_frame, text="📊 Rozkłady cech osobowości",
                   command=self.plot_personality_distributions).grid(row=0, column=0, columnspan=2,
                                                                     sticky=(tk.W, tk.E), pady=(0, 5))

        ttk.Button(viz_frame, text="📈 Częstość używania substancji",
                   command=self.plot_substance_usage).grid(row=1, column=0, columnspan=2,
                                                           sticky=(tk.W, tk.E), pady=(0, 5))

        # Wybór substancji
        ttk.Label(viz_frame, text="Wybierz substancję:").grid(row=2, column=0, sticky=tk.W)

        substance_combo = ttk.Combobox(viz_frame, textvariable=self.substance_var, width=25)
        substance_combo['values'] = SUBSTANCE_COLS
        substance_combo.set('Cannabis')
        substance_combo.grid(row=3, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 5))

        ttk.Button(viz_frame, text="📦 Analiza demograficzna - Boxploty",
                   command=self.plot_demographic_boxplots).grid(row=4, column=0, columnspan=2,
                                                                sticky=(tk.W, tk.E), pady=(0, 5))

        ttk.Button(viz_frame, text="📊 Analiza demograficzna - Histogramy",
                   command=self.plot_demographic_histograms).grid(row=5, column=0, columnspan=2,
                                                                  sticky=(tk.W, tk.E), pady=(0, 5))

        ttk.Button(viz_frame, text="🌡️ Porównanie wszystkich substancji",
                   command=self.plot_all_substances_comparison).grid(row=6, column=0, columnspan=2,
                                                                     sticky=(tk.W, tk.E), pady=(0, 5))

        return row + 1

    def create_modeling_section(self, parent, row):
        """Tworzy sekcję modelowania"""
        model_frame = ttk.LabelFrame(parent, text="🤖 7. Modelowanie Zaawansowane", padding="10")
        model_frame.grid(row=row, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))

        ttk.Button(model_frame, text="🎯 Klastrowanie K-means + Profile",
                   command=self.perform_enhanced_clustering).grid(row=0, column=0, columnspan=2,
                                                                  sticky=(tk.W, tk.E), pady=(0, 5))

        # Wybór substancji do klasyfikacji
        ttk.Label(model_frame, text="Substancja do przewidywania:").grid(row=1, column=0, sticky=tk.W)

        classification_combo = ttk.Combobox(model_frame, textvariable=self.classification_substance_var, width=25)
        classification_combo['values'] = SUBSTANCE_COLS
        classification_combo.set('Cannabis')
        classification_combo.grid(row=2, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 5))

        ttk.Button(model_frame, text="🌲 Klasyfikacja Random Forest",
                   command=self.perform_classification).grid(row=3, column=0, columnspan=2,
                                                             sticky=(tk.W, tk.E), pady=(0, 5))

        ttk.Button(model_frame, text="🏆 Porównaj wszystkie substancje",
                   command=self.compare_all_classifications).grid(row=4, column=0, columnspan=2,
                                                                  sticky=(tk.W, tk.E), pady=(0, 5))

        return row + 1

    def create_uci_section(self, parent, row):
        """Tworzy sekcję analiz specyficznych UCI"""
        uci_frame = ttk.LabelFrame(parent, text="🎓 8. Analizy Specyficzne UCI", padding="10")
        uci_frame.grid(row=row, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))

        ttk.Button(uci_frame, text="🧬 Profile Klastrów vs Substancje",
                   command=self.analyze_cluster_substance_patterns).grid(row=0, column=0, columnspan=2,
                                                                         sticky=(tk.W, tk.E), pady=(0, 5))

        ttk.Button(uci_frame, text="👥 Demografia Klastrów",
                   command=self.analyze_cluster_demographics).grid(row=1, column=0, columnspan=2,
                                                                   sticky=(tk.W, tk.E), pady=(0, 5))

        ttk.Button(uci_frame, text="🚨 Risk Assessment Tool",
                   command=self.create_risk_assessment).grid(row=2, column=0, columnspan=2,
                                                             sticky=(tk.W, tk.E), pady=(0, 5))

        return row + 1

    def setup_result_panel(self, parent):
        """Konfiguruje panel wyników"""
        self.result_frame = ttk.LabelFrame(parent, text="📊 Wyniki Analizy", padding="15")
        self.result_frame.grid(row=1, column=1, sticky=(tk.N, tk.S, tk.E, tk.W))

        self.result_frame.columnconfigure(0, weight=1)
        self.result_frame.rowconfigure(0, weight=1)

        # Notebook z zakładkami
        self.notebook = ttk.Notebook(self.result_frame)
        self.notebook.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # Zakładka tabeli edytowalnej
        self.table_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.table_frame, text="📋 Tabela Danych")
        self.setup_data_table()

        # Zakładka tekstowa
        self.text_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.text_frame, text="📝 Wyniki Tekstowe")

        self.result_text = scrolledtext.ScrolledText(self.text_frame, wrap=tk.WORD,
                                                     width=90, height=35, font=('Consolas', 10))
        self.result_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        self.text_frame.columnconfigure(0, weight=1)
        self.text_frame.rowconfigure(0, weight=1)

        # Zakładka wykresów
        self.plot_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.plot_frame, text="📊 Wykresy Interaktywne")

        self.plot_frame.columnconfigure(0, weight=1)
        self.plot_frame.rowconfigure(0, weight=1)

    # === METODY OBSŁUGI ZDARZEŃ ===

    def load_data(self):
        """Wczytuje dane z pliku"""
        file_path = filedialog.askopenfilename(
            title="Wybierz plik z danymi UCI",
            filetypes=[("Data files", "*.data"), ("CSV files", "*.csv"), ("All files", "*.*")]
        )

        if file_path:
            success, message, df = self.data_processor.load_data(file_path)

            if success:
                self.df = df
                self.processed_df = df.copy()

                # Zaktualizuj UI
                self.data_info_label.config(text=message, foreground="green")

                # Pokaż informacje o danych
                self.display_data_info()

                # Zaktualizuj combobox substancji
                self.update_substance_combos()

                # Odśwież tabelę
                self.refresh_table()

                messagebox.showinfo("Sukces", "✅ Dane UCI zostały pomyślnie wczytane!")
            else:
                messagebox.showerror("Błąd", f"❌ {message}")

    def display_data_info(self):
        """Wyświetla informacje o wczytanych danych"""
        if self.df is None:
            return

        summary = self.data_processor.get_data_summary(self.df)

        info_text = f"""
🎓 === DATASET UCI DRUG CONSUMPTION (QUANTIFIED) ===

📊 ROZMIAR DANYCH: {summary['shape'][0]} respondentów × {summary['shape'][1]} zmiennych

🧑‍🤝‍🧑 CHARAKTERYSTYKA PRÓBY:
"""

        # Demografia
        for demo_type, demo_data in summary['demographics'].items():
            if demo_data:
                info_text += f"\n{demo_type}:\n"
                for category, count in list(demo_data.items())[:5]:
                    pct = count / summary['shape'][0] * 100
                    info_text += f"  {category}: {count} ({pct:.1f}%)\n"

        # Top substancje
        if summary['substances']:
            substance_list = [(name, data['usage_rate']) for name, data in summary['substances'].items()]
            substance_list.sort(key=lambda x: x[1], reverse=True)

            info_text += f"\n💊 TOP 10 UŻYWANYCH SUBSTANCJI:\n"
            for i, (substance, rate) in enumerate(substance_list[:10], 1):
                emoji = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else f"{i:>2}."
                users = int(rate * summary['shape'][0])
                info_text += f"{emoji} {substance:<15} {users:>4} osób ({rate*100:>5.1f}%)\n"

        info_text += f"""

🎯 KLUCZOWE CHARAKTERYSTYKI:
• Próba: Głównie młodzi dorośli (18-34), wykształceni, kraje anglojęzyczne
• Substancje: Od powszechnych (caffeine 95%+) do rzadkich (heroin <5%)  
• Osobowość: Znormalizowane score'y (μ≈0, σ≈1) z modelu Big Five + impulsywność
• Format: Self-report online survey, anonimowe odpowiedzi
• Jakość: Zawiera fikcyjną substancję "Semer" do kontroli wiarygodności

⚠️ OGRANICZENIA PRÓBY:
• Geographic bias: Głównie kraje anglojęzyczne (UK, US, Canada)
• Demographic bias: Młodzi, wykształceni, z dostępem do internetu  
• Temporal: Dane z ~2012, wzorce mogły się zmienić
• Self-report: Możliwe under-reporting illegal substance use
"""

        self.result_text.delete(1.0, tk.END)
        self.result_text.insert(tk.END, info_text)

    def update_substance_combos(self):
        """Aktualizuje combobox-y substancji"""
        if self.df is None:
            return

        existing_substances = [col for col in SUBSTANCE_COLS if col in self.df.columns]

        # Znajdź widget-y combobox
        for widget in self.root.winfo_children():
            self._update_combo_recursive(widget, existing_substances)

    def _update_combo_recursive(self, widget, substances):
        """Rekurencyjnie aktualizuje combobox-y"""
        if isinstance(widget, ttk.Combobox):
            current_values = widget['values']
            if current_values and current_values[0] in SUBSTANCE_COLS:
                widget['values'] = substances
                if widget.get() not in substances and substances:
                    widget.set(substances[0])

        for child in widget.winfo_children():
            self._update_combo_recursive(child, substances)

    def calculate_basic_stats(self):
        """Oblicza podstawowe statystyki"""
        if self.processed_df is None:
            messagebox.showerror("Błąd", "Najpierw wczytaj dane!")
            return

        stats_text = self.analyzer.calculate_basic_stats(self.processed_df)
        self.result_text.delete(1.0, tk.END)
        self.result_text.insert(tk.END, stats_text)

    def analyze_personality_correlations(self):
        """Analizuje korelacje cech osobowości"""
        if self.processed_df is None:
            messagebox.showerror("Błąd", "Najpierw wczytaj dane!")
            return

        correlation_matrix, analysis_text = self.analyzer.analyze_correlations(
            self.processed_df, PERSONALITY_COLS, "personality"
        )

        if correlation_matrix.empty:
            messagebox.showwarning("Uwaga", "Brak wystarczających danych do analizy korelacji")
            return

        self.result_text.delete(1.0, tk.END)
        self.result_text.insert(tk.END, analysis_text)

        # Pokaż heatmapę
        fig = self.plot_manager.create_correlation_heatmap(
            correlation_matrix, "🧠 Korelacje Cech Osobowości"
        )
        self.plot_manager.embed_plot_in_frame(fig, self.plot_frame)

    def analyze_substance_correlations(self):
        """Analizuje korelacje substancji"""
        if self.processed_df is None:
            messagebox.showerror("Błąd", "Najpierw wczytaj dane!")
            return

        correlation_matrix, analysis_text = self.analyzer.analyze_correlations(
            self.processed_df, SUBSTANCE_COLS, "substance"
        )

        if correlation_matrix.empty:
            messagebox.showwarning("Uwaga", "Brak wystarczających danych do analizy korelacji")
            return

        self.result_text.delete(1.0, tk.END)
        self.result_text.insert(tk.END, analysis_text)

        # Pokaż heatmapę
        fig = self.plot_manager.create_correlation_heatmap(
            correlation_matrix, "💊 Korelacje Używania Substancji"
        )
        self.plot_manager.embed_plot_in_frame(fig, self.plot_frame)

    def apply_filter(self):
        """Stosuje wybrany filtr"""
        if self.df is None:
            messagebox.showerror("Błąd", "Najpierw wczytaj dane!")
            return

        filter_option = self.filter_var.get()
        self.processed_df = self.data_processor.apply_filter(self.df, filter_option)

        if len(self.processed_df) == 0:
            messagebox.showwarning("Uwaga", f"Filtr '{filter_option}' nie zwrócił żadnych danych!")
            self.processed_df = self.df.copy()
            return

        # Pokaż informacje o filtrowaniu
        filter_text = f"""
🎯 === ZASTOSOWANO FILTR: {filter_option} ===

📊 STATYSTYKI FILTROWANIA:
├── Oryginalny rozmiar: {len(self.df)} rekordów
├── Po filtrowaniu: {len(self.processed_df)} rekordów  
├── Odsetek zachowany: {len(self.processed_df) / len(self.df) * 100:.1f}%
└── Utracono: {len(self.df) - len(self.processed_df)} rekordów

✅ Filtr zastosowany pomyślnie!
"""

        self.result_text.delete(1.0, tk.END)
        self.result_text.insert(tk.END, filter_text)

        # Odśwież tabelę z nowymi danymi
        self.refresh_table()

    def standardize_features(self):
        """Standaryzuje cechy"""
        if self.processed_df is None:
            messagebox.showerror("Błąd", "Najpierw wczytaj dane!")
            return

        self.processed_df = self.data_processor.standardize_features(self.processed_df)

        text = """
📏 === STANDARYZACJA CECH OSOBOWOŚCI ===

✅ Cechy zostały wystandaryzowane (Z-score transformation: μ=0, σ=1)

🎯 KORZYŚCI STANDARYZACJI:
• Wszystkie cechy na tej samej skali
• Lepsze działanie algorytmów ML  
• Łatwiejsza interpretacja (jednostki = odchylenia standardowe)
• Eliminacja bias związanych z różnymi zakresami wartości

✅ Utworzono nowe kolumny z sufiksem '_std'
"""

        self.result_text.delete(1.0, tk.END)
        self.result_text.insert(tk.END, text)

        # Odśwież tabelę
        self.refresh_table()

    def handle_missing_values(self):
        """Obsługuje brakujące wartości"""
        if self.processed_df is None:
            messagebox.showerror("Błąd", "Najpierw wczytaj dane!")
            return

        original_size = len(self.processed_df)
        self.processed_df, missing_filled = self.data_processor.handle_missing_values(self.processed_df)

        text = f"""
🔧 === OBSŁUGA BRAKUJĄCYCH WARTOŚCI I DUPLIKATÓW ===

🔧 ZASTOSOWANE METODY:
• KNN Imputation (k=5) dla cech osobowości
• Usunięcie duplikatów

✅ WYNIKI:
• Uzupełniono: {sum(missing_filled.values())} brakujących wartości
• Usunięto: {original_size - len(self.processed_df)} duplikatów
• Finalny rozmiar: {len(self.processed_df)} rekordów

💡 KNN IMPUTATION:
Metoda szuka 5 najbardziej podobnych osób i uśrednia ich wartości.
Zachowuje naturalne korelacje między cechami osobowości.
"""

        self.result_text.delete(1.0, tk.END)
        self.result_text.insert(tk.END, text)

        # Odśwież tabelę
        self.refresh_table()

    def binary_encode(self):
        """Tworzy binarne kodowanie"""
        if self.processed_df is None:
            messagebox.showerror("Błąd", "Najpierw wczytaj dane!")
            return

        self.processed_df = self.data_processor.create_binary_encoding(self.processed_df)

        text = """
🔀 === KODOWANIE BINARNE UŻYWANIA SUBSTANCJI ===

✅ Utworzono binarne wskaźniki (0=nigdy, 1=używał)

💡 ZASTOSOWANIA KODOWANIA BINARNEGO:
• Uproszczone analizy (używa/nie używa)
• Modele klasyfikacji binarnej
• Analiza częstości występowania
• Reguły asocjacyjne

✅ Utworzono nowe kolumny z sufiksem '_binary'
"""

        self.result_text.delete(1.0, tk.END)
        self.result_text.insert(tk.END, text)

        # Odśwież tabelę
        self.refresh_table()

    def plot_personality_distributions(self):
        """Tworzy wykresy rozkładów osobowości"""
        if self.processed_df is None:
            messagebox.showerror("Błąd", "Najpierw wczytaj dane!")
            return

        fig, analysis_text = self.plot_manager.create_personality_distributions(self.processed_df)

        if fig is None:
            messagebox.showwarning("Uwaga", analysis_text)
            return

        self.result_text.delete(1.0, tk.END)
        self.result_text.insert(tk.END, analysis_text)

        self.plot_manager.embed_plot_in_frame(fig, self.plot_frame)

    def plot_substance_usage(self):
        """Tworzy wykres używania substancji"""
        if self.processed_df is None:
            messagebox.showerror("Błąd", "Najpierw wczytaj dane!")
            return

        fig, analysis_text = self.plot_manager.create_substance_usage_plot(self.processed_df)

        if fig is None:
            messagebox.showwarning("Uwaga", analysis_text)
            return

        self.result_text.delete(1.0, tk.END)
        self.result_text.insert(tk.END, analysis_text)

        self.plot_manager.embed_plot_in_frame(fig, self.plot_frame)

    def plot_demographic_boxplots(self):
        """Tworzy boxploty demograficzne"""
        if self.processed_df is None:
            messagebox.showerror("Błąd", "Najpierw wczytaj dane!")
            return

        substance = self.substance_var.get()
        if not substance:
            messagebox.showerror("Błąd", "Wybierz substancję!")
            return

        fig, analysis_text = self.plot_manager.create_demographic_boxplots(self.processed_df, substance)

        if fig is None:
            messagebox.showwarning("Uwaga", analysis_text)
            return

        self.result_text.delete(1.0, tk.END)
        self.result_text.insert(tk.END, analysis_text)

        self.plot_manager.embed_plot_in_frame(fig, self.plot_frame)

    def debug_demographic_data(self):
        """Debug: Sprawdza dostępność danych demograficznych"""
        if self.processed_df is None:
            print("DEBUG: Brak danych")
            return

        print("=== DEBUG DANYCH DEMOGRAFICZNYCH ===")
        print(f"Kolumny w DataFrame: {len(self.processed_df.columns)}")
        print(f"Wiersze w DataFrame: {len(self.processed_df)}")

        print("\n📊 KOLUMNY DEMOGRAFICZNE:")
        for col in DEMOGRAPHIC_COLS:
            if col in self.processed_df.columns:
                print(f"✅ {col}: {self.processed_df[col].dtype}")
                cat_col = f"{col}_Category"
                if cat_col in self.processed_df.columns:
                    print(f"✅ {cat_col}: {self.processed_df[cat_col].nunique()} kategorii")
                    print(f"   Przykłady: {self.processed_df[cat_col].dropna().unique()[:3]}")
                else:
                    print(f"❌ {cat_col}: BRAK")
            else:
                print(f"❌ {col}: BRAK")

        print("\n🧪 KOLUMNY SUBSTANCJI:")
        substance_count = 0
        for col in SUBSTANCE_COLS[:5]:  # Pierwsze 5
            if col in self.processed_df.columns:
                substance_count += 1
                non_zero = (self.processed_df[col] > 0).sum()
                print(f"✅ {col}: {non_zero} użytkowników")
            else:
                print(f"❌ {col}: BRAK")

        print(f"\nPODSUMOWANIE:")
        print(f"• Dostępne kolumny demograficzne: {len([col for col in DEMOGRAPHIC_COLS if col in self.processed_df.columns])}")
        print(f"• Dostępne kategorie demograficzne: {len([f'{col}_Category' for col in DEMOGRAPHIC_COLS if f'{col}_Category' in self.processed_df.columns])}")
        print(f"• Dostępne substancje: {substance_count}")

        # Test analizy demograficznej
        available_substances = [col for col in SUBSTANCE_COLS if col in self.processed_df.columns]
        if available_substances:
            test_substance = available_substances[0]
            print(f"\n🧪 TEST ANALIZY: {test_substance}")

            try:
                if hasattr(self.analyzer, 'analyze_demographic_differences'):
                    print("✅ Metoda analyze_demographic_differences istnieje")
                    # Test call (bez wyświetlania wyniku)
                    result = self.analyzer.analyze_demographic_differences(self.processed_df, test_substance)
                    print(f"✅ Analiza wykonana, długość: {len(result)} znaków")
                else:
                    print("❌ Metoda analyze_demographic_differences BRAK")

                if hasattr(self.analyzer, 'create_demographic_histograms'):
                    print("✅ Metoda create_demographic_histograms istnieje")
                else:
                    print("❌ Metoda create_demographic_histograms BRAK")

            except Exception as e:
                print(f"❌ Błąd testu: {str(e)}")

        print("=== KONIEC DEBUG ===")

    def test_demographic_histograms(self):
        """Test metoda dla histogramów demograficznych"""
        if self.processed_df is None:
            messagebox.showwarning("Uwaga", "Najpierw wczytaj dane testowe!")
            return

        # Sprawdź czy Cannabis istnieje
        if 'Cannabis' not in self.processed_df.columns:
            available_substances = [col for col in SUBSTANCE_COLS if col in self.processed_df.columns]
            if available_substances:
                test_substance = available_substances[0]
                messagebox.showinfo("Info", f"Cannabis nie znaleziono, używam: {test_substance}")
                self.substance_var.set(test_substance)
            else:
                messagebox.showerror("Błąd", "Brak substancji w danych!")
                return
        else:
            self.substance_var.set('Cannabis')

        print("TEST: Uruchamiam histogramy demograficzne...")
        self.plot_demographic_histograms()

    def plot_demographic_histograms(self):
        """Tworzy histogramy demograficzne"""
        if self.processed_df is None:
            messagebox.showerror("Błąd", "Najpierw wczytaj dane!")
            return

        substance = self.substance_var.get()
        if not substance:
            messagebox.showerror("Błąd", "Wybierz substancję!")
            return

        analysis_text = self.analyzer.analyze_demographic_differences(self.processed_df, substance)

        self.result_text.delete(1.0, tk.END)
        self.result_text.insert(tk.END, analysis_text)

    def plot_all_substances_comparison(self):
        """Porównuje wszystkie substancje"""
        if self.processed_df is None:
            messagebox.showerror("Błąd", "Najpierw wczytaj dane!")
            return

        try:
            # Znajdź istniejące kolumny substancji i osobowości
            existing_substance_cols = [col for col in SUBSTANCE_COLS if col in self.processed_df.columns]
            existing_personality_cols = [col for col in PERSONALITY_COLS if col in self.processed_df.columns]

            if len(existing_substance_cols) < 3 or len(existing_personality_cols) < 3:
                messagebox.showwarning("Uwaga", "Za mało danych do porównania substancji")
                return

            # Oblicz średnie cechy osobowości dla użytkowników każdej substancji
            substance_profiles = {}
            comparison_text = "🌡️ === PORÓWNANIE WSZYSTKICH SUBSTANCJI ===\n\n"
            comparison_text += "Średnie wartości cech osobowości dla użytkowników każdej substancji:\n\n"

            for substance in existing_substance_cols:
                users = self.processed_df[self.processed_df[substance] > 0]
                if len(users) >= 10:  # Minimum 10 użytkowników
                    means = []
                    for trait in existing_personality_cols:
                        mean_val = users[trait].mean()
                        means.append(mean_val)
                    substance_profiles[substance] = means

                    comparison_text += f"💊 {substance} (n={len(users)}):\n"
                    for trait, mean_val in zip(existing_personality_cols, means):
                        level = "🔴" if mean_val > 0.3 else "🟠" if mean_val > 0.1 else "🟡" if mean_val > -0.1 else "🔵" if mean_val > -0.3 else "🟣"
                        comparison_text += f"  {trait}: {mean_val:+.3f} {level}\n"
                    comparison_text += "\n"

            if not substance_profiles:
                messagebox.showwarning("Uwaga", "Brak wystarczających danych do porównania")
                return

            # Utwórz DataFrame dla heatmapy
            df_heatmap = pd.DataFrame(substance_profiles, index=existing_personality_cols).T

            # Wyczyść poprzednie wykresy
            for widget in self.plot_frame.winfo_children():
                widget.destroy()

            # Twórz wykresy
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 12))

            # Główna heatmapa
            import seaborn as sns
            sns.heatmap(df_heatmap, annot=True, fmt='.2f', cmap='RdBu_r', center=0,
                        cbar_kws={'label': 'Średnia znormalizowana'}, ax=ax1,
                        linewidths=0.5, square=False)

            ax1.set_title('🌡️ Profile Cech Osobowości dla Użytkowników Różnych Substancji',
                          fontsize=14, fontweight='bold')
            ax1.set_xlabel('Cechy Osobowości', fontweight='bold')
            ax1.set_ylabel('Substancje', fontweight='bold')

            # Dendrogram klastrowania
            try:
                from scipy.cluster.hierarchy import dendrogram, linkage

                # Oblicz linkage dla substancji
                linkage_matrix = linkage(df_heatmap.values, method='ward')

                dendrogram(linkage_matrix, labels=df_heatmap.index, ax=ax2,
                           leaf_rotation=45, leaf_font_size=10)
                ax2.set_title('🌳 Dendrogram Klastrów Substancji (na podstawie profili osobowości)',
                              fontweight='bold')
                ax2.set_xlabel('Substancje')
                ax2.set_ylabel('Odległość')
            except Exception as e:
                # Jeśli dendrogram nie działa, zrób wykres słupkowy
                substance_means = df_heatmap.mean(axis=1).sort_values(ascending=False)
                bars = ax2.bar(range(len(substance_means)), substance_means.values)
                ax2.set_xticks(range(len(substance_means)))
                ax2.set_xticklabels(substance_means.index, rotation=45)
                ax2.set_title('📊 Średni Poziom Cech Osobowości według Substancji', fontweight='bold')
                ax2.set_ylabel('Średnia wartość cech')

            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()

            # Identyfikacja klastrów
            comparison_text += "🎯 IDENTYFIKACJA KLASTRÓW:\n"
            comparison_text += "─" * 30 + "\n"

            # Znajdź substancje z podobnymi profilami (uproszczone klastrowanie)
            try:
                from sklearn.cluster import KMeans
                if len(df_heatmap) >= 4:
                    kmeans = KMeans(n_clusters=min(4, len(df_heatmap)), random_state=42)
                    cluster_labels = kmeans.fit_predict(df_heatmap.values)

                    for cluster_id in range(kmeans.n_clusters):
                        cluster_substances = df_heatmap.index[cluster_labels == cluster_id].tolist()
                        if cluster_substances:
                            comparison_text += f"🔗 Klaster {cluster_id + 1}: {', '.join(cluster_substances)}\n"
            except Exception as e:
                comparison_text += "Nie można wykonać klastrowania automatycznego\n"

            # Pokaż wyniki
            self.result_text.delete(1.0, tk.END)
            self.result_text.insert(tk.END, comparison_text)

            # Osadź wykres
            canvas = FigureCanvasTkAgg(fig, self.plot_frame)
            canvas.draw()
            canvas.get_tk_widget().grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        except Exception as e:
            messagebox.showerror("Błąd", f"Nie można wykonać porównania substancji: {str(e)}")
            print(f"Błąd w plot_all_substances_comparison: {str(e)}")

    def analyze_cluster_demographics(self):
        """Analizuje demografię klastrów"""
        if self.cluster_analyzer.cluster_labels is None:
            messagebox.showwarning("Uwaga", "Najpierw wykonaj analizę klastrów!")
            return

    def analyze_cluster_demographics(self):
        """Analizuje demografię klastrów"""
        if self.cluster_analyzer.cluster_labels is None:
            messagebox.showwarning("Uwaga", "Najpierw wykonaj analizę klastrów!")
            return

        try:
            # POPRAWKA: Użyj tych samych indeksów co przy klastrowaniu
            existing_personality_cols = [col for col in PERSONALITY_COLS if col in self.processed_df.columns]
            features = self.processed_df[existing_personality_cols].dropna()

            if len(self.cluster_analyzer.cluster_labels) != len(features):
                messagebox.showerror("Błąd",
                                     f"Niezgodność długości cluster_labels ({len(self.cluster_analyzer.cluster_labels)}) vs features ({len(features)})")
                return

            # Użyj tylko wierszy które były w klastrowania
            cluster_demo_df = self.processed_df.loc[features.index].copy()
            cluster_demo_df['Cluster'] = self.cluster_analyzer.cluster_labels

            demo_text = "👥 === DEMOGRAFIA KLASTRÓW OSOBOWOŚCI ===\n\n"
            demo_text += f"Analiza demograficzna {len(cluster_demo_df)} osób w 4 klastrach osobowości\n\n"

            for cluster_id in range(4):
                cluster_data = cluster_demo_df[cluster_demo_df['Cluster'] == cluster_id]
                if len(cluster_data) == 0:
                    continue

                profile = self.cluster_analyzer.cluster_profiles.get(cluster_id, {
                    'name': f'Klaster {cluster_id}',
                    'emoji': '❓',
                    'risk': 'Nieznane'
                })

                demo_text += f"{profile['emoji']} KLASTER {cluster_id}: {profile['name']}\n"
                demo_text += f"📊 Wielkość: {len(cluster_data)} osób ({len(cluster_data) / len(cluster_demo_df) * 100:.1f}%)\n"
                demo_text += f"🎯 Poziom ryzyka: {profile['risk']}\n"
                demo_text += "─" * 60 + "\n"

                # Analiza wieku
                if 'Age_Category' in cluster_data.columns:
                    age_dist = cluster_data['Age_Category'].value_counts()
                    if len(age_dist) > 0:
                        demo_text += "👶 ROZKŁAD WIEKU:\n"
                        for age_cat, count in age_dist.items():
                            pct = count / len(cluster_data) * 100
                            demo_text += f"  {age_cat}: {count} osób ({pct:.1f}%)\n"
                        demo_text += f"  Dominująca grupa: {age_dist.index[0]}\n\n"

                # Analiza płci
                if 'Gender_Category' in cluster_data.columns:
                    gender_dist = cluster_data['Gender_Category'].value_counts()
                    if len(gender_dist) > 0:
                        demo_text += "⚧️ ROZKŁAD PŁCI:\n"
                        for gender, count in gender_dist.items():
                            pct = count / len(cluster_data) * 100
                            demo_text += f"  {gender}: {count} osób ({pct:.1f}%)\n"

                        # Określ czy jest bias płciowy
                        if len(gender_dist) >= 2:
                            max_pct = gender_dist.max() / len(cluster_data) * 100
                            if max_pct > 60:
                                dominant_gender = gender_dist.index[0]
                                demo_text += f"  🎯 Dominacja: {dominant_gender} ({max_pct:.1f}%)\n"
                        demo_text += "\n"

                # Analiza wykształcenia
                if 'Education_Category' in cluster_data.columns:
                    edu_dist = cluster_data['Education_Category'].value_counts()
                    if len(edu_dist) > 0:
                        demo_text += "🎓 WYKSZTAŁCENIE (top 3):\n"
                        for edu, count in edu_dist.head(3).items():
                            pct = count / len(cluster_data) * 100
                            demo_text += f"  {edu}: {count} osób ({pct:.1f}%)\n"
                        demo_text += "\n"

                # Analiza krajów
                if 'Country_Category' in cluster_data.columns:
                    country_dist = cluster_data['Country_Category'].value_counts()
                    if len(country_dist) > 0:
                        demo_text += "🌍 POCHODZENIE (top 3):\n"
                        for country, count in country_dist.head(3).items():
                            pct = count / len(cluster_data) * 100
                            demo_text += f"  {country}: {count} osób ({pct:.1f}%)\n"
                        demo_text += "\n"

                demo_text += "\n"

            # Porównania między klastrami
            demo_text += "🔍 PORÓWNANIA DEMOGRAFICZNE:\n"
            demo_text += "=" * 40 + "\n"

            # Analiza wieku między klastrami
            if 'Age_Category' in cluster_demo_df.columns:
                demo_text += "\n👶 RÓŻNICE WIEKOWE:\n"
                age_by_cluster = cluster_demo_df.groupby(['Cluster', 'Age_Category']).size().unstack(fill_value=0)
                age_percentages = age_by_cluster.div(age_by_cluster.sum(axis=1), axis=0) * 100

                for age_group in age_percentages.columns:
                    cluster_percentages = age_percentages[age_group].sort_values(ascending=False)
                    highest_cluster = cluster_percentages.index[0]
                    highest_pct = cluster_percentages.iloc[0]

                    profile = self.cluster_analyzer.cluster_profiles.get(highest_cluster, {'name': f'Klaster {highest_cluster}'})
                    if highest_pct > 30:  # Znacząca różnica
                        demo_text += f"• {age_group}: najczęściej {profile['name']} ({highest_pct:.1f}%)\n"

            # Analiza płci między klastrami
            if 'Gender_Category' in cluster_demo_df.columns:
                demo_text += "\n⚧️ RÓŻNICE PŁCIOWE:\n"
                gender_by_cluster = cluster_demo_df.groupby(['Cluster', 'Gender_Category']).size().unstack(fill_value=0)
                if 'Male' in gender_by_cluster.columns and 'Female' in gender_by_cluster.columns:
                    gender_percentages = gender_by_cluster.div(gender_by_cluster.sum(axis=1), axis=0) * 100

                    male_percentages = gender_percentages['Male'].sort_values(ascending=False)
                    female_percentages = gender_percentages['Female'].sort_values(ascending=False)

                    most_male_cluster = male_percentages.index[0]
                    most_female_cluster = female_percentages.index[0]

                    male_profile = self.cluster_analyzer.cluster_profiles.get(most_male_cluster, {'name': f'Klaster {most_male_cluster}'})
                    female_profile = self.cluster_analyzer.cluster_profiles.get(most_female_cluster, {'name': f'Klaster {most_female_cluster}'})

                    demo_text += f"• Więcej mężczyzn: {male_profile['name']} ({male_percentages.iloc[0]:.1f}%)\n"
                    demo_text += f"• Więcej kobiet: {female_profile['name']} ({female_percentages.iloc[0]:.1f}%)\n"

            # Wnioski i implikacje
            demo_text += f"\n💡 WNIOSKI I IMPLIKACJE:\n"
            demo_text += "=" * 25 + "\n"
            demo_text += "• Profile osobowości mają związek z charakterystykami demograficznymi\n"
            demo_text += "• Różne grupy wiekowe wykazują różne wzorce osobowości\n"
            demo_text += "• Interwencje powinny uwzględniać kontekst demograficzny\n"
            demo_text += "• Targeted prevention programs dla specific demographic-personality profiles\n"

            self.result_text.delete(1.0, tk.END)
            self.result_text.insert(tk.END, demo_text)

        except Exception as e:
            messagebox.showerror("Błąd", f"Nie można wykonać analizy demografii klastrów: {str(e)}")
            print(f"Błąd w analyze_cluster_demographics: {str(e)}")
        """Przeprowadza klastrowanie K-means"""
        if self.processed_df is None:
            messagebox.showerror("Błąd", "Najpierw wczytaj dane!")
            return

        success, message, cluster_labels = self.cluster_analyzer.perform_clustering(self.processed_df)

        if not success:
            messagebox.showerror("Błąd", message)
            return

        # Pokaż analizę tekstową
        analysis_text = self.cluster_analyzer.get_cluster_analysis_text(self.processed_df)
        self.result_text.delete(1.0, tk.END)
        self.result_text.insert(tk.END, analysis_text)

        # Pokaż wizualizację
        fig, additional_text = self.cluster_analyzer.create_cluster_visualization(self.processed_df)
        if fig:
            self.plot_manager.embed_plot_in_frame(fig, self.plot_frame)

    def perform_enhanced_clustering(self):
        """Przeprowadza klastrowanie K-means"""
        if self.processed_df is None:
            messagebox.showerror("Błąd", "Najpierw wczytaj dane!")
            return

        try:
            success, message, cluster_labels = self.cluster_analyzer.perform_clustering(self.processed_df)

            if not success:
                messagebox.showerror("Błąd", message)
                return

            # Pokaż analizę tekstową
            analysis_text = self.cluster_analyzer.get_cluster_analysis_text(self.processed_df)
            self.result_text.delete(1.0, tk.END)
            self.result_text.insert(tk.END, analysis_text)

            # Pokaż wizualizację
            fig, additional_text = self.cluster_analyzer.create_cluster_visualization(self.processed_df)
            if fig:
                self.plot_manager.embed_plot_in_frame(fig, self.plot_frame)

            messagebox.showinfo("Sukces", "✅ Analiza klastrów zakończona pomyślnie!")

        except Exception as e:
            messagebox.showerror("Błąd", f"Nie można wykonać klastrowania: {str(e)}")
            print(f"Błąd w perform_enhanced_clustering: {str(e)}")

    def perform_classification(self):
        """Przeprowadza klasyfikację Random Forest"""
        if self.processed_df is None:
            messagebox.showerror("Błąd", "Najpierw wczytaj dane!")
            return

        substance = self.classification_substance_var.get()
        if not substance:
            messagebox.showerror("Błąd", "Wybierz substancję do przewidywania!")
            return

        result = self.classification_manager.perform_classification(self.processed_df, substance)

        if result['success']:
            self.result_text.delete(1.0, tk.END)
            self.result_text.insert(tk.END, result['analysis_text'])

            if result['figure']:
                self.plot_manager.embed_plot_in_frame(result['figure'], self.plot_frame)
        else:
            messagebox.showerror("Błąd", result['message'])

    def compare_all_classifications(self):
        """Porównuje klasyfikacje wszystkich substancji"""
        if self.processed_df is None:
            messagebox.showerror("Błąd", "Najpierw wczytaj dane!")
            return

        result = self.classification_manager.compare_all_substances(self.processed_df)

        self.result_text.delete(1.0, tk.END)
        self.result_text.insert(tk.END, result['analysis_text'])

        if result['figure']:
            self.plot_manager.embed_plot_in_frame(result['figure'], self.plot_frame)

    def analyze_cluster_substance_patterns(self):
        """Analizuje wzorce substancji w klastrach"""
        if self.cluster_analyzer.cluster_labels is None:
            messagebox.showwarning("Uwaga", "Najpierw wykonaj analizę klastrów!")
            return

        analysis_text = self.cluster_analyzer.analyze_cluster_substance_patterns(self.processed_df)

        self.result_text.delete(1.0, tk.END)
        self.result_text.insert(tk.END, analysis_text)

    def analyze_cluster_demographics(self):
        """Analizuje demografię klastrów"""
        if self.cluster_analyzer.cluster_labels is None:
            messagebox.showwarning("Uwaga", "Najpierw wykonaj analizę klastrów!")
            return

        # TODO: Implementacja analizy demografii klastrów
        messagebox.showinfo("Info", "Funkcja w przygotowaniu...")

    def create_risk_assessment(self):
        """Tworzy narzędzie oceny ryzyka"""
        if self.processed_df is None:
            messagebox.showerror("Błąd", "Najpierw wczytaj dane!")
            return

        try:
            # Znajdź istniejące kolumny
            existing_personality_cols = [col for col in PERSONALITY_COLS if col in self.processed_df.columns]
            existing_substance_cols = [col for col in SUBSTANCE_COLS if col in self.processed_df.columns]

            if len(existing_personality_cols) < 3:
                messagebox.showwarning("Uwaga", "Za mało cech osobowości do oceny ryzyka")
                return

            # Oblicz wskaźniki ryzyka dla każdej osoby na podstawie osobowości
            risk_scores = {}
            key_substances = ['Cannabis', 'Cocaine', 'Heroin', 'LSD', 'Alcohol']

            for substance in key_substances:
                if substance in existing_substance_cols:
                    # Proste modele ryzyka oparte na kluczowych cechach osobowości
                    if substance == 'Cannabis':
                        risk_scores[substance] = (
                                0.3 * self.processed_df.get('SensationSeeking', 0) +
                                0.25 * self.processed_df.get('Impulsiveness', 0) +
                                0.2 * self.processed_df.get('Openness', 0) -
                                0.15 * self.processed_df.get('Conscientiousness', 0)
                        )
                    elif substance == 'Cocaine':
                        risk_scores[substance] = (
                                0.35 * self.processed_df.get('Impulsiveness', 0) +
                                0.3 * self.processed_df.get('SensationSeeking', 0) +
                                0.2 * self.processed_df.get('Neuroticism', 0) -
                                0.2 * self.processed_df.get('Conscientiousness', 0)
                        )
                    elif substance == 'Heroin':
                        risk_scores[substance] = (
                                0.4 * self.processed_df.get('Impulsiveness', 0) +
                                0.3 * self.processed_df.get('Neuroticism', 0) +
                                0.2 * self.processed_df.get('SensationSeeking', 0) -
                                0.25 * self.processed_df.get('Conscientiousness', 0)
                        )
                    elif substance == 'LSD':
                        risk_scores[substance] = (
                                0.4 * self.processed_df.get('Openness', 0) +
                                0.3 * self.processed_df.get('SensationSeeking', 0) +
                                0.15 * self.processed_df.get('Extraversion', 0) -
                                0.1 * self.processed_df.get('Neuroticism', 0)
                        )
                    elif substance == 'Alcohol':
                        risk_scores[substance] = (
                                0.25 * self.processed_df.get('Extraversion', 0) +
                                0.2 * self.processed_df.get('SensationSeeking', 0) +
                                0.15 * self.processed_df.get('Impulsiveness', 0) -
                                0.1 * self.processed_df.get('Conscientiousness', 0)
                        )

            # Utwórz raport oceny ryzyka
            risk_text = "🚨 === NARZĘDZIE OCENY RYZYKA UŻYWANIA SUBSTANCJI ===\n\n"
            risk_text += "📊 Model oparty na analizie cech osobowości z UCI Dataset\n"
            risk_text += "⚠️ Tylko do celów edukacyjnych i badawczych!\n\n"

            risk_text += "🎯 PERCENTYLE RYZYKA W POPULACJI:\n"
            risk_text += "─" * 50 + "\n"

            for substance, scores in risk_scores.items():
                if len(scores) > 0:
                    percentiles = [10, 25, 50, 75, 90]
                    risk_percentiles = np.percentile(scores, percentiles)

                    risk_text += f"\n💊 {substance.upper()}:\n"
                    for i, (perc, value) in enumerate(zip(percentiles, risk_percentiles)):
                        if perc <= 25:
                            risk_level = "🟢 NISKIE"
                        elif perc <= 50:
                            risk_level = "🟡 UMIARKOWANE"
                        elif perc <= 75:
                            risk_level = "🟠 WYSOKIE"
                        else:
                            risk_level = "🔴 BARDZO WYSOKIE"

                        risk_text += f"  {perc:>2}. percentyl: {value:>6.2f} ({risk_level})\n"

            # Dystrybucja ryzyka w populacji
            risk_text += f"\n📊 DYSTRYBUCJA RYZYKA W POPULACJI:\n"
            risk_text += "─" * 40 + "\n"

            for substance, scores in risk_scores.items():
                if len(scores) > 0:
                    high_risk = (scores > np.percentile(scores, 75)).sum()
                    moderate_risk = ((scores > np.percentile(scores, 25)) &
                                     (scores <= np.percentile(scores, 75))).sum()
                    low_risk = (scores <= np.percentile(scores, 25)).sum()

                    risk_text += f"\n{substance}:\n"
                    risk_text += f"  🔴 Wysokie ryzyko: {high_risk} osób ({high_risk / len(scores) * 100:.1f}%)\n"
                    risk_text += f"  🟡 Umiarkowane: {moderate_risk} osób ({moderate_risk / len(scores) * 100:.1f}%)\n"
                    risk_text += f"  🟢 Niskie ryzyko: {low_risk} osób ({low_risk / len(scores) * 100:.1f}%)\n"

            # Analiza czynników ryzyka
            risk_text += f"\n🔍 KLUCZOWE CZYNNIKI RYZYKA:\n"
            risk_text += "─" * 30 + "\n"
            risk_text += "🔴 IMPULSYWNOŚĆ: Najsilniejszy predyktor używania substancji twardych\n"
            risk_text += "🟠 SENSATION SEEKING: Silnie związane z eksperymentowaniem\n"
            risk_text += "🟡 NEUROTYZM: Predyktor self-medication behaviors\n"
            risk_text += "🔵 OTWARTOŚĆ: Predyktor psychedelików i nowych doświadczeń\n"
            risk_text += "🟢 SUMIENNOŚĆ: Czynnik ochronny przed używaniem substancji\n"

            # Rekomendacje zastosowania
            risk_text += f"\n💡 REKOMENDACJE ZASTOSOWANIA:\n"
            risk_text += "─" * 35 + "\n"
            risk_text += "🎯 Screening populacyjny - identyfikacja grup wysokiego ryzyka\n"
            risk_text += "🏥 Planowanie interwencji - dopasowanie do poziomu ryzyka\n"
            risk_text += "📚 Badania naukowe - stratyfikacja próby badawczej\n"
            risk_text += "🎓 Edukacja - demonstracja czynników ryzyka\n"
            risk_text += "🔬 Prevention research - targeted interventions\n\n"

            risk_text += "⚠️ WAŻNE OGRANICZENIA:\n"
            risk_text += "• Model uproszczony - rzeczywiste ryzyko zależy od wielu czynników\n"
            risk_text += "• Nie uwzględnia czynników środowiskowych i społecznych\n"
            risk_text += "• Oparte na self-report data - możliwe bias\n"
            risk_text += "• Nie zastępuje profesjonalnej oceny klinicznej\n"
            risk_text += "• Tylko do celów badawczych i edukacyjnych\n"

            # Przykład interpretacji
            if len(risk_scores) > 0:
                risk_text += f"\n📋 PRZYKŁAD INTERPRETACJI WYNIKÓW:\n"
                risk_text += "─" * 35 + "\n"

                # Znajdź osobę o wysokim ryzyku jako przykład
                for substance, scores in risk_scores.items():
                    if len(scores) > 0:
                        high_risk_threshold = np.percentile(scores, 75)
                        high_risk_indices = scores[scores > high_risk_threshold].index[:1]

                        if len(high_risk_indices) > 0:
                            idx = high_risk_indices[0]
                            risk_score = scores.iloc[idx]

                            risk_text += f"👤 Osoba #{idx} - {substance}:\n"
                            risk_text += f"  📊 Wynik ryzyka: {risk_score:.3f}\n"
                            risk_text += f"  📍 Percentyl: {(scores <= risk_score).mean() * 100:.0f}\n"

                            if risk_score > np.percentile(scores, 90):
                                risk_text += f"  🚨 Interpretacja: BARDZO WYSOKIE RYZYKO\n"
                                risk_text += f"  💡 Rekomendacja: Priorytetowa interwencja, monitoring\n"
                            elif risk_score > np.percentile(scores, 75):
                                risk_text += f"  🟠 Interpretacja: WYSOKIE RYZYKO\n"
                                risk_text += f"  💡 Rekomendacja: Interwencja prewencyjna, edukacja\n"
                            break

            self.result_text.delete(1.0, tk.END)
            self.result_text.insert(tk.END, risk_text)

            # Wizualizacja dystrybucji ryzyka
            if risk_scores:
                self.create_risk_visualization(risk_scores)

        except Exception as e:
            messagebox.showerror("Błąd", f"Nie można utworzyć narzędzia oceny ryzyka: {str(e)}")
            print(f"Błąd w create_risk_assessment: {str(e)}")

    def create_risk_visualization(self, risk_scores):
        """Tworzy wizualizację rozkładów ryzyka"""
        try:
            # Wyczyść poprzednie wykresy
            for widget in self.plot_frame.winfo_children():
                widget.destroy()

            # Twórz wykresy
            n_substances = len(risk_scores)
            if n_substances == 0:
                return

            fig, axes = plt.subplots(2, (n_substances + 1) // 2, figsize=(15, 10))
            fig.suptitle('🚨 Rozkłady Ryzyka Używania Substancji\n(na podstawie cech osobowości)',
                         fontsize=14, fontweight='bold')

            if n_substances == 1:
                axes = [axes]
            elif n_substances <= 2:
                axes = axes.flatten() if hasattr(axes, 'flatten') else [axes]
            else:
                axes = axes.flatten()

            colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7']

            for i, (substance, scores) in enumerate(risk_scores.items()):
                if i >= len(axes):
                    break

                ax = axes[i]

                # Histogram rozkładu ryzyka
                n, bins, patches = ax.hist(scores, bins=30, alpha=0.7,
                                           color=colors[i % len(colors)],
                                           edgecolor='white', linewidth=0.5)

                # Linie percentyli
                percentiles = [25, 50, 75, 90]
                percentile_values = np.percentile(scores, percentiles)
                line_colors = ['green', 'orange', 'red', 'darkred']

                for perc, value, color in zip(percentiles, percentile_values, line_colors):
                    ax.axvline(value, color=color, linestyle='--', linewidth=2, alpha=0.8,
                               label=f'{perc}p: {value:.2f}')

                ax.set_title(f'{substance}\n(μ={scores.mean():.2f}, σ={scores.std():.2f})',
                             fontweight='bold')
                ax.set_xlabel('Wynik ryzyka')
                ax.set_ylabel('Liczba osób')
                ax.legend(fontsize=8)
                ax.grid(True, alpha=0.3)

            # Usuń nieużywane subploty
            for i in range(len(risk_scores), len(axes)):
                fig.delaxes(axes[i])

            plt.tight_layout()

            # Osadź wykres
            canvas = FigureCanvasTkAgg(fig, self.plot_frame)
            canvas.draw()
            canvas.get_tk_widget().grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        except Exception as e:
            print(f"Błąd w create_risk_visualization: {str(e)}")

    # === METODY POMOCNICZE I ALIASY ===

    def perform_clustering(self):
        """Alias dla perform_enhanced_clustering"""
        return self.perform_enhanced_clustering()

    def show_cluster_analysis(self):
        """Alias dla analyze_cluster_substance_patterns"""
        return self.analyze_cluster_substance_patterns()

    def show_demographic_analysis(self):
        """Alias dla analyze_cluster_demographics"""
        return self.analyze_cluster_demographics()

    def show_risk_assessment(self):
        """Alias dla create_risk_assessment"""
        return self.create_risk_assessment()

    def get_processed_data(self):
        """Zwraca przetworzone dane"""
        return self.processed_df

    def get_original_data(self):
        """Zwraca oryginalne dane"""
        return self.df

    def is_data_loaded(self):
        """Sprawdza czy dane są wczytane"""
        return self.processed_df is not None

    def get_available_substances(self):
        """Zwraca listę dostępnych substancji"""
        if self.processed_df is None:
            return []
        return [col for col in SUBSTANCE_COLS if col in self.processed_df.columns]

    def get_available_personality_traits(self):
        """Zwraca listę dostępnych cech osobowości"""
        if self.processed_df is None:
            return []
        return [col for col in PERSONALITY_COLS if col in self.processed_df.columns]

    def refresh_ui(self):
        """Odświeża całe UI"""
        if self.processed_df is not None:
            self.refresh_table()
            self.update_substance_combos()

    def reset_analysis(self):
        """Resetuje analizy (zachowuje dane)"""
        if hasattr(self, 'cluster_analyzer'):
            self.cluster_analyzer.cluster_labels = None

        # Wyczyść wyniki
        if self.result_text:
            self.result_text.delete(1.0, tk.END)
            self.result_text.insert(tk.END, "Analizy zostały zresetowane.\nMożesz rozpocząć nowe analizy.")

    def export_current_analysis(self):
        """Eksportuje bieżącą analizę do pliku tekstowego"""
        if not self.result_text:
            messagebox.showwarning("Uwaga", "Brak wyników do eksportu!")
            return

        content = self.result_text.get(1.0, tk.END)
        if not content.strip():
            messagebox.showwarning("Uwaga", "Brak wyników do eksportu!")
            return

        file_path = filedialog.asksaveasfilename(
            title="Eksportuj analizę",
            defaultextension=".txt",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")]
        )

        if file_path:
            try:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(content)
                messagebox.showinfo("Sukces", f"Analiza zapisana do: {file_path}")
            except Exception as e:
                messagebox.showerror("Błąd", f"Nie można zapisać pliku: {str(e)}")

    def show_help(self):
        """Pokazuje okno pomocy"""
        help_window = tk.Toplevel(self.root)
        help_window.title("📚 Pomoc - UCI Drug Consumption Analyzer")
        help_window.geometry("600x400")
        help_window.transient(self.root)

        help_text = scrolledtext.ScrolledText(help_window, wrap=tk.WORD)
        help_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        help_content = """
🧠 UCI DRUG CONSUMPTION ANALYZER - POMOC

📋 PODSTAWOWE KROKI:
1. Wczytaj dane: "📂 Wczytaj dane UCI" lub "🧪 Utwórz dane testowe"
2. Sprawdź tabelę: Zakładka "📋 Tabela Danych"
3. Wykonaj analizy: Użyj przycisków w panelu kontrolnym

📊 GŁÓWNE FUNKCJE:
• Statystyki opisowe - podstawowe statystyki wszystkich zmiennych
• Korelacje - związki między cechami osobowości i substancjami
• Filtrowanie - wybór konkretnych grup danych
• Klastrowanie - identyfikacja profili osobowości
• Klasyfikacja - przewidywanie używania substancji

📋 EDYCJA DANYCH:
• Podwójne kliknięcie na komórkę = edycja
• Enter = zapisz, Escape = anuluj
• "💾 Zapisz zmiany" - eksport do CSV

🎯 WSKAZÓWKI:
• Użyj danych testowych do nauki obsługi
• Sprawdź wszystkie zakładki (Tabela, Tekstowe, Wykresy)
• Filtry pomagają analizować konkretne grupy
• Klastrowanie pokazuje profile osobowości
• Eksportuj wyniki do dalszej analizy

❗ ROZWIĄZYWANIE PROBLEMÓW:
• Błędy wczytywania: Użyj danych testowych
• Brak danych: Sprawdź czy plik ma 32 kolumny
• Błędy filtrów: Sprawdź czy dane mają kolumny demograficzne
• Powolne działanie: Użyj mniejszych zbiorów danych

📞 KONTAKT:
Autor: Karol Dąbrowski
Dataset: UCI Drug Consumption (Quantified)
"""

        help_text.insert(tk.END, help_content)
        help_text.config(state=tk.DISABLED)

    def show_about(self):
        """Pokazuje okno o aplikacji"""
        about_text = """
🧠 UCI Drug Consumption Analyzer v1.2

📋 OPIS:
Zaawansowana aplikacja do analizy wzorców konsumpcji 
narkotyków na podstawie cech osobowości Big Five.

✨ FUNKCJONALNOŚCI:
• Kompletna implementacja wymagań na ocenę 3
• Edytowalna tabela danych
• Zaawansowane analizy ML (klastrowanie, klasyfikacja)
• Interaktywne wizualizacje z interpretacjami
• Narzędzia oceny ryzyka

👨‍💻 AUTOR: Karol Dąbrowski
📊 DATASET: UCI Drug Consumption (Quantified)
🎓 PROJEKT: Analiza danych - ocena 3
📅 ROK: 2024

🔧 TECHNOLOGIE:
Python, pandas, scikit-learn, matplotlib, tkinter

⚠️ DISCLAIMER:
Aplikacja służy wyłącznie celom edukacyjnym 
i badawczym. Nie zastępuje profesjonalnej 
oceny medycznej lub psychologicznej.
"""
        messagebox.showinfo("O aplikacji", about_text)

    def setup_data_table(self):
        """Konfiguruje edytowalną tabelę danych"""
        # Ramka z przyciskami
        button_frame = ttk.Frame(self.table_frame)
        button_frame.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=(0, 10))

        ttk.Button(button_frame, text="🔄 Odśwież tabelę",
                   command=self.refresh_table).grid(row=0, column=0, padx=(0, 5))
        ttk.Button(button_frame, text="💾 Zapisz zmiany",
                   command=self.save_table_changes).grid(row=0, column=1, padx=(0, 5))
        ttk.Button(button_frame, text="📤 Eksportuj CSV",
                   command=self.export_table_csv).grid(row=0, column=2, padx=(0, 5))
        ttk.Button(button_frame, text="📄 Eksportuj analizę",
                   command=self.export_current_analysis).grid(row=0, column=3, padx=(0, 5))

        # Etykieta informacyjna
        self.table_info_label = ttk.Label(button_frame, text="Tabela nie jest wczytana")
        self.table_info_label.grid(row=0, column=4, padx=(20, 0))

        # Ramka dla tabeli z scrollbarami
        table_container = ttk.Frame(self.table_frame)
        table_container.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # Tabela (Treeview)
        self.data_table = ttk.Treeview(table_container)

        # Scrollbary
        self.table_scrollbar_v = ttk.Scrollbar(table_container, orient="vertical",
                                               command=self.data_table.yview)
        self.table_scrollbar_h = ttk.Scrollbar(table_container, orient="horizontal",
                                               command=self.data_table.xview)

        # Konfiguracja scrollbarów
        self.data_table.configure(yscrollcommand=self.table_scrollbar_v.set,
                                  xscrollcommand=self.table_scrollbar_h.set)

        # Umieszczenie elementów
        self.data_table.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        self.table_scrollbar_v.grid(row=0, column=1, sticky=(tk.N, tk.S))
        self.table_scrollbar_h.grid(row=1, column=0, sticky=(tk.W, tk.E))

        # Grid weights
        self.table_frame.columnconfigure(0, weight=1)
        self.table_frame.rowconfigure(1, weight=1)
        table_container.columnconfigure(0, weight=1)
        table_container.rowconfigure(0, weight=1)

        # Bind events dla edycji
        self.data_table.bind('<Double-1>', self.on_table_double_click)
        self.data_table.bind('<Return>', self.on_table_enter)

    def refresh_table(self):
        """Odświeża tabelę z aktualnymi danymi"""
        if self.processed_df is None:
            messagebox.showwarning("Uwaga", "Najpierw wczytaj dane!")
            return

        # Wyczyść tabelę
        for item in self.data_table.get_children():
            self.data_table.delete(item)

        # Konfiguruj kolumny - pokaż WSZYSTKIE kolumny
        df_to_show = self.processed_df.copy()
        display_columns = list(df_to_show.columns)

        # USUNIĘTE OGRANICZENIE: Pokaż wszystkie kolumny zamiast tylko 20

        # Konfiguruj Treeview
        self.data_table['columns'] = display_columns
        self.data_table['show'] = 'tree headings'

        # Konfiguruj kolumny
        self.data_table.column('#0', width=50, minwidth=50, anchor='center')
        self.data_table.heading('#0', text='#', anchor='center')

        for col in display_columns:
            # Dostosuj szerokość kolumny do typu danych
            if col.endswith('_Category'):
                width = 120  # Szerzej dla kategorii tekstowych
            elif col in PERSONALITY_COLS:
                width = 80   # Węziej dla liczb
            elif col in SUBSTANCE_COLS:
                width = 70   # Najwęziej dla substancji
            else:
                width = 100  # Domyślna szerokość

            self.data_table.column(col, width=width, minwidth=60, anchor='center')
            self.data_table.heading(col, text=col, anchor='center')

        # Dodaj dane (pierwsze 1000 wierszy dla wydajności)
        max_rows = min(1000, len(df_to_show))
        for i in range(max_rows):
            row_data = []
            for col in display_columns:
                value = df_to_show.iloc[i][col]
                # Formatuj wartości
                if pd.isna(value):
                    formatted_value = "NaN"
                elif isinstance(value, float):
                    if abs(value) < 0.001 and value != 0:
                        formatted_value = f"{value:.2e}"  # Notacja naukowa dla bardzo małych liczb
                    else:
                        formatted_value = f"{value:.3f}"
                else:
                    formatted_value = str(value)
                row_data.append(formatted_value)

            self.data_table.insert('', 'end', text=str(i), values=row_data)

        # Aktualizuj etykietę
        total_rows = len(df_to_show)
        shown_rows = min(1000, total_rows)
        total_cols = len(display_columns)
        self.table_info_label.config(
            text=f"Wyświetlono {shown_rows} z {total_rows} wierszy, {total_cols} kolumn (wszystkie dostępne)"
        )

    def on_table_double_click(self, event):
        """Obsługuje podwójne kliknięcie na komórkę"""
        item = self.data_table.selection()[0]
        column = self.data_table.identify_column(event.x)

        # Uzyskaj wartość kolumny
        if column == '#0':
            return  # Nie edytuj numeru wiersza

        column_index = int(column[1:]) - 1  # Kolumny są numerowane od #1
        if column_index >= len(self.data_table['columns']):
            return

        column_name = self.data_table['columns'][column_index]
        current_value = self.data_table.item(item, 'values')[column_index]

        # Otwórz okno edycji
        self.open_edit_dialog(item, column_name, column_index, current_value)

    def on_table_enter(self, event):
        """Obsługuje naciśnięcie Enter na komórce"""
        if self.data_table.selection():
            self.on_table_double_click(event)

    def open_edit_dialog(self, item, column_name, column_index, current_value):
        """Otwiera dialog edycji komórki"""
        dialog = tk.Toplevel(self.root)
        dialog.title(f"Edytuj: {column_name}")
        dialog.geometry("300x150")
        dialog.transient(self.root)
        dialog.grab_set()

        # Etykieta
        ttk.Label(dialog, text=f"Kolumna: {column_name}").pack(pady=10)

        # Entry z aktualną wartością
        value_var = tk.StringVar(value=str(current_value))
        entry = ttk.Entry(dialog, textvariable=value_var, width=30)
        entry.pack(pady=10)
        entry.focus()
        entry.select_range(0, tk.END)

        # Ramka przycisków
        button_frame = ttk.Frame(dialog)
        button_frame.pack(pady=10)

        def save_changes():
            new_value = value_var.get()
            self.update_table_cell(item, column_index, new_value, column_name)
            dialog.destroy()

        def cancel_changes():
            dialog.destroy()

        ttk.Button(button_frame, text="💾 Zapisz",
                   command=save_changes).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="❌ Anuluj",
                   command=cancel_changes).pack(side=tk.LEFT, padx=5)

        # Bind Enter i Escape
        dialog.bind('<Return>', lambda e: save_changes())
        dialog.bind('<Escape>', lambda e: cancel_changes())

    def update_table_cell(self, item, column_index, new_value, column_name):
        """Aktualizuje komórkę w tabeli i DataFrame"""
        try:
            # Aktualizuj tabelę
            current_values = list(self.data_table.item(item, 'values'))
            current_values[column_index] = new_value
            self.data_table.item(item, values=current_values)

            # Aktualizuj DataFrame
            row_index = int(self.data_table.item(item, 'text'))

            # Konwersja wartości do odpowiedniego typu
            if column_name in self.processed_df.columns:
                original_dtype = self.processed_df[column_name].dtype

                try:
                    if pd.api.types.is_numeric_dtype(original_dtype):
                        converted_value = pd.to_numeric(new_value)
                    else:
                        converted_value = str(new_value)

                    self.processed_df.at[row_index, column_name] = converted_value

                    messagebox.showinfo("Sukces",
                                        f"Zaktualizowano {column_name}[{row_index}] = {new_value}")
                except (ValueError, TypeError) as e:
                    messagebox.showerror("Błąd",
                                         f"Nie można konwertować wartości: {str(e)}")

        except Exception as e:
            messagebox.showerror("Błąd", f"Nie można zaktualizować komórki: {str(e)}")

    def save_table_changes(self):
        """Zapisuje zmiany do pliku"""
        if self.processed_df is None:
            messagebox.showwarning("Uwaga", "Brak danych do zapisania!")
            return

        file_path = filedialog.asksaveasfilename(
            title="Zapisz dane",
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )

        if file_path:
            try:
                self.processed_df.to_csv(file_path, index=False)
                messagebox.showinfo("Sukces", f"Dane zapisane do: {file_path}")
            except Exception as e:
                messagebox.showerror("Błąd", f"Nie można zapisać pliku: {str(e)}")

    def export_table_csv(self):
        """Eksportuje aktualnie wyświetlane dane z tabeli"""
        if self.data_table.get_children():
            file_path = filedialog.asksaveasfilename(
                title="Eksportuj widoczne dane",
                defaultextension=".csv",
                filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
            )

            if file_path:
                try:
                    # Zbierz dane z tabeli
                    columns = ['Row_Index'] + list(self.data_table['columns'])
                    data = []

                    for child in self.data_table.get_children():
                        row_data = [self.data_table.item(child, 'text')] + list(self.data_table.item(child, 'values'))
                        data.append(row_data)

                    # Utwórz DataFrame i zapisz
                    export_df = pd.DataFrame(data, columns=columns)
                    export_df.to_csv(file_path, index=False)

                    messagebox.showinfo("Sukces", f"Eksport zakończony: {file_path}")
                except Exception as e:
                    messagebox.showerror("Błąd", f"Nie można eksportować: {str(e)}")
        else:
            messagebox.showwarning("Uwaga", "Tabela jest pusta!")

    def create_test_data(self):
        """Tworzy przykładowe dane testowe"""
        try:
            import numpy as np

            # Generuj przykładowe dane
            n_samples = 200
            np.random.seed(42)

            # ID
            data = {'ID': range(1, n_samples + 1)}

            # Demografia (znormalizowane wartości)
            data['Age'] = np.random.choice(list(AGE_MAPPING.keys()), n_samples)
            data['Gender'] = np.random.choice(list(GENDER_MAPPING.keys()), n_samples)
            data['Education'] = np.random.choice(list(EDUCATION_MAPPING.keys()), n_samples)
            data['Country'] = np.random.choice(list(COUNTRY_MAPPING.keys()), n_samples)
            data['Ethnicity'] = np.random.choice(list(ETHNICITY_MAPPING.keys()), n_samples)

            # Cechy osobowości (znormalizowane, μ≈0, σ≈1)
            for col in PERSONALITY_COLS:
                data[col] = np.random.normal(0, 1, n_samples)

            # Substancje (skala 0-6, większość ma niskie wartości)
            for col in SUBSTANCE_COLS:
                # Różne prawdopodobieństwa dla różnych substancji
                if col in ['Caffeine', 'Alcohol', 'Chocolate', 'Nicotine']:
                    # Legalne - wyższe prawdopodobieństwo
                    prob = 0.7
                elif col in ['Cannabis']:
                    prob = 0.3
                elif col in ['Cocaine', 'Ecstasy']:
                    prob = 0.1
                elif col in ['Heroin', 'Crack']:
                    prob = 0.02
                else:
                    prob = 0.15

                # Generuj wartości 0-6 z odpowiednim prawdopodobieństwem
                values = np.random.choice([0, 1, 2, 3, 4, 5, 6], n_samples,
                                          p=[1-prob, prob*0.3, prob*0.25, prob*0.2,
                                             prob*0.15, prob*0.07, prob*0.03])
                data[col] = values

            # Utwórz DataFrame
            self.df = pd.DataFrame(data)
            self.processed_df = self.df.copy()

            # Przetwórz dane
            success, message = self.data_processor._process_initial_data(self.df)
            if success:
                self.processed_df = self.df.copy()

                # Zaktualizuj UI
                self.data_info_label.config(
                    text=f"✅ Utworzono {len(self.df)} przykładowych rekordów",
                    foreground="green"
                )

                # Pokaż informacje
                self.display_data_info()

                # Zaktualizuj combobox-y
                self.update_substance_combos()

                # Odśwież tabelę
                self.refresh_table()

                messagebox.showinfo("Sukces", "🧪 Utworzono przykładowe dane testowe!")
            else:
                messagebox.showerror("Błąd", f"Błąd przetwarzania danych testowych: {message}")

        except Exception as e:
            messagebox.showerror("Błąd", f"Nie można utworzyć danych testowych: {str(e)}")

    # Dodatkowe metody aliasowe dla kompatybilności
    def perform_clustering(self):
        """Alias dla perform_enhanced_clustering"""
        return self.perform_enhanced_clustering()

    def show_cluster_analysis(self):
        """Alias dla analyze_cluster_substance_patterns"""
        return self.analyze_cluster_substance_patterns()

    def show_demographic_analysis(self):
        """Alias dla analyze_cluster_demographics"""
        return self.analyze_cluster_demographics()

    def show_risk_assessment(self):
        """Alias dla create_risk_assessment"""
        return self.create_risk_assessment()