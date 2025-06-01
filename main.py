import tkinter as tk
import warnings
from tkinter import ttk, filedialog, messagebox, scrolledtext

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import KNNImputer
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore')

# Set modern color palette
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


class DrugConsumptionAnalyzer:
    def __init__(self, root):
        self.data_info_label = None
        self.data_info_label = None
        self.result_frame = None
        self.root = root
        self.root.title("🧠 Analiza Wzorców Konsumpcji Narkotyków - UCI Dataset | Karol Dąbrowski")
        self.root.geometry("1920x1080")
        self.root.configure(bg='#f8f9fa')

        # Dane
        self.df = None
        self.processed_df = None
        self.cluster_labels = None

        # Kolory
        self.colors = {
            'primary': '#2E86AB',
            'secondary': '#A23B72',
            'success': '#F18F01',
            'danger': '#C73E1D',
            'warning': '#FFB700',
            'info': '#4ECDC4',
            'light': '#F8F9FA',
            'dark': '#495057',
            'cluster_colors': ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4'],
            'risk_gradient': ['#2ECC71', '#F39C12', '#E74C3C']
        }

        # Kolumny
        self.demographic_cols = ['Age', 'Gender', 'Education', 'Country', 'Ethnicity']
        self.personality_cols = ['Neuroticism', 'Extraversion', 'Openness', 'Agreeableness',
                                 'Conscientiousness', 'Impulsiveness', 'SensationSeeking']
        self.substance_cols = ['Alcohol', 'Amphetamines', 'AmylNitrite', 'Benzodiazepines',
                               'Cannabis', 'Chocolate', 'Cocaine', 'Caffeine', 'Crack',
                               'Ecstasy', 'Heroin', 'Ketamine', 'LegalHighs', 'LSD',
                               'Methadone', 'Mushrooms', 'Nicotine', 'Semer', 'VSA']

        # Mapowania
        self.age_mapping = {
            -0.95197: '18-24', -0.07854: '25-34', 0.49788: '35-44',
            1.09449: '45-54', 1.82213: '55-64', 2.59171: '65+'
        }

        self.gender_mapping = {
            0.48246: 'Female', -0.48246: 'Male'
        }

        self.education_mapping = {
            -2.43591: 'Left school before 16',
            -1.73790: 'Left school at 16',
            -1.43719: 'Left school at 17',
            -1.22751: 'Left school at 18',
            -0.61113: 'Some college',
            -0.05921: 'Professional certificate',
            0.45468: 'University degree',
            1.16365: 'Masters degree',
            1.98437: 'Doctorate degree'
        }

        self.country_mapping = {
            -0.09765: 'Australia', 0.24923: 'Canada', -0.46841: 'New Zealand',
            -0.28519: 'Other', 0.21128: 'Republic of Ireland', 0.96082: 'UK', -0.57009: 'USA'
        }

        self.ethnicity_mapping = {
            -0.50212: 'Asian', -1.10702: 'Black', 1.90725: 'Mixed-Black/Asian',
            0.12600: 'Mixed-White/Asian', -0.22166: 'Mixed-White/Black', 0.11440: 'Other',
            -0.31685: 'White'
        }

        self.cluster_profiles = {
            0: {
                'name': 'Ekstrawertyczni Poszukiwacze',
                'emoji': '🎉',
                'color': '#FF6B6B',
                'risk': 'Umiarkowane',
                'description': 'Towarzyski, stabilni emocjonalnie, eksperymentatorzy'
            },
            1: {
                'name': 'Impulsywni w Kryzysie',
                'emoji': '💥',
                'color': '#E74C3C',
                'risk': 'Bardzo Wysokie',
                'description': 'Lękliwi, impulsywni, zdezorganizowani, konfliktowi'
            },
            2: {
                'name': 'Lękliwi Izolowani',
                'emoji': '😔',
                'color': '#3498DB',
                'risk': 'Umiarkowane',
                'description': 'Introwertyczni, lękliwi, unikają ryzyka'
            },
            3: {
                'name': 'Stabilni Conservations',
                'emoji': '🛡️',
                'color': '#2ECC71',
                'risk': 'Niskie',
                'description': 'Zdyscyplinowani, kontrolowani, tradycyjni'
            }
        }

        # Inicjalizacja GUI z przewijalnym interfejsem
        self.setup_ui()

    def setup_ui(self):
        # Enhanced style
        style = ttk.Style()
        style.theme_use('clam')

        # Custom styles
        style.configure('Title.TLabel', font=('Segoe UI', 16, 'bold'),
                        foreground=self.colors['primary'])
        style.configure('Header.TLabel', font=('Segoe UI', 12, 'bold'),
                        foreground=self.colors['dark'])
        style.configure('Success.TButton', foreground='white', background=self.colors['success'])
        style.map('Success.TButton',
                  background=[('active', '#e07b00'), ('pressed', '#c76c00')])

        style.configure('Danger.TButton', foreground='white', background=self.colors['danger'])
        style.map('Danger.TButton',
                  background=[('active', '#b8321b'), ('pressed', '#a52a16')])

        style.configure('TCombobox', padding=5)
        style.configure('TLabelFrame', font=('Segoe UI', 10, 'bold'))

        # Główna ramka z paddingiem
        main_frame = ttk.Frame(self.root, padding="15")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # Grid weights (skalowalność)
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(1, weight=1)

        # Lewy panel – sterowanie (scrollable)
        control_container = ttk.LabelFrame(main_frame, text="🎛️ Panel Kontrolny", padding="0")
        control_container.grid(row=1, column=0, sticky=(tk.N, tk.S, tk.W), padx=(0, 15))

        # Canvas i scrollbar do przewijania panelu sterowania
        canvas = tk.Canvas(control_container, borderwidth=0, highlightthickness=0)
        scrollbar = ttk.Scrollbar(control_container, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)

        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(
                scrollregion=canvas.bbox("all")
            )
        )

        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.grid(row=0, column=0, sticky=(tk.N, tk.S, tk.W, tk.E))
        scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))

        # Ustawienie skalowania
        control_container.columnconfigure(0, weight=1)
        control_container.rowconfigure(0, weight=1)

        # Prawy panel – wyniki analizy
        self.result_frame = ttk.LabelFrame(main_frame, text="📊 Wyniki Analizy", padding="15")
        self.result_frame.grid(row=1, column=1, sticky=(tk.N, tk.S, tk.E, tk.W))

        # Setup pod-panele, zamiast control_frame dajemy scrollable_frame
        self.setup_control_panel(scrollable_frame)
        self.setup_result_panel()

    def setup_control_panel(self, parent):
        row = 0

        # 1. Data Loading with icon
        data_frame = ttk.LabelFrame(parent, text="📁 1. Wczytywanie Danych", padding="10")
        data_frame.grid(row=row, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))
        row += 1

        ttk.Button(data_frame, text="📂 Wczytaj dane UCI (.data/.csv)",
                   command=self.load_data).grid(row=0, column=0, columnspan=2,
                                                sticky=(tk.W, tk.E), pady=(0, 5))

        self.data_info_label = ttk.Label(data_frame, text="❌ Brak danych", foreground="red")
        self.data_info_label.grid(row=1, column=0, columnspan=2, sticky=tk.W, pady=(5, 0))

        # 2. Statistics
        stats_frame = ttk.LabelFrame(parent, text="📈 2. Statystyki Opisowe", padding="10")
        stats_frame.grid(row=row, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))
        row += 1

        ttk.Button(stats_frame, text="📊 Oblicz statystyki podstawowe",
                   command=self.calculate_basic_stats).grid(row=0, column=0, columnspan=2,
                                                            sticky=(tk.W, tk.E), pady=(0, 5))

        # 3. Correlations
        corr_frame = ttk.LabelFrame(parent, text="🔗 3. Analiza Korelacji", padding="10")
        corr_frame.grid(row=row, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))
        row += 1

        ttk.Button(corr_frame, text="🧠 Korelacje cech osobowości",
                   command=self.analyze_personality_correlations).grid(row=0, column=0, columnspan=2,
                                                                       sticky=(tk.W, tk.E), pady=(0, 5))

        ttk.Button(corr_frame, text="💊 Korelacje substancji",
                   command=self.analyze_substance_correlations).grid(row=1, column=0, columnspan=2,
                                                                     sticky=(tk.W, tk.E), pady=(0, 5))

        # 4. Filtering
        filter_frame = ttk.LabelFrame(parent, text="🎯 4. Filtrowanie Danych", padding="10")
        filter_frame.grid(row=row, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))
        row += 1

        self.filter_var = tk.StringVar()
        filter_combo = ttk.Combobox(filter_frame, textvariable=self.filter_var, width=25)
        filter_combo['values'] = ('Wszystkie dane',
                                  'Tylko mężczyźni', 'Tylko kobiety',
                                  'Wiek 18-24', 'Wiek 25-34', 'Wiek 35-44',
                                  'Wysokie wykształcenie', 'UK/USA/Canada',
                                  'Używający Cannabis', 'Używający Alcohol',
                                  'Nieużywający narkotyków')
        filter_combo.set('Wszystkie dane')
        filter_combo.grid(row=0, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 5))

        ttk.Button(filter_frame, text="✅ Zastosuj filtr",
                   command=self.apply_filter).grid(row=1, column=0, columnspan=2,
                                                   sticky=(tk.W, tk.E), pady=(0, 5))

        # 5. Data Processing
        process_frame = ttk.LabelFrame(parent, text="⚙️ 5. Przetwarzanie Danych", padding="10")
        process_frame.grid(row=row, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))
        row += 1

        ttk.Button(process_frame, text="📏 Standaryzacja cech",
                   command=self.standardize_features).grid(row=0, column=0, columnspan=2,
                                                           sticky=(tk.W, tk.E), pady=(0, 5))

        ttk.Button(process_frame, text="🔧 Obsługa brakujących wartości",
                   command=self.handle_missing_values).grid(row=1, column=0, columnspan=2,
                                                            sticky=(tk.W, tk.E), pady=(0, 5))

        ttk.Button(process_frame, text="🔀 Kodowanie binarne",
                   command=self.binary_encode).grid(row=2, column=0, columnspan=2,
                                                    sticky=(tk.W, tk.E), pady=(0, 5))

        # 6. Visualizations
        viz_frame = ttk.LabelFrame(parent, text="📈 6. Wizualizacje", padding="10")
        viz_frame.grid(row=row, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))
        row += 1

        ttk.Button(viz_frame, text="📊 Rozkłady cech osobowości",
                   command=self.plot_personality_distributions).grid(row=0, column=0, columnspan=2,
                                                                     sticky=(tk.W, tk.E), pady=(0, 5))

        ttk.Button(viz_frame, text="📈 Częstość używania substancji",
                   command=self.plot_substance_usage).grid(row=1, column=0, columnspan=2,
                                                           sticky=(tk.W, tk.E), pady=(0, 5))

        # Substance selection for demographic analysis
        ttk.Label(viz_frame, text="Wybierz substancję:").grid(row=2, column=0, sticky=tk.W)

        self.substance_var = tk.StringVar()
        substance_combo = ttk.Combobox(viz_frame, textvariable=self.substance_var, width=25)
        substance_combo['values'] = self.substance_cols
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

        # 7. Advanced Modeling
        model_frame = ttk.LabelFrame(parent, text="🤖 7. Modelowanie Zaawansowane", padding="10")
        model_frame.grid(row=row, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))
        row += 1

        ttk.Button(model_frame, text="🎯 Klastrowanie K-means + Profile",
                   command=self.perform_enhanced_clustering).grid(row=0, column=0, columnspan=2,
                                                                  sticky=(tk.W, tk.E), pady=(0, 5))

        # Substance selection for classification
        ttk.Label(model_frame, text="Substancja do przewidywania:").grid(row=1, column=0, sticky=tk.W)

        self.classification_substance_var = tk.StringVar()
        classification_combo = ttk.Combobox(model_frame, textvariable=self.classification_substance_var, width=25)
        classification_combo['values'] = self.substance_cols
        classification_combo.set('Cannabis')
        classification_combo.grid(row=2, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 5))

        ttk.Button(model_frame, text="🌲 Klasyfikacja Random Forest",
                   command=self.perform_classification).grid(row=3, column=0, columnspan=2,
                                                             sticky=(tk.W, tk.E), pady=(0, 5))

        # 8. UCI-Specific Analysis
        uci_frame = ttk.LabelFrame(parent, text="🎓 8. Analizy Specyficzne UCI", padding="10")
        uci_frame.grid(row=row, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))
        row += 1

        ttk.Button(uci_frame, text="🧬 Profile Klastrów vs Substancje",
                   command=self.analyze_cluster_substance_patterns).grid(row=0, column=0, columnspan=2,
                                                                         sticky=(tk.W, tk.E), pady=(0, 5))

        ttk.Button(uci_frame, text="👥 Demografia Klastrów",
                   command=self.analyze_cluster_demographics).grid(row=1, column=0, columnspan=2,
                                                                   sticky=(tk.W, tk.E), pady=(0, 5))

        ttk.Button(uci_frame, text="🚨 Risk Assessment Tool",
                   command=self.create_risk_assessment).grid(row=2, column=0, columnspan=2,
                                                             sticky=(tk.W, tk.E), pady=(0, 5))

    def setup_result_panel(self):
        # Enhanced notebook with better styling
        self.notebook = ttk.Notebook(self.result_frame)
        self.notebook.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # Configure grid
        self.result_frame.columnconfigure(0, weight=1)
        self.result_frame.rowconfigure(0, weight=1)

        # Text results tab
        self.text_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.text_frame, text="📝 Wyniki Tekstowe")

        self.result_text = scrolledtext.ScrolledText(self.text_frame, wrap=tk.WORD,
                                                     width=90, height=35, font=('Consolas', 10))
        self.result_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        self.text_frame.columnconfigure(0, weight=1)
        self.text_frame.rowconfigure(0, weight=1)

        # Plot tab
        self.plot_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.plot_frame, text="📊 Wykresy Interaktywne")

        # Interactive analysis tab
        self.interactive_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.interactive_frame, text="🎯 Analiza Interaktywna")

    def decode_demographics(self):
        """Dekoduje zmienne demograficzne z wartości liczbowych na kategorie"""

        def find_closest_key(value, mapping_dict):
            if pd.isna(value):
                return 'Unknown'
            closest_key = min(mapping_dict.keys(), key=lambda x: abs(x - value))
            return mapping_dict[closest_key]

        # Dekoduj zmienne demograficzne
        if 'Age' in self.df.columns:
            self.df['Age_Category'] = self.df['Age'].apply(
                lambda x: find_closest_key(x, self.age_mapping))

        if 'Gender' in self.df.columns:
            self.df['Gender_Category'] = self.df['Gender'].apply(
                lambda x: find_closest_key(x, self.gender_mapping))

        if 'Education' in self.df.columns:
            self.df['Education_Category'] = self.df['Education'].apply(
                lambda x: find_closest_key(x, self.education_mapping))

        if 'Country' in self.df.columns:
            self.df['Country_Category'] = self.df['Country'].apply(
                lambda x: find_closest_key(x, self.country_mapping))

        if 'Ethnicity' in self.df.columns:
            self.df['Ethnicity_Category'] = self.df['Ethnicity'].apply(
                lambda x: find_closest_key(x, self.ethnicity_mapping))

    def load_data(self):
        """Enhanced data loading with better validation"""
        file_path = filedialog.askopenfilename(
            title="Wybierz plik z danymi UCI",
            filetypes=[("Data files", "*.data"), ("CSV files", "*.csv"), ("All files", "*.*")]
        )

        if file_path:
            try:
                # Load with proper column structure
                if file_path.endswith('.data'):
                    column_names = (['ID', 'Age', 'Gender', 'Education', 'Country', 'Ethnicity'] +
                                    self.personality_cols + self.substance_cols)
                    self.df = pd.read_csv(file_path, header=None, names=column_names)
                else:
                    self.df = pd.read_csv(file_path)

                # Validate data structure
                if len(self.df.columns) < 30:
                    messagebox.showwarning("Uwaga",
                                           f"Plik ma tylko {len(self.df.columns)} kolumn. Oczekiwane: 32 kolumny UCI.")

                # Convert substance labels
                consumption_mapping = {
                    'CL0': 0, 'CL1': 1, 'CL2': 2, 'CL3': 3,
                    'CL4': 4, 'CL5': 5, 'CL6': 6
                }

                for col in self.substance_cols:
                    if col in self.df.columns:
                        self.df[col] = self.df[col].map(consumption_mapping)

                # Decode demographics
                self.decode_demographics()
                self.processed_df = self.df.copy()

                # Update UI
                info_text = f"✅ Wczytano {len(self.df)} rekordów, {len(self.df.columns)} kolumn"
                self.data_info_label.config(text=info_text, foreground="green")

                # Display enhanced data info
                self.display_enhanced_data_info()

                messagebox.showinfo("Sukces", "✅ Dane UCI zostały pomyślnie wczytane i przetworzone!")

            except Exception as e:
                messagebox.showerror("Błąd", f"❌ Nie można wczytać danych: {str(e)}")

    def display_enhanced_data_info(self):
        """Enhanced data information display"""
        if self.df is None:
            return

        info_text = f"""
🎓 === DATASET UCI DRUG CONSUMPTION (QUANTIFIED) ===

📊 ROZMIAR DANYCH: {self.df.shape[0]} respondentów × {self.df.shape[1]} zmiennych

🧑‍🤝‍🧑 CHARAKTERYSTYKA PRÓBY:
"""

        # Demographics if available
        if 'Age_Category' in self.df.columns:
            age_dist = self.df['Age_Category'].value_counts()
            info_text += f"\n👶 WIEK:\n{age_dist.to_string()}\n"

        if 'Gender_Category' in self.df.columns:
            gender_dist = self.df['Gender_Category'].value_counts()
            info_text += f"\n⚧️ PŁEĆ:\n{gender_dist.to_string()}\n"

        if 'Education_Category' in self.df.columns:
            edu_dist = self.df['Education_Category'].value_counts().head(5)
            info_text += f"\n🎓 WYKSZTAŁCENIE (top 5):\n{edu_dist.to_string()}\n"

        if 'Country_Category' in self.df.columns:
            country_dist = self.df['Country_Category'].value_counts()
            info_text += f"\n🌍 KRAJ:\n{country_dist.to_string()}\n"

        # Personality statistics
        info_text += f"""
🧠 CECHY OSOBOWOŚCI (znormalizowane):
{self.df[self.personality_cols].describe().round(3).to_string()}

💊 TOP 10 UŻYWANYCH SUBSTANCJI:
"""

        # Substance usage ranking
        usage_stats = {}
        for col in self.substance_cols:
            if col in self.df.columns:
                usage_count = (self.df[col] > 0).sum()
                usage_pct = usage_count / len(self.df) * 100
                usage_stats[col] = (usage_count, usage_pct)

        sorted_usage = sorted(usage_stats.items(), key=lambda x: x[1][1], reverse=True)[:10]
        for i, (substance, (count, pct)) in enumerate(sorted_usage, 1):
            emoji = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else f"{i:>2}."
            info_text += f"{emoji} {substance:<15} {count:>4} osób ({pct:>5.1f}%)\n"

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

    def get_risk_color(self, value, reverse=False):
        """Get color based on risk level"""
        if reverse:
            if value > 0.3:
                return self.colors['risk_gradient'][0]  # Green for high positive
            elif value > 0:
                return self.colors['risk_gradient'][1]  # Orange for moderate
            else:
                return self.colors['risk_gradient'][2]  # Red for negative
        else:
            if value > 0.3:
                return self.colors['risk_gradient'][2]  # Red for high risk
            elif value > 0:
                return self.colors['risk_gradient'][1]  # Orange for moderate
            else:
                return self.colors['risk_gradient'][0]  # Green for low risk

    def plot_demographic_boxplots(self):
        """Enhanced demographic boxplots with better colors"""
        global p_value
        if self.processed_df is None:
            messagebox.showerror("Błąd", "Najpierw wczytaj dane!")
            return

        selected_substance = self.substance_var.get()
        if not selected_substance or selected_substance not in self.processed_df.columns:
            messagebox.showerror("Błąd", "Wybierz prawidłową substancję!")
            return

        # Clear previous plots
        for widget in self.plot_frame.winfo_children():
            widget.destroy()

        users = self.processed_df[self.processed_df[selected_substance] > 0]
        non_users = self.processed_df[self.processed_df[selected_substance] == 0]

        if len(users) == 0 or len(non_users) == 0:
            messagebox.showwarning("Uwaga", f"Brak wystarczających danych dla {selected_substance}")
            return

        # Enhanced figure with better styling
        fig, axes = plt.subplots(2, 4, figsize=(18, 12))
        fig.suptitle(f'🧠 Analiza Demograficzna: {selected_substance}\n'
                     f'Użytkownicy ({len(users)}) vs Nieużytkownicy ({len(non_users)})',
                     fontsize=16, fontweight='bold', y=0.95)
        axes = axes.ravel()

        stats_text = f"=== 🎯 ANALIZA DEMOGRAFICZNA - {selected_substance.upper()} ===\n\n"
        stats_text += f"👥 Użytkownicy: {len(users)} osób ({len(users) / len(self.processed_df) * 100:.1f}%)\n"
        stats_text += f"🚫 Nieużytkownicy: {len(non_users)} osób ({len(non_users) / len(self.processed_df) * 100:.1f}%)\n\n"
        stats_text += "🧠 RÓŻNICE W CECHACH OSOBOWOŚCI:\n" + "=" * 50 + "\n"

        for i, col in enumerate(self.personality_cols):
            if col in self.processed_df.columns and i < 8:
                non_user_data = non_users[col].dropna()
                user_data = users[col].dropna()

                # Statistical test
                if len(non_user_data) > 5 and len(user_data) > 5:
                    from scipy.stats import mannwhitneyu
                    statistic, p_value = mannwhitneyu(non_user_data, user_data, alternative='two-sided')
                    significance = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else "ns"
                else:
                    significance = "n/a"

                # Enhanced boxplot with custom colors
                box_data = [non_user_data, user_data]
                bp = axes[i].boxplot(box_data, labels=['Nie używa', 'Używa'],
                                     patch_artist=True, medianprops={'color': 'white', 'linewidth': 2})

                # Color based on risk level
                bp['boxes'][0].set_facecolor(self.colors['info'])
                bp['boxes'][0].set_alpha(0.7)
                bp['boxes'][1].set_facecolor(self.get_risk_color(user_data.median() - non_user_data.median()))
                bp['boxes'][1].set_alpha(0.8)

                # Enhanced styling
                axes[i].set_title(f'{col}\n({significance})', fontweight='bold', fontsize=11)
                axes[i].grid(True, alpha=0.3)
                axes[i].set_facecolor('#f8f9fa')

                # Add difference annotation
                diff = user_data.median() - non_user_data.median()
                axes[i].text(0.5, 0.95, f'Δ={diff:.2f}', transform=axes[i].transAxes,
                             ha='center', va='top', fontweight='bold',
                             bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

                # Add to statistics
                non_user_median = non_user_data.median()
                user_median = user_data.median()
                difference = user_median - non_user_median
                abs(difference) / non_user_data.std() if non_user_data.std() > 0 else 0

                # Interpretation
                ""
                if significance in ['***', '**', '*']:
                    if difference > 0.2:
                        interpretation = "📈 WYŻEJ"
                    elif difference < -0.2:
                        interpretation = "📉 NIŻEJ"
                    else:
                        interpretation = "≈ PODOBNIE"
                else:
                    interpretation = "≈ BRAK RÓŻNICY"

                stats_text += f"{col:<18} {interpretation:>12} | Δ={difference:>+6.3f} | p={p_value:.3f} {significance}\n"

        # Remove unused subplots
        if len(self.personality_cols) < 8:
            for i in range(len(self.personality_cols), 8):
                fig.delaxes(axes[i])

        plt.tight_layout()
        plt.subplots_adjust(top=0.90)

        # Add interpretation
        stats_text += f"\n💡 INTERPRETACJA:\n" + "=" * 30 + "\n"

        # Count significant differences
        sig_higher = sum(1 for line in stats_text.split('\n') if
                         '📈 WYŻEJ' in line and ('***' in line or '**' in line or '*' in line))
        sig_lower = sum(1 for line in stats_text.split('\n') if
                        '📉 NIŻEJ' in line and ('***' in line or '**' in line or '*' in line))

        if sig_higher > sig_lower:
            stats_text += f"🎯 Użytkownicy {selected_substance} mają WYŻSZE wartości w {sig_higher} cechach osobowości\n"
        elif sig_lower > sig_higher:
            stats_text += f"🎯 Użytkownicy {selected_substance} mają NIŻSZE wartości w {sig_lower} cechach osobowości\n"
        else:
            stats_text += f"🎯 Użytkownicy {selected_substance} mają MIESZANY profil osobowości\n"

        # Risk assessment
        if selected_substance in ['Heroin', 'Crack', 'Cocaine']:
            stats_text += "🚨 WYSOKIE RYZYKO: Substancja związana z problematycznymi wzorcami osobowości\n"
        elif selected_substance in ['LSD', 'Mushrooms', 'Cannabis']:
            stats_text += "🟡 UMIARKOWANE RYZYKO: Substancja związana z eksperymentowaniem\n"
        else:
            stats_text += "🟢 NISKIE RYZYKO: Substancja mainstream z szerokim profilem użytkowników\n"

        # Update text results
        self.result_text.delete(1.0, tk.END)
        self.result_text.insert(tk.END, stats_text)

        # Embed plot
        canvas = FigureCanvasTkAgg(fig, self.plot_frame)
        canvas.draw()
        canvas.get_tk_widget().grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        self.plot_frame.columnconfigure(0, weight=1)
        self.plot_frame.rowconfigure(0, weight=1)

    def perform_enhanced_clustering(self):
        """Enhanced clustering with detailed cluster profiling"""
        if self.processed_df is None:
            messagebox.showerror("Błąd", "Najpierw wczytaj dane!")
            return

        # Prepare data
        features = self.processed_df[self.personality_cols].dropna()

        # Standardization
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)

        # K-means clustering
        kmeans = KMeans(n_clusters=4, n_init=10, random_state=42)
        cluster_labels = kmeans.fit_predict(features_scaled)
        self.cluster_labels = cluster_labels

        # Add cluster labels to dataframe
        features_df = features.copy()
        features_df['Cluster'] = cluster_labels

        # Enhanced cluster analysis
        cluster_text = "🎯 === ANALIZA KLASTRÓW K-MEANS - 4 PROFILE OSOBOWOŚCI ===\n\n"
        cluster_text += f"📊 Liczba klastrów: 4\n"
        cluster_text += f"👥 Liczba obserwacji: {len(features_df)}\n\n"

        # Detailed cluster profiles
        for i in range(4):
            cluster_data = features_df[features_df['Cluster'] == i]
            cluster_size = len(cluster_data)
            cluster_pct = cluster_size / len(features_df) * 100

            profile = self.cluster_profiles[i]

            cluster_text += f"{profile['emoji']} KLASTER {i}: \"{profile['name']}\" (n={cluster_size}, {cluster_pct:.1f}%)\n"
            cluster_text += f"🎯 Ryzyko: {profile['risk']} | 📝 Opis: {profile['description']}\n"
            cluster_text += "─" * 70 + "\n"

            cluster_means = cluster_data[self.personality_cols].mean()
            for trait, mean_val in cluster_means.items():
                # Color-coded interpretation
                if mean_val > 0.5:
                    level = "🔴 BARDZO WYSOKIE"
                elif mean_val > 0.2:
                    level = "🟠 WYSOKIE"
                elif mean_val > -0.2:
                    level = "🟡 UMIARKOWANE"
                elif mean_val > -0.5:
                    level = "🔵 NISKIE"
                else:
                    level = "🟣 BARDZO NISKIE"

                cluster_text += f"  {trait:<18}: {mean_val:>6.3f} {level}\n"
            cluster_text += "\n"

        # Cluster comparison and insights
        cluster_text += "🔍 KLUCZOWE RÓŻNICE MIĘDZY KLASTRAMI:\n" + "=" * 50 + "\n"

        # Find distinguishing features for each cluster
        for i in range(4):
            cluster_data = features_df[features_df['Cluster'] == i]
            cluster_means = cluster_data[self.personality_cols].mean()

            # Find top distinguishing features
            sorted_features = sorted(cluster_means.items(), key=lambda x: abs(x[1]), reverse=True)
            top_features = sorted_features[:3]

            profile = self.cluster_profiles[i]
            cluster_text += f"\n{profile['emoji']} {profile['name']}:\n"

            for trait, value in top_features:
                direction = "⬆️ Wysokie" if value > 0 else "⬇️ Niskie"
                cluster_text += f"  • {direction} {trait} ({value:+.2f})\n"

        # Risk stratification
        cluster_text += f"\n🚨 STRATYFIKACJA RYZYKA:\n" + "=" * 30 + "\n"
        cluster_text += f"🔴 BARDZO WYSOKIE: Klaster 1 - Impulsywni w Kryzysie ({len(features_df[features_df['Cluster'] == 1])} osób)\n"
        cluster_text += f"🟠 UMIARKOWANE: Klaster 0,2 - Poszukiwacze + Izolowani ({len(features_df[features_df['Cluster'].isin([0, 2])])} osób)\n"
        cluster_text += f"🟢 NISKIE: Klaster 3 - Stabilni Konserwatyści ({len(features_df[features_df['Cluster'] == 3])} osób)\n"

        self.result_text.delete(1.0, tk.END)
        self.result_text.insert(tk.END, cluster_text)

        # Enhanced PCA visualization
        pca = PCA(n_components=2)
        features_pca = pca.fit_transform(features_scaled)

        for widget in self.plot_frame.winfo_children():
            widget.destroy()

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

        # PCA plot with enhanced styling
        for i in range(4):
            mask = cluster_labels == i
            profile = self.cluster_profiles[i]
            ax1.scatter(features_pca[mask, 0], features_pca[mask, 1],
                        c=profile['color'], label=f"{profile['emoji']} {profile['name']}",
                        alpha=0.7, s=50, edgecolors='white', linewidth=0.5)

        ax1.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} wariancji)', fontweight='bold')
        ax1.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} wariancji)', fontweight='bold')
        ax1.set_title('🎯 Wizualizacja Klastrów (PCA)', fontweight='bold', fontsize=14)
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax1.grid(True, alpha=0.3)
        ax1.set_facecolor('#f8f9fa')

        # Cluster size pie chart
        cluster_sizes = [len(features_df[features_df['Cluster'] == i]) for i in range(4)]
        cluster_names = [self.cluster_profiles[i]['name'] for i in range(4)]
        cluster_colors = [self.cluster_profiles[i]['color'] for i in range(4)]

        wedges, texts, autotexts = ax2.pie(cluster_sizes, labels=cluster_names, autopct='%1.1f%%',
                                           colors=cluster_colors, startangle=90,
                                           explode=(0.05, 0.05, 0.05, 0.05))

        ax2.set_title('📊 Rozkład Wielkości Klastrów', fontweight='bold', fontsize=14)

        # Enhance text
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontweight('bold')

        plt.tight_layout()

        # Embed plot
        canvas = FigureCanvasTkAgg(fig, self.plot_frame)
        canvas.draw()
        canvas.get_tk_widget().grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        self.plot_frame.columnconfigure(0, weight=1)
        self.plot_frame.rowconfigure(0, weight=1)

    def analyze_cluster_substance_patterns(self):
        """Analyze substance use patterns by cluster"""
        if self.cluster_labels is None:
            messagebox.showwarning("Uwaga", "Najpierw wykonaj analizę klastrów!")
            return

        # Prepare data
        cluster_substance_df = self.processed_df.copy()
        cluster_substance_df['Cluster'] = self.cluster_labels

        # Calculate usage rates for each cluster-substance combination
        results = []
        for cluster in range(4):
            cluster_data = cluster_substance_df[cluster_substance_df['Cluster'] == cluster]
            cluster_profile = self.cluster_profiles[cluster]

            for substance in self.substance_cols:
                if substance in cluster_substance_df.columns:
                    usage_rate = (cluster_data[substance] > 0).mean()
                    avg_intensity = cluster_data[substance].mean()
                    results.append({
                        'Cluster': cluster,
                        'Cluster_Name': cluster_profile['name'],
                        'Substance': substance,
                        'Usage_Rate': usage_rate,
                        'Avg_Intensity': avg_intensity
                    })

        results_df = pd.DataFrame(results)

        # Create enhanced analysis text
        analysis_text = "🧬 === PROFILE KLASTRÓW vs UŻYWANIE SUBSTANCJI ===\n\n"

        for cluster in range(4):
            cluster_data = results_df[results_df['Cluster'] == cluster]
            profile = self.cluster_profiles[cluster]

            analysis_text += f"{profile['emoji']} KLASTER {cluster}: {profile['name']}\n"
            analysis_text += f"🎯 Ryzyko: {profile['risk']} | 👥 Wielkość: {len(cluster_substance_df[cluster_substance_df['Cluster'] == cluster])} osób\n"
            analysis_text += "─" * 70 + "\n"

            # Top substances for this cluster
            top_substances = cluster_data.nlargest(8, 'Usage_Rate')

            analysis_text += "🔝 TOP UŻYWANE SUBSTANCJE:\n"
            for _, row in top_substances.iterrows():
                usage_pct = row['Usage_Rate'] * 100
                intensity = row['Avg_Intensity']

                # Risk color coding
                if usage_pct > 70:
                    risk_emoji = "🔴"
                elif usage_pct > 40:
                    risk_emoji = "🟠"
                elif usage_pct > 20:
                    risk_emoji = "🟡"
                else:
                    risk_emoji = "🟢"

                analysis_text += f"  {risk_emoji} {row['Substance']:<15} {usage_pct:>5.1f}% (średnia: {intensity:.2f})\n"

            # Cluster-specific insights
            if cluster == 0:  # Ekstrawertyczni Poszukiwacze
                analysis_text += "\n💡 WZORZEC: Social/Party drugs dominują\n"
                analysis_text += "🎉 Używanie społeczne, rekreacyjne, eksperymentowanie\n"
            elif cluster == 1:  # Impulsywni w Kryzysie
                analysis_text += "\n💡 WZORZEC: Hard drugs + self-medication\n"
                analysis_text += "🚨 Chaotyczne używanie, polydrug abuse, wysokie ryzyko\n"
            elif cluster == 2:  # Lękliwi Izolowani
                analysis_text += "\n💡 WZORZEC: Self-medication + comfort substances\n"
                analysis_text += "😔 Prywatne używanie, anxiety relief, habitual patterns\n"
            elif cluster == 3:  # Stabilni Konserwatyści
                analysis_text += "\n💡 WZORZEC: Mainstream tylko, minimal use\n"
                analysis_text += "🛡️ Społecznie akceptowane, kontrolowane, odpowiedzialne\n"

            analysis_text += "\n"

        # Cross-cluster comparisons
        analysis_text += "🔍 PORÓWNANIA MIĘDZY KLASTRAMI:\n" + "=" * 40 + "\n"

        # Find substances with the biggest cluster differences
        for substance in self.substance_cols[:10]:  # Top 10 substances
            if substance in results_df['Substance'].values:
                substance_data = results_df[results_df['Substance'] == substance]
                max_usage = substance_data['Usage_Rate'].max()
                min_usage = substance_data['Usage_Rate'].min()
                diff = max_usage - min_usage

                if diff > 0.3:  # Significant difference
                    max_cluster = substance_data.loc[substance_data['Usage_Rate'].idxmax(), 'Cluster_Name']
                    min_cluster = substance_data.loc[substance_data['Usage_Rate'].idxmin(), 'Cluster_Name']
                    analysis_text += f"🎯 {substance}: {max_cluster} ({max_usage:.1%}) >> {min_cluster} ({min_usage:.1%})\n"

        # Risk stratification summary
        analysis_text += f"\n🚨 PODSUMOWANIE RYZYKA:\n" + "=" * 25 + "\n"

        # Calculate average risk scores
        risk_scores = {}
        for cluster in range(4):
            cluster_data = results_df[results_df['Cluster'] == cluster]
            # Focus on high-risk substances
            high_risk_substances = ['Heroin', 'Crack', 'Cocaine', 'Amphetamines', 'Benzodiazepines']
            risk_usage = cluster_data[cluster_data['Substance'].isin(high_risk_substances)]['Usage_Rate'].mean()
            risk_scores[cluster] = risk_usage

            profile = self.cluster_profiles[cluster]
            analysis_text += f"{profile['emoji']} {profile['name']}: {risk_usage:.1%} hard drugs use\n"

        # Recommendations
        analysis_text += f"\n💡 REKOMENDACJE INTERWENCJI:\n" + "=" * 30 + "\n"
        analysis_text += "🔴 Klaster 1: Crisis intervention, dual diagnosis treatment\n"
        analysis_text += "🟠 Klaster 2: Anxiety treatment, social support building\n"
        analysis_text += "🟡 Klaster 0: Harm reduction, alternative activities\n"
        analysis_text += "🟢 Klaster 3: Reinforcement, peer leadership roles\n"

        self.result_text.delete(1.0, tk.END)
        self.result_text.insert(tk.END, analysis_text)

        # Create visualization
        for widget in self.plot_frame.winfo_children():
            widget.destroy()

        # Heatmap of cluster-substance patterns
        pivot_data = results_df.pivot(index='Cluster_Name', columns='Substance', values='Usage_Rate')

        fig, ax = plt.subplots(figsize=(16, 8))

        # Enhanced heatmap
        sns.heatmap(pivot_data, annot=True, fmt='.2f', cmap='Reds',
                    cbar_kws={'label': 'Odsetek używających'}, ax=ax,
                    linewidths=0.5, linecolor='white')

        ax.set_title('🧬 Profile Używania Substancji według Klastrów Osobowości',
                     fontsize=14, fontweight='bold', pad=20)
        ax.set_xlabel('Substancje', fontweight='bold')
        ax.set_ylabel('Klastry Osobowości', fontweight='bold')

        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()

        # Embed plot
        canvas = FigureCanvasTkAgg(fig, self.plot_frame)
        canvas.draw()
        canvas.get_tk_widget().grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

    # Continue with other methods...
    # [Include all the remaining methods from the original with enhanced styling and colors]

    def calculate_basic_stats(self):
        """Calculate basic statistics with enhanced formatting"""
        if self.df is None:
            messagebox.showerror("Błąd", "Najpierw wczytaj dane!")
            return

        stats_text = "📊 === STATYSTYKI OPISOWE UCI DATASET ===\n\n"

        # Enhanced personality statistics
        stats_text += "🧠 CECHY OSOBOWOŚCI (Big Five + Impulsywność + Sensation Seeking):\n"
        stats_text += "─" * 70 + "\n"
        personality_stats = self.df[self.personality_cols].describe()
        stats_text += personality_stats.to_string() + "\n\n"

        # Skewness and kurtosis with interpretation
        stats_text += "📈 SKOŚNOŚĆ I KURTOZA (ocena normalności rozkładów):\n"
        stats_text += "─" * 50 + "\n"
        for col in self.personality_cols:
            if col in self.df.columns:
                skewness = stats.skew(self.df[col].dropna())
                kurtosis = stats.kurtosis(self.df[col].dropna())

                # Interpretation
                skew_interp = "👍 Normalny" if abs(skewness) < 0.5 else "⚠️ Skośny" if abs(
                    skewness) < 1 else "❌ Bardzo skośny"
                kurt_interp = "👍 Normalny" if abs(kurtosis) < 0.5 else "⚠️ Odchylenie" if abs(
                    kurtosis) < 1 else "❌ Silne odchylenie"

                stats_text += f"{col:<18}: skośność={skewness:>6.3f} {skew_interp}, kurtoza={kurtosis:>6.3f} {kurt_interp}\n"

        # Enhanced substance statistics
        stats_text += "\n💊 STATYSTYKI UŻYWANIA SUBSTANCJI:\n"
        stats_text += "─" * 40 + "\n"
        stats_text += f"{'Substancja':<15} {'Nigdy':<6} {'Ostatnio':<8} {'Intensywnie':<12} {'Popularność'}\n"
        stats_text += "─" * 60 + "\n"

        for col in self.substance_cols:
            if col in self.df.columns:
                usage_stats = self.df[col].value_counts().sort_index()
                never_used = usage_stats.get(0, 0)
                recent_use = usage_stats.get(6, 0) + usage_stats.get(5, 0)  # Last week + day
                heavy_use = usage_stats.get(6, 0)  # Daily use
                total_users = (self.df[col] > 0).sum()
                popularity = total_users / len(self.df) * 100

                # Visual popularity indicator
                if popularity > 80:
                    pop_emoji = "🔥🔥🔥"
                elif popularity > 50:
                    pop_emoji = "🔥🔥"
                elif popularity > 20:
                    pop_emoji = "🔥"
                else:
                    pop_emoji = "❄️"

                stats_text += f"{col:<15} {never_used:<6} {recent_use:<8} {heavy_use:<12} {popularity:>5.1f}% {pop_emoji}\n"

        # Key insights
        stats_text += f"\n💡 KLUCZOWE SPOSTRZEŻENIA:\n"
        stats_text += "─" * 25 + "\n"

        # Most/least popular substances
        substance_popularity = {}
        for col in self.substance_cols:
            if col in self.df.columns:
                substance_popularity[col] = (self.df[col] > 0).sum() / len(self.df) * 100

        most_popular = max(substance_popularity, key=substance_popularity.get)
        least_popular = min(substance_popularity, key=substance_popularity.get)

        stats_text += f"🏆 Najpopularniejsza substancja: {most_popular} ({substance_popularity[most_popular]:.1f}%)\n"
        stats_text += f"🏅 Najrzadsza substancja: {least_popular} ({substance_popularity[least_popular]:.1f}%)\n"

        # Risk categorization
        legal_substances = ['Alcohol', 'Caffeine', 'Chocolate', 'Nicotine']
        illegal_usage = sum(substance_popularity[s] for s in substance_popularity if s not in legal_substances) / len(
            [s for s in substance_popularity if s not in legal_substances])

        stats_text += f"📊 Średnie używanie substancji nielegalnych: {illegal_usage:.1f}%\n"
        stats_text += f"📊 Średnie używanie substancji legalnych: {sum(substance_popularity[s] for s in legal_substances) / len(legal_substances):.1f}%\n"

        self.result_text.delete(1.0, tk.END)
        self.result_text.insert(tk.END, stats_text)

    def analyze_personality_correlations(self):
        """Enhanced personality correlation analysis"""
        if self.df is None:
            messagebox.showerror("Błąd", "Najpierw wczytaj dane!")
            return

        personality_data = self.df[self.personality_cols]
        correlation_matrix = personality_data.corr()

        corr_text = "🧠 === KORELACJE CECH OSOBOWOŚCI ===\n\n"
        corr_text += correlation_matrix.round(3).to_string() + "\n\n"

        # Enhanced strong correlations analysis
        corr_text += "🔥 NAJSILNIEJSZE KORELACJE (|r| > 0.3):\n"
        corr_text += "─" * 45 + "\n"

        strong_correlations = []
        for i in range(len(self.personality_cols)):
            for j in range(i + 1, len(self.personality_cols)):
                col1, col2 = self.personality_cols[i], self.personality_cols[j]
                if col1 in correlation_matrix.columns and col2 in correlation_matrix.columns:
                    corr_val = correlation_matrix.loc[col1, col2]
                    if abs(corr_val) > 0.3:
                        strong_correlations.append((col1, col2, corr_val))

        strong_correlations.sort(key=lambda x: abs(x[2]), reverse=True)

        for col1, col2, corr_val in strong_correlations:
            direction = "📈 Pozytywna" if corr_val > 0 else "📉 Negatywna"
            strength = "🔥🔥🔥" if abs(corr_val) > 0.6 else "🔥🔥" if abs(corr_val) > 0.4 else "🔥"
            corr_text += f"{col1} ↔ {col2}: {corr_val:+.3f} {direction} {strength}\n"

        self.result_text.delete(1.0, tk.END)
        self.result_text.insert(tk.END, corr_text)

        # Enhanced heatmap
        self.plot_correlation_heatmap(correlation_matrix, "🧠 Korelacje Cech Osobowości")

    def analyze_substance_correlations(self):
        """Enhanced substance correlation analysis"""
        if self.df is None:
            messagebox.showerror("Błąd", "Najpierw wczytaj dane!")
            return

        substance_data = self.df[self.substance_cols]
        correlation_matrix = substance_data.corr()

        corr_text = "💊 === KORELACJE UŻYWANIA SUBSTANCJI ===\n\n"

        # Find the strongest correlations and group them
        strong_correlations = []
        for i in range(len(self.substance_cols)):
            for j in range(i + 1, len(self.substance_cols)):
                col1, col2 = self.substance_cols[i], self.substance_cols[j]
                if col1 in correlation_matrix.columns and col2 in correlation_matrix.columns:
                    corr_val = correlation_matrix.loc[col1, col2]
                    if abs(corr_val) > 0.2:
                        strong_correlations.append((col1, col2, corr_val))

        strong_correlations.sort(key=lambda x: abs(x[2]), reverse=True)

        # Categorize correlations
        corr_text += "🎯 KLASTRY SUBSTANCJI (najsilniejsze korelacje):\n"
        corr_text += "─" * 50 + "\n"

        party_drugs = []
        psychedelics = []
        hard_drugs = []

        for col1, col2, corr_val in strong_correlations[:15]:
            if any(x in [col1, col2] for x in ['Ecstasy', 'Cocaine', 'Amphetamines']):
                party_drugs.append((col1, col2, corr_val))
            elif any(x in [col1, col2] for x in ['LSD', 'Mushrooms']):
                psychedelics.append((col1, col2, corr_val))
            elif any(x in [col1, col2] for x in ['Heroin', 'Crack', 'Methadone']):
                hard_drugs.append((col1, col2, corr_val))

        corr_text += "🎉 PARTY DRUGS CLUSTER:\n"
        for col1, col2, corr_val in party_drugs[:5]:
            corr_text += f"  • {col1} ↔ {col2}: {corr_val:.3f}\n"

        corr_text += "\n🌈 PSYCHEDELICS CLUSTER:\n"
        for col1, col2, corr_val in psychedelics[:5]:
            corr_text += f"  • {col1} ↔ {col2}: {corr_val:.3f}\n"

        corr_text += "\n🚨 HARD DRUGS CLUSTER:\n"
        for col1, col2, corr_val in hard_drugs[:5]:
            corr_text += f"  • {col1} ↔ {col2}: {corr_val:.3f}\n"

        self.result_text.delete(1.0, tk.END)
        self.result_text.insert(tk.END, corr_text)

        self.plot_correlation_heatmap(correlation_matrix, "💊 Korelacje Używania Substancji")

    def plot_correlation_heatmap(self, correlation_matrix, title):
        """Enhanced correlation heatmap"""
        for widget in self.plot_frame.winfo_children():
            widget.destroy()

        fig, ax = plt.subplots(figsize=(14, 12))

        # Enhanced heatmap with better colors
        mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
        sns.heatmap(correlation_matrix, annot=True, cmap='RdBu_r', center=0,
                    square=True, ax=ax, fmt='.2f', cbar_kws={'shrink': 0.8},
                    mask=mask, linewidths=0.5)

        ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()

        canvas = FigureCanvasTkAgg(fig, self.plot_frame)
        canvas.draw()
        canvas.get_tk_widget().grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        self.plot_frame.columnconfigure(0, weight=1)
        self.plot_frame.rowconfigure(0, weight=1)

    def apply_filter(self):
        """Enhanced filtering with detailed feedback"""
        if self.df is None:
            messagebox.showerror("Błąd", "Najpierw wczytaj dane!")
            return

        filter_option = self.filter_var.get()

        try:
            if filter_option == 'Wszystkie dane':
                self.processed_df = self.df.copy()
            elif filter_option == 'Tylko mężczyźni':
                if 'Gender_Category' in self.df.columns:
                    self.processed_df = self.df[self.df['Gender_Category'] == 'Male']
                else:
                    messagebox.showerror("Błąd", "Brak danych demograficznych!")
                    return
            elif filter_option == 'Tylko kobiety':
                if 'Gender_Category' in self.df.columns:
                    self.processed_df = self.df[self.df['Gender_Category'] == 'Female']
                else:
                    messagebox.showerror("Błąd", "Brak danych demograficznych!")
                    return
            elif filter_option == 'Wiek 18-24':
                if 'Age_Category' in self.df.columns:
                    self.processed_df = self.df[self.df['Age_Category'] == '18-24']
                else:
                    messagebox.showerror("Błąd", "Brak danych demograficznych!")
                    return
            elif filter_option == 'Wiek 25-34':
                if 'Age_Category' in self.df.columns:
                    self.processed_df = self.df[self.df['Age_Category'] == '25-34']
                else:
                    messagebox.showerror("Błąd", "Brak danych demograficznych!")
                    return
            elif filter_option == 'Wiek 35-44':
                if 'Age_Category' in self.df.columns:
                    self.processed_df = self.df[self.df['Age_Category'] == '35-44']
                else:
                    messagebox.showerror("Błąd", "Brak danych demograficznych!")
                    return
            elif filter_option == 'Używający Cannabis':
                self.processed_df = self.df[self.df['Cannabis'] > 0]
            elif filter_option == 'Używający Alcohol':
                self.processed_df = self.df[self.df['Alcohol'] > 0]
            elif filter_option == 'Nieużywający narkotyków':
                illegal_substances = [col for col in self.substance_cols
                                      if col not in ['Alcohol', 'Caffeine', 'Chocolate', 'Nicotine']]
                mask = (self.df[illegal_substances] == 0).all(axis=1)
                self.processed_df = self.df[mask]
            elif filter_option == 'Wysokie wykształcenie':
                if 'Education_Category' in self.df.columns:
                    high_education = ['University degree', 'Masters degree', 'Doctorate degree']
                    self.processed_df = self.df[self.df['Education_Category'].isin(high_education)]
                else:
                    messagebox.showerror("Błąd", "Brak danych demograficznych!")
                    return
            elif filter_option == 'UK/USA/Canada':
                if 'Country_Category' in self.df.columns:
                    countries = ['UK', 'USA', 'Canada']
                    self.processed_df = self.df[self.df['Country_Category'].isin(countries)]
                else:
                    messagebox.showerror("Błąd", "Brak danych demograficznych!")
                    return
            else:
                self.processed_df = self.df.copy()

            if len(self.processed_df) == 0:
                messagebox.showwarning("Uwaga", f"Filtr '{filter_option}' nie zwrócił żadnych danych!")
                self.processed_df = self.df.copy()
                return

            self.update_filter_info(filter_option)

        except Exception as e:
            messagebox.showerror("Błąd", f"Problem z filtrowaniem: {str(e)}")
            self.processed_df = self.df.copy()

    def update_filter_info(self, filter_option):
        """Enhanced filter information display"""
        filter_text = f"""
🎯 === ZASTOSOWANO FILTR: {filter_option} ===

📊 STATYSTYKI FILTROWANIA:
├── Oryginalny rozmiar: {len(self.df)} rekordów
├── Po filtrowaniu: {len(self.processed_df)} rekordów  
├── Odsetek zachowany: {len(self.processed_df) / len(self.df) * 100:.1f}%
└── Utracono: {len(self.df) - len(self.processed_df)} rekordów

"""

        # Demographics in filtered group
        if 'Age_Category' in self.processed_df.columns:
            age_dist = self.processed_df['Age_Category'].value_counts()
            filter_text += f"👶 Rozkład wieku w filtrowanej grupie:\n{age_dist.to_string()}\n\n"

        if 'Gender_Category' in self.processed_df.columns:
            gender_dist = self.processed_df['Gender_Category'].value_counts()
            filter_text += f"⚧️ Rozkład płci w filtrowanej grupie:\n{gender_dist.to_string()}\n\n"

        # Top substances in filtered group
        filter_text += "🔝 TOP 5 UŻYWANYCH SUBSTANCJI W FILTROWANEJ GRUPIE:\n"
        usage_stats = {}
        for col in self.substance_cols:
            if col in self.processed_df.columns:
                usage_pct = (self.processed_df[col] > 0).sum() / len(self.processed_df) * 100
                usage_stats[col] = usage_pct

        sorted_usage = sorted(usage_stats.items(), key=lambda x: x[1], reverse=True)[:5]
        for i, (substance, pct) in enumerate(sorted_usage, 1):
            emoji = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else f"{i}."
            filter_text += f"{emoji} {substance:<15} {pct:>5.1f}%\n"

        self.result_text.delete(1.0, tk.END)
        self.result_text.insert(tk.END, filter_text)

    def perform_classification(self):
        """Enhanced Random Forest classification"""
        if self.processed_df is None:
            messagebox.showerror("Błąd", "Najpierw wczytaj dane!")
            return

        selected_substance = self.classification_substance_var.get()
        if not selected_substance or selected_substance not in self.processed_df.columns:
            messagebox.showerror("Błąd", "Wybierz prawidłową substancję!")
            return

        features = self.processed_df[self.personality_cols].dropna()
        target = (self.processed_df.loc[features.index, selected_substance] > 0).astype(int)

        if target.sum() < 10 or (len(target) - target.sum()) < 10:
            messagebox.showwarning("Uwaga",
                                   f"Za mało danych dla {selected_substance}. Potrzeba minimum 10 osób w każdej grupie.")
            return

        X_train, X_test, y_train, y_test = train_test_split(
            features, target, test_size=0.3, random_state=42, stratify=target
        )

        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf.fit(X_train, y_train)

        y_pred = rf.predict(X_test)
        y_pred_proba = rf.predict_proba(X_test)[:, 1]
        accuracy = accuracy_score(y_test, y_pred)

        try:
            auc_score = roc_auc_score(y_test, y_pred_proba)
        except:
            auc_score = 0.5

        conf_matrix = confusion_matrix(y_test, y_pred)
        baseline_accuracy = max(target.mean(), 1 - target.mean())
        improvement = accuracy - baseline_accuracy

        feature_importance = pd.DataFrame({
            'Feature': self.personality_cols,
            'Importance': rf.feature_importances_
        }).sort_values('Importance', ascending=False)

        # Enhanced classification report
        classification_text = f"""
🌲 === KLASYFIKACJA RANDOM FOREST - {selected_substance.upper()} ===

🎯 ZADANIE: Przewidywanie używania {selected_substance} na podstawie cech osobowości

📊 PODSTAWOWE INFORMACJE:
├── Całkowita próba: {len(features)} osób
├── 🟢 Używa substancji: {target.sum()} osób ({target.mean() * 100:.1f}%)  
├── 🔴 Nie używa: {len(target) - target.sum()} osób ({(1 - target.mean()) * 100:.1f}%)
├── 📚 Zbiór treningowy: {len(X_train)} obserwacji
└── 🧪 Zbiór testowy: {len(X_test)} obserwacji

🏆 WYNIKI MODELU:
├── 🎯 Dokładność (Accuracy): {accuracy:.3f} ({accuracy * 100:.1f}%)
├── 📈 AUC-ROC: {auc_score:.3f}
├── 📊 Baseline (najczęstsza klasa): {baseline_accuracy:.3f} ({baseline_accuracy * 100:.1f}%)
└── ⬆️ Poprawa nad baseline: {improvement:+.3f} ({improvement * 100:+.1f} pkt proc.)

🎭 MACIERZ KONFUZJI:
                 Przewidywane
Rzeczywiste    Nie używa  Używa
Nie używa         {conf_matrix[0, 0]:>3}      {conf_matrix[0, 1]:>3}
Używa             {conf_matrix[1, 0]:>3}      {conf_matrix[1, 1]:>3}

🏆 RANKING WAŻNOŚCI CECH:
{feature_importance.to_string(index=False)}
"""

        # Performance assessment
        if accuracy > baseline_accuracy + 0.1:
            performance = "🟢 EXCELLENT"
        elif accuracy > baseline_accuracy + 0.05:
            performance = "🟡 GOOD"
        else:
            performance = "🔴 POOR"

        classification_text += f"\n📋 OCENA WYDAJNOŚCI: {performance}\n"

        self.result_text.delete(1.0, tk.END)
        self.result_text.insert(tk.END, classification_text)

        # Enhanced visualization
        for widget in self.plot_frame.winfo_children():
            widget.destroy()

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        # Feature importance plot
        bars = ax1.barh(feature_importance['Feature'], feature_importance['Importance'],
                        color=self.colors['cluster_colors'], alpha=0.8)
        ax1.set_xlabel('Ważność cechy', fontweight='bold')
        ax1.set_title(f'🌲 Ważność Cech - {selected_substance}', fontweight='bold')
        ax1.grid(True, alpha=0.3, axis='x')

        # Performance comparison
        categories = ['Baseline', 'Random Forest']
        scores = [baseline_accuracy, accuracy]
        colors = [self.colors['danger'], self.colors['success']]

        bars2 = ax2.bar(categories, scores, color=colors, alpha=0.8)
        ax2.set_ylabel('Dokładność', fontweight='bold')
        ax2.set_title('📊 Porównanie Wydajności', fontweight='bold')
        ax2.set_ylim(0, 1)
        ax2.grid(True, alpha=0.3, axis='y')

        for bar, score in zip(bars2, scores):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width() / 2., height + 0.01,
                     f'{score:.1%}', ha='center', va='bottom', fontweight='bold')

        plt.tight_layout()

        canvas = FigureCanvasTkAgg(fig, self.plot_frame)
        canvas.draw()
        canvas.get_tk_widget().grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

    def create_risk_assessment(self):
        """Create interactive risk assessment tool"""
        if self.processed_df is None:
            messagebox.showerror("Błąd", "Najpierw wczytaj dane!")
            return

        # Calculate risk scores for each person based on personality
        risk_scores = {}
        for substance in ['Cannabis', 'Cocaine', 'Heroin', 'LSD']:
            if substance in self.processed_df.columns:
                # Simple risk model based on key predictors
                if substance == 'Cannabis':
                    risk_scores[substance] = (
                            0.3 * self.processed_df['SensationSeeking'] +
                            0.25 * self.processed_df['Impulsiveness'] +
                            0.2 * self.processed_df['Openness'] -
                            0.15 * self.processed_df['Conscientiousness']
                    )
                elif substance == 'Cocaine':
                    risk_scores[substance] = (
                            0.35 * self.processed_df['Impulsiveness'] +
                            0.3 * self.processed_df['SensationSeeking'] +
                            0.2 * self.processed_df['Neuroticism'] -
                            0.2 * self.processed_df['Conscientiousness']
                    )
                elif substance == 'Heroin':
                    risk_scores[substance] = (
                            0.4 * self.processed_df['Impulsiveness'] +
                            0.3 * self.processed_df['Neuroticism'] +
                            0.2 * self.processed_df['SensationSeeking'] -
                            0.25 * self.processed_df['Conscientiousness']
                    )
                elif substance == 'LSD':
                    risk_scores[substance] = (
                            0.4 * self.processed_df['Openness'] +
                            0.3 * self.processed_df['SensationSeeking'] +
                            0.15 * self.processed_df['Extraversion'] -
                            0.1 * self.processed_df['Neuroticism']
                    )

        # Create risk assessment report
        risk_text = "🚨 === TOOL OCENY RYZYKA UŻYWANIA SUBSTANCJI ===\n\n"
        risk_text += "📊 Model oparty na analizie cech osobowości z UCI Dataset\n"
        risk_text += "⚠️ Tylko do celów edukacyjnych i badawczych!\n\n"

        risk_text += "🎯 PERCENTYLE RYZYKA W POPULACJI:\n"
        risk_text += "─" * 50 + "\n"

        for substance, scores in risk_scores.items():
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

                risk_text += f"  {perc}. percentyl: {value:>6.2f} ({risk_level})\n"

        # Population risk distribution
        risk_text += f"\n📊 DYSTRYBUCJA RYZYKA W POPULACJI:\n"
        risk_text += "─" * 40 + "\n"

        for substance, scores in risk_scores.items():
            high_risk = (scores > np.percentile(scores, 75)).sum()
            moderate_risk = ((scores > np.percentile(scores, 25)) &
                             (scores <= np.percentile(scores, 75))).sum()
            low_risk = (scores <= np.percentile(scores, 25)).sum()

            risk_text += f"{substance}:\n"
            risk_text += f"  🔴 Wysokie ryzyko: {high_risk} osób ({high_risk / len(scores) * 100:.1f}%)\n"
            risk_text += f"  🟡 Umiarkowane: {moderate_risk} osób ({moderate_risk / len(scores) * 100:.1f}%)\n"
            risk_text += f"  🟢 Niskie ryzyko: {low_risk} osób ({low_risk / len(scores) * 100:.1f}%)\n\n"

        # Recommendations
        risk_text += "💡 REKOMENDACJE ZASTOSOWANIA:\n"
        risk_text += "─" * 35 + "\n"
        risk_text += "🎯 Screening populacyjny - identyfikacja grup wysokiego ryzyka\n"
        risk_text += "🏥 Planowanie interwencji - dopasowanie do poziomu ryzyka\n"
        risk_text += "📚 Badania naukowe - stratyfikacja próby badawczej\n"
        risk_text += "🎓 Edukacja - demonstracja czynników ryzyka\n\n"

        risk_text += "⚠️ OGRANICZENIA:\n"
        risk_text += "• Model uproszczony - rzeczywiste ryzyko zależy od wielu czynników\n"
        risk_text += "• Nie uwzględnia czynników środowiskowych i społecznych\n"
        risk_text += "• Oparte na self-report data - możliwe bias\n"
        risk_text += "• Nie zastępuje profesjonalnej oceny klinicznej\n"

        self.result_text.delete(1.0, tk.END)
        self.result_text.insert(tk.END, risk_text)

    def standardize_features(self):
        """Enhanced standardization with detailed output"""
        if self.processed_df is None:
            messagebox.showerror("Błąd", "Najpierw wczytaj dane!")
            return

        scaler = StandardScaler()
        personality_data = self.processed_df[self.personality_cols].copy()
        standardized_data = scaler.fit_transform(personality_data)

        for i, col in enumerate(self.personality_cols):
            self.processed_df[f'{col}_std'] = standardized_data[:, i]

        std_text = f"""
📏 === STANDARYZACJA CECH OSOBOWOŚCI ===

✅ Cechy zostały wystandaryzowane (Z-score transformation: μ=0, σ=1)

📊 PORÓWNANIE PRZED I PO STANDARYZACJI:

PRZED standaryzacją:
{personality_data.describe().round(3).to_string()}

PO standaryzacji:
{pd.DataFrame(standardized_data, columns=self.personality_cols).describe().round(3).to_string()}

🎯 KORZYŚCI STANDARYZACJI:
• Wszystkie cechy na tej samej skali
• Lepsze działanie algorytmów ML
• Łatwiejsza interpretacja (jednostki = odchylenia standardowe)
• Eliminacja bias związanych z różnymi zakresami wartości

✅ Utworzono nowe kolumny z sufiksem '_std'
"""

        self.result_text.delete(1.0, tk.END)
        self.result_text.insert(tk.END, std_text)

    def handle_missing_values(self):
        """Enhanced missing values handling"""
        if self.processed_df is None:
            messagebox.showerror("Błąd", "Najpierw wczytaj dane!")
            return

        missing_before = self.processed_df.isnull().sum()

        if missing_before[self.personality_cols].sum() > 0:
            imputer = KNNImputer(n_neighbors=5)
            personality_data = self.processed_df[self.personality_cols]
            imputed_data = imputer.fit_transform(personality_data)

            for i, col in enumerate(self.personality_cols):
                self.processed_df[col] = imputed_data[:, i]

        duplicates_before = self.processed_df.duplicated().sum()
        self.processed_df = self.processed_df.drop_duplicates()
        duplicates_removed = duplicates_before - self.processed_df.duplicated().sum()

        missing_after = self.processed_df.isnull().sum()

        missing_text = f"""
🔧 === OBSŁUGA BRAKUJĄCYCH WARTOŚCI I DUPLIKATÓW ===

📊 BRAKUJĄCE WARTOŚCI:
{missing_before[missing_before > 0].to_string() if missing_before.sum() > 0 else "✅ Brak brakujących wartości w danych"}

🔧 ZASTOSOWANE METODY:
• KNN Imputation (k=5) dla cech osobowości
• Usunięcie duplikatów

✅ WYNIKI:
• Uzupełniono: {missing_before.sum() - missing_after.sum()} brakujących wartości
• Usunięto: {duplicates_removed} duplikatów
• Finalny rozmiar: {len(self.processed_df)} rekordów

💡 KNN IMPUTATION:
Metoda szuka 5 najbardziej podobnych osób i uśrednia ich wartości.
Zachowuje naturalne korelacje między cechami osobowości.
"""

        self.result_text.delete(1.0, tk.END)
        self.result_text.insert(tk.END, missing_text)

    def binary_encode(self):
        """Enhanced binary encoding"""
        if self.processed_df is None:
            messagebox.showerror("Błąd", "Najpierw wczytaj dane!")
            return

        for col in self.substance_cols:
            if col in self.processed_df.columns:
                self.processed_df[f'{col}_binary'] = (self.processed_df[col] > 0).astype(int)

        binary_text = "🔀 === KODOWANIE BINARNE UŻYWANIA SUBSTANCJI ===\n\n"
        binary_text += "✅ Utworzono binarne wskaźniki (0=nigdy, 1=używał):\n\n"

        usage_stats = []
        for col in self.substance_cols:
            if col in self.processed_df.columns:
                binary_col = f'{col}_binary'
                if binary_col in self.processed_df.columns:
                    usage_count = self.processed_df[binary_col].sum()
                    usage_percent = usage_count / len(self.processed_df) * 100
                    usage_stats.append((col, usage_count, usage_percent))

        # Sort by usage percentage
        usage_stats.sort(key=lambda x: x[2], reverse=True)

        binary_text += f"{'Substancja':<15} {'Użytkownicy':<12} {'Procent':<8} {'Wizualizacja'}\n"
        binary_text += "─" * 60 + "\n"

        for substance, count, percent in usage_stats:
            bar_length = int(percent / 5)  # Scale for visualization
            bar = "█" * bar_length + "░" * (20 - bar_length)
            binary_text += f"{substance:<15} {count:<12} {percent:>6.1f}%  {bar}\n"

        binary_text += f"\n💡 ZASTOSOWANIA KODOWANIA BINARNEGO:\n"
        binary_text += "• Uproszczone analizy (używa/nie używa)\n"
        binary_text += "• Modele klasyfikacji binarnej\n"
        binary_text += "• Analiza częstości występowania\n"
        binary_text += "• Reguły asocjacyjne\n"

        self.result_text.delete(1.0, tk.END)
        self.result_text.insert(tk.END, binary_text)

    def plot_demographic_histograms(self):
        """Enhanced demographic histograms with overlapping distributions"""
        if self.processed_df is None:
            messagebox.showerror("Błąd", "Najpierw wczytaj dane!")
            return

        selected_substance = self.substance_var.get()
        if not selected_substance or selected_substance not in self.processed_df.columns:
            messagebox.showerror("Błąd", "Wybierz prawidłową substancję!")
            return

        for widget in self.plot_frame.winfo_children():
            widget.destroy()

        users = self.processed_df[self.processed_df[selected_substance] > 0]
        non_users = self.processed_df[self.processed_df[selected_substance] == 0]

        if len(users) == 0 or len(non_users) == 0:
            messagebox.showwarning("Uwaga", f"Brak wystarczających danych dla {selected_substance}")
            return

        fig, axes = plt.subplots(2, 4, figsize=(18, 12))
        fig.suptitle(f'📊 Rozkłady Cech Osobowości: {selected_substance}\n'
                     f'🔵 Nieużytkownicy (n={len(non_users)}) vs 🔴 Użytkownicy (n={len(users)})',
                     fontsize=16, fontweight='bold', y=0.95)
        axes = axes.ravel()

        analysis_text = f"📊 === ANALIZA ROZKŁADÓW - {selected_substance.upper()} ===\n\n"

        for i, col in enumerate(self.personality_cols):
            if col in self.processed_df.columns and i < 8:
                non_user_data = non_users[col].dropna()
                user_data = users[col].dropna()

                # Common range for x-axis
                all_data = pd.concat([non_user_data, user_data])
                x_min, x_max = all_data.min() - 0.5, all_data.max() + 0.5

                # Enhanced overlapping histograms
                axes[i].hist(non_user_data, bins=25, alpha=0.6, color=self.colors['info'],
                             label=f'Nie używa (n={len(non_user_data)})', density=True,
                             range=(x_min, x_max), edgecolor='white', linewidth=0.5)
                axes[i].hist(user_data, bins=25, alpha=0.7, color=self.colors['danger'],
                             label=f'Używa (n={len(user_data)})', density=True,
                             range=(x_min, x_max), edgecolor='white', linewidth=0.5)

                # Add median lines
                axes[i].axvline(non_user_data.median(), color=self.colors['info'],
                                linestyle='--', linewidth=2, alpha=0.8)
                axes[i].axvline(user_data.median(), color=self.colors['danger'],
                                linestyle='--', linewidth=2, alpha=0.8)

                axes[i].set_title(col, fontweight='bold', fontsize=12)
                axes[i].set_xlabel('Wartość cechy')
                axes[i].set_ylabel('Gęstość')
                axes[i].legend(fontsize=8)
                axes[i].grid(True, alpha=0.3)
                axes[i].set_facecolor('#f8f9fa')

                # Calculate overlap and separation
                overlap = min(user_data.max(), non_user_data.max()) - max(user_data.min(), non_user_data.min())
                total_range = max(user_data.max(), non_user_data.max()) - min(user_data.min(), non_user_data.min())
                overlap_pct = (overlap / total_range * 100) if total_range > 0 else 0

                median_diff = user_data.median() - non_user_data.median()

                if overlap_pct > 80:
                    separation = "🟢 Duże nakładanie"
                elif overlap_pct > 60:
                    separation = "🟡 Umiarkowane nakładanie"
                else:
                    separation = "🔴 Wyraźne rozdzielenie"

                analysis_text += f"{col:<18}: Δ mediana = {median_diff:+.3f}, nakładanie = {overlap_pct:.0f}% ({separation})\n"

        if len(self.personality_cols) < 8:
            for i in range(len(self.personality_cols), 8):
                fig.delaxes(axes[i])

        plt.tight_layout()
        plt.subplots_adjust(top=0.90)

        # Add interpretation
        analysis_text += f"\n💡 INTERPRETACJA ROZKŁADÓW:\n"
        analysis_text += "─" * 30 + "\n"
        analysis_text += "🔴 Wyraźne rozdzielenie (nakładanie <60%): Silny predyktor używania\n"
        analysis_text += "🟡 Umiarkowane nakładanie (60-80%): Umiarkowany predyktor\n"
        analysis_text += "🟢 Duże nakładanie (>80%): Słaby predyktor - podobne rozkłady\n"

        self.result_text.delete(1.0, tk.END)
        self.result_text.insert(tk.END, analysis_text)

        canvas = FigureCanvasTkAgg(fig, self.plot_frame)
        canvas.draw()
        canvas.get_tk_widget().grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        self.plot_frame.columnconfigure(0, weight=1)
        self.plot_frame.rowconfigure(0, weight=1)

    def plot_all_substances_comparison(self):
        """Enhanced heatmap comparison of all substances"""
        if self.processed_df is None:
            messagebox.showerror("Błąd", "Najpierw wczytaj dane!")
            return

        for widget in self.plot_frame.winfo_children():
            widget.destroy()

        substance_means = {}
        stats_text = "🌡️ === PORÓWNANIE WSZYSTKICH SUBSTANCJI ===\n\n"
        stats_text += "Średnie wartości cech osobowości dla użytkowników każdej substancji:\n\n"

        for substance in self.substance_cols:
            if substance in self.processed_df.columns:
                users = self.processed_df[self.processed_df[substance] > 0]
                if len(users) >= 10:
                    means = []
                    for trait in self.personality_cols:
                        if trait in self.processed_df.columns:
                            mean_val = users[trait].mean()
                            means.append(mean_val)
                    substance_means[substance] = means

                    stats_text += f"💊 {substance} (n={len(users)}):\n"
                    for trait, mean_val in zip(self.personality_cols, means):
                        level = "🔴" if mean_val > 0.3 else "🟠" if mean_val > 0.1 else "🟡" if mean_val > -0.1 else "🔵" if mean_val > -0.3 else "🟣"
                        stats_text += f"  {trait}: {mean_val:+.3f} {level}\n"
                    stats_text += "\n"

        if not substance_means:
            messagebox.showwarning("Uwaga", "Brak wystarczających danych do porównania")
            return

        # Create enhanced heatmap
        df_heatmap = pd.DataFrame(substance_means, index=self.personality_cols).T

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 12))

        # Main heatmap
        sns.heatmap(df_heatmap, annot=True, fmt='.2f', cmap='RdBu_r', center=0,
                    cbar_kws={'label': 'Średnia znormalizowana'}, ax=ax1,
                    linewidths=0.5, square=False)

        ax1.set_title('🌡️ Profile Cech Osobowości dla Użytkowników Różnych Substancji',
                      fontsize=14, fontweight='bold')
        ax1.set_xlabel('Cechy Osobowości', fontweight='bold')
        ax1.set_ylabel('Substancje', fontweight='bold')

        # Clustering dendrogram
        from scipy.cluster.hierarchy import dendrogram, linkage

        # Calculate linkage for substances based on personality profiles
        linkage_matrix = linkage(df_heatmap.values, method='ward')

        dendrogram(linkage_matrix, labels=df_heatmap.index, ax=ax2,
                   leaf_rotation=45, leaf_font_size=10)
        ax2.set_title('🌳 Dendrogram Klastrów Substancji (na podstawie profili osobowości)',
                      fontweight='bold')
        ax2.set_xlabel('Substancje')
        ax2.set_ylabel('Odległość')

        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()

        # Identify clusters
        stats_text += "🎯 IDENTYFIKACJA KLASTRÓW:\n"
        stats_text += "─" * 30 + "\n"

        # Find substances with similar profiles (simplified clustering)
        from sklearn.cluster import KMeans
        if len(df_heatmap) >= 4:
            kmeans = KMeans(n_clusters=4, random_state=42)
            cluster_labels = kmeans.fit_predict(df_heatmap.values)

            for cluster_id in range(4):
                cluster_substances = df_heatmap.index[cluster_labels == cluster_id].tolist()
                if cluster_substances:
                    stats_text += f"🔗 Klaster {cluster_id + 1}: {', '.join(cluster_substances)}\n"

        self.result_text.delete(1.0, tk.END)
        self.result_text.insert(tk.END, stats_text)

        canvas = FigureCanvasTkAgg(fig, self.plot_frame)
        canvas.draw()
        canvas.get_tk_widget().grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        self.plot_frame.columnconfigure(0, weight=1)
        self.plot_frame.rowconfigure(0, weight=1)

    def plot_personality_distributions(self):
        """Enhanced personality distributions with statistical insights"""
        if self.processed_df is None:
            messagebox.showerror("Błąd", "Najpierw wczytaj dane!")
            return

        for widget in self.plot_frame.winfo_children():
            widget.destroy()

        fig, axes = plt.subplots(3, 3, figsize=(16, 14))
        fig.suptitle('🧠 Rozkłady Cech Osobowości w Populacji UCI',
                     fontsize=16, fontweight='bold', y=0.95)
        axes = axes.ravel()

        dist_text = "🧠 === ANALIZA ROZKŁADÓW CECH OSOBOWOŚCI ===\n\n"

        for i, col in enumerate(self.personality_cols):
            if col in self.processed_df.columns and i < 9:
                data = self.processed_df[col].dropna()

                # Enhanced histogram with statistics
                n, bins, patches = axes[i].hist(data, bins=40, alpha=0.7,
                                                color=self.colors['cluster_colors'][i % 4],
                                                edgecolor='white', linewidth=0.5)

                # Add normal distribution overlay
                mu, sigma = data.mean(), data.std()
                x = np.linspace(data.min(), data.max(), 100)
                y = ((1 / (np.sqrt(2 * np.pi) * sigma)) *
                     np.exp(-0.5 * ((x - mu) / sigma) ** 2)) * len(data) * (bins[1] - bins[0])
                axes[i].plot(x, y, 'r--', alpha=0.8, linewidth=2, label='Rozkład normalny')

                # Add mean and median lines
                axes[i].axvline(mu, color='red', linestyle='-', linewidth=2, alpha=0.8, label=f'Średnia: {mu:.2f}')
                axes[i].axvline(data.median(), color='orange', linestyle='--', linewidth=2, alpha=0.8,
                                label=f'Mediana: {data.median():.2f}')

                axes[i].set_title(f'{col}\n(μ={mu:.2f}, σ={sigma:.2f})', fontweight='bold')
                axes[i].set_xlabel('Wartość')
                axes[i].set_ylabel('Częstość')
                axes[i].legend(fontsize=8)
                axes[i].grid(True, alpha=0.3)
                axes[i].set_facecolor('#f8f9fa')

                # Statistical analysis
                skewness = stats.skew(data)
                kurtosis = stats.kurtosis(data)

                # Normality test
                _, p_value = stats.shapiro(data[:5000] if len(data) > 5000 else data)  # Shapiro-Wilk (limited sample)

                normality = "🟢 Normalny" if p_value > 0.05 else "🔴 Nie-normalny"
                skew_desc = "🔴 Silnie skośny" if abs(skewness) > 1 else "🟡 Umiarkowanie skośny" if abs(
                    skewness) > 0.5 else "🟢 Symetryczny"
                kurt_desc = "🔴 Bardzo spłaszczony" if kurtosis < -1 else "🔴 Bardzo wysmukły" if kurtosis > 1 else "🟢 Normalny"

                dist_text += f"{col}:\n"
                dist_text += f"  📊 Średnia: {mu:.3f}, Mediana: {data.median():.3f}, Odchyl. std: {sigma:.3f}\n"
                dist_text += f"  📈 Skośność: {skewness:.3f} ({skew_desc})\n"
                dist_text += f"  📈 Kurtoza: {kurtosis:.3f} ({kurt_desc})\n"
                dist_text += f"  🧪 Normalność: p={p_value:.3f} ({normality})\n\n"

        # Remove unused subplots
        if len(self.personality_cols) < 9:
            for i in range(len(self.personality_cols), 9):
                fig.delaxes(axes[i])

        plt.tight_layout()
        plt.subplots_adjust(top=0.90)

        # Add interpretations
        dist_text += "💡 INTERPRETACJA ROZKŁADÓW:\n"
        dist_text += "─" * 30 + "\n"
        dist_text += "🟢 Większość cech ma rozkłady zbliżone do normalnych (jak oczekiwano)\n"
        dist_text += "📊 Dane zostały znormalizowane przez autorów UCI (μ≈0, σ≈1)\n"
        dist_text += "🎯 Skośność i kurtoza wskazują na naturalne variations w populacji\n"
        dist_text += "⚠️ Odstępstwa od normalności mogą wpływać na wybór testów statystycznych\n"

        self.result_text.delete(1.0, tk.END)
        self.result_text.insert(tk.END, dist_text)

        canvas = FigureCanvasTkAgg(fig, self.plot_frame)
        canvas.draw()
        canvas.get_tk_widget().grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        self.plot_frame.columnconfigure(0, weight=1)
        self.plot_frame.rowconfigure(0, weight=1)

    def plot_substance_usage(self):
        """Enhanced substance usage visualization with insights"""
        if self.processed_df is None:
            messagebox.showerror("Błąd", "Najpierw wczytaj dane!")
            return

        for widget in self.plot_frame.winfo_children():
            widget.destroy()

        # Calculate usage statistics
        usage_data = []
        for col in self.substance_cols:
            if col in self.processed_df.columns:
                total_users = (self.processed_df[col] > 0).sum()
                heavy_users = (self.processed_df[col] >= 5).sum()  # Recent use (last week/day)
                usage_rate = total_users / len(self.processed_df) * 100
                heavy_rate = heavy_users / len(self.processed_df) * 100
                avg_intensity = self.processed_df[col].mean()

                # Categorize substances
                if col in ['Alcohol', 'Caffeine', 'Chocolate', 'Nicotine']:
                    category = 'Legal'
                    cat_color = self.colors['success']
                elif col in ['Cannabis', 'LSD', 'Mushrooms']:
                    category = 'Soft Illegal'
                    cat_color = self.colors['warning']
                elif col in ['Cocaine', 'Ecstasy', 'Amphetamines']:
                    category = 'Stimulants'
                    cat_color = self.colors['secondary']
                else:
                    category = 'Hard Drugs'
                    cat_color = self.colors['danger']

                usage_data.append({
                    'Substance': col,
                    'Usage_Rate': usage_rate,
                    'Heavy_Rate': heavy_rate,
                    'Avg_Intensity': avg_intensity,
                    'Total_Users': total_users,
                    'Category': category,
                    'Color': cat_color
                })

        # Sort by usage rate
        usage_data.sort(key=lambda x: x['Usage_Rate'], reverse=True)

        # Create enhanced visualization
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 12))

        # Main usage chart
        substances = [item['Substance'] for item in usage_data]
        usage_rates = [item['Usage_Rate'] for item in usage_data]
        heavy_rates = [item['Heavy_Rate'] for item in usage_data]
        colors = [item['Color'] for item in usage_data]

        x_pos = np.arange(len(substances))

        bars1 = ax1.bar(x_pos, usage_rates, color=colors, alpha=0.8,
                        label='Ogółem używa', edgecolor='white', linewidth=0.5)
        ax1.bar(x_pos, heavy_rates, color=colors, alpha=0.5,
                label='Intensywne użycie (ostatnio)', edgecolor='white', linewidth=0.5)

        ax1.set_xlabel('Substancje', fontweight='bold')
        ax1.set_ylabel('Odsetek używających (%)', fontweight='bold')
        ax1.set_title('💊 Ranking Popularności Substancji w Populacji UCI',
                      fontsize=14, fontweight='bold')
        ax1.set_xticks(x_pos)
        ax1.set_xticklabels(substances, rotation=45, ha='right')
        ax1.legend()
        ax1.grid(True, alpha=0.3, axis='y')

        # Add value labels on bars
        for bar, rate in zip(bars1, usage_rates):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width() / 2., height + 1,
                     f'{rate:.1f}%', ha='center', va='bottom', fontweight='bold', fontsize=9)

        # Category breakdown pie chart
        category_counts = {}
        for item in usage_data:
            if item['Category'] not in category_counts:
                category_counts[item['Category']] = {'count': 0, 'total_usage': 0}
            category_counts[item['Category']]['count'] += 1
            category_counts[item['Category']]['total_usage'] += item['Usage_Rate']

        # Average usage by category
        cat_names = list(category_counts.keys())
        cat_usage = [category_counts[cat]['total_usage'] / category_counts[cat]['count']
                     for cat in cat_names]
        cat_colors = [self.colors['success'], self.colors['warning'],
                      self.colors['secondary'], self.colors['danger']][:len(cat_names)]

        wedges, texts, autotexts = ax2.pie(cat_usage, labels=cat_names, autopct='%1.1f%%',
                                           colors=cat_colors, startangle=90)
        ax2.set_title('📊 Średnie Użycie według Kategorii Substancji', fontweight='bold')

        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontweight('bold')

        plt.tight_layout()

        # Generate detailed text analysis
        usage_text = "💊 === ANALIZA UŻYWANIA SUBSTANCJI ===\n\n"
        usage_text += f"📊 RANKING POPULARNOŚCI (TOP 10):\n"
        usage_text += "─" * 50 + "\n"

        for i, item in enumerate(usage_data[:10], 1):
            emoji = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else f"{i:>2}."
            usage_text += f"{emoji} {item['Substance']:<15} {item['Usage_Rate']:>5.1f}% ({item['Total_Users']:>4} osób) - {item['Category']}\n"

        usage_text += f"\n🎯 KATEGORIE SUBSTANCJI:\n"
        usage_text += "─" * 25 + "\n"

        for category in ['Legal', 'Soft Illegal', 'Stimulants', 'Hard Drugs']:
            cat_substances = [item for item in usage_data if item['Category'] == category]
            if cat_substances:
                avg_usage = np.mean([item['Usage_Rate'] for item in cat_substances])
                count = len(cat_substances)

                risk_level = "🟢 NISKIE" if category == 'Legal' else "🟡 UMIARKOWANE" if category == 'Soft Illegal' else "🟠 WYSOKIE" if category == 'Stimulants' else "🔴 BARDZO WYSOKIE"

                usage_text += f"{category}: {avg_usage:.1f}% średnie użycie, {count} substancji ({risk_level} ryzyko)\n"

        # Key insights
        most_popular = usage_data[0]
        least_popular = usage_data[-1]

        usage_text += f"\n💡 KLUCZOWE SPOSTRZEŻENIA:\n"
        usage_text += "─" * 25 + "\n"
        usage_text += f"🏆 Najpopularniejsza: {most_popular['Substance']} ({most_popular['Usage_Rate']:.1f}%)\n"
        usage_text += f"🏅 Najrzadsza: {least_popular['Substance']} ({least_popular['Usage_Rate']:.1f}%)\n"

        legal_avg = np.mean([item['Usage_Rate'] for item in usage_data if item['Category'] == 'Legal'])
        illegal_avg = np.mean([item['Usage_Rate'] for item in usage_data if item['Category'] != 'Legal'])

        usage_text += f"📊 Średnie użycie substancji legalnych: {legal_avg:.1f}%\n"
        usage_text += f"📊 Średnie użycie substancji nielegalnych: {illegal_avg:.1f}%\n"
        usage_text += f"📈 Stosunek legalne/nielegalne: {legal_avg / illegal_avg:.1f}:1\n"

        # Population risk assessment
        high_risk_users = len(self.processed_df[
                                  (self.processed_df[['Heroin', 'Crack', 'Cocaine']].sum(axis=1) > 0) |
                                  (self.processed_df[['Cannabis', 'LSD', 'Ecstasy']].sum(axis=1) >= 2)
                                  ]) if all(col in self.processed_df.columns for col in
                                            ['Heroin', 'Crack', 'Cocaine', 'Cannabis', 'LSD', 'Ecstasy']) else 0

        usage_text += f"\n🚨 OCENA RYZYKA POPULACYJNEGO:\n"
        usage_text += f"• Wysokie ryzyko (hard drugs lub polydrug): {high_risk_users} osób ({high_risk_users / len(self.processed_df) * 100:.1f}%)\n"
        usage_text += f"• Tylko legalne substancje: {len(self.processed_df[(self.processed_df[['Cannabis', 'Cocaine', 'Heroin', 'LSD']].sum(axis=1) == 0)])} osób\n"

        self.result_text.delete(1.0, tk.END)
        self.result_text.insert(tk.END, usage_text)

        canvas = FigureCanvasTkAgg(fig, self.plot_frame)
        canvas.draw()
        canvas.get_tk_widget().grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        self.plot_frame.columnconfigure(0, weight=1)
        self.plot_frame.rowconfigure(0, weight=1)

    def compare_all_classifications(self):
        """Enhanced comparison of Random Forest for all substances"""
        if self.processed_df is None:
            messagebox.showerror("Błąd", "Najpierw wczytaj dane!")
            return

        results = []
        features = self.processed_df[self.personality_cols].dropna()

        comparison_text = "🏆 === RANKING PRZEWIDYWALNOŚCI WSZYSTKICH SUBSTANCJI ===\n\n"
        comparison_text += "🤖 Analizuję każdą substancję Random Forest...\n\n"

        for substance in self.substance_cols:
            if substance not in self.processed_df.columns:
                continue

            target = (self.processed_df.loc[features.index, substance] > 0).astype(int)
            usage_count = target.sum()
            non_usage_count = len(target) - usage_count

            if usage_count < 20 or non_usage_count < 20:
                results.append({
                    'Substance': substance,
                    'Users': usage_count,
                    'Non_Users': non_usage_count,
                    'Usage_Rate': target.mean(),
                    'Accuracy': np.nan,
                    'AUC': np.nan,
                    'Baseline': np.nan,
                    'Improvement': np.nan,
                    'Top_Feature': 'Insufficient data',
                    'Top_Importance': 0,
                    'Status': 'Insufficient data',
                    'Category': self.get_substance_category(substance)
                })
                continue

            try:
                X_train, X_test, y_train, y_test = train_test_split(
                    features, target, test_size=0.3, random_state=42, stratify=target
                )

                rf = RandomForestClassifier(n_estimators=100, random_state=42)
                rf.fit(X_train, y_train)

                y_pred = rf.predict(X_test)
                y_pred_proba = rf.predict_proba(X_test)[:, 1]

                accuracy = accuracy_score(y_test, y_pred)
                baseline = max(target.mean(), 1 - target.mean())
                improvement = accuracy - baseline

                try:
                    auc_score = roc_auc_score(y_test, y_pred_proba)
                except:
                    auc_score = 0.5

                feature_importance = pd.DataFrame({
                    'Feature': self.personality_cols,
                    'Importance': rf.feature_importances_
                }).sort_values('Importance', ascending=False)

                top_feature = feature_importance.iloc[0]['Feature']
                top_importance = feature_importance.iloc[0]['Importance']

                # Enhanced status classification
                if improvement > 0.15 and auc_score > 0.8:
                    status = "🟢 Excellent"
                elif improvement > 0.08 and auc_score > 0.7:
                    status = "🟡 Good"
                elif improvement > 0.03 and auc_score > 0.6:
                    status = "🟠 Fair"
                else:
                    status = "🔴 Poor"

                results.append({
                    'Substance': substance,
                    'Users': usage_count,
                    'Non_Users': non_usage_count,
                    'Usage_Rate': target.mean(),
                    'Accuracy': accuracy,
                    'AUC': auc_score,
                    'Baseline': baseline,
                    'Improvement': improvement,
                    'Top_Feature': top_feature,
                    'Top_Importance': top_importance,
                    'Status': status,
                    'Category': self.get_substance_category(substance)
                })

            except Exception as e:
                results.append({
                    'Substance': substance,
                    'Users': usage_count,
                    'Non_Users': non_usage_count,
                    'Usage_Rate': target.mean(),
                    'Accuracy': np.nan,
                    'AUC': np.nan,
                    'Baseline': np.nan,
                    'Improvement': np.nan,
                    'Top_Feature': f'Error: {str(e)[:20]}',
                    'Top_Importance': 0,
                    'Status': '❌ Error',
                    'Category': self.get_substance_category(substance)
                })

        results_df = pd.DataFrame(results)
        results_df = results_df.sort_values('Improvement', ascending=False, na_last=True)

        # Enhanced ranking display
        comparison_text += "🏆 RANKING PRZEWIDYWALNOŚCI:\n"
        comparison_text += "=" * 80 + "\n"
        comparison_text += f"{'Rang':<4} {'Substancja':<15} {'Użyt.':<6} {'%Użyt':<6} {'Dokł.':<6} {'+Base':<6} {'AUC':<6} {'Status':<12} {'Top Cecha':<15}\n"
        comparison_text += "=" * 80 + "\n"

        for i, (_, row) in enumerate(results_df.iterrows(), 1):
            if pd.isna(row['Accuracy']):
                accuracy_str = "N/A"
                improvement_str = "N/A"
                auc_str = "N/A"
            else:
                accuracy_str = f"{row['Accuracy']:.3f}"
                improvement_str = f"{row['Improvement']:+.3f}"
                auc_str = f"{row['AUC']:.3f}"

            usage_pct = f"{row['Usage_Rate'] * 100:.1f}%"

            comparison_text += f"{i:<4} {row['Substance']:<15} {row['Users']:<6} {usage_pct:<6} {accuracy_str:<6} {improvement_str:<6} {auc_str:<6} {row['Status']:<12} {row['Top_Feature']:<15}\n"

        # Enhanced analysis by category
        valid_results = results_df[~pd.isna(results_df['Accuracy'])]

        if len(valid_results) > 0:
            comparison_text += "\n📊 ANALIZA WEDŁUG KATEGORII:\n"
            comparison_text += "=" * 40 + "\n"

            categories = valid_results['Category'].unique()
            for category in categories:
                cat_data = valid_results[valid_results['Category'] == category]
                avg_improvement = cat_data['Improvement'].mean()
                avg_auc = cat_data['AUC'].mean()
                count = len(cat_data)

                cat_emoji = "🟢" if category == 'Legal' else "🟡" if category == 'Soft Illegal' else "🟠" if category == 'Stimulants' else "🔴"

                comparison_text += f"{cat_emoji} {category}: {count} substancji, średnie +{avg_improvement:.3f} improvement, AUC {avg_auc:.3f}\n"

            # Top predictive features across all substances
            feature_counts = valid_results['Top_Feature'].value_counts()
            comparison_text += f"\n🎯 NAJCZĘŚCIEJ NAJWAŻNIEJSZE CECHY:\n"
            comparison_text += "─" * 35 + "\n"
            for feature, count in feature_counts.head(5).items():
                comparison_text += f"🏆 {feature}: {count} substancji ({count / len(valid_results) * 100:.0f}%)\n"

            # Performance insights
            excellent_count = len(valid_results[valid_results['Status'].str.contains('Excellent')])
            good_count = len(valid_results[valid_results['Status'].str.contains('Good')])

            comparison_text += f"\n📈 PODSUMOWANIE WYDAJNOŚCI:\n"
            comparison_text += "─" * 25 + "\n"
            comparison_text += f"🟢 Excellent models: {excellent_count}\n"
            comparison_text += f"🟡 Good models: {good_count}\n"
            comparison_text += f"📊 Średnia poprawa: +{valid_results['Improvement'].mean():.3f}\n"
            comparison_text += f"📊 Średnie AUC: {valid_results['AUC'].mean():.3f}\n"

        self.result_text.delete(1.0, tk.END)
        self.result_text.insert(tk.END, comparison_text)

        # Enhanced visualization
        if len(valid_results) > 0:
            for widget in self.plot_frame.winfo_children():
                widget.destroy()

            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 12))

            # Scatter plot: Usage Rate vs Predictability
            category_colors = {
                'Legal': self.colors['success'],
                'Soft Illegal': self.colors['warning'],
                'Stimulants': self.colors['secondary'],
                'Hard Drugs': self.colors['danger']
            }

            for category in valid_results['Category'].unique():
                cat_data = valid_results[valid_results['Category'] == category]
                ax1.scatter(cat_data['Usage_Rate'] * 100, cat_data['AUC'] * 100,
                            c=category_colors.get(category, 'gray'),
                            label=category, alpha=0.7, s=100, edgecolors='white')

            ax1.set_xlabel('Odsetek użytkowników (%)', fontweight='bold')
            ax1.set_ylabel('AUC Score (%)', fontweight='bold')
            ax1.set_title('🎯 Przewidywalność vs Popularność Substancji', fontweight='bold')
            ax1.grid(True, alpha=0.3)
            ax1.legend()

            # Add substance labels
            for _, row in valid_results.iterrows():
                ax1.annotate(row['Substance'],
                             (row['Usage_Rate'] * 100, row['AUC'] * 100),
                             xytext=(5, 5), textcoords='offset points',
                             fontsize=8, alpha=0.7)

            # Feature importance frequency
            feature_counts = valid_results['Top_Feature'].value_counts()
            if len(feature_counts) > 0:
                bars = ax2.barh(range(len(feature_counts)), feature_counts.values,
                                color=self.colors['cluster_colors'][:len(feature_counts)])
                ax2.set_yticks(range(len(feature_counts)))
                ax2.set_yticklabels(feature_counts.index)
                ax2.set_xlabel('Liczba substancji', fontweight='bold')
                ax2.set_title('🏆 Najczęściej Najważniejsze Cechy Osobowości', fontweight='bold')
                ax2.grid(True, alpha=0.3, axis='x')

                for i, (bar, count) in enumerate(zip(bars, feature_counts.values)):
                    width = bar.get_width()
                    ax2.text(width + 0.1, bar.get_y() + bar.get_height() / 2,
                             f'{count}', ha='left', va='center', fontweight='bold')

            plt.tight_layout()

            canvas = FigureCanvasTkAgg(fig, self.plot_frame)
            canvas.draw()
            canvas.get_tk_widget().grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

    def get_substance_category(self, substance):
        """Categorize substances by type"""
        if substance in ['Alcohol', 'Caffeine', 'Chocolate', 'Nicotine']:
            return 'Legal'
        elif substance in ['Cannabis', 'LSD', 'Mushrooms']:
            return 'Soft Illegal'
        elif substance in ['Cocaine', 'Ecstasy', 'Amphetamines']:
            return 'Stimulants'
        else:
            return 'Hard Drugs'

    def analyze_cluster_demographics(self):
        """Analyze demographics of personality clusters"""
        if self.cluster_labels is None:
            messagebox.showwarning("Uwaga", "Najpierw wykonaj analizę klastrów!")
            return

        cluster_demo_df = self.processed_df.copy()
        cluster_demo_df['Cluster'] = self.cluster_labels

        demo_text = "👥 === DEMOGRAFIA KLASTRÓW OSOBOWOŚCI ===\n\n"

        for cluster in range(4):
            cluster_data = cluster_demo_df[cluster_demo_df['Cluster'] == cluster]
            profile = self.cluster_profiles[cluster]

            demo_text += f"{profile['emoji']} KLASTER {cluster}: {profile['name']}\n"
            demo_text += f"📊 Wielkość: {len(cluster_data)} osób ({len(cluster_data) / len(cluster_demo_df) * 100:.1f}%)\n"
            demo_text += "─" * 60 + "\n"

            # Age analysis
            if 'Age_Category' in cluster_data.columns:
                age_dist = cluster_data['Age_Category'].value_counts(normalize=True) * 100
                demo_text += "👶 WIEK:\n"
                for age, pct in age_dist.items():
                    demo_text += f"  {age}: {pct:.1f}%\n"

            # Gender analysis
            if 'Gender_Category' in cluster_data.columns:
                gender_dist = cluster_data['Gender_Category'].value_counts(normalize=True) * 100
                demo_text += "⚧️ PŁEĆ:\n"
                for gender, pct in gender_dist.items():
                    demo_text += f"  {gender}: {pct:.1f}%\n"

            # Education analysis
            if 'Education_Category' in cluster_data.columns:
                edu_dist = cluster_data['Education_Category'].value_counts(normalize=True) * 100
                demo_text += "🎓 WYKSZTAŁCENIE (top 3):\n"
                for edu, pct in edu_dist.head(3).items():
                    demo_text += f"  {edu}: {pct:.1f}%\n"

            demo_text += "\n"

        self.result_text.delete(1.0, tk.END)
        self.result_text.insert(tk.END, demo_text)


def main():
    root = tk.Tk()
    app = DrugConsumptionAnalyzer(root)
    root.mainloop()


if __name__ == "__main__":
    main()
