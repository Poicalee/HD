# -*- coding: utf-8 -*-
"""
Klasa do tworzenia wizualizacji dla aplikacji UCI Drug Consumption
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from typing import Tuple, List, Optional
from scipy import stats

from utils.constants import COLORS, PERSONALITY_COLS, SUBSTANCE_COLS
from utils.helpers import get_substance_category, calculate_usage_stats

# Ustawienia matplotlib
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

class PlotManager:
    """Klasa do zarządzania wizualizacjami"""

    def __init__(self):
        self.colors = COLORS
        self.default_figsize = (16, 10)

    def create_personality_distributions(self, df: pd.DataFrame) -> Tuple[plt.Figure, str]:
        """
        Tworzy wykres rozkładów cech osobowości
        
        Args:
            df: DataFrame z danymi
            
        Returns:
            (figura, tekst_analizy)
        """
        existing_cols = [col for col in PERSONALITY_COLS if col in df.columns]

        if not existing_cols:
            return None, "Brak danych cech osobowości"

        n_cols = min(len(existing_cols), 9)
        n_rows = int(np.ceil(n_cols / 3))

        fig, axes = plt.subplots(n_rows, 3, figsize=(16, 5 * n_rows))
        fig.suptitle('🧠 Rozkłady Cech Osobowości w Populacji UCI',
                     fontsize=16, fontweight='bold', y=0.95)

        if n_rows == 1:
            axes = axes.reshape(1, -1)
        axes = axes.ravel()

        analysis_text = "🧠 === ANALIZA ROZKŁADÓW CECH OSOBOWOŚCI ===\n\n"

        for i, col in enumerate(existing_cols[:n_cols]):
            data = df[col].dropna()

            if len(data) == 0:
                continue

            # Histogram z nakładką rozkładu normalnego
            n, bins, patches = axes[i].hist(data, bins=40, alpha=0.7,
                                            color=self.colors['cluster_colors'][i % 4],
                                            edgecolor='white', linewidth=0.5)

            # Rozkład normalny
            mu, sigma = data.mean(), data.std()
            x = np.linspace(data.min(), data.max(), 100)
            y = ((1 / (np.sqrt(2 * np.pi) * sigma)) *
                 np.exp(-0.5 * ((x - mu) / sigma) ** 2)) * len(data) * (bins[1] - bins[0])
            axes[i].plot(x, y, 'r--', alpha=0.8, linewidth=2, label='Rozkład normalny')

            # Linie średniej i mediany
            axes[i].axvline(mu, color='red', linestyle='-', linewidth=2, alpha=0.8,
                            label=f'Średnia: {mu:.2f}')
            axes[i].axvline(data.median(), color='orange', linestyle='--', linewidth=2, alpha=0.8,
                            label=f'Mediana: {data.median():.2f}')

            axes[i].set_title(f'{col}\n(μ={mu:.2f}, σ={sigma:.2f})', fontweight='bold')
            axes[i].set_xlabel('Wartość')
            axes[i].set_ylabel('Częstość')
            axes[i].legend(fontsize=8)
            axes[i].grid(True, alpha=0.3)
            axes[i].set_facecolor('#f8f9fa')

            # Analiza statystyczna
            skewness = stats.skew(data)
            kurtosis = stats.kurtosis(data)

            # Test normalności (ograniczony do 5000 próbek)
            test_data = data.sample(min(5000, len(data)), random_state=42)
            _, p_value = stats.shapiro(test_data)

            normality = "🟢 Normalny" if p_value > 0.05 else "🔴 Nie-normalny"
            skew_desc = self._interpret_skewness(skewness)
            kurt_desc = self._interpret_kurtosis(kurtosis)

            analysis_text += f"{col}:\n"
            analysis_text += f"  📊 Średnia: {mu:.3f}, Mediana: {data.median():.3f}, Odchyl. std: {sigma:.3f}\n"
            analysis_text += f"  📈 Skośność: {skewness:.3f} ({skew_desc})\n"
            analysis_text += f"  📈 Kurtoza: {kurtosis:.3f} ({kurt_desc})\n"
            analysis_text += f"  🧪 Normalność: p={p_value:.3f} ({normality})\n\n"

        # Usuń nieużywane subploty
        for i in range(len(existing_cols), len(axes)):
            fig.delaxes(axes[i])

        plt.tight_layout()
        plt.subplots_adjust(top=0.90)

        return fig, analysis_text

    def create_substance_usage_plot(self, df: pd.DataFrame) -> Tuple[plt.Figure, str]:
        """
        Tworzy wykres używania substancji
        
        Args:
            df: DataFrame z danymi
            
        Returns:
            (figura, tekst_analizy)
        """
        existing_substance_cols = [col for col in SUBSTANCE_COLS if col in df.columns]

        if not existing_substance_cols:
            return None, "Brak danych substancji"

        # Oblicz statystyki używania
        usage_data = []
        for col in existing_substance_cols:
            total_users = (df[col] > 0).sum()
            heavy_users = (df[col] >= 5).sum()  # Recent use
            usage_rate = total_users / len(df) * 100
            heavy_rate = heavy_users / len(df) * 100
            category = get_substance_category(col)

            usage_data.append({
                'Substance': col,
                'Usage_Rate': usage_rate,
                'Heavy_Rate': heavy_rate,
                'Total_Users': total_users,
                'Category': category
            })

        # Sortuj według popularności
        usage_data.sort(key=lambda x: x['Usage_Rate'], reverse=True)

        # Twórz wykres
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=self.default_figsize)

        # Główny wykres
        substances = [item['Substance'] for item in usage_data]
        usage_rates = [item['Usage_Rate'] for item in usage_data]
        heavy_rates = [item['Heavy_Rate'] for item in usage_data]

        # Kolory według kategorii
        category_colors = {
            'Legal': self.colors['success'],
            'Soft Illegal': self.colors['warning'],
            'Stimulants': self.colors['secondary'],
            'Hard Drugs': self.colors['danger'],
            'Other': self.colors['dark']
        }

        colors = [category_colors.get(item['Category'], self.colors['dark']) for item in usage_data]

        x_pos = np.arange(len(substances))

        bars1 = ax1.bar(x_pos, usage_rates, color=colors, alpha=0.8,
                        label='Ogółem używa', edgecolor='white', linewidth=0.5)
        ax1.bar(x_pos, heavy_rates, color=colors, alpha=0.5,
                label='Intensywne użycie', edgecolor='white', linewidth=0.5)

        ax1.set_xlabel('Substancje', fontweight='bold')
        ax1.set_ylabel('Odsetek używających (%)', fontweight='bold')
        ax1.set_title('💊 Ranking Popularności Substancji w Populacji UCI',
                      fontsize=14, fontweight='bold')
        ax1.set_xticks(x_pos)
        ax1.set_xticklabels(substances, rotation=45, ha='right')
        ax1.legend()
        ax1.grid(True, alpha=0.3, axis='y')

        # Dodaj etykiety wartości
        for bar, rate in zip(bars1, usage_rates):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width() / 2., height + 1,
                     f'{rate:.1f}%', ha='center', va='bottom', fontweight='bold', fontsize=9)

        # Wykres kołowy kategorii
        category_usage = {}
        for item in usage_data:
            cat = item['Category']
            if cat not in category_usage:
                category_usage[cat] = []
            category_usage[cat].append(item['Usage_Rate'])

        # Średnie użycie według kategorii
        cat_names = list(category_usage.keys())
        cat_avg_usage = [np.mean(usage) for usage in category_usage.values()]
        cat_colors = [category_colors.get(cat, self.colors['dark']) for cat in cat_names]

        wedges, texts, autotexts = ax2.pie(cat_avg_usage, labels=cat_names, autopct='%1.1f%%',
                                           colors=cat_colors, startangle=90)
        ax2.set_title('📊 Średnie Użycie według Kategorii Substancji', fontweight='bold')

        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontweight('bold')

        plt.tight_layout()

        # Generuj analizę tekstową
        analysis_text = self._generate_substance_analysis(usage_data)

        return fig, analysis_text

    def create_demographic_boxplots(self, df: pd.DataFrame, substance: str) -> Tuple[plt.Figure, str]:
        """
        Tworzy boxploty demograficzne dla substancji
        
        Args:
            df: DataFrame z danymi
            substance: Nazwa substancji
            
        Returns:
            (figura, tekst_analizy)
        """
        if substance not in df.columns:
            return None, f"Brak danych dla substancji: {substance}"

        users = df[df[substance] > 0]
        non_users = df[df[substance] == 0]

        if len(users) == 0 or len(non_users) == 0:
            return None, f"Brak wystarczających danych dla {substance}"

        existing_personality_cols = [col for col in PERSONALITY_COLS if col in df.columns]

        if not existing_personality_cols:
            return None, "Brak danych cech osobowości"

        # Ograniczamy do 8 cech
        cols_to_plot = existing_personality_cols[:8]

        fig, axes = plt.subplots(2, 4, figsize=(18, 12))
        fig.suptitle(f'🧠 Analiza Demograficzna: {substance}\n'
                     f'Użytkownicy ({len(users)}) vs Nieużytkownicy ({len(non_users)})',
                     fontsize=16, fontweight='bold', y=0.95)
        axes = axes.ravel()

        analysis_text = f"=== 🎯 ANALIZA DEMOGRAFICZNA - {substance.upper()} ===\n\n"
        analysis_text += f"👥 Użytkownicy: {len(users)} osób ({len(users) / len(df) * 100:.1f}%)\n"
        analysis_text += f"🚫 Nieużytkownicy: {len(non_users)} osób ({len(non_users) / len(df) * 100:.1f}%)\n\n"
        analysis_text += "🧠 RÓŻNICE W CECHACH OSOBOWOŚCI:\n" + "=" * 50 + "\n"

        for i, col in enumerate(cols_to_plot):
            non_user_data = non_users[col].dropna()
            user_data = users[col].dropna()

            if len(non_user_data) == 0 or len(user_data) == 0:
                continue

            # Test statystyczny
            try:
                from scipy.stats import mannwhitneyu
                statistic, p_value = mannwhitneyu(non_user_data, user_data,
                                                  alternative='two-sided')
                significance = self._get_significance_level(p_value)
            except:
                p_value = 1.0
                significance = "n/a"

            # Boxplot
            box_data = [non_user_data, user_data]
            bp = axes[i].boxplot(box_data, labels=['Nie używa', 'Używa'],
                                 patch_artist=True, medianprops={'color': 'white', 'linewidth': 2})

            # Kolorowanie pudełek
            bp['boxes'][0].set_facecolor(self.colors['info'])
            bp['boxes'][0].set_alpha(0.7)

            diff = user_data.median() - non_user_data.median()
            bp['boxes'][1].set_facecolor(self._get_risk_color(diff))
            bp['boxes'][1].set_alpha(0.8)

            axes[i].set_title(f'{col}\n({significance})', fontweight='bold', fontsize=11)
            axes[i].grid(True, alpha=0.3)
            axes[i].set_facecolor('#f8f9fa')

            # Dodaj różnicę
            axes[i].text(0.5, 0.95, f'Δ={diff:.2f}', transform=axes[i].transAxes,
                         ha='center', va='top', fontweight='bold',
                         bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

            # Interpretacja
            interpretation = self._interpret_difference(diff, significance)
            analysis_text += f"{col:<18} {interpretation:>12} | Δ={diff:>+6.3f} | p={p_value:.3f} {significance}\n"

        # Usuń nieużywane subploty
        for i in range(len(cols_to_plot), 8):
            fig.delaxes(axes[i])

        plt.tight_layout()
        plt.subplots_adjust(top=0.90)

        return fig, analysis_text

    def create_correlation_heatmap(self, correlation_matrix: pd.DataFrame,
                                   title: str) -> plt.Figure:
        """
        Tworzy heatmapę korelacji
        
        Args:
            correlation_matrix: Macierz korelacji
            title: Tytuł wykresu
            
        Returns:
            Figura matplotlib
        """
        fig, ax = plt.subplots(figsize=(14, 12))

        # Maska dla górnego trójkąta
        mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))

        # Heatmapa
        sns.heatmap(correlation_matrix, annot=True, cmap='RdBu_r', center=0,
                    square=True, ax=ax, fmt='.2f', cbar_kws={'shrink': 0.8},
                    mask=mask, linewidths=0.5)

        ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()

        return fig

    def embed_plot_in_frame(self, fig: plt.Figure, frame) -> None:
        """
        Osadza wykres w ramce tkinter
        
        Args:
            fig: Figura matplotlib
            frame: Ramka tkinter
        """
        # Usuń poprzednie wykresy
        for widget in frame.winfo_children():
            widget.destroy()

        # Osadź nowy wykres
        canvas = FigureCanvasTkAgg(fig, frame)
        canvas.draw()
        canvas.get_tk_widget().grid(row=0, column=0, sticky='nsew')

        # Konfiguracja grid
        frame.columnconfigure(0, weight=1)
        frame.rowconfigure(0, weight=1)

    def _interpret_skewness(self, skewness: float) -> str:
        """Interpretuje skośność"""
        if abs(skewness) < 0.5:
            return "👍 Normalny"
        elif abs(skewness) < 1:
            return "⚠️ Skośny"
        else:
            return "❌ Bardzo skośny"

    def _interpret_kurtosis(self, kurtosis: float) -> str:
        """Interpretuje kurtozę"""
        if abs(kurtosis) < 0.5:
            return "👍 Normalny"
        elif abs(kurtosis) < 1:
            return "⚠️ Odchylenie"
        else:
            return "❌ Silne odchylenie"

    def _get_significance_level(self, p_value: float) -> str:
        """Zwraca poziom istotności"""
        if p_value < 0.001:
            return "***"
        elif p_value < 0.01:
            return "**"
        elif p_value < 0.05:
            return "*"
        else:
            return "ns"

    def _interpret_difference(self, diff: float, significance: str) -> str:
        """Interpretuje różnicę między grupami"""
        if significance in ['***', '**', '*']:
            if diff > 0.2:
                return "📈 WYŻEJ"
            elif diff < -0.2:
                return "📉 NIŻEJ"
            else:
                return "≈ PODOBNIE"
        else:
            return "≈ BRAK RÓŻNICY"

    def _get_risk_color(self, value: float) -> str:
        """Zwraca kolor ryzyka"""
        if value > 0.3:
            return self.colors['risk_gradient'][2]  # Red for high risk
        elif value > 0:
            return self.colors['risk_gradient'][1]  # Orange for moderate
        else:
            return self.colors['risk_gradient'][0]  # Green for low risk

    def _generate_substance_analysis(self, usage_data: List[dict]) -> str:
        """Generuje analizę tekstową substancji"""
        text = "💊 === ANALIZA UŻYWANIA SUBSTANCJI ===\n\n"
        text += f"📊 RANKING POPULARNOŚCI (TOP 10):\n"
        text += "─" * 50 + "\n"

        for i, item in enumerate(usage_data[:10], 1):
            emoji = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else f"{i:>2}."
            text += f"{emoji} {item['Substance']:<15} {item['Usage_Rate']:>5.1f}% ({item['Total_Users']:>4} osób) - {item['Category']}\n"

        # Analiza według kategorii
        categories = {}
        for item in usage_data:
            cat = item['Category']
            if cat not in categories:
                categories[cat] = []
            categories[cat].append(item['Usage_Rate'])

        text += f"\n🎯 KATEGORIE SUBSTANCJI:\n"
        text += "─" * 25 + "\n"

        risk_levels = {
            'Legal': "🟢 NISKIE",
            'Soft Illegal': "🟡 UMIARKOWANE",
            'Stimulants': "🟠 WYSOKIE",
            'Hard Drugs': "🔴 BARDZO WYSOKIE"
        }

        for category, rates in categories.items():
            avg_usage = np.mean(rates)
            count = len(rates)
            risk = risk_levels.get(category, "❓ NIEZNANE")
            text += f"{category}: {avg_usage:.1f}% średnie użycie, {count} substancji ({risk} ryzyko)\n"

        return text