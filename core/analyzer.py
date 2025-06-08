# -*- coding: utf-8 -*-
"""
Klasa do przeprowadzania analiz statystycznych danych UCI
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from scipy import stats
from scipy.stats import mannwhitneyu, shapiro

from utils.constants import PERSONALITY_COLS, SUBSTANCE_COLS, DEMOGRAPHIC_COLS
from utils.helpers import (calculate_usage_stats, format_percentage,
                           create_demographic_summary, get_substance_category)

class StatisticalAnalyzer:
    """Klasa do przeprowadzania analiz statystycznych"""

    def __init__(self):
        self.correlation_cache = {}

    def calculate_basic_stats(self, df: pd.DataFrame) -> str:
        """
        Oblicza podstawowe statystyki opisowe
        
        Args:
            df: DataFrame z danymi
            
        Returns:
            Sformatowany tekst ze statystykami
        """
        stats_text = "📊 === STATYSTYKI OPISOWE UCI DATASET ===\n\n"

        # Statystyki cech osobowości
        existing_personality_cols = [col for col in PERSONALITY_COLS if col in df.columns]

        if existing_personality_cols:
            stats_text += "🧠 CECHY OSOBOWOŚCI (Big Five + Impulsywność + Sensation Seeking):\n"
            stats_text += "─" * 70 + "\n"
            personality_stats = df[existing_personality_cols].describe()
            stats_text += personality_stats.to_string() + "\n\n"

            # Skośność i kurtoza
            stats_text += "📈 SKOŚNOŚĆ I KURTOZA (ocena normalności rozkładów):\n"
            stats_text += "─" * 50 + "\n"

            for col in existing_personality_cols:
                data = df[col].dropna()
                if len(data) > 0:
                    skewness = stats.skew(data)
                    kurtosis = stats.kurtosis(data)

                    # Interpretacja
                    skew_interp = self._interpret_skewness(skewness)
                    kurt_interp = self._interpret_kurtosis(kurtosis)

                    stats_text += f"{col:<18}: skośność={skewness:>6.3f} {skew_interp}, kurtoza={kurtosis:>6.3f} {kurt_interp}\n"

        # Statystyki substancji
        existing_substance_cols = [col for col in SUBSTANCE_COLS if col in df.columns]

        if existing_substance_cols:
            stats_text += "\n💊 STATYSTYKI UŻYWANIA SUBSTANCJI:\n"
            stats_text += "─" * 40 + "\n"
            stats_text += f"{'Substancja':<15} {'Nigdy':<6} {'Ostatnio':<8} {'Intensywnie':<12} {'Popularność'}\n"
            stats_text += "─" * 60 + "\n"

            usage_stats = calculate_usage_stats(df, existing_substance_cols)

            for substance, stats_dict in usage_stats.items():
                never_used = len(df) - stats_dict['total_users']
                recent_use = stats_dict['heavy_users']
                popularity = stats_dict['usage_rate']

                # Visual popularity indicator
                pop_emoji = self._get_popularity_emoji(popularity)

                stats_text += f"{substance:<15} {never_used:<6} {recent_use:<8} {recent_use:<12} {popularity:>5.1f}% {pop_emoji}\n"

        # Kluczowe spostrzeżenia
        stats_text += self._generate_key_insights(df, existing_substance_cols)

        return stats_text

    def analyze_correlations(self, df: pd.DataFrame, columns: List[str],
                             analysis_type: str = "personality") -> Tuple[pd.DataFrame, str]:
        """
        Analizuje korelacje między zmiennymi
        
        Args:
            df: DataFrame z danymi
            columns: Lista kolumn do analizy
            analysis_type: Typ analizy ("personality" lub "substance")
            
        Returns:
            (correlation_matrix, analysis_text)
        """
        existing_cols = [col for col in columns if col in df.columns]

        if len(existing_cols) < 2:
            return pd.DataFrame(), "Brak wystarczających danych do analizy korelacji"

        correlation_matrix = df[existing_cols].corr()

        # Tekst analizy
        if analysis_type == "personality":
            analysis_text = self._analyze_personality_correlations(correlation_matrix, existing_cols)
        else:
            analysis_text = self._analyze_substance_correlations(correlation_matrix, existing_cols)

        return correlation_matrix, analysis_text

    def _analyze_personality_correlations(self, corr_matrix: pd.DataFrame,
                                          cols: List[str]) -> str:
        """Analizuje korelacje cech osobowości"""

        text = "🧠 === KORELACJE CECH OSOBOWOŚCI ===\n\n"
        text += corr_matrix.round(3).to_string() + "\n\n"

        # Znajdź silne korelacje
        strong_correlations = []
        for i in range(len(cols)):
            for j in range(i + 1, len(cols)):
                col1, col2 = cols[i], cols[j]
                if col1 in corr_matrix.columns and col2 in corr_matrix.columns:
                    corr_val = corr_matrix.loc[col1, col2]
                    if abs(corr_val) > 0.3:
                        strong_correlations.append((col1, col2, corr_val))

        strong_correlations.sort(key=lambda x: abs(x[2]), reverse=True)

        text += "🔥 NAJSILNIEJSZE KORELACJE (|r| > 0.3):\n"
        text += "─" * 45 + "\n"

        for col1, col2, corr_val in strong_correlations:
            direction = "📈 Pozytywna" if corr_val > 0 else "📉 Negatywna"
            strength = self._get_correlation_strength(abs(corr_val))
            text += f"{col1} ↔ {col2}: {corr_val:+.3f} {direction} {strength}\n"

        return text

    def _analyze_substance_correlations(self, corr_matrix: pd.DataFrame,
                                        cols: List[str]) -> str:
        """Analizuje korelacje substancji"""

        text = "💊 === KORELACJE UŻYWANIA SUBSTANCJI ===\n\n"

        # Znajdź silne korelacje
        strong_correlations = []
        for i in range(len(cols)):
            for j in range(i + 1, len(cols)):
                col1, col2 = cols[i], cols[j]
                if col1 in corr_matrix.columns and col2 in corr_matrix.columns:
                    corr_val = corr_matrix.loc[col1, col2]
                    if abs(corr_val) > 0.2:
                        strong_correlations.append((col1, col2, corr_val))

        strong_correlations.sort(key=lambda x: abs(x[2]), reverse=True)

        # Kategoryzuj korelacje
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

        text += "🎯 KLASTRY SUBSTANCJI (najsilniejsze korelacje):\n"
        text += "─" * 50 + "\n"

        text += "🎉 PARTY DRUGS CLUSTER:\n"
        for col1, col2, corr_val in party_drugs[:5]:
            text += f"  • {col1} ↔ {col2}: {corr_val:.3f}\n"

        text += "\n🌈 PSYCHEDELICS CLUSTER:\n"
        for col1, col2, corr_val in psychedelics[:5]:
            text += f"  • {col1} ↔ {col2}: {corr_val:.3f}\n"

        text += "\n🚨 HARD DRUGS CLUSTER:\n"
        for col1, col2, corr_val in hard_drugs[:5]:
            text += f"  • {col1} ↔ {col2}: {corr_val:.3f}\n"

        return text

    def analyze_demographic_differences(self, df: pd.DataFrame, substance: str) -> str:
        """
        Analizuje różnice demograficzne dla danej substancji
        
        Args:
            df: DataFrame z danymi
            substance: Nazwa substancji
            
        Returns:
            Sformatowany tekst analizy
        """
        if substance not in df.columns:
            return f"Brak danych dla substancji: {substance}"

        users = df[df[substance] > 0]
        non_users = df[df[substance] == 0]

        if len(users) == 0 or len(non_users) == 0:
            return f"Brak wystarczających danych dla {substance}"

        analysis_text = f"=== 🎯 ANALIZA DEMOGRAFICZNA - {substance.upper()} ===\n\n"
        analysis_text += f"👥 Użytkownicy: {len(users)} osób ({len(users) / len(df) * 100:.1f}%)\n"
        analysis_text += f"🚫 Nieużytkownicy: {len(non_users)} osób ({len(non_users) / len(df) * 100:.1f}%)\n\n"

        # Analiza cech osobowości
        existing_personality_cols = [col for col in PERSONALITY_COLS if col in df.columns]

        if existing_personality_cols:
            analysis_text += "🧠 RÓŻNICE W CECHACH OSOBOWOŚCI:\n"
            analysis_text += "=" * 50 + "\n"

            for col in existing_personality_cols:
                non_user_data = non_users[col].dropna()
                user_data = users[col].dropna()

                # Inicjalizuj zmienne z wartościami domyślnymi
                p_value = 1.0
                significance = "n/a"

                if len(non_user_data) > 5 and len(user_data) > 5:
                    # Test statystyczny
                    try:
                        statistic, p_value = mannwhitneyu(non_user_data, user_data,
                                                          alternative='two-sided')
                        significance = self._get_significance_level(p_value)
                    except Exception as e:
                        print(f"Błąd testu statystycznego dla {col}: {e}")
                        p_value = 1.0
                        significance = "error"

                # Różnica median
                median_diff = user_data.median() - non_user_data.median()
                interpretation = self._interpret_difference(median_diff, significance)

                analysis_text += f"{col:<18} {interpretation:>12} | Δ={median_diff:>+6.3f} | p={p_value:.3f} {significance}\n"

        # Interpretacja ogólna
        analysis_text += self._generate_interpretation(substance, users, non_users)

        return analysis_text

    def _interpret_skewness(self, skewness: float) -> str:
        """Interpretuje skośność rozkładu"""
        if abs(skewness) < 0.5:
            return "👍 Normalny"
        elif abs(skewness) < 1:
            return "⚠️ Skośny"
        else:
            return "❌ Bardzo skośny"

    def _interpret_kurtosis(self, kurtosis: float) -> str:
        """Interpretuje kurtozę rozkładu"""
        if abs(kurtosis) < 0.5:
            return "👍 Normalny"
        elif abs(kurtosis) < 1:
            return "⚠️ Odchylenie"
        else:
            return "❌ Silne odchylenie"

    def _get_popularity_emoji(self, popularity: float) -> str:
        """Zwraca emoji popularności"""
        if popularity > 80:
            return "🔥🔥🔥"
        elif popularity > 50:
            return "🔥🔥"
        elif popularity > 20:
            return "🔥"
        else:
            return "❄️"

    def _get_correlation_strength(self, abs_corr: float) -> str:
        """Zwraca emoji siły korelacji"""
        if abs_corr > 0.6:
            return "🔥🔥🔥"
        elif abs_corr > 0.4:
            return "🔥🔥"
        else:
            return "🔥"

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

    def _generate_key_insights(self, df: pd.DataFrame, substance_cols: List[str]) -> str:
        """Generuje kluczowe spostrzeżenia"""
        text = f"\n💡 KLUCZOWE SPOSTRZEŻENIA:\n"
        text += "─" * 25 + "\n"

        if substance_cols:
            # Najpopularniejsza i najrzadsza substancja
            usage_rates = {}
            for col in substance_cols:
                usage_rates[col] = (df[col] > 0).sum() / len(df) * 100

            if usage_rates:
                most_popular = max(usage_rates, key=usage_rates.get)
                least_popular = min(usage_rates, key=usage_rates.get)

                text += f"🏆 Najpopularniejsza substancja: {most_popular} ({usage_rates[most_popular]:.1f}%)\n"
                text += f"🏅 Najrzadsza substancja: {least_popular} ({usage_rates[least_popular]:.1f}%)\n"

                # Kategorie substancji
                legal_substances = ['Alcohol', 'Caffeine', 'Chocolate', 'Nicotine']
                legal_avg = np.mean([usage_rates[s] for s in legal_substances if s in usage_rates])
                illegal_avg = np.mean([usage_rates[s] for s in usage_rates if s not in legal_substances])

                text += f"📊 Średnie używanie substancji legalnych: {legal_avg:.1f}%\n"
                text += f"📊 Średnie używanie substancji nielegalnych: {illegal_avg:.1f}%\n"

        return text

    def _generate_interpretation(self, substance: str, users: pd.DataFrame,
                                 non_users: pd.DataFrame) -> str:
        """Generuje interpretację różnic demograficznych"""
        text = f"\n💡 INTERPRETACJA:\n"
        text += "=" * 30 + "\n"

        # Ocena ryzyka substancji
        if substance in ['Heroin', 'Crack', 'Cocaine']:
            text += "🚨 WYSOKIE RYZYKO: Substancja związana z problematycznymi wzorcami osobowości\n"
        elif substance in ['LSD', 'Mushrooms', 'Cannabis']:
            text += "🟡 UMIARKOWANE RYZYKO: Substancja związana z eksperymentowaniem\n"
        else:
            text += "🟢 NISKIE RYZYKO: Substancja mainstream z szerokim profilem użytkowników\n"

        return text