# -*- coding: utf-8 -*-
"""
Klasa do analizy klastrów osobowości UCI Drug Consumption
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Dict, Optional
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from utils.constants import PERSONALITY_COLS, SUBSTANCE_COLS, CLUSTER_PROFILES
from utils.helpers import get_substance_category

class ClusterAnalyzer:
    """Klasa do analizy klastrów osobowości"""

    def __init__(self, n_clusters: int = 4):
        self.n_clusters = n_clusters
        self.kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=2)
        self.cluster_labels = None
        self.features_scaled = None
        self.cluster_profiles = CLUSTER_PROFILES

    def perform_clustering(self, df: pd.DataFrame) -> Tuple[bool, str, Optional[np.ndarray]]:
        """
        Przeprowadza analizę klastrów K-means
        
        Args:
            df: DataFrame z danymi
            
        Returns:
            (success, message, cluster_labels)
        """
        try:
            # Sprawdź dostępne cechy osobowości
            existing_personality_cols = [col for col in PERSONALITY_COLS if col in df.columns]

            if len(existing_personality_cols) < 3:
                return False, "Za mało cech osobowości do klastrowania", None

            # Przygotuj dane
            features = df[existing_personality_cols].dropna()

            if len(features) < 20:
                return False, "Za mało obserwacji do klastrowania", None

            # Standaryzacja
            self.features_scaled = self.scaler.fit_transform(features)

            # Klastrowanie
            self.cluster_labels = self.kmeans.fit_predict(self.features_scaled)

            return True, f"Pomyślnie utworzono {self.n_clusters} klastrów dla {len(features)} obserwacji", self.cluster_labels

        except Exception as e:
            return False, f"Błąd klastrowania: {str(e)}", None

    def get_cluster_analysis_text(self, df: pd.DataFrame) -> str:
        """
        Generuje szczegółową analizę tekstową klastrów
        
        Args:
            df: DataFrame z danymi
            
        Returns:
            Sformatowany tekst analizy
        """
        if self.cluster_labels is None:
            return "Najpierw wykonaj klastrowanie!"

        existing_personality_cols = [col for col in PERSONALITY_COLS if col in df.columns]
        features_df = df[existing_personality_cols].dropna().copy()
        features_df['Cluster'] = self.cluster_labels

        text = "🎯 === ANALIZA KLASTRÓW K-MEANS - 4 PROFILE OSOBOWOŚCI ===\n\n"
        text += f"📊 Liczba klastrów: {self.n_clusters}\n"
        text += f"👥 Liczba obserwacji: {len(features_df)}\n\n"

        # Szczegółowe profile klastrów
        for cluster_id in range(self.n_clusters):
            cluster_data = features_df[features_df['Cluster'] == cluster_id]
            cluster_size = len(cluster_data)
            cluster_pct = cluster_size / len(features_df) * 100

            profile = self.cluster_profiles.get(cluster_id, {
                'name': f'Klaster {cluster_id}',
                'emoji': '❓',
                'risk': 'Nieznane',
                'description': 'Brak opisu'
            })

            text += f"{profile['emoji']} KLASTER {cluster_id}: \"{profile['name']}\" (n={cluster_size}, {cluster_pct:.1f}%)\n"
            text += f"🎯 Ryzyko: {profile['risk']} | 📝 Opis: {profile['description']}\n"
            text += "─" * 70 + "\n"

            # Średnie wartości cech
            cluster_means = cluster_data[existing_personality_cols].mean()
            for trait, mean_val in cluster_means.items():
                level = self._interpret_trait_level(mean_val)
                text += f"  {trait:<18}: {mean_val:>6.3f} {level}\n"
            text += "\n"

        # Porównanie klastrów
        text += self._generate_cluster_comparison(features_df, existing_personality_cols)

        # Stratyfikacja ryzyka
        text += self._generate_risk_stratification(features_df)

        return text

    def create_cluster_visualization(self, df: pd.DataFrame) -> Tuple[plt.Figure, str]:
        """
        Tworzy wizualizację klastrów
        
        Args:
            df: DataFrame z danymi
            
        Returns:
            (figura, dodatkowy_tekst)
        """
        if self.cluster_labels is None:
            return None, "Najpierw wykonaj klastrowanie!"

        existing_personality_cols = [col for col in PERSONALITY_COLS if col in df.columns]
        features_df = df[existing_personality_cols].dropna().copy()
        features_df['Cluster'] = self.cluster_labels

        # PCA dla wizualizacji
        features_pca = self.pca.fit_transform(self.features_scaled)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

        # Wykres PCA
        for cluster_id in range(self.n_clusters):
            mask = self.cluster_labels == cluster_id
            profile = self.cluster_profiles.get(cluster_id, {
                'name': f'Klaster {cluster_id}',
                'emoji': '❓',
                'color': '#808080'
            })

            ax1.scatter(features_pca[mask, 0], features_pca[mask, 1],
                        c=profile['color'], label=f"{profile['emoji']} {profile['name']}",
                        alpha=0.7, s=50, edgecolors='white', linewidth=0.5)

        ax1.set_xlabel(f'PC1 ({self.pca.explained_variance_ratio_[0]:.1%} wariancji)',
                       fontweight='bold')
        ax1.set_ylabel(f'PC2 ({self.pca.explained_variance_ratio_[1]:.1%} wariancji)',
                       fontweight='bold')
        ax1.set_title('🎯 Wizualizacja Klastrów (PCA)', fontweight='bold', fontsize=14)
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax1.grid(True, alpha=0.3)
        ax1.set_facecolor('#f8f9fa')

        # Wykres kołowy rozmiarów klastrów
        cluster_sizes = [len(features_df[features_df['Cluster'] == i]) for i in range(self.n_clusters)]
        cluster_names = [self.cluster_profiles.get(i, {})['name'] for i in range(self.n_clusters)]
        cluster_colors = [self.cluster_profiles.get(i, {}).get('color', '#808080') for i in range(self.n_clusters)]

        wedges, texts, autotexts = ax2.pie(cluster_sizes, labels=cluster_names, autopct='%1.1f%%',
                                           colors=cluster_colors, startangle=90,
                                           explode=(0.05, 0.05, 0.05, 0.05))

        ax2.set_title('📊 Rozkład Wielkości Klastrów', fontweight='bold', fontsize=14)

        # Poprawa czytelności tekstu
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontweight('bold')

        plt.tight_layout()

        additional_text = f"""
💡 INTERPRETACJA PCA:
• PC1 wyjaśnia {self.pca.explained_variance_ratio_[0]:.1%} wariancji
• PC2 wyjaśnia {self.pca.explained_variance_ratio_[1]:.1%} wariancji
• Łącznie: {sum(self.pca.explained_variance_ratio_):.1%} wariancji

🎯 JAKOŚĆ KLASTROWANIA:
• Inertia: {self.kmeans.inertia_:.2f}
• Separowalność: {'Dobra' if sum(self.pca.explained_variance_ratio_) > 0.5 else 'Umiarkowana'}
"""

        return fig, additional_text

    def analyze_cluster_substance_patterns(self, df: pd.DataFrame) -> str:
        """
        Analizuje wzorce używania substancji w klastrach
        
        Args:
            df: DataFrame z danymi
            
        Returns:
            Tekst analizy
        """
        if self.cluster_labels is None:
            return "Najpierw wykonaj klastrowanie!"

        # POPRAWKA: Użyj tych samych indeksów co przy klastrowaniu
        existing_personality_cols = [col for col in PERSONALITY_COLS if col in df.columns]
        features = df[existing_personality_cols].dropna()

        if len(self.cluster_labels) != len(features):
            return f"Błąd: Niezgodność długości cluster_labels ({len(self.cluster_labels)}) vs features ({len(features)})"

        # Użyj tylko wierszy które były w klastrowania
        cluster_substance_df = df.loc[features.index].copy()
        cluster_substance_df['Cluster'] = self.cluster_labels

        existing_substance_cols = [col for col in SUBSTANCE_COLS if col in df.columns]

        # Oblicz wskaźniki używania dla każdej kombinacji klaster-substancja
        results = []
        for cluster_id in range(self.n_clusters):
            cluster_data = cluster_substance_df[cluster_substance_df['Cluster'] == cluster_id]
            profile = self.cluster_profiles.get(cluster_id, {'name': f'Klaster {cluster_id}'})

            for substance in existing_substance_cols:
                usage_rate = (cluster_data[substance] > 0).mean()
                avg_intensity = cluster_data[substance].mean()
                results.append({
                    'Cluster': cluster_id,
                    'Cluster_Name': profile['name'],
                    'Substance': substance,
                    'Usage_Rate': usage_rate,
                    'Avg_Intensity': avg_intensity
                })

        results_df = pd.DataFrame(results)

        # Twórz analizę
        text = "🧬 === PROFILE KLASTRÓW vs UŻYWANIE SUBSTANCJI ===\n\n"

        for cluster_id in range(self.n_clusters):
            cluster_data = results_df[results_df['Cluster'] == cluster_id]
            profile = self.cluster_profiles.get(cluster_id, {
                'name': f'Klaster {cluster_id}',
                'emoji': '❓',
                'risk': 'Nieznane'
            })

            cluster_size = len(cluster_substance_df[cluster_substance_df['Cluster'] == cluster_id])

            text += f"{profile['emoji']} KLASTER {cluster_id}: {profile['name']}\n"
            text += f"🎯 Ryzyko: {profile['risk']} | 👥 Wielkość: {cluster_size} osób\n"
            text += "─" * 70 + "\n"

            # Top substancje dla tego klastra
            top_substances = cluster_data.nlargest(8, 'Usage_Rate')

            text += "🔝 TOP UŻYWANE SUBSTANCJE:\n"
            for _, row in top_substances.iterrows():
                usage_pct = row['Usage_Rate'] * 100
                intensity = row['Avg_Intensity']

                risk_emoji = self._get_risk_emoji(usage_pct)
                text += f"  {risk_emoji} {row['Substance']:<15} {usage_pct:>5.1f}% (średnia: {intensity:.2f})\n"

            # Wzorzec specyficzny dla klastra
            text += self._get_cluster_pattern_description(cluster_id)
            text += "\n"

        # Porównania między klastrami
        text += self._generate_cluster_comparisons(results_df, existing_substance_cols)

        # Podsumowanie ryzyka
        text += self._generate_risk_summary(results_df)

        return text

    def _interpret_trait_level(self, value: float) -> str:
        """Interpretuje poziom cechy osobowości"""
        if value > 0.5:
            return "🔴 BARDZO WYSOKIE"
        elif value > 0.2:
            return "🟠 WYSOKIE"
        elif value > -0.2:
            return "🟡 UMIARKOWANE"
        elif value > -0.5:
            return "🔵 NISKIE"
        else:
            return "🟣 BARDZO NISKIE"

    def _generate_cluster_comparison(self, features_df: pd.DataFrame,
                                     personality_cols: list) -> str:
        """Generuje porównanie klastrów"""
        text = "🔍 KLUCZOWE RÓŻNICE MIĘDZY KLASTRAMI:\n" + "=" * 50 + "\n"

        for cluster_id in range(self.n_clusters):
            cluster_data = features_df[features_df['Cluster'] == cluster_id]
            cluster_means = cluster_data[personality_cols].mean()

            # Znajdź cechy wyróżniające
            sorted_features = sorted(cluster_means.items(), key=lambda x: abs(x[1]), reverse=True)
            top_features = sorted_features[:3]

            profile = self.cluster_profiles.get(cluster_id, {'name': f'Klaster {cluster_id}', 'emoji': '❓'})
            text += f"\n{profile['emoji']} {profile['name']}:\n"

            for trait, value in top_features:
                direction = "⬆️ Wysokie" if value > 0 else "⬇️ Niskie"
                text += f"  • {direction} {trait} ({value:+.2f})\n"

        return text

    def _generate_risk_stratification(self, features_df: pd.DataFrame) -> str:
        """Generuje stratyfikację ryzyka"""
        text = f"\n🚨 STRATYFIKACJA RYZYKA:\n" + "=" * 30 + "\n"

        for cluster_id in range(self.n_clusters):
            cluster_size = len(features_df[features_df['Cluster'] == cluster_id])
            profile = self.cluster_profiles.get(cluster_id, {
                'name': f'Klaster {cluster_id}',
                'risk': 'Nieznane'
            })

            risk_emoji = {
                'Bardzo Wysokie': '🔴',
                'Wysokie': '🟠',
                'Umiarkowane': '🟡',
                'Niskie': '🟢'
            }.get(profile['risk'], '❓')

            text += f"{risk_emoji} {profile['risk'].upper()}: {profile['name']} ({cluster_size} osób)\n"

        return text

    def _get_risk_emoji(self, usage_pct: float) -> str:
        """Zwraca emoji ryzyka na podstawie odsetka używania"""
        if usage_pct > 70:
            return "🔴"
        elif usage_pct > 40:
            return "🟠"
        elif usage_pct > 20:
            return "🟡"
        else:
            return "🟢"

    def _get_cluster_pattern_description(self, cluster_id: int) -> str:
        """Zwraca opis wzorca dla danego klastra"""
        patterns = {
            0: "\n💡 WZORZEC: Social/Party drugs dominują\n🎉 Używanie społeczne, rekreacyjne, eksperymentowanie\n",
            1: "\n💡 WZORZEC: Hard drugs + self-medication\n🚨 Chaotyczne używanie, polydrug abuse, wysokie ryzyko\n",
            2: "\n💡 WZORZEC: Self-medication + comfort substances\n😔 Prywatne używanie, anxiety relief, habitual patterns\n",
            3: "\n💡 WZORZEC: Mainstream tylko, minimal use\n🛡️ Społecznie akceptowane, kontrolowane, odpowiedzialne\n"
        }
        return patterns.get(cluster_id, "\n💡 WZORZEC: Nieokreślony\n")

    def _generate_cluster_comparisons(self, results_df: pd.DataFrame,
                                      substance_cols: list) -> str:
        """Generuje porównania między klastrami"""
        text = "🔍 PORÓWNANIA MIĘDZY KLASTRAMI:\n" + "=" * 40 + "\n"

        # Znajdź substancje z największymi różnicami między klastrami
        for substance in substance_cols[:10]:  # Top 10 substancji
            if substance in results_df['Substance'].values:
                substance_data = results_df[results_df['Substance'] == substance]
                max_usage = substance_data['Usage_Rate'].max()
                min_usage = substance_data['Usage_Rate'].min()
                diff = max_usage - min_usage

                if diff > 0.3:  # Znacząca różnica
                    max_cluster = substance_data.loc[substance_data['Usage_Rate'].idxmax(), 'Cluster_Name']
                    min_cluster = substance_data.loc[substance_data['Usage_Rate'].idxmin(), 'Cluster_Name']
                    text += f"🎯 {substance}: {max_cluster} ({max_usage:.1%}) >> {min_cluster} ({min_usage:.1%})\n"

        return text

    def _generate_risk_summary(self, results_df: pd.DataFrame) -> str:
        """Generuje podsumowanie ryzyka"""
        text = f"\n🚨 PODSUMOWANIE RYZYKA:\n" + "=" * 25 + "\n"

        # Oblicz średnie wyniki ryzyka
        high_risk_substances = ['Heroin', 'Crack', 'Cocaine', 'Amphetamines', 'Benzodiazepines']

        for cluster_id in range(self.n_clusters):
            cluster_data = results_df[results_df['Cluster'] == cluster_id]
            risk_usage = cluster_data[cluster_data['Substance'].isin(high_risk_substances)]['Usage_Rate'].mean()

            profile = self.cluster_profiles.get(cluster_id, {'name': f'Klaster {cluster_id}', 'emoji': '❓'})
            text += f"{profile['emoji']} {profile['name']}: {risk_usage:.1%} hard drugs use\n"

        # Rekomendacje
        text += f"\n💡 REKOMENDACJE INTERWENCJI:\n" + "=" * 30 + "\n"
        text += "🔴 Klaster 1: Crisis intervention, dual diagnosis treatment\n"
        text += "🟠 Klaster 2: Anxiety treatment, social support building\n"
        text += "🟡 Klaster 0: Harm reduction, alternative activities\n"
        text += "🟢 Klaster 3: Reinforcement, peer leadership roles\n"

        return text