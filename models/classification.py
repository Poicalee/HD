# -*- coding: utf-8 -*-
"""
Klasa do klasyfikacji używania substancji UCI Drug Consumption
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Tuple
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix
from sklearn.preprocessing import StandardScaler

from utils.constants import PERSONALITY_COLS, SUBSTANCE_COLS, COLORS
from utils.helpers import get_substance_category, get_performance_status

class ClassificationManager:
    """Klasa do zarządzania modelami klasyfikacji"""

    def __init__(self):
        self.scaler = StandardScaler()
        self.models = {}
        self.results_cache = {}

    def perform_classification(self, df: pd.DataFrame, substance: str) -> Dict:
        """
        Przeprowadza klasyfikację dla danej substancji
        
        Args:
            df: DataFrame z danymi
            substance: Nazwa substancji do przewidywania
            
        Returns:
            Słownik z wynikami analizy
        """
        try:
            # Sprawdzenie danych
            if substance not in df.columns:
                return {
                    'success': False,
                    'message': f"Brak danych dla substancji: {substance}",
                    'analysis_text': '',
                    'figure': None
                }

            # Przygotowanie danych
            existing_personality_cols = [col for col in PERSONALITY_COLS if col in df.columns]

            if len(existing_personality_cols) < 3:
                return {
                    'success': False,
                    'message': "Za mało cech osobowości do klasyfikacji",
                    'analysis_text': '',
                    'figure': None
                }

            features = df[existing_personality_cols].dropna()
            target = (df.loc[features.index, substance] > 0).astype(int)

            # Sprawdzenie rozkładu klas
            usage_count = target.sum()
            non_usage_count = len(target) - usage_count

            if usage_count < 10 or non_usage_count < 10:
                return {
                    'success': False,
                    'message': f"Za mało danych dla {substance}. Potrzeba minimum 10 osób w każdej grupie.",
                    'analysis_text': '',
                    'figure': None
                }

            # Podział danych
            X_train, X_test, y_train, y_test = train_test_split(
                features, target, test_size=0.3, random_state=42, stratify=target
            )

            # Trenowanie modelu
            rf = RandomForestClassifier(n_estimators=100, random_state=42)
            rf.fit(X_train, y_train)

            # Predykcja
            y_pred = rf.predict(X_test)
            y_pred_proba = rf.predict_proba(X_test)[:, 1]

            # Metryki
            accuracy = accuracy_score(y_test, y_pred)
            baseline = max(target.mean(), 1 - target.mean())
            improvement = accuracy - baseline

            try:
                auc_score = roc_auc_score(y_test, y_pred_proba)
            except:
                auc_score = 0.5

            conf_matrix = confusion_matrix(y_test, y_pred)

            # Ważność cech
            feature_importance = pd.DataFrame({
                'Feature': existing_personality_cols,
                'Importance': rf.feature_importances_
            }).sort_values('Importance', ascending=False)

            # Przygotowanie wyników
            analysis_text = self._generate_classification_report(
                substance, features, target, accuracy, baseline, improvement,
                auc_score, conf_matrix, feature_importance, X_train, X_test
            )

            figure = self._create_classification_plots(
                feature_importance, accuracy, baseline, substance, auc_score
            )

            # Cache wyników
            self.results_cache[substance] = {
                'accuracy': accuracy,
                'baseline': baseline,
                'improvement': improvement,
                'auc': auc_score,
                'feature_importance': feature_importance,
                'usage_rate': target.mean()
            }

            return {
                'success': True,
                'message': 'Klasyfikacja zakończona pomyślnie',
                'analysis_text': analysis_text,
                'figure': figure
            }

        except Exception as e:
            return {
                'success': False,
                'message': f"Błąd klasyfikacji: {str(e)}",
                'analysis_text': '',
                'figure': None
            }

    def compare_all_substances(self, df: pd.DataFrame) -> Dict:
        """
        Porównuje klasyfikacje wszystkich substancji
        
        Args:
            df: DataFrame z danymi
            
        Returns:
            Słownik z wynikami porównania
        """
        results = []
        existing_personality_cols = [col for col in PERSONALITY_COLS if col in df.columns]
        existing_substance_cols = [col for col in SUBSTANCE_COLS if col in df.columns]

        if len(existing_personality_cols) < 3:
            return {
                'analysis_text': "Za mało cech osobowości do analizy klasyfikacji",
                'figure': None
            }

        features = df[existing_personality_cols].dropna()

        for substance in existing_substance_cols:
            target = (df.loc[features.index, substance] > 0).astype(int)
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
                    'Status': 'Insufficient data',
                    'Category': get_substance_category(substance)
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
                    'Feature': existing_personality_cols,
                    'Importance': rf.feature_importances_
                }).sort_values('Importance', ascending=False)

                top_feature = feature_importance.iloc[0]['Feature']
                status = get_performance_status(accuracy, baseline, auc_score)

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
                    'Status': status,
                    'Category': get_substance_category(substance)
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
                    'Status': '❌ Error',
                    'Category': get_substance_category(substance)
                })

        results_df = pd.DataFrame(results)

        # POPRAWKA: Użyj dropna() zamiast na_last (kompatybilność z starszymi wersjami pandas)
        # Najpierw usuń wiersze z NaN w kolumnie Improvement
        results_df_clean = results_df.dropna(subset=['Improvement'])
        results_df_nan = results_df[results_df['Improvement'].isna()]

        # Sortuj czyste dane
        results_df_clean = results_df_clean.sort_values('Improvement', ascending=False)

        # Połącz z powrotem (NaN na końcu)
        results_df = pd.concat([results_df_clean, results_df_nan], ignore_index=True)

        analysis_text = self._generate_comparison_report(results_df)
        figure = self._create_comparison_plots(results_df)

        return {
            'analysis_text': analysis_text,
            'figure': figure
        }

    def _generate_classification_report(self, substance: str, features: pd.DataFrame,
                                        target: pd.Series, accuracy: float, baseline: float,
                                        improvement: float, auc_score: float,
                                        conf_matrix: np.ndarray, feature_importance: pd.DataFrame,
                                        X_train: pd.DataFrame, X_test: pd.DataFrame) -> str:
        """Generuje raport klasyfikacji dla pojedynczej substancji"""

        text = f"""
🌲 === KLASYFIKACJA RANDOM FOREST - {substance.upper()} ===

🎯 ZADANIE: Przewidywanie używania {substance} na podstawie cech osobowości

📊 PODSTAWOWE INFORMACJE:
├── Całkowita próba: {len(features)} osób
├── 🟢 Używa substancji: {target.sum()} osób ({target.mean() * 100:.1f}%)  
├── 🔴 Nie używa: {len(target) - target.sum()} osób ({(1 - target.mean()) * 100:.1f}%)
├── 📚 Zbiór treningowy: {len(X_train)} obserwacji
└── 🧪 Zbiór testowy: {len(X_test)} obserwacji

🏆 WYNIKI MODELU:
├── 🎯 Dokładność (Accuracy): {accuracy:.3f} ({accuracy * 100:.1f}%)
├── 📈 AUC-ROC: {auc_score:.3f}
├── 📊 Baseline (najczęstsza klasa): {baseline:.3f} ({baseline * 100:.1f}%)
└── ⬆️ Poprawa nad baseline: {improvement:+.3f} ({improvement * 100:+.1f} pkt proc.)

🎭 MACIERZ KONFUZJI:
                 Przewidywane
Rzeczywiste    Nie używa  Używa
Nie używa         {conf_matrix[0, 0]:>3}      {conf_matrix[0, 1]:>3}
Używa             {conf_matrix[1, 0]:>3}      {conf_matrix[1, 1]:>3}

🏆 RANKING WAŻNOŚCI CECH:
"""

        for _, row in feature_importance.iterrows():
            text += f"{row['Feature']:<18} {row['Importance']:.4f}\n"

        # Ocena wydajności
        status = get_performance_status(accuracy, baseline, auc_score)
        text += f"\n📋 OCENA WYDAJNOŚCI: {status}\n"

        # Interpretacja
        category = get_substance_category(substance)
        text += f"\n💡 INTERPRETACJA:\n"
        text += f"• Kategoria substancji: {category}\n"
        text += f"• Przewidywalność: {'Wysoka' if improvement > 0.1 else 'Umiarkowana' if improvement > 0.05 else 'Niska'}\n"
        text += f"• Najważniejsza cecha: {feature_importance.iloc[0]['Feature']}\n"

        if improvement > 0.1:
            text += "✅ Model dobrze przewiduje używanie tej substancji na podstawie osobowości\n"
        elif improvement > 0.05:
            text += "⚠️ Model ma umiarkowaną zdolność przewidywania\n"
        else:
            text += "❌ Cechy osobowości słabo przewidują używanie tej substancji\n"

        return text

    def _generate_comparison_report(self, results_df: pd.DataFrame) -> str:
        """Generuje raport porównawczy wszystkich substancji"""

        text = "🏆 === RANKING PRZEWIDYWALNOŚCI WSZYSTKICH SUBSTANCJI ===\n\n"
        text += "🤖 Analiza każdej substancji Random Forest...\n\n"

        text += "🏆 RANKING PRZEWIDYWALNOŚCI:\n"
        text += "=" * 80 + "\n"
        text += f"{'Rang':<4} {'Substancja':<15} {'Użyt.':<6} {'%Użyt':<6} {'Dokł.':<6} {'+Base':<6} {'AUC':<6} {'Status':<12} {'Top Cecha':<15}\n"
        text += "=" * 80 + "\n"

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

            text += f"{i:<4} {row['Substance']:<15} {row['Users']:<6} {usage_pct:<6} {accuracy_str:<6} {improvement_str:<6} {auc_str:<6} {row['Status']:<12} {row['Top_Feature']:<15}\n"

        # Analiza według kategorii
        valid_results = results_df[~pd.isna(results_df['Accuracy'])]

        if len(valid_results) > 0:
            text += "\n📊 ANALIZA WEDŁUG KATEGORII:\n"
            text += "=" * 40 + "\n"

            categories = valid_results['Category'].unique()
            for category in categories:
                cat_data = valid_results[valid_results['Category'] == category]
                avg_improvement = cat_data['Improvement'].mean()
                avg_auc = cat_data['AUC'].mean()
                count = len(cat_data)

                cat_emoji = {
                    'Legal': '🟢',
                    'Soft Illegal': '🟡',
                    'Stimulants': '🟠',
                    'Hard Drugs': '🔴'
                }.get(category, '❓')

                text += f"{cat_emoji} {category}: {count} substancji, średnie +{avg_improvement:.3f} improvement, AUC {avg_auc:.3f}\n"

            # Top predyktywne cechy
            feature_counts = valid_results['Top_Feature'].value_counts()
            text += f"\n🎯 NAJCZĘŚCIEJ NAJWAŻNIEJSZE CECHY:\n"
            text += "─" * 35 + "\n"
            for feature, count in feature_counts.head(5).items():
                if feature != 'Insufficient data' and not feature.startswith('Error'):
                    text += f"🏆 {feature}: {count} substancji ({count / len(valid_results) * 100:.0f}%)\n"

            # Podsumowanie wydajności
            excellent_count = len(valid_results[valid_results['Status'].str.contains('Excellent')])
            good_count = len(valid_results[valid_results['Status'].str.contains('Good')])

            text += f"\n📈 PODSUMOWANIE WYDAJNOŚCI:\n"
            text += "─" * 25 + "\n"
            text += f"🟢 Excellent models: {excellent_count}\n"
            text += f"🟡 Good models: {good_count}\n"
            text += f"📊 Średnia poprawa: +{valid_results['Improvement'].mean():.3f}\n"
            text += f"📊 Średnie AUC: {valid_results['AUC'].mean():.3f}\n"

        return text

    def _create_classification_plots(self, feature_importance: pd.DataFrame,
                                     accuracy: float, baseline: float,
                                     substance: str, auc_score: float) -> plt.Figure:
        """Tworzy wykresy dla klasyfikacji pojedynczej substancji"""

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        # Wykres ważności cech
        bars = ax1.barh(feature_importance['Feature'], feature_importance['Importance'],
                        color=COLORS['cluster_colors'], alpha=0.8)
        ax1.set_xlabel('Ważność cechy', fontweight='bold')
        ax1.set_title(f'🌲 Ważność Cech - {substance}', fontweight='bold')
        ax1.grid(True, alpha=0.3, axis='x')

        # Porównanie wydajności
        categories = ['Baseline', 'Random Forest']
        scores = [baseline, accuracy]
        colors = [COLORS['danger'], COLORS['success']]

        bars2 = ax2.bar(categories, scores, color=colors, alpha=0.8)
        ax2.set_ylabel('Dokładność', fontweight='bold')
        ax2.set_title('📊 Porównanie Wydajności', fontweight='bold')
        ax2.set_ylim(0, 1)
        ax2.grid(True, alpha=0.3, axis='y')

        # Dodaj etykiety
        for bar, score in zip(bars2, scores):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width() / 2., height + 0.01,
                     f'{score:.1%}', ha='center', va='bottom', fontweight='bold')

        # Dodaj AUC score
        ax2.text(0.5, 0.5, f'AUC: {auc_score:.3f}', transform=ax2.transAxes,
                 ha='center', va='center', fontweight='bold', fontsize=12,
                 bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

        plt.tight_layout()
        return fig

    def _create_comparison_plots(self, results_df: pd.DataFrame) -> Optional[plt.Figure]:
        """Tworzy wykresy porównawcze wszystkich substancji"""

        valid_results = results_df[~pd.isna(results_df['Accuracy'])]

        if len(valid_results) == 0:
            return None

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 12))

        # Scatter plot: Usage Rate vs Predictability
        category_colors = {
            'Legal': COLORS['success'],
            'Soft Illegal': COLORS['warning'],
            'Stimulants': COLORS['secondary'],
            'Hard Drugs': COLORS['danger']
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

        # Dodaj etykiety substancji
        for _, row in valid_results.iterrows():
            ax1.annotate(row['Substance'],
                         (row['Usage_Rate'] * 100, row['AUC'] * 100),
                         xytext=(5, 5), textcoords='offset points',
                         fontsize=8, alpha=0.7)

        # Częstość najważniejszych cech
        feature_counts = valid_results['Top_Feature'].value_counts()
        # Filtruj błędy i brak danych
        feature_counts = feature_counts[
            ~feature_counts.index.str.contains('Error|Insufficient', na=False)
        ]

        if len(feature_counts) > 0:
            bars = ax2.barh(range(len(feature_counts)), feature_counts.values,
                            color=COLORS['cluster_colors'][:len(feature_counts)])
            ax2.set_yticks(range(len(feature_counts)))
            ax2.set_yticklabels(feature_counts.index)
            ax2.set_xlabel('Liczba substancji', fontweight='bold')
            ax2.set_title('🏆 Najczęściej Najważniejsze Cechy Osobowości', fontweight='bold')
            ax2.grid(True, alpha=0.3, axis='x')

            # Dodaj etykiety wartości
            for i, (bar, count) in enumerate(zip(bars, feature_counts.values)):
                width = bar.get_width()
                ax2.text(width + 0.1, bar.get_y() + bar.get_height() / 2,
                         f'{count}', ha='left', va='center', fontweight='bold')

        plt.tight_layout()
        return fig