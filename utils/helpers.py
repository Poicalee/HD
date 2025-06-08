# -*- coding: utf-8 -*-
"""
Funkcje pomocnicze dla aplikacji UCI Drug Consumption
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, Tuple
from .constants import MAPPING_TOLERANCE, SUBSTANCE_CATEGORIES

def find_closest_key(value: float, mapping_dict: Dict[float, str],
                     tolerance: float = MAPPING_TOLERANCE) -> str:
    """
    Znajduje najbliższy klucz w mapowaniu dla danej wartości liczbowej
    
    Args:
        value: Wartość liczbowa do zmapowania
        mapping_dict: Słownik mapowań {wartość_liczbowa: kategoria}
        tolerance: Tolerancja dla dopasowania
        
    Returns:
        Kategoria tekstowa lub 'Unknown' jeśli nie znaleziono
    """
    if pd.isna(value):
        return 'Unknown'

    try:
        value = float(value)
        closest_key = min(mapping_dict.keys(), key=lambda x: abs(x - value))

        # Sprawdź czy różnica jest w tolerancji
        if abs(closest_key - value) <= tolerance:
            return mapping_dict[closest_key]
        else:
            return 'Unknown'

    except (ValueError, TypeError):
        return 'Unknown'

def validate_dataframe(df: pd.DataFrame, expected_cols: list) -> Tuple[bool, str]:
    """
    Waliduje strukturę DataFrame
    
    Args:
        df: DataFrame do walidacji
        expected_cols: Lista oczekiwanych kolumn
        
    Returns:
        (is_valid, error_message)
    """
    if df is None or df.empty:
        return False, "Brak danych lub pusty DataFrame"

    missing_cols = set(expected_cols) - set(df.columns)
    if missing_cols:
        return False, f"Brakujące kolumny: {', '.join(missing_cols)}"

    if len(df) < 10:
        return False, "Za mało rekordów (minimum 10)"

    return True, "OK"

def get_substance_category(substance: str) -> str:
    """
    Zwraca kategorię substancji
    
    Args:
        substance: Nazwa substancji
        
    Returns:
        Kategoria substancji
    """
    for category, substances in SUBSTANCE_CATEGORIES.items():
        if substance in substances:
            return category
    return 'Other'

def get_risk_color(value: float, reverse: bool = False) -> str:
    """
    Zwraca kolor na podstawie poziomu ryzyka
    
    Args:
        value: Wartość do oceny
        reverse: Czy odwrócić skalę kolorów
        
    Returns:
        Kod koloru hex
    """
    from .constants import COLORS

    if reverse:
        if value > 0.3:
            return COLORS['risk_gradient'][0]  # Green for high positive
        elif value > 0:
            return COLORS['risk_gradient'][1]  # Orange for moderate
        else:
            return COLORS['risk_gradient'][2]  # Red for negative
    else:
        if value > 0.3:
            return COLORS['risk_gradient'][2]  # Red for high risk
        elif value > 0:
            return COLORS['risk_gradient'][1]  # Orange for moderate
        else:
            return COLORS['risk_gradient'][0]  # Green for low risk

def safe_numeric_conversion(series: pd.Series,
                            mapping: Optional[Dict] = None) -> pd.Series:
    """
    Bezpieczna konwersja serii na wartości numeryczne
    
    Args:
        series: Seria do konwersji
        mapping: Opcjonalne mapowanie wartości
        
    Returns:
        Przekonwertowana seria
    """
    try:
        if mapping:
            # Zastosuj mapowanie jeśli podane
            series = series.map(mapping)

        # Konwertuj na numeric, błędne wartości jako NaN
        return pd.to_numeric(series, errors='coerce')

    except Exception:
        return series

def calculate_usage_stats(df: pd.DataFrame, substance_cols: list) -> Dict[str, Dict]:
    """
    Oblicza statystyki używania substancji
    
    Args:
        df: DataFrame z danymi
        substance_cols: Lista kolumn substancji
        
    Returns:
        Słownik ze statystykami
    """
    stats = {}

    for substance in substance_cols:
        if substance in df.columns:
            total_users = (df[substance] > 0).sum()
            heavy_users = (df[substance] >= 5).sum()  # Recent use
            usage_rate = total_users / len(df) * 100
            heavy_rate = heavy_users / len(df) * 100
            avg_intensity = df[substance].mean()

            stats[substance] = {
                'total_users': total_users,
                'heavy_users': heavy_users,
                'usage_rate': usage_rate,
                'heavy_rate': heavy_rate,
                'avg_intensity': avg_intensity,
                'category': get_substance_category(substance)
            }

    return stats

def format_percentage(value: float, decimals: int = 1) -> str:
    """
    Formatuje wartość jako procent
    
    Args:
        value: Wartość do sformatowania (0-1)
        decimals: Liczba miejsc po przecinku
        
    Returns:
        Sformatowany string z procentem
    """
    if pd.isna(value):
        return "N/A"
    return f"{value * 100:.{decimals}f}%"

def create_demographic_summary(df: pd.DataFrame,
                               demographic_cols: list) -> str:
    """
    Tworzy podsumowanie demograficzne
    
    Args:
        df: DataFrame z danymi
        demographic_cols: Lista kolumn demograficznych
        
    Returns:
        Sformatowane podsumowanie tekstowe
    """
    summary = []

    for col in demographic_cols:
        category_col = f"{col}_Category"
        if category_col in df.columns:
            dist = df[category_col].value_counts()
            summary.append(f"\n{col.upper()}:")
            for category, count in dist.head(5).items():
                pct = count / len(df) * 100
                summary.append(f"  {category}: {count} ({pct:.1f}%)")

    return "\n".join(summary)

def detect_outliers(series: pd.Series, method: str = 'iqr') -> pd.Series:
    """
    Wykrywa wartości odstające
    
    Args:
        series: Seria do analizy
        method: Metoda wykrywania ('iqr' lub 'zscore')
        
    Returns:
        Boolean series z True dla outliers
    """
    if method == 'iqr':
        Q1 = series.quantile(0.25)
        Q3 = series.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        return (series < lower_bound) | (series > upper_bound)

    elif method == 'zscore':
        z_scores = np.abs((series - series.mean()) / series.std())
        return z_scores > 3

    else:
        raise ValueError("Method must be 'iqr' or 'zscore'")

def safe_division(numerator: float, denominator: float,
                  default: float = 0.0) -> float:
    """
    Bezpieczne dzielenie z obsługą dzielenia przez zero
    
    Args:
        numerator: Licznik
        denominator: Mianownik
        default: Wartość domyślna przy dzieleniu przez zero
        
    Returns:
        Wynik dzielenia lub wartość domyślna
    """
    try:
        if denominator == 0:
            return default
        return numerator / denominator
    except (TypeError, ZeroDivisionError):
        return default

def get_performance_status(accuracy: float, baseline: float,
                           auc: float) -> str:
    """
    Określa status wydajności modelu
    
    Args:
        accuracy: Dokładność modelu
        baseline: Baseline accuracy
        auc: AUC score
        
    Returns:
        Status z emoji
    """
    improvement = accuracy - baseline

    if improvement > 0.15 and auc > 0.8:
        return "🟢 Excellent"
    elif improvement > 0.08 and auc > 0.7:
        return "🟡 Good"
    elif improvement > 0.03 and auc > 0.6:
        return "🟠 Fair"
    else:
        return "🔴 Poor"