# -*- coding: utf-8 -*-
"""
Klasa do przetwarzania danych UCI Drug Consumption Dataset
"""

import pandas as pd
import numpy as np
from typing import Optional, Tuple, Dict
from sklearn.preprocessing import StandardScaler
from sklearn.impute import KNNImputer

from utils.constants import *
from utils.helpers import find_closest_key, validate_dataframe, safe_numeric_conversion

class DataProcessor:
    """Klasa do przetwarzania i czyszczenia danych UCI"""

    def __init__(self):
        self.scaler = StandardScaler()
        self.imputer = KNNImputer(n_neighbors=5)
        self.original_data = None
        self.processed_data = None

    def load_data(self, file_path: str) -> Tuple[bool, str, Optional[pd.DataFrame]]:
        """
        Wczytuje dane z pliku CSV lub .data
        
        Args:
            file_path: Ścieżka do pliku
            
        Returns:
            (success, message, dataframe)
        """
        try:
            # Określ typ pliku i wczytaj dane
            if file_path.endswith('.data'):
                df = pd.read_csv(file_path, header=None, names=UCI_COLUMN_NAMES)
            else:
                df = pd.read_csv(file_path)

            # Podstawowa walidacja
            if df.empty:
                return False, "Plik jest pusty", None

            # Sprawdź minimalną liczbę kolumn
            if len(df.columns) < 25:
                return False, f"Za mało kolumn ({len(df.columns)}). Oczekiwane: min. 25", None

            # Zapisz oryginalne dane
            self.original_data = df.copy()

            # Przetwórz dane
            success, message = self._process_initial_data(df)
            if not success:
                return False, message, None

            self.processed_data = df.copy()

            return True, f"✅ Wczytano {len(df)} rekordów, {len(df.columns)} kolumn", df

        except FileNotFoundError:
            return False, "Nie znaleziono pliku", None
        except pd.errors.EmptyDataError:
            return False, "Plik jest pusty", None
        except pd.errors.ParserError as e:
            return False, f"Błąd parsowania pliku: {str(e)}", None
        except Exception as e:
            return False, f"Nieoczekiwany błąd: {str(e)}", None

    def _process_initial_data(self, df: pd.DataFrame) -> Tuple[bool, str]:
        """
        Wstępne przetwarzanie danych
        
        Args:
            df: DataFrame do przetworzenia
            
        Returns:
            (success, message)
        """
        try:
            # Debugowanie - sprawdź kolumny
            print(f"Kolumny w DataFrame: {list(df.columns)}")
            print(f"Pierwsze 3 wiersze:\n{df.head(3)}")

            # 1. Konwersja substancji na wartości numeryczne
            self._convert_substance_data(df)

            # 2. Dekodowanie zmiennych demograficznych - ZAWSZE
            self._decode_demographics(df)

            # 3. Walidacja cech osobowości
            self._validate_personality_data(df)

            return True, "Dane przetworzone pomyślnie"

        except Exception as e:
            print(f"Błąd w _process_initial_data: {str(e)}")
            return False, f"Błąd przetwarzania: {str(e)}"

    def _convert_substance_data(self, df: pd.DataFrame) -> None:
        """Konwertuje dane substancji na wartości numeryczne"""

        for col in SUBSTANCE_COLS:
            if col in df.columns:
                # Jeśli już numeryczne, zostaw jak jest
                if pd.api.types.is_numeric_dtype(df[col]):
                    continue

                # Konwertuj stringi CL0-CL6 na liczby
                df[col] = df[col].map(CONSUMPTION_MAPPING).fillna(df[col])

                # Konwertuj na numeric, błędne wartości jako NaN
                df[col] = pd.to_numeric(df[col], errors='coerce')

                # Wypełnij NaN zerem (nigdy nie używał)
                df[col] = df[col].fillna(0)

    def _decode_demographics(self, df: pd.DataFrame) -> None:
        """Dekoduje zmienne demograficzne na kategorie tekstowe"""

        mappings = {
            'Age': AGE_MAPPING,
            'Gender': GENDER_MAPPING,
            'Education': EDUCATION_MAPPING,
            'Country': COUNTRY_MAPPING,
            'Ethnicity': ETHNICITY_MAPPING
        }

        for col, mapping in mappings.items():
            if col in df.columns:
                try:
                    # Sprawdź czy kolumna już nie jest zdekodowana
                    category_col = f"{col}_Category"
                    if category_col in df.columns:
                        print(f"Kolumna {category_col} już istnieje, pomijam dekodowanie")
                        continue

                    # Konwertuj na numeric jeśli to stringi
                    if df[col].dtype == 'object':
                        df[col] = pd.to_numeric(df[col], errors='coerce')

                    # Aplikuj mapowanie
                    df[category_col] = df[col].apply(
                        lambda x: find_closest_key(x, mapping) if pd.notna(x) else 'Unknown'
                    )

                    print(f"Utworzono kolumnę {category_col}, unikalne wartości: {df[category_col].unique()}")

                except Exception as e:
                    print(f"Błąd dekodowania kolumny {col}: {str(e)}")
                    # Utwórz kolumnę z wartościami domyślnymi
                    df[f"{col}_Category"] = 'Unknown'
            else:
                print(f"Kolumna {col} nie istnieje w danych")
                # Utwórz kolumnę z wartościami domyślnymi
                df[f"{col}_Category"] = 'Unknown'

    def _validate_personality_data(self, df: pd.DataFrame) -> None:
        """Waliduje i czyści dane cech osobowości"""

        for col in PERSONALITY_COLS:
            if col in df.columns:
                # Konwertuj na numeric
                df[col] = pd.to_numeric(df[col], errors='coerce')

                # Sprawdź rozsądne zakresy (standardowe wyniki powinny być w [-3, 3])
                df.loc[df[col] > 5, col] = np.nan
                df.loc[df[col] < -5, col] = np.nan

    def apply_filter(self, df: pd.DataFrame, filter_option: str) -> pd.DataFrame:
        """
        Stosuje filtr do danych
        
        Args:
            df: DataFrame do filtrowania
            filter_option: Opcja filtra
            
        Returns:
            Przefiltrowany DataFrame
        """
        if filter_option == 'Wszystkie dane':
            return df.copy()

        try:
            if filter_option == 'Tylko mężczyźni':
                return df[df['Gender_Category'] == 'Male'].copy()

            elif filter_option == 'Tylko kobiety':
                return df[df['Gender_Category'] == 'Female'].copy()

            elif filter_option == 'Wiek 18-24':
                return df[df['Age_Category'] == '18-24'].copy()

            elif filter_option == 'Wiek 25-34':
                return df[df['Age_Category'] == '25-34'].copy()

            elif filter_option == 'Wiek 35-44':
                return df[df['Age_Category'] == '35-44'].copy()

            elif filter_option == 'Wysokie wykształcenie':
                high_edu = ['University degree', 'Masters degree', 'Doctorate degree']
                return df[df['Education_Category'].isin(high_edu)].copy()

            elif filter_option == 'UK/USA/Canada':
                countries = ['UK', 'USA', 'Canada']
                return df[df['Country_Category'].isin(countries)].copy()

            elif filter_option == 'Używający Cannabis':
                return df[df['Cannabis'] > 0].copy()

            elif filter_option == 'Używający Alcohol':
                return df[df['Alcohol'] > 0].copy()

            elif filter_option == 'Nieużywający narkotyków':
                illegal_substances = [col for col in SUBSTANCE_COLS
                                      if col not in ['Alcohol', 'Caffeine', 'Chocolate', 'Nicotine']]
                # Sprawdź które kolumny istnieją
                existing_illegal = [col for col in illegal_substances if col in df.columns]
                if existing_illegal:
                    mask = (df[existing_illegal] == 0).all(axis=1)
                    return df[mask].copy()
                else:
                    return df.copy()

            else:
                return df.copy()

        except KeyError as e:
            print(f"Błąd filtrowania - brak kolumny: {e}")
            return df.copy()
        except Exception as e:
            print(f"Błąd filtrowania: {e}")
            return df.copy()

    def standardize_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Standaryzuje cechy osobowości
        
        Args:
            df: DataFrame z danymi
            
        Returns:
            DataFrame ze standaryzowanymi cechami
        """
        df_copy = df.copy()

        # Znajdź istniejące kolumny osobowości
        existing_personality_cols = [col for col in PERSONALITY_COLS if col in df_copy.columns]

        if not existing_personality_cols:
            return df_copy

        # Standaryzuj
        personality_data = df_copy[existing_personality_cols].fillna(0)
        standardized_data = self.scaler.fit_transform(personality_data)

        # Dodaj standaryzowane kolumny
        for i, col in enumerate(existing_personality_cols):
            df_copy[f'{col}_std'] = standardized_data[:, i]

        return df_copy

    def handle_missing_values(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, int]]:
        """
        Obsługuje brakujące wartości
        
        Args:
            df: DataFrame z danymi
            
        Returns:
            (cleaned_dataframe, missing_count_dict)
        """
        df_copy = df.copy()
        missing_before = df_copy.isnull().sum().to_dict()

        # Znajdź istniejące kolumny osobowości
        existing_personality_cols = [col for col in PERSONALITY_COLS if col in df_copy.columns]

        if existing_personality_cols:
            # Użyj KNN imputation dla cech osobowości
            personality_data = df_copy[existing_personality_cols]
            if personality_data.isnull().sum().sum() > 0:
                imputed_data = self.imputer.fit_transform(personality_data)

                for i, col in enumerate(existing_personality_cols):
                    df_copy[col] = imputed_data[:, i]

        # Usuń duplikaty
        df_copy = df_copy.drop_duplicates()

        missing_after = df_copy.isnull().sum().to_dict()

        # Oblicz różnicę
        missing_filled = {col: missing_before.get(col, 0) - missing_after.get(col, 0)
                          for col in missing_before}

        return df_copy, missing_filled

    def create_binary_encoding(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Tworzy binarne kodowanie substancji (używa/nie używa)
        
        Args:
            df: DataFrame z danymi
            
        Returns:
            DataFrame z dodatkowymi kolumnami binarnymi
        """
        df_copy = df.copy()

        for col in SUBSTANCE_COLS:
            if col in df_copy.columns:
                df_copy[f'{col}_binary'] = (df_copy[col] > 0).astype(int)

        return df_copy

    def get_data_summary(self, df: pd.DataFrame) -> Dict[str, any]:
        """
        Tworzy podsumowanie danych
        
        Args:
            df: DataFrame do analizy
            
        Returns:
            Słownik z podsumowaniem
        """
        summary = {
            'shape': df.shape,
            'columns': list(df.columns),
            'missing_values': df.isnull().sum().to_dict(),
            'dtypes': df.dtypes.to_dict()
        }

        # Dodaj statystyki demograficzne
        demographic_summary = {}
        for col in DEMOGRAPHIC_COLS:
            category_col = f"{col}_Category"
            if category_col in df.columns:
                demographic_summary[col] = df[category_col].value_counts().to_dict()

        summary['demographics'] = demographic_summary

        # Dodaj statystyki substancji
        substance_summary = {}
        for col in SUBSTANCE_COLS:
            if col in df.columns:
                usage_count = (df[col] > 0).sum()
                substance_summary[col] = {
                    'users': int(usage_count),
                    'usage_rate': float(usage_count / len(df))
                }

        summary['substances'] = substance_summary

        return summary