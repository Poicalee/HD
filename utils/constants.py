# -*- coding: utf-8 -*-
"""
Stałe, mapowania i konfiguracja dla aplikacji UCI Drug Consumption
"""

# Kolumny danych
DEMOGRAPHIC_COLS = ['Age', 'Gender', 'Education', 'Country', 'Ethnicity']
PERSONALITY_COLS = ['Neuroticism', 'Extraversion', 'Openness', 'Agreeableness',
                    'Conscientiousness', 'Impulsiveness', 'SensationSeeking']
SUBSTANCE_COLS = ['Alcohol', 'Amphetamines', 'AmylNitrite', 'Benzodiazepines',
                  'Cannabis', 'Chocolate', 'Cocaine', 'Caffeine', 'Crack',
                  'Ecstasy', 'Heroin', 'Ketamine', 'LegalHighs', 'LSD',
                  'Methadone', 'Mushrooms', 'Nicotine', 'Semer', 'VSA']

# Nazwy kolumn dla pliku .data
UCI_COLUMN_NAMES = (['ID', 'Age', 'Gender', 'Education', 'Country', 'Ethnicity'] +
                    PERSONALITY_COLS + SUBSTANCE_COLS)

# Mapowania wartości liczbowych na kategorie
AGE_MAPPING = {
    -0.95197: '18-24', -0.07854: '25-34', 0.49788: '35-44',
    1.09449: '45-54', 1.82213: '55-64', 2.59171: '65+'
}

GENDER_MAPPING = {
    0.48246: 'Female', -0.48246: 'Male'
}

EDUCATION_MAPPING = {
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

COUNTRY_MAPPING = {
    -0.09765: 'Australia', 0.24923: 'Canada', -0.46841: 'New Zealand',
    -0.28519: 'Other', 0.21128: 'Republic of Ireland',
    0.96082: 'UK', -0.57009: 'USA'
}

ETHNICITY_MAPPING = {
    -0.50212: 'Asian', -1.10702: 'Black', 1.90725: 'Mixed-Black/Asian',
    0.12600: 'Mixed-White/Asian', -0.22166: 'Mixed-White/Black',
    0.11440: 'Other', -0.31685: 'White'
}

# Mapowanie konsumpcji substancji
CONSUMPTION_MAPPING = {
    'CL0': 0,  # Never used
    'CL1': 1,  # Used over a decade ago
    'CL2': 2,  # Used in last decade
    'CL3': 3,  # Used in last year
    'CL4': 4,  # Used in last month
    'CL5': 5,  # Used in last week
    'CL6': 6   # Used in last day
}

# Kolory interfejsu
COLORS = {
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

# Profile klastrów osobowości
CLUSTER_PROFILES = {
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
        'name': 'Stabilni Konserwatyści',
        'emoji': '🛡️',
        'color': '#2ECC71',
        'risk': 'Niskie',
        'description': 'Zdyscyplinowani, kontrolowani, tradycyjni'
    }
}

# Kategorie substancji
SUBSTANCE_CATEGORIES = {
    'Legal': ['Alcohol', 'Caffeine', 'Chocolate', 'Nicotine'],
    'Soft Illegal': ['Cannabis', 'LSD', 'Mushrooms'],
    'Stimulants': ['Cocaine', 'Ecstasy', 'Amphetamines'],
    'Hard Drugs': ['Heroin', 'Crack', 'Methadone', 'Benzodiazepines']
}

# Opcje filtrów
FILTER_OPTIONS = [
    'Wszystkie dane',
    'Tylko mężczyźni', 'Tylko kobiety',
    'Wiek 18-24', 'Wiek 25-34', 'Wiek 35-44',
    'Wysokie wykształcenie', 'UK/USA/Canada',
    'Używający Cannabis', 'Używający Alcohol',
    'Nieużywający narkotyków'
]

# Konfiguracja GUI
GUI_CONFIG = {
    'title': '🧠 Analiza Wzorców Konsumpcji Narkotyków - UCI Dataset | Karol Dąbrowski',
    'geometry': '1920x1080',
    'bg_color': '#f8f9fa'
}

# Tolerancja dla mapowania wartości
MAPPING_TOLERANCE = 0.01