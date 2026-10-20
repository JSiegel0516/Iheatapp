"""Northern Hemisphere season labels for NOAA's overlapping ONI periods."""
import re

LABELS = {
    'JFM': 'January–March', 'FMA': 'February–April',
    'MAM': 'spring (March–May)', 'AMJ': 'April–June',
    'MJJ': 'May–July', 'JJA': 'summer (June–August)',
    'JAS': 'July–September', 'ASO': 'August–October',
    'SON': 'fall (September–November)', 'OND': 'October–December',
}


def season_label(code, year):
    if code == 'DJF':
        return f'winter {year - 1}–{year} (December–February)'
    if code == 'NDJ':
        return f'November {year}–January {year + 1}'
    return f'{LABELS[code]} {year}'


def readable_seasons(text):
    return re.sub(r'\b(DJF|JFM|FMA|MAM|AMJ|MJJ|JJA|JAS|ASO|SON|OND|NDJ)\s+((?:19|20)\d{2})\b',
                  lambda m: season_label(m[1], int(m[2])), text)
