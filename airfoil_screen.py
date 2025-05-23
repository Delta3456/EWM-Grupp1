# airfoil_screen.py

"""
airfoil_screen.py

Modul zur Vor-Selektion und Rangierung von Airfoil-Polaren aus einer großen Datenbank.

Schritte:
1. Meta-Filterung:
   - Basierend auf zulässigem Dickenverhältnis (t/c) und Reynolds-Zahl-Bereich.
2. Schnell-Rangierung:
   - Berechnung des maximalen Cl/Cd im relevanten Anstellwinkelbereich [-2°, 8°].
   - Auswahl der Top N Profile für detaillierte Simulation.

Verwendete Parameter aus config.yaml:
  - t_c_range: zulässiger Bereich des Dickenverhältnisses (z.B. [0.15, 0.25])
  - Re_range: zulässiger Reynolds-Zahlen-Bereich (z.B. [1e5, 1e7])
  - top_n: Anzahl der final ausgewählten Profile (Standard 50)
"""

import os
import numpy as np
from data_io import parse_airfoil_dat  # neu
import logging

def filter_meta(polars, t_c_range, Re_target, airfoil_dir):
    """
    Filtert anhand berechneten t/c aus den .dat-Shape-Dateien.
    Erwartet in polars nur Profilnamen, Metadaten werden hier ergänzt.
    """
    filtered = {}
    t_c_min, t_c_max = t_c_range
    for name in polars:
        dat_path = os.path.join(airfoil_dir, name + '.dat')
        try:
            meta = parse_airfoil_dat(dat_path)
        except FileNotFoundError:
            logging.warning(f"Kein .dat für {name}, überspringe Metafilter")
            filtered[name] = polars[name]
            continue
        t_c = meta['t_c']
        # Anwenden des Filters
        if t_c_min <= t_c <= t_c_max:
            # Metadaten injizieren
            data = polars[name]
            data['t_c'] = t_c
            data['x_t_max'] = meta['x_t_max']
            data['camber_max'] = meta['camber_max']
            data['x_c_max'] = meta['x_c_max']
            filtered[name] = data
    return filtered

def quick_rank(polars, alpha_range=[-2, 8], top_n=50):
    """
    Führt eine schnelle Rangierung der Airfoil-Profile durch.

    Berechnet für jedes Profil das maximale Cl/Cd-Verhältnis im Anstellwinkel-Bereich.

    Parameter:
      polars     : dict, Mapping airfoil_name -> Polar-Daten inkl. 'alpha', 'Cl', 'Cd'
      alpha_range: Liste [alpha_min, alpha_max] in Grad
      top_n      : int, Anzahl der auszuwählenden Top-Profile

    Rückgabe:
      list of tuples: [(airfoil_name, score), ...] sortiert absteigend nach score
    """
    scores = []
    alpha_min, alpha_max = alpha_range
    for name, data in polars.items():
        alpha = np.array(data['alpha'])
        Cl    = np.array(data['Cl'])
        Cd    = np.array(data['Cd'])
        # Auswahl des relevanten alpha-Intervalls
        mask = (alpha >= alpha_min) & (alpha <= alpha_max)
        if not np.any(mask):
            continue
        Cl_sel = Cl[mask]
        Cd_sel = Cd[mask]
        # Berechnung von Cl/Cd, Vermeidung von Division durch Null
        with np.errstate(divide='ignore', invalid='ignore'):
            ratios = Cl_sel / Cd_sel
        max_ratio = np.nanmax(ratios)
        if np.isfinite(max_ratio):
            scores.append((name, max_ratio))
    # Sortiere absteigend nach max Cl/Cd und gebe Top_n zurück
    scores.sort(key=lambda x: x[1], reverse=True)
    return scores[:top_n]

def select_airfoils(polars, config, Re_target):
    """
    Komplette Screening-Pipeline:
      1. Meta-Filter
      2. Quick-Rangierung
      3. Rückgabe der Top-N Profilnamen

    Parameter:
      polars    : dict, alle geladenen Polare
      config    : dict, aus config.yaml geladene Parameter
      Re_target : float, Ziel-Reynolds-Zahl

    Rückgabe:
      list of str: Namen der ausgewählten Airfoil-Profile
    """
    # 1. Meta-Filter
    filtered = filter_meta(polars,
                           t_c_range=config['t_c_range'],
                           Re_target=Re_target)
    # 2. Quick-Ranking
    ranked = quick_rank(filtered,
                        alpha_range=[-2, 8],
                        top_n=config.get('top_n', 50))
    # 3. Extrahiere Profilnamen
    selected = [name for name, _ in ranked]
    return selected