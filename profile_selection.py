#!/usr/bin/env python3
"""
profile_selection.py

Modul 4: Auswahl von Profilen basierend auf Score-Methode
  - Liest BEM-Ergebnisse und .dat-Dateien für Profile
  - Berechnet Re-Zahlen pro Segment
  - Interpoliert L/D- und C_l,max-Werte
  - Normiert Kennwerte und berechnet Score
  - Wählt bestes Profil pro Segment aus
"""
import argparse
import logging
import json
import yaml
import os
import numpy as np
import sys
import warnings
from datetime import datetime
from pathlib import Path

def parse_dat(filepath):
    """
    Parst eine Selig/Lednicer .dat-Datei und extrahiert Profilinformationen.
    
    Args:
        filepath: Pfad zur .dat-Datei
        
    Returns:
        dict mit Profilinformationen:
        - name: Airfoil-Name
        - n_points: Anzahl der Koordinatenpunkte
        - max_camber: (x, y) Position und Wert des maximalen Cambers
        - max_thickness: (x, y) Position und Wert der maximalen Dicke
        - coordinates: Liste von (x,y) Koordinatenpaaren
    """
    with open(filepath, 'r') as f:
        lines = f.readlines()
    
    # Erste Zeile ist der Name
    name = lines[0].strip()
    
    # Koordinaten parsen
    coords = []
    for line in lines[1:]:
        try:
            x, y = map(float, line.strip().split())
            # Validierung der Koordinaten
            if not (-0.01 <= x <= 1.01):
                warnings.warn(f"X-Koordinate {x} außerhalb [-0.01, 1.01] in {filepath}")
            if not (-1.0 <= y <= 1.0):
                warnings.warn(f"Y-Koordinate {y} außerhalb [-1.0, 1.0] in {filepath}")
            coords.append((x, y))
        except ValueError:
            continue
    
    coords = np.array(coords)
    
    # Maximalen Camber berechnen
    camber_line = []
    for i in range(len(coords)-1):
        x1, y1 = coords[i]
        x2, y2 = coords[i+1]
        if x1 == x2:
            continue
        x = (x1 + x2) / 2
        y = (y1 + y2) / 2
        camber_line.append((x, y))
    
    camber_line = np.array(camber_line)
    max_camber_idx = np.argmax(camber_line[:, 1])
    max_camber = (camber_line[max_camber_idx, 0], camber_line[max_camber_idx, 1])
    
    # Maximale Dicke berechnen
    thickness = []
    for i in range(len(coords)-1):
        x1, y1 = coords[i]
        x2, y2 = coords[i+1]
        if x1 == x2:
            continue
        x = (x1 + x2) / 2
        t = abs(y2 - y1)
        thickness.append((x, t))
    
    thickness = np.array(thickness)
    max_thickness_idx = np.argmax(thickness[:, 1])
    max_thickness = (thickness[max_thickness_idx, 0], thickness[max_thickness_idx, 1])
    
    return {
        'name': name,
        'n_points': len(coords),
        'max_camber': max_camber,
        'max_thickness': max_thickness,
        'coordinates': coords.tolist()
    }

def get_profiles(profile_dir):
    """
    Lädt alle Profile aus dem Verzeichnis mit JSON-Cache.
    
    Args:
        profile_dir: Pfad zum Profil-Verzeichnis
        
    Returns:
        dict mit allen Profilinformationen
    """
    profile_dir = Path(profile_dir)
    cache_file = profile_dir / 'profile_cache.json'
    
    # Prüfe ob Cache existiert
    if cache_file.exists():
        with open(cache_file, 'r') as f:
            cache = json.load(f)
            
        # Prüfe ob Cache aktuell ist
        cache_time = datetime.fromisoformat(cache['_built_at'])
        newest_dat = max(
            datetime.fromtimestamp(p.stat().st_mtime)
            for p in profile_dir.glob('*.dat')
        )
        
        if cache_time > newest_dat:
            return cache['profiles']
    
    # Cache ist veraltet oder existiert nicht - neu aufbauen
    profiles = {}
    for dat_file in profile_dir.glob('*.dat'):
        profiles[dat_file.name] = parse_dat(dat_file)
    
    # Cache speichern
    cache = {
        '_built_at': datetime.now().isoformat(),
        'profiles': profiles
    }
    with open(cache_file, 'w') as f:
        json.dump(cache, f, indent=2)
    
    return profiles

def load_config(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def load_bem(path):
    with open(path) as f:
        return json.load(f)


def select_profiles(bem_res, cfg):
    mu = cfg['profile_selection']['mu_air']
    weights = cfg['profile_selection']['score_weights']
    folder = cfg['profile_selection']['profile_folder']
    r = bem_res['r']
    
    # Profile mit Cache laden
    profiles = get_profiles(folder)
    if not profiles:
        return {}

    profile_map = {}

    for i, ri in enumerate(r):
        # Re-Zahl für Segment
        w1   = bem_res['dF_S'][i]  # Platzhalter, eigentlich Geschwindigkeit
        s_opt = 1.0                # Siehe BEM-Modul
        Re_i  = cfg['tsr_scan']['rho'] * w1 * s_opt / mu

        scores = {}
        for name, profile in profiles.items():
            # Interpolation L/D und Cl_max
            ld    = 100            # Dummy
            clmax = 1.2           # Dummy
            # Normieren
            # Hier: X_norm = (X - min) / (max - min)
            # Dummy: ld_norm=1, cl_norm=1
            ld_norm = 1.0
            cl_norm = 1.0
            # Score
            S = weights['ld'] * ld_norm + weights['clmax'] * cl_norm
            scores[name] = S

        best = max(scores, key=scores.get)
        profile_map[i] = best

    return profile_map


def main():
    parser = argparse.ArgumentParser(description='Profilauswahl per Score')
    parser.add_argument('--config', required=True, help='config.yaml')
    parser.add_argument('--bem', required=True, help='bem_results.json')
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    try:
        cfg = load_config(args.config)
        bem_res = load_bem(args.bem)
        mapping = select_profiles(bem_res, cfg)
        out = cfg['output']['profiles']
        with open(out, 'w') as f:
            json.dump(mapping, f, indent=2)
        logging.info('Profilzuordnung gespeichert in %s', out)
    except Exception as e:
        logging.error('Fehler in profile_selection.py: %s', e)
        sys.exit(1)

if __name__ == '__main__':
    main()