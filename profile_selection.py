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

# Parser für Selig/Lednicer .dat-Dateien

def parse_dat(filepath):
    # Dummy-Implementierung: Rückgabe fester Polare
    return {'Re': [], 'Cl': [], 'Cd': [], 'Cl_max': 1.2}


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
    # Alle .dat-Dateien im Ordner sammeln
    dat_files = [f for f in os.listdir(folder) if f.endswith('.dat')]
    # Falls keine Profile da sind, leeres Mapping zurückgeben
    if not dat_files:
        return {}

    profile_map = {}

    for i, ri in enumerate(r):
        # Re-Zahl für Segment
        w1   = bem_res['dF_S'][i]  # Platzhalter, eigentlich Geschwindigkeit
        s_opt = 1.0                # Siehe BEM-Modul
        Re_i  = cfg['tsr_scan']['rho'] * w1 * s_opt / mu

        scores = {}
        for dat in dat_files:
            data = parse_dat(os.path.join(folder, dat))
            # Interpolation L/D und Cl_max
            ld    = 100            # Dummy
            clmax = data['Cl_max']
            # Normieren
            # Hier: X_norm = (X - min) / (max - min)
            # Dummy: ld_norm=1, cl_norm=1
            ld_norm = 1.0
            cl_norm = 1.0
            # Score
            S = weights['ld'] * ld_norm + weights['clmax'] * cl_norm
            scores[dat] = S

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