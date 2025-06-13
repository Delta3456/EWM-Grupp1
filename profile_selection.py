#!/usr/bin/env python3
"""
profile_selection.py

Modul zur Parsing aller .dat-Aerofoil-Daten und Abspeichern als JSON.

- Liest config.yaml ein
- Durchsucht `profile_folder` nach .dat-Dateien
- Parset jede Datei nach Selig- oder Lednicer-Format
- Trennt Top- und Bottom-Surface
- Speichert alle rohen Polardaten in results/profiles_raw.json
"""

import argparse
import os
import json
import logging
import yaml


def load_config(path):
    """Lädt die YAML-Konfigurationsdatei."""
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def parse_dat(filepath):
    """
    Parst eine .dat-Datei nach Selig- oder Lednicer-Format.

    Regeln:
    - Erste Zeile: Name/Beschreibung
    - Leerzeilen werden übersprungen
    - Zweite Zeile:
        * Wenn beide ersten Werte > 1 → Lednicer-Format:
            - Erster Wert = Anzahl Top-Punkte
            - Zweiter Wert = Anzahl Bottom-Punkte
            - Danach genau (top + bottom) Koordinaten-Zeilen
        * Sonst → Selig-Format:
            - Alle folgenden Zeilen sind Koeffizientenpunkte
            - Ober- und Unterseite getrennt am minimalen X-Wert

    Gibt ein Dict zurück mit:
      name, format, top (Liste von {x,y}), bottom (Liste von {x,y})
    """
    with open(filepath, 'r') as f:
        # Rohzeilen ohne Leere
        lines = [l.strip() for l in f if l.strip()]
    name = lines[0]

    # Zweite Zeile tokens
    parts = lines[1].split()
    try:
        vals = [float(p) for p in parts]
    except ValueError:
        raise ValueError(f"Ungültige Zahlen in {filepath}: {lines[1]}")

    data_lines = lines[2:]
    # Lednicer-Format
    if vals[0] > 1 and vals[1] > 1:
        fmt = 'lednicer'
        top_n = int(vals[0])
        bot_n = int(vals[1])
        coords = []
        if len(data_lines) < top_n + bot_n:
            raise ValueError(f"{filepath}: nicht genügend Koordinaten für Lednicer-Format")
        # Lese Top
        for line in data_lines[:top_n]:
            x, y = map(float, line.split())
            coords.append(('top', x, y))
        # Lese Bottom
        for line in data_lines[top_n:top_n + bot_n]:
            x, y = map(float, line.split())
            coords.append(('bottom', x, y))

        top = [ {'x': x, 'y': y} for surf, x, y in coords if surf=='top' ]
        bottom = [ {'x': x, 'y': y} for surf, x, y in coords if surf=='bottom' ]

    # Selig-Format
    else:
        fmt = 'selig'
        coords = []
        for line in data_lines:
            x, y = map(float, line.split())
            coords.append((x, y))
        # Finde Index des minimalen x-Werts → Trennungspunkt
        xs = [x for x, _ in coords]
        min_x = min(xs)
        split_idx = xs.index(min_x)
        top_pts = coords[:split_idx + 1]
        bot_pts = coords[split_idx:]
        top = [ {'x': x, 'y': y} for x, y in top_pts ]
        bottom = [ {'x': x, 'y': y} for x, y in bot_pts ]

    return {
        'name': name,
        'format': fmt,
        'top': top,
        'bottom': bottom
    }


def main():
    parser = argparse.ArgumentParser(
        description="Parst alle .dat-Airfoil-Dateien und speichert sie als JSON"
    )
    parser.add_argument(
        '--config', default='config.yaml',
        help='Pfad zur config.yaml (Default: config.yaml)'
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

    # Konfiguration laden und Profil-Ordner ermitteln
    cfg = load_config(args.config)
    folder = cfg['profile_selection']['profile_folder']

    if not os.path.isdir(folder):
        logging.error("Profil-Ordner nicht gefunden: %s", folder)
        return

    profiles = {}
    for fname in os.listdir(folder):
        if not fname.lower().endswith('.dat'):
            continue
        path = os.path.join(folder, fname)
        try:
            parsed = parse_dat(path)
            profiles[fname] = parsed
            logging.info("Geparsed: %s (%s)", fname, parsed['format'])
        except Exception as e:
            logging.warning("Fehler beim Parsen %s: %s", fname, e)

    # Ergebnisverzeichnis anlegen
    os.makedirs('results', exist_ok=True)
    out_path = 'results/profiles_raw.json'
    with open(out_path, 'w') as f:
        json.dump(profiles, f, indent=2)
    logging.info("Alle Profile gespeichert in %s", out_path)


if __name__ == '__main__':
    main()