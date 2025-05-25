#!/usr/bin/env python3
"""
preprocessing.py

Modul 1: Vorverarbeitung der Leistungskurve
  - Excel-Datei einlesen (Windgeschwindigkeit in mph, Leistung in W)
  - Umrechnung mph → m/s
  - Glätten mit gleitendem Mittelwert (smoothing_window)
  - Spline-Interpolation auf konstante Anzahl Punkte (interpolation_points)
  - Ermittlung des Leistungs-Peaks und Auslegungsgrößen:
      v_design = 1.2 · v_peak
      omega_design = (P_peak + b) / a
  - Speichern der Parameter in design_params.json
"""
import argparse      # CLI-Parser
import logging       # Zentralisiertes Logging
import json          # JSON-Ausgabe
import yaml          # YAML-Konfigurations-Einlesung
import pandas as pd  # Datenrahmen für Tabellendaten
import numpy as np   # Numerische Operationen
from scipy.interpolate import make_interp_spline  # Spline-Interpolation
import sys           # Systemfunktionen (Exit)

# Konstante zur Umrechnung von mph in m/s
MPH_TO_MS = 0.44704  # 1 mph = 0.44704 m/s

def load_config(path):
    """Lädt die YAML-Konfigurationsdatei und gibt sie als dict zurück."""
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def read_curve(path):
    """
    Liest die Excel-Datei ein und extrahiert:
      - Windgeschwindigkeit in mph
      - Leistung in Watt
    Rückgabe: zwei Arrays v_mph, P
    """
    df = pd.read_excel(path)
    # Spaltennamen müssen mit config übereinstimmen
    v_mph = df['Windgeschwindigkeit (mph)'].values
    P = df['Leistung (W)'].values
    return v_mph, P


def smooth_and_interpolate(v_mph, P, window, n_points):
    """
    1) Umrechnung in m/s
    2) Gleitender Mittelwert zur Glättung der Leistungskurve
    3) Cubic Spline-Interpolation auf feines Raster
    """
    # Schritt 1: Umrechnung der Geschwindigkeiten
    v = v_mph * MPH_TO_MS
    # Schritt 2: Glättung mit gleitendem Mittelwert
    P_smooth = pd.Series(P).rolling(
        window, center=True, min_periods=1
    ).mean().values
    # Schritt 3: Erzeugen neuer Geschwindigkeitswerte
    v_new = np.linspace(v.min(), v.max(), n_points)
    # Erstellen und Auswerten des Splines
    spline = make_interp_spline(v, P_smooth, k=3)
    P_new = spline(v_new)
    return v_new, P_new


def compute_design_params(v, P, a, b):
    """
    Bestimmt:
      - Index des Leistungspunks: idx_peak
      - Windgeschwindigkeit beim Peak: v_peak
      - Leistung beim Peak: P_peak
      - Auslegungs-Windgeschwindigkeit: v_design = 1.2 * v_peak
      - Auslegungs-Drehzahl: omega_design = (P_peak + b) / a
    Rückgabe: dict mit diesen Werten
    """
    idx_peak = np.argmax(P)              # Index der maximalen Leistung
    v_peak = v[idx_peak]                # Geschw. am Peak
    P_peak = P[idx_peak]                # Leistung am Peak
    v_design = 1.2 * v_peak             # Sicherheitsfaktor 1.2 über Peak
    omega_design = (P_peak + b) / a      # Umrechnung Peak-Leistung → Drehzahl
    return {
        'v_peak': float(v_peak),
        'P_peak': float(P_peak),
        'v_design': float(v_design),
        'omega_design': float(omega_design)
    }


def main():
    parser = argparse.ArgumentParser(
        description='Vorverarbeitung der Leistungskurve'
    )
    parser.add_argument(
        '--config', required=True,
        help='Pfad zur config.yaml'
    )
    args = parser.parse_args()

    # Logging-Grundkonfiguration
    logging.basicConfig(
        level=logging.INFO,
        format='%(levelname)s: %(message)s'
    )

    try:
        # Konfiguration laden
        cfg = load_config(args.config)
        pre = cfg['preprocessing']
        gen = cfg['generator']

        # Rohdaten einlesen
        v_mph, P = read_curve(pre['performance_curve'])
        logging.info('Eingelesene Punkte: %d', len(v_mph))

        # Glätten und Interpolieren
        v, P_new = smooth_and_interpolate(
            v_mph, P,
            pre['smoothing_window'],
            pre['interpolation_points']
        )
        logging.info(
            'Daten geglättet (Fenster=%d) und interpoliert auf %d Punkte',
            pre['smoothing_window'], pre['interpolation_points']
        )

        # Auslegungsparameter berechnen
        design = compute_design_params(
            v, P_new,
            gen['a'], gen['b']
        )
        logging.info(
            'v_design=%.3f m/s, omega_design=%.3f rad/s',
            design['v_design'], design['omega_design']
        )

        # Ergebnisse speichern
        out_path = cfg['output']['design_params']
        with open(out_path, 'w') as f:
            json.dump(design, f, indent=2)
        logging.info('Design-Parameter gespeichert in %s', out_path)

    except Exception as e:
        logging.error('Fehler in preprocessing: %s', e)
        sys.exit(1)

if __name__ == '__main__':
    main()