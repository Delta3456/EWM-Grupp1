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
import argparse
import logging
import json
import yaml
import pandas as pd
import numpy as np
from scipy.interpolate import make_interp_spline
import sys
from typing import Tuple, Dict, Any
from pathlib import Path

# Konstante zur Umrechnung von mph in m/s
MPH_TO_MS = 0.44704  # 1 mph = 0.44704 m/s

def load_config(path: str) -> Dict[str, Any]:
    """
    Lädt die YAML-Konfigurationsdatei und gibt sie als dict zurück.
    
    Args:
        path: Pfad zur YAML-Konfigurationsdatei
        
    Returns:
        Dict mit Konfigurationsparametern
        
    Raises:
        FileNotFoundError: Wenn die Datei nicht existiert
        yaml.YAMLError: Bei ungültigem YAML-Format
    """
    try:
        with open(path, 'r') as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"Konfigurationsdatei '{path}' nicht gefunden")
    except yaml.YAMLError as e:
        raise yaml.YAMLError(f"Ungültiges YAML-Format: {e}")

def validate_input_data(v_mph: np.ndarray, P: np.ndarray) -> None:
    """
    Validiert die Eingabedaten auf Plausibilität.
    
    Args:
        v_mph: Array mit Windgeschwindigkeiten in mph
        P: Array mit Leistungswerten in Watt
        
    Raises:
        ValueError: Bei ungültigen Daten
    """
    if len(v_mph) != len(P):
        raise ValueError("Arrays für Geschwindigkeit und Leistung müssen gleich lang sein")
    if len(v_mph) < 2:
        raise ValueError("Mindestens 2 Datenpunkte erforderlich")
    if np.any(v_mph < 0):
        raise ValueError("Negative Windgeschwindigkeiten sind nicht erlaubt")
    if np.any(P < 0):
        raise ValueError("Negative Leistungswerte sind nicht erlaubt")
    if not np.all(np.diff(v_mph) > 0):
        raise ValueError("Windgeschwindigkeiten müssen streng monoton steigend sein")

def read_curve(path: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Liest die Excel-Datei ein und extrahiert Windgeschwindigkeit und Leistung.
    
    Args:
        path: Pfad zur Excel-Datei
        
    Returns:
        Tuple aus (v_mph, P) Arrays
        
    Raises:
        FileNotFoundError: Wenn die Datei nicht existiert
        ValueError: Bei ungültigen Daten
    """
    try:
        df = pd.read_excel(path)
        required_columns = ['Windgeschwindigkeit (mph)', 'Leistung (W)']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Fehlende Spalten in Excel-Datei: {missing_columns}")
            
        v_mph = df['Windgeschwindigkeit (mph)'].values
        P = df['Leistung (W)'].values
        
        validate_input_data(v_mph, P)
        return v_mph, P
        
    except FileNotFoundError:
        raise FileNotFoundError(f"Excel-Datei '{path}' nicht gefunden")
    except Exception as e:
        raise ValueError(f"Fehler beim Einlesen der Excel-Datei: {e}")

def smooth_and_interpolate(
    v_mph: np.ndarray,
    P: np.ndarray,
    window: int,
    n_points: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Glättet und interpoliert die Leistungskurve.
    
    Args:
        v_mph: Windgeschwindigkeiten in mph
        P: Leistungswerte in Watt
        window: Fenstergröße für gleitenden Mittelwert
        n_points: Anzahl der Interpolationspunkte
        
    Returns:
        Tuple aus (v_new, P_new) Arrays
        
    Raises:
        ValueError: Bei ungültigen Parametern
    """
    if window < 1:
        raise ValueError("Fenstergröße muss mindestens 1 sein")
    if n_points < 2:
        raise ValueError("Mindestens 2 Interpolationspunkte erforderlich")
        
    # Umrechnung in m/s
    v = v_mph * MPH_TO_MS
    
    # Glättung mit gleitendem Mittelwert
    P_smooth = pd.Series(P).rolling(
        window, center=True, min_periods=1
    ).mean().values
    
    # Erzeugen neuer Geschwindigkeitswerte
    v_new = np.linspace(v.min(), v.max(), n_points)
    
    # Spline-Interpolation
    try:
        spline = make_interp_spline(v, P_smooth, k=3)
        P_new = spline(v_new)
    except Exception as e:
        raise ValueError(f"Fehler bei der Spline-Interpolation: {e}")
        
    return v_new, P_new

def compute_design_params(
    v: np.ndarray,
    P: np.ndarray,
    a: float,
    b: float
) -> Dict[str, float]:
    """
    Berechnet die Auslegungsparameter.
    
    Args:
        v: Windgeschwindigkeiten in m/s
        P: Leistungswerte in Watt
        a: Generator-Parameter a
        b: Generator-Parameter b
        
    Returns:
        Dict mit Auslegungsparametern
        
    Raises:
        ValueError: Bei ungültigen Parametern
    """
    if a <= 0:
        raise ValueError("Parameter 'a' muss positiv sein")
        
    idx_peak = np.argmax(P)
    v_peak = v[idx_peak]
    P_peak = P[idx_peak]
    v_design = 1.2 * v_peak
    omega_design = (P_peak + b) / a
    
    return {
        'v_peak': float(v_peak),
        'P_peak': float(P_peak),
        'v_design': float(v_design),
        'omega_design': float(omega_design)
    }

def main() -> None:
    parser = argparse.ArgumentParser(
        description='Vorverarbeitung der Leistungskurve'
    )
    parser.add_argument(
        '--config', required=True,
        help='Pfad zur config.yaml'
    )
    args = parser.parse_args()

    # Logging-Konfiguration
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s: %(message)s'
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
        out_path = Path(cfg['output']['design_params'])
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, 'w') as f:
            json.dump(design, f, indent=2)
        logging.info('Design-Parameter gespeichert in %s', out_path)

    except Exception as e:
        logging.error('Fehler in preprocessing: %s', e)
        sys.exit(1)

if __name__ == '__main__':
    main()