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
import pandas as pd
from typing import Tuple, List, Dict
import logging

from data_io import parse_airfoil_dat

logger = logging.getLogger(__name__)

def read_airfoil_dat(file_path: str) -> Tuple[str, np.ndarray]:
    """
    Liest eine .dat Datei im Selig-Format ein.
    
    Args:
        file_path: Pfad zur .dat Datei
        
    Returns:
        Tuple aus (Name, Koordinaten als numpy array)
    """
    try:
        with open(file_path, 'r') as f:
            lines = f.readlines()
            
        # Entferne leere Zeilen und Kommentare
        lines = [line.strip() for line in lines if line.strip() and not line.startswith('#')]
        
        if not lines:
            raise ValueError("Datei ist leer")
            
        # Erste Zeile ist der Name
        name = lines[0].strip()
        
        # Restliche Zeilen sind Koordinaten
        coords = []
        for line in lines[1:]:
            try:
                x, y = map(float, line.split())
                # Validiere Koordinaten
                if not (-0.01 <= x <= 1.01):
                    logger.warning(f"X-Koordinate {x} außerhalb des erlaubten Bereichs [-0.01, 1.01]")
                if not (-1.0 <= y <= 1.0):
                    logger.warning(f"Y-Koordinate {y} außerhalb des erlaubten Bereichs [-1.0, 1.0]")
                coords.append([x, y])
            except ValueError as e:
                logger.warning(f"Konnte Zeile nicht parsen: {line}. Fehler: {e}")
                continue
                
        if not coords:
            raise ValueError("Keine gültigen Koordinaten gefunden")
            
        return name, np.array(coords)
        
    except Exception as e:
        logger.error(f"Fehler beim Lesen von {file_path}: {e}")
        raise

def calculate_airfoil_properties(coords: np.ndarray) -> Dict[str, float]:
    """
    Berechnet die wichtigsten Eigenschaften eines Profils.
    
    Args:
        coords: numpy array mit X,Y Koordinaten
        
    Returns:
        Dictionary mit den Eigenschaften
    """
    x = coords[:, 0]
    y = coords[:, 1]
    
    # Finde Leading Edge (minimaler x-Wert)
    le_idx = np.argmin(x)
    le_x, le_y = x[le_idx], y[le_idx]
    
    # Finde Trailing Edge (maximaler x-Wert)
    te_idx = np.argmax(x)
    te_x, te_y = x[te_idx], y[te_idx]
    
    # Berechne Sehnenlänge
    chord = np.sqrt((te_x - le_x)**2 + (te_y - le_y)**2)
    
    # Berechne maximale Dicke
    thickness = np.max(y) - np.min(y)
    
    # Berechne maximale Wölbung
    # Vereinfachte Berechnung: Mittelwert der oberen und unteren Kontur
    upper = y[x >= 0.5]  # Obere Kontur
    lower = y[x <= 0.5]  # Untere Kontur
    if len(upper) > 0 and len(lower) > 0:
        camber = (np.max(upper) + np.min(lower)) / 2
    else:
        camber = 0.0
        
    return {
        'chord': chord,
        'max_thickness': thickness,
        'max_camber': camber,
        'le_x': le_x,
        'le_y': le_y,
        'te_x': te_x,
        'te_y': te_y
    }

def process_airfoil_database(database_path: str) -> pd.DataFrame:
    """
    Verarbeitet alle .dat Dateien in einem Verzeichnis.
    
    Args:
        database_path: Pfad zum Verzeichnis mit .dat Dateien
        
    Returns:
        DataFrame mit den Profilinformationen
    """
    airfoil_data = []
    
    for filename in os.listdir(database_path):
        if filename.endswith('.dat'):
            file_path = os.path.join(database_path, filename)
            try:
                name, coords = read_airfoil_dat(file_path)
                properties = calculate_airfoil_properties(coords)
                
                airfoil_data.append({
                    'name': name,
                    'filename': filename,
                    **properties
                })
                
            except Exception as e:
                logger.error(f"Fehler bei {filename}: {e}")
                continue
                
    return pd.DataFrame(airfoil_data)

def filter_airfoils(df: pd.DataFrame, 
                   min_thickness: float = 0.0,
                   max_thickness: float = 1.0,
                   min_camber: float = 0.0,
                   max_camber: float = 1.0) -> pd.DataFrame:
    """
    Filtert die Profile nach Dicke und Wölbung.
    
    Args:
        df: DataFrame mit Profilinformationen
        min_thickness: Minimale Dicke
        max_thickness: Maximale Dicke
        min_camber: Minimale Wölbung
        max_camber: Maximale Wölbung
        
    Returns:
        Gefilterter DataFrame
    """
    mask = (
        (df['max_thickness'] >= min_thickness) &
        (df['max_thickness'] <= max_thickness) &
        (df['max_camber'] >= min_camber) &
        (df['max_camber'] <= max_camber)
    )
    return df[mask]

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
        except FileNotFoundError:
            logging.warning(f"Kein .dat für {name}, überspringe Metafilter")
            filtered[name] = polars[name]
            continue
        except Exception as e:
            logging.error(f"Fehler bei {name}: {e}")
            continue
    return filtered

def generate_polar(airfoil_data, alpha_range=[-2, 8], n_points=20):
    """
    Generiert eine einfache Polar für ein Profil.
    
    Args:
        airfoil_data: Dictionary mit Profilinformationen
        alpha_range: [min_alpha, max_alpha] in Grad
        n_points: Anzahl der Punkte in der Polar
        
    Returns:
        Dictionary mit alpha, Cl, Cd Arrays
    """
    alpha = np.linspace(alpha_range[0], alpha_range[1], n_points)
    
    # Vereinfachte Cl-Berechnung basierend auf Profilparametern
    t_c = airfoil_data['t_c']
    camber = airfoil_data['camber_max']
    
    # Cl = 2*pi*alpha + 2*pi*camber (vereinfachte Theorie)
    Cl = 2 * np.pi * np.radians(alpha) + 2 * np.pi * camber
    
    # Cd = Cd0 + k*Cl^2 (parabolischer Widerstand)
    Cd0 = 0.006  # Basis-Widerstand
    k = 0.1      # Induzierter Widerstand
    Cd = Cd0 + k * Cl**2
    
    return {
        'alpha': alpha,
        'Cl': Cl,
        'Cd': Cd
    }

def quick_rank(polars, alpha_range=[-2, 8], top_n=50):
    """
    Führt eine schnelle Rangierung der Airfoil-Profile durch.

    Berechnet für jedes Profil das maximale Cl/Cd-Verhältnis im Anstellwinkel-Bereich.

    Parameter:
      polars     : dict, Mapping airfoil_name -> Polar-Daten inkl. Metadaten
      alpha_range: Liste [alpha_min, alpha_max] in Grad
      top_n      : int, Anzahl der auszuwählenden Top-Profile

    Rückgabe:
      list of tuples: [(airfoil_name, score), ...] sortiert absteigend nach score
    """
    scores = []
    alpha_min, alpha_max = alpha_range
    
    for name, data in polars.items():
        try:
            # Generiere Polar wenn nicht vorhanden
            if 'alpha' not in data:
                polar = generate_polar(data, alpha_range)
                data.update(polar)
            
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
                
        except Exception as e:
            logging.warning(f"Fehler bei Profil {name}: {e}")
            continue
            
    # Sortiere absteigend nach max Cl/Cd und gebe Top_n zurück
    scores.sort(key=lambda x: x[1], reverse=True)
    return scores[:top_n]

def select_airfoils(polars, config, Re_target, airfoil_dir):
    """
    Komplette Screening-Pipeline:
      1. Meta-Filter
      2. Quick-Rangierung
      3. Rückgabe der Top-N Profilnamen

    Parameter:
      polars     : dict, alle geladenen Polare
      config     : dict, aus config.yaml geladene Parameter
      Re_target  : float, Ziel-Reynolds-Zahl
      airfoil_dir: str, Pfad zum Verzeichnis mit .dat Dateien

    Rückgabe:
      list of str: Namen der ausgewählten Airfoil-Profile
    """
    # 1. Meta-Filter
    filtered = filter_meta(polars,
                          t_c_range=config['t_c_range'],
                          Re_target=Re_target,
                          airfoil_dir=airfoil_dir)
    # 2. Quick-Ranking
    ranked = quick_rank(filtered,
                        alpha_range=[-2, 8],
                        top_n=config.get('top_n', 50))
    # 3. Extrahiere Profilnamen
    selected = [name for name, _ in ranked]
    return selected

def main():
    # Konfiguriere Logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s'
    )
    
    # Pfad zur Profildatenbank
    database_path = "data/airfoils"
    
    try:
        # Verarbeite alle Profile
        df = process_airfoil_database(database_path)
        
        # Speichere Ergebnisse
        output_path = "results/airfoil_database.csv"
        df.to_csv(output_path, index=False)
        logger.info(f"Ergebnisse gespeichert in {output_path}")
        
        # Beispiel für Filterung
        filtered = filter_airfoils(df, 
                                 min_thickness=0.1,
                                 max_thickness=0.2,
                                 min_camber=0.0,
                                 max_camber=0.05)
        
        logger.info(f"Gefundene Profile: {len(df)}")
        logger.info(f"Gefilterte Profile: {len(filtered)}")
        
    except Exception as e:
        logger.error(f"Fehler in main(): {e}")
        raise

if __name__ == "__main__":
    main()