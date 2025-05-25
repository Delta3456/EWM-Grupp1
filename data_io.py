# data_io.py

import os
import logging
import pandas as pd
import numpy as np
from scipy.interpolate import interp1d

def read_performance(path):
    """
    Liest die Leistungskurve aus einer Excel-Datei ein.
    Erwartete Spalten:
      - 'Windgeschwindigkeit (mph)': Windgeschwindigkeit in mph
      - 'Leistung': Elektrische Leistung der Air30
    
    Die Omega_vs_Power Daten werden aus der Funktion generiert:
    P = 17.192 * omega - 182.54
    
    Gibt zurück:
      - df_v: DataFrame mit Windgeschwindigkeit vs. Leistung
      - v_opt: Windgeschwindigkeit am Leistung Maximum (v_{C_p,max})
      - P_opt: Maximale Leistung
      - omega_opt: Drehzahl bei P_opt
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Performance-Datei nicht gefunden: {path}")
    
    # Lese Wind vs. Power Daten
    df_v = pd.read_excel(path)
    # Konvertiere mph in m/s (1 mph = 0.44704 m/s)
    df_v['v'] = df_v['Windgeschwindigkeit (mph)'] * 0.44704
    df_v['P'] = df_v['Windgeschwindigkeit (mph)']
    
    # Generiere Omega vs. Power Daten aus der Funktion
    omega_range = np.linspace(10, 30, 100)  # Drehzahlbereich in 1/s
    P_omega = 17.192 * omega_range - 182.54
    df_om = pd.DataFrame({
        'omega': omega_range,
        'P': P_omega
    })
    
    # Bestimmung des Leistungsmaximums
    idx = df_v['P'].idxmax()
    v_opt = df_v.at[idx, 'v']
    P_opt = df_v.at[idx, 'P']
    
    # Interpolation für die zugehörige Drehzahl
    f_omega = interp1d(df_om['P'], df_om['omega'], fill_value='extrapolate')
    omega_opt = float(f_omega(P_opt))
    
    return df_v, v_opt, P_opt, omega_opt

def read_polars(airfoil_dir):
    """
    Liest alle .dat Dateien im Airfoil-Verzeichnis ein.
    Erwartet .dat Dateien im Selig- oder Lednicer-Format:
    - Erste Zeile: Name/Description des Profils
    - Restliche Zeilen: X,Y Koordinaten (2 Spalten)
    
    Bei Lednicer-Format:
    - Zweite Zeile enthält zwei Zahlen > 1 für Anzahl der Punkte auf Ober- und Unterseite
    
    Rückgabe:
      dict: {
        airfoil_name: {
          'coords': np.ndarray,  # X,Y Koordinaten
          'name': str,          # Name/Description
          't_c': float,         # Maximale Dicke
          'x_t_max': float,     # X-Position der max. Dicke
          'camber_max': float,  # Maximale Wölbung
          'x_c_max': float      # X-Position der max. Wölbung
        },
        ...
      }
    """
    polars = {}
    if not os.path.isdir(airfoil_dir):
        raise FileNotFoundError(f"Airfoil-Verzeichnis nicht gefunden: {airfoil_dir}")

    for fname in os.listdir(airfoil_dir):
        if not fname.lower().endswith('.dat'):
            continue
        path = os.path.join(airfoil_dir, fname)
        try:
            # Lese und säubere alle Zeilen
            with open(path, 'r') as f:
                lines = [l.strip() for l in f if l.strip()]

            if not lines:
                logging.warning(f"{fname} ist leer, wird übersprungen.")
                continue

            name = lines[0]
            # Prüfe auf Lednicer-Format
            parts = lines[1].split()
            if len(parts) == 2 and float(parts[0]) > 1 and float(parts[1]) > 1:
                try:
                    N_top, N_bot = int(parts[0]), int(parts[1])
                    data_lines = lines[2:2 + N_top + N_bot]
                except (ValueError, IndexError) as e:
                    logging.error(f"{fname}: Ungültiges Lednicer-Format: {e}")
                    continue
            else:
                data_lines = lines[1:]

            coords = []
            for line in data_lines:
                try:
                    parts = line.split()
                    if len(parts) < 2:
                        logging.warning(f"{fname}: Ungültige Zeile (zu wenige Werte): {line}")
                        continue
                    x, y = map(float, parts[:2])
                    # Validiere Koordinaten
                    if not -0.01 <= x <= 1.01:
                        logging.warning(f"{fname}: x={x:.3f} außerhalb [-0.01,1.01]")
                    if not -1.0 <= y <= 1.0:
                        logging.warning(f"{fname}: y={y:.3f} außerhalb [-1.0,1.0]")
                    coords.append((x, y))
                except ValueError as e:
                    logging.warning(f"{fname}: Ungültige Zeile: {line}")
                    continue

            if not coords:
                logging.warning(f"{fname}: Keine gültigen Koordinaten gefunden")
                continue

            coords = np.array(coords)
            
            # Split in Top/Bottom
            top = coords[coords[:,1] >= 0]
            bot = coords[coords[:,1] <  0]
            
            if len(top) == 0 or len(bot) == 0:
                logging.warning(f"{fname}: Keine vollständige Profilkontur (fehlende Ober- oder Unterseite)")
                continue
            
            # Wähle Master-Seite
            master, slave = (top, bot) if len(top) >= len(bot) else (bot, top)
            
            # Interpolation der Slave-Höhe auf Master-x
            slave_x, slave_y = slave[:,0], slave[:,1]
            interp_slave = interp1d(slave_x, slave_y, bounds_error=False, fill_value="extrapolate")
            x_m = master[:,0]
            y_m = master[:,1]
            y_s = interp_slave(x_m)

            # Thickness und Camber
            thickness   = np.abs(y_m - y_s)
            idx_t       = np.nanargmax(thickness)
            t_c         = thickness[idx_t]
            x_t_max     = x_m[idx_t]
            camber      = (y_m + y_s) / 2.0
            idx_c       = np.nanargmax(np.abs(camber))
            camber_max  = camber[idx_c]
            x_c_max     = x_m[idx_c]

            name = os.path.splitext(fname)[0]
            polars[name] = {
                'coords': coords,
                'name': name,
                't_c': t_c,
                'x_t_max': x_t_max,
                'camber_max': camber_max,
                'x_c_max': x_c_max,
                'cl_fun': lambda alpha: 2 * np.pi * np.sin(np.deg2rad(alpha)),  # Simple linear theory
                'cd_fun': lambda alpha: 0.02 + 0.1 * np.sin(np.deg2rad(alpha))**2  # Simple drag model
            }

        except Exception as e:
            logging.error(f"Fehler beim Einlesen von {fname}: {e}")
            continue

    if not polars:
        raise RuntimeError(f"Keine gültigen Airfoil-.dat Dateien gefunden in {airfoil_dir}")
    return polars

def parse_airfoil_dat(path):
    """
    Parst eine Airfoil-.dat Datei (Selig- oder Lednicer-Format) und berechnet:
      - coords_top, coords_bot: Arrays mit Ober- und Unterseiten-Koordinaten.
      - t_c: max thickness (in y).
      - x_t_max: x-Position der max thickness.
      - camber_max: max Camber-Linien-Halbwert (in y).
      - x_c_max: x-Position des max Camber.
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(f"{path} nicht gefunden.")
    # Lese und säubere alle Zeilen
    with open(path, 'r') as f:
        lines = [l.strip() for l in f if l.strip()]

    name = lines[0]
    # Prüfe auf Lednicer-Format
    parts = lines[1].split()
    if len(parts) == 2 and float(parts[0]) > 1 and float(parts[1]) > 1:
        N_top, N_bot = int(parts[0]), int(parts[1])
        data_lines = lines[2:2 + N_top + N_bot]
    else:
        data_lines = lines[1:]

    coords = []
    for line in data_lines:
        try:
            x, y = map(float, line.split()[:2])
        except:
            logging.warning(f"{path}: Ungültige Zeile: {line}")
            continue
        if not -0.01 <= x <= 1.01:
            logging.warning(f"{path}: x={x:.3f} außerhalb [0,1]")
        if not -1.0 <= y <= 1.0:
            logging.warning(f"{path}: y={y:.3f} außerhalb [-1,1]")
        coords.append((x, y))
    coords = np.array(coords)

    # Split in Top/Bottom
    top = coords[coords[:,1] >= 0]
    bot = coords[coords[:,1] <  0]
    # Wähle Master-Seite
    master, slave = (top, bot) if len(top) >= len(bot) else (bot, top)
    # Interpolation der Slave-Höhe auf Master-x
    slave_x, slave_y = slave[:,0], slave[:,1]
    interp_slave = interp1d(slave_x, slave_y, bounds_error=False, fill_value="extrapolate")
    x_m = master[:,0]
    y_m = master[:,1]
    y_s = interp_slave(x_m)

    # Thickness und Camber
    thickness   = np.abs(y_m - y_s)
    idx_t       = np.nanargmax(thickness)
    t_c         = thickness[idx_t]
    x_t_max     = x_m[idx_t]
    camber      = (y_m + y_s) / 2.0
    idx_c       = np.nanargmax(np.abs(camber))
    camber_max  = camber[idx_c]
    x_c_max     = x_m[idx_c]

    return {
        'name':       name,
        'coords_top': top,
        'coords_bot': bot,
        't_c':        t_c,
        'x_t_max':    x_t_max,
        'camber_max': camber_max,
        'x_c_max':    x_c_max
    }

def validate_profile(x, y, filename=""):
    """
    Validiert ein Profil und korrigiert es wenn nötig.
    - x, y: Arrays der X- und Y-Koordinaten
    - filename: Name der Datei für Logging
    """
    # Prüfe auf vollständige Kontur
    if len(x) < 4:  # Mindestens 2 Punkte pro Seite
        logging.warning(f"{filename}: Zu wenige Punkte für vollständiges Profil")
        return None, None
        
    # Finde Leading Edge (minimaler x-Wert)
    le_idx = np.argmin(x)
    
    # Teile Profil in Ober- und Unterseite
    upper_x = x[:le_idx+1]
    upper_y = y[:le_idx+1]
    lower_x = x[le_idx:]
    lower_y = y[le_idx:]
    
    # Prüfe ob beide Seiten vorhanden sind
    if len(upper_x) < 2 or len(lower_x) < 2:
        logging.warning(f"{filename}: Keine vollständige Profilkontur (fehlende Ober- oder Unterseite)")
        return None, None
        
    # Korrigiere X-Koordinaten außerhalb [0,1]
    x_min, x_max = np.min(x), np.max(x)
    if x_min < -0.01 or x_max > 1.01:
        logging.warning(f"{filename}: X-Koordinaten außerhalb [-0.01,1.01], skaliere auf [0,1]")
        # Skaliere auf [0,1]
        x = (x - x_min) / (x_max - x_min)
        
    # Prüfe auf doppelte Punkte
    unique_mask = np.concatenate(([True], np.diff(x) != 0))
    if not np.all(unique_mask):
        logging.warning(f"{filename}: Doppelte X-Koordinaten gefunden, entferne Duplikate")
        x = x[unique_mask]
        y = y[unique_mask]
        
    # Prüfe auf monotone X-Koordinaten
    if not np.all(np.diff(x) >= 0):
        logging.warning(f"{filename}: X-Koordinaten nicht monoton, sortiere")
        sort_idx = np.argsort(x)
        x = x[sort_idx]
        y = y[sort_idx]
        
    return x, y

def read_airfoil_dat(filepath):
    """
    Liest ein Profil aus einer .dat Datei im Selig/Lednicer Format.
    - filepath: Pfad zur .dat Datei
    Rückgabe:
      x, y: Arrays der X- und Y-Koordinaten
    """
    try:
        # Versuche zuerst Selig-Format (x y)
        data = np.loadtxt(filepath, skiprows=1)
        if data.shape[1] != 2:
            raise ValueError("Kein Selig-Format")
            
        x, y = data[:, 0], data[:, 1]
        
        # Validiere und korrigiere das Profil
        x, y = validate_profile(x, y, os.path.basename(filepath))
        if x is None:
            return None
            
        return x, y
        
    except Exception as e:
        logging.error(f"Fehler beim Lesen von {filepath}: {e}")
        return None