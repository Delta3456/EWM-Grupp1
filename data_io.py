# data_io.py

import os
import logging
import pandas as pd
import numpy as np
from scipy.interpolate import interp1d

def read_performance(path):
    """
    Liest die Leistungskurve aus einer Excel-Datei ein.
    Erwartete Sheets:
      - 'Wind_vs_Power': Spalten ['v', 'P']
      - 'Omega_vs_Power': Spalten ['omega', 'P']
    Gibt zurück:
      - df_v: DataFrame mit Windgeschwindigkeit vs. Leistung
      - v_opt: Windgeschwindigkeit am Leistung Maximum (v_{C_p,max})
      - P_opt: Maximale Leistung
      - omega_opt: Drehzahl bei P_opt
    Bezug: Masterarbeit Kap. 3.6 (Bestimmung v_{C_p,max})
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Performance-Datei nicht gefunden: {path}")
    df_v  = pd.read_excel(path, sheet_name='Wind_vs_Power')
    df_om = pd.read_excel(path, sheet_name='Omega_vs_Power')
    df_v.columns  = ['v', 'P']
    df_om.columns = ['omega', 'P']
    # Bestimmung des Maximums
    idx     = df_v['P'].idxmax()
    v_opt   = df_v.at[idx, 'v']
    P_opt   = df_v.at[idx, 'P']
    # Interpolation, um omega bei P_opt zu finden
    f_omega = interp1d(df_om['P'], df_om['omega'], fill_value="extrapolate")
    omega_opt = float(f_omega(P_opt))
    return df_v, v_opt, P_opt, omega_opt

def read_polars(airfoil_dir):
    """
    Liest alle .dat Dateien im Airfoil-Verzeichnis ein.
    Baut für jede Datei Interpolationsfunktionen für Cl(alpha) und Cd(alpha).

    Rückgabe:
      dict: {
        airfoil_name: {
          'alpha': np.ndarray,
          'Cl':    np.ndarray,
          'Cd':    np.ndarray,
          'cl_fun': interp1d,
          'cd_fun': interp1d
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
            # Kopfzeile überspringen
            data = np.loadtxt(path, skiprows=1)
        except Exception as e:
            logging.warning(f"Fehler beim Einlesen von {fname}: {e}")
            continue

        if data.ndim < 2 or data.shape[1] < 3:
            logging.warning(f"{fname} hat nicht mind. 3 Spalten, wird übersprungen.")
            continue

        alpha = data[:, 0]  # Anstellwinkel [°]
        Cl    = data[:, 1]  # Auftriebsbeiwerte
        Cd    = data[:, 2]  # Widerstandsbeiwerte

        # Erzeuge Interpolationsfunktionen
        cl_fun = interp1d(alpha, Cl, kind='cubic', fill_value='extrapolate')
        cd_fun = interp1d(alpha, Cd, kind='cubic', fill_value='extrapolate')

        name = os.path.splitext(fname)[0]
        polars[name] = {
            'alpha': alpha,
            'Cl':    Cl,
            'Cd':    Cd,
            'cl_fun': cl_fun,
            'cd_fun': cd_fun
        }

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