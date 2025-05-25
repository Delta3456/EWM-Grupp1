import numpy as np
import pandas as pd
import logging

def extract_geometry_from_txt(txt_path, min_valid_sections=2):
    """
    Liest Profil-Koordinaten aus einer whitespace-separierten .txt (ohne echten CSV-Header),
    gruppiert nach z-Slices und berechnet für jede z-Ebene:
      - r (radial = z)
      - chord (Distanz LE–TE in x)
      - twist (Winkel der Sehne)
    """
    # 1) Datei einlesen, skip die beiden Kopfzeilen
    df = pd.read_csv(
        txt_path,
        delim_whitespace=True,
        skiprows=2,
        header=None,
        names=['x','y','z']
    )
    logging.info(f"Daten geladen: {len(df)} Punkte aus {txt_path}")

    # 2) nach z gruppieren
    zs = np.sort(df['z'].unique())
    sections = []
    for z in zs:
        pts = df.loc[df['z'] == z, ['x','y']].values
        if pts.shape[0] < 3:
            logging.warning(f"Zu wenige Punkte bei z={z:.5f} m")
            continue
        sections.append((z, pts))

    if len(sections) < min_valid_sections:
        raise ValueError(f"Nur {len(sections)} gültige Sektionen – mind. {min_valid_sections} benötigt")

    # 3) r, chord, twist berechnen
    rs, chords, twists = [], [], []
    for z, pts in sections:
        r, c, phi = process_section(pts, z)
        rs.append(r)
        chords.append(c)
        twists.append(phi)

    # 4) auf gleichmäßiges Gitter interpolieren
    idx = np.argsort(rs)
    r_sorted = np.array(rs)[idx]
    c_sorted = np.array(chords)[idx]
    phi_sorted = np.array(twists)[idx]

    c_arr = np.interp(r_sorted, r_sorted, c_sorted)
    phi_arr = np.interp(r_sorted, r_sorted, phi_sorted)

    return r_sorted, c_arr, phi_arr

def process_section(pts, z):
    """
    Aus Nx2-Array pts (x,y) bei Ebene z:
      r     = z
      chord = Abstand zwischen min(x) und max(x)
      twist = arctan2(dy,dx) der Sehne
    """
    xs, ys = pts[:,0], pts[:,1]
    i_le, i_te = np.argmin(xs), np.argmax(xs)
    le, te = pts[i_le], pts[i_te]

    chord = np.hypot(*(te - le))
    dx, dy = te[0]-le[0], te[1]-le[1]
    twist = np.arctan2(dy, dx)
    r = z
    return r, chord, twist