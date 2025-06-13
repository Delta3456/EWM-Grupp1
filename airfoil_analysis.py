#!/usr/bin/env python3
# combined_airfoil_with_camber.py
# Dieses Skript führt Bildanalyse und Matching gegen lokale .dat-Datenbank aus,
# berechnet und gibt für jeden Kandidaten neben RMSE und Fläche auch die Camber-Werte aus,
# und filtert zusätzlich nach Camber-Differenz.
# Benötigt: opencv-python, numpy, matplotlib
# Installation: pip install opencv-python numpy matplotlib

import cv2
import numpy as np
import argparse
import os
import sys

# --- Bildanalyse-Funktionen ---

def load_image(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"Fehler: Bild '{path}' konnte nicht geladen werden.", file=sys.stderr)
        sys.exit(1)
    return img

def extract_contour(img):
    """Extrahiert die größte Kontur mittels Canny + findContours."""
    blurred = cv2.GaussianBlur(img, (5, 5), 0)
    edges  = cv2.Canny(blurred, 50, 150)
    cnts, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not cnts:
        print("Keine Kontur gefunden!", file=sys.stderr)
        sys.exit(1)
    cnt = max(cnts, key=lambda c: cv2.arcLength(c, True))
    pts = cnt.squeeze()
    return pts.astype(float)

def normalize_profile(pts, real_chord_mm):
    """Dreht, verschiebt und skaliert die Kontur so, dass LE=(0,0), TE auf x-Achse, Chord=1."""
    le_idx = np.argmin(pts[:,0]); te_idx = np.argmax(pts[:,0])
    LE, TE = pts[le_idx], pts[te_idx]
    dx, dy = TE - LE
    theta = np.arctan2(dy, dx)
    R = np.array([[ np.cos(-theta), -np.sin(-theta)],
                  [ np.sin(-theta),  np.cos(-theta)]])
    pts_centered = pts - LE
    pts_rot = pts_centered.dot(R.T)
    chord_px = np.hypot(dx, dy)
    pts_norm = pts_rot / chord_px
    return pts_norm, chord_px

def compute_thickness_and_camber(pts_norm, n=300):
    """Berechnet Dicke- und Wölbungsverteilung aus normierter Kontur."""
    sorted_pts = pts_norm[np.argsort(pts_norm[:,0])]
    upper = sorted_pts[sorted_pts[:,1] >= 0]
    lower = sorted_pts[sorted_pts[:,1] <  0]
    x_lin = np.linspace(0, 1, n)
    up_y = np.interp(x_lin, upper[:,0], upper[:,1])
    lo_y = np.interp(x_lin, lower[:,0], lower[:,1])
    thickness = up_y - lo_y
    camber    = 0.5*(up_y + lo_y)
    return x_lin, thickness, camber

# --- DAT-Loading & Metriken ---

def load_dat(path):
    """Lädt ein Airfoil-DAT-File (x,y Paare) als Nx2 numpy-Array."""
    pts = []
    with open(path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 2: continue
            try:
                pts.append((float(parts[0]), float(parts[1])))
            except ValueError:
                continue
    return np.array(pts)

def compute_metrics(x_lin, measured_thick, measured_camber, dat_pts, real_chord_mm):
    """
    Berechnet für ein Kandidaten-Profil:
    - rmse_thick: RMSE der Dickenverteilung
    - camber_diff: absolute Abweichung der maximalen Wölbung
    - cand_cmax: maximale Wölbung des Kandidaten
    - cand_area_mm2: Profilfläche in mm²
    """
    half = len(dat_pts) // 2
    up = dat_pts[:half]; lo = dat_pts[half:]
    up_y = np.interp(x_lin, up[:,0], up[:,1])
    lo_y = np.interp(x_lin, lo[:,0], lo[:,1])
    cand_thick  = up_y - lo_y
    cand_camber = 0.5*(up_y + lo_y)

    rmse_thick = np.sqrt(np.mean((cand_thick - measured_thick)**2))
    meas_cmax   = measured_camber.max()
    cand_cmax   = cand_camber.max()
    camber_diff = abs(cand_cmax - meas_cmax)

    area_norm    = np.trapezoid(cand_thick, x_lin)
    cand_area_mm2 = area_norm * (real_chord_mm ** 2)

    return rmse_thick, camber_diff, cand_cmax, cand_area_mm2

# --- Hauptprogramm ---

def main():
    parser = argparse.ArgumentParser(
        description="Profil-Matching mit Dicken, Fläche und Camber"
    )
    parser.add_argument('image', help='Pfad zum Graustufen-Bild')
    parser.add_argument('-c','--real_chord', type=float, required=True,
                        help='Reale Chord-Länge in mm')
    parser.add_argument('-a','--measured_area_mm2', type=float, required=True,
                        help='Gemessene Querschnittsfläche in mm²')
    parser.add_argument('-d','--db_folder', default='data/AirFoil_datenbank',
                        help='Ordner mit .dat-Profilen')
    parser.add_argument('--area_threshold', type=float, default=0.05,
                        help='Max. relative Flächenabweichung (z.B. 0.05=5%)')
    parser.add_argument('--camber_threshold', type=float, default=0.01,
                        help='Max. absolute Camber-Abweichung (z.B. 0.01=1%)')
    parser.add_argument('--top', type=int, default=5,
                        help='Anzahl der Top-Matches')
    args = parser.parse_args()

    # 1) Bildanalyse
    img = load_image(args.image)
    pts = extract_contour(img)
    pixel_chord = np.max(pts[:,0]) - np.min(pts[:,0])
    print(f"Gemessene Chord in Pixeln: {pixel_chord:.1f} px")

    pts_norm, _ = normalize_profile(pts, args.real_chord)
    x_lin, meas_thick, meas_camber = compute_thickness_and_camber(pts_norm)
    print(f"Maximale Dicke: {meas_thick.max()*100:.2f}% bei x/c = {x_lin[np.argmax(meas_thick)]:.3f}")
    print(f"Maximale Wölbung: {meas_camber.max()*100:.2f}% bei x/c = {x_lin[np.argmax(meas_camber)]:.3f}")

    meas_area_norm = np.trapezoid(meas_thick, x_lin)
    meas_area_calc = meas_area_norm * (args.real_chord ** 2)
    print(f"Berechnete Fläche: {meas_area_calc:.3f} mm² (Soll: {args.measured_area_mm2:.3f} mm²)")

    # 2) Matching gegen Datenbank mit Camber-Filter
    results = []
    for fn in os.listdir(args.db_folder):
        if not fn.lower().endswith('.dat'):
            continue
        dat_pts = load_dat(os.path.join(args.db_folder, fn))
        if dat_pts.size == 0:
            continue

        rmse, cam_diff, cand_cmax, cand_area = compute_metrics(
            x_lin, meas_thick, meas_camber,
            dat_pts, args.real_chord
        )
        rel_area = abs(cand_area - args.measured_area_mm2) / args.measured_area_mm2
        if rel_area > args.area_threshold or cam_diff > args.camber_threshold:
            continue

        results.append((rmse, cam_diff, cand_cmax, rel_area, fn))

    if not results:
        print("Keine Profile innerhalb der Flächen- oder Camber-Toleranz gefunden.")
        sys.exit(0)

    # Sortierung: zuerst Camber-Differenz, dann RMSE
    results.sort(key=lambda x: (x[1], x[0]))

    print(f"\nTop {args.top} Matches (RMSE, ΔCamber, Camber, ΔFläche):\n")
    for rmse, cam_diff, cmax, rel_a, fn in results[:args.top]:
        print(f"{fn:25s}  RMSE={rmse:.5f}  ΔCamber={cam_diff*100:.2f}%  "
              f"Camber={cmax*100:.2f}%  ΔFläche={rel_a*100:.2f}%")

if __name__ == '__main__':
    main()