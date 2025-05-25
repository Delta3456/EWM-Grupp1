#!/usr/bin/env python3
"""
bem.py

Modul 3: Blade Element Momentum (BEM) ohne Prandtl-Korrekturen
  - Berechnet axiale und tangentiale Induktionsfaktoren
  - Bestimmt lokale Anströmung, Kräfte und Momentbeiträge
  - Nutzt Kontinuitäts- und Impulserhaltung
  - Summiert Leistung und berechnet c_P
"""
import argparse
import logging
import json
import yaml
import numpy as np
import sys

# Kappa: rad/s für Drehzahl

def load_config(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def load_inputs(design_path, tsr_path):
    with open(design_path) as f:
        design = json.load(f)
    with open(tsr_path) as f:
        tsr = json.load(f)
    return design, tsr

def ensure_float(value, name):
    try:
        return float(value)
    except (TypeError, ValueError):
        raise ValueError(f"Expected float for {name}, got {type(value)}")

def bem(design, tsr_cfg, bem_cfg):
    # Geometrische Daten
    R = ensure_float(tsr_cfg['R'], 'R')
    r_min = ensure_float(bem_cfg['r_min'], 'r_min')
    r_max = ensure_float(bem_cfg['r_max'], 'r_max')
    N = int(bem_cfg['N_segments'])
    B = int(bem_cfg['B'])  # Blattzahl
    rho = ensure_float(tsr_cfg['rho'], 'rho')
    v = ensure_float(design['v_design'], 'v_design')
    omega = ensure_float(tsr_cfg['omega_design'], 'omega_design')
    tsr_lambda_opt = ensure_float(tsr_cfg['lambda_opt'], 'lambda_opt')
    tol_induction = ensure_float(bem_cfg['tol_induction'], 'tol_induction')

    # Radiales diskretes Gitter
    r = np.linspace(r_min, r_max, N)
    dr = r[1] - r[0]

    # Speicherung
    dM = np.zeros_like(r)
    dF_S = np.zeros_like(r)

    # Induktionsfaktoren initialisieren
    a = np.zeros_like(r)
    aprime = np.zeros_like(r)

    for i, ri in enumerate(r):
        # Anfangsschätzung
        a_i = 0.3
        apr_i = 0.0
        for _ in range(100):
            # Geschwindigkeitsteil c1
            c1 = v * (1 - a_i)
            # Umfangsgeschwindigkeit u1
            u1 = omega * ri * (1 + apr_i)
            # Resultierende Geschwindigkeit
            w1 = np.hypot(c1, u1)
            # Anströmwinkel phi
            phi = np.arctan2(c1, u1)
            # Geometrie: Profilsehnenlänge (Nähe Betz)
            s_opt = (16/9 * np.pi * R *  # siehe Formel
                     float(bem_cfg.get('C_l_ref', 1.0)) /  # Referenz C_l (~1)
                     (B * tsr_lambda_opt) *
                     np.sqrt((tsr_lambda_opt*R/ri)**2 + 4/9))
            # Re-Zahl
            Re = rho * w1 * s_opt / float(bem_cfg.get('mu_air', 1.8e-5))
            # Profilpolare laden / interpolieren → C_l, C_d
            # (Hier Annahme: C_l = 1.0, C_d = 0.01 als Platzhalter)
            C_l = 1.0
            C_d = 0.01
            # Blattkräfte
            dM_i = 0.5 * rho * w1**2 * s_opt * B * (C_l*np.sin(phi) - C_d*np.cos(phi)) * ri * dr
            dF_S_i = 0.5 * rho * w1**2 * s_opt * B * (C_l*np.cos(phi) + C_d*np.sin(phi)) * dr
            # Berechne Induktionsfaktoren neu (Momentum-Theorie)
            a_new = dF_S_i / (4*np.pi * ri * rho * v**2 * (1 - a_i))
            apr_new = dM_i / (4*np.pi * ri**3 * rho * v * omega * (1 - a_i))
            if abs(a_new - a_i) < tol_induction and abs(apr_new - apr_i) < tol_induction:
                break
            a_i, apr_i = a_new, apr_new
        # speichern
        a[i], aprime[i] = a_i, apr_i
        dM[i], dF_S[i] = dM_i, dF_S_i

    # Gesamtmoment
    M = np.sum(dM)
    # Leistung
    P = M * omega
    # Leistungsbeiwert
    c_P = P / (0.5 * rho * np.pi * R**2 * v**3)

    return {
        'r': r.tolist(),
        'a': a.tolist(),
        'aprime': aprime.tolist(),
        'dM': dM.tolist(),
        'dF_S': dF_S.tolist(),
        'M_total': float(M),
        'P': float(P),
        'c_P': float(c_P)
    }


def main():
    parser = argparse.ArgumentParser(description='BEM-Berechnung')
    parser.add_argument('--config', required=True, help='Pfad zu config.yaml')
    parser.add_argument('--design', required=True, help='Design-Params JSON')
    parser.add_argument('--tsr', required=True, help='TSR-Scan JSON')
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    try:
        cfg = load_config(args.config)
        design, tsr = load_inputs(args.design, args.tsr)
        res = bem(design, tsr, cfg['bem'])
        out = cfg['output']['bem']
        with open(out, 'w') as f:
            json.dump(res, f, indent=2)
        logging.info('BEM-Ergebnisse gespeichert in %s', out)
    except Exception as e:
        logging.error('Fehler in bem.py: %s', e)
        sys.exit(1)

if __name__ == '__main__':
    main()