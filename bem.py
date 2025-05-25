#!/usr/bin/env python3
"""
tsr_scan.py

Modul 2: Sweep über den Tip Speed Ratio (TSR)-Bereich
  - Liest Design-Parameter aus Modul 1
  - Durchläuft λ ∈ [lambda_min, lambda_max] in Schritten lambda_step
  - Berechnet vereinfachten Leistungsbeiwert c_P(λ)
  - Findet λ_opt mit maximalem c_P
  - Aktualisiert omega_design auf Basis von λ_opt
  - Speichert Scan-Ergebnisse in tsr_scan.json
"""
import argparse
import logging
import json
import yaml
import numpy as np
import sys


def load_config(path):
    """Lädt die YAML-Konfigurationsdatei."""
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def load_design(path):
    """Lädt design_params.json aus Modul 1."""
    with open(path, 'r') as f:
        return json.load(f)


def tsr_scan(design, tsr_cfg):
    """
    Führt den TSR-Scan durch:
      P(λ) approximiert durch lineares Modell P = a·λ + b
      c_P(λ) = P(λ) / (0.5·ρ·π·R²·v_design³)
    Return: dict mit allen λ, zugehörigen c_P, λ_opt und neuer omega_design
    """
    # Auslegungsgeschwindigkeit und Luftdichte
    v = design['v_design']
    rho = tsr_cfg['rho']
    R = tsr_cfg['R']                    # Fester Rotorradius

    # Aufbau des λ-Vektors
    lambdas = np.arange(
        tsr_cfg['lambda_min'],
        tsr_cfg['lambda_max'] + 1e-8,
        tsr_cfg['lambda_step']
    )
    # Parameter der linearen Leistungskennlinie
    a = tsr_cfg.get('a', design.get('a_gen', None)) or design['a_gen']
    b = tsr_cfg.get('b', design.get('b_gen', None)) or design['b_gen']

    c_p = []
    # Kalkuliere c_P für jeden λ
    for lam in lambdas:
        P_l = a * lam + b                # Vereinfachte P(λ)
        cp = P_l / (0.5 * rho * np.pi * R**2 * v**3)
        c_p.append(cp)
    c_p = np.array(c_p)

    # Optimum finden
    idx_opt = np.argmax(c_p)
    lam_opt = float(lambdas[idx_opt])

    # Neue Drehzahl basierend auf λ_opt
    omega_new = lam_opt * v / R

    return {
        'lambdas': lambdas.tolist(),
        'c_p': c_p.tolist(),
        'lambda_opt': lam_opt,
        'omega_design': float(omega_new)
    }


def main():
    parser = argparse.ArgumentParser(
        description='TSR-Scan zur Optimierung des Leistungsbeiwerts'
    )
    parser.add_argument('--config', required=True, help='Pfad zu config.yaml')
    parser.add_argument('--design', required=True, help='Pfad zu design_params.json')
    args = parser.parse_args()

    # Logging initialisieren
    logging.basicConfig(
        level=logging.INFO,
        format='%(levelname)s: %(message)s'
    )
    try:
        cfg = load_config(args.config)
        tsr_cfg = cfg['tsr_scan']
        design = load_design(args.design)

        # TSR-Scan durchführen
        result = tsr_scan(design, tsr_cfg)
        logging.info('λ_optimal = %.3f', result['lambda_opt'])

        # Ergebnisse speichern
        out_path = cfg['output']['tsr_scan']
        with open(out_path, 'w') as f:
            json.dump(result, f, indent=2)
        logging.info('TSR-Scan-Ergebnisse gespeichert in %s', out_path)

    except Exception as e:
        logging.error('Fehler in tsr_scan: %s', e)
        sys.exit(1)

if __name__ == '__main__':
    main()