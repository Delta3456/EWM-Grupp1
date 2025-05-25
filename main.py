# main.py

"""
main.py

Orchestriert den Optimierungs-Workflow mit Laufzeitmessung pro Schritt.
"""

import os
import time
import logging
import numpy as np

from config import load_config
from utils import setup_logging, ensure_dir
from data_io import read_performance, read_polars
from geometry import extract_geometry_from_txt
from airfoil_screen import select_airfoils
from bem import BladeElementMomentum
from optimize import run_optimization
from validate import prepare_cfd_input, prepare_fem_input, summarize_results
from plotting import plot_cp_curves, plot_geometry_comparison

def main():
    try:
        # 1) Initialisierung
        setup_logging()
        cfg = load_config("config.yaml")
        ensure_dir(cfg['output_dir'])
        logging.info("Starte Windturbinen-Optimierungs-Pipeline")

        # 2) Performance-Daten einlesen
        t0 = time.time()
        df_v, v_opt_orig, P_opt, omega_opt = read_performance(cfg['performance_file'])
        A = np.pi * cfg['R']**2
        Cp_orig = P_opt / (0.5 * cfg['rho'] * A * v_opt_orig**3)
        logging.info(f"Leistung eingelesen: v_opt={v_opt_orig:.2f} m/s, Cp_max={Cp_orig:.4f}")
        logging.info(f"  Dauer Schritt 2: {time.time() - t0:.2f}s")

        # 3) Geometrie-Extraktion aus STEP
        t1 = time.time()
        coords_file = cfg['coords_file']
        r, c0, phi0 = extract_geometry_from_txt(coords_file, min_valid_sections=3)
        logging.info("Geometrie aus STEP extrahiert")
        logging.info(f"  Dauer Schritt 3: {time.time() - t1:.2f}s")

        # 4) Airfoil-Screening
        t2 = time.time()
        airfoil_dir = cfg.get('airfoil_dir', 'data/airfoils')
        polars = read_polars(airfoil_dir)
        if not polars:
            raise RuntimeError("Keine gültigen Airfoil-Daten gefunden")
            
        v_target = cfg['v_target_factor'] * v_opt_orig
        mid_idx = len(r) // 2
        Re_target = v_target * c0[mid_idx] / 1.5e-5
        
        selected = select_airfoils(polars, cfg, Re_target, airfoil_dir)
        if not selected:
            raise RuntimeError("Keine passenden Profile gefunden")
            
        best_profile = selected[0]
        logging.info(f"Ausgewähltes Airfoil (Top-1): {best_profile}")
        logging.info(f"  Dauer Schritt 4: {time.time() - t2:.2f}s")

        # 5) BEM-Setup & Optimierung
        t3 = time.time()
        polar_data = polars[best_profile]
        bem = BladeElementMomentum(r, c0, phi0, polar_data, cfg['Z'], cfg['rho'], cfg['R'])
        result = run_optimization(bem, cfg, v_opt_orig, Cp_orig, omega_opt)
        factors = result.x[:len(r)]
        offsets = result.x[len(r):]
        c_opt = c0 * (1 + factors)
        phi_opt = phi0 + offsets
        logging.info("Optimierung abgeschlossen")
        logging.info(f"  Dauer Schritt 5: {time.time() - t3:.2f}s")

        # 6) Ergebnis-Plots
        t4 = time.time()
        v_range = np.linspace(0.5 * v_target, 1.5 * v_target, 50)
        _, Cp_curve_orig = bem.power_curve(v_range, omega_opt)
        bem_opt = BladeElementMomentum(r, c_opt, phi_opt, polar_data,
                                     cfg['Z'], cfg['rho'], cfg['R'])
        _, Cp_curve_opt = bem_opt.power_curve(v_range, omega_opt)
        plot_cp_curves(v_range, Cp_curve_orig, v_range, Cp_curve_opt, cfg['output_dir'])
        plot_geometry_comparison(r, c0, c_opt, phi0, phi_opt, cfg['output_dir'])
        logging.info("Plots erstellt")
        logging.info(f"  Dauer Schritt 6: {time.time() - t4:.2f}s")

        # 7) CFD/FEM-Input vorbereiten
        t5 = time.time()
        cfd_script = prepare_cfd_input(
            cfg.get('cfd_case_dir', 'cfd_case'),
            cfg.get('cfd_mesh_file', 'mesh.msh'),
            os.path.join(cfg['output_dir'], 'cfd_results'),
            solver=cfg.get('cfd_solver', 'ansys_cfx')
        )
        fem_script = prepare_fem_input(
            cfg.get('fem_case_dir', 'fem_case'),
            cfg.get('fem_model_file', 'fem_input.inp'),
            os.path.join(cfg['output_dir'], 'fem_results'),
            solver=cfg.get('fem_solver', 'ansys')
        )
        summary = summarize_results(
            os.path.join(cfg['output_dir'], 'cfd_results'),
            os.path.join(cfg['output_dir'], 'fem_results')
        )
        logging.info("Validierungs-Vorbereitung abgeschlossen")
        logging.info(f"  Dauer Schritt 7: {time.time() - t5:.2f}s")

        total = time.time() - t0  # ab Schritt 2
        logging.info(f"Gesamtdauer (ohne Init): {total:.2f}s")
        
    except Exception as e:
        logging.error(f"Fehler in der Pipeline: {e}")
        raise

if __name__ == "__main__":
    main()