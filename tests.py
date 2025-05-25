# tests.py

import json
import os
import pytest

from preprocessing import smooth_and_interpolate, compute_design_params
from tsr_scan import tsr_scan
from bem import bem
from profile_selection import select_profiles

def test_smoothing_interpolation():
    # Prüft, ob glätten und interpolieren die korrekte Länge erzeugt
    v_mph = [0, 10, 20, 30]
    P = [0, 100, 200, 150]
    v, P_new = smooth_and_interpolate(v_mph, P, window=2, n_points=10)
    assert len(v) == 10
    assert len(P_new) == 10

def test_design_params():
    # Prüft, ob v_design und omega_design berechnet werden
    v = [1, 2, 3]
    P = [10, 20, 15]
    res = compute_design_params(v, P, a=2, b=5)
    assert 'v_design' in res and res['v_design'] == pytest.approx(1.2 * 2)
    assert 'omega_design' in res

def test_tsr_basic():
    # Einfacher TSR-Scan
    design = {'v_design': 1.0, 'a_gen': 2, 'b_gen': 0}
    tsr_cfg = {'rho':1.225, 'R':0.5, 'lambda_min':1, 'lambda_max':2, 'lambda_step':1}
    res = tsr_scan(design, tsr_cfg)
    # λ_opt muss 1 oder 2 sein
    assert res['lambda_opt'] in [1.0, 2.0]

def test_bem_output_shape():
    # Form der BEM-Ergebnisse
    design = {'v_design':1.0}
    tsr_cfg = {'rho':1.225, 'R':0.5, 'omega_design':10, 'lambda_opt':7}
    bem_cfg = {
        'r_min':0.05, 'r_max':0.5,
        'N_segments':5, 'tol_induction':1e-4,
        'B':3, 'mu_air':1.8e-5
    }
    res = bem(design, tsr_cfg, bem_cfg)
    assert 'c_P' in res
    assert len(res['r']) == bem_cfg['N_segments']

def test_profile_mapping_empty(tmp_path, monkeypatch):
    # Wenn kein Profil im Ordner liegt, kommt dennoch ein dict zurück
    cfg = {
        'profile_selection': {
            'mu_air':1,
            'score_weights':{'ld':0.5,'clmax':0.5},
            'profile_folder': str(tmp_path)
        },
        'tsr_scan': {'rho':1.225}
    }
    bem_res = {'r':[0.1,0.2], 'dF_S':[1,1]}
    mapping = select_profiles(bem_res, cfg)
    assert isinstance(mapping, dict)