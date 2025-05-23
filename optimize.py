# optimize.py

"""
optimize.py

Enthält Funktionen zur parametrischen Optimierung des Rotorblatts.
Verwendet Differential Evolution aus SciPy.

Zielfunktion (Eigenentwicklung auf Basis Kap. 3.6):
    f(p) = |v_opt(p) - v_target| + weight * max(0, Cp_orig - Cp(p))

wobei:
  - v_opt(p)   = Windgeschwindigkeit des maximalen Cp nach Parametern p
  - Cp(p)      = maximaler Cp nach Parametern p
  - v_target   = v_orig * v_target_factor
  - Cp_orig    = maximaler Cp der Originalgeometrie
  - weight     = Gewichtungsfaktor für Cp-Verlust (konfigurierbar)

Formelbezüge:
  - v_opt und Cp-Berechnung über power_curve() in bem.py (Kap. 3.6)
  - Parametrisierte Geometrie-Updates siehe geometry.py
"""

import numpy as np
from scipy.optimize import differential_evolution
from utils import deg2rad

def objective(p, bem, v_target, Cp_orig, Omega, weight):
    """
    Zielfunktion für die Optimierung.

    Parameter:
      p         : Parametervektor [Δc1, ..., ΔcN, Δφ1, ..., ΔφN]
                  Δci sind relative Änderungen der Sehnen (z.B. 0.1 für +10%),
                  Δφi sind Twist-Offsets in Radiant.
      bem       : Instanz von BladeElementMomentum
      v_target  : Ziel-Windgeschwindigkeit (m/s)
      Cp_orig   : Originaler maximaler Cp
      Omega     : Betriebsdrehzahl für die Analyse (rad/s)
      weight    : Gewicht für den Cp-Verlust

    Rückgabe:
      float: Zielfunktionswert
    """
    N = len(bem.r)
    # Sicherung der Originalgeometrie
    c0, phi0 = bem.c.copy(), bem.phi.copy()

    # Parametereinteilung
    chord_factors = p[:N]
    twist_offsets  = p[N:]

    # Geometrie aktualisieren
    bem.c   = c0 * (1 + chord_factors)
    bem.phi = phi0 + twist_offsets

    # Cp-Kurve berechnen
    v_range = np.linspace(0.5 * v_target, 1.5 * v_target, 50)
    _, Cp = bem.power_curve(v_range, Omega)

    # Neues Optimum finden
    idx_opt   = np.nanargmax(Cp)
    v_opt_new = v_range[idx_opt]
    Cp_new    = Cp[idx_opt]

    # Strafe für Cp-Verlust
    penalty = max(0.0, Cp_orig - Cp_new)

    # Zielfunktionswert
    f = abs(v_opt_new - v_target) + weight * penalty

    # Geometrie zurücksetzen
    bem.c, bem.phi = c0, phi0
    return f

def run_optimization(bem, config, v_opt_orig, Cp_orig, omega_opt):
    """
    Führt die Optimierung mit Differential Evolution durch.

    Parameter:
      bem        : BladeElementMomentum-Instanz
      config     : geladene Konfigurationsparameter (dict)
      v_opt_orig : ursprüngliche Windgeschwindigkeit bei Cp_max
      Cp_orig    : ursprünglicher Cp_max
      omega_opt  : Betriebsdrehzahl (rad/s) bei Cp_max

    Rückgabe:
      result: OptimizeResult-Objekt von SciPy mit den optimalen Parametern
    """
    # Ziel definieren
    v_target = config['v_target_factor'] * v_opt_orig
    weight   = config.get('cp_weight', 1.0)

    N = len(bem.r)
    # Bounds für Δc: ± chord_delta
    cd = config['chord_delta']
    bounds_chord = [(-cd, cd)] * N
    # Bounds für Δφ: ± twist_delta (in Radiant)
    td = deg2rad(config['twist_delta'])
    bounds_twist = [(-td, td)] * N
    bounds = bounds_chord + bounds_twist

    # Aufruf von Differential Evolution
    result = differential_evolution(
        func=objective,
        bounds=bounds,
        args=(bem, v_target, Cp_orig, omega_opt, weight),
        strategy='best1bin',
        popsize=15,
        tol=1e-3,
        maxiter=100,
        disp=True
    )
    return result