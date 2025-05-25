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

    # Strafe für Cp-Verlust (quadratische Strafe für stärkere Gewichtung)
    penalty = max(0.0, Cp_orig - Cp_new)**2

    # Zielfunktionswert mit normalisierter Geschwindigkeitsdifferenz
    v_diff = abs(v_opt_new - v_target) / v_target
    f = v_diff + weight * penalty

    # Zusätzliche Strafe für zu große Änderungen
    max_chord_change = np.max(np.abs(chord_factors))
    max_twist_change = np.max(np.abs(twist_offsets))
    f += 0.1 * (max_chord_change**2 + max_twist_change**2)

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

    # Parameter für Stagnationserkennung
    max_stagnation = 5  # Maximale Anzahl gleicher Ergebnisse
    best_result = None
    best_f = float('inf')

    class StagnationChecker:
        def __init__(self):
            self.stagnation_count = 0
            self.last_f = float('inf')
            
        def __call__(self, xk, convergence=None):
            if convergence is not None:
                current_f = convergence
                if abs(current_f - self.last_f) < 1e-10:
                    self.stagnation_count += 1
                    if self.stagnation_count >= max_stagnation:
                        return True
                else:
                    self.stagnation_count = 0
                self.last_f = current_f
            return False

    # Aufruf von Differential Evolution mit mehreren Versuchen
    for attempt in range(3):  # Maximal 3 Versuche
        stagnation_checker = StagnationChecker()
        result = differential_evolution(
            func=objective,
            bounds=bounds,
            args=(bem, v_target, Cp_orig, omega_opt, weight),
            strategy='best1bin',
            popsize=20,        # Reduzierte Populationsgröße
            tol=1e-2,
            maxiter=30,        # Reduzierte maximale Iterationen
            mutation=(0.5, 1.0),
            recombination=0.7,
            disp=True,
            callback=stagnation_checker,
            polish=False       # Deaktiviere den finalen Polishing-Schritt
        )
        
        # Speichere das beste Ergebnis
        if result.fun < best_f:
            best_result = result
            best_f = result.fun
            
        # Wenn wir ein gutes Ergebnis haben, brechen wir ab
        if result.fun < 1e-2:  # Erhöhte Toleranz für früheren Abbruch
            break
            
        # Ansonsten erweitern wir die Suchgrenzen leicht
        cd *= 1.2
        td *= 1.2
        bounds_chord = [(-cd, cd)] * N
        bounds_twist = [(-td, td)] * N
        bounds = bounds_chord + bounds_twist

    return best_result