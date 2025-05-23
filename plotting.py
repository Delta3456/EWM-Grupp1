# plotting.py

"""
plotting.py

Enthält Funktionen zur Visualisierung der Aerodynamik- und Geometrieergebnisse.

Module:
  - plot_cp_curves: Vergleich Original- vs. Optimiert-Cp(v)
  - plot_geometry_comparison: Visualisierung von c(r)/φ(r) alt vs. neu

Verwendet Matplotlib. Keine Farbangaben, um Flexibilität für Nutzerdarstellungen zu erhalten.
"""

import os
import matplotlib.pyplot as plt

def plot_cp_curves(v_orig, Cp_orig, v_opt, Cp_opt, output_dir):
    """
    Zeichnet Cp-Kurven Original vs. Optimiert.

    Parameter:
      v_orig    : np.ndarray Windgeschwindigkeiten original
      Cp_orig   : np.ndarray Cp-Werte original
      v_opt     : np.ndarray Windgeschwindigkeiten optimiert
      Cp_opt    : np.ndarray Cp-Werte optimiert
      output_dir: Verzeichnis zum Speichern der Abbildung
    """
    os.makedirs(output_dir, exist_ok=True)
    plt.figure()
    plt.plot(v_orig, Cp_orig, label='Original', linewidth=2)
    plt.plot(v_opt, Cp_opt, label='Optimiert', linestyle='--', linewidth=2)
    plt.xlabel('Windgeschwindigkeit (m/s)')
    plt.ylabel('Leistungsbeiwert Cp')
    plt.title('Vergleich der Cp-Kurven')
    plt.legend()
    plt.grid(True)
    save_path = os.path.join(output_dir, 'cp_comparison.png')
    plt.savefig(save_path)
    plt.close()

def plot_geometry_comparison(r, c_orig, c_opt, phi_orig, phi_opt, output_dir):
    """
    Zeichnet den Vergleich der Geometrie-Verteilungen c(r) und φ(r).

    Parameter:
      r           : np.ndarray Radialpositionen
      c_orig      : np.ndarray Sehnenlängen original
      c_opt       : np.ndarray Sehnenlängen optimiert
      phi_orig    : np.ndarray Twist original [rad]
      phi_opt     : np.ndarray Twist optimiert [rad]
      output_dir  : Verzeichnis zum Speichern der Abbildung
    """
    os.makedirs(output_dir, exist_ok=True)
    # Plot c(r)
    plt.figure()
    plt.plot(r, c_orig, label='c original', linewidth=2)
    plt.plot(r, c_opt, label='c optimiert', linestyle='--', linewidth=2)
    plt.xlabel('Radialposition r (m)')
    plt.ylabel('Sehnenlänge c (m)')
    plt.title('Sehnenverteilung Original vs. Optimiert')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'chord_comparison.png'))
    plt.close()

    # Plot φ(r)
    plt.figure()
    plt.plot(r, phi_orig, label='φ original', linewidth=2)
    plt.plot(r, phi_opt, label='φ optimiert', linestyle='--', linewidth=2)
    plt.xlabel('Radialposition r (m)')
    plt.ylabel('Twist φ (rad)')
    plt.title('Twist-Verteilung Original vs. Optimiert')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'twist_comparison.png'))
    plt.close()
