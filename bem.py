# bem.py

"""
bem.py

Implementiert die Blattelement-Momentum-Methode (BEM) für Rotorblätter.
Alle Formeln basieren auf Kapitel 3.6 der Masterarbeit:
  - Axialer Induktionsfaktor a:   Eq. (3.7) & Eq. (3.8)  [oai_citation:0‡M02-24_Masterarbeit_Alexander_Malchow.pdf](file-service://file-XJnpmUy5eA5ihGj1XRTCpe)
  - Radialer Induktionsfaktor a': Eq. (3.9) & Eq. (3.10)  [oai_citation:1‡M02-24_Masterarbeit_Alexander_Malchow.pdf](file-service://file-XJnpmUy5eA5ihGj1XRTCpe)
  - Tangentialkraft dM:           Eq. (3.11)  [oai_citation:2‡M02-24_Masterarbeit_Alexander_Malchow.pdf](file-service://file-XJnpmUy5eA5ihGj1XRTCpe)
  - Schubkraft dF_S:              Eq. (3.12)  [oai_citation:3‡M02-24_Masterarbeit_Alexander_Malchow.pdf](file-service://file-XJnpmUy5eA5ihGj1XRTCpe)
"""

import numpy as np

class BladeElementMomentum:
    def __init__(self, r, c, phi, polars, Z, rho, R, nu=1.5e-5):
        """
        Initialisierung der BEM-Klasse.

        Parameter:
          r      : np.ndarray, Radialpositionen [m]
          c      : np.ndarray, Sehnenlängen c(r) [m]
          phi    : np.ndarray, Twist-Winkel φ(r) [rad]
          polars : dict, Airfoil-Polare mit 'cl_fun' und 'cd_fun'
          Z      : int, Zahl der Rotorblätter
          rho    : float, Luftdichte [kg/m^3]
          R      : float, Rotorradius [m]
          nu     : float, kinematische Viskosität [m^2/s] (Standard 1.5e-5)
        """
        self.r = r
        self.c = c
        self.phi = phi
        self.polars = polars
        self.Z = Z
        self.rho = rho
        self.R = R
        self.nu = nu
        # Radialabstand delta r
        self.dr = np.diff(np.concatenate(([0], r)))

    def compute_induction(self, v_inf, Omega, tol=1e-5, max_iter=100):
        """
        Iterative Berechnung der Induktionsfaktoren a(r), a'(r).

        Eq. (3.7): a = (v_inf - c1) / v_inf
        Eq. (3.8): c1 = v_inf * (1 - a)
        Eq. (3.9): a' = omega / Omega
        Eq. (3.10): u1 = Omega * r * (1 + a')

        Rückgabe:
          a   : np.ndarray, axialer Induktionsfaktor
          a_p : np.ndarray, radialer Induktionsfaktor
          c1  : np.ndarray, axiale Geschwindigkeit
          u1  : np.ndarray, Umfangsgeschwindigkeit
        """
        # Startwerte
        a   = np.full_like(self.r, 1/3)    # Betz-Ansatz  [oai_citation:4‡M02-24_Masterarbeit_Alexander_Malchow.pdf](file-service://file-XJnpmUy5eA5ihGj1XRTCpe)
        a_p = np.zeros_like(self.r)
        for _ in range(max_iter):
            # Berechne c1 und u1
            c1 = v_inf * (1 - a)                       # Eq. (3.8)
            u1 = Omega * self.r * (1 + a_p)           # Eq. (3.10)
            # resultierende Anströmgeschwindigkeit
            w = np.sqrt(c1**2 + u1**2)
            # Geometrischer Anstellwinkel β
            beta = np.arctan2(c1, u1)
            # Aerodynamischer Anstellwinkel α = β - φ
            alpha = beta - self.phi
            # Reynolds-Zahl schätzen
            Re = w * self.c / self.nu
            # Profilbeiwerte
            Cl = self.polars['cl_fun'](np.rad2deg(alpha))  # in Funktionsaufruf Grad!
            Cd = self.polars['cd_fun'](np.rad2deg(alpha))
            # Blattbeladung σ = (Z * c) / (2πr)
            sigma = self.Z * self.c / (2 * np.pi * self.r)
            # Aktualisierung a und a'
            # Klassische Gleichungen aus Kap. 3.6 Derivation
            # a_new = 1 / ( (4 sin^2β) / (σ Cl cosβ) + 1 )
            # a_p_new = 1 / ( (4 sinβ cosβ) / (σ Cl sinβ) - 1 )
            a_new   = 1 / ( (4 * np.sin(beta)**2) / (sigma * Cl * np.cos(beta)) + 1 )
            a_p_new = 1 / ( (4 * np.sin(beta) * np.cos(beta)) / (sigma * Cl * np.sin(beta)) - 1 )
            # Relaxation / Konvergenz
            if np.max(np.abs(a_new - a)) < tol and np.max(np.abs(a_p_new - a_p)) < tol:
                a, a_p = a_new, a_p_new
                break
            a, a_p = a_new, a_p_new
        return a, a_p, c1, u1

    def element_forces(self, v_inf, Omega):
        """
        Berechnet die radialen Verteilungen der Kräfte dM und dF_S.

        Eq. (3.11): dM    = ½ρ w² c Z (Cl sinσ - Cd cosσ) r dr
        Eq. (3.12): dF_S  = ½ρ w² c Z (Cl cosσ + Cd sinσ) dr

        Rückgabe:
          dM  : np.ndarray, Drehmoment­differential je Segment [Nm]
          dF  : np.ndarray, Schubkraft­differential je Segment [N]
        """
        a, a_p, c1, u1 = self.compute_induction(v_inf, Omega)
        # resultierende Geschwindigkeit & Anströmwinkel
        w    = np.sqrt(c1**2 + u1**2)
        beta = np.arctan2(c1, u1)
        alpha = beta - self.phi
        # Profilbeiwerte
        Cl = self.polars['cl_fun'](np.rad2deg(alpha))
        Cd = self.polars['cd_fun'](np.rad2deg(alpha))
        # Blattbeladung
        sigma = self.Z * self.c / (2 * np.pi * self.r)
        # Differentiale
        dM = 0.5 * self.rho * w**2 * self.c * self.Z * \
             (Cl * np.sin(sigma) - Cd * np.cos(sigma)) * self.r * self.dr
        dF = 0.5 * self.rho * w**2 * self.c * self.Z * \
             (Cl * np.cos(sigma) + Cd * np.sin(sigma)) * self.dr
        return dM, dF

    def power_curve(self, v_range, Omega):
        """
        Erzeugt Leistungs- und Cp-Kurve über einen Bereich von Windgeschwindigkeiten.

        P(v)   = Ω * ∫ dM    über r
        Cp(v)  = P / (½ρ A v³),  A = πR²
        """
        P = []
        Cp = []
        A = np.pi * self.R**2
        for v in v_range:
            dM, _ = self.element_forces(v, Omega)
            M = np.sum(dM)
            P_v = Omega * M
            Cp_v = P_v / (0.5 * self.rho * A * v**3)
            P.append(P_v)
            Cp.append(Cp_v)
        return np.array(P), np.array(Cp)