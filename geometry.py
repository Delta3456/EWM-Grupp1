# geometry.py

"""
geometry.py

Extrahiert aus einer STEP-Datei mit pythonOCC-core 7.9.0
die radiale Sehnen- (c) und Twist-Verteilung (φ) eines Rotorblatts.

Anpassung:
 - Nutzung des Konstruktors BRepAlgoAPI_Section(shape, tool, performNow)
   statt nicht vorhandener Methode Init().
 - Beibehaltung robuster Endpunkt-Sammlung und Logging.
"""

import numpy as np
import logging

from OCC.Core.STEPControl import STEPControl_Reader
from OCC.Core.BRepAlgoAPI import BRepAlgoAPI_Section
from OCC.Core.gp import gp_Pln, gp_Pnt, gp_Dir, gp_Vec
from OCC.Core.BRep import BRep_Tool
from OCC.Core.TopExp import TopExp_Explorer
from OCC.Core.TopAbs import TopAbs_EDGE
from OCC.Core.TopoDS import topods

def extract_geometry_from_step(step_path: str,
                               R: float,
                               n_sections: int = 30):
    """
    Extrahiert c(r) und φ(r) aus STEP.

    Args:
      step_path  : Pfad zur .stp Datei
      R          : Rotorradius [m]
      n_sections : Anzahl radiale Schnitte

    Returns:
      r   : np.ndarray, radiale Positionen [m]
      c   : np.ndarray, Sehnenlängen [m]
      phi : np.ndarray, Twist-Winkel [rad]
    """

    # 1) STEP einlesen
    reader = STEPControl_Reader()
    status = reader.ReadFile(step_path)
    if status != 1:
        raise IOError(f"STEP-Datei konnte nicht gelesen werden: {step_path}")
    reader.TransferRoots()
    blade_shape = reader.OneShape()

    # 2) Vorbereitung
    r   = np.linspace(0.05 * R, R, n_sections)
    c   = np.zeros_like(r)
    phi = np.zeros_like(r)

    # 3) Schnitte durchführen
    for i, ri in enumerate(r):
        plane = gp_Pln(gp_Pnt(0.0, 0.0, ri), gp_Dir(0.0, 0.0, 1.0))

        # 3a) Section direkt per Konstruktor erstellen und builden
        section = BRepAlgoAPI_Section(blade_shape, plane, True)
        # Optional: Genauere Approximation und PCurves auf beiden Shapes
        section.Approximation(True)
        section.ComputePCurveOn1(True)
        section.ComputePCurveOn2(True)
        if not section.IsDone():
            logging.warning(f"Sektion fehlgeschlagen bei z={ri:.3f} m")
            continue
        sec_shape = section.Shape()

        # 3b) Endpunkte aller Kanten sammeln
        points = []
        exp = TopExp_Explorer(sec_shape, TopAbs_EDGE)
        while exp.More():
            edge = topods.Edge(exp.Current())
            try:
                curve_handle, u1, u2 = BRep_Tool.Curve(edge)
                curve = curve_handle.GetObject()
            except Exception:
                exp.Next()
                continue

            pA = gp_Pnt(); pB = gp_Pnt()
            curve.D0(u1, pA); curve.D0(u2, pB)
            points.append((pA.X(), pA.Y(), pA))
            points.append((pB.X(), pB.Y(), pB))
            exp.Next()

        if len(points) < 2:
            logging.warning(f"Zu wenige Punkte bei z={ri:.3f} m")
            continue

        # 4) Sehne über größtes Punktepaar finden
        pts_xy = np.array([(x, y) for x, y, _ in points])
        dists = np.linalg.norm(pts_xy[:, None, :] - pts_xy[None, :, :], axis=2)
        idx_flat = np.nanargmax(dists)
        idx1, idx2 = divmod(idx_flat, len(points))
        p_le, p_te = points[idx1][2], points[idx2][2]

        # 5) Sehnenlänge speichern
        c[i] = p_le.Distance(p_te)

        # 6) Twist-Winkel von LE → TE
        vec = gp_Vec(p_le, p_te)
        phi[i] = np.arctan2(vec.Y(), vec.X())

    return r, c, phi