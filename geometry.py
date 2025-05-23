from OCC.Core.STEPControl import STEPControl_Reader
from OCC.Core.BRepAlgoAPI import BRepAlgoAPI_Section
from OCC.Core.gp import gp_Pln, gp_Pnt, gp_Dir
import numpy as np

def extract_geometry_from_step(step_path, R, n_sections=30):
    """
    Extrahiert für n_sections radiale Stationen die Sehne c(r) und den Twist φ(r)
    aus der STEP-Datei.
    - step_path: Pfad zur .stp Datei
    - R: Rotorradius in Metern
    - n_sections: Anzahl radialer Schnitte
    Rückgabe:
      r: np.array der Radialpositionen
      c: np.array der Sehnenlängen
      phi: np.array der Twist-Winkel (rad)
    """
    # STEP-Datei einlesen
    reader = STEPControl_Reader()
    status = reader.ReadFile(step_path)
    if status != 1:
        raise IOError(f"STEP-Datei konnte nicht gelesen werden: {step_path}")
    reader.TransferRoots()
    shape = reader.OneShape()

    # Erzeuge gleichmäßige Radialpositionen
    r = np.linspace(0.05 * R, R, n_sections)
    c = np.zeros_like(r)
    phi = np.zeros_like(r)

    for i, ri in enumerate(r):
        # Ebene senkrecht zur Rotationsachse auf Höhe z = ri
        plane = gp_Pln(gp_Pnt(0, 0, ri), gp_Dir(0, 0, 1))
        # Schnitt der Schaufel mit der Ebene
        section = BRepAlgoAPI_Section(shape, plane).Shape()
        # Kontur-Analyse (Platzhalter):
        #   - Bestimme Leading-Edge-Punkt (le_point)
        #   - Bestimme Trailing-Edge-Punkt (te_point)
        #   - Berechne Sehnenlänge c = Abstand(le_point, te_point)
        #   - Berechne Twist φ = Winkel der Profilsehne zur Referenz
        le_point = ...   # gp_Pnt, muss implementiert werden
        te_point = ...   # gp_Pnt, muss implementiert werden
        chord = le_point.Distance(te_point)
        c[i] = chord
        twist_angle = ...  # in Radiant, muss implementiert werden
        phi[i] = twist_angle

    return r, c, phi