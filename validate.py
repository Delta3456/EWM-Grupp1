# validate.py

"""
validate.py

Modul zur Vorbereitung der CFD- und FEM-Validierung.
Erzeugt nur Batch-/Shell-Skripte (run_cfd.sh, run_fem.sh), startet die Solver nicht.

Funktionen:
  - prepare_cfd_input()
  - prepare_fem_input()
  - summarize_results()  (Parser-Platzhalter)
"""

import os
import logging

def prepare_cfd_input(case_dir, mesh_file, result_dir, solver="ansys_cfx"):
    """
    Erzeugt ein Shell-Skript zur CFD-Analyse, ohne den Solver aufzurufen.

    Parameter:
      case_dir   : Verzeichnis für CFD-Input (werden hier abgelegt)
      mesh_file  : Pfad zur Mesh-Datei (.msh o.ä.)
      result_dir : Zielverzeichnis für CFD-Ergebnisse
      solver     : 'ansys_cfx' oder 'openfoam'

    Rückgabe:
      script_path: Pfad zum generierten Skript (run_cfd.sh)
    """
    os.makedirs(case_dir, exist_ok=True)
    os.makedirs(result_dir, exist_ok=True)
    script_path = os.path.join(case_dir, "run_cfd.sh")
    with open(script_path, "w") as f:
        f.write("#!/bin/bash\n")
        f.write(f"mkdir -p \"{result_dir}\"\n")
        if solver == "ansys_cfx":
            f.write(
                f"cfx5solve -def \"{case_dir}/setup.def\" "
                f"-input \"{mesh_file}\" "
                f"-output \"{result_dir}/cfx_results.cse\"\n"
            )
        else:
            f.write(f"simpleFoam -case \"{case_dir}\"\n")
    os.chmod(script_path, 0o755)
    logging.info(f"CFD-Input-Skript erstellt: {script_path}")
    return script_path

def prepare_fem_input(case_dir, model_file, result_dir, solver="ansys"):
    """
    Erzeugt ein Shell-Skript zur FEM-Analyse, ohne den Solver aufzurufen.

    Parameter:
      case_dir   : Verzeichnis für FEM-Input (werden hier abgelegt)
      model_file : Pfad zur FEM-Modelldatei (.inp, .cdb)
      result_dir : Zielverzeichnis für FEM-Ergebnisse
      solver     : 'ansys' (erweitbar für andere Solver)

    Rückgabe:
      script_path: Pfad zum generierten Skript (run_fem.sh)
    """
    os.makedirs(case_dir, exist_ok=True)
    os.makedirs(result_dir, exist_ok=True)
    script_path = os.path.join(case_dir, "run_fem.sh")
    with open(script_path, "w") as f:
        f.write("#!/bin/bash\n")
        f.write(f"mkdir -p \"{result_dir}\"\n")
        if solver == "ansys":
            f.write(
                f"ansys -b -i \"{case_dir}/fem_input.inp\" "
                f"-o \"{result_dir}/fem_output.out\"\n"
            )
        else:
            logging.warning(f"Unbekannter FEM-Solver '{solver}', generiere leeres Skript.")
            f.write(f"# Befehl für Solver {solver} hier eintragen\n")
    os.chmod(script_path, 0o755)
    logging.info(f"FEM-Input-Skript erstellt: {script_path}")
    return script_path

def summarize_results(cfd_result_dir, fem_result_dir):
    """
    Platzhalter zum späteren Parsen der Solver-Ergebnisse:
      - CFD: max Druck, Geschwindigkeitsprofil
      - FEM: max Spannungen, Sicherheitsfaktoren

    Parameter:
      cfd_result_dir : Verzeichnis mit CFD-Ergebnissen
      fem_result_dir : Verzeichnis mit FEM-Ergebnissen

    Rückgabe:
      dict mit später zu füllenden Kennzahlen
    """
    summary = {}
    return summary