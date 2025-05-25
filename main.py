#!/usr/bin/env python3
"""
main.py

Orchestriert alle Schritte:
  1) Preprocessing
  2) TSR-Scan
  3) BEM
  4) Profile Selection

Misst und gibt für jeden Schritt die benötigte Zeit aus.
Liest automatisch 'config.yaml' im aktuellen Verzeichnis.
"""

import subprocess
import sys
import time
import os

def run_step(name, cmd):
    """Führt einen externen Befehl aus und misst die Dauer."""
    print(f"\n>> Starte Schritt: {name}")
    start = time.perf_counter()
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"✗ Schritt {name} fehlgeschlagen (Code {e.returncode})")
        sys.exit(e.returncode)
    duration = time.perf_counter() - start
    print(f"✓ {name} abgeschlossen in {duration:.2f} s")
    return duration

def main():
    config = "config.yaml"
    if not os.path.isfile(config):
        print(f"Konfigurationsdatei '{config}' nicht gefunden.")
        sys.exit(1)

    # Sichert, dass das Ausgabeverzeichnis existiert
    os.makedirs("results", exist_ok=True)

    total_start = time.perf_counter()

    # 1) Preprocessing
    run_step(
        "Preprocessing",
        ["python", "preprocessing.py", "--config", config]
    )

    # 2) TSR-Scan
    run_step(
        "TSR-Scan",
        ["python", "tsr_scan.py", "--config", config,
         "--design", "results/design_params.json"]
    )

    # 3) BEM
    run_step(
        "BEM",
        ["python", "bem.py", "--config", config,
         "--design", "results/design_params.json",
         "--tsr",    "results/tsr_scan.json"]
    )

    # 4) Profile Selection
    run_step(
        "Profile Selection",
        ["python", "profile_selection.py", "--config", config,
         "--bem", "results/bem_results.json"]
    )

    total = time.perf_counter() - total_start
    print(f"\n■ Gesamtzeit: {total:.2f} s")

if __name__ == "__main__":
    main()