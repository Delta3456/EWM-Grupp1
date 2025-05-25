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
from typing import List, Tuple
import yaml
import logging

# Minimum required Python version
MIN_PYTHON = (3, 8)

def check_python_version() -> None:
    """Überprüft die Python-Version."""
    if sys.version_info < MIN_PYTHON:
        sys.exit(f"Python {MIN_PYTHON[0]}.{MIN_PYTHON[1]} oder höher wird benötigt.")

def validate_config(config: dict) -> None:
    """Validiert die Konfigurationsdatei auf Vollständigkeit."""
    required_sections = ['preprocessing', 'generator', 'output']
    for section in required_sections:
        if section not in config:
            raise ValueError(f"Fehlender Abschnitt '{section}' in der Konfigurationsdatei")

def run_step(name: str, cmd: List[str]) -> float:
    """Führt einen externen Befehl aus und misst die Dauer."""
    print(f"\n>> Starte Schritt: {name}")
    start = time.perf_counter()
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        if result.stdout:
            logging.info(result.stdout)
    except subprocess.CalledProcessError as e:
        logging.error(f"Fehler in Schritt {name}: {e.stderr}")
        sys.exit(e.returncode)
    duration = time.perf_counter() - start
    print(f"✓ {name} abgeschlossen in {duration:.2f} s")
    return duration

def main() -> None:
    # Logging-Konfiguration
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s: %(message)s'
    )
    
    # Python-Version prüfen
    check_python_version()
    
    config_path = "config.yaml"
    if not os.path.isfile(config_path):
        logging.error(f"Konfigurationsdatei '{config_path}' nicht gefunden.")
        sys.exit(1)

    try:
        # Konfiguration laden und validieren
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        validate_config(config)
    except Exception as e:
        logging.error(f"Fehler beim Laden der Konfiguration: {e}")
        sys.exit(1)

    # Sichert, dass das Ausgabeverzeichnis existiert
    os.makedirs("results", exist_ok=True)

    total_start = time.perf_counter()

    # 1) Preprocessing
    run_step(
        "Preprocessing",
        ["python", "preprocessing.py", "--config", config_path]
    )

    # 2) TSR-Scan
    run_step(
        "TSR-Scan",
        ["python", "tsr_scan.py", "--config", config_path,
         "--design", "results/design_params.json"]
    )

    # 3) BEM
    run_step(
        "BEM",
        ["python", "bem.py", "--config", config_path,
         "--design", "results/design_params.json",
         "--tsr",    "results/tsr_scan.json"]
    )

    # 4) Profile Selection
    run_step(
        "Profile Selection",
        ["python", "profile_selection.py", "--config", config_path,
         "--bem", "results/bem_results.json"]
    )

    total = time.perf_counter() - total_start
    print(f"\n■ Gesamtzeit: {total:.2f} s")

if __name__ == "__main__":
    main()