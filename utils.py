# utils.py

"""
utils.py

Enthält Hilfsfunktionen und -klassen:
- Logging-Setup für konsistente Ausgaben.
- Einheiten-Konvertierung (optional).
- Interpolationshelfer.
- Timer-Dekorator für Performance-Analysen.
"""

import logging
import time
from functools import wraps
import numpy as np
import os

def setup_logging(log_level=logging.INFO, log_file=None):
    """
    Konfiguriert das Logging-Modul.
    - log_level: Logging-Level (z.B. DEBUG, INFO).
    - log_file: Optionaler Pfad für eine Logdatei.
    """
    handlers = [logging.StreamHandler()]
    if log_file:
        handlers.append(logging.FileHandler(log_file))
    logging.basicConfig(level=log_level,
                        format="%(asctime)s [%(levelname)s] %(message)s",
                        handlers=handlers)

def timer(func):
    """
    Dekorator, um die Laufzeit einer Funktion zu messen und im Log zu senden.
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        logging.info(f"Function {func.__name__} took {end - start:.3f}s")
        return result
    return wrapper

def deg2rad(deg):
    """
    Konvertiert Winkel in Grad zu Radiant.
    """
    return deg * (np.pi / 180.0)

def rad2deg(rad):
    """
    Konvertiert Winkel in Radiant zu Grad.
    """
    return rad * (180.0 / np.pi)

def ensure_dir(path):
    """
    Erstellt das Verzeichnis, falls es nicht existiert.
    """
    if not os.path.isdir(path):
        os.makedirs(path, exist_ok=True)