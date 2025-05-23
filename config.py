import os
import yaml

def load_config(path="config.yaml"):
    """
    Liest die Konfigurationsdatei im YAML-Format ein und gibt ein Dictionary zurück.
    Mit yaml.safe_load werden nur sichere Konstrukte geladen.
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Konfigurationsdatei nicht gefunden: {path}")
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    return cfg