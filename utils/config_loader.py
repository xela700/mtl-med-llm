"""
This utility script is for loading the configuration set out in the YAML file
located in \"config/config.yaml\"
"""

import yaml
from pathlib import Path

def load_config(path: str = "config/config.yaml") -> dict:
    """
    Loads YAML configuration as a dictionary
    """

    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, "r") as file:
        return yaml.safe_load(file)


def verify_files(config: dict) -> list:
    """
    Verifies that the required model file exists.
    Will return a list of missing file names.
    """

    model_dir = Path(model_dir["dir"])
    missing = []

    for file in config.get("files", []):
        file_path = model_dir / file
        if not file_path.exists():
            missing.append(str(file_path))
            
    return missing