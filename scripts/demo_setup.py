"""
This script is for extracting the current model components to the correct
paths for inference based on the config.yaml settings.

If you have not already done so, please place 'demo_models.zip' in the project root directory.
This zip file contains the fine-tuned model weights for demoing the model pipeline.
"""

import zipfile
import pathlib
import sys

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
MODEL_DIR = PROJECT_ROOT / "model"
ZIP_PATH = PROJECT_ROOT / "demo_models.zip"

def main():

    if not ZIP_PATH.exists():
        print(f"ERROR: {ZIP_PATH} not found. Please place 'demo_models.zip' in the project root directory.")
        sys.exit(1)

    if (MODEL_DIR / "saved_model").exists():
        print(f"Demo models already exist in {MODEL_DIR / 'saved_model'}. Setup skipped.")
        return
    
    print(f"Extracting demo models into {MODEL_DIR}...")

    with zipfile.ZipFile(ZIP_PATH, 'r') as zip_ref:
        zip_ref.extractall(MODEL_DIR)
    
    expected = MODEL_DIR / "saved_model"
    if not expected.exists():
        print(f"ERROR: Extraction failed. saved_model was not created.")
        sys.exit(1)
    
    print("Extraction complete. Demo models are set up for inference.")

if __name__ == "__main__":
    main()
