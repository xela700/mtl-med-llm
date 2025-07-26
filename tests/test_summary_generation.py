"""
Script to test summary generation on smaller dataset
"""

from data.fetch_data import load_data
from data.preprocessing_data import SummarizationTargetCreation
from utils.config_loader import load_config
from config.log_config import logging_setup
import pandas as pd

def test_generation() -> None:
    """
    Tests synthetic summary generation on a smaller amount of data to confirm effectiveness.
    """

    config = load_config()

    base_data = load_data(config["data"]["task_3"]["synthetic_data_path"])
    test_data = base_data.head(2)
    checkpoint = checkpoint = config["data"]["task_3"]["model"]
    clean_path = config["data"]["task_3"]["clean_path"]
    real_target_path = config["data"]["task_3"]["real_data_path"]
    synthetic_target_path = config["data"]["task_3"]["synthetic_data_path"]
    generator = SummarizationTargetCreation(checkpoint=checkpoint, text_col="discharge_note", cleaned_path=clean_path, real_target_path=real_target_path, synthetic_target_path=synthetic_target_path)

    synthetic_summary_path = "tests/test_data/synthetic_summaries"

    generator.generate_synthetic_targets(test_data, save_path=synthetic_summary_path, chunk_size=1, resume=False)

def print_generation() -> None:
    """
    Prints created summaries for visual testing
    """

    summary_data = load_data("tests/test_data/synthetic_summaries")

    print(summary_data.shape)
    print(summary_data.head(2))
    val = summary_data.at[0, "target"]
    print(val)


if __name__ == "__main__":
    logging_setup()
    # test_generation()
    print_generation()