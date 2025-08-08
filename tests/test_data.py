"""
A simple module for examining datasets from MIMIC-IV
"""

from data.fetch_data import load_data
from datasets import load_from_disk
from utils.config_loader import load_config

def examine_classification_data():
    config = load_config()
    task_1_data = config["data"]["task_1"]["data_path"]

    classification_data = load_data(task_1_data)
    print("Number of pulled records for classification", len(classification_data))

    task_1_tokenized = config["data"]["task_1"]["tokenized_path"]
    classification_data_tokenized = load_from_disk(task_1_tokenized)

    print("Total classification dataset length:", len(classification_data_tokenized))

def examine_summarization_data():
    config = load_config()
    task_3_data = config["data"]["task_3"]["data_path"]

    summarization_data = load_data(task_3_data)
    print("Number of pulled records for summarization", len(summarization_data))

    task_3_real = config["data"]["task_3"]["real_data_path"]
    task_3_synthetic = config["data"]["task_3"]["synthetic_data_path"]

    real_data = load_data(task_3_real)
    synthetic_data = load_data(task_3_synthetic)

    print("Initial real records:", len(real_data), "Initial synthetic records:", len(synthetic_data))

    final_real = load_data(config["data"]["task_3"]["real_summary_path"])
    final_syn = load_data(config["data"]["task_3"]["synthetic_summary_path"])

    print("Total real records:", len(final_real), "Total synthetic records:", len(final_syn))

    combined = load_from_disk(config["data"]["task_3"]["tokenized_path"])
    print("Total for summary:", len(combined))

if __name__ == "__main__":
    # examine_classification_data()
    examine_summarization_data()