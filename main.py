"""
Main script with command line parser arguments to control which steps occur within the pipeline.

Current steps:
Fetch Data (Both for classification and summarization)
Preprocess Data
"""

import argparse
from utils.config_loader import load_config
from data.fetch_data import fetch_and_save_query, load_data
import data.preprocessing_data

def main(args: list[str]) -> None:
    config = load_config()
    if args.command == "fetch":
        if args.target == "classification":
            fetch_and_save_query(query=config["data"]["task_1"]["query"], save_path=config["data"]["task_1"]["data_path"])
            fetch_and_save_query(query=config["data"]["task_2"]["query"], save_path=config["data"]["task_2"]["data_path"])
        elif args.target == "summarization":
            fetch_and_save_query(query=config["data"]["task_3"]["query"], save_path=config["data"]["task_3"]["data_path"])
        elif args.target == "full":
            fetch_and_save_query(query=config["data"]["task_1"]["query"], save_path=config["data"]["task_1"]["data_path"])
            fetch_and_save_query(query=config["data"]["task_2"]["query"], save_path=config["data"]["task_2"]["data_path"])
            fetch_and_save_query(query=config["data"]["task_3"]["query"], save_path=config["data"]["task_3"]["data_path"])
    
    elif args.command == "preprocess":
        if args.target == "classification":
            pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command")
    data_parser = subparsers.add_parser("fetch")
    data_parser.add_argument("target", choices=["classification", "summarization", "full"], help="Specify which dataset(s) to pull from BigQuery")
    preprocess_parser = subparsers.add_parser("preprocess")
    preprocess_parser.add_argument("target", choices=["classification", "summarization", "full"], help="Specify which preprocess pipeline(s) to initiate")