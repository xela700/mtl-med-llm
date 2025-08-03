"""
Main script with command line parser arguments to control which steps occur within the pipeline.

Current steps:
Fetch Data (Both for classification and summarization)
Preprocess Data for both classification and summarization
Training (with interim evaluation metrics) for classification and summarization
"""
import transformers
transformers.logging.set_verbosity_error()
import argparse
import pandas as pd
import os
from config.log_config import logging_setup
from utils.config_loader import load_config
from data.fetch_data import fetch_and_save_query, load_data
from data.preprocessing_data import ClassificationPreprocessor, SummarizationPreprocessor, SummarizationTargetCreation, IntentTargetingPreprocessor
from model.train_model import classification_model_training, summarization_model_training

def main(args: list[str]) -> None:
    config = load_config()
    if args.command == "fetch":
        if args.target == "classification":
            fetch_and_save_query(query=config["data"]["task_1"]["query"], save_path=config["data"]["task_1"]["data_path"])
            fetch_and_save_query(query=config["data"]["task_2"]["query"], save_path=config["data"]["task_2"]["data_path"])
        elif args.target == "summarization":
            fetch_and_save_query(query=config["data"]["task_3"]["query"], save_path=config["data"]["task_3"]["data_path"])
        elif args.target == "all":
            fetch_and_save_query(query=config["data"]["task_1"]["query"], save_path=config["data"]["task_1"]["data_path"])
            fetch_and_save_query(query=config["data"]["task_2"]["query"], save_path=config["data"]["task_2"]["data_path"])
            fetch_and_save_query(query=config["data"]["task_3"]["query"], save_path=config["data"]["task_3"]["data_path"])

    
    elif args.command == "preprocess":
        if args.target == "classification":
            base_data = load_data(config["data"]["task_1"]["data_path"])
            label_data = load_data(config["data"]["task_2"]["data_path"])
            label_ids = {code: i for i, code in enumerate(sorted(label_data["icd_code"].unique()))}
            checkpoint = config["model"]["classification_checkpoint"]

            preprocessor = ClassificationPreprocessor(
                checkpoint=checkpoint, 
                label_ids=label_ids, 
                text_col="discharge_note", 
                label_col="icd_codes", 
                cleaned_path=config["data"]["task_1"]["tokenized_path"]
                )

            model_save_dir = config["model"]["classification_model"]

            preprocessor.preprocess(base_data)
        
        elif args.target == "summary_generation": # Must be done before command "generation"
            base_data = load_data(config["data"]["task_3"]["data_path"])
            checkpoint = config["data"]["task_3"]["model"]
            clean_path = config["data"]["task_3"]["clean_path"]
            real_target_path = config["data"]["task_3"]["real_data_path"]
            synthetic_target_path = config["data"]["task_3"]["synthetic_data_path"]
            preprocessor = SummarizationTargetCreation(
                checkpoint=checkpoint, 
                text_col="discharge_note", 
                cleaned_path=clean_path, 
                real_target_path=real_target_path, 
                synthetic_target_path=synthetic_target_path
                )

            preprocessor.preprocess(base_data)
        
        elif args.target == "summarization": # Must be done after targets are generated
            base_data = load_data(config["data"]["task_3"]["clean_path"]) # Combined summaries
            tokenized_path = config["data"]["task_3"]["tokenized_path"]
            checkpoint = config["model"]["summarization_checkpoint"]
            preprocessor = SummarizationPreprocessor(
                checkpoint=checkpoint, 
                text_col="discharge_note", 
                target_col="target", 
                source_type_col="source_type", 
                cleaned_path=tokenized_path
                )

            model_save_dir = config["model"]["summarization_model"]

            preprocessor.preprocess(base_data, save_dir=model_save_dir)
        
        elif args.target == "intent_targeting":
            intent_path = config["data"]["task_4"]["data_path"]
            if os.path.exists(intent_path):
                intent_data = load_data(config["data"]["task_4"]["data_path"])
            else:
                base_data = load_data(config["data"]["task_3"]["data_path"])
                intent_data = base_data.sample(n=300, random_state=42) # Don't need thousands of samples to train small intent targeting model at this time.
                intent_data.to_parquet(intent_path)
            
            checkpoint = config["model"]["intent_checkpoint"]
            clean_path = config["data"]["task_4"]["tokenized_path"]
            
            preprocessor = IntentTargetingPreprocessor(
                checkpoint=checkpoint, 
                text_col="discharge_note", 
                cleaned_path=clean_path)

            preprocessor.preprocess(intent_data)

                

    
    elif args.command == "summary_generation":
        if args.target == "real":
            base_data = load_data(config["data"]["task_3"]["real_data_path"])
            checkpoint = checkpoint = config["data"]["task_3"]["model"]
            clean_path = config["data"]["task_3"]["clean_path"]
            real_target_path = config["data"]["task_3"]["real_data_path"]
            synthetic_target_path = config["data"]["task_3"]["synthetic_data_path"]
            generator = SummarizationTargetCreation(
                checkpoint=checkpoint, 
                text_col="discharge_note", 
                cleaned_path=clean_path, 
                real_target_path=real_target_path, 
                synthetic_target_path=synthetic_target_path
                )

            real_summary_path = config["data"]["task_3"]["real_summary_path"]

            generator.extract_real_target(base_data, save_path=real_summary_path)
        
        elif args.target == "synthetic":
            base_data = load_data(config["data"]["task_3"]["synthetic_data_path"])
            checkpoint = checkpoint = config["data"]["task_3"]["model"]
            clean_path = config["data"]["task_3"]["clean_path"]
            real_target_path = config["data"]["task_3"]["real_data_path"]
            synthetic_target_path = config["data"]["task_3"]["synthetic_data_path"]
            generator = SummarizationTargetCreation(
                checkpoint=checkpoint, 
                text_col="discharge_note", 
                cleaned_path=clean_path, 
                real_target_path=real_target_path, 
                synthetic_target_path=synthetic_target_path
                )

            synthetic_summary_path = config["data"]["task_3"]["synthetic_summary_path"]

            generator.generate_synthetic_targets(base_data, save_path=synthetic_summary_path, resume=False)

        elif args.target == "synthetic_continue": # If generation is interrupted
            base_data = load_data(config["data"]["task_3"]["synthetic_data_path"])
            checkpoint = checkpoint = config["data"]["task_3"]["model"]
            clean_path = config["data"]["task_3"]["clean_path"]
            real_target_path = config["data"]["task_3"]["real_data_path"]
            synthetic_target_path = config["data"]["task_3"]["synthetic_data_path"]
            generator = SummarizationTargetCreation(
                checkpoint=checkpoint, 
                text_col="discharge_note", 
                cleaned_path=clean_path, 
                real_target_path=real_target_path, 
                synthetic_target_path=synthetic_target_path
                )

            synthetic_summary_path = config["data"]["task_3"]["synthetic_summary_path"]

            generator.generate_synthetic_targets(base_data, save_path=synthetic_summary_path, resume=True)

        elif args.target == "combine":
            checkpoint = checkpoint = config["data"]["task_3"]["model"]
            clean_path = config["data"]["task_3"]["clean_path"]
            real_target_path = config["data"]["task_3"]["real_data_path"]
            synthetic_target_path = config["data"]["task_3"]["synthetic_data_path"]
            generator = SummarizationTargetCreation(
                checkpoint=checkpoint, 
                text_col="discharge_note", 
                cleaned_path=clean_path, 
                real_target_path=real_target_path, 
                synthetic_target_path=synthetic_target_path
                )

            real_summary_data = load_data(config["data"]["task_3"]["real_summary_path"])
            synthetic_summary_data = load_data(config["data"]["task_3"]["synthetic_summary_path"])

            generator.combine_data(real_summary_dataset=real_summary_data, synthetic_summary_dataset=synthetic_summary_data)
    
    elif args.command == "training":
        if args.target == "classification":
            tokenized_data_dir = config["data"]["task_1"]["tokenized_path"]
            label_dir = config["data"]["task_2"]["data_path"]
            checkpoint = config["model"]["classification_checkpoint"]
            model_weights_dir = config["model"]["classification_model"]
            training_checkpoints = config["model"]["classification_training_checkpoints"]
            test_data_dir = config["data"]["classification_test_data"]

            classification_model_training(
                data_dir=tokenized_data_dir, 
                label_dir=label_dir, 
                checkpoint=checkpoint, 
                save_dir=model_weights_dir, 
                training_checkpoint_dir=training_checkpoints,
                test_data_dir=test_data_dir
                )
        
        elif args.target == "summarization":
            tokenized_data_dir = config["data"]["task_3"]["tokenized_path"]
            checkpoint = config["model"]["summarization_checkpoint"]
            model_weights_dir = config["model"]["summarization_model"]
            training_checkpoints = config["model"]["summarization_training_checkpoints"]
            test_data_dir = config["data"]["summarization_test_data"]

            summarization_model_training(
                data_dir=tokenized_data_dir,
                checkpoint=checkpoint,
                save_dir=model_weights_dir,
                training_checkpoint_dir=training_checkpoints,
                test_data_dir=test_data_dir
            )




if __name__ == "__main__":
    logging_setup()
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command")

    data_parser = subparsers.add_parser("fetch")
    data_parser.add_argument("target", choices=["classification", "summarization", "all"], help="Specify which dataset(s) to pull from BigQuery")

    generation_parser = subparsers.add_parser("summary_generation")
    generation_parser.add_argument("target", choices=["real", "synthetic", "synthetic_continue", "combine"], help="Specify target creation for summary task. Combine unites both types of targets into one dataset.")

    preprocess_parser = subparsers.add_parser("preprocess")
    preprocess_parser.add_argument("target", choices=["classification", "summary_generation", "summarization", "intent_targeting"], help="Specify which preprocess pipeline(s) to initiate")

    training_parser = subparsers.add_parser("training")
    training_parser.add_argument("target", choices=["classification", "summarization"], help="Denote which training pipeline is being used.")

    args = parser.parse_args()

    if args.command is None:
        print("No command provided, Skipping execution.")
    else:
        main(args)