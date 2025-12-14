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
import json
import logging
from config.log_config import logging_setup
from utils.config_loader import load_config
from data.fetch_data import fetch_and_save_query, load_data
from data.preprocessing_data import ClassificationPreprocessor, SummarizationPreprocessor, SummarizationTargetCreation, IntentDataPreprocessor
from model.train_model import classification_model_training, summarization_model_training, intent_model_training
from data.intent_prompt_data import create_intent_dataset

logger = logging.getLogger(__name__)

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
        elif args.target == "intent_targeting":
            create_intent_dataset(
                note_path=config["data"]["task_3"]["data_path"],
                text_col="discharge_note",
                save_dir=config["data"]["task_4"]["temp_intent_data_path"],
                limit=500
            )

    
    elif args.command == "preprocess":
        if args.target == "classification_base":
            base_data = load_data(config["data"]["task_1"]["data_path"])
            label_data = load_data(config["data"]["task_2"]["temp_data_path"]) # Modifying this for initial training. Uses only 50 ICD-10 codes.
            label_map_path = config["data"]["task_2"]["label_map_path"]
            labels = sorted(label_data["icd_code"].unique())
            label2id = {label: i for i, label in enumerate(labels)}
            id2label = {i: label for i, label in enumerate(labels)}
            checkpoint = config["model"]["classification_checkpoint"]

            # For use in training to set up config
            with open(label_map_path, "w") as f:
                json.dump({"label2id": label2id, "id2label": id2label}, f)

            preprocessor = ClassificationPreprocessor(
                checkpoint=checkpoint, 
                label_ids=label2id, 
                text_col="discharge_note", 
                label_col="icd_codes", 
                cleaned_path=config["data"]["task_1"]["temp_tokenized_path"] # Modifying this for initial training. Uses only 50 ICD-10 codes.
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
        
        elif args.target == "intent_targeting": # Modified intent targeting preprocessing
            intent_data = load_data(config["data"]["task_4"]["temp_intent_data_path"])
            label_dir = config["data"]["task_4"]["temp_intent_label_path"]
            intent_ds_path = config["data"]["task_4"]["temp_intent_dataset_path"]
            checkpoint = config["model"]["intent_checkpoint"]

            preprocessor = IntentDataPreprocessor(tokenizer=checkpoint)

            preprocessor.build_dataset(intent_data=intent_data, save_path=intent_ds_path)
            preprocessor.save_label_mappings(save_path=label_dir)
                

    
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
            tokenized_data_dir = config["data"]["task_1"]["temp_tokenized_path"] # modified to use fewer labels for initial training.
            label_dir = config["data"]["task_2"]["temp_data_path"] # modified to use fewer labels for initial training.
            label_map_path = config["data"]["task_2"]["label_map_path"]
            checkpoint = config["model"]["classification_checkpoint"]
            model_weights_dir = config["model"]["classification_model_temp"] # modified to use fewer labels for initial training.
            training_checkpoints = config["model"]["classification_training_checkpoints_temp"] # modified to use fewer labels for initial training.
            test_data_dir = config["data"]["classification_test_data_temp"] # modified to use fewer labels for initial training.
            metrics_dir = config["results"]["classification_wo_code_LORA_high_cap_Mixed_MoE_8_save_det"]
            num_runs = args.num_runs

            classification_model_training(
                data_dir=tokenized_data_dir, 
                active_labels_dir=label_dir,
                label_mapping_dir=label_map_path, 
                checkpoint=checkpoint, 
                save_dir=model_weights_dir, 
                training_checkpoint_dir=training_checkpoints,
                test_data_dir=test_data_dir,
                metric_dir=metrics_dir,
                run_number=num_runs
                )
        
        elif args.target == "summarization":
            print(f"Running Summarization model for {args.num_runs} runs")
            for i in range(args.num_runs):
                tokenized_data_dir = config["data"]["task_3"]["tokenized_path"]
                checkpoint = config["model"]["summarization_checkpoint"]
                model_weights_dir = config["model"]["summarization_model"]
                training_checkpoints = config["model"]["summarization_training_checkpoints"]
                test_data_dir = config["data"]["summarization_test_data"]
                metric_dir = f"results/reporting/summarization_higher_samp_w_proj_high_lora_2/summarization_rouge_results_run_{i+1}.json"

                summarization_model_training(
                    data_dir=tokenized_data_dir,
                    checkpoint=checkpoint,
                    save_dir=model_weights_dir,
                    training_checkpoint_dir=training_checkpoints,
                    test_data_dir=test_data_dir,
                    metric_dir=metric_dir
                )
        
        elif args.target == "intent_targeting":
            # Modified to load train and validation data prepared at preprocessing. Data no longer split directly prior to training.
            intent_dataset = config["data"]["task_4"]["temp_intent_dataset_path"]
            intent_label_map = config["data"]["task_4"]["temp_intent_label_path"]
            checkpoint = config["model"]["intent_checkpoint"]
            model_weights_dir = config["model"]["intent_model"]
            training_checkpoints = config["model"]["intent_training_checkpoints"]
            metric_dir = config["results"]["intent_targeting_mod_data_low_lr"]

            intent_model_training(
                dataset_dir=intent_dataset,
                label_dir=intent_label_map,
                checkpoint=checkpoint,
                save_dir=model_weights_dir,
                training_checkpoint_dir=training_checkpoints,
                metric_dir=metric_dir
            )




if __name__ == "__main__":
    logging_setup()
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command")

    data_parser = subparsers.add_parser("fetch")
    data_parser.add_argument("target", choices=["classification", "summarization", "all", "intent_targeting"], help="Specify which dataset(s) to pull from BigQuery/generate")

    generation_parser = subparsers.add_parser("summary_generation")
    generation_parser.add_argument("target", choices=["real", "synthetic", "synthetic_continue", "combine"], help="Specify target creation for summary task. Combine unites both types of targets into one dataset.")

    preprocess_parser = subparsers.add_parser("preprocess")
    preprocess_parser.add_argument("target", choices=["classification_base", "summary_generation", "summarization", "intent_targeting"], help="Specify which preprocess pipeline(s) to initiate")

    training_parser = subparsers.add_parser("training")
    training_parser.add_argument("target", choices=["classification", "summarization", "intent_targeting"], help="Denote which training pipeline is being used.")
    training_parser.add_argument("num_runs", type=int, default=1, nargs="?", help="Number of runs the model will both train and evaluate (optional, default 1)")

    args = parser.parse_args()

    if args.command is None:
        print("No command provided, Skipping execution.")
    else:
        main(args)