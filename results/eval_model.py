"""
Performs evaluation of trained models on test datasets.
"""

from transformers import AutoTokenizer
from model.model_projection import Seq2SeqWProjection
from peft import PeftModel
from datasets import Dataset
from model.evaluate_model import rouge_metrics
from utils.config_loader import load_config
import os
import json

config = load_config()

def summarization_evaluation(base_model: str, model_path: str, test_metric_dir: str, test_dataset: Dataset, batch_size: int = 8, device: str = "cuda") -> None:
    """
    Evaluates the summarization model on the test dataset. Dumps results to json file.

    Args:
        base_model (str): base model checkpoint
        model_path (str): path to fine-tuned model
        test_dataset (Dataset): test dataset for evaluation
        batch_size (int): evaluation batch size
        device (str): device for evaluation ("cuda" or "cpu")
    
    Returns:
        None
    """

    tokenizer = AutoTokenizer.from_pretrained(base_model)
    model = Seq2SeqWProjection.from_pretrained(base_model)
    peft_model = PeftModel.from_pretrained(model, os.path.join(model_path, "peft_model"))

    metrics = rouge_metrics(peft_model, test_dataset, tokenizer, batch_size=batch_size, device=device)

    with open(os.path.join(test_metric_dir, "summarization_test_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=4)

if __name__ == "__main__":
    base_model = config["model"]["summarization_checkpoint"]
    model_path = config["model"]["summarization_model"]
    test_data_path = config["data"]["summarization_test_data"]
    test_metric_dir = config["results"]["test_summarization"]

    test_dataset = Dataset.load_from_disk(test_data_path)

    device = "cuda" if os.getenv("CUDA_AVAILABLE", "true").lower() == "true" else "cpu"

    summarization_evaluation(base_model, model_path, test_metric_dir, test_dataset, batch_size=8, device=device)