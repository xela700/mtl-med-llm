"""
Script to evaluate both classification and summarization models using sequested test data.
"""

from datasets import load_from_disk
from transformers import AutoConfig, AutoTokenizer, AutoModelForSequenceClassification, AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq
from transformers.trainer import Trainer
from peft import PeftModel, PeftConfig
from utils.config_loader import load_config
from model.evaluate_model import classification_compute_metric, SummarizationMetrics
from data.fetch_data import load_data
import os
import logging

config = load_config()
logger = logging.getLogger(__name__)

def classification_model_testing(test_data_path: str) -> None:
    """
    Evaluates BioBERT PEFT fine-tuned model using saved test data.

    Args:
        test_data_path (str): Path to the saved test data.
    
    Returns:
        None
    """

    if not os.path.exists(test_data_path):
        logger.error(f"No dataset available at path: {test_data_path}")
        raise FileNotFoundError(f"No dataset available at path: {test_data_path}")

    dataset = load_from_disk(test_data_path)

    peft_model_path = "model/saved_model/ft-biobert-large"

    labels = config["data"]["task_2"]["data_path"]
    num_labels = len(load_data(labels))

    peft_config = PeftConfig.from_pretrained(peft_model_path)
    base_config = AutoConfig.from_pretrained(peft_config.base_model_name_or_path)
    base_config.num_labels = num_labels
    base_config.problem_type = "multi_label_classification"

    tokenizer = AutoTokenizer.from_pretrained(peft_config.base_model_name_or_path)
    base_model = AutoModelForSequenceClassification.from_pretrained(peft_config.base_model_name_or_path, config=base_config)
    model = PeftModel.from_pretrained(base_model, peft_model_path)

    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        compute_metrics=classification_compute_metric
    )

    metrics = trainer.evaluate(eval_dataset=dataset)
    print(metrics)

def summarization_model_testing(test_data_path: str) -> None:
    """
    Evaluates BioBART PEFT fine-tuned model using saved test data.

    Args:
        test_data_path (str): Path to the saved test data.
    
    Returns:
        None
    """

    if not os.path.exists(test_data_path):
        logger.error(f"No dataset available at path: {test_data_path}")
        raise FileNotFoundError(f"No dataset available at path: {test_data_path}")

    dataset = load_from_disk(test_data_path)

    peft_model_path = "model/saved_model/ft-biobart-large"

    peft_config = PeftConfig.from_pretrained(peft_model_path)

    tokenizer = AutoTokenizer.from_pretrained(peft_config.base_model_name_or_path)
    base_model = AutoModelForSeq2SeqLM.from_pretrained(peft_config.base_model_name_or_path)
    model = PeftModel.from_pretrained(base_model, peft_model_path)

    compute_metrics = SummarizationMetrics(tokenizer=tokenizer)

    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True, model=model, label_pad_token_id=-100, pad_to_multiple_of=8)

    trainer = Trainer(
        model=model,
        processing_class=tokenizer,
        compute_metrics=compute_metrics,
        data_collator=data_collator
    )

    metrics = trainer.evaluate(eval_dataset=dataset)
    print(metrics)


if __name__ == "__main__":
    # classification_test_data = config["data"]["classification_test_data"]
    # classification_model_testing(classification_test_data)

    summarzation_test_data = config["data"]["summarization_test_data"]
    summarization_model_testing(summarzation_test_data)

