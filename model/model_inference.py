"""
Script for running inference based on saved model weights. Used for both classification and summarization.
"""

from transformers import AutoTokenizer
from peft import PeftModelForSequenceClassification, PeftModelForSeq2SeqLM
from utils.config_loader import load_config
from data.fetch_data import load_data
import torch.nn.functional as F

config = load_config()

def classification_prediction(text: str) -> list[str]:
    """
    Takes clinical note input and makes predictions of possible ICD-10 code labels using
    fine-tuned classification LLM.

    Parameters:
    text (str): Text to classify

    Returns:
    list[str] -> predicted ICD-10 codes in list format
    """

    peft_model_path = config["model"]["classification_model"]

    tokenizer = AutoTokenizer.from_pretrained(peft_model_path)
    model = PeftModelForSequenceClassification.from_pretrained(peft_model_path)

    inputs = tokenizer(text, return_tensor="pt").to(model.device)
    outputs = model(**inputs)

    logits = outputs.logits
    probs = F.sigmoid(logits)

    predicted_labels = (probs > 0.5).int()

    label_data_path = config["data"]["task_2"]["data_path"] # path to full list of labels in str form
    label_data = load_data(label_data_path)
    labels = label_data["icd_code"].tolist() # Need to more saving the label list to part of preprocessing (json file)

    predictions = predicted_labels[0].tolist()
    predicted_classes = [labels[i] for i, val in enumerate(predictions) if val == 1]

    return predicted_classes

def summarization_prediction(text: str) -> str:
    """
    Takes clinical note input and returns a summary using the fine-tune summarization LLM.

    Parameters:
    text (str): Clinical note text to summarize

    Returns:
    str -> Summary of clinical note
    """

    peft_model_path = config["model"]["summarization_model"]
    
    tokenizer = AutoTokenizer.from_pretrained(peft_model_path)
    model = PeftModelForSeq2SeqLM.from_pretrained(peft_model_path)

    inputs = tokenizer(text, return_tensor="pt").to(model.device)
    summary = model.generate(**inputs, max_new_tokens=200)

    return tokenizer.decode(summary[0], skip_special_tokens=True)
