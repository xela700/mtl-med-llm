"""
Script for running inference based on saved model weights. Used for both classification and summarization.
"""

from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import PeftModelForSequenceClassification, PeftModelForSeq2SeqLM
from utils.config_loader import load_config
from data.fetch_data import load_data
import torch.nn.functional as F
import json
import torch
import logging

config = load_config()
logger = logging.getLogger(__name__)

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
    labels = label_data["icd_code"].tolist() # Need to move saving the label list to part of preprocessing (json file)

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

def intent_prediction(text: str) -> str:
    """
    Takes clinical note w/ intent for either classification or summarization and predicts intent.

    Parameters:
    text (str): Clinical note to infer intent.

    Returns:
    str: Intent label (classification or summarization currently)
    """

    model_path = config["model"]["intent_model"]

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)

    with open("data/cleaned_data/intent_id2label.json", "r") as file:
        id2label = {int(k): v for k, v in json.load(file).items()}
    
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(model.device)

    with torch.no_grad():
        logits = model(**inputs).logits
        probs = F.softmax(logits, dim=-1)
        prediction = torch.argmax(probs, dim=-1).item()
        confidence = probs[0, prediction].item()
    
    logger.info(f"Intent prediction {id2label[prediction]}. Confidence: {confidence:.4f}")
    
    return id2label[prediction]


def model_routing_pipeline(texts: list[str]) -> None:
    """
    Takes text or list of texts through pipeline.
    Intent prediction -> either classification or summarization -> output

    Parameters:
    texts list[str]: text(s) to feed into pipeline and return either associated ICD-10 codes (classification) or
    a summary of the clinical note (summarization).
    """

    if isinstance(texts, str):
        inputs = [texts]
    elif isinstance(texts, list) and all(isinstance(item, str) for item in texts):
        inputs = texts
    else:
        logger.error("Input must be a string or list of strings")
        raise ValueError("Input must be a string or list of strings")
    
    for input in inputs:

        intent = intent_prediction(input)

        if intent == "classification":
            logger.info("Intent to classify ICD codes found.")
            predicted_codes = classification_prediction(input)
            print(f"Predicted ICD-10 codes for clinical note: {predicted_codes}")
        
        elif intent == "summarization":
            logger.info("Intent to summarize clinical note found.")
            generated_summary = summarization_prediction(input)
            print(f"Following summary generated: {generated_summary}")

        else: # Should not happen
            logger.error(f"Intent target must be either classification or summarization. Got {intent}.")
            raise ValueError("Invalid intent target found.")



