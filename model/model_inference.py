"""
Script for running inference based on saved model weights. Used for both classification and summarization.
"""

from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForSeq2SeqLM, AutoConfig
from peft import PeftModel, PeftModelForSequenceClassification, PeftModelForSeq2SeqLM
from utils.config_loader import load_config
from data.fetch_data import load_data
from model.model_projection import Seq2SeqWProjection, CodelessWrapper
import torch.nn.functional as F
import json
import torch
import os
import logging

config = load_config()
logger = logging.getLogger(__name__)

def classification_prediction(text: str) -> list[str]:
    """
    Takes clinical note input and makes predictions of possible ICD-10 code labels using
    fine-tuned classification LLM.

    Args:
        text (str): Text to classify

    Returns:
        list[str]: predicted ICD-10 codes in list format
    """

    model_path = config["model"]["classification_model_temp"]

    wrapper = CodelessWrapper.load_custom(model_path)
    model = wrapper.model
    tokenizer = wrapper.tokenizer

    model.eval()

    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512).to(model.device)

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probs = torch.sigmoid(logits)

    threshold = 0.5 # modifiable prediction threshold

    predicted_ids = (probs > threshold).nonzero(as_tuple=True)[1].tolist()

    predicted_labels = [model.config.id2label[i] for i in predicted_ids]

    return predicted_labels

def summarization_prediction(text: str) -> str:
    """
    Takes clinical note input and returns a summary using the fine-tune summarization LLM.

    Args:
        text (str): Clinical note text to summarize

    Returns:
        str: Summary of clinical note
    """

    base_model = config["model"]["summarization_checkpoint"]
    model_path = config["model"]["summarization_model"]

    config = AutoConfig.from_pretrained(base_model)
    tokenizer = AutoTokenizer.from_pretrained(base_model)

    model = Seq2SeqWProjection.from_pretrained(base_model, config=config)

    proj_state = torch.load(os.path.join(model_path, "proj.pt"), map_location="cpu")
    model.proj.load_state_dict(proj_state)

    model = PeftModel.from_pretrained(model, os.path.join(model_path, "peft_model"))

    model.to("cuda" if torch.cuda.is_available() else "cpu")

    model.eval()

    inputs = tokenizer(text, return_tensors="pt", max_length=512, truncation=True).to(model.device)
    
    with torch.no_grad():
        output_ids = model.generate(**inputs, max_new_tokens=200, do_sample=False, pad_token_id=tokenizer.pad_token_id)

    return tokenizer.decode(output_ids[0], skip_special_tokens=True)

def intent_prediction(text: str) -> str:
    """
    Takes clinical note w/ intent for either classification or summarization and predicts intent.

    Args:
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

    Args:
        texts list[str]: text(s) to feed into pipeline and return either associated ICD-10 codes (classification) or a summary of the clinical note (summarization).
    
    Returns:
        None
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



