from sklearn.metrics import (
    f1_score, accuracy_score, hamming_loss,
    precision_score, recall_score, roc_auc_score
)
from evaluate import load
import numpy as np
import torch
import logging

logger = logging.getLogger(__name__)

def classification_compute_metric(eval_preds):
    logits, labels = eval_preds

    probs = torch.sigmoid(torch.tensor(logits)).numpy()
    preds = (probs > 0.5).astype(int)
    labels = np.array(labels)

    metrics = {}

    metrics["accuracy"] = accuracy_score(labels, preds)
    metrics["f1_micro"] = f1_score(labels, preds, average="micro", zero_division=0)
    metrics["f1_macro"] = f1_score(labels, preds, average="macro", zero_division=0)
    metrics["precision_micro"] = precision_score(labels, preds, average="micro", zero_division=0)
    metrics["recall_micro"] = recall_score(labels, preds, average="micro", zero_division=0)
    metrics["hamming_loss"] = hamming_loss(labels, preds)

    try:
        metrics["roc_auc_micro"] = roc_auc_score(labels, preds, average="micro")
        metrics["roc_auc_macro"] = roc_auc_score(labels, preds, average="macro")
    except ValueError as ve:
        logger.error(f"ROC-AUC not computed: {ve}")
        metrics["roc_auc_micro"] = None
        metrics["roc_auc_macro"] = None
    
    return metrics

class SummarizationMetrics:

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.rouge = load("rouge")
        self.bertscore = load("bertscore")

    def __call__(self, eval_preds):
        predictions, labels = eval_preds
        labels = np.where(labels == -100, self.tokenizer.pad_token_id, labels)

        decoded_preds = self.tokenizer.batch_decode(predictions, skip_special_tokens=True)
        decoded_labels = self.tokenizer.batch_decode(predictions, skip_special_tokens=True)

        decoded_preds = [pred.strip() for pred in decoded_preds]
        decoded_labels = [label.strip() for label in decoded_labels]

        rouge_results = rouge.compute(predictions=decoded_preds, references=decoded_labels, rouge_types=["rougeL"], use_stemmer=True)
        bert_results = bertscore.compute(predictions=decoded_preds, references=decoded_labels, lang="en")

        return {
            "rougeL": rouge_results["rougeL"].mid.fmeasure * 100,
            "bertscore_f1": np.mean(bert_results["f1"]) * 100
        }