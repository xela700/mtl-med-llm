from sklearn.metrics import (
    f1_score, accuracy_score, hamming_loss,
    precision_score, recall_score, roc_auc_score
)
from evaluate import load
import numpy as np
import torch
import logging
from tqdm import tqdm

logger = logging.getLogger(__name__)

def classification_compute_metric(eval_preds):
    """
    Method for use in model training to evaluation performance. Includes accuracy, F1 (micro & macro), precision, recall, hamming loss and ROC-AUC (micro & macro)

    Args:
        eval_preds (transformers.EvalPrediction): logits and labels for computing metrics
    
    Returns:
        dict[str:float]: metrics based on evaluations
    """
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
        """
        Method for use in model training to evaluation performance. Includes ROUGE-L and BERT Score F1.

        Args:
            eval_preds (transformers.EvalPrediction): logits and labels for computing metrics
    
        Returns:
            dict[str:float]: metrics based on evaluations
        """
        predictions, labels = eval_preds
        labels = np.where(labels == -100, self.tokenizer.pad_token_id, labels)

        decoded_preds = self.tokenizer.batch_decode(predictions, skip_special_tokens=True)
        decoded_labels = self.tokenizer.batch_decode(predictions, skip_special_tokens=True)

        decoded_preds = [pred.strip() for pred in decoded_preds]
        decoded_labels = [label.strip() for label in decoded_labels]

        rouge_results = self.rouge.compute(predictions=decoded_preds, references=decoded_labels, rouge_types=["rougeL"], use_stemmer=True)
        bert_results = self.bertscore.compute(predictions=decoded_preds, references=decoded_labels, lang="en")

        return {
            "rougeL": rouge_results["rougeL"].mid.fmeasure * 100,
            "bertscore_f1": np.mean(bert_results["f1"]) * 100
        }

def rouge_metrics(model, dataset, tokenizer, batch_size=8, device="cuda", max_gen_length=256):
    """
    Used to evaluate the summarization model on ROUGE between epochs using the current checkpoint.
    Substitute for running evaluations during model training (GPU resource conservation)

    Args:
        model (transformers.AutoModelForSeq2SeqLM): model being evaluated
        dataset (Dataset): validation dataset
        tokenizer (transformers.AutoTokenizer): tokenizer associated with model
        batch_size (int): size of evaluation batch
        device (str): device for evaluation ("cuda" or "cpu")
        max_gen_length (int): max number of tokens generated for prediction
    
    Returns:
        dict[str:float]: rouge metrics
    """
    rouge = load("rouge")
    model.eval()
    model.to(device)
    preds, refs = [], []
    for i in tqdm(range(0, len(dataset), batch_size)):
        batch = dataset[i:i+batch_size]
        input_texts = batch["discharge_note"]
        reference_texts = batch["target"]

        inputs = tokenizer(
            input_texts,
            return_tensors="pt",
            truncation=True,
            max_length=1024,
            padding=True
        ).to(device)
        
        with torch.no_grad():
            outputs = model.generate(**inputs, 
                                     max_new_tokens=max_gen_length,
                                     do_sample=False,
                                     pad_token_id=tokenizer.pad_token_id)
        
        decoded_preds = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        decoded_refs = reference_texts

        preds.extend(decoded_preds)
        refs.extend(decoded_refs)

    result = rouge.compute(predictions=preds, references=refs, use_stemmer=True)
    return result

def intent_compute_metrics(eva_pred):
    """
    Method for use in model training to evaluation performance. Includes accuracy.

    Args:
        eval_preds (transformers.EvalPrediction): logits and labels for computing metrics
    
    Returns:
        dict[str:float]: metrics based on evaluations
    """
    accuracy = load("accuracy")

    logits, labels = eva_pred
    predictions = torch.tensor(logits).argmax(dim=-1)
    return accuracy.compute(predictions=predictions, references=labels)