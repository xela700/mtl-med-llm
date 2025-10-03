"""
Module to set up training tasks for all NLP models.
"""

from transformers import AutoModelForSequenceClassification, AutoModelForSeq2SeqLM, AutoTokenizer, AutoConfig, DataCollatorWithPadding, DataCollatorForSeq2Seq, Seq2SeqTrainer, Seq2SeqTrainingArguments
from transformers.training_args import TrainingArguments
from transformers.trainer import Trainer
from transformers.modeling_outputs import SequenceClassifierOutput
from peft import get_peft_model, LoraConfig, TaskType
from datasets import load_from_disk
from data.fetch_data import load_data
from model.evaluate_model import classification_compute_metric, SummarizationMetrics, intent_compute_metrics, rouge_metrics
import torch
import numpy as np
import json
import logging
from torch import Tensor
from torch.utils.data import DataLoader
from datasets import Dataset
from typing import Union, Dict, Tuple

logger = logging.getLogger(__name__)

def classification_model_training(data_dir: str, label_mapping_dir: str, active_labels_dir: str, checkpoint: str, save_dir: str, training_checkpoint_dir: str, test_data_dir: str) -> None:
    """
    Training method for fine-tuning a pre-trained encoder model on ICD-10 code
    classification task.

    Args:
        data_dir (str): path to dataset being used for training
        label_map_dir (str): path to label mapping (id2label and label2id)
        active_labels_dir (str): path to dictionary of active labels
        checkpoint (str): pre-trained HuggingFace model used for training
        training_checkpoint_dir (str): directory to save in-training model config
        save_dir (str): path to saved PEFT weights for specified task
        test_data_dir (str): path to save test data to
    
    Returns:
        None
    """

    dataset = load_from_disk(data_dir)
    train_val_test = dataset.train_test_split(test_size=0.2, seed=42)
    train_val = train_val_test["train"].train_test_split(test_size=0.1, seed=42)

    train_dataset = train_val["train"]
    val_dataset = train_val["test"]
    test_dataset = train_val_test["test"] # For use when evaluating model performance after training

    test_dataset.save_to_disk(test_data_dir)

    try:
        with open(label_mapping_dir) as f:
            mappings = json.load(f)
    except FileNotFoundError:
        logger.error("JSON file for mapping does not exist. Must preprocess data before training.")

    active_labels = load_data(active_labels_dir)
    label2id = mappings["label2id"]
    active_labels = list(active_labels["icd_code"])
    id2label = {int(k): v for k, v in mappings["id2label"].items()}
    num_labels = len(label2id)
    active_label_indices = [label2id[label] for label in active_labels]
    mask = [1 if i in active_label_indices else 0 for i in range(num_labels)]

    config = AutoConfig.from_pretrained(checkpoint)
    config.num_labels = num_labels
    config.label2id = label2id
    config.id2label = id2label
    config.problem_type = "multi_label_classification"

    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    model = AutoModelForSequenceClassification.from_pretrained(checkpoint, config=config)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    lora_config = LoraConfig( # PERF tuning
        r=8,
        lora_alpha=16,
        target_modules=["query", "value"],
        lora_dropout=0.1,
        bias="none",
        task_type=TaskType.SEQ_CLS
    )

    model = get_peft_model(model=model, peft_config=lora_config)

    training_args = TrainingArguments(
        output_dir=training_checkpoint_dir,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=10,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_dir="./logs",
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        greater_is_better=True
    )

    pos_weight = compute_pos_weight(train_dataset)

    class WeightedLossTrainer(Trainer):
        """
        Custom Trainer subclass intended to help tackle ICD-10 class imbalance in MIMIC-IV dataset.
        Only used with trainer for classification. 
        """

        def __init__(self, pos_weight=None, active_label_mask=None, model = None, args = None, data_collator = None, train_dataset = None, eval_dataset = None, processing_class = None, model_init = None, compute_loss_func = None, compute_metrics = None, callbacks = None, optimizers = (None, None), optimizer_cls_and_kwargs = None, preprocess_logits_for_metrics = None):
            super().__init__(model, args, data_collator, train_dataset, eval_dataset, processing_class, model_init, compute_loss_func, compute_metrics, callbacks, optimizers, optimizer_cls_and_kwargs, preprocess_logits_for_metrics)
            
            self.pos_weight = (
                torch.tensor(pos_weight, dtype=torch.float) if pos_weight is not None else None
            )

            self.active_label_mask = (
                torch.tensor(active_label_mask, dtype=torch.float) if active_label_mask is not None else None
            )

            self.criterion = torch.nn.BCEWithLogitsLoss(
                pos_weight=self.pos_weight,
                reduction="none"
            )  

        def compute_loss(self, model: AutoModelForSequenceClassification, inputs: Dict[str, Tensor], return_outputs:bool=False, **kwargs) -> Union[Tensor, Tuple[Tensor, SequenceClassifierOutput]]:
            """
            Compute loss function that provides new loss function based on positive class weights.

            Args:
                model (AutoModel): classification training model
                inputs (dict[str:Tensor]): dictionary of the training inputs
                return_outputs (bool): controlled by HuggingFace Trainer class. False during training (only loss needed). True for evaluation/prediction.

            Returns: 
                Tensor | (Tensor, SequenceClassifierOutput): tuple with the loss and model outputs or just the loss, depending on stage.
            """
            labels = inputs.get("labels")
            outputs = model(**inputs)
            logits = outputs.get("logits")

            labels = labels.to(logits.device).float()

            if self.pos_weight is not None:
                self.criterion.pos_weight = self.pos_weight.to(logits.device)

            loss_matrix = self.criterion(logits, labels)

            if self.active_label_mask is not None:
                mask = self.active_label_mask.to(logits.device)
                loss_matrix = loss_matrix * mask

            loss = loss_matrix.mean()

            return (loss, outputs) if return_outputs else loss

    trainer = WeightedLossTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=classification_compute_metric,
        pos_weight=pos_weight,
        active_label_mask=mask
    )

    trainer.train()

    model.save_pretrained(save_dir) # PEFT saved weights
    tokenizer.save_pretrained(save_dir)
    model.config.save_pretrained(save_dir)


def compute_pos_weight(dataset: Dataset) -> Tensor:
    """
    Function to calculate positive class weight for classification pipeline. Intended to help with class imbalance
    by increasing the penalty of positive labels.

    Args:
        dataset (Dataset): HuggingFace dataset with a "labels" columm. Labels are expected to contain multi-hot vectors

    Returns:
        Tensor: tensor of positive class weight for each class
    """
    all_labels = np.array(dataset["labels"])

    positive_counts = all_labels.sum(axis=0)
    total_samples = all_labels.shape[0]
    negative_counts = total_samples - positive_counts

    pos_weight = negative_counts / (positive_counts + 1e-5)
    return torch.tensor(pos_weight, dtype=torch.float32)


def summarization_model_training(data_dir: str, checkpoint: str, save_dir: str, training_checkpoint_dir: str, test_data_dir: str, metric_dir: str) -> None:
    """
    Training method for fine-tuning a pre-trained encoder-decoder model on clinical note summarization task.

    Args:
        data_dir (str): path to dataset being used for training
        checkpoint (str): pre-trained HuggingFace model used for training
        training_checkpoint_dir (str): directory to save in-training model config
        save_dir (str): path to saved PEFT weights for specified task
        test_data_dir (str): path to save test data to
        metric_dir (str): path to save metrics from evaluation to

    Returns:
        None
    """

    dataset = load_from_disk(data_dir)
    train_val_test = dataset.train_test_split(test_size=0.2, seed=42)
    train_val = train_val_test["train"].train_test_split(test_size=0.1, seed=42)

    train_dataset = train_val["train"]
    val_dataset = train_val["test"]
    test_dataset = train_val_test["test"] # For use when evaluating model performance after training

    test_dataset.save_to_disk(test_data_dir)

    tokenizer = AutoTokenizer.from_pretrained(checkpoint, model_max_length=1024)
    model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint, torch_dtype=torch.float16)
    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True, model=model, label_pad_token_id=-100, pad_to_multiple_of=8)

    lora_config = LoraConfig( # PERF tuning
        r=8,
        lora_alpha=16,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.1,
        bias="none",
        task_type=TaskType.SEQ_2_SEQ_LM
    )

    model = get_peft_model(model, lora_config)
    model.to("cuda")

    training_args = Seq2SeqTrainingArguments(
        output_dir=training_checkpoint_dir,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        num_train_epochs=1, # modified by epoch loop below
        logging_dir="./logs",
        save_strategy="epoch",
        eval_strategy="no",
        logging_strategy="steps",
        logging_steps=50,
        save_total_limit=2,
        predict_with_generate=False,
        fp16=True
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=None
    )

    # Modifying summarization training to halt training every epoch to evaluate on ROUGE
    num_epochs = 10
    for epoch in range(num_epochs):
        print(f"Starting epoch {epoch+1} out of {num_epochs}")

        trainer.train()

        trainer.save_model(f"{training_checkpoint_dir}/epoch_{epoch+1}") # saving checkpoint

        metrics = rouge_metrics(model, val_dataset, tokenizer) # all rouge metrics
        print(f"Epoch {epoch+1} ROUGE-L:", metrics["rougeL"])

        with open(metric_dir, "a") as f:
            f.write(json.dumps({"epoch": epoch+1, **metrics}) + "\n")
        
        torch.cuda.empty_cache() # freeing GPU resources for training restart

    model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)


def intent_model_training(data_dir: str, checkpoint: str, save_dir: str, training_checkpoint_dir: str) -> None:
    """
    Training pipeline for intent targeting (between classification and summarization for now).

    Args:
        data_dir (str): path to dataset being used for training
        checkpoint (str): pre-trained HuggingFace model used for training
        save_dir (str): path to saved PEFT weights for specified task
        training_checkpoint_dir (str): directory to save in-training model config
    
    Returns:
        None
    """

    dataset = Dataset.load_from_disk(data_dir)

    dataset = dataset.train_test_split(test_size=0.2)
    train_dataset = dataset["train"]
    test_dataset = dataset["test"]

    tokenizer = AutoTokenizer.from_pretrained(checkpoint)

    with open("data\cleaned_data\intent_label2id.json") as file:
        label2id = json.load(file)
    
    with open("data\cleaned_data\intent_id2label.json") as file:
        id2label = json.load(file)

    model = AutoModelForSequenceClassification.from_pretrained(
        checkpoint,
        num_labels=len(label2id),
        label2id=label2id,
        id2label=id2label
    )

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    training_args = TrainingArguments(
        output_dir=training_checkpoint_dir,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="epoch",
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=3,
        learning_rate=2e-5,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=intent_compute_metrics
    )

    trainer.train()

    model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)






