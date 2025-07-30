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
from model.evaluate_model import classification_compute_metric, SummarizationMetrics
import torch
import numpy as np
from torch import Tensor
from torch.utils.data import DataLoader
from datasets import Dataset
from typing import Union, Dict, Tuple

def classification_model_training(data_dir: str, label_dir: str, checkpoint: str, save_dir: str, training_checkpoint_dir: str, test_data_dir: str) -> None:
    """
    Training method for fine-tuning a pre-trained encoder model on ICD-10 code
    classification task.

    Parameters:
    data_dir (str): path to dataset being used for training
    label_dir (str): path to labels
    checkpoint (str): pre-trained HuggingFace model used for training
    save_dir (str): path to saved PEFT weights for specified task
    test_data_dir (str): path to save test data to
    """

    dataset = load_from_disk(data_dir)
    train_val_test = dataset.train_test_split(test_size=0.2, seed=42)
    train_val = train_val_test["train"].train_test_split(test_size=0.1, seed=42)

    train_dataset = train_val["train"]
    val_dataset = train_val["test"]
    test_dataset = train_val_test["test"] # For use when evaluating model performance after training

    test_dataset.save_to_disk(test_data_dir)

    num_labels = len(load_data(label_dir))

    config = AutoConfig.from_pretrained(checkpoint)
    config.num_labels = num_labels
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
        num_train_epochs=3,
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
        def compute_loss(self, model: AutoModelForSequenceClassification, inputs: Dict[str, Tensor], return_outputs:bool=False, **kwargs) -> Union[Tensor, Tuple[Tensor, SequenceClassifierOutput]]:
            """
            Compute loss function that provides new loss function based on positive class weights.

            Parameters:
            model: classification training model
            inputs: dictionary of the training inputs
            return_outputs: controlled by HuggingFace Trainer class. False during training (only loss needed). True for evaluation/prediction.

            Returns: tuple with the loss and model outputs or just the loss, depending on stage.
            """
            labels = inputs.get("labels")
            outputs = model(**inputs)
            logits = outputs.logits

            loss_function = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(logits.device))
            loss = loss_function(logits, labels.float())

            return (loss, outputs) if return_outputs else loss

    trainer = WeightedLossTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=classification_compute_metric
    )

    trainer.train()

    model.save_pretrained(save_dir) # PEFT saved weights
    tokenizer.save_pretrained(save_dir)

def summarization_model_training(data_dir: str, checkpoint: str, save_dir: str, training_checkpoint_dir: str, test_data_dir: str) -> None:
    """
    Training method for fine-tuning a pre-trained encoder-decoder model on clinical note summarization task.

    Parameters:
    data_dir (str): path to dataset being used for training
    checkpoint (str): pre-trained HuggingFace model used for training
    save_dir (str): path to saved PEFT weights for specified task
    test_data_dir (str): path to save test data to
    """

    dataset = load_from_disk(data_dir)
    train_val_test = dataset.train_test_split(test_size=0.2, seed=42)
    train_val = train_val_test["train"].train_test_split(test_size=0.1, seed=42)

    train_dataset = train_val["train"]
    val_dataset = train_val["test"]
    test_dataset = train_val_test["test"] # For use when evaluating model performance after training

    test_dataset.save_to_disk(test_data_dir)

    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
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

    training_args = Seq2SeqTrainingArguments(
        output_dir=training_checkpoint_dir,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        num_train_epochs=3,
        logging_dir="./logs",
        save_strategy="steps",
        eval_strategy="steps",
        eval_steps=100,
        save_steps=100,
        logging_steps=50,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=True,
        predict_with_generate=False,
        fp16=True
    )

    # compute_metrics = SummarizationMetrics(tokenizer=tokenizer)

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=None
    )

    trainer.train()

    model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)
    
def compute_pos_weight(dataset: Dataset) -> Tensor:
    """
    Function to calculate positive class weight for classification pipeline. Intended to help with class imbalance
    by increasing the penalty of positive labels.

    Parameters:
    dataset: HuggingFace dataset with a "labels" column
    labels is expected to contain multi-hot vectors

    Returns:
    Tensor of positive class weight for each class
    """
    all_labels = np.array(dataset["labels"])

    positive_counts = all_labels.sum(axis=0)
    total_samples = all_labels.shape[0]
    negative_counts = total_samples - positive_counts

    pos_weight = negative_counts / (positive_counts + 1e-5)
    return torch.tensor(pos_weight, dtype=torch.float32)




