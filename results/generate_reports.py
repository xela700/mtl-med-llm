"""
Script to extract metrics from training checkpoints.
"""

from utils.config_loader import load_config
import matplotlib.pyplot as plt
import os
import json

config = load_config()

def classification_metrics() -> None:
    """
    Pulls classification model training metrics and creates graphs for them. Graphs are saved to results/reporting/...
    """

    checkpoint_directory = "model/saved_model/ft-biobert-large-checkpoints/checkpoint-32991"

    training_epochs = []
    training_losses = []

    eval_epochs = []
    eval_losses = []
    eval_f1_macros = []
    eval_f1_micros = []
    eval_hamming_losses = []
    eval_roc_auc_micros = []
    eval_runtimes = []


    trainer_state_path = os.path.join(checkpoint_directory, "trainer_state.json")
    
    if os.path.exists(trainer_state_path):
        with open(trainer_state_path, "r") as file:
            state = json.load(file)

        for entry in state.get("log_history", []):
            if "loss" in entry:
                training_epochs.append(entry["epoch"])
                training_losses.append(entry["loss"])

            if "eval_loss" in entry:
                eval_epochs.append(entry["epoch"])
                eval_losses.append(entry["eval_loss"])
                eval_f1_macros.append(entry["eval_f1_macro"])
                eval_f1_micros.append(entry["eval_f1_micro"])
                eval_hamming_losses.append(entry["eval_hamming_loss"])
                eval_roc_auc_micros.append(entry["eval_roc_auc_micro"])
                eval_runtimes.append(entry["eval_runtime"])

    
    plt.figure(figsize=(10, 5))
    plt.plot(training_epochs, training_losses, marker='o', color='blue')
    plt.title("Training Loss per Epoch for BioBERT Fine-Tuning")
    plt.xlabel("Training Epoch")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("results/reporting/classification_training_loss.jpg")
    plt.show()
    plt.clf()

    fig, axis = plt.subplots(2, 2, figsize=(12, 8))
    axis[0, 0].plot(eval_epochs, eval_losses, marker='o', color='blue')
    axis[0, 0].set_title("Evaluation Loss per Epoch for BioBERT Fine-Tuning")
    axis[0, 0].set_xlabel("Evaluation Epoch")
    axis[0, 0].set_ylabel("Evaluation Loss")

    axis[0, 1].plot(eval_epochs, eval_roc_auc_micros, marker='o', color='blue')
    axis[0, 1].set_title("ROC-AUC Micro per Epoch for BioBERT Fine-Tuning")
    axis[0, 1].set_xlabel("Evaluation Epoch")
    axis[0, 1].set_ylabel("Evaluation ROC-AUC Micro")

    axis[1, 0].plot(eval_epochs, eval_f1_micros, marker='o', color='blue')
    axis[1, 0].set_title("F1 Micro per Epoch for BioBERT Fine-Tuning")
    axis[1, 0].set_xlabel("Evaluation Epoch")
    axis[1, 0].set_ylabel("Evaluation F1 Micro")

    axis[1, 1].plot(eval_epochs, eval_f1_macros, marker='o', color='blue')
    axis[1, 1].set_title("F1 Macro per Epoch for BioBERT Fine-Tuning")
    axis[1, 1].set_xlabel("Evaluation Epoch")
    axis[1, 1].set_ylabel("Evaluation F1 Macro")

    plt.tight_layout()
    plt.savefig("results/reporting/classification_training_metrics.jpg")
    plt.show()
    plt.clf()

    plt.figure(figsize=(5, 5))
    plt.plot(eval_epochs, eval_runtimes, marker='o', color='blue')
    plt.title("Runtime per Epoch for BioBERT Fine-Tuning")
    plt.xlabel("Evaluation Epoch")
    plt.ylabel("Runtime (Seconds)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("results/reporting/classification_training_runtime.jpg")
    plt.show()
    plt.clf()

def summarization_metrics():
    """
    Pulls summarization model training metrics and creates graphs for them. Graphs are saved to results/reporting/...
    """

    checkpoint_directory = "model/saved_model/ft-biobart-large-checkpoints/checkpoint-4818"

    training_epochs = []
    training_losses = []

    trainer_state_path = os.path.join(checkpoint_directory, "trainer_state.json")
    
    if os.path.exists(trainer_state_path):
        with open(trainer_state_path, "r") as file:
            state = json.load(file)

        for entry in state.get("log_history", []):
            if "loss" in entry:
                training_epochs.append(entry["epoch"])
                training_losses.append(entry["loss"])

    
    plt.figure(figsize=(10, 5))
    plt.plot(training_epochs, training_losses, marker='o', color='blue')
    plt.title("Training Loss per Epoch for BioBART Fine-Tuning")
    plt.xlabel("Training Epoch")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("results/reporting/summarization_training_loss.jpg")
    plt.show()
    plt.clf()

def intent_metrics() -> None:
    """
    Pulls intent-targeting model training metrics and creates graphs for them. Graphs are saved to results/reporting/...
    """

    checkpoint_directory = "model/saved_model/ft-distilbert-base-checkpoints/checkpoint-90"

    training_epochs = []
    training_losses = []

    eval_epochs = []
    eval_losses = []
    eval_accuracy = []


    trainer_state_path = os.path.join(checkpoint_directory, "trainer_state.json")
    
    if os.path.exists(trainer_state_path):
        with open(trainer_state_path, "r") as file:
            state = json.load(file)

        for entry in state.get("log_history", []):
            if "loss" in entry:
                training_epochs.append(entry["epoch"])
                training_losses.append(entry["loss"])

            if "eval_loss" in entry:
                eval_epochs.append(entry["epoch"])
                eval_losses.append(entry["eval_loss"])
                eval_accuracy.append(entry["eval_accuracy"])

    
    plt.figure(figsize=(10, 5))
    plt.plot(training_epochs, training_losses, marker='o', color='blue')
    plt.title("Training Loss per Epoch for DistilBERT Fine-Tuning")
    plt.xlabel("Training Epoch")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("results/reporting/intent_training_loss.jpg")
    plt.show()
    plt.clf()

    fig, axis = plt.subplots(1, 2, figsize=(10, 6))
    axis[0].plot(eval_epochs, eval_losses, marker='o', color='blue')
    axis[0].set_title("Evaluation Loss per Epoch for DistilBERT Fine-Tuning")
    axis[0].set_xlabel("Evaluation Epoch")
    axis[0].set_ylabel("Evaluation Loss")

    axis[1].plot(eval_epochs, eval_accuracy, marker='o', color='blue')
    axis[1].set_title("Class Accuracy per Epoch for DistilBERT Fine-Tuning")
    axis[1].set_xlabel("Evaluation Epoch")
    axis[1].set_ylabel("Class Accuracy")

    plt.tight_layout()
    plt.savefig("results/reporting/intent_training_metrics.jpg")
    plt.show()
    plt.clf()

if __name__ == "__main__":
    # classification_metrics()
    # summarization_metrics()
    intent_metrics()



