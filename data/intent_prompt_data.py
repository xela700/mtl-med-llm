"""
Script to create a series of prompts to train intent targetting model.
"""

import pandas as pd
import random
import itertools
from data.fetch_data import load_data

summarization_prompts = [
                "Summarize these notes:",
                "Write a short summary of the following:",
                "Can you provide a brief summary?",
                "Give me a brief description of this patient's visit:",
                "Can you shorten the following?",
                "Describe these clinical notes:",
                "How would you describe the patient's visit?",
                "Based on this note, what happened?",
                "What is the key takeaway from this note?",
                "Describe the main points for this:",
                "Please summarize this note:",
                "Give me a short version of this clinical note:",
                "Summarize the following record:",
                "Write a brief overview of this note:",
                "Condense this patient report into a summary",
                "Shorten this as best you can:",
                "Please make this patient record briefer:",
                ""
]

classification_prompts = [
                "Classify this note:",
                "What ICD codes relate to this patient's visit?",
                "Determine ICD codes based on the following text:",
                "Codes associated with patient",
                "Potential diagnosis for the following patient:",
                "Please identify ICD codes for this visit:",
                "ICD codes from note:",
                "What labels should be assigned here?",
                "Assign suitable categories for this note:",
                "What labels would you apply to this patient's visit?"
]

neutral_prompts = [
                "Process the following input:",
                "Review the text below:",
                "Consider the following note:",
                "Analyze this record:",
                "Read the following:"
]

class PromptCreator:
    """
    Class to construct prompts for generating an intent-targeting dataset

    Attributes:
        summarize_verbs (list[str]): list of summarization intended verb phrases
        classify_verbs (list[str]): list of classification intended verb phrases
        modifiers (list[str]): list of politeness phrases
        targets (list[str]): list of object-based phrases
        suffixes (list[str]): list of potential sentence suffixes
    """

    def __init__(self, seed: int = 42):
        random.seed(seed)

        self.summarize_verbs = [
            "summarize", "provide a summary of", "briefly summarize", "give a brief summary of", "outline", "condense", "shorten"
        ]

        self.classify_verbs = [
            "classify", "categorize", "identify ICD codes for", "label", "assign codes to", "find potential ICD codes for", "categorize", "give medical diagnosis", "provide codes for"
        ]

        self.modifiers = [
            "", "please", "can you", "could you", "kindly", "would you mind", "I need you to", "help me"
        ]

        self.targets = [
            "this note", "the following text", "the content below", "this clinical note", "the text provided"
        ]

        self.suffixes = [
            "", "for me", "if possible", "when you can", "if you can", "please", "as soon as possible"
        ]
    
    def _permute(self, verbs: list[str]) -> list[str]:
        """
        Builds list of unique NLP prompts for intent targeting
        """
        prompts = set()

        for verb, modifier, target, suffix in itertools.product(verbs, self.modifiers, self.targets, self.suffixes):
            if modifier:
                command = f"{modifier} {verb} {target} {suffix}".strip()
            else:
                command = f"{verb.capitalize()} {target} {suffix}".strip()
            
            prompts.add(command)

            if modifier in ["can you", "could you", "would you mind"]:
                question = f"{modifier} {verb} {target} {suffix}?".strip()
                prompts.add(question)
            
        prompts = list({p.replace(" ", " ").strip() for p in prompts})

        random.shuffle(prompts)
        return prompts
    
    def generate_summarization_prompts(self, limit: int | None = None) -> list[str]:
        """
        Generates prompts specific to summarization

        Args:
            limit (int | None): Limit the number of potential premutations
        
        Returns:
            list[str]: List of premutations
        """
        prompts = self._permute(self.summarize_verbs)
        return prompts[:limit] if limit else prompts
    
    def generate_classification_prompts(self, limit: int | None = None) -> list[str]:
        """
        Generates prompts specific to classification

        Args:
            limit (int | None): Limit the number of potential premutations
        
        Returns:
            list[str]: List of premutations
        """
        prompts = self._permute(self.classify_verbs)
        return prompts[:limit] if limit else prompts

class NeutralPromptCreator:
    """
    Creates neutral (no target) prompts to vary data for intent targeting model

    Attributes:
        neutral_verbs (list[str]): list of neutral verb phrases
        modifiers (list[str]): list of politeness phrases
        targets (list[str]): list of object-based phrases
        suffixes (list[str]): list of potential sentence suffixes
    """

    def __init__(self):
        self.neutral_verbs = [
            "process", "look at", "review", "analyze", "help me with", "work with", "handle"
        ]

        self.modifiers = [
            "", "please", "can you", "could you", "kindly", "would you mind", "I need you to", "help me"
        ]

        self.targets = [
            "this note", "the following text", "the content below", "this clinical note", "the text provided"
        ]

        self.suffixes = [
            "", "for me", "if possible", "when you can", "if you can", "please", "as soon as possible"
        ]
    
    def generate(self, limit: int | None = None) -> list[str]:
        """
        Generates list of neutral-based prompts

        Args:
            limit (int | None): Limit the number of potential premutations
        
        Returns:
            list[str]: List of premutations
        """
        prompts = set()

        for verb, modifier, target, suffix in itertools.product(self.neutral_verbs, self.modifiers, self.targets, self.suffixes):

            if modifier:
                prompt = f"{modifier} {verb} {target} {suffix}".strip()
            else:
                prompt = f"{verb.capitalize()} {target} {suffix}".strip()
            
            prompts.add(prompt)

            if modifier in ["can you", "could you", "would you mind"]:
                prompts.add(f"{modifier} {verb} {target} {suffix}?".strip())
            
        prompts = list({p.replace(" ", " ").strip() for p in prompts})
        random.shuffle(prompts)

        return prompts[:limit] if limit else prompts

def create_intent_dataset(note_path: str, text_col: str, save_dir: str, limit: int | None = None) -> None:
    """
    Creates a complete dataset for intent-targeting model training and saves to file. Aims for a balance of correct label prompts, ambiguous prompts, and raw text.

    Args:
        save_dir (str): location to save dataset
        limit (int | None): limit for number of premutations for each class of target (or just max number of possible prompts)
    
    Returns:
        None
    """

    prompter = PromptCreator()
    neutral_prompter = NeutralPromptCreator()

    summary_prompts = prompter.generate_summarization_prompts(limit=limit)
    classify_prompts = prompter.generate_classification_prompts(limit=limit)
    neutral_prompts = neutral_prompter.generate(limit=limit)

    dataset = []

    for p in summary_prompts:
        dataset.append({"text": p, "label": "summarization"})
    
    for p in classify_prompts:
        dataset.append({"text": p, "label": "classification"})
    
    for p in neutral_prompts:
        label = random.choice(["summarization", "classification"])
        dataset.append({"text": p, "label": label})
    
    raw_text = load_data(note_path)

    raw_text = raw_text[text_col].tolist()[:limit]

    for p in raw_text:
        label = random.choice(["summarization", "classification"])
        dataset.append({"text": p, "label": label})
    
    dataframe = pd.DataFrame(dataset)

    dataframe.to_parquet(save_dir)