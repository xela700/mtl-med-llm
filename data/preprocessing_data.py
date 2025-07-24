"""
Script to preprocess data for model training for both ICD classification
and clinical note summarization.
"""

import re
import logging
import pandas as pd
import os
import torch
from tqdm import tqdm
from pathlib import Path
from datasets import Dataset
from abc import ABC, abstractmethod
from transformers import AutoTokenizer, DataCollatorWithPadding, DataCollatorForSeq2Seq, pipeline, AutoModelForSeq2SeqLM
from transformers.tokenization_utils_base import PreTrainedTokenizerBase


logger = logging.getLogger(__name__)

class TextPreprocessor(ABC):
    """
    Parent class for preprocessing text imported from dataset.

    Attributes:
    checkpoint (str): The associated HuggingFace model. Must match model used in training.
    text_col (str): Column that contains the text to be cleaned and tokenized
    remove_sections (list[str]): Optional section headers used to remove sections of clinical
    notes. Default of 'None' results in removal of sections that will always be blank due to 
    de-identification
    cleaned_path (str): path string for saving cleaned dataset
    """

    __slots__ = ("tokenizer", "text_col", "remove_sections", "cleaned_path")

    def __init__(self, checkpoint: str, text_col: str, cleaned_path: str, remove_sections: list[str] = None):
        self.tokenizer = self._create_tokenizer(checkpoint)
        self.text_col = text_col
        self.remove_sections = remove_sections
        self.cleaned_path = cleaned_path

    
    def _create_tokenizer(self, checkpoint: str) -> PreTrainedTokenizerBase:
        """
        Initializes the tokenizer to be used in preprocessing.

        Parameters:
        checkpoint (str): Checkpoint to be used to determine the model. Must be the 
        same as the checkpoint used in training.

        Returns:
        PreTrainedTokenizerBase: tokenizer for the model

        Raises:
        ValueError/OSError if checkpoint provided does not exist or is inaccessible
        """

        try:
            tokenizer = AutoTokenizer.from_pretrained(checkpoint)
        except (ValueError, OSError, ImportError) as err:
            logger.error(f"Unable to access model using checkpoint {checkpoint}")
            raise ValueError(f"Unable to access model using checkpoint {checkpoint}") from err
        
        return tokenizer

    def remove_blank_sections(self, text: str) -> str:
        """
        Removed redacted sections from MIMIC data that occur as a result of the
        deindentification process.

        Parameters:
        note (str): clinical note to be stripped of unneeded sections

        Return:
        str: clinical note stripped of blank sections
        """

        if self.remove_sections is None:
            logger.info("Using default removable sections to remove always de-identified sections.")
            self.remove_sections = [
                "Name:", "Unit No:", "Admission Date:", 
                "Discharge Date:", "Date of Birth", "Attending:",
                "Social History:", "Followup Instructions:", "Facility:" 
            ]
        
        lines = text.split("\n")
        cleaned = []
        i = 0

        while i < len(lines):
            line = lines[i].strip()

            if line == "" or re.fullmatch(r"=+\s*", line):
                i += 1
                continue

            # Detection of removable headers
            has_header = any(header.lower() in line.lower() for header in self.remove_sections)

            # Checks for redactions on same/next line
            is_redacted_line = re.search(r"_+", line)
            is_redacted_next = i + 1 < len(lines) and re.fullmatch(r"_+\s*", lines[i + 1].strip())

            # Removes redactable sections
            if has_header and (is_redacted_line or is_redacted_next):
                i += 2 if is_redacted_next else 1
                continue
            # Section kept otherwise
            cleaned.append(line)
            i += 1
        
        return "\n".join(cleaned)
    

    def normalize_deidentified_blanks(self, text: str) -> str:
        """
        Inserts standard placeholder text inplace of de-identified blanks where applicable.

        Parameters:
        text (str): Input text w/ de-identified blanks from MIMIC notes

        Returns:
        str: Text w/ replacement placeholders for training
        """

        replacements = {
            r"\b_{2,}\s*(years?\s*old|y\.?o\.?|y/o|yo)": "<AGE>",
            r"\b_{2,}-years?-old\b": "<AGE>",
            r"\b(mrs?|ms)\.?\s_{2,}": "<PATIENT>",
            r"\b(doctor|dr.?)\s_{2,}": "<DOCTOR>",

        }

        for pattern, replacement in replacements.items():
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
        
        text = re.sub(r"\b_{2,}\b", "<REDACTED>", text) # Clean up for remaining unknown de-identified fields
        return text
    
    
    @abstractmethod
    def preprocess_function(self, batch: dict) -> dict[str, list[int]]:
        """
        Preprocessing method that may vary depending on NLP task. To be determined by child classes.
        Intended for use with dataset.map()

        Parameters:
        batch (dict): batch on which the preprocess function will be applied

        Returns:
        dictionary structure expected by dataset.map()
        """
        pass

    @abstractmethod
    def preprocess(self, dataframe: pd.DataFrame) -> None:
        """
        Full preprocess pipeline for specific task. Takes a dataframe, cleans it, applies tokenization and filtering where applicable,
        and saves it to a new location for use during model training.

        Parameters:
        dataframe: data to undergo preprocessing.
        """
        pass
    


class ClassificationPreprocessor(TextPreprocessor):
    """
    Preprocessor child class for classification training pipeline for use with encoder models.
    Requires labels (ICD-10 codes) paired with input text.

    Attributes:
    label_ids -> Separate label-specific query. Current plan to use initial dataset and group by
    ICD-10 codes. Only training on most frequent 2k - 5k codes (out of 70k) to reduce chance of no relevant
    samples.
    label_col -> Column in the full dataset that contains the ICD-10 labels for a given clinical note.
    extract_sections -> Sections of full text most relevant for classification.
    """

    __slots__ = ("label_ids", "extract_sections", "label_col", "data_collator")

    def __init__(self, checkpoint: str, label_ids: dict[str:int], text_col: str, label_col: str, cleaned_path: str, remove_sections: list[str] = None, extract_sections: list[str] = None):
        super().__init__(checkpoint, text_col, remove_sections, cleaned_path)
        self.extract_sections = extract_sections
        self.label_ids = label_ids
        self.label_col = label_col
        self.data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)

    def classification_extract_sections(self, text: str) -> str:
        """
        Captures relevant sections for ICD code classification. Used to limit the input size of a given sample while minimizing
        context loss. Only intended for classification task and not summarization.

        Parameters:
        text (str): Input discharge notes with extractable sections
        sections (list[str]): sections to keep from text. If none provided, a default set is used

        Returns:
        str: Extracted sections from text joined together 
        """

        if self.extract_sections is None:
            logger.info("Using default sections to extract data for classification.")
            self.extract_sections = ['discharge diagnosis', 'brief hospital course', 'hospital course',
                        'final diagnosis', 'principal diagnosis', 'primary diagnosis', 'history of present illness']
        
        # Normalize section names for comparison
        target_sections = [s.lower().strip() for s in self.extract_sections]

        # Find all sections with their content
        section_pattern = re.compile(
            r'(?i)(?P<header>^[A-Z][A-Z \-/]{3,}):\s*\n(?P<body>.*?)(?=^[A-Z][A-Z \-/]{3,}:|\Z)',
            re.DOTALL | re.MULTILINE
        )

        extracted = []

        for match in section_pattern.finditer(text):
            header = match.group('header').strip().lower()
            body = match.group('body').strip()

            if header in target_sections:
                extracted.append(body)

        return ' '.join(extracted)
    

    def preprocess_function(self, batch: dict[str, list[str]]) -> dict[str, list[int]]:
        """
        Preprocess function to be mapped using dataset specific to classification task.
        To be used with dataset.map() method.

        Parameters:
        batch (dict[list[str]]): batch from dataset

        Returns:
        Modified data in dataset. Text field tokenized based on tokenizer.
        Multi-hot label vectors created using top k labels.
        """

        processed_texts = []
        for text in batch[self.text_col]:
            text = self.remove_blank_sections(text)
            text = self.normalize_deidentified_blanks(text)
            text = self.classification_extract_sections(text)
            processed_texts.append(text)

        tokenized_data = self.tokenizer(processed_texts, truncation=True, return_tensors=None)

        multi_hot_labels = []
        for labels in batch[self.label_col]:
            label_vector = [0] * len(self.label_ids)
            for label in labels:
                if label in self.label_ids:
                    label_vector[self.label_ids[label]] = 1
            multi_hot_labels.append(label_vector)
        tokenized_data["labels"] = multi_hot_labels

        return tokenized_data
    

    def filter_text_by_label(self, sample: dict[str, list[str]]) -> bool:
        """
        Filtering method to check for the presence of one (or more) ICD labels that are being used during
        training. To be used with dataset.filter() method

        Parameters:
        sample: dataset sample being provided to check for label presence

        Return:
        bool: true if at least one label is present
        """
        icd_codes = sample.get("labels", []) # gets the labels associated with a clinical note, or an empty list if none present
        return any(code in self.label_ids for code in icd_codes)

    def preprocess(self, dataframe: pd.DataFrame) -> None:
        """
        Full preprocess steps for classification. Renames columns to appropriate names.
        Converts dataframe to a dataset. Filters samples without any applicable training labels.
        Maps preprocess function. Finally, saves dataset to disk for reuse.

        Parameters:
        dataframe: full dataframe for classification preprocessing
        """
        if self.text_col != "input":
            dataframe.rename(columns={self.text_col: "input"}, inplace=True)
        if self.label_col != "labels":
            dataframe.rename(columns={self.label_col: "labels"}, inplace=True)

        dataset = Dataset.from_pandas(dataframe)

        dataset.filter(self.filter_text_by_label, batched=True)
        dataset.map(self.preprocess_function, batched=True)

        if os.path.exists(self.cleaned_path):
            logger.info(f"Warning: {self.cleaned_path} exists and will be overwritten with new cleaned dataset.")
        
        clean_path = Path(self.cleaned_path)

        dataset.save_to_disk(str(clean_path))
    
        



class SummarizationPreprocessor(TextPreprocessor):
    """
    Preprocessor for summarization task for use with encoder-decoder models. 
    Requires target summaries for training.
    Current plan to mix section within clinical notes that most resembles a summary with synthetic 
    summaries generated by an LLM.

    Attributes:
    target_col -> summary targets in the dataframe
    source_type_col -> tracking column for whether the summary target is 'real' (pulled from data) or 'synthetic' (generated by LLM)
    data_collator -> preparing data for encoder-decoder model training
    """

    __slots__ = ("target_col", "source_type_col", "data_collator")

    def __init__(self, checkpoint: str, text_col: str, target_col: str, source_type_col: str, cleaned_path: str, remove_sections: list[str] = None):
        super().__init__(checkpoint, text_col, remove_sections, cleaned_path)
        self.target_col = target_col
        self.source_type_col = source_type_col
        self.data_collator = DataCollatorForSeq2Seq(tokenizer=checkpoint, model=checkpoint, padding=True)

    
    def preprocess_function(self, batch: dict[str, list[str]]) -> dict[str, list[str]]:
        """
        Preprocess function to be mapped using dataset specific to summarization task.
        To be used with dataset.map() method.

        Parameters:
        batch (dict[list[str]]): batch from dataset

        Returns:
        Modified data in dataset. Text field tokenized based on tokenizer.
        Multi-hot label vectors created using top k labels.
        """
        input_text = batch[self.text_col]
        target_text = batch[self.target_col]

        # preprocess input text (i.e. entire clinical notes)
        tokenized_data = self.tokenizer( 
            input_text,
            truncation=True
        )

        with self.tokenizer.as_target_tokenizer(): # preprocess target text (i.e. summaries)
            labels = self.tokenizer(
                target_text,
                truncation=True
            )["input_ids"]

        tokenized_data["labels"] = labels
        return tokenized_data


class SummarizationTargetPreprocessor(TextPreprocessor):
    """
    Preprocessor for raw data intended for summarization. Used to clean data and prepare for input into LLM to create synthetic targets.

    Attributes:
    extract_sections -> section from text to be extracted for use as real training targets
    model -> model for generating synthetic summaries (uses same checkpoint)
    device -> GPU or CPU for summary generation. Will use GPU if available.
    summarizer -> pipeline for generating summaries
    """

    __slots__ = ("extract_sections", "model", "device", "summarizer")

    def __init__(self, checkpoint: str, text_col: str, cleaned_path: str, remove_sections: list[str] = None, extract_sections: list[str] = None):
        super().__init__(checkpoint, text_col, remove_sections, cleaned_path)
        self.extract_sections = extract_sections
        self.model = AutoModelForSeq2SeqLM(checkpoint)
        self.device = 0 if torch.cuda.is_available() else -1
        self.summarizer = pipeline("summarization", model=self.model, tokenizer=self.tokenizer, device=self.device)

    
    def extract_from_text(self, text: str, extract_section_list: list[str] = None) -> str | None:
        """
        Helper function for extract_target. Extracts sections from text and returns them if they exist in a combined 
        format. (Only one should exist)

        Parameters:
        text (str): text to have portion(s) extracted
        extract_sections (list[str]): section(s) being extracted from text

        Returns:
        str: text from extracted sections
        None: returns nothing if section doesn't exist
        """


        # Normalize section names for comparison
        target_sections = [s.lower().strip() for s in extract_section_list]

        # Find all sections with their content
        section_pattern = re.compile(
            r'(?i)(?P<header>^[A-Z][A-Z \-/]{3,}):\s*\n(?P<body>.*?)(?=^[A-Z][A-Z \-/]{3,}:|\Z)',
            re.DOTALL | re.MULTILINE
        )

        extracted = []

        for match in section_pattern.finditer(text):
            header = match.group('header').strip().lower()
            body = match.group('body').strip()

            if header in target_sections:
                extracted.append(body)
        
        combined = ' '.join(extracted)

        return combined if combined else None
        # parsed_text = {}
        # for section in extract_sections:
        #     # Pattern captures everything after the name of the section up until either the next section title or EOF
        #     pattern = rf'{section.lower()}[\s:\n](.*?)(\n[A-Z][^:\n]*:|\Z)'
        #     match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
        #     if match:
        #         parsed_text[section] = match.group(1).strip()

        # combined = ' '.join(filter(None, parsed_text.values()))
        
        # return combined if combined else None


    def extract_real_target(self, dataframe: pd.DataFrame, save_path: str) -> None:
        """
        Takes a training dataframe and pulls out section intended to represent a target summarization.
        This dataframe is not to be used for creating synthetic targets.

        Parameters:
        dataframe: Dataframe with only text (SEPARATE FROM SYNTHETIC DF) -> to have representative summary extracted
        save_path (str): Path to save dataframe with real summary targets
        """

        if self.extract_sections is None:
            logger.info("Using default section(s) to extract target data.")
            self.extract_sections = ["brief hospital course", "hospital course"]
        
        df = dataframe.copy()
        df["target"] = df[self.text_col].apply(lambda x: self.extract_from_text(x, self.extract_sections))
        df = df.dropna(subset=["target"]) # Removes samples w/o any applicable sections
        df["source_type"] = "real" # Targets are "real" in that they come from the data

        df.to_parquet(save_path)
    
    def generate_synthetic_targets(self, dataframe: pd.DataFrame, save_path: str, chunk_size: int = 100, resume: bool = True) -> None:
        """
        Takes training input and generates synthetic summaries using a pretrained medical LLM (checkpoint).
        Generates in chunks to limit chance of interruptions.

        Parameters:
        dataframe: Dataframe with only text (SEPARATE FROM REAL DF) -> to have summary generated
        chunk_size (int): Number of rows processed at a time
        save_path (str): Path where the summaries will be saved
        resume (bool): If True, will skip rows with generated summaries
        """

        if resume and os.path.exists(save_path):
            df_summaries = pd.read_parquet(save_path)
            processed_ids = set(df_summaries.index)
            logger.info(f"Resuming summary generation from file. {len(processed_ids)} rows already generated.")
        else:
            df_summaries = pd.DataFrame()
            processed_ids = set()
            logger.info(f"Starting summary generation using {self.model}")

        # Loop to generate summaries in chunks to help protect completed work
        for start in tqdm(range(0, len(dataframe), chunk_size), desc="Summarizing"):
            end = min(start + chunk_size, len(dataframe))
            chunk = dataframe.iloc[start:end].copy() # current chunk being summarized

            if resume:
                # Only using ids that don't exist in current save
                chunk = chunk[~chunk.index.isin(processed_ids)]
                if chunk.empty:
                    continue
            
            chunk["target"] = chunk[self.text_col].apply(
                lambda x: self.summarizer(x, max_length=256, min_length=30, do_sample=False)[0]["summary_text"]
                )
            chunk["source_type"] = "synthetic" # Targets are synthetic in that they are generated
            
            chunk.to_parquet(
                save_path, 
                engine='pyarrow', 
                index=True, 
                append=os.path.exists(save_path)
                )

    



    
