"""
Script to preprocess data for model training for both ICD classification
and clinical note summarization.
"""

import re
import logging
from abc import ABC, abstractmethod
from transformers import AutoTokenizer, DataCollatorWithPadding, DataCollatorForSeq2Seq
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
    """

    __slots__ = ("tokenizer", "text_col", "remove_sections")

    def __init__(self, checkpoint: str, text_col: str, remove_sections: list[str] = None):
        self.tokenizer = self._create_tokenizer(checkpoint)
        self.text_col = text_col
        self.remove_sections = remove_sections

    
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

            if line == "":
                i +=1
                continue

            if any(header.lower() in line.lower() for header in self.remove_sections):
                # Captures and removes sections followed by underscore blanks
                if re.fullmatch(r".*:\s*_+\s*", line):
                    i += 1
                    continue
                # Captures and removes sections with underscore blanks on the following line
                elif i + 1 < len(lines) and re.fullmatch(r"^_+\s*", lines[i + 1].strip()):
                    i += 2
                    continue
                # Catch all for a header that should be removed
                else:
                    i += 1
                    continue
            
            else:
                cleaned.append(lines[i])
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

    def __init__(self, checkpoint: str, label_ids: dict[str:int], text_col: str, label_col: str, remove_sections: list[str] = None, extract_sections: list[str] = None):
        super().__init__(checkpoint, text_col, remove_sections)
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
                        'final diagnosis', 'principal diagnosis', 'history of present illness']
        
        parsed_text = {}

        for section in self.extract_sections:
            # Pattern captures everything after the name of the section up until either the next section title or EOF
            pattern = rf'{section.lower()}[\s:\n](.*?)(\n[A-Z][^:\n]*:|\Z)'
            match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
            if match:
                parsed_text[section] = match.group(1).strip()
        
        return ' '.join(filter(None, parsed_text.values()))
    

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
    
        



class SummarizationPreprocessor(TextPreprocessor):
    """
    Preprocessor for summarization task for use with encoder-decoder models. 
    Requires target summaries for training.
    Current plan to mix section within clinical notes that most resembles a summary with synthetic 
    summaries generated by an LLM.
    """

    __slots__ = ("target_col", "source_type_col", "data_collator")

    def __init__(self, checkpoint: str, text_col: str, target_col: str, source_type_col: str, remove_sections: list[str] = None):
        super().__init__(checkpoint, text_col, remove_sections)
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



    



    
